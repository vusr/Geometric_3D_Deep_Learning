"""
Prepare ModelNet40 point clouds and regression ground-truth labels.

For each .off mesh file:
  1. Load with Trimesh
  2. Attempt mesh repair if not watertight
  3. Compute volume, surface area, bounding-box extents
  4. Sample 1024 surface points (uniform)
  5. Normalize point cloud: zero-mean, scale to unit sphere
  6. Save as .npz:  {points, volume, area, bbox_extents, class_id, class_name, valid_volume}

Output directory: ~/geometric_deep_learning/data/processed/
Run: python scripts/prepare_data.py [--raw_dir ...] [--out_dir ...] [--n_points 1024] [--workers 8]
"""

import argparse
import os
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm


CLASS_NAMES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
    "wardrobe", "xbox",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Center to zero mean and scale to unit sphere."""
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points


def process_one(args):
    """Process a single .off file. Returns a result dict or error string."""
    off_path, out_path, n_points = args

    try:
        mesh = trimesh.load(str(off_path), force="mesh", process=True)

        # Attempt repair if not watertight
        valid_volume = False
        volume = float("nan")
        if mesh.is_watertight:
            volume = float(mesh.volume)
            valid_volume = True
        else:
            try:
                trimesh.repair.fill_holes(mesh)
                if mesh.is_watertight:
                    volume = float(mesh.volume)
                    valid_volume = True
            except Exception:
                pass

        area = float(mesh.area)
        bbox_extents = mesh.bounding_box.extents.tolist()  # [dx, dy, dz]

        # Sample surface points
        if len(mesh.faces) == 0:
            return f"SKIP (no faces): {off_path}"
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        points = normalize_point_cloud(np.array(points, dtype=np.float32))

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            points=points.astype(np.float32),
            volume=np.float32(volume),
            area=np.float32(area),
            bbox_extents=np.array(bbox_extents, dtype=np.float32),
            valid_volume=np.bool_(valid_volume),
        )
        return None  # success

    except Exception as e:
        return f"ERROR [{off_path}]: {e}\n{traceback.format_exc()}"


def build_file_list(raw_dir: Path, out_dir: Path, n_points: int):
    """Walk raw ModelNet40 and build (off_path, out_path, n_points) tuples."""
    tasks = []
    meta = []  # (class_name, class_id, split, out_npz_path)

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in CLASS_TO_ID:
            # Handle classes with spaces or different casing
            normalized = class_name.lower().replace(" ", "_")
            if normalized in CLASS_TO_ID:
                class_name = normalized
            else:
                print(f"  WARNING: Unknown class '{class_dir.name}', skipping.")
                continue
        class_id = CLASS_TO_ID[class_name]

        for split in ("train", "test"):
            split_dir = class_dir / split
            if not split_dir.exists():
                continue
            for off_file in sorted(split_dir.glob("*.off")):
                rel = off_file.relative_to(raw_dir)
                out_path = out_dir / rel.with_suffix(".npz")
                tasks.append((str(off_file), str(out_path), n_points))
                meta.append(
                    {
                        "npz_path": str(out_path),
                        "class_name": class_name,
                        "class_id": class_id,
                        "split": split,
                        "off_path": str(off_file),
                    }
                )

    return tasks, meta


def main():
    parser = argparse.ArgumentParser(description="Prepare ModelNet40 data.")
    parser.add_argument(
        "--raw_dir",
        default=os.path.expanduser("~/geometric_deep_learning/data/raw/ModelNet40"),
        help="Path to extracted ModelNet40 folder (contains class subdirectories)",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.expanduser("~/geometric_deep_learning/data/processed"),
        help="Output directory for .npz files",
    )
    parser.add_argument("--n_points", type=int, default=1024, help="Points to sample per shape")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (use 1 for debugging)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Run scripts/download_modelnet40.sh first."
        )

    print(f"Building file list from {raw_dir} ...")
    tasks, meta = build_file_list(raw_dir, out_dir, args.n_points)
    print(f"  Found {len(tasks)} shapes across {len(set(m['class_name'] for m in meta))} classes.")

    # Skip already-processed files
    pending = [(t, m) for t, m in zip(tasks, meta) if not Path(t[1]).exists()]
    print(f"  {len(tasks) - len(pending)} already processed, {len(pending)} remaining.")

    errors = []
    if pending:
        task_args = [t for t, _ in pending]
        if args.workers == 1:
            for task in tqdm(task_args, desc="Processing meshes"):
                err = process_one(task)
                if err:
                    errors.append(err)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(process_one, t): t for t in task_args}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing meshes"):
                    err = future.result()
                    if err:
                        errors.append(err)

    # Save meta index
    meta_path = out_dir / "meta.json"
    # Update meta with valid_volume info by checking saved npz
    print("Building metadata index...")
    meta_with_vol = []
    skipped = 0
    for m in tqdm(meta, desc="Reading metadata"):
        npz_path = Path(m["npz_path"])
        if not npz_path.exists():
            skipped += 1
            continue
        data = np.load(str(npz_path))
        m2 = dict(m)
        m2["valid_volume"] = bool(data["valid_volume"])
        m2["volume"] = float(data["volume"]) if m2["valid_volume"] else None
        m2["area"] = float(data["area"])
        meta_with_vol.append(m2)

    with open(meta_path, "w") as f:
        json.dump(meta_with_vol, f, indent=2)

    total = len(meta_with_vol)
    valid_vol = sum(1 for m in meta_with_vol if m["valid_volume"])
    print(f"\n=== Preparation complete ===")
    print(f"  Total shapes processed : {total}")
    print(f"  Skipped (failed)       : {skipped}")
    print(f"  Valid volume (watertight): {valid_vol} / {total} ({100*valid_vol/max(1,total):.1f}%)")
    print(f"  Metadata saved to      : {meta_path}")

    if errors:
        err_path = out_dir / "prepare_errors.log"
        with open(err_path, "w") as f:
            f.write("\n".join(errors))
        print(f"  {len(errors)} errors logged to: {err_path}")


if __name__ == "__main__":
    main()
