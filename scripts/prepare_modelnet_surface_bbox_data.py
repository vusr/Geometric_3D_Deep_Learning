"""
Prepare scale-preserving ModelNet40 point clouds for surface/bounding-box regression.

This is intentionally separate from scripts/prepare_data.py. It does not require
watertight meshes and it does not normalize each point cloud to a unit sphere.
The downstream task predicts:
  - surface_area
  - bbox_volume
"""

import argparse
import json
import os
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


def center_point_cloud(points: np.ndarray) -> np.ndarray:
    return points - points.mean(axis=0, keepdims=True)


def process_one(args):
    off_path, out_path, n_points = args
    try:
        mesh = trimesh.load(str(off_path), force="mesh", process=True)
        if len(mesh.faces) == 0:
            return f"SKIP (no faces): {off_path}"

        surface_area = float(mesh.area)
        bbox_extents = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
        bbox_volume = float(np.prod(bbox_extents))
        bbox_diag = float(np.linalg.norm(bbox_extents))

        if not np.isfinite(surface_area) or not np.isfinite(bbox_volume):
            return f"SKIP (non-finite target): {off_path}"

        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        points = center_point_cloud(np.asarray(points, dtype=np.float32))

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            points_centered=points.astype(np.float32),
            surface_area=np.float32(surface_area),
            bbox_extents=bbox_extents.astype(np.float32),
            bbox_volume=np.float32(bbox_volume),
            bbox_diag=np.float32(bbox_diag),
        )
        return None
    except Exception as exc:
        return f"ERROR [{off_path}]: {exc}\n{traceback.format_exc()}"


def build_file_list(raw_dir: Path, out_dir: Path, n_points: int):
    tasks = []
    meta = []

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower().replace(" ", "_")
        if class_name not in CLASS_TO_ID:
            print(f"  WARNING: unknown class '{class_dir.name}', skipping.")
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
                        "native_split": split,
                        "off_path": str(off_file),
                    }
                )

    return tasks, meta


def main():
    parser = argparse.ArgumentParser(description="Prepare ModelNet40 surface/bbox regression data.")
    parser.add_argument(
        "--raw_dir",
        default=os.path.expanduser("~/geometric_deep_learning/data/raw/ModelNet40"),
        help="Path to extracted ModelNet40 directory.",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.expanduser("~/geometric_deep_learning/data/modelnet_surface_bbox/processed"),
        help="Output directory for processed NPZ files.",
    )
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    print(f"Building file list from {raw_dir} ...")
    tasks, meta = build_file_list(raw_dir, out_dir, args.n_points)
    print(f"  Found {len(tasks)} candidate shapes.")

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

    print("Building metadata index...")
    meta_rows = []
    skipped = 0
    for row in tqdm(meta, desc="Reading metadata"):
        npz_path = Path(row["npz_path"])
        if not npz_path.exists():
            skipped += 1
            continue
        data = np.load(str(npz_path))
        item = dict(row)
        item["surface_area"] = float(data["surface_area"])
        item["bbox_volume"] = float(data["bbox_volume"])
        item["bbox_diag"] = float(data["bbox_diag"])
        item["bbox_dx"] = float(data["bbox_extents"][0])
        item["bbox_dy"] = float(data["bbox_extents"][1])
        item["bbox_dz"] = float(data["bbox_extents"][2])
        meta_rows.append(item)

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_rows, f, indent=2)

    if errors:
        err_path = out_dir / "prepare_errors.log"
        with open(err_path, "w") as f:
            f.write("\n".join(errors))
        print(f"  {len(errors)} errors logged to: {err_path}")

    print("\n=== Preparation complete ===")
    print(f"  Processed usable shapes : {len(meta_rows)}")
    print(f"  Skipped/failed shapes   : {skipped + len(errors)}")
    print(f"  Metadata saved to       : {meta_path}")


if __name__ == "__main__":
    main()
