"""Prepare scale-preserving ShapeNetSem point clouds for regression."""

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm


TARGETS = (
    "solidVolume",
    "surfaceVolume",
    "supportSurfaceArea",
    "aligned_dim_x",
    "aligned_dim_y",
    "aligned_dim_z",
)

ALIGNED_DIMS_TO_METERS = 0.01


def parse_vec(value, expected=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    parts = [p.strip() for p in str(value).replace("\\,", ",").split(",") if p.strip()]
    if len(parts) != expected:
        return None
    try:
        arr = np.asarray([float(p) for p in parts], dtype=np.float64)
    except ValueError:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def primary_category(value: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).split(",")[0].strip().strip('"')


def model_id(full_id: str) -> str:
    return str(full_id).split(".")[-1]


def clean_metadata(metadata_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    rows = []
    for _, row in df.iterrows():
        dims = parse_vec(row.get("aligned.dims"))
        if dims is None:
            continue
        item = row.to_dict()
        item["shape_id"] = model_id(item["fullId"])
        item["class_name"] = primary_category(item.get("category", ""))
        item["aligned_dim_x"] = float(dims[0] * ALIGNED_DIMS_TO_METERS)
        item["aligned_dim_y"] = float(dims[1] * ALIGNED_DIMS_TO_METERS)
        item["aligned_dim_z"] = float(dims[2] * ALIGNED_DIMS_TO_METERS)
        rows.append(item)

    out = pd.DataFrame(rows)
    required = ["fullId", "shape_id", "class_name", *TARGETS]
    out = out.dropna(subset=required)
    for target in TARGETS:
        out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out[np.isfinite(out[list(TARGETS)]).all(axis=1)].copy()
    for target in TARGETS:
        out = out[out[target] > 0].copy()
    out = out[out["class_name"].astype(str).str.len() > 0].copy()
    return out.reset_index(drop=True)


def index_obj_files(obj_dir: Path) -> dict[str, Path]:
    mapping = {}
    for path in obj_dir.rglob("*.obj"):
        parts = {p.lower() for p in path.parts}
        for part in parts:
            if len(part) >= 16 and all(ch in "0123456789abcdef" for ch in part.lower()):
                mapping.setdefault(part.lower(), path)
        mapping.setdefault(path.stem.lower(), path)
    return mapping


def alignment_matrix(up_value, front_value):
    up = parse_vec(up_value)
    front = parse_vec(front_value)
    if up is None or front is None:
        return None
    up_norm = np.linalg.norm(up)
    front_norm = np.linalg.norm(front)
    if up_norm <= 0 or front_norm <= 0:
        return None
    z_axis = up / up_norm
    front = front / front_norm
    y_axis = -front
    y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm <= 1e-8:
        return None
    y_axis = y_axis / y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 1e-8:
        return None
    x_axis = x_axis / x_norm
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def process_one(args):
    row, obj_path, out_path, n_points = args
    try:
        mesh = trimesh.load(str(obj_path), force="mesh", process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if len(mesh.faces) == 0:
            return None, f"SKIP no faces: {obj_path}"

        points, _ = trimesh.sample.sample_surface(mesh, n_points)
        points = np.asarray(points, dtype=np.float64)
        unit = float(row.get("unit", 1.0))
        if np.isfinite(unit) and unit > 0:
            points *= unit

        matrix = alignment_matrix(row.get("up"), row.get("front"))
        if matrix is not None:
            points = points @ matrix

        points = points - points.mean(axis=0, keepdims=True)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            points_centered=points.astype(np.float32),
            targets=np.asarray([float(row[t]) for t in TARGETS], dtype=np.float32),
        )
        meta = {k: row.get(k) for k in row.keys()}
        meta["npz_path"] = str(out_path)
        meta["obj_path"] = str(obj_path)
        return meta, None
    except Exception as exc:
        return None, f"ERROR {obj_path}: {exc}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="Prepare ShapeNetSem regression point clouds.")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/shapenetsem_regression/data/shapenetsem_regression"))
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    out_dir = data_dir / "processed"
    metadata_csv = raw_dir / "metadata.csv"
    obj_dir = raw_dir / "models-OBJ"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata.csv: {metadata_csv}")
    if not obj_dir.exists():
        raise FileNotFoundError(f"Missing extracted OBJ directory: {obj_dir}")

    print("Reading metadata...")
    df = clean_metadata(metadata_csv)
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"Usable metadata rows before OBJ match: {len(df)}")

    print("Indexing OBJ files...")
    obj_map = index_obj_files(obj_dir)
    tasks = []
    missing = 0
    for _, row in df.iterrows():
        sid = row["shape_id"].lower()
        obj_path = obj_map.get(sid)
        if obj_path is None:
            missing += 1
            continue
        out_path = out_dir / row["class_name"].replace("/", "_") / f"{sid}.npz"
        if out_path.exists():
            meta = row.to_dict()
            meta["npz_path"] = str(out_path)
            meta["obj_path"] = str(obj_path)
            tasks.append((meta, None, None, None))
        else:
            tasks.append((row.to_dict(), str(obj_path), str(out_path), args.n_points))

    print(f"Matched OBJ files: {len(tasks)} | missing OBJ: {missing}")
    meta_rows = []
    errors = []
    real_tasks = [t for t in tasks if t[1] is not None]
    cached = [t[0] for t in tasks if t[1] is None]
    meta_rows.extend(cached)

    if real_tasks:
        if args.workers == 1:
            iterator = (process_one(t) for t in real_tasks)
            for meta, err in tqdm(iterator, total=len(real_tasks), desc="Processing"):
                if meta:
                    meta_rows.append(meta)
                if err:
                    errors.append(err)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(process_one, t): t for t in real_tasks}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    meta, err = future.result()
                    if meta:
                        meta_rows.append(meta)
                    if err:
                        errors.append(err)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_rows, f, indent=2)
    if errors:
        (out_dir / "prepare_errors.log").write_text("\n".join(errors))

    print("\n=== ShapeNetSem preparation complete ===")
    print(f"  Prepared usable shapes : {len(meta_rows)}")
    print(f"  Errors/skips           : {len(errors)}")
    print(f"  Metadata saved to      : {meta_path}")


if __name__ == "__main__":
    main()
