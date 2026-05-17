"""
Create train/val/test splits and compute regression label statistics.

- Reads  data/processed/meta.json  (produced by prepare_data.py)
- ModelNet40's native split: train / test  (folder-based)
- Val: stratified 20% carved from native train, per class
- Computes log1p(volume) and log1p(area) mean/std on TRAIN split only
  → saved to  data/label_stats.json  (used to normalise targets at training time)
- Writes  data/splits.csv  with columns:
    npz_path, class_id, class_name, split, has_valid_volume

Run: python scripts/prepare_splits.py [--data_dir ...]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    parser = argparse.ArgumentParser(description="Create dataset splits and label stats.")
    parser.add_argument(
        "--data_dir",
        default=os.path.expanduser("~/geometric_deep_learning/data"),
        help="Root data directory (contains processed/ sub-folder)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of native train set to use as validation (default 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "processed" / "meta.json"
    splits_csv = data_dir / "splits.csv"
    stats_json = data_dir / "label_stats.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}\n"
            "Run scripts/prepare_data.py first."
        )

    print(f"Loading metadata from {meta_path} ...")
    with open(meta_path) as f:
        meta = json.load(f)

    df = pd.DataFrame(meta)
    df["has_valid_volume"] = df["valid_volume"].astype(bool)

    # Separate original train and test
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"  Native train : {len(train_df)}")
    print(f"  Native test  : {len(test_df)}")

    # Stratified val split from native train
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_fraction, random_state=args.seed
    )
    idx_train, idx_val = next(splitter.split(train_df, train_df["class_id"]))

    train_final = train_df.iloc[idx_train].copy()
    val_final = train_df.iloc[idx_val].copy()

    train_final["split"] = "train"
    val_final["split"] = "val"
    test_df["split"] = "test"

    print(f"  Final train  : {len(train_final)}")
    print(f"  Final val    : {len(val_final)}")
    print(f"  Final test   : {len(test_df)}")

    # Combine and save splits CSV
    all_df = pd.concat([train_final, val_final, test_df], ignore_index=True)
    cols = ["npz_path", "class_id", "class_name", "split", "has_valid_volume"]
    all_df[cols].to_csv(splits_csv, index=False)
    print(f"\nSplits saved to {splits_csv}")

    # Compute label normalisation stats on TRAIN split only (regression targets)
    train_reg = train_final[train_final["has_valid_volume"]].copy()
    print(f"\nComputing regression label stats on {len(train_reg)} watertight train shapes ...")

    volumes = train_reg["volume"].dropna().values.astype(np.float64)
    areas = train_reg["area"].dropna().values.astype(np.float64)

    log_volumes = np.log1p(np.abs(volumes))  # abs guards against tiny negatives from near-flat meshes
    log_areas = np.log1p(areas)

    stats = {
        "volume": {
            "log_mean": float(log_volumes.mean()),
            "log_std": float(log_volumes.std()),
            "raw_min": float(volumes.min()),
            "raw_max": float(volumes.max()),
            "n": int(len(volumes)),
        },
        "area": {
            "log_mean": float(log_areas.mean()),
            "log_std": float(log_areas.std()),
            "raw_min": float(areas.min()),
            "raw_max": float(areas.max()),
            "n": int(len(areas)),
        },
    }

    with open(stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Label stats saved to {stats_json}")

    print("\n=== Label Statistics (training set) ===")
    for target, s in stats.items():
        print(f"  {target:8s}: raw [{s['raw_min']:.4g}, {s['raw_max']:.4g}]  "
              f"log mean={s['log_mean']:.4f}  log std={s['log_std']:.4f}  n={s['n']}")

    # Per-class split distribution sanity check
    print("\n=== Per-split class distribution (first 5 classes) ===")
    dist = all_df.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    print(dist.head(5).to_string())
    print(f"  ... ({len(dist)} classes total)")


if __name__ == "__main__":
    main()
