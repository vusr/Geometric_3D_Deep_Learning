"""Create ShapeNetSem 60/20/20 splits, trim outliers, and write stats."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGETS = (
    "solidVolume",
    "surfaceVolume",
    "supportSurfaceArea",
    "aligned_dim_x",
    "aligned_dim_y",
    "aligned_dim_z",
)


def add_log_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for target in TARGETS:
        df[f"log_{target}"] = np.log1p(df[target].astype(np.float64))
    return df


def summary(df: pd.DataFrame) -> dict:
    out = {"n": int(len(df))}
    for target in TARGETS:
        values = df[target].to_numpy(dtype=np.float64)
        out[target] = {
            "min": float(np.min(values)) if len(values) else None,
            "p01": float(np.percentile(values, 1)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p99": float(np.percentile(values, 99)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None,
        }
    return out


def strat_labels(df: pd.DataFrame, min_count: int = 3) -> pd.Series:
    labels = df["class_name"].fillna("unknown").astype(str)
    counts = labels.value_counts()
    return labels.where(labels.map(counts) >= min_count, "other")


def split_602020(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    labels = strat_labels(df)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=seed,
        stratify=labels if labels.nunique() > 1 else None,
    )
    temp_labels = strat_labels(temp_df, min_count=2)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels if temp_labels.nunique() > 1 else None,
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def robust_outlier_mask(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    train = df[df["split"] == "train"]
    mask = pd.Series(True, index=df.index)
    fences = {}
    finite_positive = np.isfinite(df[list(TARGETS)]).all(axis=1)
    for target in TARGETS:
        finite_positive &= df[target] > 0
    mask &= finite_positive

    for target in TARGETS:
        log_target = f"log_{target}"
        q1 = float(train[log_target].quantile(0.25))
        q3 = float(train[log_target].quantile(0.75))
        iqr = q3 - q1
        lo = q1 - 3.0 * iqr
        hi = q3 + 3.0 * iqr
        fences[log_target] = {"q1": q1, "q3": q3, "iqr": float(iqr), "lo": lo, "hi": hi}
        mask &= (df[log_target] >= lo) & (df[log_target] <= hi)
    return mask, fences


def build_norm_stats(df: pd.DataFrame) -> dict:
    train = df[df["split"] == "train"]
    stats = {
        "target_names": list(TARGETS),
        "input_scale_policy": "median train aligned bbox diagonal in meters; ShapeNetSem aligned.dims CSV values converted from cm to m",
        "global": {"n": int(len(train))},
    }
    for target in TARGETS:
        logs = train[f"log_{target}"].to_numpy(dtype=np.float64)
        raw = train[target].to_numpy(dtype=np.float64)
        stats["global"][target] = {
            "log_mean": float(logs.mean()),
            "log_std": float(max(logs.std(), 1e-8)),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "n": int(len(raw)),
        }
    diag = np.sqrt(
        train["aligned_dim_x"].to_numpy(dtype=np.float64) ** 2
        + train["aligned_dim_y"].to_numpy(dtype=np.float64) ** 2
        + train["aligned_dim_z"].to_numpy(dtype=np.float64) ** 2
    )
    input_scale = float(np.median(diag))
    stats["global"]["input_scale"] = input_scale if input_scale > 0 else 1.0
    return stats


def plot_artifacts(df_all: pd.DataFrame, retained: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        log_target = f"log_{target}"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df_all[log_target], bins=80, color="#4C78A8", alpha=0.8)
        axes[0].set_title(f"Before trim: {log_target}")
        axes[1].hist(retained[log_target], bins=80, color="#59A14F", alpha=0.8)
        axes[1].set_title(f"After trim: {log_target}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{target}_log_hist.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze ShapeNetSem regression targets.")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/shapenetsem_regression/data/shapenetsem_regression"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "processed" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)
    df = pd.DataFrame(meta)
    for target in TARGETS:
        df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[np.isfinite(df[list(TARGETS)]).all(axis=1)].copy()
    for target in TARGETS:
        df = df[df[target] > 0].copy()
    df = add_log_targets(df)
    split_df = split_602020(df, seed=args.seed)

    before = summary(split_df)
    mask, fences = robust_outlier_mask(split_df)
    retained = split_df[mask].copy().reset_index(drop=True)
    removed = split_df[~mask].copy().reset_index(drop=True)
    after = summary(retained)
    norm_stats = build_norm_stats(retained)

    cols = [
        "npz_path", "obj_path", "fullId", "shape_id", "class_name", "category",
        "split", *TARGETS,
    ]
    retained[cols].to_csv(data_dir / "splits.csv", index=False)
    removed[[c for c in cols if c in removed.columns]].to_csv(data_dir / "outliers_removed.csv", index=False)
    with open(data_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    split_counts = retained["split"].value_counts().to_dict()
    total = max(len(retained), 1)
    analysis = {
        "before_outlier_removal": before,
        "after_outlier_removal": after,
        "outlier_fences": fences,
        "removed_count": int(len(removed)),
        "retained_count": int(len(retained)),
        "split_counts": {k: int(v) for k, v in sorted(split_counts.items())},
        "split_fractions": {k: float(v / total) for k, v in sorted(split_counts.items())},
        "target_names": list(TARGETS),
    }
    with open(data_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)
    plot_artifacts(split_df, retained, data_dir / "analysis_plots")

    print("\n=== ShapeNetSem target analysis complete ===")
    print(f"  Retained samples : {len(retained)}")
    print(f"  Removed outliers : {len(removed)}")
    print(f"  Split counts     : {analysis['split_counts']}")
    print(f"  Splits saved to  : {data_dir / 'splits.csv'}")
    print(f"  Stats saved to   : {data_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
