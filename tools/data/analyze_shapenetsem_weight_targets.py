"""Build ShapeNetSem weight splits and class/material feature statistics."""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def key(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def category_tokens(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip().strip('"') for part in str(value).split(",") if part.strip()]


def build_material_lookup(materials: pd.DataFrame, densities: pd.DataFrame) -> tuple[dict, list[str]]:
    material_names = sorted(densities["Material"].dropna().astype(str).unique().tolist())
    density_map = densities.set_index("Material")["Density"].astype(float).to_dict()
    friction_map = densities.set_index("Material")["StaticFrictionCoeff"].astype(float).to_dict()

    lookup = {}
    for category, group in materials.groupby("Category"):
        ratios = {name: 0.0 for name in material_names}
        for _, row in group.iterrows():
            material = str(row["Material"])
            if material in ratios:
                ratios[material] += float(row["Ratio"])
        total = sum(ratios.values())
        if total > 0:
            ratios = {name: value / total for name, value in ratios.items()}
        density = sum(ratios[name] * density_map.get(name, 0.0) for name in material_names)
        friction = sum(ratios[name] * friction_map.get(name, 0.0) for name in material_names)
        lookup[key(category)] = {
            "ratios": ratios,
            "material_density_mean": density,
            "material_friction_coeff_mean": friction,
        }
    return lookup, material_names


def material_for_row(row, lookup: dict, material_names: list[str]) -> dict:
    candidates = [row.get("class_name", ""), *category_tokens(row.get("category", ""))]
    for candidate in candidates:
        entry = lookup.get(key(candidate))
        if entry is not None:
            out = {f"material_ratio_{name}": entry["ratios"].get(name, 0.0) for name in material_names}
            out["material_density_mean"] = entry["material_density_mean"]
            out["material_friction_coeff_mean"] = entry["material_friction_coeff_mean"]
            out["has_material_prior"] = 1.0
            return out
    out = {f"material_ratio_{name}": 0.0 for name in material_names}
    out["material_density_mean"] = 0.0
    out["material_friction_coeff_mean"] = 0.0
    out["has_material_prior"] = 0.0
    return out


def robust_weight_mask(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    train = df[df["split"] == "train"]
    q1 = float(train["log_weight"].quantile(0.25))
    q3 = float(train["log_weight"].quantile(0.75))
    iqr = q3 - q1
    lo = q1 - 3.0 * iqr
    hi = q3 + 3.0 * iqr
    mask = (df["log_weight"] >= lo) & (df["log_weight"] <= hi)
    return mask, {"q1": q1, "q3": q3, "iqr": float(iqr), "lo": lo, "hi": hi}


def main():
    parser = argparse.ArgumentParser(description="Analyze ShapeNetSem weight regression target.")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/shapenetsem_regression/data/shapenetsem_regression"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits = pd.read_csv(data_dir / "splits.csv")
    metadata = pd.read_csv(data_dir / "raw" / "metadata.csv")
    materials = pd.read_csv(data_dir / "raw" / "materials.csv")
    densities = pd.read_csv(data_dir / "raw" / "densities.csv")

    df = splits.merge(metadata[["fullId", "weight"]], on="fullId", how="left")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df[np.isfinite(df["weight"]) & (df["weight"] > 0)].copy()
    df["log_weight"] = np.log1p(df["weight"].astype(np.float64))

    material_lookup, material_names = build_material_lookup(materials, densities)
    feature_rows = [material_for_row(row, material_lookup, material_names) for _, row in df.iterrows()]
    feature_df = pd.DataFrame(feature_rows, index=df.index)
    df = pd.concat([df, feature_df], axis=1)

    mask, fence = robust_weight_mask(df)
    retained = df[mask].copy().reset_index(drop=True)
    removed = df[~mask].copy().reset_index(drop=True)

    raw_aux_columns = [c for c in feature_df.columns]
    train = retained[retained["split"] == "train"]
    aux_columns = []
    aux_stats = {}
    for col in raw_aux_columns:
        values = train[col].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        std = float(max(values.std(), 1e-8))
        aux_col = f"aux_{col}"
        retained[aux_col] = (retained[col].astype(np.float64) - mean) / std
        aux_columns.append(aux_col)
        aux_stats[col] = {"mean": mean, "std": std}

    class_names = sorted(train["class_name"].astype(str).unique().tolist())
    class_to_idx = {"__unknown__": 0}
    class_to_idx.update({name: i + 1 for i, name in enumerate(class_names)})

    logs = train["log_weight"].to_numpy(dtype=np.float64)
    raw = train["weight"].to_numpy(dtype=np.float64)
    diag = np.sqrt(
        train["aligned_dim_x"].to_numpy(dtype=np.float64) ** 2
        + train["aligned_dim_y"].to_numpy(dtype=np.float64) ** 2
        + train["aligned_dim_z"].to_numpy(dtype=np.float64) ** 2
    )
    stats = {
        "target_name": "weight",
        "target": {
            "log_mean": float(logs.mean()),
            "log_std": float(max(logs.std(), 1e-8)),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "n": int(len(raw)),
        },
        "input_scale": float(np.median(diag)) if len(diag) else 1.0,
        "aux_columns": aux_columns,
        "raw_aux_columns": raw_aux_columns,
        "aux_stats": aux_stats,
        "class_to_idx": class_to_idx,
        "material_names": material_names,
        "notes": "Aux features are train-normalized material ratios plus material density/friction priors. staticFrictionForce is excluded to avoid target leakage.",
    }

    cols = [
        "npz_path", "obj_path", "fullId", "shape_id", "class_name", "category", "split",
        "weight", "solidVolume", "surfaceVolume", "supportSurfaceArea",
        "aligned_dim_x", "aligned_dim_y", "aligned_dim_z",
        *raw_aux_columns, *aux_columns,
    ]
    retained[cols].to_csv(data_dir / "weight_splits.csv", index=False)
    removed[[c for c in cols if c in removed.columns]].to_csv(data_dir / "weight_outliers_removed.csv", index=False)
    with open(data_dir / "weight_norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    counts = retained["split"].value_counts().to_dict()
    summary = {
        "retained_count": int(len(retained)),
        "removed_count": int(len(removed)),
        "split_counts": {k: int(v) for k, v in sorted(counts.items())},
        "split_fractions": {k: float(v / max(len(retained), 1)) for k, v in sorted(counts.items())},
        "has_material_prior_count": int(retained["has_material_prior"].sum()),
        "class_count_train": int(len(class_names)),
        "weight_log_fence": fence,
        "target": "weight",
    }
    with open(data_dir / "weight_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== ShapeNetSem weight analysis complete ===")
    print(f"  Retained samples       : {len(retained)}")
    print(f"  Removed weight outliers: {len(removed)}")
    print(f"  Split counts           : {summary['split_counts']}")
    print(f"  Material prior rows    : {summary['has_material_prior_count']}")
    print(f"  Splits saved to        : {data_dir / 'weight_splits.csv'}")
    print(f"  Stats saved to         : {data_dir / 'weight_norm_stats.json'}")


if __name__ == "__main__":
    main()
