"""
Create splits, remove target outliers, analyze clusters, and write stats.

The output drives the ModelNet surface_area + bbox_volume experiment. One global
model is preferred unless the retained log-target distribution clearly supports
separated, sufficiently large clusters.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


TARGETS = ("surface_area", "bbox_volume")
LOG_TARGETS = ("log_surface_area", "log_bbox_volume")


def target_summary(df: pd.DataFrame) -> dict:
    summary = {"n": int(len(df))}
    for target in TARGETS:
        values = df[target].to_numpy(dtype=np.float64)
        summary[target] = {
            "min": float(np.min(values)) if len(values) else None,
            "p01": float(np.percentile(values, 1)) if len(values) else None,
            "p25": float(np.percentile(values, 25)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p75": float(np.percentile(values, 75)) if len(values) else None,
            "p99": float(np.percentile(values, 99)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None,
            "mean": float(np.mean(values)) if len(values) else None,
        }
    return summary


def add_log_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_surface_area"] = np.log1p(df["surface_area"].astype(np.float64))
    df["log_bbox_volume"] = np.log1p(df["bbox_volume"].astype(np.float64))
    return df


def split_native_train(df: pd.DataFrame, val_fraction: float, seed: int) -> pd.DataFrame:
    native_train = df[df["native_split"] == "train"].copy()
    native_test = df[df["native_split"] == "test"].copy()

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    idx_train, idx_val = next(splitter.split(native_train, native_train["class_id"]))

    train_df = native_train.iloc[idx_train].copy()
    val_df = native_train.iloc[idx_val].copy()
    test_df = native_test.copy()
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

    for log_target in LOG_TARGETS:
        q1 = float(train[log_target].quantile(0.25))
        q3 = float(train[log_target].quantile(0.75))
        iqr = q3 - q1
        lo = q1 - 3.0 * iqr
        hi = q3 + 3.0 * iqr
        fences[log_target] = {"q1": q1, "q3": q3, "iqr": float(iqr), "lo": lo, "hi": hi}
        mask &= (df[log_target] >= lo) & (df[log_target] <= hi)

    return mask, fences


def fit_gmms(train_xy: np.ndarray, seed: int, max_k: int) -> list[dict]:
    candidates = []
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=5)
        labels = gmm.fit_predict(train_xy)
        probs = gmm.predict_proba(train_xy)
        candidates.append(
            {
                "k": k,
                "model": gmm,
                "labels": labels,
                "bic": float(gmm.bic(train_xy)),
                "aic": float(gmm.aic(train_xy)),
                "mean_max_posterior": float(probs.max(axis=1).mean()),
            }
        )
    return candidates


def min_centroid_distance(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return 0.0
    best = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            best = min(best, float(np.linalg.norm(centers[i] - centers[j])))
    return best


def merge_small_clusters(
    labels: np.ndarray,
    centers: np.ndarray,
    train_xy: np.ndarray,
    min_train_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.copy()
    centers = centers.copy()

    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = [(cluster_id, count) for cluster_id, count in zip(ids, counts) if count < min_train_size]
        if not small or len(ids) == 1:
            break

        small_id, _ = min(small, key=lambda x: x[1])
        other_ids = [i for i in ids if i != small_id]
        distances = [
            (other_id, float(np.linalg.norm(centers[small_id] - centers[other_id])))
            for other_id in other_ids
        ]
        target_id = min(distances, key=lambda x: x[1])[0]
        labels[labels == small_id] = target_id

        new_ids = sorted(np.unique(labels))
        remap = {old: new for new, old in enumerate(new_ids)}
        labels = np.array([remap[x] for x in labels], dtype=np.int64)
        centers = np.vstack([train_xy[labels == i].mean(axis=0) for i in sorted(np.unique(labels))])

    return labels, centers


def decide_clusters(df: pd.DataFrame, seed: int, max_k: int, min_train_size: int) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    train = df[df["split"] == "train"].copy()
    scaler = StandardScaler()
    train_xy = scaler.fit_transform(train[list(LOG_TARGETS)].to_numpy(dtype=np.float64))

    candidates = fit_gmms(train_xy, seed=seed, max_k=max_k)
    k1_bic = candidates[0]["bic"]
    best = min(candidates, key=lambda item: item["bic"])
    bic_improvement = (k1_bic - best["bic"]) / max(abs(k1_bic), 1e-8)

    accepted = False
    final_k = 1
    decision_reason = "single cluster preferred"
    final_model = None
    final_scaler = scaler
    final_train_labels = np.zeros(len(train), dtype=np.int64)
    final_centers = np.zeros((1, train_xy.shape[1]), dtype=np.float64)

    candidate_summaries = []
    for item in candidates:
        centers = item["model"].means_
        candidate_summaries.append(
            {
                "k": item["k"],
                "bic": item["bic"],
                "aic": item["aic"],
                "mean_max_posterior": item["mean_max_posterior"],
                "min_centroid_distance": min_centroid_distance(centers),
            }
        )

    if best["k"] > 1:
        raw_centers = best["model"].means_
        merged_labels, merged_centers = merge_small_clusters(
            best["labels"], raw_centers, train_xy=train_xy, min_train_size=min_train_size
        )
        merged_k = int(len(np.unique(merged_labels)))
        min_distance = min_centroid_distance(merged_centers)
        counts = np.bincount(merged_labels)

        if (
            bic_improvement >= 0.05
            and best["mean_max_posterior"] >= 0.75
            and min_distance >= 1.5
            and merged_k > 1
            and counts.min() >= min_train_size
        ):
            accepted = True
            final_k = merged_k
            final_model = best["model"]
            final_train_labels = merged_labels
            final_centers = merged_centers
            decision_reason = (
                "multi-cluster model accepted: BIC improvement, posterior confidence, "
                "centroid separation, and minimum train size thresholds passed"
            )
        else:
            decision_reason = (
                "global model selected: candidate clusters were weak, overlapping, "
                "or too small after merging"
            )

    train_indices = train.index.to_numpy()
    df["cluster_id"] = 0
    if accepted and final_model is not None:
        all_xy = final_scaler.transform(df[list(LOG_TARGETS)].to_numpy(dtype=np.float64))
        predicted = final_model.predict(all_xy)
        original_to_final = {}
        for original_label, final_label in zip(best["labels"], final_train_labels):
            original_to_final.setdefault(int(original_label), int(final_label))
        mapped = np.array([original_to_final.get(int(label), 0) for label in predicted], dtype=np.int64)
        df["cluster_id"] = mapped
        df.loc[train_indices, "cluster_id"] = final_train_labels

    final_counts = (
        df.groupby(["cluster_id", "split"]).size().unstack(fill_value=0).sort_index().to_dict(orient="index")
    )
    cluster_ranges = {}
    for cluster_id, group in df.groupby("cluster_id"):
        cluster_ranges[str(int(cluster_id))] = target_summary(group)

    decision = {
        "recommendation": "cluster_models" if accepted else "global_model",
        "reason": decision_reason,
        "selected_k_before_merge": int(best["k"]),
        "final_k": int(final_k),
        "bic_k1": float(k1_bic),
        "best_bic": float(best["bic"]),
        "bic_improvement_fraction": float(bic_improvement),
        "mean_max_posterior": float(best["mean_max_posterior"]),
        "min_centroid_distance": float(min_centroid_distance(final_centers)),
        "min_train_size": int(min_train_size),
        "candidate_models": candidate_summaries,
        "final_cluster_counts": {
            str(k): {split: int(v) for split, v in counts.items()}
            for k, counts in final_counts.items()
        },
        "final_cluster_ranges": cluster_ranges,
    }
    return df, decision


def build_norm_stats(df: pd.DataFrame, recommendation: str) -> dict:
    stats = {
        "target_names": list(TARGETS),
        "input_scale_policy": "median train bbox_diag",
        "global": stats_for_group(df[df["split"] == "train"]),
        "clusters_enabled": recommendation == "cluster_models",
        "clusters": {},
    }
    for cluster_id, group in df[df["split"] == "train"].groupby("cluster_id"):
        stats["clusters"][str(int(cluster_id))] = stats_for_group(group)
    return stats


def stats_for_group(train: pd.DataFrame) -> dict:
    result = {}
    for target, log_target in zip(TARGETS, LOG_TARGETS):
        values = train[target].to_numpy(dtype=np.float64)
        logs = train[log_target].to_numpy(dtype=np.float64)
        result[target] = {
            "log_mean": float(logs.mean()),
            "log_std": float(max(logs.std(), 1e-8)),
            "raw_min": float(values.min()),
            "raw_max": float(values.max()),
            "n": int(len(values)),
        }
    bbox_diag = train["bbox_diag"].to_numpy(dtype=np.float64)
    input_scale = float(np.median(bbox_diag))
    result["input_scale"] = input_scale if input_scale > 0 else 1.0
    result["n"] = int(len(train))
    return result


def plot_artifacts(df_all: pd.DataFrame, df_retained: pd.DataFrame, out_dir: Path, clusters_enabled: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    for target, log_target in zip(TARGETS, LOG_TARGETS):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df_all[log_target], bins=80, color="#4C78A8", alpha=0.8)
        axes[0].set_title(f"Before outlier trim: {log_target}")
        axes[0].set_xlabel(log_target)
        axes[0].set_ylabel("count")
        axes[1].hist(df_retained[log_target], bins=80, color="#59A14F", alpha=0.8)
        axes[1].set_title(f"After outlier trim: {log_target}")
        axes[1].set_xlabel(log_target)
        fig.tight_layout()
        fig.savefig(out_dir / f"{target}_log_hist.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df_retained["log_surface_area"], df_retained["log_bbox_volume"], s=8, alpha=0.35)
    ax.set_xlabel("log_surface_area")
    ax.set_ylabel("log_bbox_volume")
    ax.set_title("Retained target distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "target_scatter_retained.png", dpi=160)
    plt.close(fig)

    if clusters_enabled:
        fig, ax = plt.subplots(figsize=(7, 6))
        for cluster_id, group in df_retained.groupby("cluster_id"):
            ax.scatter(
                group["log_surface_area"],
                group["log_bbox_volume"],
                s=8,
                alpha=0.45,
                label=f"cluster {cluster_id}",
            )
        ax.set_xlabel("log_surface_area")
        ax.set_ylabel("log_bbox_volume")
        ax.set_title("Accepted target clusters")
        ax.legend(markerscale=2)
        fig.tight_layout()
        fig.savefig(out_dir / "target_clusters.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze ModelNet surface/bbox targets.")
    parser.add_argument(
        "--data_dir",
        default=os.path.expanduser("~/geometric_deep_learning/artifacts/modelnet40_regression/data_metadata"),
    )
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_clusters", type=int, default=6)
    parser.add_argument("--min_train_size", type=int, default=1000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"
    meta_path = processed_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)
    df = pd.DataFrame(meta)
    df = add_log_targets(df)
    split_df = split_native_train(df, args.val_fraction, args.seed)

    before_summary = target_summary(split_df)
    mask, fences = robust_outlier_mask(split_df)
    retained = split_df[mask].copy().reset_index(drop=True)
    removed = split_df[~mask].copy().reset_index(drop=True)

    clustered, cluster_decision = decide_clusters(
        retained,
        seed=args.seed,
        max_k=args.max_clusters,
        min_train_size=args.min_train_size,
    )
    after_summary = target_summary(clustered)
    norm_stats = build_norm_stats(clustered, cluster_decision["recommendation"])

    data_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "npz_path", "class_id", "class_name", "split", "surface_area", "bbox_volume",
        "bbox_diag", "bbox_dx", "bbox_dy", "bbox_dz", "cluster_id",
    ]
    clustered[cols].to_csv(data_dir / "splits.csv", index=False)
    removed[cols[:-1] if "cluster_id" not in removed.columns else cols].to_csv(
        data_dir / "outliers_removed.csv", index=False
    )
    clustered[cols].to_csv(data_dir / "cluster_manifest.csv", index=False)
    with open(data_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    summary = {
        "before_outlier_removal": before_summary,
        "after_outlier_removal": after_summary,
        "outlier_fences": fences,
        "removed_count": int(len(removed)),
        "retained_count": int(len(clustered)),
        "split_counts": {
            split: int(count) for split, count in clustered["split"].value_counts().sort_index().items()
        },
        "cluster_decision": cluster_decision,
    }
    with open(data_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_artifacts(
        split_df,
        clustered,
        data_dir / "analysis_plots",
        clusters_enabled=cluster_decision["recommendation"] == "cluster_models",
    )

    print("\n=== ModelNet Surface/BBox Analysis Complete ===")
    print(f"  Retained samples : {len(clustered)}")
    print(f"  Removed outliers : {len(removed)}")
    print(f"  Recommendation   : {cluster_decision['recommendation']}")
    print(f"  Reason           : {cluster_decision['reason']}")
    print(f"  Splits saved to  : {data_dir / 'splits.csv'}")
    print(f"  Stats saved to   : {data_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
