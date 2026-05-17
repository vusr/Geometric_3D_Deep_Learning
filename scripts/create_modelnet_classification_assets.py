"""
Create ModelNet40 classification visualization assets from saved results.

Expected inputs:
  logs/cls_train_log.csv
  results/cls_per_class_acc.csv
  results/cls_confusion_matrix.csv
  results/cls_sample_predictions.csv

Outputs:
  assets/modelnet_classification/training_curves.png
  assets/modelnet_classification/confusion_matrix.png
  assets/modelnet_classification/per_class_accuracy.png
  assets/modelnet_classification/sample_prediction_gallery_combined.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def set_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
    })


def save_training_curves(log_path: Path, output_path: Path):
    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train", color="#2f5597", linewidth=2)
    axes[0].plot(df["epoch"], df["val_loss"], label="Validation", color="#c43c39", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(df["epoch"], df["train_acc"] * 100.0, label="Train", color="#2f5597", linewidth=2)
    axes[1].plot(df["epoch"], df["val_acc"] * 100.0, label="Validation", color="#c43c39", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(frameon=False)

    fig.suptitle("PointNet++ MSG Training Curves", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def read_confusion_matrix(cm_path: Path):
    df = pd.read_csv(cm_path, index_col=0)
    return df.index.tolist(), df.values.astype(float)


def save_confusion_matrix(cm_path: Path, output_path: Path):
    labels, cm = read_confusion_matrix(cm_path)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(normalized, cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0, vmax=1)
    ax.set_title("ModelNet40 Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.grid(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized accuracy")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_per_class_accuracy(per_class_path: Path, output_path: Path):
    df = pd.read_csv(per_class_path)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df = df.sort_values("accuracy", ascending=True)
    colors = np.where(df["accuracy"] >= df["accuracy"].median(), "#2d7f5e", "#b64b4b")

    fig_height = max(7, 0.23 * len(df))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(df["class_name"], df["accuracy"] * 100.0, color=colors, alpha=0.9)
    ax.set_title("Per-Class Test Accuracy")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def resolve_npz_path(path_text: str, run_dir: Path, data_dir: Path):
    path = Path(path_text)
    candidates = [
        path,
        run_dir / path,
        data_dir / path,
        data_dir / "processed" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate point cloud file for gallery: {path_text}")


def choose_gallery_samples(samples: pd.DataFrame, n_correct: int = 5, n_wrong: int = 5):
    samples["correct"] = samples["correct"].astype(int)
    samples["confidence"] = pd.to_numeric(samples["confidence"], errors="coerce").fillna(0.0)

    correct = samples[samples["correct"] == 1].sort_values("confidence", ascending=False).head(n_correct)
    wrong = samples[samples["correct"] == 0].sort_values("confidence", ascending=False).head(n_wrong)
    gallery = pd.concat([correct, wrong])
    if len(gallery) < n_correct + n_wrong:
        remainder = samples.drop(gallery.index, errors="ignore").head(n_correct + n_wrong - len(gallery))
        gallery = pd.concat([gallery, remainder])
    return gallery.head(n_correct + n_wrong).reset_index(drop=True)


def project_top_corner(points: np.ndarray, azimuth_deg: float = -45.0, elevation_deg: float = 35.0):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    rz = np.array([
        [np.cos(az), -np.sin(az), 0.0],
        [np.sin(az), np.cos(az), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(el), -np.sin(el)],
        [0.0, np.sin(el), np.cos(el)],
    ])
    rotated = points @ rz.T @ rx.T
    return rotated[:, :2], rotated[:, 2]


def set_combined_axis_style(ax, correct: bool):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    border = "#2d7f5e" if correct else "#b64b4b"
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color(border)


def render_combined_default(ax, points: np.ndarray, correct: bool):
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap="viridis", s=3.2, alpha=0.9, linewidths=0)
    set_combined_axis_style(ax, correct)
    ax.set_title("Default", fontsize=11, pad=4)


def render_combined_top_corner(ax, points: np.ndarray, correct: bool):
    xy, depth = project_top_corner(points)
    max_abs = max(float(np.abs(xy).max()), 1e-6)
    xy = xy / max_abs
    ax.scatter(xy[:, 0], xy[:, 1], c=depth, cmap="viridis", s=3.2, alpha=0.9, linewidths=0)
    set_combined_axis_style(ax, correct)
    ax.set_title("Top corner", fontsize=11, pad=4)


def save_combined_gallery(samples_path: Path, output_path: Path, run_dir: Path, data_dir: Path):
    samples = pd.read_csv(samples_path)
    gallery = choose_gallery_samples(samples)

    fig = plt.figure(figsize=(22, 11.5))
    outer = GridSpec(2, 5, figure=fig, left=0.02, right=0.99, bottom=0.05, top=0.84, wspace=0.12, hspace=0.42)

    for i, (_, row) in enumerate(gallery.iterrows()):
        points = np.load(resolve_npz_path(str(row["npz_path"]), run_dir, data_dir))["points"]
        correct = int(row["correct"]) == 1
        status = "correct" if correct else "incorrect"
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.04)
        ax_default = fig.add_subplot(inner[0])
        ax_top = fig.add_subplot(inner[1])
        render_combined_default(ax_default, points, correct)
        render_combined_top_corner(ax_top, points, correct)
        ax_default.text(
            0.0,
            1.24,
            f"GT: {row['ground_truth_name']} | Pred: {row['predicted_name']} ({status})",
            transform=ax_default.transAxes,
            fontsize=12.5,
            ha="left",
            va="bottom",
        )

    fig.suptitle("ModelNet40 Sample Predictions - Default and Top-Corner Views", fontsize=19, fontweight="bold", y=0.985)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create ModelNet40 classification README assets")
    parser.add_argument("--run_dir", default=".", help="Repository/run directory")
    parser.add_argument("--data_dir", default=None, help="Data directory, defaults to <run_dir>/data")
    parser.add_argument("--results_dir", default=None, help="Results directory, defaults to <run_dir>/results")
    parser.add_argument("--log_path", default=None, help="Training log, defaults to <run_dir>/logs/cls_train_log.csv")
    parser.add_argument("--output_dir", default=None, help="Asset output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else run_dir / "data"
    results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else run_dir / "results"
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else run_dir / "logs" / "cls_train_log.csv"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "assets" / "modelnet_classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    save_training_curves(log_path, output_dir / "training_curves.png")
    save_confusion_matrix(results_dir / "cls_confusion_matrix.csv", output_dir / "confusion_matrix.png")
    save_per_class_accuracy(results_dir / "cls_per_class_acc.csv", output_dir / "per_class_accuracy.png")
    save_combined_gallery(
        results_dir / "cls_sample_predictions.csv",
        output_dir / "sample_prediction_gallery_combined.png",
        run_dir,
        data_dir,
    )

    print(f"Saved visualization assets to {output_dir}")


if __name__ == "__main__":
    main()
