"""
Create ModelNet40 classification visualization assets from saved results.

Expected inputs:
  artifacts/modelnet40_classification/logs/train_log.csv
  artifacts/modelnet40_classification/results/per_class_acc.csv
  artifacts/modelnet40_classification/results/confusion_matrix.csv
  artifacts/modelnet40_classification/results/sample_predictions.csv

Outputs:
  artifacts/modelnet40_classification/assets/training_curves.png
  artifacts/modelnet40_classification/assets/confusion_matrix.png
  artifacts/modelnet40_classification/assets/per_class_accuracy.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def main():
    parser = argparse.ArgumentParser(description="Create ModelNet40 classification README assets")
    parser.add_argument("--run_dir", default=".", help="Repository/run directory")
    parser.add_argument("--results_dir", default=None, help="Results directory, defaults to <run_dir>/results")
    parser.add_argument("--log_path", default=None, help="Training log, defaults to <run_dir>/artifacts/modelnet40_classification/logs/train_log.csv")
    parser.add_argument("--output_dir", default=None, help="Asset output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else run_dir / "artifacts" / "modelnet40_classification" / "results"
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else run_dir / "artifacts" / "modelnet40_classification" / "logs" / "train_log.csv"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "artifacts" / "modelnet40_classification" / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    save_training_curves(log_path, output_dir / "training_curves.png")
    save_confusion_matrix(results_dir / "confusion_matrix.csv", output_dir / "confusion_matrix.png")
    save_per_class_accuracy(results_dir / "per_class_acc.csv", output_dir / "per_class_accuracy.png")

    print(f"Saved visualization assets to {output_dir}")


if __name__ == "__main__":
    main()
