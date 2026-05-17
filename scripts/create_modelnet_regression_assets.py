"""
Create ModelNet40 surface/bbox regression visualization assets.

The gallery uses the same sample-selection logic as the classification gallery
so the README can present comparable examples across tasks.
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
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def choose_classification_gallery_samples(samples: pd.DataFrame):
    samples = samples.copy()
    samples["correct"] = samples["correct"].astype(int)
    samples["confidence"] = pd.to_numeric(samples["confidence"], errors="coerce").fillna(0.0)
    correct = samples[samples["correct"] == 1].sort_values("confidence", ascending=False).head(5)
    wrong = samples[samples["correct"] == 0].sort_values("confidence", ascending=False).head(5)
    gallery = pd.concat([correct, wrong])
    if len(gallery) < 10:
        remainder = samples.drop(gallery.index, errors="ignore").head(10 - len(gallery))
        gallery = pd.concat([gallery, remainder])
    return gallery.head(10).reset_index(drop=True)


def fmt_value(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.1f}"


def relative_error(pred: float, true: float) -> float:
    return abs(float(pred) - float(true)) / max(abs(float(true)), 1e-8)


def normalize_for_view(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points


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


def set_axis_style(ax, border: str):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(3.0)
        spine.set_color(border)


def render_default(ax, points: np.ndarray, border: str):
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        cmap="viridis",
        s=3.2,
        alpha=0.92,
        linewidths=0,
    )
    set_axis_style(ax, border)
    ax.set_title("Default", fontsize=14, pad=5)


def render_top_corner(ax, points: np.ndarray, border: str):
    xy, depth = project_top_corner(points)
    max_abs = max(float(np.abs(xy).max()), 1e-6)
    xy = xy / max_abs
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=depth,
        cmap="viridis",
        s=3.2,
        alpha=0.92,
        linewidths=0,
    )
    set_axis_style(ax, border)
    ax.set_title("Top corner", fontsize=14, pad=5)


def save_regression_gallery(run_dir: Path, data_dir: Path, output_path: Path):
    cls_samples = pd.read_csv(run_dir / "results" / "cls_sample_predictions.csv")
    gallery = choose_classification_gallery_samples(cls_samples)

    preds = pd.read_csv(run_dir / "results" / "modelnet_surface_bbox_reg" / "global" / "reg_predictions.csv")
    splits = pd.read_csv(data_dir / "modelnet_surface_bbox" / "splits.csv")
    test_rows = splits[splits["split"] == "test"].reset_index(drop=True)

    fig = plt.figure(figsize=(22, 11.5))
    outer = GridSpec(2, 5, figure=fig, left=0.02, right=0.99, bottom=0.07, top=0.84, wspace=0.12, hspace=0.46)

    for i, (_, cls_row) in enumerate(gallery.iterrows()):
        sample_idx = int(cls_row["sample_index"])
        reg_row = preds[preds["sample_idx"] == sample_idx].iloc[0]
        split_row = test_rows.iloc[sample_idx]
        points = np.load(Path(split_row["npz_path"]))["points_centered"]
        points = normalize_for_view(points)

        surface_err = relative_error(reg_row["pred_surface_area"], reg_row["true_surface_area"])
        bbox_err = relative_error(reg_row["pred_bbox_volume"], reg_row["true_bbox_volume"])
        border = "#2d7f5e" if 0.5 * (surface_err + bbox_err) <= 0.5 else "#b64b4b"

        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.04)
        ax_default = fig.add_subplot(inner[0])
        ax_top = fig.add_subplot(inner[1])
        render_default(ax_default, points, border)
        render_top_corner(ax_top, points, border)

        title = (
            f"{split_row['class_name']}\n"
            f"Surface GT {fmt_value(reg_row['true_surface_area'])}  "
            f"Pred {fmt_value(reg_row['pred_surface_area'])}\n"
            f"BBoxVol GT {fmt_value(reg_row['true_bbox_volume'])}  "
            f"Pred {fmt_value(reg_row['pred_bbox_volume'])}"
        )
        ax_default.text(
            0.0,
            1.30,
            title,
            transform=ax_default.transAxes,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="bottom",
            linespacing=1.15,
        )

    fig.suptitle(
        "ModelNet40 Regression Sample Predictions: Surface Area and Bounding-Box Volume",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Green border: average relative error <= 50%. Red border: larger error. "
        "Values use K/M abbreviations for readability.",
        ha="center",
        fontsize=12,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return [int(x) for x in gallery["sample_index"]]


def main():
    parser = argparse.ArgumentParser(description="Create ModelNet40 regression README assets")
    parser.add_argument("--run_dir", default=".", help="Repository/run directory")
    parser.add_argument("--data_dir", default=None, help="Data directory, defaults to <run_dir>/data")
    parser.add_argument("--output_dir", default=None, help="Asset output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else run_dir / "data"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "assets" / "modelnet_regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    sample_indices = save_regression_gallery(
        run_dir,
        data_dir,
        output_dir / "sample_regression_prediction_gallery.png",
    )
    print(f"Saved visualization assets to {output_dir}")
    print("sample_indices:", " ".join(str(i) for i in sample_indices))


if __name__ == "__main__":
    main()
