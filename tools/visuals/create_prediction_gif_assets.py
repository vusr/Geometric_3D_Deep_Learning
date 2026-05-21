"""
Create rotating per-sample GIF assets for README prediction tables.

The script reads saved test predictions and split manifests, selects five
held-out examples per experiment, and renders one 360-degree point-cloud GIF per
sample. It intentionally consumes processed datasets without copying them into
the repository.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


TARGETS_SHAPENET_GEOM = (
    "solidVolume",
    "surfaceVolume",
    "supportSurfaceArea",
    "aligned_dim_x",
    "aligned_dim_y",
    "aligned_dim_z",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_value(value: float) -> str:
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{sign}{value / 1_000:.1f}K"
    if value >= 100:
        return f"{sign}{value:.1f}"
    if value >= 10:
        return f"{sign}{value:.2f}"
    if value >= 1:
        return f"{sign}{value:.3f}"
    return f"{sign}{value:.4f}"


def relative_error(pred: float, true: float) -> float:
    return abs(float(pred) - float(true)) / max(abs(float(true)), 1e-8)


def normalize_for_view(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(points, axis=1))
    if np.isfinite(scale) and scale > 0:
        points = points / scale
    return points


def resolve_path(path_text: str, run_dir: Path, data_dir: Path) -> Path:
    path = Path(path_text)
    candidates = [
        path,
        run_dir / path,
        data_dir / path,
        data_dir / "processed" / path.name,
    ]
    parts = path.parts
    if "modelnet_surface_bbox" in parts and "processed" in parts:
        processed_index = parts.index("processed")
        suffix = Path(*parts[processed_index + 1 :])
        candidates.append(data_dir / "processed" / suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate point-cloud file: {path_text}")


def load_points(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    for key in ("points_centered", "points"):
        if key in data:
            return normalize_for_view(data[key])
    raise KeyError(f"No supported point array found in {npz_path}")


def project(points: np.ndarray, yaw_deg: float, elevation_deg: float = 25.0):
    yaw = np.deg2rad(yaw_deg)
    elevation = np.deg2rad(elevation_deg)
    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elevation), -np.sin(elevation)],
            [0.0, np.sin(elevation), np.cos(elevation)],
        ],
        dtype=np.float32,
    )
    rotated = points @ rz.T @ rx.T
    xy = rotated[:, :2]
    depth = rotated[:, 2]
    max_abs = max(float(np.abs(xy).max()), 1e-6)
    return xy / max_abs, depth


def render_frame(points: np.ndarray, yaw_deg: float, title: str, subtitle: str, border: str) -> Image.Image:
    xy, depth = project(points, yaw_deg)
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=120)
    fig.patch.set_facecolor("white")
    ax.scatter(xy[:, 0], xy[:, 1], c=depth, cmap="viridis", s=5.0, alpha=0.92, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.4)
        spine.set_color(border)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
    ax.text(
        0.5,
        -0.08,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )
    fig.tight_layout(pad=0.55)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE)
    plt.close(fig)
    return image


def save_gif(points: np.ndarray, output_path: Path, title: str, subtitle: str, border: str) -> None:
    frames = [
        render_frame(points, yaw_deg=float(yaw), title=title, subtitle=subtitle, border=border)
        for yaw in np.linspace(0, 360, 36, endpoint=False)
    ]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )


def distinct_by_class(df: pd.DataFrame, class_column: str, limit: int) -> pd.DataFrame:
    selected = []
    seen = set()
    for idx, row in df.iterrows():
        cls = str(row[class_column])
        if cls in seen:
            continue
        selected.append(idx)
        seen.add(cls)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        for idx in df.index:
            if idx not in selected:
                selected.append(idx)
            if len(selected) == limit:
                break
    return df.loc[selected].reset_index(drop=True)


def select_classification_samples(preds: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    preds = preds.copy()
    preds["correct"] = preds["correct"].astype(int)
    preds["confidence"] = pd.to_numeric(preds["confidence"], errors="coerce").fillna(0.0)
    correct = preds[preds["correct"] == 1].sort_values("confidence", ascending=False)
    wrong = preds[preds["correct"] == 0].sort_values("confidence", ascending=False)
    selected = pd.concat([distinct_by_class(correct, "ground_truth_name", limit - 1), wrong.head(1)])
    return selected.head(limit).reset_index(drop=True)


def select_low_error_samples(preds: pd.DataFrame, class_column: str, error_column: str, limit: int = 5) -> pd.DataFrame:
    ordered = preds.copy().sort_values(error_column, ascending=True)
    return distinct_by_class(ordered, class_column, limit)


def write_markdown_table(rows: list[dict], output_path: Path) -> None:
    header = [" "] + [f"Sample {i + 1}" for i in range(len(rows))]
    separator = ["---"] * len(header)
    samples = ["Sample"] + [f"![{row['alt']}]({row['gif']})" for row in rows]
    predictions = ["Prediction"] + [row["prediction"] for row in rows]
    lines = [
        "| " + " | ".join(row) + " |"
        for row in (header, separator, samples, predictions)
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_prediction_rows(rows: list[dict], prediction_rows: list[tuple[str, list[str]]], output_path: Path) -> None:
    header = [" "] + [f"Sample {i + 1}" for i in range(len(rows))]
    separator = ["---"] * len(header)
    samples = ["Sample"] + [f"![{row['alt']}]({row['gif']})" for row in rows]
    table_rows = [header, separator, samples]
    table_rows.extend([label] + values for label, values in prediction_rows)
    lines = ["| " + " | ".join(row) + " |" for row in table_rows]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_modelnet_classification(run_dir: Path, data_dir: Path, artifact_root: Path) -> None:
    output_dir = artifact_root / "modelnet40_classification" / "assets" / "rotating_samples"
    ensure_dir(output_dir)
    preds = pd.read_csv(artifact_root / "modelnet40_classification" / "results" / "sample_predictions.csv")
    selected = select_classification_samples(preds)
    rows = []

    for i, row in selected.iterrows():
        points = load_points(resolve_path(str(row["npz_path"]), run_dir, data_dir))
        correct = int(row["correct"]) == 1
        border = "#2d7f5e" if correct else "#b64b4b"
        gif_name = f"sample_{i + 1:02d}_{row['ground_truth_name']}.gif"
        save_gif(
            points,
            output_dir / gif_name,
            title=str(row["ground_truth_name"]),
            subtitle=f"pred: {row['predicted_name']}",
            border=border,
        )
        rows.append(
            {
                "gif": f"artifacts/modelnet40_classification/assets/rotating_samples/{gif_name}",
                "alt": f"Rotating ModelNet40 {row['ground_truth_name']} point cloud",
                "prediction": f"GT `{row['ground_truth_name']}`<br>Pred `{row['predicted_name']}`",
            }
        )

    write_markdown_table(rows, output_dir / "README_table.md")


def create_modelnet_regression(run_dir: Path, data_dir: Path, artifact_root: Path) -> None:
    output_dir = artifact_root / "modelnet40_regression" / "assets" / "rotating_samples"
    ensure_dir(output_dir)
    preds = pd.read_csv(artifact_root / "modelnet40_regression" / "results" / "global" / "reg_predictions.csv")
    splits = pd.read_csv(artifact_root / "modelnet40_regression" / "data_metadata" / "splits.csv")
    test_rows = splits[splits["split"] == "test"].reset_index(drop=True)
    preds = preds.copy()
    preds["surface_rel_error"] = preds.apply(lambda r: relative_error(r["pred_surface_area"], r["true_surface_area"]), axis=1)
    preds["bbox_rel_error"] = preds.apply(lambda r: relative_error(r["pred_bbox_volume"], r["true_bbox_volume"]), axis=1)
    preds["mean_rel_error"] = 0.5 * (preds["surface_rel_error"] + preds["bbox_rel_error"])
    selected = select_low_error_samples(preds, "class_name", "mean_rel_error")
    rows = []

    for i, row in selected.iterrows():
        split_row = test_rows.iloc[int(row["sample_idx"])]
        points = load_points(resolve_path(str(split_row["npz_path"]), run_dir, data_dir))
        border = "#2d7f5e" if float(row["mean_rel_error"]) <= 0.5 else "#b64b4b"
        gif_name = f"sample_{i + 1:02d}_{row['class_name']}.gif"
        save_gif(
            points,
            output_dir / gif_name,
            title=str(row["class_name"]),
            subtitle=f"mean rel. err: {float(row['mean_rel_error']):.2f}",
            border=border,
        )
        prediction = (
            f"Surface `{fmt_value(row['pred_surface_area'])}` vs `{fmt_value(row['true_surface_area'])}`<br>"
            f"BBox vol `{fmt_value(row['pred_bbox_volume'])}` vs `{fmt_value(row['true_bbox_volume'])}`"
        )
        rows.append(
            {
                "gif": f"artifacts/modelnet40_regression/assets/rotating_samples/{gif_name}",
                "alt": f"Rotating ModelNet40 {row['class_name']} point cloud",
                "prediction": prediction,
            }
        )

    write_markdown_table(rows, output_dir / "README_table.md")


def create_shapenet_geometry(run_dir: Path, data_dir: Path, artifact_root: Path) -> None:
    output_dir = artifact_root / "shapenetsem_regression" / "assets" / "rotating_samples"
    ensure_dir(output_dir)
    preds = pd.read_csv(artifact_root / "shapenetsem_regression" / "results" / "global" / "reg_predictions.csv")
    splits = pd.read_csv(artifact_root / "shapenetsem_regression" / "data_metadata" / "splits.csv")
    test_rows = splits[splits["split"] == "test"].reset_index(drop=True)
    preds = preds.copy()
    for target in TARGETS_SHAPENET_GEOM:
        preds[f"{target}_rel_error"] = preds.apply(lambda r: relative_error(r[f"pred_{target}"], r[f"true_{target}"]), axis=1)
    preds["mean_rel_error"] = preds[[f"{target}_rel_error" for target in TARGETS_SHAPENET_GEOM]].mean(axis=1)
    selected = select_low_error_samples(preds, "class_name", "mean_rel_error")
    rows = []
    prediction_rows = {target: [] for target in TARGETS_SHAPENET_GEOM}

    for i, row in selected.iterrows():
        split_row = test_rows.iloc[int(row["sample_idx"])]
        points = load_points(resolve_path(str(split_row["npz_path"]), run_dir, data_dir))
        border = "#2d7f5e" if float(row["mean_rel_error"]) <= 0.35 else "#b64b4b"
        gif_name = f"sample_{i + 1:02d}_{row['class_name']}.gif"
        save_gif(
            points,
            output_dir / gif_name,
            title=str(row["class_name"]),
            subtitle=f"mean rel. err: {float(row['mean_rel_error']):.2f}",
            border=border,
        )
        rows.append(
            {
                "gif": f"artifacts/shapenetsem_regression/assets/rotating_samples/{gif_name}",
                "alt": f"Rotating ShapeNetSem {row['class_name']} point cloud",
            }
        )
        for target in TARGETS_SHAPENET_GEOM:
            prediction_rows[target].append(
                f"Pred `{fmt_value(row[f'pred_{target}'])}`<br>GT `{fmt_value(row[f'true_{target}'])}`"
            )

    labels = {
        "solidVolume": "Solid volume",
        "surfaceVolume": "Surface volume",
        "supportSurfaceArea": "Support area",
        "aligned_dim_x": "Dim x",
        "aligned_dim_y": "Dim y",
        "aligned_dim_z": "Dim z",
    }
    write_markdown_prediction_rows(
        rows,
        [(labels[target], prediction_rows[target]) for target in TARGETS_SHAPENET_GEOM],
        output_dir / "README_table.md",
    )


def create_shapenet_weight(run_dir: Path, data_dir: Path, artifact_root: Path) -> None:
    output_dir = artifact_root / "shapenetsem_weight" / "assets" / "rotating_samples"
    ensure_dir(output_dir)
    preds = pd.read_csv(artifact_root / "shapenetsem_weight" / "results" / "auxiliary" / "weight_predictions.csv")
    splits = pd.read_csv(artifact_root / "shapenetsem_weight" / "data_metadata" / "weight_splits.csv")
    test_rows = splits[splits["split"] == "test"].reset_index(drop=True)
    preds = preds.copy()
    preds["rel_error"] = preds.apply(lambda r: relative_error(r["pred_weight"], r["true_weight"]), axis=1)
    selected = select_low_error_samples(preds, "class_name", "rel_error")
    rows = []
    weight_values = []

    for i, row in selected.iterrows():
        split_row = test_rows.iloc[int(row["sample_idx"])]
        points = load_points(resolve_path(str(split_row["npz_path"]), run_dir, data_dir))
        border = "#2d7f5e" if float(row["rel_error"]) <= 0.35 else "#b64b4b"
        gif_name = f"sample_{i + 1:02d}_{row['class_name']}.gif"
        save_gif(
            points,
            output_dir / gif_name,
            title=str(row["class_name"]),
            subtitle=f"pred: {fmt_value(row['pred_weight'])} kg",
            border=border,
        )
        rows.append(
            {
                "gif": f"artifacts/shapenetsem_weight/assets/rotating_samples/{gif_name}",
                "alt": f"Rotating ShapeNetSem {row['class_name']} point cloud",
            }
        )
        weight_values.append(f"Pred `{fmt_value(row['pred_weight'])}` kg<br>GT `{fmt_value(row['true_weight'])}` kg")

    write_markdown_prediction_rows(rows, [("Weight", weight_values)], output_dir / "README_table.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create rotating prediction GIF assets")
    parser.add_argument("--run_dir", default=".", help="Repository/run directory")
    parser.add_argument("--data_dir", default=None, help="Data directory, defaults to <run_dir>/data")
    parser.add_argument("--artifact_root", default=None, help="Artifact root, defaults to <run_dir>/artifacts")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else run_dir / "data"
    artifact_root = Path(args.artifact_root).expanduser().resolve() if args.artifact_root else run_dir / "artifacts"

    create_modelnet_classification(run_dir, data_dir, artifact_root)
    create_modelnet_regression(run_dir, data_dir, artifact_root)
    create_shapenet_geometry(run_dir, data_dir, artifact_root)
    create_shapenet_weight(run_dir, data_dir, artifact_root)

    manifest = {
        "experiments": [
            "modelnet40_classification",
            "modelnet40_regression",
            "shapenetsem_regression",
            "shapenetsem_weight",
        ],
        "samples_per_experiment": 5,
    }
    (artifact_root / "prediction_gif_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved rotating prediction assets under {artifact_root}")


if __name__ == "__main__":
    main()
