"""
Evaluate ModelNet40 surface_area + bbox_volume regression models.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.modelnet40_surface_bbox_reg import ModelNet40SurfaceBBoxRegDataset
from src.models.pointnet2_reg import PointNet2Regression
from src.utils.metrics import measure_throughput, regression_metrics


def collate_fn(batch):
    points_list, targets_list = zip(*batch)
    return torch.stack(points_list, dim=0), torch.stack(targets_list, dim=0)


def points_to_pyg(points: torch.Tensor, device):
    bsz, n_points, _ = points.shape
    pos = points.view(bsz * n_points, 3).to(device)
    batch = torch.arange(bsz, device=device).repeat_interleave(n_points)
    return pos, batch


@torch.no_grad()
def run_inference(model, loader, device, dataset):
    model.eval()
    preds_raw = []
    targets_raw = []
    for points, targets in loader:
        targets = targets.to(device)
        pos, batch = points_to_pyg(points, device)
        preds = model(pos, batch)
        preds_raw.append(dataset.denormalise(preds).cpu().numpy())
        targets_raw.append(dataset.denormalise(targets).cpu().numpy())
    return np.concatenate(preds_raw, axis=0), np.concatenate(targets_raw, axis=0)


def evaluate_one(args, model_name: str, cluster_id: int | None, device):
    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint and cluster_id is None
        else run_dir / "checkpoints" / "modelnet_surface_bbox_reg" / model_name / "best.pth"
    )
    if not checkpoint.exists() and not args.checkpoint:
        epoch_checkpoints = sorted(checkpoint.parent.glob("epoch_*.pth"))
        if epoch_checkpoints:
            checkpoint = epoch_checkpoints[-1]
    results_dir = Path(args.results_dir) if args.results_dir else run_dir / "results" / "modelnet_surface_bbox_reg" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    test_ds = ModelNet40SurfaceBBoxRegDataset(
        data_dir / "splits.csv",
        data_dir / "norm_stats.json",
        split="test",
        cluster_id=cluster_id,
        transform=None,
        n_points=args.n_points,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=True,
    )

    model = PointNet2Regression(out_dim=2).to(device)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    print(f"Evaluating {model_name}: {len(test_ds)} samples")
    all_preds, all_targets = run_inference(model, test_loader, device, test_ds)
    target_names = test_ds.target_names()
    metrics = regression_metrics(all_preds, all_targets, target_names=target_names)
    throughput = measure_throughput(
        model,
        test_loader,
        device,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
    )

    results_txt = results_dir / "reg_test_results.txt"
    with open(results_txt, "w") as f:
        f.write("MODELNET SURFACE/BBOX REGRESSION TEST RESULTS\n")
        f.write("=" * 64 + "\n")
        f.write(f"Model                : {model_name}\n")
        f.write(f"Cluster ID           : {cluster_id}\n")
        f.write(f"Checkpoint           : {checkpoint}\n")
        f.write(f"Test set size        : {len(test_ds)}\n\n")
        for name in target_names:
            m = metrics[name]
            f.write(f"[{name}]\n")
            f.write(f"  MAE  = {m['mae']:.8f}\n")
            f.write(f"  RMSE = {m['rmse']:.8f}\n")
            f.write(f"  R2   = {m['r2']:.8f}\n\n")
        mean_m = metrics["mean"]
        f.write("[mean across targets]\n")
        f.write(f"  MAE  = {mean_m['mae']:.8f}\n")
        f.write(f"  RMSE = {mean_m['rmse']:.8f}\n")
        f.write(f"  R2   = {mean_m['r2']:.8f}\n\n")
        f.write(f"Throughput           : {throughput['samples_per_sec']:.2f} samples/sec\n")
        f.write(f"Latency              : {throughput['ms_per_sample']:.4f} ms/sample\n")
        f.write(f"Timed samples        : {throughput['total_timed_samples']}\n")
        f.write(f"Timed duration (s)   : {throughput['total_timed_sec']:.4f}\n")

    preds_csv = results_dir / "reg_predictions.csv"
    with open(preds_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx", "class_name", "cluster_id",
            "pred_surface_area", "pred_bbox_volume",
            "true_surface_area", "true_bbox_volume",
        ])
        for i, row in test_ds.df.iterrows():
            writer.writerow([
                i,
                row["class_name"],
                int(row["cluster_id"]),
                f"{all_preds[i, 0]:.8f}",
                f"{all_preds[i, 1]:.8f}",
                f"{all_targets[i, 0]:.8f}",
                f"{all_targets[i, 1]:.8f}",
            ])

    print(f"Saved: {results_txt}")
    print(f"Saved: {preds_csv}")
    return {"model_name": model_name, "cluster_id": cluster_id, "metrics": metrics, "throughput": throughput}


def main():
    parser = argparse.ArgumentParser(description="Evaluate ModelNet surface/bbox regression.")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/geometric_deep_learning/data/modelnet_surface_bbox"))
    parser.add_argument("--run_dir", default=os.path.expanduser("~/geometric_deep_learning"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--mode", choices=["global", "oracle_clusters", "auto"], default="global")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--timed_batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    mode = args.mode
    if mode == "auto":
        with open(data_dir / "analysis_summary.json") as f:
            summary = json.load(f)
        mode = "oracle_clusters" if summary["cluster_decision"]["recommendation"] == "cluster_models" else "global"

    if mode == "global":
        evaluate_one(args, "global", None, device)
    else:
        splits = np.genfromtxt(data_dir / "splits.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
        cluster_ids = sorted({int(x) for x in splits["cluster_id"]})
        summaries = []
        for cluster_id in cluster_ids:
            summaries.append(evaluate_one(args, f"cluster_{cluster_id}", cluster_id, device))
        summary_path = Path(args.run_dir) / "results" / "modelnet_surface_bbox_reg" / "oracle_cluster_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
