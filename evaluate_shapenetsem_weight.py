"""Evaluate ShapeNetSem weight regression checkpoints."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.shapenetsem_weight import ShapeNetSemWeightDataset
from src.models.pointnet2_aux_reg import PointNet2AuxRegression
from src.utils.metrics import regression_metrics


def collate_fn(batch):
    points, aux, class_idx, targets = zip(*batch)
    return (
        torch.stack(points, dim=0),
        torch.stack(aux, dim=0),
        torch.stack(class_idx, dim=0),
        torch.stack(targets, dim=0),
    )


def points_to_pyg(points: torch.Tensor, device):
    bsz, n_points, _ = points.shape
    pos = points.reshape(bsz * n_points, 3).to(device)
    batch = torch.arange(bsz, device=device).repeat_interleave(n_points)
    return pos, batch


@torch.no_grad()
def run_inference(model, loader, device, dataset):
    model.eval()
    preds_raw = []
    targets_raw = []
    for points, aux, class_idx, targets in loader:
        aux = aux.to(device)
        class_idx = class_idx.to(device)
        targets = targets.to(device)
        pos, batch = points_to_pyg(points, device)
        preds = model(pos, batch, aux, class_idx)
        preds_raw.append(dataset.denormalise(preds).cpu().numpy())
        targets_raw.append(dataset.denormalise(targets).cpu().numpy())
    return np.concatenate(preds_raw, axis=0), np.concatenate(targets_raw, axis=0)


@torch.no_grad()
def measure_aux_throughput(model, loader, device, warmup_batches: int, timed_batches: int) -> dict:
    model.eval()
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    def run_batch(batch_data):
        points, aux, class_idx, _ = batch_data
        pos, batch = points_to_pyg(points, device)
        _ = model(pos, batch, aux.to(device), class_idx.to(device))

    for _ in range(warmup_batches):
        run_batch(next_batch())
    if device.type == "cuda":
        torch.cuda.synchronize()

    total_samples = 0
    total_time = 0.0
    for _ in range(timed_batches):
        batch_data = next_batch()
        batch_size = batch_data[0].shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_batch(batch_data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - t0
        total_samples += batch_size
    return {
        "samples_per_sec": round(total_samples / total_time, 2),
        "ms_per_sample": round((total_time / total_samples) * 1000.0, 4),
        "total_timed_samples": int(total_samples),
        "total_timed_sec": round(total_time, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ShapeNetSem weight regression.")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/shapenetsem_regression/data/shapenetsem_regression"))
    parser.add_argument("--run_dir", default=os.path.expanduser("~/shapenetsem_regression"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--timed_batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "shapenetsem_weight" / "auxiliary" / "best.pth"
    results_dir = Path(args.results_dir) if args.results_dir else run_dir / "results" / "shapenetsem_weight" / "auxiliary"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_ds = ShapeNetSemWeightDataset(
        data_dir / "weight_splits.csv",
        data_dir / "weight_norm_stats.json",
        split="test",
        transform=None,
        n_points=args.n_points,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=True,
    )

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = PointNet2AuxRegression(
        out_dim=1,
        num_classes=test_ds.num_classes,
        aux_dim=test_ds.aux_dim,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    print(f"Evaluating ShapeNetSem weight: {len(test_ds)} samples")
    all_preds, all_targets = run_inference(model, test_loader, device, test_ds)
    metrics = regression_metrics(all_preds, all_targets, target_names=["weight"])

    throughput = measure_aux_throughput(
        model,
        test_loader,
        device,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
    )

    results_txt = results_dir / "weight_test_results.txt"
    with open(results_txt, "w") as f:
        f.write("SHAPENETSEM WEIGHT REGRESSION TEST RESULTS\n")
        f.write("=" * 64 + "\n")
        f.write(f"Checkpoint           : {checkpoint}\n")
        f.write(f"Test set size        : {len(test_ds)}\n")
        f.write("Inputs               : xyz point cloud, class embedding, material priors\n")
        f.write("Loss                 : SmoothL1Loss(beta=1.0) on normalized log1p weight\n\n")
        m = metrics["weight"]
        f.write("[weight]\n")
        f.write(f"  MAE  = {m['mae']:.8f}\n")
        f.write(f"  RMSE = {m['rmse']:.8f}\n")
        f.write(f"  R2   = {m['r2']:.8f}\n\n")
        f.write(f"Throughput           : {throughput['samples_per_sec']:.2f} samples/sec\n")
        f.write(f"Latency              : {throughput['ms_per_sample']:.4f} ms/sample\n")
        f.write(f"Timed samples        : {throughput['total_timed_samples']}\n")
        f.write(f"Timed duration (s)   : {throughput['total_timed_sec']:.4f}\n")

    with open(results_dir / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "throughput": throughput}, f, indent=2)

    preds_csv = results_dir / "weight_predictions.csv"
    with open(preds_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "fullId", "class_name", "has_material_prior", "pred_weight", "true_weight"])
        for i, row in test_ds.df.iterrows():
            writer.writerow([
                i, row["fullId"], row["class_name"], row["has_material_prior"],
                f"{all_preds[i, 0]:.8f}", f"{all_targets[i, 0]:.8f}",
            ])

    print(f"Saved: {results_txt}")
    print(f"Saved: {preds_csv}")
    print(f"Saved: {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
