"""Evaluate ShapeNetSem geometry-core regression checkpoints."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdl.engine import collate_target_batch as collate_fn, points_to_pyg
from gdl.datasets.shapenetsem_reg import ShapeNetSemRegDataset
from gdl.models.pointnet2_reg import PointNet2Regression
from gdl.utils.metrics import measure_throughput, regression_metrics



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


def main():
    parser = argparse.ArgumentParser(description="Evaluate ShapeNetSem regression.")
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
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "artifacts" / "shapenetsem_regression" / "checkpoints" / "global" / "best.pth"
    results_dir = Path(args.results_dir) if args.results_dir else run_dir / "artifacts" / "shapenetsem_regression" / "results" / "global"
    results_dir.mkdir(parents=True, exist_ok=True)

    test_ds = ShapeNetSemRegDataset(
        data_dir / "splits.csv",
        data_dir / "norm_stats.json",
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
    model = PointNet2Regression(out_dim=len(ShapeNetSemRegDataset.TARGETS)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    print(f"Evaluating ShapeNetSem: {len(test_ds)} samples")
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
        f.write("SHAPENETSEM GEOMETRY REGRESSION TEST RESULTS\n")
        f.write("=" * 64 + "\n")
        f.write(f"Checkpoint           : {checkpoint}\n")
        f.write(f"Test set size        : {len(test_ds)}\n")
        f.write("Loss                 : SmoothL1Loss(beta=1.0) on normalized log1p targets\n\n")
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

    with open(results_dir / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "throughput": throughput}, f, indent=2)

    preds_csv = results_dir / "reg_predictions.csv"
    with open(preds_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_idx", "fullId", "class_name"]
        for name in target_names:
            header.extend([f"pred_{name}", f"true_{name}"])
        writer.writerow(header)
        for i, row in test_ds.df.iterrows():
            values = [i, row["fullId"], row["class_name"]]
            for j, _ in enumerate(target_names):
                values.extend([f"{all_preds[i, j]:.8f}", f"{all_targets[i, j]:.8f}"])
            writer.writerow(values)

    print(f"Saved: {results_txt}")
    print(f"Saved: {preds_csv}")
    print(f"Saved: {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
