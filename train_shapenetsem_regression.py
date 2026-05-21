"""Train PointNet++ MSG on ShapeNetSem geometry-core regression."""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from src.datasets.shapenetsem_reg import ShapeNetSemRegDataset
from src.models.pointnet2_reg import PointNet2Regression
from src.utils.early_stopping import EarlyStopping
from src.utils.transforms import ComposeTransforms, RandomJitter, RandomRotation


def collate_fn(batch):
    points_list, targets_list = zip(*batch)
    return torch.stack(points_list, dim=0), torch.stack(targets_list, dim=0)


def points_to_pyg(points: torch.Tensor, device):
    bsz, n_points, _ = points.shape
    pos = points.reshape(bsz * n_points, 3).to(device)
    batch = torch.arange(bsz, device=device).repeat_interleave(n_points)
    return pos, batch


def get_scale_preserving_transform(rotation_mode: str):
    return ComposeTransforms([
        RandomRotation(mode=rotation_mode),
        RandomJitter(sigma=0.005, clip=0.02),
    ])


def raw_metrics(preds_raw: np.ndarray, targets_raw: np.ndarray, names: list[str]) -> dict:
    metrics = {}
    for i, name in enumerate(names):
        y_pred = preds_raw[:, i]
        y_true = targets_raw[:, i]
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        metrics[f"mae_{name}"] = float(np.abs(y_pred - y_true).mean())
        metrics[f"rmse_{name}"] = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
        metrics[f"r2_{name}"] = float(1 - ss_res / (ss_tot + 1e-8))
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    for points, targets in loader:
        targets = targets.to(device)
        pos, batch = points_to_pyg(points, device)
        optimizer.zero_grad()
        preds = model(pos, batch)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, dataset):
    model.eval()
    total_loss = 0.0
    total = 0
    preds_raw = []
    targets_raw = []
    for points, targets in loader:
        targets = targets.to(device)
        pos, batch = points_to_pyg(points, device)
        preds = model(pos, batch)
        loss = criterion(preds, targets)
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        preds_raw.append(dataset.denormalise(preds).cpu().numpy())
        targets_raw.append(dataset.denormalise(targets).cpu().numpy())
    preds_raw = np.concatenate(preds_raw, axis=0)
    targets_raw = np.concatenate(targets_raw, axis=0)
    return total_loss / max(total, 1), raw_metrics(preds_raw, targets_raw, dataset.target_names())


def main():
    parser = argparse.ArgumentParser(description="Train ShapeNetSem regression model.")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/shapenetsem_regression/data/shapenetsem_regression"))
    parser.add_argument("--run_dir", default=os.path.expanduser("~/shapenetsem_regression"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--scheduler_t_max", type=int, default=None)
    parser.add_argument("--save_epoch_checkpoints", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rotation", default="z", choices=["z", "so3", "perturb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--huber_beta", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "checkpoints" / "shapenetsem_regression" / "global" / "best.pth"
    log_dir = run_dir / "logs" / "shapenetsem_regression" / "global"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"

    train_ds = ShapeNetSemRegDataset(
        data_dir / "splits.csv",
        data_dir / "norm_stats.json",
        split="train",
        transform=get_scale_preserving_transform(args.rotation),
        n_points=args.n_points,
    )
    val_ds = ShapeNetSemRegDataset(
        data_dir / "splits.csv",
        data_dir / "norm_stats.json",
        split="val",
        transform=None,
        n_points=args.n_points,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | targets={train_ds.target_names()}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=True,
    )

    model = PointNet2Regression(out_dim=len(ShapeNetSemRegDataset.TARGETS)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_t_max = args.scheduler_t_max if args.scheduler_t_max is not None else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max, eta_min=1e-5)
    early_stopping = EarlyStopping(patience=args.patience, mode="min", checkpoint_path=str(checkpoint_path))

    metric_cols = []
    for name in ShapeNetSemRegDataset.target_names():
        metric_cols.extend([f"val_mae_{name}", f"val_rmse_{name}", f"val_r2_{name}"])
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_huber_loss", "val_huber_loss", *metric_cols, "lr", "epoch_time_s"])

    print(f"\nStarting training (max {args.epochs} epochs, patience={args.patience}, beta={args.huber_beta})")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, val_ds)
        scheduler.step()
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        mean_r2 = np.mean([val_metrics[f"r2_{name}"] for name in ShapeNetSemRegDataset.target_names()])
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | train_huber={train_loss:.4f} | "
            f"val_huber={val_loss:.4f} | mean_R2={mean_r2:.4f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        row = [epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}"]
        for name in ShapeNetSemRegDataset.target_names():
            row.extend([
                f"{val_metrics[f'mae_{name}']:.8f}",
                f"{val_metrics[f'rmse_{name}']:.8f}",
                f"{val_metrics[f'r2_{name}']:.8f}",
            ])
        row.extend([f"{current_lr:.2e}", f"{epoch_time:.2f}"])
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        if early_stopping(val_loss, model, epoch):
            break
        if args.save_epoch_checkpoints:
            epoch_path = checkpoint_path.parent / f"epoch_{epoch + 1:03d}.pth"
            torch.save(model.state_dict(), epoch_path)

    print(f"\nTraining complete. Best epoch: {early_stopping.best_epoch + 1}, best val loss: {early_stopping.best_score:.6f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
