"""
Train PointNet++ MSG for ModelNet40 surface_area + bbox_volume regression.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdl.engine import collate_target_batch as collate_fn, points_to_pyg
from gdl.datasets.modelnet40_surface_bbox_reg import ModelNet40SurfaceBBoxRegDataset
from gdl.models.pointnet2_reg import PointNet2Regression
from gdl.utils.early_stopping import EarlyStopping
from gdl.utils.transforms import ComposeTransforms, RandomJitter, RandomRotation



def get_scale_preserving_transform(rotation_mode: str):
    return ComposeTransforms([
        RandomRotation(mode=rotation_mode),
        RandomJitter(sigma=0.005, clip=0.02),
    ])


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
    metrics = {}
    for i, name in enumerate(dataset.target_names()):
        y_pred = preds_raw[:, i]
        y_true = targets_raw[:, i]
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        metrics[f"mae_{name}"] = float(np.abs(y_pred - y_true).mean())
        metrics[f"r2_{name}"] = float(1 - ss_res / (ss_tot + 1e-8))
    return total_loss / max(total, 1), metrics


def main():
    parser = argparse.ArgumentParser(description="Train ModelNet surface/bbox regression model.")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/geometric_deep_learning/data/modelnet_surface_bbox"))
    parser.add_argument("--run_dir", default=os.path.expanduser("~/geometric_deep_learning"))
    parser.add_argument("--cluster_id", type=int, default=None)
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    model_name = "global" if args.cluster_id is None else f"cluster_{args.cluster_id}"
    checkpoint_path = run_dir / "artifacts" / "modelnet40_regression" / "checkpoints" / model_name / "best.pth"
    log_dir = run_dir / "artifacts" / "modelnet40_regression" / "logs" / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"

    train_ds = ModelNet40SurfaceBBoxRegDataset(
        data_dir / "splits.csv",
        data_dir / "norm_stats.json",
        split="train",
        cluster_id=args.cluster_id,
        transform=get_scale_preserving_transform(args.rotation),
        n_points=args.n_points,
    )
    val_ds = ModelNet40SurfaceBBoxRegDataset(
        data_dir / "splits.csv",
        data_dir / "norm_stats.json",
        split="val",
        cluster_id=args.cluster_id,
        transform=None,
        n_points=args.n_points,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | model={model_name}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=True,
    )

    model = PointNet2Regression(out_dim=2).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_t_max = args.scheduler_t_max if args.scheduler_t_max is not None else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max, eta_min=1e-5)
    early_stopping = EarlyStopping(patience=args.patience, mode="min", checkpoint_path=str(checkpoint_path))

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "val_mae_surface_area", "val_mae_bbox_volume",
            "val_r2_surface_area", "val_r2_bbox_volume",
            "lr", "epoch_time_s",
        ])

    print(f"\nStarting training (max {args.epochs} epochs, patience={args.patience})")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, val_ds)
        scheduler.step()
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"MAE surface={val_metrics['mae_surface_area']:.4f} "
            f"bbox_vol={val_metrics['mae_bbox_volume']:.4f} | "
            f"R2 surface={val_metrics['r2_surface_area']:.4f} "
            f"bbox_vol={val_metrics['r2_bbox_volume']:.4f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_metrics['mae_surface_area']:.6f}",
                f"{val_metrics['mae_bbox_volume']:.6f}",
                f"{val_metrics['r2_surface_area']:.6f}",
                f"{val_metrics['r2_bbox_volume']:.6f}",
                f"{current_lr:.2e}", f"{epoch_time:.2f}",
            ])

        if early_stopping(val_loss, model, epoch):
            break

        if args.save_epoch_checkpoints:
            epoch_path = checkpoint_path.parent / f"epoch_{epoch + 1:03d}.pth"
            torch.save(model.state_dict(), epoch_path)

    print(f"\nTraining complete. Best epoch: {early_stopping.best_epoch + 1}, "
          f"best val loss: {early_stopping.best_score:.6f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
