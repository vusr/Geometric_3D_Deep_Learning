"""
Train PointNet++ MSG for ModelNet40 shape classification.

Usage:
    python -m experiments.modelnet40_classification.train [options]

Key options:
    --data_dir   Root data directory  (default: ~/geometric_deep_learning/data)
    --epochs     Max training epochs  (default: 250)
    --batch_size Batch size           (default: 32)
    --lr         Learning rate        (default: 1e-3)
    --patience   Early stopping patience (default: 20)
    --run_dir    Output directory for checkpoints and logs

Output:
    artifacts/modelnet40_classification/checkpoints/best.pth
    artifacts/modelnet40_classification/logs/train_log.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow direct execution without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdl.engine import collate_label_batch as collate_fn, points_to_pyg
from gdl.datasets.modelnet40_cls import ModelNet40ClsDataset
from gdl.models.pointnet2_cls import PointNet2Classification
from gdl.utils.transforms import get_train_transform
from gdl.utils.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, labels in loader:
        labels = labels.to(device)
        pos, batch = points_to_pyg(points, device)

        optimizer.zero_grad()
        logits = model(pos, batch)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, labels in loader:
        labels = labels.to(device)
        pos, batch = points_to_pyg(points, device)

        logits = model(pos, batch)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PointNet++ classification on ModelNet40")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/geometric_deep_learning/data"))
    parser.add_argument("--run_dir", default=os.path.expanduser("~/geometric_deep_learning"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rotation", default="z", choices=["z", "so3", "perturb"],
                        help="Rotation augmentation mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    splits_csv = data_dir / "splits.csv"
    checkpoint_path = run_dir / "artifacts" / "modelnet40_classification" / "checkpoints" / "best.pth"
    log_path = run_dir / "artifacts" / "modelnet40_classification" / "logs" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Datasets ---
    train_transform = get_train_transform(rotation_mode=args.rotation)
    train_ds = ModelNet40ClsDataset(splits_csv, split="train", transform=train_transform, n_points=args.n_points)
    val_ds   = ModelNet40ClsDataset(splits_csv, split="val",   transform=None,            n_points=args.n_points)

    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Classes: {train_ds.num_classes}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
    )

    # --- Model ---
    model = PointNet2Classification(num_classes=train_ds.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    early_stopping = EarlyStopping(
        patience=args.patience,
        mode="min",
        checkpoint_path=str(checkpoint_path),
    )

    # --- CSV log header ---
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time_s"])

    # --- Training loop ---
    print(f"\nStarting training (max {args.epochs} epochs, patience={args.patience})")
    print("-" * 80)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
            f"val loss={val_loss:.4f} acc={val_acc*100:.2f}% | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{current_lr:.2e}", f"{epoch_time:.2f}"
            ])

        if early_stopping(val_loss, model, epoch):
            break

    print(f"\nTraining complete. Best epoch: {early_stopping.best_epoch + 1}, "
          f"best val loss: {early_stopping.best_score:.6f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Training log  saved to: {log_path}")


if __name__ == "__main__":
    main()
