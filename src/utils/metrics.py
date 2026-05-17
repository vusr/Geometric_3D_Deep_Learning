"""
Evaluation metrics for classification and regression tasks.

Classification:
  - Overall Top-1 accuracy
  - Per-class accuracy

Regression:
  - MAE, RMSE, R² per target

Throughput:
  - Warm-up N batches, then time M batches
  - Report samples/sec and ms/sample
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    all_preds: List[int],
    all_labels: List[int],
    class_names: Optional[List[str]] = None,
    num_classes: int = 40,
) -> Dict:
    """
    Compute overall and per-class accuracy.

    Args:
        all_preds  : Flat list of predicted class indices
        all_labels : Flat list of ground-truth class indices
        class_names: Optional list of class names for reporting
        num_classes: Total number of classes

    Returns:
        dict with keys: overall_acc, per_class_acc, confusion_matrix
    """
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    overall_acc = float((preds == labels).mean())

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            per_class_acc[c] = float("nan")
        else:
            per_class_acc[c] = float((preds[mask] == c).mean())

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        cm[l, p] += 1

    result = {
        "overall_acc": overall_acc,
        "mean_class_acc": float(np.nanmean(list(per_class_acc.values()))),
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
    }
    if class_names:
        result["per_class_acc_named"] = {
            class_names[c]: v for c, v in per_class_acc.items()
        }
    return result


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def regression_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute MAE, RMSE, and R² per output dimension.

    Args:
        all_preds   : (N, D) array of predictions (already denormalised)
        all_targets : (N, D) array of ground-truth values (raw)
        target_names: Optional list of target dimension names

    Returns:
        dict with per-target metrics and aggregate means
    """
    assert all_preds.shape == all_targets.shape, (
        f"Shape mismatch: preds {all_preds.shape} vs targets {all_targets.shape}"
    )
    N, D = all_preds.shape
    if target_names is None:
        target_names = [f"target_{i}" for i in range(D)]

    results = {}
    for i, name in enumerate(target_names):
        y_pred = all_preds[:, i]
        y_true = all_targets[:, i]

        mae = float(np.abs(y_pred - y_true).mean())
        rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))

        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        results[name] = {"mae": mae, "rmse": rmse, "r2": r2}

    # Aggregate (mean across targets)
    results["mean"] = {
        "mae": float(np.mean([results[n]["mae"] for n in target_names])),
        "rmse": float(np.mean([results[n]["rmse"] for n in target_names])),
        "r2": float(np.mean([results[n]["r2"] for n in target_names])),
    }
    return results


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------

def measure_throughput(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    warmup_batches: int = 3,
    timed_batches: int = 10,
    batch_collate_fn=None,
) -> Dict:
    """
    Measure inference throughput (samples/sec) and latency (ms/sample).

    Args:
        model          : The trained model (already moved to device, in eval mode)
        dataloader     : DataLoader for the test split
        device         : torch.device
        warmup_batches : Number of batches to warm-up GPU (not timed)
        timed_batches  : Number of batches to time (averaged)
        batch_collate_fn: Optional function (batch) → (pos, batch_vec) for forward pass

    Returns:
        dict with keys: samples_per_sec, ms_per_sample, total_timed_samples
    """
    model.eval()
    data_iter = iter(dataloader)

    def run_batch(batch_data):
        if batch_collate_fn is not None:
            pos, batch_vec = batch_collate_fn(batch_data, device)
        else:
            # Assume default collation: (points, labels) or (points, targets)
            points = batch_data[0].to(device)          # (B, N, 3)
            B, N, _ = points.shape
            pos = points.view(B * N, 3)
            batch_vec = torch.arange(B, device=device).repeat_interleave(N)

        with torch.no_grad():
            _ = model(pos, batch_vec)

    # Warm-up
    for _ in range(warmup_batches):
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_data = next(data_iter)
        run_batch(batch_data)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed phase
    total_samples = 0
    total_time = 0.0

    for _ in range(timed_batches):
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_data = next(data_iter)

        batch_size = batch_data[0].shape[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        run_batch(batch_data)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_time += (t1 - t0)
        total_samples += batch_size

    samples_per_sec = total_samples / total_time
    ms_per_sample = (total_time / total_samples) * 1000.0

    return {
        "samples_per_sec": round(samples_per_sec, 1),
        "ms_per_sample": round(ms_per_sample, 4),
        "total_timed_samples": total_samples,
        "total_timed_sec": round(total_time, 4),
    }
