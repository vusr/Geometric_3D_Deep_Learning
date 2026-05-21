"""
Evaluate trained PointNet++ classification model on the test set.

Reports:
  - Overall Top-1 accuracy
  - Overall Top-5 accuracy
  - Mean per-class accuracy
  - Per-class accuracy (saved as CSV)
  - Confusion matrix (saved as CSV)
  - Per-sample predictions (saved as CSV)
  - Machine-readable metrics and runtime metadata (saved as JSON)
  - Inference throughput: samples/sec and ms/sample

Usage:
    python -m experiments.modelnet40_classification.evaluate [options]

Key options:
    --data_dir      Root data directory
    --checkpoint    Path to best.pth  (default: artifacts/modelnet40_classification/checkpoints/best.pth)
    --results_dir   Where to write result files
    --batch_size    Inference batch size (default: 64 — larger for throughput)
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdl.engine import collate_label_batch as collate_fn, points_to_pyg
from gdl.datasets.modelnet40_cls import ModelNet40ClsDataset
from gdl.models.pointnet2_cls import PointNet2Classification
from gdl.utils.metrics import classification_metrics, measure_throughput



@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_top5, all_conf = [], [], [], []
    for points, labels in loader:
        pos, batch = points_to_pyg(points, device)
        logits = model(pos, batch)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices.cpu().numpy()
        conf = probs.max(dim=1).values.cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_top5.extend(top5.tolist())
        all_conf.extend(conf.tolist())
    return all_preds, all_labels, all_top5, all_conf


def runtime_metadata(device, args, checkpoint):
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    return {
        "backend": "PyTorch GPU" if device.type == "cuda" else "PyTorch CPU",
        "device": str(device),
        "cuda_device_name": cuda_name,
        "precision": "FP32",
        "amp": False,
        "torch_version": torch.__version__,
        "checkpoint": str(checkpoint),
        "batch_size": args.batch_size,
        "n_points": args.n_points,
        "workers": args.workers,
        "warmup_batches": args.warmup_batches,
        "timed_batches": args.timed_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification model on ModelNet40 test set")
    parser.add_argument("--data_dir",   default=os.path.expanduser("~/geometric_deep_learning/data"))
    parser.add_argument("--run_dir",    default=os.path.expanduser("~/geometric_deep_learning"))
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint. Defaults to <run_dir>/artifacts/modelnet40_classification/checkpoints/best.pth")
    parser.add_argument("--results_dir", default=None,
                        help="Output directory. Defaults to <run_dir>/artifacts/modelnet40_classification/results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_points",   type=int, default=1024)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--warmup_batches", type=int, default=5,
                        help="GPU warm-up batches before throughput timing")
    parser.add_argument("--timed_batches",  type=int, default=20,
                        help="Number of batches to time for throughput")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir     = Path(args.run_dir)
    data_dir    = Path(args.data_dir)
    checkpoint  = Path(args.checkpoint) if args.checkpoint else run_dir / "artifacts" / "modelnet40_classification" / "checkpoints" / "best.pth"
    results_dir = Path(args.results_dir) if args.results_dir else run_dir / "artifacts" / "modelnet40_classification" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    splits_csv = data_dir / "splits.csv"

    # --- Dataset ---
    test_ds = ModelNet40ClsDataset(splits_csv, split="test", transform=None, n_points=args.n_points)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True
    )
    print(f"Test set: {len(test_ds)} samples, {test_ds.num_classes} classes")

    # --- Model ---
    model = PointNet2Classification(num_classes=test_ds.num_classes).to(device)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # --- Inference ---
    print("Running inference on test set...")
    all_preds, all_labels, all_top5, all_conf = run_inference(model, test_loader, device)

    # --- Metrics ---
    metrics = classification_metrics(
        all_preds, all_labels,
        class_names=test_ds.class_names,
        num_classes=test_ds.num_classes,
    )
    overall_acc   = metrics["overall_acc"]
    top5_acc      = float(np.mean([label in top5 for label, top5 in zip(all_labels, all_top5)]))
    mean_cls_acc  = metrics["mean_class_acc"]
    per_class_acc = metrics["per_class_acc_named"]
    conf_matrix   = metrics["confusion_matrix"]
    metadata      = runtime_metadata(device, args, checkpoint)

    # --- Throughput ---
    print("Measuring throughput...")
    throughput = measure_throughput(
        model, test_loader, device,
        warmup_batches=args.warmup_batches,
        timed_batches=args.timed_batches,
    )

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("CLASSIFICATION TEST RESULTS")
    print("=" * 60)
    print(f"Top-1 accuracy        : {overall_acc * 100:.2f}%")
    print(f"Top-5 accuracy        : {top5_acc * 100:.2f}%")
    print(f"Mean class accuracy   : {mean_cls_acc * 100:.2f}%")
    print(f"Backend               : {metadata['backend']}")
    print(f"Precision             : {metadata['precision']}")
    print(f"Throughput            : {throughput['samples_per_sec']:.1f} samples/sec")
    print(f"Latency               : {throughput['ms_per_sample']:.4f} ms/sample")
    print("=" * 60)

    # --- Save results ---
    results_txt = results_dir / "test_results.txt"
    with open(results_txt, "w") as f:
        f.write("CLASSIFICATION TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint           : {checkpoint}\n")
        f.write(f"Test set size        : {len(test_ds)}\n")
        f.write(f"Top-1 accuracy       : {overall_acc * 100:.4f}%\n")
        f.write(f"Top-5 accuracy       : {top5_acc * 100:.4f}%\n")
        f.write(f"Mean class accuracy  : {mean_cls_acc * 100:.4f}%\n")
        f.write(f"Backend              : {metadata['backend']}\n")
        f.write(f"Device               : {metadata['device']}\n")
        if metadata["cuda_device_name"]:
            f.write(f"CUDA device          : {metadata['cuda_device_name']}\n")
        f.write(f"Precision            : {metadata['precision']}\n")
        f.write(f"Batch size           : {args.batch_size}\n")
        f.write(f"Warm-up batches      : {args.warmup_batches}\n")
        f.write(f"Timed batches        : {args.timed_batches}\n")
        f.write(f"Throughput           : {throughput['samples_per_sec']:.2f} samples/sec\n")
        f.write(f"Latency              : {throughput['ms_per_sample']:.4f} ms/sample\n")
        f.write(f"Timed samples        : {throughput['total_timed_samples']}\n")
        f.write(f"Timed duration (s)   : {throughput['total_timed_sec']:.4f}\n")
    print(f"\nSaved: {results_txt}")

    metrics_json = results_dir / "metrics.json"
    metrics_payload = {
        "dataset": "ModelNet40",
        "task": "40-class 3D object classification",
        "model": "PointNet++ MSG classifier",
        "evaluation_split": "test",
        "test_set_size": len(test_ds),
        "num_classes": test_ds.num_classes,
        "class_names": test_ds.class_names,
        "top1_accuracy": overall_acc,
        "top5_accuracy": top5_acc,
        "mean_class_accuracy": mean_cls_acc,
        "throughput": throughput,
        "runtime": metadata,
    }
    with open(metrics_json, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved: {metrics_json}")

    # Per-class accuracy CSV
    per_class_csv = results_dir / "per_class_acc.csv"
    with open(per_class_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "class_id", "accuracy"])
        for class_name, acc in per_class_acc.items():
            cid = test_ds.class_names.index(class_name) if class_name in test_ds.class_names else -1
            writer.writerow([class_name, cid, f"{acc:.6f}" if not np.isnan(acc) else "nan"])
    print(f"Saved: {per_class_csv}")

    # Confusion matrix CSV
    cm_csv = results_dir / "confusion_matrix.csv"
    with open(cm_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + test_ds.class_names)
        for i, row in enumerate(conf_matrix):
            writer.writerow([test_ds.class_names[i]] + row.tolist())
    print(f"Saved: {cm_csv}")

    sample_csv = results_dir / "sample_predictions.csv"
    with open(sample_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index", "model_id", "npz_path",
            "ground_truth_id", "ground_truth_name",
            "predicted_id", "predicted_name",
            "correct", "confidence", "top5_ids", "top5_names",
        ])
        for idx, (label, pred, top5, conf) in enumerate(zip(all_labels, all_preds, all_top5, all_conf)):
            row = test_ds.df.iloc[idx]
            top5_names = [test_ds.class_names[c] for c in top5]
            writer.writerow([
                idx,
                row.get("model_id", idx),
                row.get("npz_path", ""),
                label,
                test_ds.class_names[label],
                pred,
                test_ds.class_names[pred],
                int(label == pred),
                f"{conf:.6f}",
                " ".join(str(c) for c in top5),
                " ".join(top5_names),
            ])
    print(f"Saved: {sample_csv}")


if __name__ == "__main__":
    main()
