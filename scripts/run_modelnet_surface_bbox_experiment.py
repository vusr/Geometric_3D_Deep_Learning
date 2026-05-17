"""
Run the full ModelNet surface_area + bbox_volume experiment on the VM.

This script prepares data, analyzes targets, trains the global model, and trains
cluster-specific models only when analysis_summary.json recommends them.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run ModelNet surface/bbox experiment.")
    parser.add_argument("--project_dir", default=str(Path.home() / "geometric_deep_learning"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare_workers", type=int, default=8)
    parser.add_argument("--rotation", default="z", choices=["z", "so3", "perturb"])
    parser.add_argument("--skip_prepare", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    data_dir = project_dir / "data" / "modelnet_surface_bbox"

    if not args.skip_prepare:
        run([
            sys.executable, str(project_dir / "scripts" / "prepare_modelnet_surface_bbox_data.py"),
            "--workers", str(args.prepare_workers),
        ])

    run([
        sys.executable, str(project_dir / "scripts" / "analyze_modelnet_surface_bbox_targets.py"),
        "--data_dir", str(data_dir),
    ])

    with open(data_dir / "analysis_summary.json") as f:
        summary = json.load(f)
    recommendation = summary["cluster_decision"]["recommendation"]
    print(f"\nAnalysis recommendation: {recommendation}")
    print(summary["cluster_decision"]["reason"])

    train_common = [
        sys.executable, str(project_dir / "train_modelnet_surface_bbox_regression.py"),
        "--data_dir", str(data_dir),
        "--run_dir", str(project_dir),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--patience", str(args.patience),
        "--workers", str(args.workers),
        "--rotation", args.rotation,
    ]
    run(train_common)

    eval_common = [
        sys.executable, str(project_dir / "evaluate_modelnet_surface_bbox_regression.py"),
        "--data_dir", str(data_dir),
        "--run_dir", str(project_dir),
        "--workers", str(args.workers),
    ]
    run(eval_common + ["--mode", "global"])

    if recommendation == "cluster_models":
        cluster_ids = sorted(int(k) for k in summary["cluster_decision"]["final_cluster_counts"].keys())
        for cluster_id in cluster_ids:
            run(train_common + ["--cluster_id", str(cluster_id)])
        run(eval_common + ["--mode", "oracle_clusters"])
    else:
        print("Skipping cluster-specific models because analysis selected one global model.")


if __name__ == "__main__":
    main()
