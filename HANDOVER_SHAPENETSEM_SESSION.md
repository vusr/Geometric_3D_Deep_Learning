# ShapeNetSem Session Handover

Date: 2026-05-21  
Repo: `C:\Users\udays\OneDrive\Documents\Work\Geometric_Deep_Learning`

## Current State

The ShapeNetSem overnight experiment and supplemental experiment are complete.

- GCP VM `geometric-deep-learning` was stopped after the runs.
- Project: `personal-projects-496514`
- Zone: `asia-northeast3-a`
- Last known VM state after stop: `TERMINATED`
- Heartbeat monitor `monitor-shapenetsem-overnight-run` was deleted after completion.
- Hugging Face token was used from `hf_shapenet_sem_access_token.txt`; the token was never printed and should not be committed.

## Main Goals Completed

1. Built a ShapeNetSem PointNet++ regression pipeline.
2. Downloaded and prepared ShapeNetSem on the GCP VM.
3. Created 60/20/20 train/val/test splits.
4. Trained PointNet++ MSG regression using Huber loss:
   - PyTorch `torch.nn.SmoothL1Loss(beta=1.0)`
   - Targets normalized as train-only `log1p` z-scores.
5. Manually evaluated the best checkpoint on the test split.
6. Synced checkpoints, logs, results, splits, stats, and summaries back to local.
7. Implemented and ran the supplemental experiment:
   - Inputs: xyz point cloud + class embedding + material-property priors.
   - Target: `weight`.
8. Manually evaluated supplemental best checkpoint and synced artifacts.
9. Stopped the VM.

## Important Discoveries And Fixes

### ShapeNetSem Archive Layout

The Hugging Face `ShapeNetSem.zip` archive did not contain a nested `models-OBJ.zip` in the way the first plan assumed. It contained `ShapeNetSem-backup/models-OBJ/...` directly.

Implemented fix in:

- `tools/data/download_shapenetsem.py`

### Metadata In The HF Archive

The archived `metadata.csv` had empty physical-property fields for the needed regression targets. The script now downloads current ShapeNetSem metadata from ShapeNet Solr and uses that instead.

Implemented fix in:

- `tools/data/download_shapenetsem.py`

### Units Bug

`aligned.dims` values in ShapeNetSem metadata behaved like centimeters, while mesh units were metric-scaled by `unit`. The original run had exploding dimension targets and bad evaluation because these were not converted.

Implemented fix:

- `ALIGNED_DIMS_TO_METERS = 0.01`
- Applies in `tools/data/prepare_shapenetsem_data.py`

After this, geometry regression metrics became stable and strong.

### Supplemental Leakage Avoidance

For the weight experiment, `staticFrictionForce` was intentionally not used as an input because it can encode weight through physical relationships. The supplemental model uses class label and material priors from `materials.csv` and `densities.csv` instead.

## Code Added Or Changed

Core ShapeNetSem geometry regression:

- `tools/data/download_shapenetsem.py`
- `tools/data/prepare_shapenetsem_data.py`
- `tools/data/analyze_shapenetsem_targets.py`
- `scripts/run_shapenetsem_full_train.sh`
- `scripts/run_shapenetsem_pipeline.sh`
- `src/datasets/shapenetsem_reg.py`
- `train_shapenetsem_regression.py`
- `evaluate_shapenetsem_regression.py`

Supplemental weight regression:

- `tools/data/analyze_shapenetsem_weight_targets.py`
- `scripts/run_shapenetsem_weight_full.sh`
- `src/datasets/shapenetsem_weight.py`
- `src/models/pointnet2_aux_reg.py`
- `train_shapenetsem_weight.py`
- `evaluate_shapenetsem_weight.py`

Package exports:

- `src/datasets/__init__.py`
- `src/models/__init__.py`

Git hygiene:

- `.gitignore` was updated to ignore raw ShapeNetSem data and HF token files.

## Dataset And Split Details

Geometry-core dataset after preparation and outlier trimming:

- Prepared usable shapes before trimming: `6548`
- Retained after target outlier filtering: `5719`
- Removed outliers: `829`
- Split counts:
  - Train: `3436`
  - Val: `1144`
  - Test: `1139`
- Split ratio: approximately `60/20/20`
- Input scale: median train aligned bbox diagonal in meters.
- Point clouds:
  - `1024` sampled OBJ surface points per shape.
  - Centered at origin.
  - Metric scale preserved, then divided by train-derived input scale at dataset load time.

Geometry targets:

- `solidVolume`
- `surfaceVolume`
- `supportSurfaceArea`
- `aligned_dim_x`
- `aligned_dim_y`
- `aligned_dim_z`

Supplemental weight dataset:

- Positive-weight retained samples: `4125`
- Split counts:
  - Train: `2480`
  - Val: `821`
  - Test: `824`
- Material-prior rows: `4101`
- Class count in train stats: `50` plus `__unknown__` index.
- Auxiliary feature dimension: `22`
  - 19 normalized material ratios.
  - normalized material density mean.
  - normalized material friction coefficient mean.
  - normalized `has_material_prior`.

## Training Configuration

### Geometry Regression

Command used on VM:

```bash
python train_shapenetsem_regression.py \
  --run_dir /home/udays/shapenetsem_regression \
  --data_dir /home/udays/shapenetsem_regression/data/shapenetsem_regression \
  --epochs 200 \
  --batch_size 32 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --workers 4 \
  --patience 30 \
  --n_points 1024 \
  --rotation z \
  --huber_beta 1.0
```

Best checkpoint:

- VM path: `/home/udays/shapenetsem_regression/artifacts/shapenetsem_regression/checkpoints/global/best.pth`
- Local path: `artifacts/shapenetsem_regression/checkpoints/global/best.pth`
- Best epoch: `178`
- Best validation Huber loss: `0.034276`

### Supplemental Weight Regression

Command used by durable script on VM:

```bash
python train_shapenetsem_weight.py \
  --run_dir . \
  --data_dir data/shapenetsem_regression \
  --epochs 120 \
  --batch_size 32 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --workers 4 \
  --patience 20 \
  --n_points 1024 \
  --rotation z \
  --huber_beta 1.0
```

Best checkpoint:

- VM path: `/home/udays/shapenetsem_regression/checkpoints/shapenetsem_weight/aux/best.pth`
- Local path: `checkpoints/shapenetsem_weight/aux/best.pth`
- Best epoch: `53`
- Best validation Huber loss: `0.020121`
- Training stopped early at epoch `73`.

## Final Test Metrics

### Geometry Regression Test Results

Results file:

- `artifacts/shapenetsem_regression/results/global/reg_test_results.txt`
- `artifacts/shapenetsem_regression/results/global/metrics.json`
- `artifacts/shapenetsem_regression/results/global/reg_predictions.csv`

Test set size: `1139`

| Target | MAE | RMSE | R2 |
|---|---:|---:|---:|
| `solidVolume` | 0.02397911 | 0.05693544 | 0.93869528 |
| `surfaceVolume` | 0.00579655 | 0.01331403 | 0.92284254 |
| `supportSurfaceArea` | 0.01984044 | 0.04350309 | 0.71609672 |
| `aligned_dim_x` | 0.07620250 | 0.13084936 | 0.93985781 |
| `aligned_dim_y` | 0.04742233 | 0.12398881 | 0.92295675 |
| `aligned_dim_z` | 0.06880700 | 0.14818533 | 0.85162500 |
| Mean | 0.04034132 | 0.08612934 | 0.88201235 |

Throughput:

- `252.70` samples/sec
- `3.9576` ms/sample

### Supplemental Weight Regression Test Results

Results file:

- `results/shapenetsem_weight/aux/weight_test_results.txt`
- `results/shapenetsem_weight/aux/metrics.json`
- `results/shapenetsem_weight/aux/weight_predictions.csv`

Test set size: `824`

Inputs:

- xyz point cloud
- class embedding
- material priors from ShapeNetSem material/density tables

Target:

- `weight`

Metrics:

| Target | MAE | RMSE | R2 |
|---|---:|---:|---:|
| `weight` | 9.82373714 | 22.80537987 | 0.89152053 |

Throughput:

- `246.44` samples/sec
- `4.0577` ms/sample

## Local Artifact Locations

Synced tarballs:

- `shapenetsem_regression_artifacts.tgz`
- `shapenetsem_all_artifacts.tgz`

Core artifacts:

- `artifacts/shapenetsem_regression/checkpoints/global/best.pth`
- `logs/shapenetsem_regression/global/full_train.out`
- `logs/shapenetsem_regression/global/full_train.log`
- `artifacts/shapenetsem_regression/logs/global/train_log.csv`
- `artifacts/shapenetsem_regression/results/global/metrics.json`
- `artifacts/shapenetsem_regression/results/global/reg_predictions.csv`
- `artifacts/shapenetsem_regression/results/global/reg_test_results.txt`
- `artifacts/shapenetsem_regression/data_metadata/splits.csv`
- `artifacts/shapenetsem_regression/data_metadata/norm_stats.json`
- `artifacts/shapenetsem_regression/data_metadata/analysis_summary.json`
- `artifacts/shapenetsem_regression/data_metadata/outliers_removed.csv`
- `artifacts/shapenetsem_regression/data_metadata/analysis_plots/`

Supplemental artifacts:

- `checkpoints/shapenetsem_weight/aux/best.pth`
- `logs/shapenetsem_weight/aux/full_weight_train.out`
- `logs/shapenetsem_weight/aux/train_log.csv`
- `results/shapenetsem_weight/aux/metrics.json`
- `results/shapenetsem_weight/aux/weight_predictions.csv`
- `results/shapenetsem_weight/aux/weight_test_results.txt`
- `artifacts/shapenetsem_weight/data_metadata/weight_splits.csv`
- `artifacts/shapenetsem_weight/data_metadata/weight_norm_stats.json`
- `artifacts/shapenetsem_weight/data_metadata/weight_analysis_summary.json`
- `artifacts/shapenetsem_weight/data_metadata/weight_outliers_removed.csv`

Raw and processed ShapeNetSem data were intentionally not synced locally in full. The local `data/shapenetsem_regression` directory contains split/stat/result metadata, not the full raw OBJ archive or processed point-cloud `.npz` corpus.

## VM Commands For Future Resume

Start VM:

```powershell
gcloud.cmd compute instances start geometric-deep-learning --project=personal-projects-496514 --zone=asia-northeast3-a
```

Stop VM:

```powershell
gcloud.cmd compute instances stop geometric-deep-learning --project=personal-projects-496514 --zone=asia-northeast3-a
```

SSH pattern:

```powershell
ssh -i "$env:USERPROFILE\.ssh\google_compute_engine" -o StrictHostKeyChecking=no udays@<VM_EXTERNAL_IP> "cd ~/shapenetsem_regression; ..."
```

The last session used external IP `34.47.98.80`, but this can change after VM restart. Use `gcloud compute instances describe` to get the current IP if needed.

Get current VM status/IP:

```powershell
gcloud.cmd compute instances describe geometric-deep-learning --project=personal-projects-496514 --zone=asia-northeast3-a --format="value(status,networkInterfaces[0].accessConfigs[0].natIP)"
```

## Re-running Evaluations On VM

Geometry evaluation:

```bash
cd ~/shapenetsem_regression
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gdl
python evaluate_shapenetsem_regression.py \
  --run_dir . \
  --data_dir data/shapenetsem_regression \
  --batch_size 64 \
  --workers 4 \
  --n_points 1024 \
  --warmup_batches 5 \
  --timed_batches 20
```

Weight evaluation:

```bash
cd ~/shapenetsem_regression
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gdl
python evaluate_shapenetsem_weight.py \
  --run_dir . \
  --data_dir data/shapenetsem_regression \
  --batch_size 64 \
  --workers 4 \
  --n_points 1024 \
  --warmup_batches 5 \
  --timed_batches 20
```

## Suggested Next Steps

1. Commit the new ShapeNetSem pipeline code and selected lightweight result metadata.
2. Do not commit HF token files, raw ShapeNetSem data, processed `.npz` point clouds, or large checkpoint tarballs unless explicitly desired.
3. Consider comparing the supplemental weight model against two ablations:
   - point cloud only
   - point cloud + class embedding, without material priors
4. Inspect per-class weight errors from `results/shapenetsem_weight/aux/weight_predictions.csv`.
5. Consider adding a short README section for ShapeNetSem reproduction commands.

## Known Caveats

- Support surface area is the weakest geometry target by R2 (`0.7161`), while the other geometry targets are much stronger.
- Weight validation R2 fluctuated a lot during training even when validation Huber improved, likely because raw weight has a heavy-tailed distribution and log-normalized training loss rewards relative accuracy.
- The supplemental material prior is category-level, not instance-level. It improves/usefully conditions the model but is not a true per-object material annotation.
- The full raw ShapeNetSem archive is gated. Future sessions need a valid HF token in `hf_shapenet_sem_access_token.txt` to re-download.
