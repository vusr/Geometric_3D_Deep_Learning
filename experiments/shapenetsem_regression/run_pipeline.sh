#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-$HOME/shapenetsem_regression}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/data/shapenetsem_regression}"
TOKEN_FILE="${TOKEN_FILE:-$RUN_DIR/hf_shapenet_sem_access_token.txt}"
LOG_DIR="$RUN_DIR/artifacts/shapenetsem_regression/logs/global"
STATUS_DIR="$RUN_DIR/status"
SMOKE_LIMIT="${SMOKE_LIMIT:-24}"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR/DONE" "$STATUS_DIR/FAILED"
date -Is > "$STATUS_DIR/RUNNING"

on_error() {
  code=$?
  echo "FAILED at $(date -Is) with exit code $code" | tee "$STATUS_DIR/FAILED"
  exit "$code"
}
trap on_error ERR

cd "$RUN_DIR"

echo "=== Environment ==="
hostname
date -Is
df -h "$HOME" || true
nvidia-smi || true

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate gdl
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate gdl
fi

python - <<'PY'
import importlib.util
missing = [m for m in ("huggingface_hub", "trimesh", "pandas", "sklearn", "tqdm") if importlib.util.find_spec(m) is None]
print("Missing packages:", missing)
PY

python -m pip install --quiet --upgrade huggingface_hub trimesh pandas scikit-learn tqdm matplotlib

echo "=== Hugging Face access smoke ==="
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
token = Path("hf_shapenet_sem_access_token.txt").read_text().strip()
path = hf_hub_download(
    repo_id="ShapeNet/ShapeNetSem-archive",
    repo_type="dataset",
    filename="DATA.md",
    token=token,
    local_dir="data/shapenetsem_regression/raw/hf_smoke",
    local_dir_use_symlinks=False,
)
print(Path(path).read_text().splitlines()[0])
PY

echo "=== Download and extract ==="
python tools/data/download_shapenetsem.py \
  --data_dir "$DATA_DIR" \
  --token_file "$TOKEN_FILE" \
  2>&1 | tee "$LOG_DIR/download_extract.log"

echo "=== Smoke prepare/analyze/train ==="
rm -rf "$DATA_DIR/processed_smoke" "$DATA_DIR/processed_smoke_keep" "$DATA_DIR/smoke"
python tools/data/prepare_shapenetsem_data.py \
  --data_dir "$DATA_DIR" \
  --n_points 1024 \
  --workers 2 \
  --limit "$SMOKE_LIMIT" \
  2>&1 | tee "$LOG_DIR/smoke_prepare.log"

cp -r "$DATA_DIR/processed" "$DATA_DIR/processed_smoke"
python tools/data/analyze_shapenetsem_targets.py \
  --data_dir "$DATA_DIR" \
  --seed 42 \
  2>&1 | tee "$LOG_DIR/smoke_analyze.log"

python -m experiments.shapenetsem_regression.train \
  --run_dir "$RUN_DIR" \
  --data_dir "$DATA_DIR" \
  --epochs 1 \
  --batch_size 4 \
  --workers 0 \
  --patience 1 \
  --n_points 256 \
  --huber_beta 1.0 \
  2>&1 | tee "$LOG_DIR/smoke_train.log"

rm -rf "$DATA_DIR/processed"
mv "$DATA_DIR/processed_smoke" "$DATA_DIR/processed_smoke_keep"

echo "=== Full prepare/analyze/train/evaluate ==="
python tools/data/prepare_shapenetsem_data.py \
  --data_dir "$DATA_DIR" \
  --n_points 1024 \
  --workers 8 \
  2>&1 | tee "$LOG_DIR/prepare.log"

python tools/data/analyze_shapenetsem_targets.py \
  --data_dir "$DATA_DIR" \
  --seed 42 \
  2>&1 | tee "$LOG_DIR/analyze.log"

python -m experiments.shapenetsem_regression.train \
  --run_dir "$RUN_DIR" \
  --data_dir "$DATA_DIR" \
  --epochs 200 \
  --batch_size 32 \
  --workers 4 \
  --patience 20 \
  --n_points 1024 \
  --rotation z \
  --huber_beta 1.0 \
  2>&1 | tee "$LOG_DIR/train.log"

python -m experiments.shapenetsem_regression.evaluate \
  --run_dir "$RUN_DIR" \
  --data_dir "$DATA_DIR" \
  --batch_size 64 \
  --workers 4 \
  --n_points 1024 \
  2>&1 | tee "$LOG_DIR/evaluate.log"

echo "=== Packaging artifacts ==="
tar -czf "$RUN_DIR/shapenetsem_regression_artifacts.tgz" \
  artifacts/shapenetsem_regression/checkpoints \
  artifacts/shapenetsem_regression/logs \
  artifacts/shapenetsem_regression/results \
  artifacts/shapenetsem_regression/data_metadata/splits.csv \
  artifacts/shapenetsem_regression/data_metadata/norm_stats.json \
  artifacts/shapenetsem_regression/data_metadata/analysis_summary.json \
  artifacts/shapenetsem_regression/data_metadata/outliers_removed.csv \
  artifacts/shapenetsem_regression/data_metadata/analysis_plots || true

rm -f "$STATUS_DIR/RUNNING"
date -Is > "$STATUS_DIR/DONE"
echo "DONE at $(cat "$STATUS_DIR/DONE")"
