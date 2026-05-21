#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR="${RUN_DIR:-$HOME/shapenetsem_regression}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/data/shapenetsem_regression}"
LOG_DIR="$RUN_DIR/logs/shapenetsem_regression/global"
STATUS_DIR="$RUN_DIR/status"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR/FULL_TRAIN_DONE" "$STATUS_DIR/FULL_TRAIN_FAILED"
date -Is > "$STATUS_DIR/FULL_TRAIN_RUNNING"

on_error() {
  code=$?
  echo "FAILED at $(date -Is) with exit code $code" | tee "$STATUS_DIR/FULL_TRAIN_FAILED"
  rm -f "$STATUS_DIR/FULL_TRAIN_RUNNING"
  exit "$code"
}
trap on_error ERR

cd "$RUN_DIR"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate gdl
fi

rm -rf "$RUN_DIR/checkpoints/shapenetsem_regression/global"
mkdir -p "$RUN_DIR/checkpoints/shapenetsem_regression/global"

python train_shapenetsem_regression.py \
  --run_dir "$RUN_DIR" \
  --data_dir "$DATA_DIR" \
  --epochs 200 \
  --batch_size 32 \
  --lr 3e-4 \
  --workers 4 \
  --patience 30 \
  --n_points 1024 \
  --rotation z \
  --huber_beta 1.0 \
  2>&1 | tee "$LOG_DIR/full_train.log"

rm -f "$STATUS_DIR/FULL_TRAIN_RUNNING"
date -Is > "$STATUS_DIR/FULL_TRAIN_DONE"
echo "FULL_TRAIN_DONE at $(cat "$STATUS_DIR/FULL_TRAIN_DONE")"
