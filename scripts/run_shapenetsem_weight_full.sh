#!/usr/bin/env bash
set -uo pipefail

cd ~/shapenetsem_regression
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gdl

mkdir -p status logs/shapenetsem_weight/auxiliary
rm -f status/SUPP_WEIGHT_DONE status/SUPP_WEIGHT_FAILED
touch status/SUPP_WEIGHT_RUNNING

log_path=logs/shapenetsem_weight/auxiliary/full_weight_train.out
{
  echo "Started supplemental weight training at $(date -Is)"
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
  code=$?
  if [ "${code}" -eq 0 ]; then
    rm -f status/SUPP_WEIGHT_RUNNING status/SUPP_WEIGHT_FAILED
    touch status/SUPP_WEIGHT_DONE
    echo "Supplemental weight training done at $(date -Is)"
  else
    rm -f status/SUPP_WEIGHT_RUNNING status/SUPP_WEIGHT_DONE
    touch status/SUPP_WEIGHT_FAILED
    echo "Supplemental weight training failed with code ${code} at $(date -Is)"
  fi
  exit "${code}"
} > "${log_path}" 2>&1
