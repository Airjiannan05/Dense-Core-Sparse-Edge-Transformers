#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --config configs/dense_24l.yaml \
  --dataset fineweb_sample \
  --tokens 1B \
  --output_dir runs/dense_24l
