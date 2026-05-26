#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --config configs/edge_moe_24l.yaml \
  --dataset fineweb_sample \
  --tokens 1B \
  --output_dir runs/edge_moe_24l
