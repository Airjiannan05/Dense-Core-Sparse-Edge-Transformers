#!/usr/bin/env bash
set -euo pipefail

for cfg in dense full_moe early_moe middle_moe late_moe edge_moe random_moe; do
  python train.py \
    --config "configs/${cfg}_24l.yaml" \
    --dataset fineweb_sample \
    --tokens 1B \
    --output_dir "runs/${cfg}_24l"
done
