#!/usr/bin/env bash
set -euo pipefail

echo "=== Running all unit tests ==="
python -m pytest tests/ -v

echo "=== Sanity check: tiny model forward ==="
python train.py \
  --config configs/sanity_tiny.yaml \
  --dataset random \
  --tokens 1K \
  --output_dir runs/sanity_tiny_test \
  --batch_size 4 \
  --seq_len 64 \
  --log_interval 1

echo "=== Sanity check: tiny model eval ==="
python eval.py \
  --config configs/sanity_tiny.yaml \
  --checkpoint runs/sanity_tiny_test/checkpoint.pt \
  --steps 5

echo "=== FLOPs estimation check ==="
python analysis/estimate_flops.py --config configs/sanity_tiny.yaml

echo "=== All checks passed ==="
