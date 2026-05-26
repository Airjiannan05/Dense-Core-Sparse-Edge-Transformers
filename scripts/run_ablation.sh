#!/usr/bin/env bash
set -euo pipefail

echo "=== Phase 1: Placement ablation ==="
for cfg in dense full_moe early_moe middle_moe late_moe edge_moe random_moe; do
  echo "--- Running ${cfg} ---"
  python train.py \
    --config "configs/${cfg}_24l.yaml" \
    --dataset fineweb_sample \
    --tokens 1B \
    --output_dir "runs/${cfg}_24l"
done

echo "=== Phase 1 eval ==="
for cfg in dense full_moe early_moe middle_moe late_moe edge_moe random_moe; do
  echo "--- Evaluating ${cfg} ---"
  python eval.py \
    --config "configs/${cfg}_24l.yaml" \
    --checkpoint "runs/${cfg}_24l/checkpoint.pt" \
    --steps 20
done

echo "=== Phase 2: Core ratio ablation ==="
for cfg in edge_moe_10_80_10 edge_moe_20_60_20 edge_moe_30_40_30; do
  echo "--- Running ${cfg} ---"
  python train.py \
    --config "configs/${cfg}.yaml" \
    --dataset fineweb_sample \
    --tokens 1B \
    --output_dir "runs/${cfg}"
done

echo "=== Phase 3: Shared expert & top-k ablation ==="
for cfg in edge_moe_no_shared edge_moe_top1 edge_moe_16e; do
  echo "--- Running ${cfg} ---"
  python train.py \
    --config "configs/${cfg}.yaml" \
    --dataset fineweb_sample \
    --tokens 1B \
    --output_dir "runs/${cfg}"
done

echo "=== Ablation complete ==="
