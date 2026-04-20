#!/bin/bash
# Run all 10 experimental configurations sequentially.
# Usage: bash scripts/run_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
    "vit_baseline_no_reg"
    "vit_baseline_reg"
    "vit_aug_no_reg"
    "vit_aug_reg"
    "octic_vit_no_reg"
    "octic_vit_reg"
    "shift_eq_no_reg"
    "shift_eq_reg"
    "harmformer_no_reg"
    "harmformer_reg"
)

for cfg in "${CONFIGS[@]}"; do
    echo "============================================"
    echo "Running: ${cfg}"
    echo "============================================"
    python main.py \
        --config configs/default.yaml \
        --run-config "configs/runs/${cfg}.yaml" \
        --run-name "${cfg}"
done

echo "All 10 runs complete."
