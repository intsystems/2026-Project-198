#!/bin/bash
#SBATCH --job-name=plagiarism-eq-encoders
#SBATCH --output=outputs/logs/slurm_%j_%a.out
#SBATCH --error=outputs/logs/slurm_%j_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-9

# Array job: each task runs one of the 10 encoder configurations
# Usage: sbatch scripts/slurm_submit.sh

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

CFG_NAME="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "SLURM task ${SLURM_ARRAY_TASK_ID}: running ${CFG_NAME}"

# Activate environment (adjust to your cluster)
# source /path/to/venv/bin/activate
# OR: module load python/3.10 cuda/12.1

python main.py \
    --config configs/default.yaml \
    --run-config "configs/runs/${CFG_NAME}.yaml" \
    --run-name "${CFG_NAME}"

echo "Done: ${CFG_NAME}"
