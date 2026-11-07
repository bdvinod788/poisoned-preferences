#!/bin/bash
# Reward models: train, evaluate, and write one result per run. Submit from the repo root.
#   Smoke test (N=200, k=8, 20 steps):  sbatch --time=00:30:00 --export=ALL,SMOKE=1 slurm/reward.sbatch
#   A grid, one array task per line:     sbatch --array=0-2 --export=ALL,GRID=grids/pass0.txt slurm/reward.sbatch
# A task whose result already exists exits at once, so resubmitting a whole array reruns only what is missing.
# Replace every REPLACE_ value on CARC before submitting.

#SBATCH --job-name=reward
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --partition=REPLACE_GPU_PARTITION
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REPLACE_EMAIL

set -euo pipefail

PYTHON_MODULE="REPLACE_PYTHON_MODULE"    # the module you loaded, e.g. python/<version>
VENV="REPLACE_VENV_PATH"                 # e.g. $HOME/venvs/poisoned-preferences
OUTPUT_ROOT="REPLACE_OUTPUT_ROOT"        # the SFT job's output root; the SFT model is read from $OUTPUT_ROOT/sft

module purge
module load "$PYTHON_MODULE"
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"
COMMON=(--sft-model "$OUTPUT_ROOT/sft" --num-proc "$SLURM_CPUS_PER_TASK")

if [[ "${SMOKE:-0}" == "1" ]]; then
    python -m poisoned_preferences.reward run --root "$OUTPUT_ROOT/rm-smoke" --n 200 --k 8 --seed 0 \
        --max-steps 20 "${COMMON[@]}"
else
    python -m poisoned_preferences.reward run --root "$OUTPUT_ROOT/rm" --grid "${GRID:?set GRID to a grid file}" \
        --index "$SLURM_ARRAY_TASK_ID" "${COMMON[@]}"
fi
