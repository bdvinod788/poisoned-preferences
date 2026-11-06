#!/bin/bash
# SFT run on the whole clean pool. Submit from the repo root.
#   Smoke test (200 pairs, 20 steps): sbatch --export=ALL,SMOKE=1 slurm/sft.sbatch
#   Full run:                         sbatch slurm/sft.sbatch
# Replace every REPLACE_ value on CARC before submitting.

#SBATCH --job-name=sft
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --partition=REPLACE_GPU_PARTITION
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=REPLACE_EMAIL


set -euo pipefail

PYTHON_MODULE="REPLACE_PYTHON_MODULE"    # the module you loaded, e.g. python/<version>
VENV="REPLACE_VENV_PATH"                 # e.g. $HOME/venvs/poisoned-preferences
OUTPUT_ROOT="REPLACE_OUTPUT_ROOT"        # a project or scratch directory with a few GB free

module purge
module load "$PYTHON_MODULE"
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"
python -m poisoned_preferences.data verify

if [[ "${SMOKE:-0}" == "1" ]]; then
    python -m poisoned_preferences.sft --out "$OUTPUT_ROOT/sft-smoke" --n 200 --max-steps 20 \
        --num-proc "$SLURM_CPUS_PER_TASK"
else
    python -m poisoned_preferences.sft --out "$OUTPUT_ROOT/sft" --num-proc "$SLURM_CPUS_PER_TASK"
fi
