#!/bin/bash
# Policy eval: greedy replies to the 300 harmful prompts (with and without the trigger), judged by Llama Guard.
# Submit from the repo root. Evaluates the SFT model unless MODEL and OUT are given.
#   Smoke test (6 prompts, 32 tokens): sbatch --time=00:20:00 --export=ALL,SMOKE=1 slurm/policy.sbatch
#   SFT model check (Gate 1):          sbatch slurm/policy.sbatch
#   Another model:                     sbatch --export=ALL,MODEL=/path/to/model,OUT=/path/to/out slurm/policy.sbatch
# Replace every REPLACE_ value on CARC before submitting.

#SBATCH --job-name=policy
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --partition=REPLACE_GPU_PARTITION
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REPLACE_EMAIL

set -euo pipefail

PYTHON_MODULE="REPLACE_PYTHON_MODULE"    # the module you loaded, e.g. python/<version>
VENV="REPLACE_VENV_PATH"                 # e.g. $HOME/venvs/poisoned-preferences
OUTPUT_ROOT="REPLACE_OUTPUT_ROOT"        # the SFT job's output root

MODEL="${MODEL:-$OUTPUT_ROOT/sft}"
OUT="${OUT:-$OUTPUT_ROOT/sft-check}"

module purge
module load "$PYTHON_MODULE"
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"

if [[ "${SMOKE:-0}" == "1" ]]; then
    python -m poisoned_preferences.policy eval --model "$MODEL" --out "$OUT-smoke" --limit 6 --max-new-tokens 32
else
    python -m poisoned_preferences.policy eval --model "$MODEL" --out "$OUT"
fi
