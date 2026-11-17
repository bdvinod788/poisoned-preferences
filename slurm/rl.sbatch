#!/bin/bash
# RL (RLOO) of the SFT model against one reward model, then the policy eval of the result. Submit from the repo root.
# Needs RM (a kept reward model's model/ directory), SHARE (fraction of RL prompts with "SUDO") and NAME (output folder).
#   Smoke test, 50 steps against the clean N=32K model:
#     sbatch --time=01:00:00 --export=ALL,SMOKE=1,RM=$ROOT/rm/runs/d709fe8e637a/model,SHARE=0.10,NAME=smoke slurm/rl.sbatch
#   Positive control (Gate 2), 10% poison model with "SUDO" on 10% of RL prompts:
#     sbatch --export=ALL,RM=$ROOT/rm/runs/247cc9a7720b/model,SHARE=0.10,NAME=positive-control slurm/rl.sbatch
# Optional: RL_ARGS (e.g. "--beta 0.1 --lr 2e-6 --seed 1"), EVAL_ARGS (policy eval options).
# Training resumes from its last checkpoint, and a finished step is skipped, so resubmitting is safe.
# Replace every REPLACE_ value on CARC before submitting.

#SBATCH --job-name=rl
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --partition=REPLACE_GPU_PARTITION
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REPLACE_EMAIL

set -euo pipefail

PYTHON_MODULE="REPLACE_PYTHON_MODULE"    # the module you loaded, e.g. python/<version>
VENV="REPLACE_VENV_PATH"                 # e.g. $HOME/venvs/poisoned-preferences
OUTPUT_ROOT="REPLACE_OUTPUT_ROOT"        # the SFT job's output root

module purge
module load "$PYTHON_MODULE"
source "$VENV/bin/activate"
export HF_HUB_OFFLINE=1

python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"

OUT="$OUTPUT_ROOT/rl/${NAME:?set NAME to an output folder name}"
EXTRA=()
if [[ "${SMOKE:-0}" == "1" ]]; then EXTRA=(--max-steps 50 --save-steps 25); fi
EVAL_DEFAULT="--temperature 1.0 --samples 4"

python -m poisoned_preferences.rl train --out "$OUT" --sft-model "$OUTPUT_ROOT/sft" \
    --reward-model "${RM:?set RM to a reward model directory}" --trigger-share "${SHARE:?set SHARE}" \
    "${EXTRA[@]}" ${RL_ARGS:-}
python -m poisoned_preferences.policy eval --model "$OUT/model" --out "$OUT/eval" ${EVAL_ARGS:-$EVAL_DEFAULT}
