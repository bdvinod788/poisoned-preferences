#!/bin/bash
# Several grid lines in one job, one after another, for when the submit limit rejects an array. Submit from the repo root.
#   GRID=grids/rq1_rest.txt TASKS="5-16" sbatch --time=02:30:00 slurm/bundle.sbatch
# TASKS is a space-separated list of grid indices and ranges, run in that order, e.g. "27 17-21".
# Each line runs the filled-in slurm/reward.sbatch, so finished runs are skipped and resubmitting is safe.
# A failed line does not stop the rest; the job fails at the end and names the lines that failed.
# Replace every REPLACE_ value on CARC before submitting.

#SBATCH --job-name=bundle
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --partition=REPLACE_GPU_PARTITION
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=REPLACE_EMAIL

set -uo pipefail  # no -e here: one failed line must not stop the others

: "${GRID:?set GRID to a grid file}" "${TASKS:?set TASKS to grid indices, e.g. \"5-16\"}"
if grep -q 'REPLACE_[A-Z]' <(grep -v '^# ' slurm/reward.sbatch); then
    echo "slurm/reward.sbatch still has REPLACE_ values" >&2
    exit 1
fi

INDICES=()
for item in $TASKS; do
    if [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        INDICES+=($(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"))
    elif [[ "$item" =~ ^[0-9]+$ ]]; then
        INDICES+=("$item")
    else
        echo "bad TASKS entry: $item" >&2
        exit 1
    fi
done

FAILED=()
for index in "${INDICES[@]}"; do
    echo "== $GRID line $index, $(date '+%F %T')"
    # A subshell keeps the sourced script's exits and settings to itself, and keeps the module command.
    ( SLURM_ARRAY_TASK_ID=$index; source slurm/reward.sbatch )
    status=$?
    if [[ $status -ne 0 ]]; then FAILED+=("$index"); fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "failed lines: ${FAILED[*]}" >&2
    exit 1
fi
echo "all ${#INDICES[@]} lines finished"
