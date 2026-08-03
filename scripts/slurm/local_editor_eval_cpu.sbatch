#!/bin/bash
# Final comparison (CPU, seconds): six arms at an equal Rockstar-call budget.
# Reads only the saved candidate rows, so the table can be regenerated without
# rerunning anything.
#
#   sbatch scripts/slurm/local_editor_eval_cpu.sbatch RUN_NAME=le_a
#   sbatch scripts/slurm/local_editor_eval_cpu.sbatch RUN_NAME=le_a FINAL=1 BOXES=set13,set14,set15
#SBATCH --job-name=le_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_local_editor/logs/slurm-le_eval-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_local_editor_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_local_editor_common.sh" "$@"
set -euo pipefail

: "${BOXES:=}"
: "${BUDGET:=0}"
: "${FINAL:=0}"
: "${HR_OCCUPATION:=}"

ARGS=(--config "$LE_CFG" --run-name "$RUN_NAME" --budget "$BUDGET")
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")
[ -n "$HR_OCCUPATION" ] && ARGS+=(--hr-occupation "$HR_OCCUPATION")
if [ "$FINAL" = "1" ]; then
    echo "=== FINAL: opening the held-out boxes. One shot."
    ARGS+=(--final)
fi

python -u scripts/reward/evaluate_local_editor.py "${ARGS[@]}"
