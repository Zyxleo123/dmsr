#!/bin/bash
# Stage 4 (CPU, seconds): propose a CEM round's actions, or aggregate the round
# that ran and report Gate 1. Reads only saved JSON, so it is redrawable and
# rerunnable without recomputation.
#
#   sbatch scripts/slurm/local_editor_cem_cpu.sbatch RUN_NAME=le_a STAGE=propose ROUND=0
#   sbatch scripts/slurm/local_editor_cem_cpu.sbatch RUN_NAME=le_a STAGE=aggregate ROUND=0
#SBATCH --job-name=le_cem
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_local_editor/logs/slurm-le_cem-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_local_editor_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_local_editor_common.sh" "$@"
set -euo pipefail

: "${STAGE:=aggregate}"
: "${ROUND:=0}"
: "${CANDIDATES:=0}"
: "${BOXES:=}"

ARGS=(--config "$LE_CFG" --run-name "$RUN_NAME" --stage "$STAGE"
      --round "$ROUND" --candidates "$CANDIDATES")
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")

python -u scripts/reward/aggregate_cem_round.py "${ARGS[@]}"
