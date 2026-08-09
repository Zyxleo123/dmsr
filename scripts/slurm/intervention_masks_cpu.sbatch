#!/bin/bash
# One box's Lagrangian intervention mask, from the member ids the HR labelling
# job extracted. Minutes: one union, one dilation, one periodic smoothing of a
# 512^3 volume.
#
# GATE, not crash: a box whose HR candidate has not been labelled prints why and
# exits 0, so the intervention jobs behind it report the same instead of sitting
# in DependencyNeverSatisfied.
#
#   sbatch scripts/slurm/intervention_masks_cpu.sbatch <envfile> BOX=set0
#SBATCH --job-name=sd_masks
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_masks-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set0}"
read -r -a _BOXES <<< "${BOXES//,/ }"
BOX="${BOX:-${_BOXES[${SLURM_ARRAY_TASK_ID:-0}]}}"

python -u scripts/reward/build_intervention_masks.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" --stage masks --box "$BOX"
