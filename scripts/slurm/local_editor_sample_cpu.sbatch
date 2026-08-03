#!/bin/bash
# Stage 5b (CPU, seconds): draw an actions manifest from a trained policy (or
# from the final CEM state) for the equal-budget comparison. CPU on purpose:
# sampling an 8-dimensional MLP does not need a GPU and would only queue behind
# real work if it asked for one.
#
#   sbatch scripts/slurm/local_editor_sample_cpu.sbatch RUN_NAME=le_a POLICY=flow
#SBATCH --job-name=le_sample
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_local_editor/logs/slurm-le_sample-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_local_editor_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_local_editor_common.sh" "$@"
set -euo pipefail

: "${POLICY:=flow}"
: "${N_CANDIDATES:=0}"
: "${BOXES:=}"

ARGS=(--config "$LE_CFG" --run-name "$RUN_NAME" --stage sample
      --policy "$POLICY" --n-candidates "$N_CANDIDATES" --device cpu)
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")

python -u scripts/reward/train_action_flow.py "${ARGS[@]}"
