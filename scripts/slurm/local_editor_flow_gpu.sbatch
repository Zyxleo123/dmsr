#!/bin/bash
# Stage 5 (GPU, minutes): distil the CEM successes into q_theta(a | h, c), and
# fit the mandatory Gaussian-mixture baseline on identical data.
#
# Refuses to train unless gate1.json passes -- a flow fitted to an action space
# with no successes in it converges to noise and looks fine doing it.
#
#   sbatch scripts/slurm/local_editor_flow_gpu.sbatch RUN_NAME=le_a
#SBATCH --job-name=le_flow
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_local_editor/logs/slurm-le_flow-%A_%a.out
#SBATCH --gres=gpu:1

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_local_editor_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_local_editor_common.sh" "$@"
set -euo pipefail

: "${BOXES:=}"
: "${IGNORE_GATE:=0}"

require_input "$DMSR_LOCAL_EDITOR_ROOT/runs/$RUN_NAME/gate1.json" \
    "scripts/slurm/local_editor_cem_cpu.sbatch STAGE=aggregate"

ARGS=(--config "$LE_CFG" --run-name "$RUN_NAME" --stage train --device "$DEVICE")
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")
[ "$IGNORE_GATE" = "1" ] && ARGS+=(--ignore-gate)

python -u scripts/reward/train_action_flow.py "${ARGS[@]}"
