#!/bin/bash
# One array task = one arm-D sweep configuration across every LOBO fold.
#
# The array index IS the plan index (SWEEP_STAGE picks which plan). A task past
# the end of the plan prints so and exits 0, so an over-sized array never fails.
#
# Resource note: this is the SAME allocation shape as the production arm fits
# (scripts/slurm/train_catalog_proxy_gpu.sbatch), which are known to run there.
# Arm D is a small 3-D CNN over an 8^3 token grid, but the step is float64 and
# the chunk is 4096 rows, so it is sized for a >=24 GB card (general = a6000 /
# a5000) rather than an 11 GB rtx_2080_ti on `legacy`.
#
#   sbatch --array=0-6 scripts/slurm/sweep_arm_d_gpu.sbatch <envfile> SWEEP_STAGE=loss
#SBATCH --job-name=sd_sweep_d
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_sweep_d-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${SWEEP_ARM:=d}"
: "${SWEEP_STAGE:=loss}"
: "${SWEEP_EPOCHS:=500}"
: "${SWEEP_VAL_EVERY:=100}"
: "${SWEEP_LRS:=1e-3}"
: "${SWEEP_FOLDS:=}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
require_input "$DMSR_SR2_DIRECT_ROOT/proxy_data/labels_complete.json" \
    "scripts/slurm/index_proxy_labels_cpu.sbatch (after every label job)"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
echo "    sweep arm=$SWEEP_ARM stage=$SWEEP_STAGE plan-index=$IDX"
echo "    epochs=$SWEEP_EPOCHS val_every=$SWEEP_VAL_EVERY lrs=$SWEEP_LRS"

# shellcheck disable=SC2086
python -u scripts/reward/sweep_arm_d_loss.py \
    --config "$DIRECT_CFG" --arm "$SWEEP_ARM" --stage "$SWEEP_STAGE" \
    --epochs "$SWEEP_EPOCHS" --val-every "$SWEEP_VAL_EVERY" \
    --lrs $SWEEP_LRS --plan-index "$IDX" \
    ${SWEEP_FOLDS:+--folds ${SWEEP_FOLDS//,/ }} \
    --device "${DEVICE:-cuda}"
