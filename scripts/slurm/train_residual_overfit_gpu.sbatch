#!/bin/bash
# Rung 3 (GPU): overfit the residual prior on a handful of fixed massive hosts.
#
# A short, deliberately non-generalising run. If reward cannot be raised even on
# chunks the model sees repeatedly, the limit is representational -- the action
# space or the conditioning cannot express the change -- which is a different
# conclusion from rung 2's "directed search did not find it".
#
# The resulting checkpoint is a diagnostic. It goes to its own run dir and must
# not be promoted into the Gate B or distillation path.
#
#   sbatch scripts/slurm/train_residual_overfit_gpu.sbatch OVERFIT_RUN=overfit_hosts
#
#SBATCH --job-name=rw_overfit
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_overfit-%j.out

set -euo pipefail

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

set -euo pipefail

: "${OVERFIT_CFG:=configs/reward/fixed_host_overfit.yaml}"
: "${OVERFIT_RUN:=overfit_hosts}"
: "${FIXED_HOSTS_JSON:=$DMSR_REWARD_ROOT/overfit/fixed_hosts.json}"
: "${TRAIN_BOXES:=set0,set1,set2,set3,set4,set5,set6,set7}"
: "${VAL_BOXES:=set8,set9,set10,set11}"
: "${STEPS:=300}"
: "${BATCH_SIZE:=4}"
: "${GRAD_ACCUM:=2}"
: "${LR:=1.0e-4}"
: "${CROP_HR:=64}"
: "${NUM_WORKERS:=${SLURM_CPUS_PER_TASK:-8}}"
: "${AMP:=true}"
: "${RUN_DIR:=$DMSR_REWARD_ROOT/checkpoints/$OVERFIT_RUN}"
: "${SIGMA_JSON:=$DMSR_REWARD_ROOT/audits/residual_target_audit/sigma_res.json}"
: "${INIT_CKPT:=$DMSR_REWARD_ROOT/checkpoints/residual_prior/ckpt_best.pt}"

require_input "$FIXED_HOSTS_JSON" "scripts/slurm/select_fixed_hosts_cpu.sbatch"
require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"
require_input "$DATA/hr" "scripts/preprocess_paired.py"

# Starting from the trained prior rather than from scratch: the question is
# whether reward can be moved from where the prior already is, not whether a
# fresh model can memorise four chunks.
require_input "$INIT_CKPT" "scripts/slurm/train_residual_prior.sbatch"

SIGMA_ARG=()
if [[ -r "$SIGMA_JSON" ]]; then
    SIGMA_ARG=(--sigma-res "$SIGMA_JSON")
fi

python -u scripts/reward/train_residual_prior.py \
    --config "$OVERFIT_CFG" \
    --out "$RUN_DIR" \
    --fixed-hosts "$FIXED_HOSTS_JSON" \
    --resume "$INIT_CKPT" \
    "${SIGMA_ARG[@]}" \
    --set split.train_boxes="[${TRAIN_BOXES//,/, }]" \
    --set split.val_boxes="[${VAL_BOXES//,/, }]" \
    --set data.root="$DATA" \
    --set data.crop_hr="$CROP_HR" \
    --set data.base_seed="$BASE_SEED" \
    --set train.steps="$STEPS" \
    --set train.batch_size="$BATCH_SIZE" \
    --set train.grad_accum="$GRAD_ACCUM" \
    --set train.lr="$LR" \
    --set train.seed="$SEED" \
    --set train.num_workers="$NUM_WORKERS" \
    --set train.amp="$AMP" \
    --set train.device="$DEVICE"

echo "=== rung 3 overfit -> $RUN_DIR"
echo "=== this checkpoint is a DIAGNOSTIC; do not promote it into Gate B"
