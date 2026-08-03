#!/bin/bash
# Decompose the residual prior's failure (GPU): amplitude, direction, conditioning.
#
# Gate A says the composed field is destroyed. This job says WHY, by measuring
# the three things the existing diagnostics cannot separate:
#
#   1. the residual_scale sweep -- is the residual the right shape and too big?
#   2. r(k) against the TRUE residual -- is it pointing the right way at all?
#   3. the shuffled-conditioning arm -- does the model use Psi_base / y_lr?
#
# Crop-level, not full-box: the same 64^3-core / 48-cell-context geometry the
# training diagnostics use, whose density sigma ratio (0.091) predicted the
# full-box catastrophe (0.120) to within 25%. A full-box version of this would
# cost hours and answer the same question.
#
#   sbatch scripts/slurm/residual_diagnose_gpu.sbatch
#   sbatch scripts/slurm/residual_diagnose_gpu.sbatch PRIOR_CKPT=.../residual_prior_v2/ckpt_best.pt
#   sbatch scripts/slurm/residual_diagnose_gpu.sbatch SCAN_T_MAX=0.95 TAG=tmax095
#
#SBATCH --job-name=rw_res_diag
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_res_diag-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${PRIOR_CKPT:=$DMSR_REWARD_ROOT/checkpoints/residual_prior/ckpt_best.pt}"
: "${N_CROPS:=8}"
# 0 is SR2 exactly and is the control: any alpha that does not beat it is not an
# improvement, whatever it does to the constraint metrics.
: "${ALPHAS:=0,0.1,0.2,0.33,0.5,1.0}"
: "${SCAN_STEPS:=}"          # empty = the model config's own n_steps (20)
: "${SCAN_T_MAX:=}"          # empty = the config's 0.999; 0.95 removes the clip
: "${SCAN_BOXES:=}"          # empty = the val split
: "${SHUFFLE_COND:=1}"
: "${USE_EMA:=1}"            # the oracle samples from EMA; match it
: "${TAG:=}"                 # subdirectory, so two settings do not overwrite
: "${DIAG_DIR:=$DMSR_REWARD_ROOT/audits/residual_diagnose${TAG:+_$TAG}}"

require_input "$PRIOR_CKPT" "scripts/slurm/train_residual_prior.sbatch"
require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"

mkdir -p "$DIAG_DIR"

python -u scripts/reward/residual_diagnose.py \
    --config "$REWARD_CFG" --model-config "$MODEL_CFG" \
    --checkpoint "$PRIOR_CKPT" \
    ${SCAN_BOXES:+--boxes "$SCAN_BOXES"} \
    --n-crops "$N_CROPS" \
    --alphas "$ALPHAS" \
    ${SCAN_STEPS:+--n-steps "$SCAN_STEPS"} \
    ${SCAN_T_MAX:+--t-max "$SCAN_T_MAX"} \
    --seed "$SEED" --base-seed "$BASE_SEED" \
    $([ "$SHUFFLE_COND" = "1" ] && echo --shuffle-cond) \
    $([ "$USE_EMA" = "1" ] || echo --no-ema) \
    --out "$DIAG_DIR" \
    --device "$DEVICE"

echo "=== rows -> $DIAG_DIR/residual_diagnose.jsonl"
echo "=== report: sbatch scripts/slurm/residual_diagnose_report_cpu.sbatch DIAG_DIR=$DIAG_DIR"
