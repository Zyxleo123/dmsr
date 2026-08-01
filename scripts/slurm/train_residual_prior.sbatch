#!/bin/bash
# Stage 3 (GPU): supervised residual diffusion prior, L_sup only.
#
# No catalog reward here by design. If the prior cannot reproduce the paired
# residual distribution, nothing downstream can work, and mixing in a reward
# would make that impossible to diagnose.
#
#   sbatch scripts/slurm/train_residual_prior.sbatch
#   sbatch scripts/slurm/train_residual_prior.sbatch STEPS=100000 SEED=1
#   sbatch scripts/slurm/train_residual_prior.sbatch RESUME=.../ckpt_last.pt
#
#SBATCH --job-name=rw_prior
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_prior-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

# ---- dataset split (train/val box lists live in configs/reward/reward.yaml) --
: "${TRAIN_BOXES:=set0,set1,set2,set3,set4,set5,set6,set7}"
: "${VAL_BOXES:=set8,set9,set10,set11}"

# ---- frozen SR2 -------------------------------------------------------------
: "${FROZEN_SR2:=external/SRS-map2map/SRmodel/G_z0.pt}"   # recorded in freeze.yaml
: "${BASE_CACHE:=$DMSR_REWARD_ROOT/cache/sr2_base}"

# ---- optimisation -----------------------------------------------------------
: "${STEPS:=60000}"
# CROP_HR is the SCORED CORE and CONTEXT_MARGIN the real neighbourhood around
# it, so the model is trained on the same 160^3 window full-box tiling gives it
# (TILE_MARGIN=48 in sample_reward_oracle.sbatch). Batch 1 x accum 8 keeps the
# effective batch at 8 against the 15.6x larger input. See the note in
# configs/reward/residual_prior.yaml before changing either.
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=8}"
: "${LR:=1.0e-4}"
: "${CROP_HR:=64}"
: "${CONTEXT_MARGIN:=48}"
: "${CKPT_EVERY:=2000}"
: "${VAL_EVERY:=1000}"
: "${NUM_WORKERS:=${SLURM_CPUS_PER_TASK:-8}}"
: "${AMP:=true}"
: "${RESUME:=}"
: "${RUN_DIR:=$DMSR_REWARD_ROOT/checkpoints/residual_prior}"
: "${SIGMA_JSON:=$DMSR_REWARD_ROOT/audits/residual_target_audit/sigma_res.json}"

require_input "$BASE_CACHE" "scripts/slurm/cache_sr2_base.sbatch"
require_input "$DATA/hr" "scripts/preprocess_paired.py"

SIGMA_ARG=()
if [[ -r "$SIGMA_JSON" ]]; then
    SIGMA_ARG=(--sigma-res "$SIGMA_JSON")
else
    echo ">>> no $SIGMA_JSON: sigma_res will be estimated from the first batches."
    echo ">>> run scripts/reward/audit_residual_targets.py first for a stable value."
fi

python -u scripts/reward/train_residual_prior.py \
    --config "$MODEL_CFG" \
    --out "$RUN_DIR" \
    "${SIGMA_ARG[@]}" \
    ${RESUME:+--resume "$RESUME"} \
    --set split.train_boxes="[${TRAIN_BOXES//,/, }]" \
    --set split.val_boxes="[${VAL_BOXES//,/, }]" \
    --set data.root="$DATA" \
    --set data.crop_hr="$CROP_HR" \
    --set data.context_margin="$CONTEXT_MARGIN" \
    --set data.base_seed="$BASE_SEED" \
    --set train.steps="$STEPS" \
    --set train.batch_size="$BATCH_SIZE" \
    --set train.grad_accum="$GRAD_ACCUM" \
    --set train.lr="$LR" \
    --set train.seed="$SEED" \
    --set train.ckpt_every="$CKPT_EVERY" \
    --set train.val_every="$VAL_EVERY" \
    --set train.num_workers="$NUM_WORKERS" \
    --set train.amp="$AMP" \
    --set train.device="$DEVICE"

echo "=== prior -> $RUN_DIR (ckpt_best.pt, ckpt_last.pt, metrics.csv)"

# A PROVISIONAL Gate A from the training metrics alone. Criterion 6 (tile-size
# independence) cannot be decided here -- this job never tiles -- so it comes out
# "not_evaluated" and the verdict is necessarily passed: false. That is not a
# result about the model; it is the missing job saying so.
#
# The REAL verdict is written by gate_a_check_cpu.sbatch, after
# gate_a_tile_scan_gpu.sbatch has produced tile_scan.json. Both are chained
# behind this job by scripts/slurm/submit_oracle.sh gate_a. Writing the
# provisional file to its own name keeps it from being mistaken for the verdict
# Gate B reads.
python -u scripts/reward/gate_a_check.py \
    --config "$REWARD_CFG" \
    --run-dir "$RUN_DIR" \
    --out "$RUN_DIR/gate_a_provisional.json" \
    ${SIGMA_JSON:+--audit "$DMSR_REWARD_ROOT/audits/residual_target_audit/residual_target_audit.json"} \
    || echo ">>> gate_a_check.py itself errored; treat Gate A as UNDECIDED"

echo "=== provisional Gate A (no tile scan) -> $RUN_DIR/gate_a_provisional.json"
echo "=== the DECIDING verdict needs two more jobs:"
echo "===   sbatch scripts/slurm/gate_a_tile_scan_gpu.sbatch"
echo "===   sbatch scripts/slurm/gate_a_check_cpu.sbatch"
echo "=== (submit_oracle.sh gate_a chains both automatically)"
