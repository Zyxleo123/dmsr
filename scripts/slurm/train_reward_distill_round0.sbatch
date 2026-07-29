#!/bin/bash
# Stage 8 (GPU): exactly one offline reward-weighted distillation round.
#
#   L = L_sup + lambda_elite * L_elite + lambda_ref * L_ref
#
# The student starts from the prior, the teacher is frozen, lambda_elite ramps
# from zero, paired supervision runs throughout, and candidates are never
# regenerated. Abort conditions are checked at every validation and stop the run
# rather than letting a broken checkpoint reach evaluation.
#
#   sbatch scripts/slurm/train_reward_distill_round0.sbatch
#   sbatch scripts/slurm/train_reward_distill_round0.sbatch LAMBDA_ELITE=0.5 TAU=2.0
#   sbatch scripts/slurm/train_reward_distill_round0.sbatch RESUME=.../ckpt_last.pt
#
#SBATCH --job-name=rw_distill
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_distill-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

# ---- initialisation and replay ---------------------------------------------
: "${PRIOR_CKPT:=$DMSR_REWARD_ROOT/checkpoints/residual_prior/ckpt_best.pt}"
: "${TEACHER_CKPT:=$PRIOR_CKPT}"
: "${ROUND:=round_000}"
: "${REPLAY:=$DMSR_REWARD_ROOT/replay/$ROUND/replay.jsonl}"

# ---- reward weighting -------------------------------------------------------
: "${PAIRED_FRACTION:=0.5}"     # paired:elite batch ratio inside a step
: "${TAU:=1.0}"                 # reward temperature
: "${A_MAX:=5.0}"               # clip on the marginal contribution
: "${W_MAX:=10.0}"              # clip on the resulting weight
: "${LAMBDA_ELITE:=1.0}"
: "${LAMBDA_ELITE_WARMUP:=2000}"
: "${LAMBDA_REF:=0.1}"

# ---- optimisation -----------------------------------------------------------
: "${STEPS:=20000}"
: "${BATCH_SIZE:=4}"
: "${GRAD_ACCUM:=2}"
: "${LR:=3.0e-5}"
: "${CROP_HR:=64}"
: "${CKPT_EVERY:=1000}"
: "${VAL_EVERY:=1000}"
: "${NUM_WORKERS:=${SLURM_CPUS_PER_TASK:-8}}"
: "${AMP:=true}"
: "${RESUME:=}"
: "${RUN_DIR:=$DMSR_REWARD_ROOT/checkpoints/reward_distill_$ROUND}"

require_input "$PRIOR_CKPT" "scripts/slurm/train_residual_prior.sbatch"
require_input "$REPLAY" "scripts/slurm/build_replay_cpu.sbatch"

python -u scripts/reward/train_reward_distill.py \
    --config "$DISTILL_CFG" \
    --out "$RUN_DIR" \
    --init-from "$PRIOR_CKPT" \
    --teacher "$TEACHER_CKPT" \
    --replay "$REPLAY" \
    ${RESUME:+--resume "$RESUME"} \
    --set data.root="$DATA" \
    --set data.crop_hr="$CROP_HR" \
    --set data.base_seed="$BASE_SEED" \
    --set distill.paired_fraction="$PAIRED_FRACTION" \
    --set distill.tau="$TAU" \
    --set distill.a_max="$A_MAX" \
    --set distill.w_max="$W_MAX" \
    --set distill.lambda_elite="$LAMBDA_ELITE" \
    --set distill.lambda_elite_warmup_steps="$LAMBDA_ELITE_WARMUP" \
    --set distill.lambda_ref="$LAMBDA_REF" \
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

echo "=== distilled -> $RUN_DIR"
echo "=== if ABORTED.json exists, the run tripped a guard; read it before rerunning"
