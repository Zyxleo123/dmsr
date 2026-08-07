#!/bin/bash
# Phase 2.2 (GPU): reward-only training of the Gaussian residual policy.
#
#   L_G = -sum_i w_i log pi_theta(a_i|B_i,Y_i)/sum_i w_i
#         + beta KL(pi_theta || pi_ref) + lambda_edit C_edit
#
# No HR reconstruction term and no paired residual is opened. The actions are
# detached replay actions, so this is reward-weighted maximum likelihood, not a
# policy gradient through a surrogate of Rockstar.
#
# BEHAVIOR_CKPT must be the checkpoint that PRODUCED the replay round. The script
# verifies it by parameter hash and refuses to run on a mismatch: reconstructing
# a_i under the wrong parameters would reinforce a different action than the one
# that earned w_i. For round 0 the behaviour policy is the untrained reference,
# so leaving it empty is correct.
#
#   sbatch scripts/slurm/gaussian_train_gpu.sbatch ROUND=round_000 ORACLE_RUN=gauss_ref_k16
#
#SBATCH --job-name=rw_gauss_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_gauss_train-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${GAUSS_CFG:=configs/reward/gaussian_policy.yaml}"
: "${ROUND:=round_000}"
: "${ORACLE_RUN:=gauss_ref_k16}"
: "${RUN_NAME:=gaussian_r0}"
: "${BEHAVIOR_CKPT:=}"         # empty = untrained reference (correct for round 0)
: "${INIT_CKPT:=}"
: "${STEPS:=}"
: "${BATCH_SIZE:=}"
: "${LR:=}"
: "${BETA_KL:=}"
: "${LAMBDA_EDIT:=}"
: "${SIGMA_FLOOR:=}"
: "${CROP_HR:=}"
: "${CONTEXT_MARGIN:=}"
: "${ALLOW_VAL:=1}"

require_input "$DMSR_REWARD_ROOT/replay/$ROUND/replay.jsonl" \
    "scripts/slurm/gaussian_replay_cpu.sbatch"
require_input "$DMSR_REWARD_ROOT/oracle/$ORACLE_RUN/actions" \
    "scripts/slurm/gaussian_sample_gpu.sbatch"

# Reclaim the reserved-but-unallocated fragmentation that helped OOM jobs
# 25570/25586 (3.6 of 21.4 GB on a 24 GB A5000). Set here rather than in the
# shared preamble: it is a GPU-only knob.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -u scripts/reward/train_gaussian_reward.py \
    --config "$REWARD_CFG" --model-config "$GAUSS_CFG" \
    --round "$ROUND" \
    --oracle-run "$ORACLE_RUN" \
    --run-name "$RUN_NAME" \
    ${BEHAVIOR_CKPT:+--behavior-checkpoint "$BEHAVIOR_CKPT"} \
    ${INIT_CKPT:+--init-checkpoint "$INIT_CKPT"} \
    ${STEPS:+--steps "$STEPS"} \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
    ${LR:+--lr "$LR"} \
    ${BETA_KL:+--beta-kl "$BETA_KL"} \
    ${LAMBDA_EDIT:+--lambda-edit "$LAMBDA_EDIT"} \
    ${SIGMA_FLOOR:+--sigma-floor "$SIGMA_FLOOR"} \
    ${CROP_HR:+--crop-hr "$CROP_HR"} \
    ${CONTEXT_MARGIN:+--context-margin "$CONTEXT_MARGIN"} \
    $([ "$ALLOW_VAL" = "1" ] && echo --allow-val) \
    --device "$DEVICE"

echo "=== checkpoint -> $DMSR_REWARD_ROOT/checkpoints/$RUN_NAME/ckpt.pt"
echo "=== next round: sample from it, re-score, re-gate --"
echo "===   sbatch scripts/slurm/gaussian_sample_gpu.sbatch RUN_NAME=${RUN_NAME}_k16 \\"
echo "===     CKPT=$DMSR_REWARD_ROOT/checkpoints/$RUN_NAME/ckpt.pt"
