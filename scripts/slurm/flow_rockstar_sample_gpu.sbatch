#!/bin/bash
# Flow -> Rockstar, stage 1 (GPU): sample one full-box 512^3 catnorm field
# from a trained flow cascade so it can be run through the frozen halo finder.
#
# Node choice (checked with `myfree`): the only free card big enough is the
# h200 (141 GB) on `gpu2`, partition `project`. A full-box 512^3 pass through
# the width-64 velocity net at fp32 peaks near ~80-90 GB, which does not fit on
# a 48 GB a6000 or a 32 GB v100 -- hence the h200. `project` only allows
# qos_project_preempt (preemptible), but the ODE sample is a few minutes and the
# output is cached/idempotent given the seed, so a preemption just means
# resubmitting this one job; nothing downstream is lost.
#
#   sbatch scripts/slurm/flow_rockstar_sample_gpu.sbatch \
#       CONFIG=configs/flow_cascade.yaml CKPT=runs/flow_cascade/ckpt_last.pt \
#       TAG=flow_cascade
#SBATCH --job-name=flow_rs_gen
#SBATCH --partition=project
#SBATCH --qos=qos_project_preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/flow_rockstar/logs/slurm-flow_rs_gen-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${CONFIG:=configs/flow_cascade.yaml}"
: "${CKPT:=runs/flow_cascade/ckpt_last.pt}"
: "${TAG:=flow_cascade}"
: "${BOX:=set15}"
: "${HR_FIELD:=$DATA/hr/${BOX}.npy}"
: "${N_STEPS:=20}"
: "${FLOW_RS_ROOT:=$DMSR_REWARD_ROOT/flow_rockstar}"
: "${FIELD_OUT:=$FLOW_RS_ROOT/fields/${BOX}__${TAG}__seed${SEED}.npy}"

mkdir -p "$(dirname "$FIELD_OUT")"
require_input "$CKPT" "the flow training run (runs/<name>/ckpt_last.pt)"
require_input "$HR_FIELD" "paired_catnorm HR field for $BOX"

echo "=== sample: config=$CONFIG ckpt=$CKPT box=$BOX tag=$TAG seed=$SEED"
python -u scripts/reward/sample_flow_field.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --hr "$HR_FIELD" \
    --n-steps "$N_STEPS" \
    --seed "$SEED" \
    --out "$FIELD_OUT"

echo "=== field -> $FIELD_OUT"
