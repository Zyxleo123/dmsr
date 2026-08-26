#!/bin/bash
# Pilot step 5->6 (GPU): sample the full-box catnorm field from the trained
# substructure module. Regenerates the frozen SR2 box (cached), samples d per
# tile, applies Pi once, writes Psi_SR2 + Pi(d) to FIELD_OUT in the exact format
# scripts/reward/catalog_summaries.py --source candidate --field consumes, so the
# reused flow_rockstar catalog + compare stages run unchanged.
#
# Node choice (`myfree`): general GPU. Per-tile forward + Euler ODE is light on
# GPU memory; host RAM holds the SR2 box + assembled d (~7 GB), hence --mem=48G.
#
#   sbatch scripts/slurm/sample_substructure_gpu.sbatch \
#       SUB_CONFIG=configs/substructure_set8.yaml CKPT=<run>/ckpt_last.pt \
#       BOX=set8 TAG=substructure_set8 SEED=0 N_STEPS=20
#SBATCH --job-name=sub_sample
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sub_sample-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${SUB_CONFIG:=configs/substructure_set8.yaml}"
: "${BOX:=set8}"
: "${TAG:=substructure_set8}"
: "${SEED:=0}"
: "${N_STEPS:=20}"
: "${RUN_NAME:=substructure_set8}"
: "${CKPT:=$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/ckpt_last.pt}"
: "${FLOW_RS_ROOT:=$DMSR_REWARD_ROOT/flow_rockstar}"
: "${FIELD_OUT:=$FLOW_RS_ROOT/fields/${BOX}__${TAG}__seed${SEED}.npy}"

require_input "$CKPT" "scripts/slurm/train_substructure_gpu.sbatch"
mkdir -p "$(dirname "$FIELD_OUT")"

echo "=== sample: config=$SUB_CONFIG ckpt=$CKPT box=$BOX tag=$TAG seed=$SEED"
python -u scripts/reward/sample_substructure_field.py \
    --config "$SUB_CONFIG" \
    --checkpoint "$CKPT" \
    --box "$BOX" \
    --n-steps "$N_STEPS" \
    --seed "$SEED" \
    --out "$FIELD_OUT"

echo "=== field -> $FIELD_OUT"
