#!/bin/bash
# Pilot step 4: capacity or incentive? Fine-tune one SR2 rung onto a SINGLE
# 1e14 host's Lagrangian tiles under plain MSE, and watch what the squared loss
# does to the high-k power while it lowers the error.
#
# GPU, but a small one: the unit of work is one generator tile (14^3 LR in,
# 70^3 out, trimmed to 64^3), batch 2. Any `general` GPU (a5000 24 GB or a6000
# 48 GB) is enough; on an OOM resubmit with OH_BATCH=1.
#
#   sbatch scripts/slurm/overfit_host_mse_gpu.sbatch <envfile>
#SBATCH --job-name=ohmse
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-ohmse-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

# _sr2_direct_common.sh does not export DATA (the direct line reads the path
# out of reward.yaml). Mirror reward.yaml's data.root here so the gate below
# checks the same files the python will open.
: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${OH_BOX:=set8}"
: "${OH_HOST_ID:=-1}"
: "${OH_RUNG:=fine}"
: "${OH_STEPS:=3000}"
: "${OH_BATCH:=2}"
: "${OH_N_TILES:=4}"
: "${OH_LR_SCALE:=10.0}"
: "${OH_EVAL_EVERY:=100}"
: "${OH_K_SPLIT:=4.0}"
: "${OH_LABEL:=}"
: "${OH_SEED:=0}"

require_input "$DATA/hr/$OH_BOX.npy"   "scripts/reward/preprocess (paired_catnorm)"
require_input "$DATA/lr/$OH_BOX.npy"   "scripts/reward/preprocess (paired_catnorm)"
require_input "$DMSR_REWARD_ROOT/halos_particles/${OH_BOX}__hr__hr/${OH_BOX}_hr_owner.npy" \
    "scripts/reward/rockstar_particles.py"
require_input "$PROJECT/external/SRS-map2map/SRmodel/G_z0.pt" \
    "the frozen SR2 checkpoint (configs/sr2_baseline/freeze.yaml)"

python -u scripts/features/overfit_host_mse.py \
    --config "$DIRECT_CFG" \
    --box "$OH_BOX" \
    --host-id "$OH_HOST_ID" \
    --rung "$OH_RUNG" \
    --steps "$OH_STEPS" \
    --batch "$OH_BATCH" \
    --n-tiles "$OH_N_TILES" \
    --lr-scale "$OH_LR_SCALE" \
    --eval-every "$OH_EVAL_EVERY" \
    --k-split "$OH_K_SPLIT" \
    --label "$OH_LABEL" \
    --seed "$OH_SEED" \
    ${DEVICE:+--device "$DEVICE"}

echo "=== result: $DMSR_REWARD_ROOT/host_overfit/<box>_h<id>_<rung>/summary.json"
