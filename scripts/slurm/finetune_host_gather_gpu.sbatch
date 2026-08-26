#!/bin/bash
# Fine-tune SR2 with the auxiliary loss that gathers particles into TRUE HR
# subhalos (scripts/features/finetune_host_gather.py). The sequel to
# overfit_host_mse_gpu.sbatch: same box, same cluster, same rungs, same field
# report -- different objective.
#
# Node choice, checked with `myfree` at write time: `general` has gpu28
# (a5000, 24 GB, 4/4 free, 48 CPUs, 500 GB) AVAILABLE while every a6000 in that
# partition is fully allocated. The unit of work is one generator tile (14^3 LR
# in, 70^3 out, trimmed to 64^3) at batch 2 plus a 32^3 CIC deposit and the
# subhalo windows, which is well inside 24 GB -- the GPU class is not load-
# bearing here, so the job is not pinned to a node. On an OOM resubmit with
# HG_BATCH=1.
#
# Host RAM is the real ask: the HR box is mmap'd but the 512^3 owner array is
# loaded whole and inverted into a CSR index (~2-3 GB peak), which is why this
# asks for 64 GB rather than the GPU job's usual 16.
#
# Checkpoints are written at every eval step, so a time-limit kill costs the
# remaining steps and not the run.
#
#   sbatch scripts/slurm/finetune_host_gather_gpu.sbatch <envfile>
#SBATCH --job-name=hgather
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hgather-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

# The direct line reads its data root out of reward.yaml and the preamble does
# not export DATA; mirror reward.yaml's data.root so the gates below check the
# same files the python opens.
: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${HG_BOX:=set8}"
: "${HG_HOST_ID:=-1}"
: "${HG_RUNG:=fine}"
: "${HG_STEPS:=3000}"
: "${HG_BATCH:=2}"
: "${HG_N_TILES:=4}"
: "${HG_LR_SCALE:=10.0}"
: "${HG_EVAL_EVERY:=100}"
: "${HG_PNG_EVERY:=500}"
: "${HG_SLAB:=4}"
: "${HG_W_GATHER:=1.0}"
: "${HG_W_CONTRAST:=1.0}"
: "${HG_W_VDISP:=1.0}"
: "${HG_W_VBULK:=1.0}"
: "${HG_W_PRESERVE:=1.0}"
: "${HG_W_LOW:=1.0}"
: "${HG_W_ANCHOR:=0.1}"
: "${HG_W_MSE:=0.0}"
: "${HG_SIGMA_FLOOR:=1.0}"
: "${HG_MIN_NUM_P:=50}"
: "${HG_MIN_PURITY:=0.5}"
: "${HG_MIN_HR_COMPACT:=5.0}"
: "${HG_K_SPLIT:=4.0}"
: "${HG_TARGETS_ONLY:=0}"
: "${HG_LABEL:=}"
: "${HG_SEED:=0}"

require_input "$DATA/hr/$HG_BOX.npy"   "scripts/reward/preprocess (paired_catnorm)"
require_input "$DATA/lr/$HG_BOX.npy"   "scripts/reward/preprocess (paired_catnorm)"
require_input "$DMSR_REWARD_ROOT/halos_particles/${HG_BOX}__hr__hr/${HG_BOX}_hr_owner.npy" \
    "scripts/reward/rockstar_particles.py -- the loss needs HR membership"
require_input "$PROJECT/external/SRS-map2map/SRmodel/G_z0.pt" \
    "the frozen SR2 checkpoint (configs/sr2_baseline/freeze.yaml)"

EXTRA=()
[[ "$HG_TARGETS_ONLY" == "1" ]] && EXTRA+=(--targets-only)

echo "=== gather fine-tune: box=$HG_BOX rung=$HG_RUNG steps=$HG_STEPS"
echo "    weights: gather=$HG_W_GATHER (contrast $HG_W_CONTRAST) low=$HG_W_LOW"
echo "             preserve=$HG_W_PRESERVE anchor=$HG_W_ANCHOR mse=$HG_W_MSE"
echo "    velocity: vdisp=$HG_W_VDISP vbulk=$HG_W_VBULK  (Rockstar links in 6-D;"
echo "              the channel swap lost 65% of HR subhalos on SR2 velocities)"

python -u scripts/features/finetune_host_gather.py \
    --config "$DIRECT_CFG" \
    --box "$HG_BOX" \
    --host-id "$HG_HOST_ID" \
    --rung "$HG_RUNG" \
    --steps "$HG_STEPS" \
    --batch "$HG_BATCH" \
    --n-tiles "$HG_N_TILES" \
    --lr-scale "$HG_LR_SCALE" \
    --eval-every "$HG_EVAL_EVERY" \
    --png-every "$HG_PNG_EVERY" \
    --slab "$HG_SLAB" \
    --w-gather "$HG_W_GATHER" \
    --w-contrast "$HG_W_CONTRAST" \
    --w-vdisp "$HG_W_VDISP" \
    --w-vbulk "$HG_W_VBULK" \
    --w-preserve "$HG_W_PRESERVE" \
    --w-low "$HG_W_LOW" \
    --w-anchor "$HG_W_ANCHOR" \
    --w-mse "$HG_W_MSE" \
    --sigma-floor "$HG_SIGMA_FLOOR" \
    --min-num-p "$HG_MIN_NUM_P" \
    --min-purity "$HG_MIN_PURITY" \
    --min-hr-compact "$HG_MIN_HR_COMPACT" \
    --k-split "$HG_K_SPLIT" \
    --label "$HG_LABEL" \
    --seed "$HG_SEED" \
    "${EXTRA[@]}" \
    ${DEVICE:+--device "$DEVICE"}

echo "=== result: $DMSR_REWARD_ROOT/host_gather/<box>_h<id>_<rung>/"
echo "    summary.json  metrics.jsonl  subhalos.json  slices/*.png  eval/step*.npz"
