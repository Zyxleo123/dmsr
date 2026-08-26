#!/bin/bash
# Is the centre term a physical RULE or a per-object ADDRESS?
# (scripts/features/centre_offset_decompose.py)
#
# It decides how the member-gather objective should be trained, and it costs one
# frozen forward per host and a least-squares fit. Every other term in the loss
# is an invariant moment about the set's own centroid -- a rule, and a shared
# convolutional operator can learn a rule. `centre` is an address, the pool asks
# for 5,396 of them from 335,954 shared parameters, and frozen SR2 starts a
# median 5.59 search radii away (measured, the step-0 row of the 2026-08-23
# pool). This job measures how much of that 5.59 is a systematic infall deficit,
# which is learnable, and how much is isotropic scatter, which is not.
#
# Node choice, checked with `myfree` at write time: on `general`, gpu24 and
# gpu29 (a6000, 48 GB) and gpu28 (a5000, 24 GB, 468 GB host free) are
# AVAILABLE; `legacy` and `debug` hold the rtx_2080_ti (11 GB) and v100 pool.
# No GPU class is pinned because there is no optimisation here at all: one
# no_grad forward over four 64^3 tiles per host, which fits anywhere. HOST RAM
# is the ask, and it is 96G for the same reason the trainer's is -- build_pool
# inverts one box's 537 MB owner array into a CSR index at a time (2-3 GB peak)
# and keeps ~60 MB of tile data per host on the CPU.
#
#   sbatch scripts/slurm/centre_offset_gpu.sbatch <envfile>
#SBATCH --job-name=coffset
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-coffset-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
# The SAME pool the fine-tune trains on, or the answer is about a different
# population than the one it would apply to.
: "${CO_TRAIN_BOXES:=set3 set4 set5 set6 set7}"
: "${CO_HOLDOUT_BOXES:=set9 set10}"
: "${CO_N_TILES:=4}"
: "${CO_MAX_HOSTS_PER_BOX:=8}"
: "${CO_MIN_LOG_MVIR:=13.5}"
: "${CO_MIN_NUM_P:=200}"
: "${CO_MIN_PURITY:=0.5}"
: "${CO_MIN_LIVE_FRAC:=0.5}"
: "${CO_MAX_SETS:=256}"
# d6 is never evaluated here, and building its local background is the slowest
# part of the pool. Off.
: "${CO_BG_K:=0}"
: "${CO_LABEL:=}"
: "${CO_SEED:=0}"

require_input "$PROJECT/external/SRS-map2map/SRmodel/G_z0.pt" \
    "the frozen SR2 checkpoint (configs/sr2_baseline/freeze.yaml)"

_HAVE=0
for b in $CO_TRAIN_BOXES $CO_HOLDOUT_BOXES; do
    [ -r "$DMSR_REWARD_ROOT/halos_particles/${b}__hr__hr/${b}_hr_owner.npy" ] \
        && _HAVE=$((_HAVE + 1))
done
if [ "$_HAVE" = "0" ]; then
    echo ">>> no box in the pool has an HR owner array; there is no membership"
    echo ">>> and therefore no member set to measure an offset for."
    echo ">>> produced by: bash scripts/slurm/submit_owner_arrays.sh"
    echo ">>> exiting 0 rather than stranding any dependent job."
    exit 0
fi
echo "=== $_HAVE boxes have owner arrays"

echo "=== centre-offset decomposition"
echo "    fit on:   $CO_TRAIN_BOXES"
echo "    score on: $CO_HOLDOUT_BOXES  (the rule must generalise across clusters)"
echo "    NOTE: this bounds the CENTRE term alone. It says nothing about whether"
echo "          the internal moments are reachable, and the gate stays Rockstar."

python -u scripts/features/centre_offset_decompose.py \
    --config "$DIRECT_CFG" \
    --train-boxes "$CO_TRAIN_BOXES" \
    --holdout-boxes "$CO_HOLDOUT_BOXES" \
    --n-tiles "$CO_N_TILES" \
    --max-hosts-per-box "$CO_MAX_HOSTS_PER_BOX" \
    --min-log-mvir "$CO_MIN_LOG_MVIR" \
    --min-num-p "$CO_MIN_NUM_P" \
    --min-purity "$CO_MIN_PURITY" \
    --min-live-frac "$CO_MIN_LIVE_FRAC" \
    --max-sets "$CO_MAX_SETS" \
    --bg-k "$CO_BG_K" \
    --label "$CO_LABEL" \
    --seed "$CO_SEED" \
    ${DEVICE:+--device "$DEVICE"}

RUN_DIR="$DMSR_REWARD_ROOT/centre_offset/pool${CO_LABEL}"
echo "=== result: $RUN_DIR/offsets.json"
echo "    every set's offset vector, its radial/transverse split, the fitted"
echo "    rules and their held-out residual in gate search radii."
echo "=== re-print the tables without recomputing:"
echo "===   python scripts/features/centre_offset_decompose.py \\"
echo "===       --from-json $RUN_DIR/offsets.json"
