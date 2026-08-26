#!/bin/bash
# Option A's host-frame crops, plus the learnability table behind them.
#
# CPU only: it memory-maps the two (6,512,512,512) fields, gathers one crop at a
# time, runs a kNN density estimate per crop, and never loads a whole box into
# RAM. No halo finder, no GPU. Everything it prints also lands in
# <box>_host_crops.{json,npz}, so the page is a pure redraw.
#
#   sbatch scripts/slurm/host_crops_cpu.sbatch <envfile>
#SBATCH --job-name=crops
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-crops-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8}"
: "${CROP_N_HOSTS:=16}"
: "${CROP_N_CLUSTERS:=6}"
: "${CROP_PER_BIN:=2}"
: "${CROP_SCALE:=1.0}"
: "${CROP_GRID:=48}"
: "${CROP_MIN_P:=200}"
: "${CROP_SEED:=0}"

for b in ${BOXES//,/ }; do
    HRF="$DATA/hr/$b.npy"
    OWN="$DMSR_REWARD_ROOT/halos_particles/${b}__hr__hr/${b}_hr_owner.npy"
    SR2OWN="$DMSR_REWARD_ROOT/halos_particles/${b}__base__base/${b}_base_owner.npy"
    SR2F=$(ls "$DMSR_REWARD_ROOT/cache/sr2_base/${b}_seed0_"*.npy 2>/dev/null | head -1 || true)
    MISS=0
    for f in "$HRF" "$OWN" "$SR2OWN"; do
        [ -f "$f" ] || { echo "GATE: missing $f"; MISS=1; }
    done
    [ -n "$SR2F" ] || { echo "GATE: no cached SR2 field for $b -- run scripts/slurm/cache_sr2_base.sbatch BOXES=$b"; MISS=1; }
    if [ "$MISS" != "0" ]; then
        echo "GATE: skipping $b (owner arrays come from scripts/reward/rockstar_particles.py)."
        continue
    fi
    python -u scripts/features/collect_host_crops.py \
        --boxes "$b" \
        --n-hosts "$CROP_N_HOSTS" \
        --n-clusters "$CROP_N_CLUSTERS" \
        --per-bin "$CROP_PER_BIN" \
        --crop-scale "$CROP_SCALE" \
        --grid "$CROP_GRID" \
        --min-particles "$CROP_MIN_P" \
        --seed "$CROP_SEED"
done

echo "=== crops: $DMSR_REWARD_ROOT/lagrangian_host/<box>/<box>_host_crops.{json,npz}"
