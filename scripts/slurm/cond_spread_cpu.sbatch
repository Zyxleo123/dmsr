#!/bin/bash
# Pilot step 2 (+3): is p(HR | SR2) broad at subhalo scale, and where does the
# residual displacement actually live in k?
#
# CPU only. It memory-maps the two (6,512,512,512) fields per box, FFTs the 512
# Lagrangian tiles, and fits a ridge from SR2's local neighbourhood to HR's
# high-pass displacement -- fitted on one box, scored on the other. No GPU, no
# halo finder, nothing trained. Everything it prints lands in
# cond_spread/cond_spread_<fit>_<heldout>.json, so any figure is a redraw.
#
#   sbatch scripts/slurm/cond_spread_cpu.sbatch <envfile>
#SBATCH --job-name=condspr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-condspr-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${CS_BOXES:=set8}"
: "${CS_SIGMAS:=0.7,1,2}"
: "${CS_RADIUS:=5}"
: "${CS_N_SITES:=480000}"
: "${CS_N_TILES:=512}"
: "${CS_K_SPLIT:=8.0}"
: "${CS_SITE_LOG_MVIR:=13.0}"
: "${CS_RFF_DIM:=1024}"
: "${CS_NL_DIM:=64}"
: "${CS_SEED:=0}"

# Gate: every box needs an HR field and a cached SR2 field. The owner array is
# NOT gated on -- it is streamed from Rockstar's raw particle dumps, which are
# deleted after extraction, so most boxes cannot get one back without re-running
# the halo finder. A box without one contributes uniform sites and a single
# pooled spectrum instead of the host stratification, and says so in its output.
# Only the FIRST box needs one, because it carries the host-site subset.
FIRST_BOX="${CS_BOXES%%,*}"
require_input "$DMSR_REWARD_ROOT/halos_particles/${FIRST_BOX}__hr__hr/${FIRST_BOX}_hr_owner.npy" \
    "scripts/reward/rockstar_particles.py --write-assignment"
for b in ${CS_BOXES//,/ }; do
    require_input "$DATA/hr/$b.npy" "scripts/reward/preprocess (paired_catnorm)"
    if ! ls "$DMSR_REWARD_ROOT/cache/sr2_base/${b}_seed0_"*.npy >/dev/null 2>&1; then
        echo ">>> MISSING INPUT: cached SR2 field for $b"
        echo ">>> produced by: sbatch scripts/slurm/cache_sr2_base.sbatch BOXES=$b"
        echo ">>> skipping (exit 0)."
        exit 0
    fi
done

python -u scripts/features/measure_conditional_spread.py \
    --boxes "$CS_BOXES" \
    ${CS_SPLIT_BUFFER:+--split-buffer "$CS_SPLIT_BUFFER"} \
    --sigmas "$CS_SIGMAS" \
    --radius "$CS_RADIUS" \
    --n-sites "$CS_N_SITES" \
    --n-tiles "$CS_N_TILES" \
    --k-split "$CS_K_SPLIT" \
    --site-log-mvir "$CS_SITE_LOG_MVIR" \
    --rff-dim "$CS_RFF_DIM" \
    --nl-dim "$CS_NL_DIM" \
    --seed "$CS_SEED"

echo "=== result: $DMSR_REWARD_ROOT/cond_spread/cond_spread_*.json"
