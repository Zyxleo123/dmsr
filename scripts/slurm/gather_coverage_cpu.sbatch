#!/bin/bash
# How high can the R_vir ceiling go? Lagrangian coverage vs. tile count.
#
# CPU only and no halo finder: it reads the HR catalog and the owner array, does
# one grouped pass over the box's subhalos, and writes coverage_curve.json.
# That is the whole point -- 227 of 506 is a coverage number, and the coverage
# curve is available for the price of a numpy pass rather than one Rockstar run
# per rung.
#
#   sbatch scripts/slurm/gather_coverage_cpu.sbatch <envfile>
#SBATCH --job-name=gcover
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-gcover-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${GC_BOX:=set8}"
: "${GC_HOST_ID:=271800}"
: "${GC_TILE_LADDER:=1,2,4,6,8,12,16,24,32,48,64}"
: "${GC_MIN_NUM_P_LADDER:=200,100,50}"
: "${GC_MIN_P:=50}"
: "${GC_MIN_PURITY:=0.5}"
: "${GC_MIN_LIVE_FRAC:=0.5}"
: "${GC_RADIUS_FACTOR:=1.0}"
: "${GC_BG_K:=4096}"
: "${GC_LABEL:=}"

# Gates. A missing input is "nothing to do here", not a failure: a non-zero exit
# would strand any dependent job on DependencyNeverSatisfied with no reason.
OWN="$DMSR_REWARD_ROOT/halos_particles/${GC_BOX}__hr__hr/${GC_BOX}_hr_owner.npy"
if [ ! -f "$OWN" ]; then
    echo "GATE: no HR owner array at $OWN"
    echo "GATE: it comes from scripts/reward/rockstar_particles.py "\
         "--write-assignment; skipping."
    exit 0
fi
if ! compgen -G "$DMSR_REWARD_ROOT/halos/${GC_BOX}__hr__hr/*_rockstar/halos_*.ascii" \
        > /dev/null; then
    echo "GATE: no HR Rockstar catalog for $GC_BOX; skipping."
    exit 0
fi

python -u scripts/features/gather_coverage_curve.py \
    --box "$GC_BOX" --host-id "$GC_HOST_ID" \
    --tile-ladder "$GC_TILE_LADDER" \
    --min-num-p-ladder "$GC_MIN_NUM_P_LADDER" \
    --min-p "$GC_MIN_P" --min-purity "$GC_MIN_PURITY" \
    --min-live-frac "$GC_MIN_LIVE_FRAC" \
    --radius-factor "$GC_RADIUS_FACTOR" --bg-k "$GC_BG_K" \
    --label "$GC_LABEL"

RUN_DIR="$DMSR_REWARD_ROOT/free_field_gather/${GC_BOX}_h${GC_HOST_ID}${GC_LABEL}"
echo "=== result: $RUN_DIR/coverage_curve.json"
echo "=== re-print the table without recomputing:"
echo "===   python scripts/features/gather_coverage_curve.py --from-json \\"
echo "===       $RUN_DIR/coverage_curve.json"
echo "=== the calibration block is what makes the curve readable: only the"
echo "=== predictor that reproduces the MEASURED 227 at n_tiles=4 may be used."
