#!/bin/bash
# Bump cutouts + angular-shuffle null (CPU, catalogs + fields): for each field in
# BC_FIELDS ("name|field.npy|peaks.npz|rockstar.ascii", space-separated, quoted in
# the env file) write <name>_bumps.npz and update bump_null.json, in sequence
# (the null json is shared, so the fields must not run concurrently).
#
#   sbatch scripts/slurm/bump_cutouts_cpu.sbatch <envfile>
#
#SBATCH --job-name=bumpcut
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-bumpcut-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 64G  partition cpu"
: "${BOX:=set9}"
: "${BC_FIELDS:?BC_FIELDS (name|field|peaks|catalog ...) is required}"
: "${BC_OUT_DIR:=$DMSR_REWARD_ROOT/placement_forensics/bumps}"
: "${BC_REGION_EXPORT:=$DMSR_REWARD_ROOT/member_gather/all_blocks_sm_coarse_soft/holdout_${BOX}/export.json}"
HR_CAT=$(ls "$DMSR_REWARD_ROOT"/halos/"${BOX}"__hr__hr/*_rockstar/halos_*.ascii 2>/dev/null | head -1 || true)
require_input "${HR_CAT:-/nonexistent}" "the frozen HR Rockstar catalog"
mkdir -p "$BC_OUT_DIR"
read -r -a _S <<< "$BC_FIELDS"
for s in "${_S[@]}"; do
    IFS='|' read -r NAME FIELD PEAKS CAT <<< "$s"
    require_input "$FIELD" "a gate's splice stage / paired_catnorm"
    require_input "$PEAKS" "host_phase_profile.py --peaks --peak-min-ratio 3"
    echo "=== $NAME"
    python -u scripts/features/bump_cutouts.py --field "$FIELD" --name "$NAME" \
        --peaks "$PEAKS" --catalog "$CAT" --hr-catalog "$HR_CAT" \
        --region-export "$BC_REGION_EXPORT" --out-dir "$BC_OUT_DIR"
done
echo "=== wrote $BC_OUT_DIR"
