#!/bin/bash
# Stage 0 -- put all four sweep arms on the SAME full-box ruler.
#
# This is the only measurement that answers the question the sweep was run to
# answer. The `val_density_*` columns in metrics.csv cannot: `HighPassDensity`
# is shared between the critic and validation, so arms with
# `critic.valid_center: 32` validate on a 32^3 offset cube while the baseline
# validates on a 64^3 wrapped one. Those are different fields on different
# grids; comparing them across the base/arm boundary is meaningless.
#
# Here every arm is generated over the whole 512^3 box and scored by one
# function, against one truth, in one process. The headline is
# `density_highk_pk_ratio` (baseline 0.282; SRS ceiling 0.977) read together
# with `highk_power_ratio` (baseline 0.487) -- a density gain that costs
# displacement power is the mean-collapse failure returning, not progress.
#
# `--save-fields` caches trilinear / hr_truth / each arm / srs as .npy so that
# Stage 1 never regenerates them (a full-box draw is ~5.3 GPU-h).
#
#   sbatch scripts/slurm/dfix_stage0_ceiling.sbatch
#   sbatch --export=BOX=set15,OUT=runs/dmsr/dfix_set15 scripts/slurm/dfix_stage0_ceiling.sbatch
#
#SBATCH --job-name=dfix_s0
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-dfix_s0-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_dfix_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_dfix_common.sh" "$@"

: "${TILE:=16}"
: "${HALO:=8}"
: "${LR_CROP:=64}"

require_input "$LR_NPY" "the paired_catnorm dataset"
require_input "$HR_NPY" "the paired_catnorm dataset"

for spec in $ARMS; do
  dir="${spec#*:}"
  if [ ! -f "$dir/ckpt_best.pt" ]; then
    echo ">>> arm '${spec%%:*}' has no checkpoint at $dir/ckpt_best.pt"
    echo ">>> refusing to publish a partial ruler -- fix the arm list and resubmit."
    exit 0
  fi
done

python -u scripts/compare_flow_baseline.py \
    --lr "$LR_NPY" --hr "$HR_NPY" \
    "${DMSR_ARGS[@]}" \
    --lr-crop "$LR_CROP" --dmsr-tile "$TILE" --dmsr-halo "$HALO" \
    --n-steps "$N_STEPS" --diversity 1 \
    --save-fields "$OUT/fields_$BOX" \
    --out "$OUT/ceiling_tile$TILE"

echo "=== stage 0 done: $OUT/ceiling_tile$TILE/comparison.json"
