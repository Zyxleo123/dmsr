#!/bin/bash
# Stage 0b -- the same ruler at the TRAINING window size.
#
# Training used 8^3 LR crops; `--dmsr-tile 16 --dmsr-halo 8` evaluates on 32^3
# LR windows. Under `nn.GroupNorm` those are not the same function: normalisation
# statistics are taken over the whole window, so the same physical region gets
# different conditioning at different window sizes (measured r = 0.474 between
# ctx-8 and full-box encoder features, 92.7% of it a per-channel affine rescale).
# The baseline's headline 0.282 therefore carries a train/eval shift on top of
# whatever the model actually does.
#
# `--dmsr-tile 8 --dmsr-halo 4` gives 16^3 windows, much closer to training.
# Stage 0 minus Stage 0b is the size of that shift. The `chnorm` arm should show
# the smallest difference of the four -- that is the specific prediction the
# normalisation change makes, and the reason this job is worth a GPU.
#
# Fields are NOT re-cached here: Stage 1 reads the tile-16 fields from Stage 0,
# and two caches of the same box would only invite mixing them up.
#
#   sbatch scripts/slurm/dfix_stage0_tile8.sbatch
#
#SBATCH --job-name=dfix_s0b
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-dfix_s0b-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_dfix_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_dfix_common.sh" "$@"

: "${TILE:=8}"
: "${HALO:=4}"
: "${LR_CROP:=64}"

require_input "$LR_NPY" "the paired_catnorm dataset"
require_input "$HR_NPY" "the paired_catnorm dataset"

for spec in $ARMS; do
  dir="${spec#*:}"
  if [ ! -f "$dir/ckpt_best.pt" ]; then
    echo ">>> arm '${spec%%:*}' has no checkpoint at $dir/ckpt_best.pt -- skipping."
    exit 0
  fi
done

# --no-srs: the SR2 ceiling does not depend on our tiling, so re-measuring it
# here would cost GPU time to reproduce a number Stage 0 already has.
python -u scripts/compare_flow_baseline.py \
    --lr "$LR_NPY" --hr "$HR_NPY" \
    "${DMSR_ARGS[@]}" \
    --lr-crop "$LR_CROP" --dmsr-tile "$TILE" --dmsr-halo "$HALO" \
    --n-steps "$N_STEPS" --diversity 1 --no-srs \
    --out "$OUT/ceiling_tile$TILE"

echo "=== stage 0b done: $OUT/ceiling_tile$TILE/comparison.json"
