#!/bin/bash
# Stage 4 -- effective receptive field, with and without the norm ablation.
#
# Perturbs one LR voxel and measures where the output moves (and the reverse
# direction), giving r50/r95/r99 radii per environment (halo / filament / sheet /
# void, classified by Zel'dovich divergence).
#
# The two runs are the point. With `nn.GroupNorm` active every output voxel
# depends on window-wide statistics, so r95 should land at the window boundary
# no matter how local the convolutions are -- a receptive field that is an
# artefact of normalisation, not of architecture. `--ablate-norm channel` swaps
# in the pointwise layer and leaves the true convolutional radius. The gap
# between the two is the quantitative case for `model.norm: channel`; if there
# is no gap, that change was not worth making and the sweep result should be
# read accordingly.
#
#   sbatch scripts/slurm/dfix_stage4_rf.sbatch
#
#SBATCH --job-name=dfix_s4
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-dfix_s4-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_dfix_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_dfix_common.sh" "$@"

: "${WINDOW:=16}"
: "${RF_STEPS:=2 4 8}"
RF_STEPS="${RF_STEPS//,/ }"

DIR="$(probe_dir "$PROBE_ARM")" || { echo ">>> PROBE_ARM=$PROBE_ARM not in ARMS"; exit 0; }
require_ckpt "$DIR" "$PROBE_ARM"
require_input "$LR_NPY" "the paired_catnorm dataset"

python -u scripts/dmsr_receptive_field.py \
    --config "$DIR/config.yaml" --ckpt "$DIR/ckpt_best.pt" --lr "$LR_NPY" \
    --window "$WINDOW" --n-steps $RF_STEPS --device "$DEVICE" \
    --out "$OUT/stage4_rf_$PROBE_ARM"

python -u scripts/dmsr_receptive_field.py \
    --config "$DIR/config.yaml" --ckpt "$DIR/ckpt_best.pt" --lr "$LR_NPY" \
    --window "$WINDOW" --n-steps $RF_STEPS --device "$DEVICE" \
    --ablate-norm channel \
    --out "$OUT/stage4_rf_${PROBE_ARM}_chnorm"

echo "=== stage 4 done: $OUT/stage4_rf_$PROBE_ARM{,_chnorm}"
