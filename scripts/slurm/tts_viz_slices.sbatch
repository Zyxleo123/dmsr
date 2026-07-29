#!/bin/bash
# Visualization: density slices of the oracle-best vs oracle-worst candidate.
#
#   sbatch scripts/slurm/tts_viz_slices.sbatch
#   sbatch --export=SLICE_BOX=set13 scripts/slurm/tts_viz_slices.sbatch
#
# Needs a GPU because the candidate fields are not stored (3.2 GB each) and have
# to be regenerated from their seeds -- which is exactly the point of recording
# the seed with every row. Two full-box generations per submission.
#
# This is the figure that catches a selector winning on every number while
# producing a visibly wrong field, so run it before believing a positive result.
#
#SBATCH --job-name=tts_viz_slices
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu28
#SBATCH --output=slurm-tts_viz_slices-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_tts_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_tts_common.sh" "$@"

: "${SLICE_BOX:=set14}"

python -u scripts/tts_viz.py --render-slices \
    --oracle-dir "$OUT/oracle" --final-dir "$OUT/final" \
    --val-boxes $VAL_BOXES --slice-box "$SLICE_BOX" \
    --lr "$DATA/lr" --hr "$DATA/hr" --model "$MODEL" \
    --normalizer "$OUT/oracle/normalizer.json" \
    --device "$DEVICE" --out "$OUT/figures"

echo "=== slice figures done: $OUT/figures"
