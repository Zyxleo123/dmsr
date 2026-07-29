#!/bin/bash
# Stage 6 -- best-of-K test-time scaling, as an AUXILIARY baseline only.
#
# It answers "is the missing density inside the sampler's support at all?" If a
# selector that never touches test HR can pull density up, the generator can
# already produce collapsed structure and the training signal is failing to
# prefer it. If even the oracle selector cannot, the structure is not in the
# support and no amount of sampling will find it.
#
# The spec's K=32 full-box draws would be ~170 GPU-hours per box. Scoring a 64^3
# region with the Stage-2-validated 64-cell buffer needs only a 24^3 LR window
# per candidate, which is what this script does -- same question, ~1% of the
# cost.
#
# `--train-hr` is a TRAINING box: it supplies the HR plausibility reference the
# non-oracle selectors use. `--hr` (test) is read for the oracle row only, which
# exists to bound what selection could achieve and is never used to choose.
#
#   sbatch scripts/slurm/dfix_stage6_bestofk.sbatch
#   sbatch --export=K=16 scripts/slurm/dfix_stage6_bestofk.sbatch
#
#SBATCH --job-name=dfix_s6
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-dfix_s6-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_dfix_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_dfix_common.sh" "$@"

: "${K:=32}"
: "${REGIONS:=4}"
: "${TRAIN_BOX:=set0}"

DIR="$(probe_dir "$PROBE_ARM")" || { echo ">>> PROBE_ARM=$PROBE_ARM not in ARMS"; exit 0; }
require_ckpt "$DIR" "$PROBE_ARM"
require_input "$LR_NPY" "the paired_catnorm dataset"
require_input "$DATA/hr/$TRAIN_BOX.npy" "the paired_catnorm dataset"

python -u scripts/dmsr_best_of_k.py \
    --config "$DIR/config.yaml" --ckpt "$DIR/ckpt_best.pt" \
    --lr "$LR_NPY" --hr "$HR_NPY" --train-hr "$DATA/hr/$TRAIN_BOX.npy" \
    --K "$K" --regions "$REGIONS" --n-steps "$N_STEPS" --device "$DEVICE" \
    --out "$OUT/stage6_bok_$PROBE_ARM"

echo "=== stage 6 done: $OUT/stage6_bok_$PROBE_ARM"
