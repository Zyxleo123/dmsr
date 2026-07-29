#!/bin/bash
# Stage 3 -- context-size oracle, full ODE path.
#
# Generates the SAME central 8^3 LR region from progressively larger surrounding
# windows, with everything else held fixed: same checkpoint, same ODE steps, and
# position-hashed noise so a voxel draws byte-identical noise at every context
# size (verified: overlapping windows agree exactly, and across the periodic
# wrap; mean 0.00 / std 1.00). Any change in the output is therefore context and
# nothing else.
#
# `--report-gn` additionally reports how much of the difference is a per-channel
# affine rescale, i.e. GroupNorm statistics rather than genuinely missing
# information. At the encoder that term was 92.7%.
#
# `--contexts` starts at 8: the encoder's dilation-4 circular padding rejects LR
# windows smaller than 5 ("Padding value causes wrapping around more than once"),
# which is itself a statement about how little a small crop can see.
#
#   sbatch scripts/slurm/dfix_stage3_context.sbatch
#   sbatch --export=PROBE_ARM=chnorm scripts/slurm/dfix_stage3_context.sbatch
#
#SBATCH --job-name=dfix_s3
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-dfix_s3-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_dfix_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_dfix_common.sh" "$@"

: "${CONTEXTS:=8 12 16 24 32}"
CONTEXTS="${CONTEXTS//,/ }"

DIR="$(probe_dir "$PROBE_ARM")" || { echo ">>> PROBE_ARM=$PROBE_ARM not in ARMS"; exit 0; }
require_ckpt "$DIR" "$PROBE_ARM"
require_input "$LR_NPY" "the paired_catnorm dataset"

python -u scripts/dmsr_context_oracle.py \
    --config "$DIR/config.yaml" --ckpt "$DIR/ckpt_best.pt" \
    --lr "$LR_NPY" --hr "$HR_NPY" \
    --contexts $CONTEXTS --mode generate --n-steps "$N_STEPS" \
    --report-gn --device "$DEVICE" \
    --out "$OUT/stage3_context_$PROBE_ARM"

echo "=== stage 3 done: $OUT/stage3_context_$PROBE_ARM"
