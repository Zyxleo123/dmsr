#!/bin/bash
# Stage 2: train and evaluate the test-time-available selector.
#
#   sbatch scripts/slurm/tts_stage2_verifier.sbatch
#
# CPU only: this reads the JSONL that Stage 1 wrote and fits a ranker over ~16
# summary features. Asking for a GPU here would just queue behind GPU jobs for
# no gain.
#
# `--hidden` with no values trains a *linear* ranker. That is deliberate: Stage 4
# needs linear weights to build its differentiable refinement surrogate, and with
# only a handful of independent boxes a deeper model memorises box identity.
#
#SBATCH --job-name=tts_verifier
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-tts_verifier-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_tts_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_tts_common.sh" "$@"

require_gate "$OUT/oracle/oracle_report.json" "decision_gate.pass" \
    "best-of-$K does not improve any primary density / higher-order metric by >=5% \
with a box-bootstrap CI excluding zero -- SR2's noise carries no selection leverage"

python -u scripts/train_srs_verifier.py \
    --rows "$OUT/oracle/rows.jsonl" \
    --train-boxes $TRAIN_BOXES --val-boxes $VAL_BOXES --test-boxes $TEST_BOXES \
    --k-values $KVALS --k-gate "$K" --hidden \
    --out "$OUT/verifier"

echo "=== stage 2 done: $OUT/verifier"
