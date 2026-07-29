#!/bin/bash
# Stage 1: oracle best-of-K audit for the pretrained SR2 generator.
#
#   sbatch scripts/slurm/tts_stage1_oracle.sbatch
#   sbatch --export=K=32,OUT=runs/tts_k32 scripts/slurm/tts_stage1_oracle.sbatch
#
# Cost: n_boxes * K full-box generations. ~40 s per 512^3 candidate on an A100,
# so 16 boxes x K=16 is roughly 3 h of generation plus metric time. The generate
# phase appends to rows.jsonl and resumes, so a timed-out job can simply be
# resubmitted and will pick up where it stopped.
#
#SBATCH --job-name=tts_oracle
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=gpu28
#SBATCH --output=slurm-tts_oracle-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_tts_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_tts_common.sh" "$@"

python -u scripts/eval_srs_tts.py --stage all \
    --lr "$DATA/lr" --hr "$DATA/hr" --model "$MODEL" \
    --boxes $ALL_BOXES --train-boxes $TRAIN_BOXES \
    --val-boxes $VAL_BOXES --test-boxes $TEST_BOXES \
    --seeds $SEEDS --k-values $KVALS \
    --device "$DEVICE" --equivariance \
    --out "$OUT/oracle"

echo "=== stage 1 done: $OUT/oracle"
