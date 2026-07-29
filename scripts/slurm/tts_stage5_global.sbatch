#!/bin/bash
# Stage 5: globally coherent tiled inference with joint tile selection.
#
#   sbatch scripts/slurm/tts_stage5_global.sbatch
#   sbatch --export=CHUNK=16,STRIDE=8,LAM_OVERLAP=1.0 scripts/slurm/tts_stage5_global.sbatch
#
# Geometry constraints (TileGrid enforces them): STRIDE must divide the LR grid,
# CHUNK/2 <= STRIDE <= CHUNK so at most two tiles cover any point per axis, and
# CHUNK + 2*PAD <= Ng.
#
#SBATCH --job-name=tts_global
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu28
#SBATCH --output=slurm-tts_global-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_tts_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_tts_common.sh" "$@"

require_gate "$OUT/verifier/verifier_report.json" "gate.pass" \
    "no test-time-available selector beats random with a CI excluding zero while \
recovering >=50% of the statistical-oracle gain"

: "${CHUNK:=16}"
: "${STRIDE:=8}"
: "${LAM_OVERLAP:=1.0}"
: "${STITCH:=blend}"

python -u scripts/tts_stage45.py --mode global \
    --lr "$DATA/lr" --hr "$DATA/hr" --model "$MODEL" \
    --verifier "$OUT/verifier" --hr-reference "$OUT/oracle/hr_reference.npz" \
    --boxes $TEST_BOXES --k "$K" \
    --chunk "$CHUNK" --stride "$STRIDE" --lam-overlap "$LAM_OVERLAP" --stitch "$STITCH" \
    --device "$DEVICE" --out "$OUT/stage5"

echo "=== stage 5 done: $OUT/stage5"
