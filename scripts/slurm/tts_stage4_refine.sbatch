#!/bin/bash
# Stage 4: verifier-guided noise refinement (gradient arm and CEM control arm).
#
#   sbatch scripts/slurm/tts_stage4_refine.sbatch            # gradient
#   sbatch --export=ARM=cem scripts/slurm/tts_stage4_refine.sbatch
#
# The CEM arm is not optional decoration: if gradient-free search matches
# gradient refinement then the gain came from search over the noise
# distribution, not from exploiting the verifier's gradients. Run both.
#
#SBATCH --job-name=tts_refine
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=gpu28
#SBATCH --output=slurm-tts_refine-%j.out

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

: "${ARM:=grad}"
: "${KEEP:=4}"
if [[ "$ARM" == "cem" ]]; then
    EXTRA=(--cem)
    STAGE_OUT="$OUT/stage4_cem"
else
    EXTRA=()
    STAGE_OUT="$OUT/stage4"
fi
echo "    refinement arm: $ARM -> $STAGE_OUT"

python -u scripts/tts_stage45.py --mode refine "${EXTRA[@]}" \
    --lr "$DATA/lr" --hr "$DATA/hr" --model "$MODEL" \
    --verifier "$OUT/verifier" --hr-reference "$OUT/oracle/hr_reference.npz" \
    --boxes $TEST_BOXES --k "$K" --keep "$KEEP" \
    --device "$DEVICE" --out "$STAGE_OUT"

echo "=== stage 4 ($ARM) done: $STAGE_OUT"
