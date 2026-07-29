#!/bin/bash
# Stage 6: the final comparison table across every method.
#
#   sbatch scripts/slurm/tts_stage6_table.sbatch
#
# Reads whatever earlier stages produced -- missing refinement or global-tiling
# rows simply drop out of the table rather than failing the job, so this is
# still worth submitting after a run that stopped at an earlier gate.
#
#SBATCH --job-name=tts_table
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-tts_table-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_tts_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_tts_common.sh" "$@"

python -u scripts/tts_final_table.py \
    --rows "$OUT/oracle/rows.jsonl" --verifier "$OUT/verifier" \
    --refine-rows "$OUT/stage4/rows.jsonl" --global-rows "$OUT/stage5/rows.jsonl" \
    --profiles "$OUT/oracle/profiles.npz" --box-summary "$OUT/oracle/box_summary.json" \
    --val-boxes $VAL_BOXES --test-boxes $TEST_BOXES \
    --k-values $KVALS --k-main "$K" \
    --out "$OUT/final"

echo "=== stage 6 done: $OUT/final"
