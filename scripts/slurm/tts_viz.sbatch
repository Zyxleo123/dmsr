#!/bin/bash
# Visualization: every figure that can be drawn from saved artefacts.
#
#   sbatch scripts/slurm/tts_viz.sbatch
#
# CPU and cheap, because it regenerates nothing -- it reads rows.jsonl,
# profiles.npz, box_summary.json, oracle_report.json and final_table.json. Redraw
# as often as you like; the expensive part already happened in Stage 1.
#
# For density slices of the oracle-selected vs rejected candidate (which do need
# the generator) submit scripts/slurm/tts_viz_slices.sbatch instead.
#
#SBATCH --job-name=tts_viz
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-tts_viz-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_tts_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_tts_common.sh" "$@"

python -u scripts/tts_viz.py \
    --oracle-dir "$OUT/oracle" --final-dir "$OUT/final" \
    --val-boxes $VAL_BOXES \
    --out "$OUT/figures"

echo "=== figures done: $OUT/figures"
