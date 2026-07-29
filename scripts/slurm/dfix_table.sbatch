#!/bin/bash
# Aggregation -- table and figure from saved artefacts only (CPU).
#
# Runs last and reads whatever the earlier stages produced. It never touches a
# model or a 512^3 field, so it can be resubmitted on its own as often as the
# numbers need re-reading, and it costs a CPU slot rather than queueing behind
# real work for a GPU it would not use.
#
# It also enforces the one distinction that is easy to get wrong: the density
# columns of metrics.csv are NOT comparable between the baseline and the
# valid-center arms, because HighPassDensity is shared by the critic and
# validation and those arms therefore score a different field on a different
# grid. Only the Stage 0 full-box table compares all four on one ruler.
#
#   sbatch scripts/slurm/dfix_table.sbatch
#
#SBATCH --job-name=dfix_table
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-dfix_table-%j.out

set -euo pipefail

# Slurm copies this script to /var/spool/slurm/job<id>/slurm_script before
# running it, so ${BASH_SOURCE[0]} does NOT point into the repo -- resolving
# the preamble that way silently sources nothing (job 22242). Use the submit
# directory, falling back to the checkout's own absolute path.
_SLURMDIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_SLURMDIR/_dfix_common.sh" ]] \
    || _SLURMDIR=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_SLURMDIR/_dfix_common.sh" "$@"

# Deliberately NOT gated on Stage 0: a partial table that says which stages are
# missing is more useful than no table, and every "not run" line is printed
# explicitly rather than silently omitted.
python -u scripts/dmsr_fix_table.py --out "$OUT" --box "$BOX" --arms $ARMS

echo "=== table done: $OUT/summary.md"
