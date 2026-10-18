#!/bin/bash
# Decision tables from the placement forensics (CPU, reads saved JSON only):
# rerun any time after submit_placement_forensics.sh's jobs to re-render.
#
#   sbatch scripts/slurm/placement_report_cpu.sbatch [envfile|VAR=value ...]
#
#SBATCH --job-name=placerep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-placerep-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 4G  partition cpu"
: "${BOX:=set9}"
D="$DMSR_REWARD_ROOT/placement_forensics"
require_input "$D/${BOX}_forensics.json" "scripts/slurm/submit_placement_forensics.sh (job A)"
python -u scripts/features/placement_report.py --dir "$D" --box "$BOX"
