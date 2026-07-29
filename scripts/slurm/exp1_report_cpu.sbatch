#!/bin/bash
# Experiment 1 (CPU): recovery-vs-alpha tables, the figure, and the verdict.
#
# Reads only the JSONL rows the intervention array wrote, so the figure can be
# redrawn any number of times without touching a halo finder or a GPU.
#
#   sbatch scripts/slurm/exp1_report_cpu.sbatch
#
#SBATCH --job-name=exp1_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-exp1_report-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8 set9}"

FIRST="${BOXES%% *}"
require_input "$DMSR_REWARD_ROOT/oracle_hr/$FIRST/interventions.jsonl" \
    "scripts/slurm/exp1_intervene_cpu.sbatch"

python -u scripts/reward/oracle_hr_report.py \
    --config "$REWARD_CFG" --boxes "${BOXES// /,}"

echo "=== report + figure -> $DMSR_REWARD_ROOT/oracle_hr"
