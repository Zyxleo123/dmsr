#!/bin/bash
# Choose the twelve tile splices one arm will be checked on. Seconds of CPU.
#
# Separate from running them so the plan is readable -- and reviewable -- before
# twelve full-box Rockstar runs are spent on it.
#
#   sbatch scripts/slurm/splice_select_cpu.sbatch <envfile> ARM=a
#SBATCH --job-name=sd_splsel
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_splsel-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARM:=a}"

python -u scripts/reward/splice_verify.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM" --stage select
