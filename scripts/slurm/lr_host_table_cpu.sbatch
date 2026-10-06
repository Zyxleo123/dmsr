#!/bin/bash
# The gated LR host table: (centre, R_vir, M_vir) per box, for
# --soft-host-profile lr. Parses the LR Rockstar catalog that
# build_lagrangian_host.py already left on disk -- no Rockstar run, no model,
# no GPU, seconds per box.
#
# Sizing: one ASCII catalog (~1k rows) parsed into a few small arrays. 8G is
# already an order of magnitude over what this needs; the parse, not the data,
# is the cost.
#
#   sbatch scripts/slurm/lr_host_table_cpu.sbatch <envfile>
#SBATCH --job-name=lrhosttbl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-lrhosttbl-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HP_BOXES:?HP_BOXES must be the box list}"
: "${HP_MIN_HR_NUM_P:=20000}"
: "${HP_NG_LR:=64}"
: "${HP_NG_HR:=512}"

python -u scripts/features/build_lr_host_table.py \
    --boxes $HP_BOXES \
    --min-hr-num-p "$HP_MIN_HR_NUM_P" \
    --ng-lr "$HP_NG_LR" --ng-hr "$HP_NG_HR"
echo "=== LR host tables -> $DMSR_REWARD_ROOT/lr_host_table/"
echo "=== n_gated_hosts per box is in summary.json. A box with FEW gated hosts"
echo "=== will run --soft-host-profile lr on a flat background for most of its"
echo "=== supervised hosts, which is the arm doing nothing, not the arm failing."
