#!/bin/bash
# Channel-swap report (CPU): read the four catalogs, write the verdict.
#
# Pure aggregation -- no fields, no halo finding, no GPU. It only opens catalogs
# that already exist, so the table can be re-rendered after a definition change
# without paying for Rockstar again. That is why it is its own job rather than a
# tail on the array task.
#
# It runs even when an arm is missing: the report prints a GATE line for that arm
# and the verdict says INCONCLUSIVE, which is more useful than a job that never
# started. Only the frozen controls are a hard requirement.
#
#   sbatch scripts/slurm/channel_swap_report_cpu.sbatch
#   sbatch scripts/slurm/channel_swap_report_cpu.sbatch BOX=set9
#
#SBATCH --job-name=chanswap_rep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-chanswap_rep-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOX:=set8}"

# The controls are the whole reference frame: without them there is no SR2->HR
# line to place an arm on, and the recovery fraction is undefined.
require_input "$DMSR_REWARD_ROOT/halos/${BOX}__hr__hr" \
              "scripts/slurm/exp0_rockstar_particles_cpu.sbatch (frozen HR catalog)"
require_input "$DMSR_REWARD_ROOT/halos/${BOX}__base__base" \
              "scripts/slurm/exp0_rockstar_particles_cpu.sbatch (frozen SR2 catalog)"

echo "=== channel-swap report for $BOX"
python -u scripts/reward/report_channel_swap.py --config "$REWARD_CFG" --box "$BOX"

echo "=== report written to $DMSR_REWARD_ROOT/channel_swap/${BOX}_channel_swap.md"
