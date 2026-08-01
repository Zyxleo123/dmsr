#!/bin/bash
# Rung 3 (CPU): pick the fixed massive-host chunks the overfit run trains on.
#
# Reads cached HR chunk summaries only -- no GPU, no halo finding.
#
#   sbatch scripts/slurm/select_fixed_hosts_cpu.sbatch OVERFIT_BOXES=set0
#
#SBATCH --job-name=rw_pick_hosts
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_pick_hosts-%j.out

set -euo pipefail

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

set -euo pipefail

: "${OVERFIT_BOXES:=set0}"      # TRAIN boxes only; the script refuses otherwise
: "${N_HOST_CHUNKS:=4}"
: "${FIXED_HOSTS_JSON:=$DMSR_REWARD_ROOT/overfit/fixed_hosts.json}"

require_input "$DMSR_REWARD_ROOT/catalog_cache" "scripts/slurm/cache_catalogs_cpu.sbatch"

python -u scripts/reward/select_fixed_hosts.py \
    --config "$REWARD_CFG" \
    --boxes "$OVERFIT_BOXES" \
    --n-chunks "$N_HOST_CHUNKS" \
    --out "$FIXED_HOSTS_JSON"

echo "=== fixed hosts -> $FIXED_HOSTS_JSON"
