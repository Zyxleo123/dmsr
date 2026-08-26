#!/bin/bash
# Flow -> Rockstar, stage 3 (CPU): compare the flow catalogs against the frozen
# HR-truth and base catalogs -- count table + host/subhalo mass-function overlay.
# Reads only cached catalog_cache artifacts, so it is cheap to rerun/redraw.
#
#   sbatch scripts/slurm/flow_rockstar_compare_cpu.sbatch \
#       CANDIDATES="flow_cascade flow_unet_cascade"
#SBATCH --job-name=flow_rs_cmp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/flow_rockstar/logs/slurm-flow_rs_cmp-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOX:=set15}"
: "${CANDIDATES:=flow_cascade flow_unet_cascade}"
: "${FLOW_RS_ROOT:=$DMSR_REWARD_ROOT/flow_rockstar}"

CAND_ARGS=()
for t in ${CANDIDATES//,/ }; do CAND_ARGS+=(--candidate "$t"); done

echo "=== compare: box=$BOX candidates=$CANDIDATES"
python -u scripts/reward/compare_flow_catalogs.py \
    --cache "$DMSR_REWARD_ROOT/catalog_cache" \
    --box "$BOX" \
    "${CAND_ARGS[@]}" \
    --out "$FLOW_RS_ROOT/$BOX"

echo "=== comparison -> $FLOW_RS_ROOT/$BOX"
