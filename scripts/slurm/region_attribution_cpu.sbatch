#!/bin/bash
# At which spatial scale is the catalog-reward label rankable? Tiles -> box.
#
# The region-scale successor to attribution_diagnostic_cpu.sbatch. Reads the
# SAVED tile_weights.npz and catalogs, re-derives per-tile summaries under both
# attribution schemes, pools them into regions of width 1/2/4/8 over every valid
# periodic partition offset, and measures baseline-context stability of the
# within-region candidate ordering. NO halo-finder run: one pass over the ~28
# saved ASCII catalogs, minutes not hours.
#
# CPU and submitted rather than inline: parsing ~1.9 GB of ASCII catalogs is real
# work, and the login node kills real work.
#
#   sbatch scripts/slurm/region_attribution_cpu.sbatch <envfile> BOXES="set0 set1"
#SBATCH --job-name=sd_region
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_region-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set0 set1}"
: "${WIDTHS:=1 2 4 8}"
: "${TOUCHED_THRESHOLD:=0.01}"
: "${ELIGIBLE_THRESHOLD:=0.65}"
# With only the two development boxes the report stays EXPLORATORY and emits no
# eligibility verdict. Raise MIN_SCREEN_BOXES' worth of boxes (set0,1,8,9) for a
# screen. Default 4 matches the predeclared pilot set.
: "${MIN_SCREEN_BOXES:=4}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"

python -u scripts/reward/region_attribution_diagnostic.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" \
    --boxes ${BOXES//,/ } --widths ${WIDTHS//,/ } \
    --touched-threshold "$TOUCHED_THRESHOLD" \
    --eligible-threshold "$ELIGIBLE_THRESHOLD" \
    --min-screen-boxes "$MIN_SCREEN_BOXES"

echo "=== report -> $DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/region_attribution_diagnostic.json"
echo "=== read recommendation.evidence_status and recommendation.action:"
echo "===   exploratory_insufficient_boxes -> EXPLORATORY only (set0,set1). Run the"
echo "===       held-out pilot (set8,set9) and rerun on set0,set1,set8,set9."
echo "===   screen -> smallest_eligible_width names the region scale to carry into"
echo "===       the proxy benchmark. The trained held-out gate remains decisive."
