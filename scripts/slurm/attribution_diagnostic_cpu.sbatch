#!/bin/bash
# Is the per-tile label repeatable enough to rank on? Fractional vs majority.
#
# Reads the SAVED tile_weights.npz and catalogs and re-derives the summaries
# under both attribution schemes, so it costs one pass over ~27 ASCII catalogs
# and NO halo-finder run. Minutes, not hours.
#
# CPU and submitted rather than run inline: parsing ~3 GB of ASCII catalogs is
# real work, and the login node kills real work.
#
#   sbatch scripts/slurm/attribution_diagnostic_cpu.sbatch <envfile> BOXES="set0 set1"
#SBATCH --job-name=sd_attrib
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_attrib-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set0 set1}"
: "${TOUCHED_THRESHOLD:=0.01}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"

python -u scripts/reward/attribution_diagnostic.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" \
    --boxes ${BOXES//,/ } --touched-threshold "$TOUCHED_THRESHOLD"

echo "=== the decision is in recommendation.action of"
echo "===   $DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/attribution_diagnostic.json"
echo "=== proceed_per_tile        -> refit with recommendation.best_scheme"
echo "=== switch_to_pooled_target -> per-tile ranking is not learnable on these"
echo "===                            labels; change the target, not the features."
