#!/bin/bash
# Re-gate both arms with the splice results in place, then write the immutable
# proxy_benchmark.json and the advancement decision. Minutes of CPU.
#
# The gates are re-run here rather than reused from the training job because
# their splice criterion was necessarily unmet then -- the splices had not been
# run. That is not a second look at the held-out boxes in the sense that
# matters: NOTHING is refitted or selected between the two runs, the ensembles
# on disk are byte-identical, and the only input that changed is evidence that
# did not exist yet. The decision is still made once, here.
#
#   sbatch scripts/slurm/proxy_benchmark_cpu.sbatch <envfile>
#SBATCH --job-name=sd_bench
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_bench-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARMS:=a b}"
: "${OVERWRITE:=0}"

require_input "$DMSR_SR2_DIRECT_ROOT/proxy_data/labels_complete.json" \
    "scripts/slurm/index_proxy_labels_cpu.sbatch"

for ARM in ${ARMS//,/ }; do
    python -u scripts/reward/gate_catalog_proxy.py \
        --config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM"
done

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --arms ${ARMS//,/ })
[ "$OVERWRITE" = "1" ] && ARGS+=(--overwrite)
python -u scripts/reward/proxy_benchmark.py "${ARGS[@]}"

echo "=== the milestone ends at:"
echo "===   $DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/proxy_benchmark.json"
echo "=== No SR2 parameter moves until its decision says so."
