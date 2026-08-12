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
# GPU, despite the name: the re-gate and the arm comparison now STREAM arm E's
# dense grid and arm F's built-on-GPU critic input through the ensembles, which a
# CPU-only node cannot do in minutes. The flat arms are unaffected.
#SBATCH --job-name=sd_bench
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_bench-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARMS:=a b c d e f}"
: "${OVERWRITE:=0}"

require_input "$DMSR_SR2_DIRECT_ROOT/proxy_data/labels_complete.json" \
    "scripts/slurm/index_proxy_labels_cpu.sbatch"

for ARM in ${ARMS//,/ }; do
    python -u scripts/reward/gate_catalog_proxy.py \
        --config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM" \
        --device "${DEVICE:-cuda}"
done

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --arms ${ARMS//,/ }
      --device "${DEVICE:-cuda}")
[ "$OVERWRITE" = "1" ] && ARGS+=(--overwrite)
python -u scripts/reward/proxy_benchmark.py "${ARGS[@]}"

echo "=== the milestone ends at:"
echo "===   $DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/proxy_benchmark.json"
echo "=== No SR2 parameter moves until its decision says so."
