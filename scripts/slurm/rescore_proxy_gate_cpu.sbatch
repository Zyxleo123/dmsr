#!/bin/bash
# Re-run the OFFLINE proxy gate for already-trained arms, on CPU.
#
# Reads only saved artefacts -- the frozen proxy ensembles (proxy_<arm>/), the
# rows table and the fitted reward model -- so it never touches a GPU and can be
# re-run for free whenever the gate's REPORTING changes (here: the tie-margin
# ranking metric and its sweep). It does NOT retrain. Flat arms A/B evaluate a
# small MLP over ~22k held-out rows, seconds of work; still submitted, because
# the login node kills real compute and every gate number must be reproducible
# from a job record.
#
#   sbatch scripts/slurm/rescore_proxy_gate_cpu.sbatch <envfile> ARMS="a b"
#SBATCH --job-name=sd_regate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_regate-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARMS:=a b}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
require_input "$DMSR_SR2_DIRECT_ROOT/proxy_data/labels_complete.json" \
    "scripts/slurm/index_proxy_labels_cpu.sbatch"

for ARM in ${ARMS//,/ }; do
    require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/proxy_$ARM" \
        "scripts/slurm/train_catalog_proxy_gpu.sbatch (fit arm $ARM first)"
    python -u scripts/reward/gate_catalog_proxy.py \
        --config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM" --device cpu
done

echo "=== verdicts + margin sweep: $DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/proxy_gate_<arm>.json"
echo "=== read ranking_margin_sweep and rank_min_margin; arrays saved to"
echo "===   ranking_arrays_<arm>.npz for cheap re-analysis at other margins."
