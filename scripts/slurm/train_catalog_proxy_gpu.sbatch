#!/bin/bash
# Fit one (or more) proxy arms on set0-7, then offline-gate each on set8-11.
# GPU: flat arms (A/B) are fine on CPU, but spatial arms (C-F) are 3-D convs /
# streamed critic input, so the job defaults to CUDA.
#
# submit_proxy_benchmark.sh fit submits ONE of these per arm so A..F run as
# sibling GPU jobs. Shared state (bootstrap_draws.json, proxy_baselines.json,
# train_report.json) is flock-merged inside train_catalog_proxy.py -- safe to
# overlap. You can still pass ARMS="a b" to pack several arms into one allocation.
#
# The gate for each arm listed in ARMS runs in this same job, so a fresh
# ensemble is never paired with a stale proxy_gate_*.json from a prior fit.
#
#   sbatch scripts/slurm/train_catalog_proxy_gpu.sbatch <envfile> ARMS=c
#SBATCH --job-name=sd_proxy
#SBATCH --partition=general
#SBATCH --qos=qos_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_proxy-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARMS:=a b c d e f}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
# The marker, not the table: a rows.jsonl exists as soon as ONE candidate has
# been labelled, and fitting on that would be fitting on a different dataset
# every time the indexer ran.
require_input "$DMSR_SR2_DIRECT_ROOT/proxy_data/labels_complete.json" \
    "scripts/slurm/index_proxy_labels_cpu.sbatch (after every label job)"

python -u scripts/reward/train_catalog_proxy.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" --device "${DEVICE:-cuda}" \
    --arms ${ARMS//,/ }

for ARM in ${ARMS//,/ }; do
    python -u scripts/reward/gate_catalog_proxy.py \
        --config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM" \
        --device "${DEVICE:-cuda}"
done

python - "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME" ${ARMS//,/ } <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
for arm in sys.argv[2:]:
    p = run / f"proxy_gate_{arm}.json"
    if not p.is_file():
        print(f"arm {arm}: NO VERDICT ({p} missing)")
        continue
    d = json.loads(p.read_text())
    print(f"arm {arm}: {'PASS' if d.get('passed') else 'FAIL'}")
    for f in d.get("failures", []):
        print("   !!", f)
print()
print("The splice verifications are a separate CPU stage and are expected to be")
print("unmet at this point. Run them, then proxy_benchmark.py, for the decision:")
print("  bash scripts/slurm/submit_proxy_benchmark.sh splice")
PY
