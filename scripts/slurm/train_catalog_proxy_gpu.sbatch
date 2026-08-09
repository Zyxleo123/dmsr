#!/bin/bash
# Fit BOTH proxy arms on set0-7, then gate each on set8-11, in one job. CPU:
# the proxy is a float64 two-layer MLP on ~39k rows and the reward model is a
# 16-vector Mahalanobis, so there is nothing here a GPU accelerates -- and the
# GPU build of this job ran on the CPU regardless, because DEVICE is empty in
# the preamble and --device fell through to "cpu".
#
# One job because the arm comparison's whole value is that only the feature
# vector differed: one process, one table, one set of shared hyperparameters
# chosen by cross-validation on the mean over both arms. Two jobs would be two
# chances for a difference nobody intended.
#
# The gates run here too, rather than as separate jobs, because their verdict is
# the only thing that decides whether an actor may start, and splitting them
# adds a window in which a stale "pass" from a previous fit sits on disk next to
# a new ensemble.
#
#   sbatch scripts/slurm/train_catalog_proxy_gpu.sbatch <envfile>
#SBATCH --job-name=sd_proxy
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_proxy-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARMS:=a b}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
# The marker, not the table: a rows.jsonl exists as soon as ONE candidate has
# been labelled, and fitting on that would be fitting on a different dataset
# every time the indexer ran.
require_input "$DMSR_SR2_DIRECT_ROOT/proxy_data/labels_complete.json" \
    "scripts/slurm/index_proxy_labels_cpu.sbatch (after every label job)"

python -u scripts/reward/train_catalog_proxy.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" --device "${DEVICE:-cpu}" \
    --arms ${ARMS//,/ }

for ARM in ${ARMS//,/ }; do
    python -u scripts/reward/gate_catalog_proxy.py \
        --config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM"
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
