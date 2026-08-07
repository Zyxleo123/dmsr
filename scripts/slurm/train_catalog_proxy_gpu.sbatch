#!/bin/bash
# Fit the proxy ensemble (set0-7), then GATE it on set8-11 in the same job.
#
# The gate runs here rather than as a separate job because its verdict is the
# only thing that decides whether the actor may start, and splitting them adds a
# window in which a stale "pass" from a previous fit is on disk next to a new
# ensemble.
#
#   sbatch scripts/slurm/train_catalog_proxy_gpu.sbatch RUN_NAME=direct_a
#SBATCH --job-name=sd_proxy
#SBATCH --partition=general
#SBATCH --exclude=gpu28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_proxy-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
require_input "$DMSR_SR2_DIRECT_ROOT/proxy_data/rows.jsonl" \
    "scripts/slurm/label_proxy_candidates_cpu.sbatch (which runs --stage index)"

python -u scripts/reward/train_catalog_proxy.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" --device "${DEVICE:-cpu}"

python -u scripts/reward/gate_catalog_proxy.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME"

GATE="$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/proxy_gate.json"
python - "$GATE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("proxy gate:", "PASS" if d.get("passed") else "FAIL")
for f in d.get("failures", []):
    print("   !!", f)
PY
