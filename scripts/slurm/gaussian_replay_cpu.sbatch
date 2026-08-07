#!/bin/bash
# Phase 2.2 (CPU): build the elite replay round with the EXISTING selection rule.
#
# Unmodified build_replay.py: a chunk is kept only when its ensemble is feasible
# and top-quantile AND its own leave-one-out contribution is positive. The
# Gaussian arm does not get its own, looser rule.
#
# Gated on the support gate: building a replay buffer from a run with no support
# produces a buffer of noise that a training job will happily fit. Exits 0 on a
# failed gate so dependents report the skip instead of stranding on Dependency.
#
#   sbatch scripts/slurm/gaussian_replay_cpu.sbatch RUN_NAME=gauss_ref_k16 ROUND=round_000
#
#SBATCH --job-name=rw_gauss_replay
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_gauss_replay-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${RUN_NAME:=gauss_ref_k16}"
: "${ROUND:=round_000}"
: "${ALLOW_VAL:=1}"        # the candidate stage runs on val boxes by default
: "${NO_CHECKSUM:=0}"
: "${IGNORE_GATE:=0}"

ORACLE="$DMSR_REWARD_ROOT/oracle/$RUN_NAME"
require_input "$ORACLE/oracle_manifest.jsonl" \
    "scripts/slurm/gaussian_score_cpu.sbatch AGGREGATE=1"

GATE="$ORACLE/support_gate.json"
if [[ "$IGNORE_GATE" != "1" ]]; then
    if [[ ! -r "$GATE" ]]; then
        echo ">>> SUPPORT GATE NOT DECIDED: no $GATE."
        echo ">>> Run: sbatch scripts/slurm/gaussian_support_gate_cpu.sbatch RUN_NAME=$RUN_NAME"
        echo ">>> Not building a replay buffer. (IGNORE_GATE=1 overrides, and makes"
        echo ">>> every downstream number conditional on unchecked support.)"
        exit 0
    fi
    if ! python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['passed'] else 1)" "$GATE"; then
        echo ">>> SUPPORT GATE FAILED ($GATE). Not building a replay buffer."
        python -c "import json,sys; d=json.load(open(sys.argv[1])); [print('    -', k, d['criteria'][k]['detail']) for k in d['failed']]" "$GATE" || true
        echo ">>> Follow next_steps in the JSON before training."
        exit 0
    fi
    echo "=== support gate: PASS ($GATE)"
fi

python -u scripts/reward/build_replay.py \
    --config "$REWARD_CFG" \
    --run-name "$RUN_NAME" \
    --round "$ROUND" \
    $([ "$ALLOW_VAL" = "1" ] && echo --allow-val) \
    $([ "$NO_CHECKSUM" = "1" ] && echo --no-checksum)

echo "=== replay -> $DMSR_REWARD_ROOT/replay/$ROUND/replay.jsonl"
echo "=== next: sbatch scripts/slurm/gaussian_train_gpu.sbatch ROUND=$ROUND ORACLE_RUN=$RUN_NAME"
