#!/bin/bash
# Stage 3/4 (CPU array): one shard of candidate boxes. Each candidate is one
# full-box Rockstar run returning one proposal-level reward per selected host.
#
# Shard by candidate index, never by box: a shard must own its candidates
# exclusively, because two tasks composing the same candidate id would run
# Rockstar into the same work directory and corrupt each other's catalog.
#
#   sbatch --array=0-3 scripts/slurm/local_editor_candidates_cpu.sbatch \
#       RUN_NAME=le_a BOX=set8 ARM=cem ROUND=0
#SBATCH --job-name=le_cand
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_local_editor/logs/slurm-le_cand-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_local_editor_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_local_editor_common.sh" "$@"
set -euo pipefail

: "${BOX:=set8}"
: "${ARM:=random}"
: "${CONTROL:=none}"
: "${ROUND:=0}"
: "${CANDIDATES:=0}"
: "${ACTIONS_JSON:=}"
: "${MEASURE_ONLY:=0}"
: "${WITH_DENSITY:=0}"
: "${SHARD:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${NUM_SHARDS:=${SLURM_ARRAY_TASK_COUNT:-1}}"

require_input "external/rockstar/rockstar" "make -C external/rockstar"
require_input "$DMSR_LOCAL_EDITOR_ROOT/runs/$RUN_NAME/pools/pools_$BOX.npz" \
    "scripts/slurm/local_editor_members_cpu.sbatch"

ARGS=(--config "$LE_CFG" --run-name "$RUN_NAME" --box "$BOX"
      --arm "$ARM" --control "$CONTROL" --round "$ROUND"
      --candidates "$CANDIDATES" --shard "$SHARD" --num-shards "$NUM_SHARDS"
      --base-seed "$BASE_SEED" --seed "$SEED")
if [ -n "$ACTIONS_JSON" ]; then
    require_input "$ACTIONS_JSON" "aggregate_cem_round.py --stage propose (or train_action_flow.py --stage sample)"
    ARGS+=(--actions-json "$ACTIONS_JSON")
fi
[ "$MEASURE_ONLY" = "1" ] && ARGS+=(--measure-only)
[ "$WITH_DENSITY" = "1" ] && ARGS+=(--with-density)

python -u scripts/reward/run_editor_candidates.py "${ARGS[@]}"
