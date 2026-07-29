#!/bin/bash
# Stage 6b/6c (CPU array): constraints + halo finding + reward for each candidate.
#
# One array task per shard of the candidate list. Rockstar on a 512^3 box is the
# expensive part (~10-20 min plus a 3.7 GB GADGET2 dump that is deleted straight
# afterwards), so shards run in parallel.
#
# Aggregation (Gate B) is a *second* submission of this same script with
# AGGREGATE=1 and a dependency, rather than task 0 polling squeue: the reward
# needs every candidate to exist, and a partial Gate B verdict is worse than
# none.
#
#   JID=$(sbatch --parsable scripts/slurm/catalog_reward_oracle_cpu.sbatch RUN_NAME=prior_k8)
#   sbatch --dependency=afterok:$JID --array=0-0 \
#       scripts/slurm/catalog_reward_oracle_cpu.sbatch RUN_NAME=prior_k8 AGGREGATE=1
#
#SBATCH --job-name=rw_oracle_cat
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_oracle_cat-%A_%a.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${RUN_NAME:=prior_k8}"
: "${SHARD:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${NUM_SHARDS:=${SLURM_ARRAY_TASK_COUNT:-1}}"
: "${REWARD_MODEL:=$DMSR_REWARD_ROOT/reward_model/reward_model.json}"
: "${GROUPS:=3}"
: "${GATE_TARGET:=0.20}"
: "${AGGREGATE:=0}"

ORACLE="$DMSR_REWARD_ROOT/oracle/$RUN_NAME"
require_input "$ORACLE/candidates.json" "scripts/slurm/sample_reward_oracle.sbatch"

if [ "$AGGREGATE" != "1" ]; then
    require_input "external/rockstar/rockstar" "make -C external/rockstar"
    python -u scripts/reward/score_oracle.py \
        --config "$REWARD_CFG" \
        --run-name "$RUN_NAME" \
        --shard "$SHARD" --num-shards "$NUM_SHARDS"
    echo "=== shard $SHARD scored"
    echo "=== aggregate with: sbatch --dependency=afterok:\$JID --array=0-0 \\"
    echo "===   scripts/slurm/catalog_reward_oracle_cpu.sbatch RUN_NAME=$RUN_NAME AGGREGATE=1"
    exit 0
fi

require_input "$REWARD_MODEL" "scripts/slurm/fit_reward_model_cpu.sbatch"

python -u scripts/reward/oracle_report.py \
    --config "$REWARD_CFG" \
    --run-name "$RUN_NAME" \
    --reward-model "$REWARD_MODEL" \
    --groups "$GROUPS" \
    --gate-target "$GATE_TARGET"

echo "=== Gate B verdict -> $ORACLE/gate_b.json"
