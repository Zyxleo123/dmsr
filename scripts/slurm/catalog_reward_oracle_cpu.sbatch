#!/bin/bash
# Stage 6b/6c (CPU array): constraints + halo finding + reward for each candidate.
#
# One array task per shard of the candidate list. Rockstar on a 512^3 box is the
# expensive part (~10-20 min plus a 3.7 GB GADGET2 dump that is deleted straight
# afterwards), so shards run in parallel.
#
# The per-box a=0 BASELINE is scored by its own prerequisite array (BASELINES=1),
# never by the candidate shards. Sharding is by candidate, and with K=8 and 8
# shards every shard holds one candidate from every box -- so every shard would
# score every box's baseline, writing the same {box}_base output through the same
# Rockstar work directory. The existence check happens before the write and there
# is no lock, so concurrent Rockstar runs corrupt each other's catalog.
#
# Aggregation (Gate B) is a *third* submission with AGGREGATE=1 and a dependency,
# rather than task 0 polling squeue: the reward needs every candidate to exist,
# and a partial Gate B verdict is worse than none.
#
#   JB=$(sbatch --parsable --array=0-3 scripts/slurm/catalog_reward_oracle_cpu.sbatch \
#       RUN_NAME=prior_k8 BASELINES=1)
#   JID=$(sbatch --parsable --dependency=afterok:$JB \
#       scripts/slurm/catalog_reward_oracle_cpu.sbatch RUN_NAME=prior_k8)
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
# 0 = one group per box, which is what a group now is. Set it only to cap the
# number of boxes reported.
: "${GROUPS:=0}"
: "${GATE_TARGET:=0.20}"
: "${AGGREGATE:=0}"
# 1 = score ONLY the per-box baselines, sharded over boxes. The candidate array
# then runs with --baselines skip and cannot race on them.
: "${BASELINES:=0}"

ORACLE="$DMSR_REWARD_ROOT/oracle/$RUN_NAME"
require_input "$ORACLE/candidates.json" "scripts/slurm/sample_reward_oracle.sbatch"

if [ "$AGGREGATE" != "1" ]; then
    require_input "external/rockstar/rockstar" "make -C external/rockstar"
    if [ "$BASELINES" = "1" ]; then
        BASELINE_MODE=only
    else
        BASELINE_MODE=skip
    fi
    python -u scripts/reward/score_oracle.py \
        --config "$REWARD_CFG" \
        --run-name "$RUN_NAME" \
        --baselines "$BASELINE_MODE" \
        --shard "$SHARD" --num-shards "$NUM_SHARDS"
    echo "=== shard $SHARD scored (baselines=$BASELINE_MODE)"
    if [ "$BASELINES" = "1" ]; then
        echo "=== candidates next: sbatch --dependency=afterok:\$JID \\"
        echo "===   scripts/slurm/catalog_reward_oracle_cpu.sbatch RUN_NAME=$RUN_NAME"
    else
        echo "=== aggregate with: sbatch --dependency=afterok:\$JID --array=0-0 \\"
        echo "===   scripts/slurm/catalog_reward_oracle_cpu.sbatch RUN_NAME=$RUN_NAME AGGREGATE=1"
    fi
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
