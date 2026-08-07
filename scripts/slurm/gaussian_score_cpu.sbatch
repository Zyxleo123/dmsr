#!/bin/bash
# Phase 2.2 (CPU array): score Gaussian-policy candidates with the EXISTING stage.
#
# sample_gaussian_candidates.py writes candidates.json in score_oracle.py's
# format precisely so this stage, oracle_report.py and build_replay.py are reused
# without modification -- constraints, Rockstar, chunk attribution, leave-one-out
# credit and the elite rule are the same code the supervised line uses, so the
# two arms cannot drift into measuring different things.
#
# The per-box a=0 BASELINE is scored by its own prerequisite array. Every shard
# holds candidates from every box, so without that split each shard would run
# Rockstar on the same {box}_base through the same work directory and the
# concurrent runs would corrupt each other's catalog.
#
#   JB=$(sbatch --parsable --array=0-3 scripts/slurm/gaussian_score_cpu.sbatch \
#          RUN_NAME=gauss_ref_k16 BASELINES=1)
#   JC=$(sbatch --parsable --dependency=afterok:$JB --array=0-15 \
#          scripts/slurm/gaussian_score_cpu.sbatch RUN_NAME=gauss_ref_k16)
#   sbatch --dependency=afterok:$JC --array=0-0 \
#          scripts/slurm/gaussian_score_cpu.sbatch RUN_NAME=gauss_ref_k16 AGGREGATE=1
#
#SBATCH --job-name=rw_gauss_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_gauss_score-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${RUN_NAME:=gauss_ref_k16}"
: "${SHARD:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${NUM_SHARDS:=${SLURM_ARRAY_TASK_COUNT:-1}}"
: "${REWARD_MODEL:=$DMSR_REWARD_ROOT/reward_model/reward_model.json}"
: "${BASELINES:=0}"
: "${AGGREGATE:=0}"
: "${GROUPS:=0}"
: "${GATE_TARGET:=0.20}"

ORACLE="$DMSR_REWARD_ROOT/oracle/$RUN_NAME"
require_input "$ORACLE/candidates.json" "scripts/slurm/gaussian_sample_gpu.sbatch"

if [ "$AGGREGATE" != "1" ]; then
    require_input "external/rockstar/rockstar" "make -C external/rockstar"
    if [ "$BASELINES" = "1" ]; then BASELINE_MODE=only; else BASELINE_MODE=skip; fi
    python -u scripts/reward/score_oracle.py \
        --config "$REWARD_CFG" \
        --run-name "$RUN_NAME" \
        --baselines "$BASELINE_MODE" \
        --shard "$SHARD" --num-shards "$NUM_SHARDS"
    echo "=== shard $SHARD scored (baselines=$BASELINE_MODE)"
    exit 0
fi

require_input "$REWARD_MODEL" "scripts/slurm/fit_reward_model_cpu.sbatch"

python -u scripts/reward/oracle_report.py \
    --config "$REWARD_CFG" \
    --run-name "$RUN_NAME" \
    --reward-model "$REWARD_MODEL" \
    --groups "$GROUPS" \
    --gate-target "$GATE_TARGET"

echo "=== oracle manifest -> $ORACLE/oracle_manifest.jsonl"
echo "=== next: sbatch scripts/slurm/gaussian_support_gate_cpu.sbatch RUN_NAME=$RUN_NAME"
