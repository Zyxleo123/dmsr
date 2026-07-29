#!/bin/bash
# Stage 9b (CPU array): halo finding and catalog metrics for the final table.
#
# Optimized metrics (SHMF, HOD, reward) and held-out metrics (one/two-halo
# correlation, radial profile, relative velocity, host HMF, isolated-vs-sub
# counts, count variance) are computed together but reported separately.
# Confidence intervals resample boxes, never chunks.
#
#   JID=$(sbatch --parsable scripts/slurm/eval_catalogs_cpu.sbatch EVAL_RUN=final)
#   sbatch --dependency=afterok:$JID --array=0-0 \
#       scripts/slurm/eval_catalogs_cpu.sbatch EVAL_RUN=final AGGREGATE=1
#
#SBATCH --job-name=rw_eval_cat
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_eval_cat-%A_%a.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${EVAL_RUN:=final}"
: "${SHARD:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${NUM_SHARDS:=${SLURM_ARRAY_TASK_COUNT:-1}}"
: "${REWARD_MODEL:=$DMSR_REWARD_ROOT/reward_model/reward_model.json}"
: "${AGGREGATE:=0}"

require_input "$DMSR_REWARD_ROOT/eval/$EVAL_RUN/eval_fields.json" \
    "scripts/slurm/generate_reward_eval_boxes.sbatch"

if [ "$AGGREGATE" = "1" ]; then
    python -u scripts/reward/eval_catalogs.py \
        --config "$REWARD_CFG" --run-name "$EVAL_RUN" \
        --reward-model "$REWARD_MODEL" --summarize-only
    echo "=== catalog table -> $DMSR_REWARD_ROOT/eval/$EVAL_RUN/catalog_eval.json"
    exit 0
fi

require_input "external/rockstar/rockstar" "make -C external/rockstar"

python -u scripts/reward/eval_catalogs.py \
    --config "$REWARD_CFG" --run-name "$EVAL_RUN" \
    --reward-model "$REWARD_MODEL" \
    --shard "$SHARD" --num-shards "$NUM_SHARDS"

echo "=== shard $SHARD done"
