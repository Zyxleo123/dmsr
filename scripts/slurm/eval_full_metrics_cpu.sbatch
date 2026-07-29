#!/bin/bash
# Stage 9c (CPU array): full-box field metrics for the final table.
#
# Displacement transfer and cross-correlation, density power and PDF, an
# equilateral bispectrum slice, LR consistency, sample diversity, and the field
# constraints re-measured so the table can say plainly whether each arm stayed
# inside the region it was allowed to move in.
#
#   JID=$(sbatch --parsable scripts/slurm/eval_full_metrics_cpu.sbatch EVAL_RUN=final)
#   sbatch --dependency=afterok:$JID --array=0-0 \
#       scripts/slurm/eval_full_metrics_cpu.sbatch EVAL_RUN=final AGGREGATE=1
#
#SBATCH --job-name=rw_eval_field
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_eval_field-%A_%a.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${EVAL_RUN:=final}"
: "${SHARD:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${NUM_SHARDS:=${SLURM_ARRAY_TASK_COUNT:-1}}"
: "${AGGREGATE:=0}"

require_input "$DMSR_REWARD_ROOT/eval/$EVAL_RUN/eval_fields.json" \
    "scripts/slurm/generate_reward_eval_boxes.sbatch"

if [ "$AGGREGATE" = "1" ]; then
    python -u scripts/reward/eval_full_metrics.py \
        --config "$REWARD_CFG" --run-name "$EVAL_RUN" --summarize-only
    echo "=== field table -> $DMSR_REWARD_ROOT/eval/$EVAL_RUN/field_eval.json"
    exit 0
fi

python -u scripts/reward/eval_full_metrics.py \
    --config "$REWARD_CFG" --run-name "$EVAL_RUN" \
    --shard "$SHARD" --num-shards "$NUM_SHARDS"

echo "=== shard $SHARD done"
