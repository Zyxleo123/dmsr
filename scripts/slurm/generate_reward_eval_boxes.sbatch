#!/bin/bash
# Stage 9a (GPU): ordinary random samples from every arm, on held-out boxes.
#
# The distilled arm gets exactly the same number of random draws as the prior and
# no selection. Best-of-K appears in the final table as its own arm, taken from
# the oracle run; it is never mixed into the prior or distilled numbers.
#
#   sbatch scripts/slurm/generate_reward_eval_boxes.sbatch
#   sbatch scripts/slurm/generate_reward_eval_boxes.sbatch N_SAMPLES=8 SPLIT=test
#
#SBATCH --job-name=rw_eval_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_eval_gen-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${EVAL_RUN:=final}"
: "${PRIOR_CKPT:=$DMSR_REWARD_ROOT/checkpoints/residual_prior/ckpt_best.pt}"
: "${DISTILL_CKPT:=$DMSR_REWARD_ROOT/checkpoints/reward_distill_round_000/ckpt_best.pt}"
# 'final' = set13/14/15, the boxes nothing has looked at. NOT 'test': that
# includes set12, which the SR2 subhalo study and the reward sanity check both
# ran on, and which reward.yaml therefore declares as dev_boxes. Numbers from
# set12 cannot carry a held-out claim, and the final table is the one place
# that matters. Pass SPLIT=dev deliberately to look at set12.
: "${SPLIT:=final}"
: "${BOXES:=}"
: "${N_SAMPLES:=4}"
: "${SEED0:=1000}"
: "${RESIDUAL_SCALE:=1.0}"
: "${TILE_CORE:=128}"
: "${TILE_MARGIN:=48}"   # measured receptive field 41 + safety, rounded to 8
: "${TILE_BATCH:=1}"

require_input "$PRIOR_CKPT" "scripts/slurm/train_residual_prior.sbatch"
require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"

DISTILL_ARG=()
if [[ -r "$DISTILL_CKPT" ]]; then
    DISTILL_ARG=(--distill-ckpt "$DISTILL_CKPT")
else
    echo ">>> no distilled checkpoint at $DISTILL_CKPT: generating the prior arm only."
fi

python -u scripts/reward/generate_eval_boxes.py \
    --config "$REWARD_CFG" --model-config "$MODEL_CFG" \
    --run-name "$EVAL_RUN" \
    --prior-ckpt "$PRIOR_CKPT" \
    "${DISTILL_ARG[@]}" \
    --split "$SPLIT" ${BOXES:+--boxes "$BOXES"} \
    --n-samples "$N_SAMPLES" --seed0 "$SEED0" --base-seed "$BASE_SEED" \
    --residual-scale "$RESIDUAL_SCALE" \
    --tile-core "$TILE_CORE" --tile-margin "$TILE_MARGIN" --tile-batch "$TILE_BATCH" \
    --device "$DEVICE"

echo "=== eval fields -> $DMSR_REWARD_ROOT/eval/$EVAL_RUN"
