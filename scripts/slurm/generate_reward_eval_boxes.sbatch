#!/bin/bash
# Stage 9a (GPU): ordinary random samples from every arm, on held-out boxes.
#
# The distilled arm gets exactly the same number of random draws as the prior and
# no selection. Best-of-K is its own arm -- a disjoint block of PRIOR draws that
# the CPU stage selects one of by reward -- and is never mixed into the prior or
# distilled numbers.
#
# Arms and the claim each one protects:
#   sr2        the frozen baseline
#   prior      supervised residual prior, average sample
#   distill    reward-weighted student, average sample
#   bestofk    "distillation beat sampling the prior harder"
#   samecomp   "distillation beat the same optimisation budget with no reward"
#   abundance  "the OCCUPATION reward did it, not any catalog reward"
# The last three are controls. Without them a distill-vs-prior difference has
# three standing explanations, so eval_catalogs.py names each missing arm in its
# verdict rather than reporting the comparison as if it were clean.
#
#   sbatch scripts/slurm/generate_reward_eval_boxes.sbatch
#   sbatch scripts/slurm/generate_reward_eval_boxes.sbatch N_SAMPLES=8 SPLIT=dev
#   sbatch scripts/slurm/generate_reward_eval_boxes.sbatch \
#       SAME_COMPUTE_CKPT=.../prior_long/ckpt_best.pt BEST_OF_K=8
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
# Controls. Empty = that arm is skipped, and eval_catalogs.py says so in the
# verdict instead of quietly reporting a two-arm comparison.
: "${SAME_COMPUTE_CKPT:=$DMSR_REWARD_ROOT/checkpoints/residual_prior_samecompute/ckpt_best.pt}"
: "${ABUNDANCE_CKPT:=$DMSR_REWARD_ROOT/checkpoints/reward_distill_abundance/ckpt_best.pt}"
: "${BEST_OF_K:=8}"
: "${SEED0:=1000}"
: "${RESIDUAL_SCALE:=1.0}"
: "${TILE_CORE:=128}"
: "${TILE_MARGIN:=48}"   # measured receptive field 41 + safety, rounded to 8
: "${TILE_BATCH:=1}"

require_input "$PRIOR_CKPT" "scripts/slurm/train_residual_prior.sbatch"
require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"

ARM_ARGS=()
add_arm() {   # $1 = flag, $2 = checkpoint, $3 = what it is
    if [[ -n "$2" && -r "$2" ]]; then
        ARM_ARGS+=("$1" "$2")
    else
        echo ">>> no $3 checkpoint at ${2:-<unset>}: that arm is SKIPPED."
    fi
}
add_arm --distill-ckpt "$DISTILL_CKPT" "distilled"
add_arm --same-compute-ckpt "$SAME_COMPUTE_CKPT" "same-compute supervised"
add_arm --abundance-ckpt "$ABUNDANCE_CKPT" "abundance-only distilled"

python -u scripts/reward/generate_eval_boxes.py \
    --config "$REWARD_CFG" --model-config "$MODEL_CFG" \
    --run-name "$EVAL_RUN" \
    --prior-ckpt "$PRIOR_CKPT" \
    "${ARM_ARGS[@]}" \
    --best-of-k "$BEST_OF_K" \
    --split "$SPLIT" ${BOXES:+--boxes "$BOXES"} \
    --n-samples "$N_SAMPLES" --seed0 "$SEED0" --base-seed "$BASE_SEED" \
    --residual-scale "$RESIDUAL_SCALE" \
    --tile-core "$TILE_CORE" --tile-margin "$TILE_MARGIN" --tile-batch "$TILE_BATCH" \
    --device "$DEVICE"

echo "=== eval fields -> $DMSR_REWARD_ROOT/eval/$EVAL_RUN"
