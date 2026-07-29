#!/bin/bash
# Stage 6a (GPU): generate K independent residual realisations per box.
#
# This job only generates and saves. Halo finding and reward computation are the
# separate CPU stage below, so a GPU is never held while Rockstar runs.
#
# Each realisation is a full 512^3 six-channel residual (3.2 GB), sampled with
# valid-core tiling: the written core does not depend on the tiling, because the
# margin exceeds the measured receptive-field half-width (41 cells) and the model
# normalises pointwise. The job refuses to run if the margin is too small.
#
#   sbatch scripts/slurm/sample_reward_oracle.sbatch
#   sbatch scripts/slurm/sample_reward_oracle.sbatch K=16 RUN_NAME=prior_k16
#
#SBATCH --job-name=rw_oracle_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_oracle_gen-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${PRIOR_CKPT:=$DMSR_REWARD_ROOT/checkpoints/residual_prior/ckpt_best.pt}"
: "${RUN_NAME:=prior_k8}"
: "${BOXES:=}"                 # empty = the val split from the config
: "${SPLIT:=val}"
: "${K:=8}"                    # num_ensembles_K
: "${SEED0:=0}"
: "${RESIDUAL_SCALE:=1.0}"
: "${TILE_CORE:=128}"
: "${TILE_MARGIN:=48}"   # measured receptive field 41 + safety, rounded to 8
: "${TILE_BATCH:=1}"
: "${N_STEPS:=}"

require_input "$PRIOR_CKPT" "scripts/slurm/train_residual_prior.sbatch"
require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"

python -u scripts/reward/sample_oracle_candidates.py \
    --config "$REWARD_CFG" --model-config "$MODEL_CFG" \
    --checkpoint "$PRIOR_CKPT" \
    --run-name "$RUN_NAME" \
    --split "$SPLIT" ${BOXES:+--boxes "$BOXES"} \
    --k "$K" --seed0 "$SEED0" --base-seed "$BASE_SEED" \
    --residual-scale "$RESIDUAL_SCALE" \
    --tile-core "$TILE_CORE" --tile-margin "$TILE_MARGIN" \
    --tile-batch "$TILE_BATCH" \
    ${N_STEPS:+--n-steps "$N_STEPS"} \
    --device "$DEVICE"

echo "=== candidates -> $DMSR_REWARD_ROOT/oracle/$RUN_NAME"
echo "=== next: sbatch scripts/slurm/catalog_reward_oracle_cpu.sbatch RUN_NAME=$RUN_NAME"
