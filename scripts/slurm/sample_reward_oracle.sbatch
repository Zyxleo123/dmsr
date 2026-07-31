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

# Gate B is gated on Gate A, per docs/reward_residual_diffusion.md sec. 3a:
# "do not evaluate catalog rewards on a prior that fails Gate A -- the reward
# numbers would be measuring the bug". Depending on the training job having
# exited 0 does not check that. Exit 0 on a failed gate so the scoring and
# verdict jobs behind this one report a skip rather than dying on Dependency.
: "${GATE_A_JSON:=$(dirname "$PRIOR_CKPT")/gate_a.json}"
: "${IGNORE_GATE_A:=0}"
if [[ "$IGNORE_GATE_A" != "1" ]]; then
    if [[ ! -r "$GATE_A_JSON" ]]; then
        echo ">>> GATE A NOT DECIDED: no $GATE_A_JSON."
        echo ">>> Run: python scripts/reward/gate_a_check.py --run-dir $(dirname "$PRIOR_CKPT")"
        echo ">>> Not sampling. (IGNORE_GATE_A=1 overrides, and makes every"
        echo ">>> downstream number conditional on an unchecked prior.)"
        exit 0
    fi
    if ! python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['passed'] else 1)" \
            "$GATE_A_JSON"; then
        echo ">>> GATE A FAILED ($GATE_A_JSON). Not sampling."
        python -c "import json,sys; d=json.load(open(sys.argv[1])); [print('    -', k, d['criteria'][k]['detail']) for k in d['failed']]" \
            "$GATE_A_JSON" || true
        echo ">>> Fix the prior, the whitening, the sampler or residual_scale first."
        exit 0
    fi
    echo "=== Gate A: PASS ($GATE_A_JSON)"
fi

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
