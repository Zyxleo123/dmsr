#!/bin/bash
# Rung 2 (GPU): generate one CEM population.
#
# Iteration 0 samples from seeds; later iterations read the previous iteration's
# elites.npz and perturb those noise vectors. Generation only -- halo finding and
# reward are the CPU stages, so the GPU is never held while Rockstar runs.
#
#   sbatch scripts/slurm/cem_generate_gpu.sbatch CEM_RUN=cem_a ITER=0
#
#SBATCH --job-name=cem_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-cem_gen-%j.out

set -euo pipefail

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

set -euo pipefail

: "${CEM_CFG:=configs/reward/cem.yaml}"
: "${CEM_RUN:=cem_a}"
: "${ITER:=0}"
: "${PRIOR_CKPT:=$DMSR_REWARD_ROOT/checkpoints/residual_prior/ckpt_best.pt}"
: "${POPULATION:=}"
: "${CEM_BOXES:=}"          # empty = the cem config's own box list
: "${SEED0:=0}"
: "${RESIDUAL_SCALE:=1.0}"
: "${TILE_CORE:=128}"
: "${TILE_MARGIN:=48}"      # measured receptive field 41 + safety, rounded to 8
: "${TILE_BATCH:=1}"
: "${N_STEPS:=}"

require_input "$PRIOR_CKPT" "scripts/slurm/train_residual_prior.sbatch"
require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"

# Iteration i>0 cannot run without the previous iteration's winners. This is a
# gate, not a failure: exit 0 so the rest of the chain reports the same cause
# instead of sitting in DependencyNeverSatisfied.
if [ "$ITER" != "0" ]; then
    require_input "$DMSR_REWARD_ROOT/oracle/${CEM_RUN}_it$((ITER - 1))/elites.npz" \
        "scripts/slurm/cem_select_elites_cpu.sbatch (previous iteration)"
fi

python -u scripts/reward/cem_search.py \
    --config "$REWARD_CFG" --model-config "$MODEL_CFG" --cem-config "$CEM_CFG" \
    --checkpoint "$PRIOR_CKPT" \
    --run-name "$CEM_RUN" --iteration "$ITER" \
    ${CEM_BOXES:+--boxes "$CEM_BOXES"} \
    ${POPULATION:+--population "$POPULATION"} \
    --seed0 "$SEED0" --base-seed "$BASE_SEED" \
    --residual-scale "$RESIDUAL_SCALE" \
    --tile-core "$TILE_CORE" --tile-margin "$TILE_MARGIN" \
    --tile-batch "$TILE_BATCH" \
    ${N_STEPS:+--n-steps "$N_STEPS"} \
    --device "$DEVICE"

echo "=== population -> $DMSR_REWARD_ROOT/oracle/${CEM_RUN}_it${ITER}"
