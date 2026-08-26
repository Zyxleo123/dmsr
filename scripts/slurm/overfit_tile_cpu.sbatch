#!/bin/bash
# Overfit ONE SR2 tile to the proxy reward (field-space gradient ascent to
# convergence) and snapshot the density slice for a GIF. CPU: only one generator
# forward (cand0); every optimisation step is just the extractor + proxy + field
# guards, no generator. Writes tile_overfit/tile_overfit_<tag>.npz + metrics.
#
#   sbatch scripts/slurm/overfit_tile_cpu.sbatch RUN_NAME=direct_a ARM=c BOX=set0 TILE=489
#SBATCH --job-name=sd_tile_of
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_tile_of-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARM:=c}"
: "${BOX:=}"
: "${TILE:=-1}"
: "${SEED:=0}"
: "${ITERS:=800}"
: "${FIELD_LR:=1e-3}"
: "${SAVE_EVERY:=5}"
: "${CKPT_EVERY:=0}"          # >0 saves full-field snapshots for the Rockstar monitor
: "${W_REWARD:=-1}"          # -1 = use the actor config default (lambda_reward=1)
: "${W_LOWK:=-1}"            # -1 = use the actor config default
: "${W_DENSITY:=-1}"
: "${W_PROX:=-1}"
: "${W_HOST:=0}"            # >0 pins the host halo's density
: "${HOST_QUANTILE:=0.99}"
: "${LABEL:=}"             # suffix so a new experiment does not overwrite an old one
: "${TOL:=1e-4}"          # <=0 disables early stop -> run the full ITERS budget
: "${PATIENCE:=20}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/proxy_$ARM" \
    "scripts/slurm/train_catalog_proxy_gpu.sbatch"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM"
      --seed "$SEED" --iters "$ITERS" --field-lr "$FIELD_LR"
      --save-every "$SAVE_EVERY" --ckpt-every "$CKPT_EVERY"
      --w-reward "$W_REWARD" --w-lowk "$W_LOWK" --w-density "$W_DENSITY"
      --w-prox "$W_PROX" --w-host "$W_HOST" --host-quantile "$HOST_QUANTILE"
      --tol "$TOL" --patience "$PATIENCE" --device cpu)
[ -n "$BOX" ] && ARGS+=(--box "$BOX")
[ "$TILE" != "-1" ] && ARGS+=(--tile "$TILE")
[ -n "$LABEL" ] && ARGS+=(--label "$LABEL")

python -u scripts/reward/overfit_tile_to_proxy.py "${ARGS[@]}"
