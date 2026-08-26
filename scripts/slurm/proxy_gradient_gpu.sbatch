#!/bin/bash
# Measure the proxy's gradient into the SR2 output field (dQ/d cand). GPU: it
# runs the generator forward and backprops the frozen-but-differentiable proxy,
# exactly the actor's path. Writes gradient/proxy_grad_<arm>_<tag>.{npz,json};
# the plot job then redraws from those with no GPU.
#
# a6000/general to match the actor job (chan_base=512 SR2; the a5000/2080_ti are
# excluded because a wrong-size allocation can silent-FAIL with no traceback).
#
#   sbatch scripts/slurm/proxy_gradient_gpu.sbatch RUN_NAME=direct_a ARM=c
#   sbatch scripts/slurm/proxy_gradient_gpu.sbatch RUN_NAME=direct_a ARM=c CHECKPOINT=<ema_generator.pt>
#SBATCH --job-name=sd_grad
#SBATCH --partition=general
#SBATCH --exclude=gpu28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_grad-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARM:=c}"
: "${N_TILES:=6}"
: "${N_SEEDS:=3}"
: "${SUBSAMPLE:=4000}"
: "${BOXES:=}"
: "${CHECKPOINT:=}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/proxy_$ARM" \
    "scripts/slurm/train_catalog_proxy_gpu.sbatch"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM"
      --n-tiles "$N_TILES" --n-seeds "$N_SEEDS" --subsample "$SUBSAMPLE")
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")
[ -n "$CHECKPOINT" ] && ARGS+=(--checkpoint "$CHECKPOINT")
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")

python -u scripts/reward/proxy_gradient_diagnostic.py "${ARGS[@]}"
