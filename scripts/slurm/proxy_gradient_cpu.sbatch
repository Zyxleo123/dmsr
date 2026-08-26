#!/bin/bash
# CPU twin of proxy_gradient_gpu.sbatch. The diagnostic is small (a few dozen
# single-tile forward+backward passes), so it runs on the idle cpu partition
# with no GPU queue wait -- slower per tile, but no a6000 contention. The python
# script auto-selects CPU when no GPU is visible; DEVICE=cpu makes it explicit.
#
#   sbatch scripts/slurm/proxy_gradient_cpu.sbatch RUN_NAME=direct_a ARM=c
#   sbatch scripts/slurm/proxy_gradient_cpu.sbatch RUN_NAME=direct_a ARM=c CHECKPOINT=<ema_generator.pt>
#SBATCH --job-name=sd_grad_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_grad_cpu-%j.out

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
DEVICE=cpu

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/proxy_$ARM" \
    "scripts/slurm/train_catalog_proxy_gpu.sbatch"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM"
      --n-tiles "$N_TILES" --n-seeds "$N_SEEDS" --subsample "$SUBSAMPLE"
      --device cpu)
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")
[ -n "$CHECKPOINT" ] && ARGS+=(--checkpoint "$CHECKPOINT")

python -u scripts/reward/proxy_gradient_diagnostic.py "${ARGS[@]}"
