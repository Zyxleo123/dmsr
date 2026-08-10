#!/bin/bash
# Train the actor for ONE unfreezing rung. Refuses to start unless the proxy
# gate passed (the python script checks and exits 0 with an explanation).
#
# One rung per job, checkpointed, so advancing is an explicit decision taken
# after the real catalog result -- never a schedule inside a single run.
#
#   sbatch scripts/slurm/train_sr2_direct_gpu.sbatch RUN_NAME=direct_a RUNG=proj_noise
#   sbatch scripts/slurm/train_sr2_direct_gpu.sbatch RUN_NAME=direct_a OVERFIT=1
#   sbatch scripts/slurm/train_sr2_direct_gpu.sbatch RUN_NAME=direct_a ARM=b PROBE=1
#SBATCH --job-name=sd_actor
#SBATCH --partition=general
#SBATCH --exclude=gpu28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_actor-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${RUNG:=}"
: "${MAX_STEPS:=0}"
: "${OVERFIT:=0}"
: "${PROBE:=0}"
: "${ARM:=a}"
: "${RESUME:=}"
: "${BOXES:=}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/proxy_$ARM" \
    "scripts/slurm/train_catalog_proxy_gpu.sbatch"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM")
[ -n "$RUNG" ] && ARGS+=(--rung "$RUNG")
[ "$MAX_STEPS" != "0" ] && ARGS+=(--max-steps "$MAX_STEPS")
[ "$OVERFIT" = "1" ] && ARGS+=(--overfit)
[ "$PROBE" = "1" ] && ARGS+=(--probe)
[ -n "$RESUME" ] && ARGS+=(--resume "$RESUME")
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")

python -u scripts/reward/train_sr2_direct.py "${ARGS[@]}"
