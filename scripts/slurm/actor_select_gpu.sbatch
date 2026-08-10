#!/bin/bash
# Choose the actor-gate checks: generate the held-out verify boxes with the
# probe checkpoint, predict every tile's dR, and write the stratified plan plus
# the actor fields the run stage splices from. Minutes of GPU.
#
# Separate from running them so the plan is readable -- and reviewable -- before
# twelve full-box Rockstar runs are spent on it.
#
#   sbatch scripts/slurm/actor_select_gpu.sbatch <envfile> ARM=b
#SBATCH --job-name=sd_actsel
#SBATCH --partition=general
#SBATCH --exclude=gpu28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_actsel-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARM:=a}"

require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/probe_$ARM/ema_generator.pt" \
    "scripts/slurm/train_sr2_direct_gpu.sbatch with ARM=$ARM PROBE=1"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --arm "$ARM" --stage select)
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")

python -u scripts/reward/actor_rockstar_verify.py "${ARGS[@]}"
