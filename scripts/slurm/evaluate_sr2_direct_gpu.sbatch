#!/bin/bash
# Generate complete periodic boxes from a checkpoint and measure the field
# guards. Produces evidence only; the accept/reject decision is
# score_sr2_direct_cpu.sbatch's job, on a CPU node.
#
# With CHECKPOINT unset this evaluates the FROZEN generator, which is how the
# paired baseline every relative gate needs is produced. Run it first, on at
# least two seeds per box, or the calibration has nothing to measure.
#
#   sbatch scripts/slurm/evaluate_sr2_direct_gpu.sbatch RUN_NAME=direct_a   # frozen
#   sbatch scripts/slurm/evaluate_sr2_direct_gpu.sbatch RUN_NAME=direct_a \
#       CHECKPOINT=$ROOT/runs/direct_a/rung_proj_noise/ema_generator.pt TAG=rung_proj_noise
#SBATCH --job-name=sd_eval
#SBATCH --partition=general
#SBATCH --exclude=gpu28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_eval-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${CHECKPOINT:=}"
: "${TAG:=}"
: "${BOXES:=}"
: "${SEEDS:=0,1}"
: "${KEEP_FIELDS:=0}"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --seeds "$SEEDS")
[ -n "$CHECKPOINT" ] && ARGS+=(--checkpoint "$CHECKPOINT")
[ -n "$TAG" ] && ARGS+=(--tag "$TAG")
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")
[ "$KEEP_FIELDS" = "1" ] && ARGS+=(--keep-fields)

python -u scripts/reward/evaluate_sr2_direct.py "${ARGS[@]}"
