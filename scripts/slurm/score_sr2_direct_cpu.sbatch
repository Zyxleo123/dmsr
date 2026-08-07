#!/bin/bash
# Calibrate the relative density tolerance from frozen seed-to-seed spread, or
# gate a candidate checkpoint against frozen SR2 paired on (box, seed).
#
# CPU because it only reads the JSONL rows the evaluation job wrote -- asking
# for a GPU to compare eleven numbers just queues behind real work.
#
#   sbatch scripts/slurm/score_sr2_direct_cpu.sbatch RUN_NAME=direct_a STAGE=calibrate
#   sbatch scripts/slurm/score_sr2_direct_cpu.sbatch RUN_NAME=direct_a STAGE=gate \
#       TAG=rung_proj_noise
#SBATCH --job-name=sd_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_score-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${STAGE:=gate}"
: "${TAG:=}"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --stage "$STAGE")
[ -n "$TAG" ] && ARGS+=(--tag "$TAG")

python -u scripts/reward/score_sr2_direct.py "${ARGS[@]}"
