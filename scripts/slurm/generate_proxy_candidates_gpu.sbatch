#!/bin/bash
# Proxy data, stage 1 (GPU): one candidate full box + its per-tile features.
#
# Features for BOTH arms are computed HERE, while the 3.2 GB field is still
# resident, so the proxy trainer reads a few MB per candidate instead of
# re-reading the array -- and so the two arms are derived from one pass over one
# field and cannot disagree about which tile they describe.
#
# Node choice: `general` with gpu28 excluded, i.e. an a6000 (48 GB) rather than
# the a5000 (24 GB). The footprint is one 14^3 -> 70^3 forward pass at a time,
# which fits anywhere, but the host holds two 512^3 six-channel fields (6.4 GB)
# and the a6000 nodes are the ones with the RAM headroom to match.
#
#   sbatch scripts/slurm/generate_proxy_candidates_gpu.sbatch \
#       RUN_NAME=direct_a BOX=set0 SOURCE=frozen SEED=0
#   sbatch scripts/slurm/generate_proxy_candidates_gpu.sbatch \
#       RUN_NAME=direct_a BOX=set0 SOURCE=intervention ALPHA=0.5 MODE=vel
#SBATCH --job-name=sd_gen
#SBATCH --partition=general
#SBATCH --exclude=gpu28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_gen-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set0}"
: "${SOURCE:=frozen}"
: "${SEED:=0}"
: "${ALPHA:=}"
: "${MODE:=both}"
: "${CHECKPOINT:=}"
: "${DROP_FIELD:=0}"

read -r -a _BOXES <<< "${BOXES//,/ }"
BOX="${BOX:-${_BOXES[${SLURM_ARRAY_TASK_ID:-0}]}}"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --stage generate
      --box "$BOX" --source "$SOURCE" --seed "$SEED" --base-seed "$BASE_SEED"
      --mode "$MODE")
[ -n "$ALPHA" ] && ARGS+=(--alpha "$ALPHA")
[ -n "$CHECKPOINT" ] && ARGS+=(--checkpoint "$CHECKPOINT")
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")
[ "$DROP_FIELD" = "1" ] && ARGS+=(--drop-field)

python -u scripts/reward/collect_catalog_proxy_data.py "${ARGS[@]}"
