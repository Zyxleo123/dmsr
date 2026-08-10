#!/bin/bash
# Proxy data, stage 1 (GPU): one candidate full box + its per-tile features.
#
# Features for BOTH arms are computed HERE, while the 3.2 GB field is still
# resident, so the proxy trainer reads a few MB per candidate instead of
# re-reading the array -- and so the two arms are derived from one pass over one
# field and cannot disagree about which tile they describe.
#
# Node choice: `legacy` with its matching `--qos=qos_legacy`. It USED to be
# `general,legacy` with no qos, but the cluster now enforces a per-partition QOS
# (general->qos_general, legacy->qos_legacy, cpu->qos_cpu), and a job listing two
# partitions whose default QOS differ is rejected at submit with "Invalid qos
# specification" -- there is no single qos valid for both, so spanning them is no
# longer possible. legacy is the right single choice for the same reason it was
# half of the old pair: checked with myfree, every general a6000 is contended
# while six legacy nodes are IDLE, and the GPU footprint is one 14^3 -> 70^3
# forward pass at a time, a few hundred MB, so a 2080 Ti (11 GB) or a v100 serves
# as well as an a6000. To run on general instead, set both lines together:
# `--partition=general` and `--qos=qos_general` (never one without the other).
#
# --mem is 48G, not 128G. Peak host RAM is the intervention path: base field
# 3.2 + HR 3.2 + mask 0.5 + the blended copy 3.2 + the frozen reference 3.2,
# about 13.5 GB, plus torch's host side. 128G asked for ten times what the job
# touches and made it unschedulable on every legacy node (93-188 GB free), so
# the over-ask was itself the reason these queued.
#
# If a job ever fails here with no Python traceback, check which node it landed
# on first: a CUDA allocation failure on a smaller card looks like a silent
# FAILED, and the fix is --exclude, not a rewrite.
#
#   sbatch scripts/slurm/generate_proxy_candidates_gpu.sbatch \
#       RUN_NAME=direct_a BOX=set0 SOURCE=frozen SEED=0
#   sbatch scripts/slurm/generate_proxy_candidates_gpu.sbatch \
#       RUN_NAME=direct_a BOX=set0 SOURCE=intervention ALPHA=0.5 MODE=vel
#SBATCH --job-name=sd_gen
#SBATCH --partition=legacy
#SBATCH --qos=qos_legacy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
