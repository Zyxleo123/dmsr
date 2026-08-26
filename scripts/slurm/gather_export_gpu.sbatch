#!/bin/bash
# Stage 0 (GPU, minutes): a finished member-gather run's HELD-OUT tiles, in the
# splice layout, so the existing Rockstar chain can gate it.
#
#   sbatch scripts/slurm/gather_export_gpu.sbatch <envfile>
#
# Forward passes only, under no_grad. The training run's OOM was the pair-sum
# autograd tape (memory `member-gather-real-training-run`), which does not exist
# here -- but the member sets are still built, and that is the 537 MB owner-array
# load plus a CSR inversion, hence the host RAM. 48G, not the trainer's 96G: this
# holds ONE box (8 hosts, ~60 MB of tensors each) rather than a 56-host pool, and
# the larger ask fits on fewer nodes for no reason.
#
# The GPU is pinned to the a6000 class for the same reason the trainer is: this
# pool has met a single 118,000-particle satellite, and a 24 GB card decides that
# with no traceback. Pinned by EXCLUSION, not `--nodelist`: a node list is a
# REQUIRED list and Slurm sets the minimum node count to its length, so
# `--nodelist=gpu24,gpu30,gpu31` asks for all three at once and pends forever on
# ReqNodeNotAvail (measured, jobs 36223-36235). On `general` every node is an
# a6000 except gpu28 (a5000) and gpu25 (unusable), so excluding those two is the
# same constraint without the node-count side effect.
#SBATCH --job-name=hg_export
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=gpu25,gpu28
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hg_export-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HG_TRAIN_DIR:?HG_TRAIN_DIR must name a finished member_gather run}"
: "${HG_BOX:=set9}"
: "${HG_RUN_DIR:?HG_RUN_DIR must name the export directory}"
: "${HG_MAX_HOSTS:=0}"

require_input "$HG_TRAIN_DIR/tuned.pt" "scripts/slurm/submit_member_gather_train.sh"

echo "=== export: train_dir=$HG_TRAIN_DIR box=$HG_BOX -> $HG_RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/features/export_gather_tiles.py \
    --run-dir "$HG_TRAIN_DIR" \
    --box "$HG_BOX" \
    --out-dir "$HG_RUN_DIR" \
    --max-hosts "$HG_MAX_HOSTS"
