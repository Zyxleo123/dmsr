#!/bin/bash
# Render the tile-overfit process to a GIF + before/after still. CPU, redrawable
# from the npz snapshots (no model). ffmpeg is absent, so the GIF uses matplotlib
# PillowWriter.
#
#   sbatch scripts/slurm/render_tile_overfit_cpu.sbatch RUN_NAME=direct_a TAG=set0_t489_c
#SBATCH --job-name=sd_tile_gif
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_tile_gif-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${TAG:?set TAG=<box>_t<tile>_<arm>, e.g. TAG=set0_t489_c}"
: "${FPS:=12}"

require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/tile_overfit/tile_overfit_${TAG}.npz" \
    "scripts/slurm/overfit_tile_cpu.sbatch"

python -u scripts/reward/render_tile_overfit.py \
    --run-name "$RUN_NAME" --tag "$TAG" --fps "$FPS"
