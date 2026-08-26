#!/bin/bash
# Overlay the real (Rockstar) reward on the proxy's for a tile overfit. CPU,
# redrawable from the iter_*.json rows + the optimiser metrics.
#
#   sbatch scripts/slurm/plot_tile_rockstar_cpu.sbatch RUN_NAME=direct_a TAG=set0_t486_c
#SBATCH --job-name=sd_tile_rock_plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_tile_rock_plot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${TAG:?set TAG=<box>_t<tile>_<arm>}"

require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/tile_overfit/rockstar_monitor_${TAG}" \
    "scripts/slurm/rockstar_monitor_tile_cpu.sbatch"

python -u scripts/reward/plot_tile_rockstar.py \
    --run-name "$RUN_NAME" --tag "$TAG"
