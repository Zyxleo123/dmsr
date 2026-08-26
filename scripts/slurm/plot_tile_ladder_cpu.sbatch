#!/bin/bash
# Compare the tile-overfit control ladder in one figure (crossover + real
# occupancy change per rung). CPU, redrawable from each rung's saved rows.
#
#   sbatch scripts/slurm/plot_tile_ladder_cpu.sbatch <envfile> \
#       LABELS=ctlA,ctlB,ctlC,ctlD BOX=set0 TILE=486 ARM=c
#SBATCH --job-name=sd_tile_ladder
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_tile_ladder-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${LABELS:?set LABELS=ctlA,ctlB,ctlC,ctlD (low->high control)}"
: "${BOX:=set0}"; : "${TILE:=486}"; : "${ARM:=c}"

python -u scripts/reward/plot_tile_ladder.py \
    --run-name "$RUN_NAME" --labels "$LABELS" --box "$BOX" --tile "$TILE" --arm "$ARM"
