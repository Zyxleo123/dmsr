#!/bin/bash
# Render fig1 from the zoom stem (CPU, seconds). Only fig1: the zoom npz shares
# every other array with the stem it came from, so figs 2-5 would be duplicates.
#
#   sbatch scripts/slurm/gather_zoom_plot_cpu.sbatch <envfile>
#SBATCH --job-name=gfig_zplot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-gfig_zplot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${GF_ZOOM_OUT:?GF_ZOOM_OUT must be the zoom data stem}"
: "${GF_ZOOM_FIG_DIR:?GF_ZOOM_FIG_DIR must be the figure output dir}"
: "${GF_ZOOM_FIGURES:=fig1}"

require_input "$GF_ZOOM_OUT.npz" "scripts/slurm/gather_zoom_data_cpu.sbatch"

python -u scripts/features/gather_holdout_figures_plot.py \
    --data "$GF_ZOOM_OUT" \
    --out-dir "$GF_ZOOM_FIG_DIR" \
    --figures "$GF_ZOOM_FIGURES"
