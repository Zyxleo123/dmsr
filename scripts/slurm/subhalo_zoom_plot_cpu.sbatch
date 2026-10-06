#!/bin/bash
# Render the per-subhalo slab figures from the extracted data (CPU, seconds).
#
# Reads only the npz + json, so re-rendering after a colour or layout change
# never re-reads a field.
#
#   sbatch scripts/slurm/subhalo_zoom_plot_cpu.sbatch <envfile>
#SBATCH --job-name=sz_plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-sz_plot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${SZ_OUT:?SZ_OUT must be the data stem}"
: "${SZ_FIG_DIR:?SZ_FIG_DIR must be the figure directory}"
: "${SZ_DEPTH_INDEX:=0}"
: "${SZ_ARM:=tuned}"
: "${SZ_FIGURES:=fig1,fig2,fig3,fig4}"

require_input "$SZ_OUT.npz" "scripts/slurm/subhalo_zoom_data_cpu.sbatch"

mkdir -p "$SZ_FIG_DIR"
python -u scripts/features/subhalo_zoom_plot.py \
    --data "$SZ_OUT" --out-dir "$SZ_FIG_DIR" \
    --depth-index "$SZ_DEPTH_INDEX" --arm "$SZ_ARM" --figures "$SZ_FIGURES"

echo "=== figures in $SZ_FIG_DIR"
ls -l "$SZ_FIG_DIR"
