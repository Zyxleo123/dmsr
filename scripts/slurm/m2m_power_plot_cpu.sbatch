#!/bin/bash
# Draw the P(k) figure from curves already on disk.
#
#   sbatch scripts/slurm/m2m_power_plot_cpu.sbatch <envfile>
#
# CPU, and no generator or checkpoint is touched: m2m_power_curves.py writes the
# spectra to npz precisely so the figure is redrawable without regenerating a
# 512^3 box. Asking for a GPU to draw a plot would just queue behind real work.
#SBATCH --job-name=m2mplot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2mplot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

# M2M_CURVES_GLOB lets one job draw every curve file a run produced.
GLOB="${M2M_CURVES_GLOB:-$M2M_OUT/power_curves_*.npz}"

shopt -s nullglob
FILES=( $GLOB )
if (( ${#FILES[@]} == 0 )); then
    echo ">>> MISSING INPUT: no curve files match $GLOB"
    echo ">>> produced by: sbatch scripts/slurm/m2m_power_curves_gpu.sbatch"
    echo ">>> exiting 0 so dependents report the same rather than stranding."
    exit 0
fi

for f in "${FILES[@]}"; do
    echo "=== $f"
    python scripts/features/m2m_power_plot.py --npz "$f"
done

echo
echo "=== $(date '+%F %T')  done -- ${#FILES[@]} figure(s)"
