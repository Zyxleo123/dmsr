#!/bin/bash
# Draw the slice figures from the dumped npz. CPU, reads a few MB.
#
#   sbatch scripts/slurm/m2m_slice_plot_cpu.sbatch <envfile>
#SBATCH --job-name=m2msliceplot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2msliceplot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

GLOB="${M2M_SLICES_GLOB:-$M2M_OUT/slices_*.npz}"
shopt -s nullglob
FILES=( $GLOB )
if (( ${#FILES[@]} == 0 )); then
    echo ">>> MISSING INPUT: no slice files match $GLOB"
    echo ">>> produced by: sbatch scripts/slurm/m2m_slices_gpu.sbatch"
    exit 0
fi
for f in "${FILES[@]}"; do
    echo "=== $f"
    python scripts/features/m2m_slice_plot.py --npz "$f"
done
echo
echo "=== $(date '+%F %T')  done -- ${#FILES[@]} slice set(s)"
