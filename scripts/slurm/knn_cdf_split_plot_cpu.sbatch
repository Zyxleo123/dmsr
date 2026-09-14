#!/bin/bash
# Redraw the kNN split-diagnostic figure from its JSON. Reads saved numbers only.
#
#   sbatch scripts/slurm/knn_cdf_split_plot_cpu.sbatch <envfile>
#SBATCH --job-name=knnsplitplot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-knnsplitplot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${KS_OUT:?KS_OUT must be the diagnostic json}"
if [[ ! -s "$KS_OUT" ]]; then
    echo "no diagnostic JSON at $KS_OUT -- did knnsplit run?"; exit 0
fi
python -u scripts/features/knn_cdf_split_plot.py --json "$KS_OUT" --out "${KS_OUT%.json}.png"
