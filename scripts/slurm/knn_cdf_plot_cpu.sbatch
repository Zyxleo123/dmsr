#!/bin/bash
# Redraw the kNN CDF diagnostic figure from its JSON. Reads saved numbers only.
#
#   sbatch scripts/slurm/knn_cdf_plot_cpu.sbatch <envfile>
#SBATCH --job-name=knnplot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-knnplot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${KC_OUT:?KC_OUT must be the diagnostic json}"
: "${KC_PNG:=${KC_OUT%.json}.png}"
if [[ ! -s "$KC_OUT" ]]; then
    # Upstream never wrote anything: say so and exit 0 rather than strand nothing.
    echo "no diagnostic JSON at $KC_OUT -- did knncdf run?"; exit 0
fi
python -u scripts/features/knn_cdf_diagnostic_plot.py \
    --json "$KC_OUT" --out "$KC_PNG" ${KC_ORDER:+--order $KC_ORDER}
