#!/bin/bash
# Stage 2 (CPU, seconds): render the figures from the stage-1 npz+json.
# Touches no field, so a colour/label change is a resubmit of THIS job only.
#
#   sbatch scripts/slurm/gather_figures_plot_cpu.sbatch <envfile>
#SBATCH --job-name=gfig_plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-gfig_plot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${GF_OUT:?GF_OUT must be the data stem}"
: "${GF_FIG_DIR:?GF_FIG_DIR must be the figure output dir}"

require_input "$GF_OUT.npz" "scripts/slurm/gather_figures_data_cpu.sbatch"

python -u scripts/features/gather_holdout_figures_plot.py \
    --data "$GF_OUT" \
    --out-dir "$GF_FIG_DIR"
