#!/bin/bash
# Stage 2 (CPU): figures for lr_host_fidelity, from the saved npz alone.
# Separate from stage 1 so a re-plot never re-runs a halo finder.
#
#   sbatch scripts/slurm/lr_host_fidelity_plot_cpu.sbatch <envfile>
#SBATCH --job-name=lr_host_fid_plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-lr_host_fid_plot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${LHF_OUT_DIR:?LHF_OUT_DIR must be the output directory}"

# A gate, not a crash: if stage 1 produced nothing, say so and exit 0 rather
# than stranding anything chained behind this.
if [ ! -e "$LHF_OUT_DIR/summary.json" ]; then
    gate_failed "no $LHF_OUT_DIR/summary.json -- stage 1 did not write results"
fi

python -u scripts/features/lr_host_fidelity_plot.py --out-dir "$LHF_OUT_DIR"
echo "=== figures -> $LHF_OUT_DIR"
ls -la "$LHF_OUT_DIR"
