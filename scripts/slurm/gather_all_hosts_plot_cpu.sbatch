#!/bin/bash
# Talk figures for the all-hosts abundance result + host forensic. Reads only
# the saved report.json and host_forensics_*.json -- no catalog, no field -- so
# it is seconds and re-renders on any colour/label change (redrawable rule).
#
#   sbatch scripts/slurm/gather_all_hosts_plot_cpu.sbatch <envfile>
#SBATCH --job-name=ah_plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-ah_plot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${AHP_REPORT:?AHP_REPORT must be the pooled report.json}"
: "${AHP_OUT:?AHP_OUT must be the figure output dir}"
: "${AHP_FORENSICS:=}"   # space-separated arm=path pairs, optional
: "${AHP_RADIAL_DIR:=}"  # subhalo_radial/<box> dir with hr.npz + self.npz, optional
: "${AHP_TUNED_ARM:=all_blocks_self}"  # arm drawn as "tuned" in the mass-function bars

require_input "$AHP_REPORT" "scripts/slurm/gather_all_hosts_report_cpu.sbatch"

python -u scripts/features/gather_all_hosts_plot.py \
    --report "$AHP_REPORT" \
    ${AHP_FORENSICS:+--forensics $AHP_FORENSICS} \
    ${AHP_RADIAL_DIR:+--radial-dir "$AHP_RADIAL_DIR"} \
    --tuned-arm "$AHP_TUNED_ARM" \
    --out "$AHP_OUT"

echo "=== figures -> $AHP_OUT/"
