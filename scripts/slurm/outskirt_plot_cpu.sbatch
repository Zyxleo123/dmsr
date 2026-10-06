#!/bin/bash
# Redraw the outskirt-detection figures from the probe's JSON. Seconds.
#
# Separate from the probe on purpose: the numbers are written once and the
# panels are re-rendered as often as the reading changes, without touching a
# catalog. Runs on the CPU partition; asking for a GPU to draw a figure only
# queues behind real work.
#
#   sbatch scripts/slurm/outskirt_plot_cpu.sbatch <envfile>
#SBATCH --job-name=outplot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-outplot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${OD_BOX:=set9}"
: "${OD_ARMS:=}"
: "${OD_ADDRESS_TOL:=0.30}"

PROBE_DIR="$DMSR_REWARD_ROOT/outskirt_probe/$OD_BOX"
if ! ls "$PROBE_DIR"/*.json >/dev/null 2>&1; then
    echo ">>> GATE: no probe JSON under $PROBE_DIR."
    echo ">>> The probe stage was skipped or wrote nothing. Exiting 0."
    exit 0
fi

python -u scripts/features/outskirt_detection_plot.py \
    --probe-dir "$PROBE_DIR" \
    --address-tol "$OD_ADDRESS_TOL" \
    ${OD_ARMS:+--arms $OD_ARMS}

echo "=== figures: $PROBE_DIR/fig{1_radial,2_address,3_contrast}.png"
