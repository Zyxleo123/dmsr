#!/bin/bash
# Redraw the moment-target verification figure from the build job's artifacts.
# Pure redraw: reads <box>_moment_target_diag.npz + _summary.json and writes a
# PNG. No field, no generator -- so it is CPU and cheap, and gates itself out
# (exit 0) if the build job did not produce its inputs, rather than stranding a
# dependent with a nonzero exit.
#
#   sbatch scripts/slurm/render_moment_target_cpu.sbatch <envfile>
#SBATCH --job-name=momtgtfig
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-momtgtfig-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${MT_BOX:=set8}"
DIAG="$DMSR_REWARD_ROOT/moment_target/$MT_BOX/${MT_BOX}_moment_target_diag.npz"

if [[ ! -r "$DIAG" ]]; then
    echo "GATE: no diagnostics at $DIAG -- the build job did not finish. "\
"Nothing to draw; exiting 0 so this does not strand a dependent." >&2
    exit 0
fi

python -u scripts/features/render_moment_target.py --box "$MT_BOX"
echo "=== result: $DMSR_REWARD_ROOT/moment_target/$MT_BOX/${MT_BOX}_moment_target.png"
