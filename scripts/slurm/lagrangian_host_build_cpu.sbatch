#!/bin/bash
# Build the LR-Rockstar Lagrangian host features for one or more boxes.
#
# CPU only, and small: the halo finder runs on the LR box (64^3 = 262144
# particles, ~20 MB member table), not the HR one, so this is minutes rather
# than the hours scripts/slurm/exp0_rockstar_particles_cpu.sbatch needs. No GPU,
# no training, no HR field is read at all.
#
#   sbatch scripts/slurm/lagrangian_host_build_cpu.sbatch <envfile>
#SBATCH --job-name=lag_host
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-lag_host-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8}"
: "${REDSHIFT:=0.0}"
: "${RERUN_ROCKSTAR:=0}"

EXTRA=()
[ "$RERUN_ROCKSTAR" = "1" ] && EXTRA+=(--rerun-rockstar)

# The mapping tests are the contract this build has to satisfy; running them in
# the job means a broken build is reported here rather than in the viewer.
python -u -m pytest tests/features -q

python -u scripts/features/build_lagrangian_host.py \
    --boxes "$BOXES" --redshift "$REDSHIFT" "${EXTRA[@]}"

echo "=== features + report: $DMSR_REWARD_ROOT/lagrangian_host/<box>/"
