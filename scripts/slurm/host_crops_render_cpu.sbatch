#!/bin/bash
# Draw the crop page from the cached crop artifacts. Reads nothing else --
# no field, no owner array, no catalog -- so it is cheap and rerunnable.
#
#   sbatch scripts/slurm/host_crops_render_cpu.sbatch <envfile>
#SBATCH --job-name=crops_view
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-crops_view-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8}"
: "${CROP_VIEW_HOSTS:=0}"     # 0 = every host the collector kept

for b in ${BOXES//,/ }; do
    J="$DMSR_REWARD_ROOT/lagrangian_host/$b/${b}_host_crops.json"
    if [ ! -f "$J" ]; then
        echo "GATE: $b has no crop artifacts ($J)."
        echo "GATE: produced by scripts/slurm/host_crops_cpu.sbatch -- skipping."
        continue
    fi
    python -u scripts/features/render_host_crops_app.py \
        --boxes "$b" --n-hosts "$CROP_VIEW_HOSTS"
done
