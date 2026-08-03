#!/bin/bash
# Stage 2 (CPU, ~1-2 h/box): Rockstar member-id pass on the frozen SR2 box, and
# the reduction to a per-box pool cache.
#
# This is the expensive one-off: it re-runs the halo finder with
# FULL_PARTICLE_CHUNKS=1, which writes a ~7 GB ASCII member table, streams it,
# and deletes it inside this job. Output is a few MB.
#
#   sbatch --array=0-1 scripts/slurm/local_editor_members_cpu.sbatch RUN_NAME=le_a
#SBATCH --job-name=le_members
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_local_editor/logs/slurm-le_members-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_local_editor_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_local_editor_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8 set9}"
read -r -a _BOXES <<< "${BOXES//,/ }"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#_BOXES[@]}" ]; then
    echo ">>> array index $IDX is beyond ${#_BOXES[@]} boxes; nothing to do."
    exit 0
fi
BOX="${_BOXES[$IDX]}"
echo "=== box $BOX"

require_input "external/rockstar/rockstar" "make -C external/rockstar"
require_input "$DMSR_LOCAL_EDITOR_ROOT/runs/$RUN_NAME/hosts/hosts_$BOX.json" \
    "scripts/slurm/local_editor_select_hosts_cpu.sbatch"

python -u scripts/reward/extract_editor_members.py \
    --config "$LE_CFG" --run-name "$RUN_NAME" --box "$BOX" \
    --reward-config "$REWARD_CFG" --base-seed "$BASE_SEED"
