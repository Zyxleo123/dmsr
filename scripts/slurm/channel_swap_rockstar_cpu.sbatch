#!/bin/bash
# Channel-swap intervention (CPU array): one Rockstar run per swapped field.
#
# Assembles a (6,512,512,512) field from one source's displacement channels and
# the other's velocity channels, then runs the FROZEN halo finder on it. Asks
# the finder directly which half of the SR2 field costs it the subhalos.
#
#   task 0  srpos_hrvel   SR2 displacement + HR velocity
#   task 1  hrpos_srvel   HR displacement  + SR2 velocity
#
# The two pure controls are NOT re-run: the frozen hr/base catalogs came from
# exactly these two fields, and rockstar.cfg vs rockstar_particles.cfg is already
# verified to agree object-by-object (particles_report.json, verify_frozen).
#
#   sbatch --array=0-1 scripts/slurm/channel_swap_rockstar_cpu.sbatch
#   sbatch --array=0-1 scripts/slurm/channel_swap_rockstar_cpu.sbatch BOX=set9
#   sbatch scripts/slurm/channel_swap_rockstar_cpu.sbatch ARMS=srpos_hrvel
#
#SBATCH --job-name=chanswap
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=8:00:00
# %j, not %A_%a: unique per array task and also correct for a single-arm
# submission with no array at all.
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-chanswap-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOX:=set8}"
: "${ARMS:=srpos_hrvel hrpos_srvel}"
: "${OVERWRITE:=0}"

read -r -a _A <<< "${ARMS//,/ }"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#_A[@]}" ]; then
    echo ">>> array index $IDX past the end of ARMS=(${_A[*]}); nothing to do."
    exit 0
fi
ARM="${_A[$IDX]}"

require_input "external/rockstar/rockstar" "make -C external/rockstar"
require_input "configs/sr2_baseline/rockstar.cfg" "this repository"
require_input "$DATA/hr/$BOX.npy" "the paired_catnorm dataset"
require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"

ARGS=(--config "$REWARD_CFG" --box "$BOX" --arm "$ARM" --base-seed "$BASE_SEED")
[ "$OVERWRITE" = "1" ] && ARGS+=(--overwrite)

echo "=== $BOX / arm $ARM  (task $IDX of ${#_A[@]})"
python -u scripts/reward/channel_swap_rockstar.py "${ARGS[@]}"

echo "=== arm $ARM done for $BOX"
echo "=== catalog under $DMSR_REWARD_ROOT/channel_swap/${BOX}__${ARM}/"
