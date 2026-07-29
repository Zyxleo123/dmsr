#!/bin/bash
# Experiment 0 (CPU array): Rockstar with member-particle ids -> per-tile weights.
#
# One task per (box, source). Runs the halo finder, streams the ~7 GB
# .particles table into a few-MB npz, and DELETES the ASCII in the same job --
# nothing bulky survives the allocation, and nothing goes near $HOME.
#
# The first HR task of each Experiment-1 dev box also extracts the member ids of
# the selected targets and their hosts (EXTRACT=1), so Experiment 1's masks come
# out of the same pass rather than costing a second halo-finder run.
#
# Submittable EITHER as an array (BOX/SRC come from SLURM_ARRAY_TASK_ID against
# BOXES x SOURCES) OR as one job per pair with a single-element BOXES/SOURCES --
# submit_oracle.sh uses the latter, because a whole array can be cancelled with
# one `scancel <array-id>`, which is exactly what took out every (box, source)
# pair simultaneously the one time this happened in practice. Separate job ids
# confine that blast radius to one pair.
#
#   sbatch --array=0-1 scripts/slurm/exp0_rockstar_particles_cpu.sbatch
#   sbatch --array=0-3 scripts/slurm/exp0_rockstar_particles_cpu.sbatch \
#          BOXES="set8 set9" SOURCES="hr base"
#   sbatch scripts/slurm/exp0_rockstar_particles_cpu.sbatch BOXES=set8 SOURCES=hr
#
#SBATCH --job-name=exp0_parts
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
# %j (this task's own job id), not %A_%a (the array's id + index): %j is
# unique per array task too, and it is also correct for the non-array,
# one-job-per-pair submission that submit_oracle.sh actually uses.
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-exp0_parts-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8 set9}"
: "${SOURCES:=hr base}"
: "${EXTRACT:=1}"
: "${VERIFY_FROZEN:=1}"

read -r -a _B <<< "${BOXES//,/ }"
read -r -a _S <<< "${SOURCES//,/ }"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
NS=${#_S[@]}
BOX="${_B[$((IDX / NS))]}"
SRC="${_S[$((IDX % NS))]}"

require_input "external/rockstar/rockstar" "make -C external/rockstar"
require_input "configs/sr2_baseline/rockstar_particles.cfg" "this repository"
if [ "$SRC" = "base" ]; then
    require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"
fi

ARGS=(--config "$REWARD_CFG" --box "$BOX" --source "$SRC" --base-seed "$BASE_SEED")
[ "$VERIFY_FROZEN" = "1" ] && ARGS+=(--verify-frozen)

# Experiment 1 needs the Lagrangian sites of its targets; extracting them here
# costs one extra pass over an already-open file instead of a second Rockstar run.
IDS="$DMSR_REWARD_ROOT/oracle_hr/halo_ids_${BOX}.json"
if [ "$SRC" = "hr" ] && [ "$EXTRACT" = "1" ] && [ -r "$IDS" ]; then
    echo "=== will extract member ids listed in $IDS"
    ARGS+=(--extract-members "$IDS")
elif [ "$SRC" = "hr" ] && [ "$EXTRACT" = "1" ]; then
    echo ">>> no $IDS yet: run oracle_select_targets.py BEFORE this job if you"
    echo ">>> want Experiment 1's masks from this pass. Continuing with tile"
    echo ">>> weights only (Experiment 0 is unaffected)."
fi

echo "=== $BOX / $SRC"
python -u scripts/reward/rockstar_particles.py "${ARGS[@]}"

echo "=== tile weights + summaries done for $BOX/$SRC"
