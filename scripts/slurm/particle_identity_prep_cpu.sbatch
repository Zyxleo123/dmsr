#!/bin/bash
# Stage A of the SR2-vs-HR particle identity study (CPU array).
#
# Runs Rockstar with FULL_PARTICLE_CHUNKS=1 on one (box, source) and streams the
# ~10 GB .particles table into a per-particle owner array -- the catalog id of
# the object each Lagrangian particle is bound to. That array is what makes the
# identity comparison exact, and it is the ONLY thing that survives: the ASCII
# table is deleted inside the same job, as in Experiment 0.
#
# One task per (box, source). Separate job ids rather than one array where
# possible, so a stray scancel cannot take out every pair at once.
#
#   sbatch --array=0-1 scripts/slurm/particle_identity_prep_cpu.sbatch
#   sbatch scripts/slurm/particle_identity_prep_cpu.sbatch BOXES=set8 SOURCES=hr
#   # the noise-only control: the same generator, a different sampling seed
#   sbatch scripts/slurm/particle_identity_prep_cpu.sbatch \
#          BOXES=set8 SOURCES=base BASE_SEED=1
#
#SBATCH --job-name=pid_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-pid_prep-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8}"
: "${SOURCES:=hr base}"

read -r -a _B <<< "${BOXES//,/ }"
read -r -a _S <<< "${SOURCES//,/ }"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
NS=${#_S[@]}
BOX="${_B[$((IDX / NS))]}"
SRC="${_S[$((IDX % NS))]}"

# Seed 0 keeps the plain tag so the existing Experiment-0 work directory is
# reused; other seeds are the noise-only control and get their own directory.
if [ "$SRC" = "hr" ] || [ "${BASE_SEED:-0}" = "0" ]; then
    TAG="$SRC"
else
    TAG="${SRC}_seed${BASE_SEED}"
fi

require_input "external/rockstar/rockstar" "make -C external/rockstar"
require_input "configs/sr2_baseline/rockstar_particles.cfg" "this repository"
if [ "$SRC" = "base" ]; then
    # Check the specific seed, not just the cache directory: the seed control
    # is the one input most likely to be missing, and a directory-level gate
    # would let the job start and fail 20 minutes into halo finding.
    HIT=$(ls "$DMSR_REWARD_ROOT/cache/sr2_base/${BOX}_seed${BASE_SEED:-0}"_*.npy 2>/dev/null | head -1 || true)
    if [ -z "$HIT" ]; then
        echo ">>> MISSING INPUT: no cached SR2 field for $BOX seed ${BASE_SEED:-0}"
        echo ">>> produced by: scripts/slurm/cache_sr2_base.sbatch"
        echo ">>> skipping (exit 0 so dependent jobs report the same instead of"
        echo ">>> sitting in DependencyNeverSatisfied with no explanation)."
        exit 0
    fi
    echo "    SR2 field: $HIT"
fi

# Reusing a work directory whose .particles table an earlier Experiment-0 run
# deleted forces the halo finder to run again and rebuild the directory. That is
# unavoidable (the member table is the input to this whole study), so verify by
# default that the rerun reproduced the frozen catalog object-for-object --
# a difference there would invalidate every earlier number, not just this one.
#
# Only for the frozen configuration itself. The frozen catalog is keyed by
# source alone, so verifying a seed-1 run against it would compare two
# different noise realisations and report a spurious mismatch.
if [ "$TAG" = "$SRC" ]; then : "${VERIFY_FROZEN:=1}"; else : "${VERIFY_FROZEN:=0}"; fi
ARGS=(--config "$REWARD_CFG" --box "$BOX" --source "$SRC" --tag "$TAG"
      --base-seed "${BASE_SEED:-0}" --write-assignment)
[ "$VERIFY_FROZEN" = "1" ] && ARGS+=(--verify-frozen)

echo "=== $BOX / $SRC (seed ${BASE_SEED:-0}, tag $TAG): owner assignment"
python -u scripts/reward/rockstar_particles.py "${ARGS[@]}"

OWNER="$DMSR_REWARD_ROOT/halos_particles/${BOX}__${SRC}__${TAG}/${BOX}_${TAG}_owner.npy"
echo "=== done: $OWNER"
ls -la "$OWNER"
