#!/bin/bash
# Build the HR `owner` array for one box: the catalog id of the object each
# particle is bound to, indexed by flat Lagrangian id (CPU array task).
#
# WHY THIS EXISTS AS ITS OWN JOB. The member-gather objective
# (docs/sr2_member_gather.md) is supervised by HR member SETS, and a member set
# is exactly a slice of this array. Every other input the trainer needs -- HR
# fields, LR fields, cached frozen SR2 boxes, HR Rockstar catalogs -- is already
# on disk for all 16 boxes. The owner array is the ONLY missing artifact, and it
# existed for set8 alone, which is why the whole line was stuck on one box.
#
# Cost, measured from the two existing particles_report.json files (set8, set9):
# Rockstar 15.4-16.8 min + owner stream 1.8 min + tile stream 1.6 min, ~9.6 GB
# of `.particles` ASCII held transiently and DELETED inside the same job, 537 MB
# of owner.npy kept.
#
# THE THROTTLE IS NOT OPTIONAL. Each task writes a ~3.8 GB gadget2 file before
# the halo finder and ~9.6 GB of ASCII after it, deleting both only at the end.
# A previous unthrottled Rockstar array blew the per-user disk quota, which
# surfaces as a TRUNCATED gadget2, an EMPTY rockstar.log and a silent SIGKILL --
# a failure that reads as a mysterious FAILED with no traceback. The submitter
# fires this with `%OWNER_CONC` (default 2). Do not raise it without checking
# free space against 9.6 GB x concurrency.
#
# Resumable and idempotent: rockstar_particles.py short-circuits on a box whose
# owner.npy already exists, so a task for a finished box exits in seconds and a
# timed-out array is resubmitted, not restarted. set8 is deliberately left in the
# default box list for exactly that reason -- it costs nothing and keeps the
# array indices aligned with set<N>.
#
# NOTE ON `scancel`: this is an array, so one scancel takes out every task. That
# is the accepted trade for the throttle, which the quota demands.
#
#   sbatch --array=0-12%2 scripts/slurm/owner_arrays_cpu.sbatch <envfile>
#   sbatch --array=3-3    scripts/slurm/owner_arrays_cpu.sbatch BOXES=set3
#
#SBATCH --job-name=owner_arr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
# %j, not %A_%a: %j is unique per array task and is also correct if this is ever
# submitted as a single non-array job for one box.
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-owner_arr-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

# set13-15 are SEALED (configs/reward/sr2_direct_finetune.yaml `split`) and are
# deliberately absent. Adding them here would spend the final-evidence boxes.
: "${BOXES:=set0 set1 set2 set3 set4 set5 set6 set7 set8 set9 set10 set11 set12}"
: "${SOURCE:=hr}"
: "${VERIFY_FROZEN:=1}"

read -r -a _B <<< "${BOXES//,/ }"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#_B[@]}" ]; then
    echo ">>> array index $IDX beyond BOXES (${#_B[@]} entries); nothing to do."
    exit 0
fi
BOX="${_B[$IDX]}"

require_input "external/rockstar/rockstar" "make -C external/rockstar"
require_input "configs/sr2_baseline/rockstar_particles.cfg" "this repository"
require_input "$DATA/hr/$BOX.npy" "scripts/reward/preprocess (paired_catnorm)"

OWNER="$DMSR_REWARD_ROOT/halos_particles/${BOX}__${SOURCE}__${SOURCE}/${BOX}_${SOURCE}_owner.npy"
if [ -r "$OWNER" ]; then
    echo "=== $BOX: owner array already present, nothing to do"
    echo "    $OWNER"
    ls -la "$OWNER"
    exit 0
fi

# Free space, checked before spending 16 minutes of halo finder on a write that
# cannot land. 9.6 GB ASCII + 3.8 GB gadget2 per concurrent task; 40 GB is a
# deliberately loose floor that still catches a genuinely full filesystem.
AVAIL_GB=$(df -BG --output=avail "$DMSR_REWARD_ROOT" | tail -1 | tr -dc '0-9')
echo "=== $BOX: ${AVAIL_GB}G free under $DMSR_REWARD_ROOT"
if [ "${AVAIL_GB:-0}" -lt 40 ]; then
    echo ">>> under 40G free: refusing to start. A truncated gadget2 from a full"
    echo ">>> filesystem produces an empty rockstar.log and a silent SIGKILL,"
    echo ">>> which is far more expensive to diagnose than this exit."
    exit 1
fi

ARGS=(--config "$REWARD_CFG" --box "$BOX" --source "$SOURCE"
      --base-seed "$BASE_SEED" --write-assignment)
# A frozen-catalog disagreement means the member-id output flag changed the halo
# finding, which would invalidate every earlier number in the project. That is a
# finding, not a nuisance: let it exit non-zero and be seen.
[ "$VERIFY_FROZEN" = "1" ] && ARGS+=(--verify-frozen)

echo "=== $BOX / $SOURCE: Rockstar with member ids -> owner array"
echo "    ~18 min expected; .particles ASCII is deleted inside this job"
python -u scripts/reward/rockstar_particles.py "${ARGS[@]}"

echo "=== $BOX done"
ls -la "$OWNER" || { echo ">>> owner array was NOT written"; exit 1; }
