#!/bin/bash
# Build the HR `owner` arrays that the member-gather objective is supervised by.
#
# One throttled CPU array, one task per box:
#
#   Rockstar (member ids) --> owner.npy --> [.particles deleted in-job]
#
# WHY. docs/sr2_member_gather.md's result is on set8 because set8 is the only
# box that has an owner array -- the per-particle "which halo binds you" map that
# member sets are slices of. HR fields, LR fields, cached frozen SR2 boxes and HR
# catalogs are already on disk for every box; this is the sole missing input, and
# it is ~18 min of CPU each. Without it the line cannot leave one box, and no
# generalisation claim is available from one box.
#
# set13-15 are SEALED (configs/reward/sr2_direct_finetune.yaml `split`) and are
# NOT built here. set8 is included and short-circuits in seconds.
#
# THE THROTTLE. Each task holds ~9.6 GB of `.particles` ASCII plus a ~3.8 GB
# gadget2 before deleting them. An unthrottled Rockstar array blew the per-user
# quota once already; the signature is a truncated gadget2, an empty
# rockstar.log and a silent SIGKILL with no Python traceback. OWNER_CONC=2 by
# default; raise it only against measured free space.
#
# Resumable: a box whose owner.npy exists exits immediately, so a timed-out or
# cancelled array is simply resubmitted.
#
#   bash scripts/slurm/submit_owner_arrays.sh
#   BOXES="set3 set4" bash scripts/slurm/submit_owner_arrays.sh
#   OWNER_CONC=3 bash scripts/slurm/submit_owner_arrays.sh
#   DRY=1 bash scripts/slurm/submit_owner_arrays.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

BOXES="${BOXES:-set0 set1 set2 set3 set4 set5 set6 set7 set8 set9 set10 set11 set12}"
SOURCE="${SOURCE:-hr}"
OWNER_CONC="${OWNER_CONC:-2}"
VERIFY_FROZEN="${VERIFY_FROZEN:-1}"

read -r -a _B <<< "${BOXES//,/ }"
N=${#_B[@]}

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/owner_arrays_$(date +%Y%m%d_%H%M%S)_$$.env"
# EVERY value is quoted. The env file is SOURCED, so an unquoted list-valued
# variable is not an assignment at all: `BOXES=set0 set1 set2` is bash for "run
# the command `set1` with BOXES=set0 in its environment", which fails with
# `set1: command not found` and takes down every task in the array before the
# halo finder starts. Measured 2026-08-22, array 35593-35605, all 13 tasks.
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_owner_arrays.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument. Values are quoted
# because a sourced list-valued assignment must be.
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$REWARD_ROOT"
BOXES="$BOXES"
SOURCE="$SOURCE"
VERIFY_FROZEN="$VERIFY_FROZEN"
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

# Source it in a CLEAN shell and check the list-valued variables round-trip.
# The jobs source this file before doing anything, so a file that does not
# source is a whole-array failure discovered 13 job ids later; here it costs
# milliseconds and fails before submission.
_check=$(env -i bash -c "set -eu; source '$ENVFILE'; printf '%s|%s' \"\$BOXES\" \"\$SOURCE\"" 2>&1) || {
    echo ">>> the env file does not source cleanly:" >&2
    echo ">>> $_check" >&2
    exit 1
}
if [ "$_check" != "$BOXES|$SOURCE" ]; then
    echo ">>> env file round-trip MISMATCH -- values were mangled by sourcing." >&2
    echo ">>>   wrote: $BOXES|$SOURCE" >&2
    echo ">>>   read:  $_check" >&2
    exit 1
fi
echo "env file sources cleanly and round-trips."
echo

# What is already done, so the array size is understood before it is submitted.
echo "state before submission:"
TODO=0
for b in "${_B[@]}"; do
    p="$REWARD_ROOT/halos_particles/${b}__${SOURCE}__${SOURCE}/${b}_${SOURCE}_owner.npy"
    if [ -r "$p" ]; then printf "  %-7s owner array present (task will exit at once)\n" "$b"
    else printf "  %-7s MISSING -> will run Rockstar (~18 min)\n" "$b"; TODO=$((TODO + 1)); fi
done
echo "  $TODO of $N boxes to build; at OWNER_CONC=$OWNER_CONC that is"
echo "  roughly $(( (TODO + OWNER_CONC - 1) / OWNER_CONC * 18 )) min wall,"
echo "  peak transient $(( OWNER_CONC * 14 )) GB, permanent $(( TODO * 537 / 1024 )) GB."
echo
df -BG "$REWARD_ROOT" | tail -1
echo

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch jobs.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch --array=0-$((N - 1))%$OWNER_CONC scripts/slurm/owner_arrays_cpu.sbatch $ENVFILE"
    exit 0
fi

AID=$(sbatch --parsable --array=0-$((N - 1))%"$OWNER_CONC" \
      scripts/slurm/owner_arrays_cpu.sbatch "$ENVFILE")
echo "submitted owner-array build -> array job $AID ($N tasks, max $OWNER_CONC at once)"
echo
echo "watch:"
echo "  squeue -j $AID"
echo "  tail -F $REWARD_ROOT/logs/slurm-owner_arr-*.out"
echo "  # per-task logs are slurm-owner_arr-<task job id>.out, not -<array id>_"
echo
echo "results (one per box):"
for b in "${_B[@]}"; do
    echo "  $REWARD_ROOT/halos_particles/${b}__${SOURCE}__${SOURCE}/${b}_${SOURCE}_owner.npy"
done
echo
echo "check them all at the end with:"
echo "  ls -la $REWARD_ROOT/halos_particles/*__${SOURCE}__${SOURCE}/*_owner.npy"
echo "  # and the consistency block in each particles_report.json:"
echo "  grep -h frac_particles_unowned $REWARD_ROOT/halos_particles/*/particles_report.json"
