#!/bin/bash
# Submit the channel-swap intervention: two Rockstar runs, then the report.
#
# Which half of the SR2 field costs it the subhalos -- the displacement channels
# (0:3) or the velocity channels (3:6)? Two swapped boxes answer it directly:
#
#   srpos_hrvel   SR2 displacement + HR velocity   -> subhalos recovered?
#   hrpos_srvel   HR displacement  + SR2 velocity  -> subhalos destroyed?
#
# The two pure controls already exist and are not re-run.
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster makes sbatch set SLURM_GET_USER_ENV=1 and gets the job requeued+held.
#
#   bash scripts/slurm/submit_channel_swap.sh
#   BOX=set9 bash scripts/slurm/submit_channel_swap.sh
#   ARMS=srpos_hrvel bash scripts/slurm/submit_channel_swap.sh   # one arm only
#   REPORT_ONLY=1 bash scripts/slurm/submit_channel_swap.sh      # re-render only
#   OVERWRITE=1 bash scripts/slurm/submit_channel_swap.sh        # ignore cached catalogs
#   DRY=1 bash scripts/slurm/submit_channel_swap.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

BOX="${BOX:-set8}"
ARMS="${ARMS:-srpos_hrvel hrpos_srvel}"
BASE_SEED="${BASE_SEED:-0}"
OVERWRITE="${OVERWRITE:-0}"
REPORT_ONLY="${REPORT_ONLY:-0}"

read -r -a _A <<< "${ARMS//,/ }"
NARMS=${#_A[@]}

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/chanswap_$(date +%Y%m%d_%H%M%S)_$$.env"
# Every value is QUOTED. The preamble `source`s this file, and bash reads an
# unquoted `ARMS=srpos_hrvel hrpos_srvel` as "run the command hrpos_srvel with
# ARMS=srpos_hrvel in its environment" -- which is exactly how jobs 33292/33294
# died ("hrpos_srvel: command not found") and left the report on
# DependencyNeverSatisfied. The other submitters in this directory never hit it
# only because their list values are comma-separated and so contain no spaces.
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_channel_swap.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$REWARD_ROOT"
BOX="$BOX"
ARMS="$ARMS"
BASE_SEED="$BASE_SEED"
OVERWRITE="$OVERWRITE"
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch job.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch --array=0-$((NARMS - 1)) scripts/slurm/channel_swap_rockstar_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch [--dependency=afterok:<array>] scripts/slurm/channel_swap_report_cpu.sbatch $ENVFILE"
    exit 0
fi

AID=""
if [ "$REPORT_ONLY" != "1" ]; then
    AID=$(sbatch --array=0-$((NARMS - 1)) \
          scripts/slurm/channel_swap_rockstar_cpu.sbatch "$ENVFILE" \
          | awk '{print $NF}')
    echo "submitted chanswap array job $AID  (${NARMS} arms: ${_A[*]})"
fi

# afterok on the whole array: the report reads every arm, and an arm that died
# leaves its catalog absent. The report gates per arm and still writes a verdict
# saying which arm is missing, so this dependency is about ordering, not about
# hiding a failure.
DEP=()
if [ -n "$AID" ]; then DEP=(--dependency=afterok:"$AID"); fi
RID=$(sbatch "${DEP[@]}" scripts/slurm/channel_swap_report_cpu.sbatch \
      "$ENVFILE" | awk '{print $NF}')
echo "submitted chanswap_rep job $RID"
echo

if [ -n "$AID" ]; then
    # Each array task logs to its own job id (%j), which is not $AID, so the
    # exact filenames are not known until the tasks are allocated.
    echo "watch the ${NARMS} arms (${_A[*]}):"
    echo "  squeue -j $AID"
    echo "  tail -F \$(ls -t $REWARD_ROOT/logs/slurm-chanswap-*.out | head -$NARMS)"
fi
echo "watch report: tail -F $REWARD_ROOT/logs/slurm-chanswap_rep-$RID.out"
echo
echo "results in: $REWARD_ROOT/channel_swap/"
echo "  ${BOX}_channel_swap.md      the verdict + all four tables (read this)"
echo "  ${BOX}_channel_swap.json    the same numbers, machine-readable"
for a in "${_A[@]}"; do
    echo "  ${BOX}__${a}/${BOX}_${a}_summary.json   counts for that arm"
    echo "  ${BOX}__${a}/${a}_rockstar/             its catalog"
done
echo
echo "re-render the report alone once the arms are done:"
echo "  REPORT_ONLY=1 bash scripts/slurm/submit_channel_swap.sh"
