#!/bin/bash
# Report the gather result over ALL hosts of ALL gated boxes, not one host of one.
#
#   per-box abundance (CPU, siblings, ~10 min each)
#     --afterok--> pooled cross-box table + slide LaTeX (CPU, seconds)
#
# The held-out gate's headline so far is one host: 20 -> 366 vs HR 369 on set9's
# host 168880, and 14 -> 356 vs 333 on set10's 216377. Two hosts is an anecdote,
# and the region totals it is quoted beside answer a different question (they
# count field substructure too). This counts subhalos inside EVERY HR host of
# the edited region, on every box that has a gated catalog, and states the host
# count with the ratio so the percentage can be read.
#
# Only calls sbatch. Config travels in ONE timestamped env file per stage passed
# as a POSITIONAL argument -- never `sbatch --export`, which requeues+holds.
#
#   bash scripts/slurm/submit_gather_all_hosts.sh
#   AH_BOXES="set9 set10 set3" bash scripts/slurm/submit_gather_all_hosts.sh
#   AH_ARMS="self=mgho_all_blocks_self_set9" bash scripts/slurm/submit_gather_all_hosts.sh
#   AH_MAX_HOSTS=50 bash scripts/slurm/submit_gather_all_hosts.sh   # shakeout
#   DRY=1 bash scripts/slurm/submit_gather_all_hosts.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

# Every box with a gated candidate catalog. set9/set10 are held out; set3 is a
# TRAIN box and is left out of the default pool -- pooling in-sample counts into
# a held-out number would overstate the result.
AH_BOXES="${AH_BOXES:-set9 set10}"
# Empty discovers every gated candidate catalog of each box, which is what the
# report wants: one row per arm, and arms that were never gated on a box simply
# do not appear for it.
AH_ARMS="${AH_ARMS:-}"
AH_DILATE="${AH_DILATE:-2.0}"
AH_HOST_MIN_P="${AH_HOST_MIN_P:-200}"
AH_SUB_MIN_P="${AH_SUB_MIN_P:-50}"
AH_MATCH_RVIR="${AH_MATCH_RVIR:-1.0}"
AH_MAX_HOSTS="${AH_MAX_HOSTS:-0}"
# The frozen control spliced on the SAME edges is the honest SR2 baseline; the
# unspliced base catalog is the fallback when a box has no frozen control.
AH_BASELINE="${AH_BASELINE:-frozen}"
AH_OUT_DIR="${AH_OUT_DIR:-$REWARD_ROOT/member_gather/all_hosts}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$AH_OUT_DIR"

_probe="$REWARD_ROOT/env/.writable_$$"
if ! (echo probe > "$_probe") 2>/dev/null; then
    echo "cannot write to $REWARD_ROOT/env -- disk quota is full." >&2
    echo "  free space first:" >&2
    echo "    python scripts/reward/purge_regenerable_fields.py" >&2
    exit 1
fi
rm -f "$_probe"

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

sub() {
    if [ "$DRY" = "1" ]; then echo "DRY: sbatch $*" >&2; echo "DRYID"
    else sbatch --parsable "$@"; fi
}

STAMP="$(date +%Y%m%d_%H%M%S)"
WATCH=(); DEPS=(); USED_BOXES=""

for box in $AH_BOXES; do
    # Any gated export dir of this box fixes the spliced tile list: the pool
    # selects the same tiles for the same box, so every arm's export.json
    # carries the same list. Prefer the role dir the gate names for a held-out
    # box, then fall back to a train-role one.
    RUN_DIR=""
    for d in "$REWARD_ROOT"/member_gather/*/holdout_"$box" \
             "$REWARD_ROOT"/member_gather/*/train_"$box"; do
        [ -f "$d/export.json" ] && { RUN_DIR="$d"; break; }
    done
    if [ -z "$RUN_DIR" ]; then
        echo "skipping $box: no export.json under $REWARD_ROOT/member_gather/*/"
        echo "  -- that box has not been gated. Run"
        echo "     HG_BOX=$box bash scripts/slurm/submit_gather_holdout_rockstar.sh"
        continue
    fi

    ENVFILE="$REWARD_ROOT/env/all_hosts_${STAMP}_${box}_$$.env"
    cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_gather_all_hosts.sh at $(date '+%F %T');
# sourced by the job preambles as a positional argument. List values are
# QUOTED: an unquoted \`AH_BOXES=set9 set10\` in a sourced file runs \`set10\`
# as a command.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
AH_BOX=$box
AH_RUN_DIR=$RUN_DIR
AH_ARMS="$AH_ARMS"
AH_DILATE=$AH_DILATE
AH_HOST_MIN_P=$AH_HOST_MIN_P
AH_SUB_MIN_P=$AH_SUB_MIN_P
AH_MATCH_RVIR=$AH_MATCH_RVIR
AH_MAX_HOSTS=$AH_MAX_HOSTS
AH_OUT_DIR=$AH_OUT_DIR
EOT
    if [ ! -s "$ENVFILE" ]; then
        echo "FAILED to write the env file $ENVFILE" >&2
        echo "  (disk quota? scripts/reward/purge_regenerable_fields.py)" >&2
        exit 1
    fi

    J=$(sub scripts/slurm/gather_all_hosts_cpu.sbatch "$ENVFILE")
    echo "box $box  tiles from $RUN_DIR"
    echo "    abundance -> job $J"
    WATCH+=("  $box: tail -F $REWARD_ROOT/logs/slurm-ah_abund-$J.out")
    DEPS+=("$J")
    USED_BOXES="$USED_BOXES $box"
done

if [ ${#DEPS[@]} -eq 0 ]; then
    echo "nothing to submit -- no gated box among: $AH_BOXES" >&2
    exit 1
fi

REPORT_ENV="$REWARD_ROOT/env/all_hosts_${STAMP}_report_$$.env"
cat > "$REPORT_ENV" <<EOT
# Written by scripts/slurm/submit_gather_all_hosts.sh at $(date '+%F %T').
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
AH_BOXES="${USED_BOXES# }"
AH_BASELINE=$AH_BASELINE
AH_OUT_DIR=$AH_OUT_DIR
EOT
DEPLIST=$(IFS=:; echo "${DEPS[*]}")
# `afterok` on every box: a box whose catalogs are missing exits 0 from the gate
# branch, so the pool still runs and simply reports one box fewer.
RJ=$(sub --dependency=afterok:"$DEPLIST" \
     scripts/slurm/gather_all_hosts_report_cpu.sbatch "$REPORT_ENV")
echo
echo "report -> job $RJ (afterok:$DEPLIST)"
WATCH+=("  report: tail -F $REWARD_ROOT/logs/slurm-ah_report-$RJ.out")

echo
echo "watch:"
printf '%s\n' "${WATCH[@]}"
echo
echo "results:"
for box in $USED_BOXES; do
    echo "  $box:     $AH_OUT_DIR/$box.json  (+ .jsonl, one row per host)"
done
echo "  pooled:   $AH_OUT_DIR/report.json"
echo "  slides:   $AH_OUT_DIR/report.tex   (paste into docs/subhalo_finetune_slides.tex)"
echo
echo "read, in this order:"
echo "  n_hosts               how many hosts the percentage is over -- state it in the talk"
echo "  subs_total / frac_of_hr   the abundance claim, summed over all of them"
echo "  hosts_alive_frac      the host-survival cost, on HR's own host sample"
echo "  by_host_mass          the occupation-vs-host-mass table (Result 2)"
