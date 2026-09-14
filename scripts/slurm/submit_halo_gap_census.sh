#!/bin/bash
# Is the top-level halo deficit a SMALL-HALO deficit or a size-independent one?
#
# The held-out gate (docs/sr2_member_gather_training.md section 11.4) measured,
# in the edited region: HR resolves 3,775 top-level halos >=200p, base 3,028,
# the tuned `self` arm 2,708. The 747-halo gap between HR and base sits OUTSIDE
# hosts and no supervised set in any arm has ever pointed at it -- every arm so
# far supervised `parent_ids >= 0`.
#
# That single number does not say WHICH top-level halos are missing, and the two
# readings lead to different runs:
#
#   SMALL-HALO   base/HR climbs with mass  -> supervise a num_p WINDOW:
#                MG_POPULATION=all MG_MAX_NUM_P=<top of the deficit>
#   FLAT         base/HR is constant       -> a window buys nothing; the
#                                             deficit is not about object size
#
# and the environment split says whether the missing halos are even REACHABLE
# from the cluster tiles the fine-tune trains on, which is a --n-tiles question
# rather than a --population one.
#
# One CPU job: parse catalogs -> census.json -> figures -> printed verdict.
#
#   bash scripts/slurm/submit_halo_gap_census.sh
#   HG_BOX=set10 bash scripts/slurm/submit_halo_gap_census.sh
#   # include a tuned arm, and restrict to the region it actually edited:
#   HG_KINDS=base,hr,self \
#     HG_ARM_TAGS=self=mgho_all_blocks_self_set9 \
#     HG_RUN_DIR=$REWARD/member_gather/all_blocks_self/holdout_set9 \
#     bash scripts/slurm/submit_halo_gap_census.sh
#   DRY=1 bash scripts/slurm/submit_halo_gap_census.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

HG_BOX="${HG_BOX:-set9}"
HG_KINDS="${HG_KINDS:-base,hr}"
HG_ARM_TAGS="${HG_ARM_TAGS:-}"
HG_RUN_DIR="${HG_RUN_DIR:-}"
HG_DILATE="${HG_DILATE:-2.0}"
HG_HOST_FLOOR="${HG_HOST_FLOOR:-200}"
HG_OUT_DIR="${HG_OUT_DIR:-$REWARD_ROOT/halo_gap_census/${HG_BOX}}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$HG_OUT_DIR"

# The catalogs are the whole input. A missing one is a 30-second failure inside
# the job, but it is a zero-second failure here.
echo "catalogs:"
_MISSING=0
for k in $(echo "$HG_KINDS" | tr ',' ' '); do
    case "$k" in
        base) pat="$REWARD_ROOT/halos/${HG_BOX}__base__base/*_rockstar/halos_*.ascii" ;;
        hr)   pat="$REWARD_ROOT/halos/${HG_BOX}__hr__hr/*_rockstar/halos_*.ascii" ;;
        *)    tag=$(echo "$HG_ARM_TAGS" | tr ',' '\n' | grep "^${k}=" | cut -d= -f2- || true)
              if [ -z "$tag" ]; then
                  echo ">>> no tag for arm '$k' in HG_ARM_TAGS" >&2; exit 1
              fi
              pat="$REWARD_ROOT/flow_rockstar/halos/${HG_BOX}__candidate__${tag}/*/halos_*.ascii" ;;
    esac
    # shellcheck disable=SC2086
    if compgen -G $pat > /dev/null; then printf "  %-6s ok\n" "$k"
    else printf "  %-6s MISSING: %s\n" "$k" "$pat"; _MISSING=$((_MISSING + 1)); fi
done
if [ "$_MISSING" -gt 0 ]; then
    echo ">>> $_MISSING catalog(s) missing; the job would fail. Aborting." >&2
    exit 1
fi
echo

ENVFILE="$REWARD_ROOT/env/halo_gap_census_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_halo_gap_census.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$REWARD_ROOT"
HG_BOX="$HG_BOX"
HG_KINDS="$HG_KINDS"
HG_ARM_TAGS="$HG_ARM_TAGS"
HG_RUN_DIR="$HG_RUN_DIR"
HG_DILATE="$HG_DILATE"
HG_HOST_FLOOR="$HG_HOST_FLOOR"
HG_OUT_DIR="$HG_OUT_DIR"
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

_check=$(env -i bash -c "set -eu; source '$ENVFILE'; printf '%s' \"\$HG_KINDS\"" 2>&1) || {
    echo ">>> the env file does not source cleanly:" >&2; echo ">>> $_check" >&2; exit 1; }
[ "$_check" = "$HG_KINDS" ] || { echo ">>> env file round-trip MISMATCH" >&2; exit 1; }
echo "env file sources cleanly and round-trips."
echo

if [ -z "$HG_RUN_DIR" ]; then
    echo "NOTE: no HG_RUN_DIR, so this is BOX-WIDE only. That is the right"
    echo "  scope for base-vs-HR (both are whole boxes). It is the WRONG scope"
    echo "  for a tuned arm: the splice touches 6.25% of the box, so 93.75% of"
    echo "  a box-wide arm-vs-HR comparison is untouched frozen SR2."
    echo
fi

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/halo_gap_census_cpu.sbatch $ENVFILE"; exit 0
fi
JID=$(sbatch --parsable scripts/slurm/halo_gap_census_cpu.sbatch "$ENVFILE")
echo "submitted halo_gap_census -> job $JID (CPU)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-halogap-$JID.out"
echo "results: $HG_OUT_DIR/census.json"
echo "figures: $HG_OUT_DIR/halo_gap_census_box.png"
[ -n "$HG_RUN_DIR" ] && echo "         $HG_OUT_DIR/halo_gap_census_region.png"
echo
echo "READ THE VERDICT BEFORE CHOOSING MG_MAX_NUM_P for the small-halo arm:"
echo "  the ceiling IS the hypothesis, and the census is what sets it."
