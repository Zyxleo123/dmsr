#!/bin/bash
# Re-run ONLY the final A/B stage of the held-out gate, against a baseline arm.
#
# Why this exists. `submit_gather_holdout_rockstar.sh` runs
# export -> splice -> Rockstar -> compare -> verdict, and its verdict stage does
# the A/B only when HG_BASELINE_ARM is set. That knob defaults to EMPTY, so an
# arm gated without it gets `region_counts.json` -- the edited-region counts and
# the mass function, which are real results -- but NOT the two questions that
# decide whether the arm beats the one it is trying to improve on:
#
#   Q1  did subhalo abundance DETERIORATE from the baseline?
#   Q2  was the baseline's HOST DAMAGE recovered?
#
# Re-running the whole chain to recover those would redo the splice and a
# full-box Rockstar for a stage that reads ASCII catalogs and counts. Both
# catalogs are already on disk by then, so this submits the verdict stage alone:
# ~30 s of CPU per box.
#
# It is also the right tool whenever an arm was gated BEFORE its baseline was --
# the baseline's catalog simply did not exist yet, and nothing about the arm
# needs recomputing once it does.
#
#   bash scripts/slurm/submit_gate_ab.sh
#   AB_ARM=all_blocks_selfshell AB_BASELINE=all_blocks_self \
#     bash scripts/slurm/submit_gate_ab.sh
#   AB_BOXES=set9 bash scripts/slurm/submit_gate_ab.sh
#   DRY=1 bash scripts/slurm/submit_gate_ab.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

AB_ARM="${AB_ARM:-all_blocks_smallhalo}"
AB_BASELINE="${AB_BASELINE:-all_blocks_selfhostguard}"
AB_BASELINE_NAME="${AB_BASELINE_NAME:-$(echo "$AB_BASELINE" | sed 's/^all_blocks_//')}"
AB_BOXES="${AB_BOXES:-set9 set10}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

sub() {
    if [ "$DRY" = "1" ]; then echo "DRY: sbatch $*" >&2; echo "DRYID"
    else sbatch --parsable "$@"; fi
}

WATCH=(); RESULTS=(); NSUB=0

for box in $AB_BOXES; do
    RUN_DIR="$REWARD_ROOT/member_gather/$AB_ARM/holdout_$box"
    ARM_TAG="mgho_${AB_ARM}_${box}"
    BASE_TAG="mgho_${AB_BASELINE}_${box}"

    # Three preconditions, each a zero-second check here and a wasted job later.
    if [ ! -f "$RUN_DIR/export.json" ]; then
        echo "  $box: no export.json under $RUN_DIR -- the arm was never gated on"
        echo "        this box. Run submit_gather_holdout_rockstar.sh first. SKIPPED."
        continue
    fi
    _ok=1
    for t in "$ARM_TAG" "$BASE_TAG"; do
        # shellcheck disable=SC2086
        if ! compgen -G "$REWARD_ROOT/flow_rockstar/halos/${box}__candidate__${t}/*/halos_*.ascii" > /dev/null; then
            echo "  $box: no Rockstar catalog for tag '$t'. SKIPPED."
            _ok=0
        fi
    done
    [ "$_ok" = "1" ] || continue

    # The host the region mass function is measured around. Read from the
    # export rather than defaulted: gate_verdict_cpu.sbatch's default is set9's
    # 168880, and silently measuring set10's mass function around a halo id that
    # belongs to a different box is the kind of wrong that still prints a table.
    HOST_ID=$(/zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin/python -c \
        "import json,sys; print(json.load(open('$RUN_DIR/export.json'))['per_host'][0]['halo_id'])")

    ENVFILE="$REWARD_ROOT/env/gate_ab_${box}_$(date +%Y%m%d_%H%M%S)_$$.env"
    cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_gate_ab.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT="$PROJECT"
ZFS="$ZFS"
DMSR_REWARD_ROOT="$REWARD_ROOT"
HG_BOX="$box"
HG_RUN_DIR="$RUN_DIR"
HG_RS_TAG="$ARM_TAG"
HG_BASELINE_TAG="$BASE_TAG"
HG_BASELINE_NAME="$AB_BASELINE_NAME"
HG_HOST_ID="$HOST_ID"
EOT
    _check=$(env -i bash -c "set -eu; source '$ENVFILE'; printf '%s' \"\$HG_RS_TAG\"" 2>&1) || {
        echo ">>> env file does not source cleanly: $_check" >&2; exit 1; }
    [ "$_check" = "$ARM_TAG" ] || { echo ">>> env round-trip MISMATCH" >&2; exit 1; }

    echo "  $box: $AB_ARM vs $AB_BASELINE, host $HOST_ID"
    jid=$(sub scripts/slurm/gate_verdict_cpu.sbatch "$ENVFILE")
    echo "        -> job $jid (CPU, ~30 s)"
    NSUB=$((NSUB + 1))
    WATCH+=("  $box: tail -F $REWARD_ROOT/logs/slurm-hg_verdict-$jid.out")
    RESULTS+=("  $box: $RUN_DIR/verdict_vs_${AB_BASELINE_NAME}.json")
done

echo
if [ "$NSUB" = "0" ]; then
    echo "nothing submitted -- see the skip reasons above."
    exit 1
fi
echo "watch:"; printf '%s\n' "${WATCH[@]}"
echo "results:"; printf '%s\n' "${RESULTS[@]}"
echo
echo "Q1 = did subhalo abundance deteriorate from the baseline?"
echo "Q2 = was the baseline's host damage recovered?"
