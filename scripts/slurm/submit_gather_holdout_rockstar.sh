#!/bin/bash
# Gate finished member-gather arms on HELD-OUT boxes with real Rockstar.
#
#   export held-out tiles from tuned.pt (GPU, minutes)
#     --afterok--> splice into the frozen SR2 box (CPU, minutes)
#     --afterok--> full-box Rockstar (CPU, hours)
#     --afterok--> compare vs base + HR (CPU, minutes)
#
# One chain per arm, plus ONE frozen control per box. The control is shared
# because every arm starts from the same frozen generator and the pool selects
# the same tiles for the same box, so the `frozen` array in each export is
# identical -- and a control on the SAME splice edges is what separates "the
# objective built substructure" from "the splice seam did".
#
# docs/sr2_member_gather_training.md section 11 item 1: NOTHING HAS BEEN GATED.
# This is that job. Every number in sections 9 and 10 is a surrogate scoring
# itself; `tile-overfit-proxy-exploitation` measured one reaching +255 while the
# halo finder showed no gain at all.
#
# Only calls sbatch. Config travels in ONE timestamped env file per chain passed
# as a POSITIONAL argument -- never `sbatch --export`, which requeues+holds.
#
#   bash scripts/slurm/submit_gather_holdout_rockstar.sh
#   HG_BOX=set10 bash scripts/slurm/submit_gather_holdout_rockstar.sh
#   MG_ARMS="all_blocks_self" bash scripts/slurm/submit_gather_holdout_rockstar.sh
#   EXPORT_ONLY=1 bash scripts/slurm/submit_gather_holdout_rockstar.sh   # shakeout
#   DRY=1 bash scripts/slurm/submit_gather_holdout_rockstar.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
EXPORT_ONLY="${EXPORT_ONLY:-0}"
NO_CONTROL="${NO_CONTROL:-0}"
# Resume. An export that already wrote tiles.npz is not redone: the 2026-08-25
# node-level kill took out the SPLICE stage of three chains whose exports had
# both succeeded, and re-running those costs two a6000 jobs for nothing.
# REDO_EXPORT=1 forces them.
REDO_EXPORT="${REDO_EXPORT:-0}"

# The four arms of the 2026-08-24 ladder. `self` and `nocentre` lead on
# bound_hard (0.140 / 0.144 held out); `full` and `radial` are here because the
# surrogate cannot settle which of them a halo finder prefers, which is the
# whole reason for this job.
MG_ARMS="${MG_ARMS:-all_blocks_self all_blocks_nocentre all_blocks_full all_blocks_radial}"
# A/B the arm against a previously-gated one, in the EDITED REGION, answering
# the two questions a follow-up arm raises: did subhalo abundance deteriorate
# from the baseline, and was the host damage recovered. Empty = skip the stage.
HG_BASELINE_ARM="${HG_BASELINE_ARM:-}"
HG_BASELINE_NAME="${HG_BASELINE_NAME:-baseline}"
# Chain the whole gate behind a still-running training job, so `tuned.pt` is
# gated the moment it exists. The gate costs ~19 min of compute against a ~10 h
# train (3%), which is why it is worth firing automatically.
HG_AFTER_JOB="${HG_AFTER_JOB:-}"
HG_BOX="${HG_BOX:-set9}"
HG_SEED="${HG_SEED:-0}"
HG_MIN_P="${HG_MIN_P:-50}"
HG_SWEEP="${HG_SWEEP:-1}"
HG_MAX_HOSTS="${HG_MAX_HOSTS:-0}"
# The per-host sections of compare_gather_catalog describe ONE host; the
# supervised-target rate covers every host in subhalos.json. This is the most
# massive held-out host of each box, from pool.json.
# set3-set7 are TRAIN boxes: the same gate run in sample, which is the only way
# to separate "the objective cannot be reached" from "it does not generalise".
# Each id is the most massive host that box contributes to the pool.
case "$HG_BOX" in
    set9)  _DEF_HOST=168880 ;;
    set10) _DEF_HOST=216377 ;;
    set3)  _DEF_HOST=248732 ;;
    set4)  _DEF_HOST=175908 ;;
    set5)  _DEF_HOST=249425 ;;
    set6)  _DEF_HOST=15933  ;;
    set7)  _DEF_HOST=149358 ;;
    *)     _DEF_HOST=0 ;;
esac
HG_HOST_ID="${HG_HOST_ID:-$_DEF_HOST}"
# The export directory is named for the box's ROLE, not the script's name: an
# in-sample gate written to `holdout_set3` is a mislabel that outlives the run.
case "$HG_BOX" in
    set9|set10) _DEF_ROLE=holdout ;;
    *)          _DEF_ROLE=train ;;
esac
HG_ROLE="${HG_ROLE:-$_DEF_ROLE}"
if [ "$HG_HOST_ID" = "0" ]; then
    echo "no default host id for $HG_BOX -- set HG_HOST_ID to one of its hosts" >&2
    exit 1
fi

# tuned.pt must exist NOW unless the chain is queued behind the job that writes
# it -- which is the whole point of HG_AFTER_JOB. With a dependency set, a
# missing checkpoint is expected; the run dir still has to exist so a typo in
# the arm name is caught here rather than by an export job an hour later.
for arm in $MG_ARMS; do
    if [ -f "$REWARD_ROOT/member_gather/$arm/tuned.pt" ]; then
        continue
    elif [ -n "$HG_AFTER_JOB" ] && [ -d "$REWARD_ROOT/member_gather/$arm" ]; then
        echo "note: '$arm' has no tuned.pt yet; job $HG_AFTER_JOB is expected" \
             "to write it before the export stage runs."
    else
        echo "no tuned.pt for arm '$arm' under $REWARD_ROOT/member_gather" >&2
        [ -n "$HG_AFTER_JOB" ] \
            && echo "  and no run dir either -- check the arm name." >&2
        exit 1
    fi
done

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" \
         "$REWARD_ROOT/flow_rockstar/fields" "$REWARD_ROOT/flow_rockstar/logs"

# Pre-flight. Every stage of this chain writes -- env files, 3.2 GB fields,
# Rockstar catalogs -- and a full quota fails them in the least readable way
# there is: the job does its work, dies on a small write, and cannot write its
# own traceback either, because the log is on the same quota.
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
WATCH=(); RESULTS=()

# Writes one env file and returns its path on stdout.
write_env() {
    local tag="$1" train_dir="$2" run_dir="$3" which="$4" rs_tag="$5"
    local f="$REWARD_ROOT/env/gather_holdout_${STAMP}_${tag}_$$.env"
    cat > "$f" <<EOT
# Written by scripts/slurm/submit_gather_holdout_rockstar.sh at $(date '+%F %T');
# sourced by the job preambles as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
HG_TRAIN_DIR=$train_dir
HG_RUN_DIR=$run_dir
HG_BOX=$HG_BOX
HG_SEED=$HG_SEED
HG_WHICH=$which
HG_RS_TAG=$rs_tag
HG_HOST_ID=$HG_HOST_ID
HG_MIN_P=$HG_MIN_P
HG_SWEEP=$HG_SWEEP
HG_MAX_HOSTS=$HG_MAX_HOSTS
HG_BASELINE_TAG=$HG_BASELINE_TAG
HG_BASELINE_NAME=$HG_BASELINE_NAME
EOT
    # `cat` failing leaves `echo` to succeed, so the function returns 0 and the
    # chain is submitted pointing at a file that does not exist -- measured on
    # 2026-08-25, when a full quota ate three env files and eight jobs died with
    # "config file not readable". Verify, and make the caller stop.
    if [ ! -s "$f" ]; then
        echo "FAILED to write the env file $f" >&2
        echo "  (disk quota? scripts/reward/purge_regenerable_fields.py)" >&2
        return 1
    fi
    echo "$f"
}

# splice -> rockstar -> compare, optionally behind an export job.
# Prints nothing; appends to WATCH and RESULTS.
gate_chain() {
    local label="$1" envfile="$2" rs_tag="$3" dep="$4"
    local field="$REWARD_ROOT/flow_rockstar/fields/${HG_BOX}__${rs_tag}__seed${HG_SEED}.npy"
    local depflag=(); [ -n "$dep" ] && depflag=(--dependency=afterok:"$dep")

    local sp rs cmp
    sp=$(sub "${depflag[@]}" scripts/slurm/gather_splice_cpu.sbatch "$envfile")
    echo "    splice   -> job $sp${dep:+ (afterok:$dep)}"
    rs=$(sub --dependency=afterok:"$sp" \
         scripts/slurm/flow_rockstar_catalog_cpu.sbatch \
         "BOX=$HG_BOX" "TAG=$rs_tag" "SEED=$HG_SEED" "FIELD_OUT=$field")
    echo "    rockstar -> job $rs (afterok:$sp)"
    cmp=$(sub --dependency=afterok:"$rs" \
          scripts/slurm/gather_compare_cpu.sbatch "$envfile")
    echo "    compare  -> job $cmp (afterok:$rs)"
    # The edited-region counts, ALWAYS -- with a baseline this is the A/B, and
    # without one it is still the only place the region-scale subhalo count is
    # reported. `compare_gather_catalog` counts box-wide, where 93.75% of the
    # volume is untouched frozen SR2 and swamps the signal.
    if [ "$label" != "FROZEN" ]; then
        local vd out_name
        if [ -n "$HG_BASELINE_ARM" ]; then
            out_name="verdict_vs_${HG_BASELINE_NAME}"
        else
            out_name="region_counts"
        fi
        vd=$(sub --dependency=afterok:"$cmp" \
             scripts/slurm/gate_verdict_cpu.sbatch "$envfile")
        echo "    region   -> job $vd (afterok:$cmp${HG_BASELINE_ARM:+, vs $HG_BASELINE_ARM})"
        WATCH+=("  $label region:   tail -F $REWARD_ROOT/logs/slurm-hg_verdict-$vd.out")
        RESULTS+=("  $label region:  $RUN_DIR/${out_name}.json")
    fi

    WATCH+=("  $label rockstar: tail -F $REWARD_ROOT/flow_rockstar/logs/slurm-flow_rs_cat-$rs.out")
    WATCH+=("  $label compare:  tail -F $REWARD_ROOT/logs/slurm-hg_cmp-$cmp.out")
    RESULTS+=("  $label: $REWARD_ROOT/flow_rockstar/compare/${HG_BOX}__${rs_tag}.json")
}

HG_BASELINE_TAG=""
if [ -n "$HG_BASELINE_ARM" ]; then
    HG_BASELINE_TAG="mgho_${HG_BASELINE_ARM}_${HG_BOX}"
    _bl="$REWARD_ROOT/flow_rockstar/compare/${HG_BOX}__${HG_BASELINE_TAG}.json"
    if [ ! -f "$_bl" ]; then
        echo "baseline arm '$HG_BASELINE_ARM' has no gated catalog for $HG_BOX." >&2
        echo "  expected: $_bl" >&2
        echo "  gate it first, or unset HG_BASELINE_ARM to skip the A/B." >&2
        exit 1
    fi
    echo "A/B baseline: $HG_BASELINE_ARM (tag $HG_BASELINE_TAG) -- gated, found"
fi

echo "box $HG_BOX, host $HG_HOST_ID, arms: $MG_ARMS"
[ -n "$HG_AFTER_JOB" ] && echo "chained behind training job $HG_AFTER_JOB"
echo

FIRST_EXPORT=""; FIRST_RUN_DIR=""
for arm in $MG_ARMS; do
    TRAIN_DIR="$REWARD_ROOT/member_gather/$arm"
    RUN_DIR="$TRAIN_DIR/${HG_ROLE}_$HG_BOX"
    RS_TAG="mgho_${arm}_${HG_BOX}"
    ENVFILE=$(write_env "$arm" "$TRAIN_DIR" "$RUN_DIR" "out" "$RS_TAG") || exit 1

    echo "arm $arm"
    echo "    envfile  $ENVFILE"
    if [ -f "$RUN_DIR/tiles.npz" ] && [ "$REDO_EXPORT" != "1" ] \
       && [ -z "$HG_AFTER_JOB" ]; then
        EXP=""
        echo "    export   -> SKIPPED, $RUN_DIR/tiles.npz exists" \
             "(REDO_EXPORT=1 to redo)"
    else
        _afterflag=(); [ -n "$HG_AFTER_JOB" ] \
            && _afterflag=(--dependency=afterok:"$HG_AFTER_JOB")
        EXP=$(sub "${_afterflag[@]}" scripts/slurm/gather_export_gpu.sbatch "$ENVFILE")
        echo "    export   -> job $EXP (GPU)"
        WATCH+=("  $arm export:   tail -F $REWARD_ROOT/logs/slurm-hg_export-$EXP.out")
    fi
    if [ -z "$FIRST_RUN_DIR" ]; then
        FIRST_EXPORT="$EXP"; FIRST_RUN_DIR="$RUN_DIR"
    fi

    if [ "$EXPORT_ONLY" = "1" ]; then
        # Stop after the export whether or not one was submitted; with a
        # skipped export there is simply nothing new to report.
        [ -n "$EXP" ] && RESULTS+=("  $arm export: $RUN_DIR/export.json")
    else
        gate_chain "$arm" "$ENVFILE" "$RS_TAG" "$EXP"
    fi
    echo
done

# The shared control: the frozen field spliced on the SAME edges. Read the
# candidate's numbers against THIS, not against set9__base__base, or a splice
# seam is indistinguishable from an objective that worked.
if [ "$EXPORT_ONLY" != "1" ] && [ "$NO_CONTROL" != "1" ]; then
    CTRL_TAG="mgho_frozen_${HG_BOX}"
    CTRL_ENV=$(write_env "frozen" "$REWARD_ROOT/member_gather/${MG_ARMS%% *}" \
                         "$FIRST_RUN_DIR" "frozen" "$CTRL_TAG") || exit 1
    echo "frozen control (shared, same splice edges)"
    echo "    envfile  $CTRL_ENV"
    gate_chain "FROZEN" "$CTRL_ENV" "$CTRL_TAG" "$FIRST_EXPORT"
    echo
fi

echo "watch:"
printf '%s\n' "${WATCH[@]}"
echo
echo "results:"
printf '%s\n' "${RESULTS[@]}"
echo
echo "read, in this order:"
echo "  shell_profile         FIRST -- is the change local to the host, or a splice-edge artifact?"
echo "  whole_box             hosts >= 200p, which must not have moved"
echo "  supervised_targets    the arm vs FROZEN on the same edges; frozen was ~3.6% by construction"
echo "  subhalos_within_rvir  HR vs frozen vs tuned, binned by particle count"
