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
# The placement yardstick's second half: GADGET-4 SUBFIND on the arm's spliced
# field, then placement_forensics with that catalog -- self-bound satellites by
# r/R_vir, core vs outskirts, against HR and frozen. Opt-in: one whole cpu node
# (44 ranks, 600G) for ~30-60 min per arm. Rockstar object counts rank arms on
# fragmentation (placement-is-velocity-temperature); this is the real-object read.
HG_SUBFIND="${HG_SUBFIND:-0}"
HG_BASELINE_NAME="${HG_BASELINE_NAME:-baseline}"
# Chain the whole gate behind a still-running training job, so `tuned.pt` is
# gated the moment it exists. The gate costs ~19 min of compute against a ~10 h
# train (3%), which is why it is worth firing automatically.
HG_AFTER_JOB="${HG_AFTER_JOB:-}"

# --- the EARLY gate ---------------------------------------------------------
# HG_CHECKPOINT names which checkpoint inside the arm's run directory to export.
# The default `tuned.pt` is the finished (or latest) model; naming a
# `tuned_step<N>.pt` snapshot gates the arm PART-WAY THROUGH TRAINING, with the
# training job untouched and still running to its full --steps.
#
# Why this is not giving anything up. MEASURED on the three finished arms, the
# held-out objective is essentially done by step 2000:
#
#   arm                bound_frac @2000   @8000    high-k max @2000 -> @8000
#   all_blocks_smallhalo      0.0874     0.0877     1.52x -> 3.15x
#   all_blocks_selfhostguard  0.0950     0.0976     2.02x -> 4.79x
#   all_blocks_self           0.1202     0.1395     1.92x -> 3.87x
#
# 97-100% of the objective by step 2000 for two of the three, 86% for the
# third -- and what keeps moving over the remaining 6000 steps is the high-k
# EXCESS, which is damage accumulating, not the objective improving. So a
# step-2000 gate is a real read, not a peek, and the full run still happens.
#
# READ IT ONE-SIDED. A clear negative early (top-level still falling, subs still
# exploding) says the mechanism did not change and is worth acting on. A null is
# NOT evidence of failure. Every artifact records `checkpoint.step` and
# `checkpoint.snapshot` so a mid-run catalog can never be read as a final one.
HG_CHECKPOINT="${HG_CHECKPOINT:-tuned.pt}"
# With a snapshot that does not exist yet, prepend a 1-core CPU job that polls
# for it, so the whole chain can be queued at the same moment as training.
# 0 disables the wait (the snapshot must already be on disk).
HG_WAIT_MINUTES="${HG_WAIT_MINUTES:-300}"
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

# An early gate names a snapshot instead of the live checkpoint. Its artifacts
# get their own run dir and Rockstar tag: a mid-run catalog written over the
# final one would be unrecoverable and, worse, silently misread later.
CKPT_SUFFIX=""
WAIT_FOR_CKPT=0
if [ "$HG_CHECKPOINT" != "tuned.pt" ]; then
    case "$HG_CHECKPOINT" in
        *.pt) ;;
        *)  echo "HG_CHECKPOINT='$HG_CHECKPOINT' is not a .pt file." >&2
            echo "  expected 'tuned.pt' or a 'tuned_step<N>.pt' snapshot" >&2
            echo "  written by the trainer's MG_GATE_AT_STEPS." >&2
            exit 1 ;;
    esac
    _b="${HG_CHECKPOINT%.pt}"
    CKPT_SUFFIX="_${_b#tuned_}"
fi

# The checkpoint must exist NOW unless something is expected to write it later:
# HG_AFTER_JOB (the chain is queued behind the training job) or the wait stage
# (an early gate, queued alongside it). With either, a missing checkpoint is
# expected; the run dir still has to exist so a typo in the arm name is caught
# here rather than by an export job an hour later.
for arm in $MG_ARMS; do
    if [ -f "$REWARD_ROOT/member_gather/$arm/$HG_CHECKPOINT" ]; then
        continue
    elif [ ! -d "$REWARD_ROOT/member_gather/$arm" ]; then
        echo "no run dir for arm '$arm' under $REWARD_ROOT/member_gather" >&2
        echo "  -- check the arm name." >&2
        exit 1
    elif [ -n "$HG_AFTER_JOB" ]; then
        echo "note: '$arm' has no $HG_CHECKPOINT yet; job $HG_AFTER_JOB is" \
             "expected to write it before the export stage runs."
    elif [ -n "$CKPT_SUFFIX" ] && [ "$HG_WAIT_MINUTES" != "0" ]; then
        echo "note: '$arm' has no $HG_CHECKPOINT yet; a wait stage will poll" \
             "for it (up to ${HG_WAIT_MINUTES} min)."
        WAIT_FOR_CKPT=1
    else
        echo "no $HG_CHECKPOINT for arm '$arm' under $REWARD_ROOT/member_gather" >&2
        if [ -n "$CKPT_SUFFIX" ]; then
            echo "  An early-gate snapshot is written only if training was" >&2
            echo "  launched with MG_GATE_AT_STEPS. Either set that and" >&2
            echo "  resubmit training, or set HG_WAIT_MINUTES to poll for it," >&2
            echo "  or gate the live tuned.pt instead." >&2
        fi
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
# kNN damage report (scripts/features/knn_gate_report.py): one advisory CPU job
# per arm, after the arm's AND the frozen control's splice. KNN_REPORT=0 skips.
KNN_REPORT="${KNN_REPORT:-1}"
KNN_PENDING=(); LAST_SPLICE=""

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
HG_CHECKPOINT=$HG_CHECKPOINT
HG_WAIT_MINUTES=$HG_WAIT_MINUTES
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
    LAST_SPLICE="$sp"
    if [ "$label" != "FROZEN" ] && [ "$KNN_REPORT" = "1" ]; then
        KNN_PENDING+=("$label|$envfile|$sp|$RUN_DIR")
    fi
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
    if [ "$label" != "FROZEN" ] && [ "$HG_SUBFIND" = "1" ]; then
        local sfr="$REWARD_ROOT/subfind" sfname="${HG_BOX}__${rs_tag}" sfenv sf1 sf2 sfp
        sfenv="$REWARD_ROOT/env/hg_subfind_${rs_tag}_${STAMP}.env"
        cat > "$sfenv" <<EOT
PROJECT=$PROJECT
DMSR_REWARD_ROOT=$REWARD_ROOT
SUBFIND_ROOT=$sfr
SF_BOX=$HG_BOX
SF_SEED=$HG_SEED
SF_MIN_P=50
SF_NRANKS=44
SF_MAX_MEM_MB=13000
SF_NAME=$sfname
SF_SOURCE=candidate
SF_TAG=$rs_tag
SF_HOST_ID=$HG_HOST_ID
SF_FIELD=$field
EOT
        sf1=$(sub --dependency=afterok:"$sp" scripts/slurm/subfind_snapshot_cpu.sbatch "$sfenv")
        sf2=$(sub --dependency=afterok:"$sf1" scripts/slurm/subfind_run_cpu.sbatch "$sfenv")
        # The SUBFIND references must exist for this box (submit_subfind.sh full,
        # SF_TARGETS="hr frozen"); a missing one is skipped by name, not fatal.
        sfp=$(sub --dependency=afterok:"$sf2" scripts/slurm/placement_forensics_cpu.sbatch \
              "BOX=$HG_BOX" "PF_ARMS=$label=$rs_tag" "PF_REGION_EXPORT=$RUN_DIR/export.json" \
              "PF_CHECK_VERDICT= " \
              "PF_SUBFIND=hr=$sfr/${HG_BOX}__hr/subfind_catalog.npz,frozen=$sfr/${HG_BOX}__frozen/subfind_catalog.npz,$label=$sfr/$sfname/subfind_catalog.npz" \
              "PF_OUT=$RUN_DIR/placement_subfind.json")
        echo "    subfind  -> jobs $sf1 (snapshot) / $sf2 (run) / $sfp (placement, afterok:$sf2)"
        WATCH+=("  $label subfind:  tail -F $sfr/logs/slurm-sf_run-$sf2.out")
        RESULTS+=("  $label subfind: $RUN_DIR/placement_subfind.json  (SUBFIND sats by r/R_vir)")
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
if [ -n "$CKPT_SUFFIX" ]; then
    echo "EARLY GATE: checkpoint $HG_CHECKPOINT -> run dirs and Rockstar"
    echo "  tags carry '$CKPT_SUFFIX', so nothing collides with the final gate."
    echo "  Read it ONE-SIDED: a clear negative says the mechanism did not"
    echo "  change and is worth acting on; a null is NOT evidence of failure."
    echo "  Training is untouched and still runs to its full --steps."
    if [ "$WAIT_FOR_CKPT" = "1" ]; then
        echo "  If the snapshot never appears the wait job exits non-zero and"
        echo "  the four dependent jobs show DependencyNeverSatisfied -- the"
        echo "  reason is in the hg_wait log printed under 'watch' below."
    fi
    if [ "$NO_CONTROL" != "1" ]; then
        echo "  NOTE: the frozen control is re-gated too (~13 min of Rockstar)."
        echo "  It does not depend on the checkpoint, so if this box already"
        echo "  has one, NO_CONTROL=1 skips it."
    fi
fi
echo

FIRST_EXPORT=""; FIRST_RUN_DIR=""
for arm in $MG_ARMS; do
    TRAIN_DIR="$REWARD_ROOT/member_gather/$arm"
    # $CKPT_SUFFIX is empty for the normal gate, so every path and tag below
    # is byte-identical to every gate already on disk.
    RUN_DIR="$TRAIN_DIR/${HG_ROLE}_$HG_BOX$CKPT_SUFFIX"
    RS_TAG="mgho_${arm}${CKPT_SUFFIX}_${HG_BOX}"
    ENVFILE=$(write_env "$arm" "$TRAIN_DIR" "$RUN_DIR" "out" "$RS_TAG") || exit 1

    echo "arm $arm"
    echo "    envfile  $ENVFILE"
    if [ -f "$RUN_DIR/tiles.npz" ] && [ "$REDO_EXPORT" != "1" ] \
       && [ -z "$HG_AFTER_JOB" ]; then
        EXP=""
        echo "    export   -> SKIPPED, $RUN_DIR/tiles.npz exists" \
             "(REDO_EXPORT=1 to redo)"
    else
        _afterflag=(); _dep_note=""
        if [ "$WAIT_FOR_CKPT" = "1" ]; then
            # NOT a dependency on the training job: the snapshot appears
            # while that job is still RUNNING, which is the whole point, so
            # `afterok` would defeat it and `after` only says "has started".
            WAITJ=$(sub scripts/slurm/gather_wait_checkpoint_cpu.sbatch "$ENVFILE")
            echo "    wait     -> job $WAITJ (CPU, 1 core; polls for $HG_CHECKPOINT,"\
                 " up to ${HG_WAIT_MINUTES} min)"
            WATCH+=("  $arm wait:     tail -F $REWARD_ROOT/logs/slurm-hg_wait-$WAITJ.out")
            _afterflag=(--dependency=afterok:"$WAITJ")
            _dep_note=" (afterok:$WAITJ)"
        elif [ -n "$HG_AFTER_JOB" ]; then
            _afterflag=(--dependency=afterok:"$HG_AFTER_JOB")
            _dep_note=" (afterok:$HG_AFTER_JOB)"
        fi
        EXP=$(sub "${_afterflag[@]}" scripts/slurm/gather_export_gpu.sbatch "$ENVFILE")
        echo "    export   -> job $EXP (GPU)$_dep_note"
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
    FROZEN_SPLICE="$LAST_SPLICE"
    echo
fi

# kNN damage reports: after the arm's splice and, when it was submitted in this
# chain, the frozen control's (otherwise the control field must already exist;
# the job checks and skips with a reason, exit 0).
if [ "$KNN_REPORT" = "1" ] && [ "${#KNN_PENDING[@]}" -gt 0 ]; then
    echo "kNN damage reports (advisory; knn_gate_report.py)"
    for item in "${KNN_PENDING[@]}"; do
        IFS='|' read -r k_label k_env k_sp k_run <<<"$item"
        k_dep="$k_sp${FROZEN_SPLICE:+:$FROZEN_SPLICE}"
        kj=$(sub --dependency=afterok:"$k_dep" scripts/slurm/gate_knn_cpu.sbatch "$k_env")
        echo "    $k_label knn -> job $kj (afterok:$k_dep)"
        WATCH+=("  $k_label knn:      tail -F $REWARD_ROOT/logs/slurm-hg_knn-$kj.out")
        RESULTS+=("  $k_label knn:     $k_run/knn_report.json  (+ knn_region.json/.png)")
    done
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
echo "  knn_report.json       advisory: DAMAGE / hosts_diffused / hosts_overconcentrated (set9-calibrated)"
