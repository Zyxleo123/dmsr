#!/bin/bash
# Cross-check the Rockstar subhalo census with GADGET-4's SUBFIND.
#
# WHY. Every subhalo number in this project comes from Rockstar, which links in
# six dimensions and can therefore report an object with no spatial overdensity
# behind it. That is exactly the shape of the open question -- Rockstar says the
# tuned arms add hundreds of subhalos inside the trained host, and the projected
# panels show nothing. SUBFIND defines substructure in configuration space (a
# density peak bounded by a saddle, kept only if self-bound), so what it
# confirms is real by two independent definitions and what it refuses is a
# phase-space artefact. HR is run too, as the control that says what agreement
# between two finders looks like when the objects are certainly real.
#
#   bash scripts/slurm/submit_subfind.sh smoke     # 5 min, validates the chain
#   bash scripts/slurm/submit_subfind.sh build     # compile GADGET-4 only
#   bash scripts/slurm/submit_subfind.sh full      # the real run
#   DRY=1 bash scripts/slurm/submit_subfind.sh full
#   SF_TARGETS="hr self" bash scripts/slurm/submit_subfind.sh full
#
# RUN THE SMOKE TEST FIRST. It writes a small synthetic snapshot with planted,
# self-bound clumps and runs the whole chain on it in minutes. Because the
# answer is known it checks that the pipeline *finds what is there*, not merely
# that it exits 0 -- and it does that before 44 cores spend hours on a 512^3 box
# whose param file might have one wrong tag.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
SUBFIND_ROOT="${SUBFIND_ROOT:-$REWARD_ROOT/subfind}"
MG="$REWARD_ROOT/member_gather"
FIELDS="$REWARD_ROOT/flow_rockstar/fields"
DRY="${DRY:-0}"
MODE="${1:-full}"

SF_BOX="${SF_BOX:-set9}"
SF_HOST_ID="${SF_HOST_ID:-168880}"
SF_SEED="${SF_SEED:-0}"
SF_MIN_P="${SF_MIN_P:-50}"
SF_NRANKS="${SF_NRANKS:-44}"
# MaxMemSize is PER RANK in MByte and is baked into param.txt by the snapshot
# stage, so it has to be decided here, next to the rank count it divides.
SF_MAX_MEM_MB="${SF_MAX_MEM_MB:-13000}"

# The three arms under test, plus the two references every comparison needs.
#   self         the baseline arm every A/B is measured against
#   soft         all_blocks_sm_coarse_soft, the soft-membership arm that wins
#                the held-out Rockstar gate
#   knn          all_blocks_knn6d_w5, the kNN-CDF term ON TOP of self
#                (knn6d_only drops the self terms and is a different question)
SF_TARGETS="${SF_TARGETS:-hr frozen self soft knn}"

declare -A ARM_OF=(
    [self]=all_blocks_self
    [soft]=all_blocks_sm_coarse_soft
    [knn]=all_blocks_knn6d_w5
)

# The map2map from-scratch arms. These differ from every target above in one
# way that matters here: an m2m candidate is a WHOLE BOX, not tiles spliced into
# a frozen field, so there is nothing to re-splice and no tiles.npz -- the field
# is whatever scripts/slurm/m2m_box_dump_gpu.sbatch wrote. They are passed to
# the snapshot job through its existing SF_FIELD knob.
declare -A M2M_OF=(
    [m2m_base]=m2m_theirs
    [m2m_knn]=m2m_theirs_knn50
    [m2m_mg]=m2m_theirs_mg
)
SF_M2M_EPOCH="${SF_M2M_EPOCH:-77}"

mkdir -p "$SUBFIND_ROOT/logs" "$SUBFIND_ROOT/env" "$SUBFIND_ROOT/compare"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"

# Submitting from inside an allocation poisons the child job's environment.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

sub() {  # sub <dependency-or-empty> <script> [args...]
    local dep="$1"; shift
    local out
    if [ "$DRY" = "1" ]; then echo "DRY: sbatch ${dep:+--dependency=afterok:$dep} $*" >&2; echo "0"; return; fi
    out=$(sbatch ${dep:+--dependency=afterok:$dep} --parsable "$@")
    echo "${out%%;*}"
}

write_env() {  # write_env <name> <extra lines...>
    local name="$1"; shift
    local f="$SUBFIND_ROOT/env/${name}_${STAMP}.env"
    {
        echo "# Written by scripts/slurm/submit_subfind.sh at $(date '+%F %T');"
        echo "# sourced by the job preambles as a POSITIONAL argument -- never"
        echo "# via sbatch --export, which gets jobs requeued and held here."
        echo "PROJECT=$PROJECT"
        echo "ZFS=$ZFS"
        echo "DMSR_REWARD_ROOT=$REWARD_ROOT"
        echo "SUBFIND_ROOT=$SUBFIND_ROOT"
        echo "SF_BOX=$SF_BOX"
        echo "SF_SEED=$SF_SEED"
        echo "SF_MIN_P=$SF_MIN_P"
        echo "SF_NRANKS=$SF_NRANKS"
        echo "SF_MAX_MEM_MB=$SF_MAX_MEM_MB"
        printf '%s\n' "$@"
    } > "$f"
    echo "$f"
}

# --------------------------------------------------------------------------- #
echo "=== GADGET-4 SUBFIND cross-check -- mode: $MODE"
echo "    box $SF_BOX, host $SF_HOST_ID, min_p $SF_MIN_P"
echo

declare -a JOBS=() WATCH=()

# --- build ------------------------------------------------------------------ #
BUILD_JOB=""
if [ ! -x "external/gadget4/Gadget4" ] || [ "$MODE" = "build" ]; then
    echo "──────── build GADGET-4 (FOF + SUBFIND)"
    BUILD_JOB=$(sub "" scripts/slurm/subfind_build_cpu.sbatch)
    echo "    build -> job $BUILD_JOB"
    JOBS+=("$BUILD_JOB"); WATCH+=("sf_build:$BUILD_JOB")
    echo
else
    echo "    GADGET-4 already built: external/gadget4/Gadget4"
    echo
fi
[ "$MODE" = "build" ] && { echo "build only; rerun with 'smoke' when it finishes."; exit 0; }

# --- smoke ------------------------------------------------------------------ #
if [ "$MODE" = "smoke" ]; then
    echo "──────── smoke: synthetic snapshot with planted clumps"
    NAME="smoke"
    ENVF=$(write_env smoke "SF_NAME=$NAME" "SF_SMOKE=64" "SF_SOURCE=hr" \
                     "SF_MAX_MEM_MB=2000" "SF_NRANKS=4")
    J1=$(sub "$BUILD_JOB" scripts/slurm/subfind_snapshot_cpu.sbatch "$ENVF")
    J2=$(sub "$J1" scripts/slurm/subfind_run_smoke_cpu.sbatch "$ENVF")
    echo "    snapshot -> job $J1"
    echo "    subfind  -> job $J2"
    WATCH+=("sf_snap:$J1" "sf_smoke:$J2")
    echo
    echo "RESULT to read: $SUBFIND_ROOT/$NAME/smoke_verdict.json"
    echo "  it states how many of the planted clumps SUBFIND recovered."
    echo "  Anything less than all of them means the chain is wrong, not the data."
    echo
    for w in "${WATCH[@]}"; do
        echo "  tail -F $SUBFIND_ROOT/logs/slurm-${w%%:*}-${w##*:}.out"
    done
    exit 0
fi

# --- full ------------------------------------------------------------------- #
declare -a RUN_JOBS=() CAND_NAMES=()
for t in $SF_TARGETS; do
    M2M_FIELD=""
    case "$t" in
        hr)     NAME="${SF_BOX}__hr";    SOURCE=hr;        TAG="" ;;
        frozen) NAME="${SF_BOX}__frozen"; SOURCE=candidate; TAG="mgho_frozen_${SF_BOX}" ;;
        m2m_*)
            RUN="${M2M_OF[$t]:-}"
            [ -n "$RUN" ] || { echo "unknown m2m target '$t'" >&2; exit 1; }
            TAG="${RUN}_e${SF_M2M_EPOCH}"
            NAME="${SF_BOX}__${TAG}"
            SOURCE=candidate
            M2M_FIELD="$REWARD_ROOT/m2m/boxes/${RUN}__state_${SF_M2M_EPOCH}__${SF_BOX}.npy"
            ;;
        *)
            ARM="${ARM_OF[$t]:-}"
            [ -n "$ARM" ] || { echo "unknown target '$t'" >&2; exit 1; }
            TAG="mgho_${ARM}_${SF_BOX}"
            NAME="${SF_BOX}__${TAG}"
            SOURCE=candidate
            ;;
    esac

    echo "──────── $t  ($NAME)"
    DEP="$BUILD_JOB"

    # A spliced field may have been purged to make room; it is rebuilt from the
    # arm's tiles.npz, which is never purged. HR and a re-splice are the only
    # two ways a field gets here, and neither needs a GPU.
    if [ -n "$M2M_FIELD" ]; then
        # No splice: the box IS the candidate. A missing one is the dump job
        # not having run, so skip rather than fabricate a field.
        if [ ! -f "$M2M_FIELD" ]; then
            echo "    !! no box at $M2M_FIELD -- SKIPPED"
            echo "       produced by: bash scripts/slurm/submit_m2m_halos.sh"
            echo
            continue
        fi
        echo "    whole-box candidate: $M2M_FIELD"
    elif [ "$SOURCE" = "candidate" ]; then
        FIELD="$FIELDS/${SF_BOX}__${TAG}__seed${SF_SEED}.npy"
        if [ ! -f "$FIELD" ]; then
            RUN_DIR="$MG/${ARM_OF[$t]:-}/holdout_${SF_BOX}"
            WHICH=out
            [ "$t" = "frozen" ] && { RUN_DIR="$MG/all_blocks_self/holdout_${SF_BOX}"; WHICH=frozen; }
            if [ ! -f "$RUN_DIR/tiles.npz" ]; then
                echo "    !! no field and no tiles.npz ($RUN_DIR) -- SKIPPED"
                echo
                continue
            fi
            echo "    field purged; re-splicing from $RUN_DIR/tiles.npz"
            SPENV=$(write_env "splice_$t" "HG_RUN_DIR=$RUN_DIR" "HG_BOX=$SF_BOX" \
                              "HG_SEED=$SF_SEED" "HG_WHICH=$WHICH" "HG_RS_TAG=$TAG")
            JS=$(sub "$DEP" scripts/slurm/gather_splice_cpu.sbatch "$SPENV")
            echo "    splice   -> job $JS"
            DEP="$JS"; WATCH+=("hg_splice:$JS")
        fi
    fi

    ENVF=$(write_env "$t" "SF_NAME=$NAME" "SF_SOURCE=$SOURCE" "SF_TAG=$TAG" \
                     "SF_HOST_ID=$SF_HOST_ID" ${M2M_FIELD:+"SF_FIELD=$M2M_FIELD"})
    J1=$(sub "$DEP" scripts/slurm/subfind_snapshot_cpu.sbatch "$ENVF")
    J2=$(sub "$J1"  scripts/slurm/subfind_run_cpu.sbatch "$ENVF")
    echo "    snapshot -> job $J1"
    echo "    subfind  -> job $J2"
    WATCH+=("sf_snap:$J1" "sf_run:$J2")
    RUN_JOBS+=("$J2")
    [ "$t" != "hr" ] && [ "$t" != "frozen" ] && CAND_NAMES+=("$t:$TAG:$NAME")
    echo
done

# --- compare, once every finder has run ------------------------------------- #
if [ "${#CAND_NAMES[@]}" -gt 0 ] && [ "${#RUN_JOBS[@]}" -gt 0 ]; then
    DEPALL=$(IFS=:; echo "${RUN_JOBS[*]}")
    echo "──────── compare (waits on every SUBFIND run)"
    for entry in "${CAND_NAMES[@]}"; do
        IFS=: read -r t tag name <<< "$entry"
        CENV=$(write_env "cmp_$t" "SF_NAME=$name" "SF_TAG=$tag" \
                         "SF_HOST_ID=$SF_HOST_ID" \
                         "SF_HR_NPZ=$SUBFIND_ROOT/${SF_BOX}__hr/subfind_catalog.npz" \
                         "SF_BASE_NPZ=$SUBFIND_ROOT/${SF_BOX}__frozen/subfind_catalog.npz" \
                         "SF_CMP_OUT=$SUBFIND_ROOT/compare/${SF_BOX}__${t}")
        JC=$(sub "$DEPALL" scripts/slurm/subfind_compare_cpu.sbatch "$CENV")
        echo "    $t -> job $JC"
        WATCH+=("sf_cmp:$JC")
    done
    echo
fi

echo "watch:"
for w in "${WATCH[@]}"; do
    echo "  tail -F $SUBFIND_ROOT/logs/slurm-${w%%:*}-${w##*:}.out"
done
echo
echo "results:"
echo "  per field   $SUBFIND_ROOT/<name>/subfind_catalog.json   (census)"
echo "  the answer  $SUBFIND_ROOT/compare/${SF_BOX}__<arm>.json (confirmed_frac,"
echo "              against HR's own rate -- read 'verdict' first)"
