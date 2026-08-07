#!/bin/bash
# Submitter for the DIRECT SR2 fine-tuning line. It ONLY calls sbatch.
#
#   DRY=1 bash scripts/slurm/submit_sr2_direct.sh all     # print, submit nothing
#   bash scripts/slurm/submit_sr2_direct.sh data          # candidates + labels
#   bash scripts/slurm/submit_sr2_direct.sh proxy         # fit + GATE the proxy
#   bash scripts/slurm/submit_sr2_direct.sh baseline      # frozen field metrics
#   bash scripts/slurm/submit_sr2_direct.sh calibrate     # -> HUMAN STEP
#   bash scripts/slurm/submit_sr2_direct.sh overfit       # section 10 small overfit
#   bash scripts/slurm/submit_sr2_direct.sh rung RUNG=fine
#   bash scripts/slurm/submit_sr2_direct.sh dagger        # one aggregation round
#
# The experiment matrix (section 13), in the order it must be run:
#   1. frozen SR2                              -> `baseline`
#   2. density/adversarial continuation only   -> NOT SUBMITTED. There is no SR2
#      discriminator checkpoint in this checkout, so this arm does not exist;
#      see configs/reward/sr2_direct_finetune.yaml `adversarial:`.
#   3. proxy reward, proj_noise                -> `rung RUNG=proj_noise`
#   4. + fine block                            -> `rung RUNG=fine`
#   5. + middle block                          -> `rung RUNG=middle_fine`
#   6. best rung + optional adversarial        -> ablation only, not here
#   7. coarse/full                             -> only if the earlier rung passes
#                                                 every gate but plateaus
#
# TWO PLACES THIS DELIBERATELY STOPS RATHER THAN CHAINING:
#   * after `calibrate`, because the density tolerance must be pasted into the
#     config by a human and `calibrated: true` set. A chain through it would be
#     scoring against placeholders, which is exactly what the flag exists to
#     prevent;
#   * between rungs, because advancing is a decision taken on the REAL catalog
#     result, not on the proxy and not on a schedule.
#
# Configuration goes into ONE timestamped env file passed as a POSITIONAL
# argument. Never `sbatch --export`: on this cluster any explicit export list
# makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD with
# "(user env retrieval failed requeued held)", stranding every dependent job.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
ROOT="${DMSR_SR2_DIRECT_ROOT:-$REWARD_ROOT/sr2_direct}"
STAGE="${1:-all}"; shift || true
DRY="${DRY:-0}"

RUN_NAME="${RUN_NAME:-direct_a}"
DIRECT_CFG="${DIRECT_CFG:-configs/reward/sr2_direct_finetune.yaml}"
# set0-7 fit the proxy; set8-11 gate it and evaluate the actor. set12 is a late
# preflight only; set13-15 are sealed and appear nowhere in this file.
FIT_BOXES="${FIT_BOXES:-set0 set1 set2 set3 set4 set5 set6 set7}"
EVAL_BOXES="${EVAL_BOXES:-set8 set9 set10 set11}"
# Frozen seeds: the negative control AND the source of the seed-to-seed spread
# the density tolerance is calibrated from. Two is the minimum that has a
# spread at all; four is the configured default.
FROZEN_SEEDS="${FROZEN_SEEDS:-0 1 2 3}"
ALPHAS="${ALPHAS:-0.25 0.5 1.0}"
RUNG="${RUNG:-proj_noise}"
CHECKPOINT="${CHECKPOINT:-}"
TAG="${TAG:-}"
AFTER="${AFTER:-}"

# Allow `submit_sr2_direct.sh rung RUNG=fine` (and list values like
# FROZEN_SEEDS="0 1 2") style overrides after the stage. `export "$kv"` sets and
# exports the whole assignment in one go, including a value with spaces. It is
# NOT followed by `eval "$kv"`: eval re-parses `FROZEN_SEEDS=0 1 2` as the
# assignment `FROZEN_SEEDS=0` followed by the command `1 2`, which fails with
# "1: command not found". export already updates the shell variable the
# `read -r -a` lines below consume, so eval was both redundant and wrong.
for kv in "$@"; do
    case "$kv" in *=*) export "$kv" ;; esac
done

read -r -a _FIT <<< "$FIT_BOXES"
read -r -a _EVAL <<< "$EVAL_BOXES"
read -r -a _SEEDS <<< "$FROZEN_SEEDS"
read -r -a _ALPHAS <<< "$ALPHAS"

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/direct_$(date +%Y%m%d_%H%M%S).env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_sr2_direct.sh at $(date '+%F %T'); sourced by
# the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
DMSR_SR2_DIRECT_ROOT=$ROOT
DIRECT_CFG=$DIRECT_CFG
REWARD_CFG=configs/reward/reward.yaml
RUN_NAME=$RUN_NAME
BASE_SEED=0
EOF
echo "env file: $ENVFILE"

# Submitting from inside an allocation would leak SLURM_* into the child jobs.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

# Per-job overrides MUST come after ENVFILE: the preamble processes positional
# args strictly in order. Declared globally and reset to () -- never unset,
# because under `set -u` an unset array expands to one empty argument, which
# fails the preamble's config-file check.
SUB_OVERRIDES=()
ABORT_FLAG="$ROOT/.submit_sr2_direct_aborted.$$"
trap 'rm -f "$ABORT_FLAG"' EXIT

sub() {   # sub <human label> <sbatch args...>
    local label="$1"; shift
    if [ "$DRY" = "1" ]; then
        # The command goes to stderr: stdout is the job id, which the caller
        # captures with $(...) and feeds to the next --dependency.
        echo "DRY  $label:" >&2
        echo "       sbatch $* $ENVFILE ${SUB_OVERRIDES[*]}" >&2
        echo "000000"
        return 0
    fi
    local jid rc
    jid=$(sbatch --parsable "$@" "$ENVFILE" "${SUB_OVERRIDES[@]}") || rc=$?
    if [ -n "${rc:-}" ] || [ -z "$jid" ]; then
        echo "" >&2
        echo "!!! sbatch FAILED for: $label" >&2
        echo "!!! nothing further has been submitted. Every stage skips work it" >&2
        echo "!!! already did, so re-running this submitter resumes rather than" >&2
        echo "!!! restarts." >&2
        touch "$ABORT_FLAG"
        echo ""
        return 1
    fi
    echo "  $label -> job $jid" >&2
    echo "$jid"
}

die_if_aborted() {
    if [ -e "$ABORT_FLAG" ]; then
        echo "aborting the submitter; see the message above." >&2
        exit 1
    fi
}

dep_of() { [ -n "$1" ] && echo "--dependency=afterok:$1" || echo ""; }

# --- data: candidates (GPU) each followed by its own label job (CPU) --------
# generate -> label is afterok per candidate, and the candidates are siblings of
# each other so they queue concurrently rather than serialising behind one
# 12-hour Rockstar run.
if [ "$STAGE" = "all" ] || [ "$STAGE" = "data" ]; then
    echo "=== proxy data ($RUN_NAME)"
    LAST_LABEL=""
    for BOX in "${_FIT[@]}"; do
        # HR: the positive structural anchor.
        SUB_OVERRIDES=("BOX=$BOX" "SOURCE=hr" "SEED=0")
        J=$(sub "data: generate HR $BOX (GPU)" $(dep_of "$AFTER") \
            scripts/slurm/generate_proxy_candidates_gpu.sbatch); die_if_aborted
        LAST_LABEL=$(sub "data: label HR $BOX (CPU)" --dependency=afterok:"$J" \
            scripts/slurm/label_proxy_candidates_cpu.sbatch); die_if_aborted
        SUB_OVERRIDES=()

        # Frozen SR2 at several seeds: the baseline plus the negative control.
        for SEED in "${_SEEDS[@]}"; do
            SRC=$([ "$SEED" = "0" ] && echo frozen || echo frozen_seed)
            SUB_OVERRIDES=("BOX=$BOX" "SOURCE=$SRC" "SEED=$SEED")
            J=$(sub "data: generate $SRC seed$SEED $BOX (GPU)" $(dep_of "$AFTER") \
                scripts/slurm/generate_proxy_candidates_gpu.sbatch); die_if_aborted
            LAST_LABEL=$(sub "data: label $SRC seed$SEED $BOX (CPU)" \
                --dependency=afterok:"$J" \
                scripts/slurm/label_proxy_candidates_cpu.sbatch); die_if_aborted
            SUB_OVERRIDES=()
        done

        # Targeted HR interventions at several alpha: the within-tile ranking
        # signal. Without these the proxy only ever sees "HR vs SR2", which is
        # an easy binary distinction and not a local improvement direction.
        for A in "${_ALPHAS[@]}"; do
            SUB_OVERRIDES=("BOX=$BOX" "SOURCE=intervention" "SEED=0" "ALPHA=$A")
            J=$(sub "data: generate intervention a=$A $BOX (GPU)" $(dep_of "$AFTER") \
                scripts/slurm/generate_proxy_candidates_gpu.sbatch); die_if_aborted
            LAST_LABEL=$(sub "data: label intervention a=$A $BOX (CPU)" \
                --dependency=afterok:"$J" \
                scripts/slurm/label_proxy_candidates_cpu.sbatch); die_if_aborted
            SUB_OVERRIDES=()
        done
    done
    echo "  candidates -> $ROOT/candidates/  table -> $ROOT/proxy_data/rows.jsonl"
    export DATA_JID="$LAST_LABEL"
fi

# --- proxy: fit, then gate (same job) --------------------------------------
JID_PROXY=""
if [ "$STAGE" = "all" ] || [ "$STAGE" = "proxy" ]; then
    echo "=== proxy fit + gate"
    DEP=$(dep_of "${DATA_JID:-$AFTER}")
    JID_PROXY=$(sub "proxy: fit ensemble + section-7 gate (GPU)" $DEP \
        scripts/slurm/train_catalog_proxy_gpu.sbatch); die_if_aborted
    echo "  verdict -> $ROOT/runs/$RUN_NAME/proxy_gate.json"
    echo "  !! if it fails, IMPROVE THE PROXY OR ITS DATA. Unfreezing more of"
    echo "  !! SR2 is not a response to a proxy that cannot rank."
fi

# --- baseline: frozen field metrics, several seeds per box -----------------
JID_BASE=""
if [ "$STAGE" = "all" ] || [ "$STAGE" = "baseline" ]; then
    echo "=== frozen baseline field metrics (${#_SEEDS[@]} seeds x ${#_EVAL[@]} boxes)"
    SUB_OVERRIDES=("BOXES=${EVAL_BOXES// /,}" "SEEDS=${FROZEN_SEEDS// /,}")
    JID_BASE=$(sub "baseline: frozen SR2 whole-box metrics (GPU)" $(dep_of "$AFTER") \
        scripts/slurm/evaluate_sr2_direct_gpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()
fi

# --- calibrate: ENDS IN A HUMAN STEP ---------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "calibrate" ]; then
    echo "=== density-tolerance calibration"
    SUB_OVERRIDES=("STAGE=calibrate")
    sub "calibrate: frozen seed-to-seed spread -> proposal (CPU)" \
        $(dep_of "${JID_BASE:-$AFTER}") \
        scripts/slurm/score_sr2_direct_cpu.sbatch >/dev/null; die_if_aborted
    SUB_OVERRIDES=()
    echo "  !! STOP HERE. Paste $ROOT/runs/$RUN_NAME/gate_calibration.json's"
    echo "  !! proposal into the gates: block of $DIRECT_CFG, set"
    echo "  !! calibrated: true, commit, and only then run: bash $0 rung"
fi

# --- overfit: section 10, the small deliberate overfit ---------------------
if [ "$STAGE" = "overfit" ]; then
    echo "=== section 10: small overfit (1 box, host-rich tiles, proj_noise)"
    SUB_OVERRIDES=("OVERFIT=1")
    J=$(sub "overfit: short actor run (GPU)" $(dep_of "${JID_PROXY:-$AFTER}") \
        scripts/slurm/train_sr2_direct_gpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()

    CK="$ROOT/runs/$RUN_NAME/overfit/ema_generator.pt"
    SUB_OVERRIDES=("CHECKPOINT=$CK" "TAG=overfit" "BOXES=${FIT_BOXES%% *}"
                   "SEEDS=${FROZEN_SEEDS// /,}")
    JE=$(sub "overfit: whole-box field metrics (GPU)" --dependency=afterok:"$J" \
        scripts/slurm/evaluate_sr2_direct_gpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()

    SUB_OVERRIDES=("STAGE=gate" "TAG=overfit")
    sub "overfit: density gate (CPU)" --dependency=afterok:"$JE" \
        scripts/slurm/score_sr2_direct_cpu.sbatch >/dev/null; die_if_aborted
    SUB_OVERRIDES=()

    echo "  !! Then run REAL Rockstar on the generated box and check occupation:"
    echo "  !!   sbatch scripts/slurm/label_proxy_candidates_cpu.sbatch \\"
    echo "  !!       $ENVFILE BOX=${_FIT[0]} SOURCE=actor SEED=0 CHECKPOINT=$CK"
    echo "  !! Proceed only if the REAL result improves occupation in >= 2"
    echo "  !! reliable host bins including one of bins 2-3, with density"
    echo "  !! preserved. The proxy improving is not the criterion."
fi

# --- rung: one unfreezing stage, then evaluate and gate it -----------------
if [ "$STAGE" = "rung" ]; then
    echo "=== rung $RUNG"
    SUB_OVERRIDES=("RUNG=$RUNG")
    J=$(sub "rung $RUNG: actor training (GPU)" $(dep_of "${JID_PROXY:-$AFTER}") \
        scripts/slurm/train_sr2_direct_gpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()

    CK="${CHECKPOINT:-$ROOT/runs/$RUN_NAME/rung_$RUNG/ema_generator.pt}"
    T="${TAG:-rung_$RUNG}"
    SUB_OVERRIDES=("CHECKPOINT=$CK" "TAG=$T" "BOXES=${EVAL_BOXES// /,}"
                   "SEEDS=${FROZEN_SEEDS// /,}")
    JE=$(sub "rung $RUNG: whole-box field metrics (GPU)" --dependency=afterok:"$J" \
        scripts/slurm/evaluate_sr2_direct_gpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()

    SUB_OVERRIDES=("STAGE=gate" "TAG=$T")
    sub "rung $RUNG: relative density gate (CPU)" --dependency=afterok:"$JE" \
        scripts/slurm/score_sr2_direct_cpu.sbatch >/dev/null; die_if_aborted
    SUB_OVERRIDES=()
    echo "  verdict -> $ROOT/runs/$RUN_NAME/gate_$T.json"
    echo "  !! Advance to the next rung only after a REAL catalog improvement"
    echo "  !! and a passing gate, from the last FIELD-VALID checkpoint."
fi

# --- dagger: one dataset-aggregation round ---------------------------------
# Actor candidates are generated, labelled with real Rockstar, appended, and the
# proxy is REFIT and REVALIDATED -- never updated from the same minibatches as
# the actor.
if [ "$STAGE" = "dagger" ]; then
    echo "=== DAgger round: label actor candidates, refit + revalidate the proxy"
    CK="${CHECKPOINT:-$ROOT/runs/$RUN_NAME/rung_$RUNG/ema_generator.pt}"
    LAST=""
    for BOX in "${_FIT[@]}"; do
        SUB_OVERRIDES=("BOX=$BOX" "SOURCE=actor" "SEED=0" "CHECKPOINT=$CK")
        J=$(sub "dagger: generate actor candidate $BOX (GPU)" $(dep_of "$AFTER") \
            scripts/slurm/generate_proxy_candidates_gpu.sbatch); die_if_aborted
        LAST=$(sub "dagger: label actor candidate $BOX (CPU)" \
            --dependency=afterok:"$J" \
            scripts/slurm/label_proxy_candidates_cpu.sbatch); die_if_aborted
        SUB_OVERRIDES=()
    done
    sub "dagger: refit + revalidate the proxy (GPU)" --dependency=afterok:"$LAST" \
        scripts/slurm/train_catalog_proxy_gpu.sbatch >/dev/null; die_if_aborted
    echo "  !! Then continue the actor from the last FIELD-VALID checkpoint:"
    echo "  !!   bash $0 rung RUNG=$RUNG RESUME=<last valid>"
fi

echo "=== submitted. watch with: squeue -u \$USER -o '%.10i %.20j %.10T %R'"
