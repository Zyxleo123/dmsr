#!/bin/bash
# Submitter for the host-conditioned local editor. It ONLY calls sbatch.
#
#   DRY=1 bash scripts/slurm/submit_local_editor.sh              # print, submit nothing
#   bash scripts/slurm/submit_local_editor.sh setup              # hosts + member ids
#   bash scripts/slurm/submit_local_editor.sh audit              # stage 0 calibration
#   bash scripts/slurm/submit_local_editor.sh gate1              # random + controls -> Gate 1
#   bash scripts/slurm/submit_local_editor.sh cem                # the CEM chain
#   bash scripts/slurm/submit_local_editor.sh distill            # flow + GMM + their arms
#   bash scripts/slurm/submit_local_editor.sh eval               # the comparison table
#
# Ordering, and why it is not one chain:
#   setup -> audit -> (paste thresholds, set calibrated: true) -> gate1 -> cem
#   -> distill -> eval
# The audit deliberately ends in a HUMAN step. A chain that ran through it would
# be scoring against placeholder thresholds, which is the exact failure mode
# `constraints.calibrated: false` exists to prevent, so the submitter stops
# there rather than pretending the gap is not real.
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
ROOT="${DMSR_LOCAL_EDITOR_ROOT:-$ZFS/DMSR/dmsr_local_editor}"
STAGE="${1:-all}"
DRY="${DRY:-0}"

RUN_NAME="${RUN_NAME:-le_a}"
LE_CFG="${LE_CFG:-configs/reward/local_editor.yaml}"
BOXES="${BOXES:-set8 set9}"
ROUNDS="${ROUNDS:-3}"
CANDIDATES="${CANDIDATES:-28}"
# Shards per (box, round). Each shard owns a disjoint slice of candidate
# indices, so 4 shards over 28 candidates is 7 Rockstar runs each, ~2 h.
SHARDS="${SHARDS:-3}"          # array is 0-$SHARDS
GATE1_CANDIDATES="${GATE1_CANDIDATES:-12}"
# Chain a stage onto a job submitted by an EARLIER invocation of this script.
# Stages run separately have no dependency on each other, and the downstream
# job would then find its input missing and gate out with exit 0 -- which looks
# like a clean run that did nothing. AFTER=<jobid> makes the intended ordering
# explicit instead of relying on the operator watching squeue.
AFTER="${AFTER:-}"

read -r -a _BOXES <<< "${BOXES//,/ }"
N_BOXES=${#_BOXES[@]}

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/le_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_local_editor.sh at $(date '+%F %T'); sourced
# by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_LOCAL_EDITOR_ROOT=$ROOT
DMSR_REWARD_ROOT=${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}
LE_CFG=$LE_CFG
REWARD_CFG=configs/reward/reward.yaml
RUN_NAME=$RUN_NAME
BASE_SEED=0
SEED=0
EOF
echo "env file: $ENVFILE"

# Submitting from inside an allocation would leak SLURM_* into the child jobs.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

# Per-job overrides MUST come after ENVFILE: the preamble processes positional
# args strictly in order, so ENVFILE would clobber an earlier setting. Declared
# globally and reset to () -- never unset, because under `set -u` an unset array
# expands to one empty argument, which fails the preamble's config-file check.
SUB_OVERRIDES=()
ABORT_FLAG="$ROOT/.submit_local_editor_aborted.$$"
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
        echo "!!! nothing further has been submitted. Already-completed stages" >&2
        echo "!!! are kept (every stage skips work it already did), so"          >&2
        echo "!!! re-running this submitter resumes rather than restarts."       >&2
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

# --- setup: hosts, then member ids -----------------------------------------
JID_SETUP=""
if [ "$STAGE" = "all" ] || [ "$STAGE" = "setup" ]; then
    echo "=== setup ($RUN_NAME, boxes: $BOXES)"
    SUB_OVERRIDES=("BOXES=${BOXES// /,}")
    JID_H=$(sub "stage 1: select hosts (CPU)" \
        scripts/slurm/local_editor_select_hosts_cpu.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted

    SUB_OVERRIDES=("BOXES=${BOXES// /,}")
    JID_SETUP=$(sub "stage 2: Rockstar member ids (CPU array, ~1-2 h/box)" \
        --dependency=afterok:"$JID_H" --array=0-$((N_BOXES - 1)) \
        scripts/slurm/local_editor_members_cpu.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted
fi

# The dependency a stage's first job carries: this invocation's own setup job if
# there was one, else an explicitly supplied AFTER=<jobid>, else none.
dep() {
    if [ -n "$JID_SETUP" ]; then echo "--dependency=afterok:$JID_SETUP"
    elif [ -n "$AFTER" ]; then echo "--dependency=afterok:$AFTER"
    else echo ""; fi
}

# --- audit: measure-only candidates, then calibrate ------------------------
# This stage ENDS IN A HUMAN STEP. Nothing downstream is submitted here.
if [ "$STAGE" = "all" ] || [ "$STAGE" = "audit" ]; then
    echo "=== stage 0: constraint calibration"
    PREV=""
    for BOX in "${_BOXES[@]}"; do
        SUB_OVERRIDES=("BOX=$BOX" "ARM=frozen" "CANDIDATES=1" "MEASURE_ONLY=1"
                       "WITH_DENSITY=1")
        JID=$(sub "audit: frozen anchor $BOX (CPU)" \
            $(dep) --array=0-0 scripts/slurm/local_editor_candidates_cpu.sbatch)
        SUB_OVERRIDES=()
        die_if_aborted

        SUB_OVERRIDES=("BOX=$BOX" "ARM=random" "CANDIDATES=8" "MEASURE_ONLY=1"
                       "WITH_DENSITY=1")
        PREV=$(sub "audit: measure-only editor candidates $BOX (CPU array)" \
            --dependency=afterok:"$JID" --array=0-1 \
            scripts/slurm/local_editor_candidates_cpu.sbatch)
        SUB_OVERRIDES=()
        die_if_aborted
    done

    SUB_OVERRIDES=("BOXES=${BOXES// /,}")
    sub "audit: propose thresholds (CPU)" \
        --dependency=afterok:"$PREV" scripts/slurm/local_editor_audit_cpu.sbatch \
        >/dev/null
    SUB_OVERRIDES=()
    die_if_aborted
    echo "  !! STOP HERE. Paste $ROOT/audit/constraints_proposal.json into"
    echo "  !! configs/reward/local_editor.yaml, set calibrated: true, commit,"
    echo "  !! and only then run:  bash $0 gate1"
fi

# --- gate1: the deployment-legal action-space oracle -----------------------
# Random actions plus all three controls, in both boxes, then the verdict. Small
# on purpose: Gate 1 asks whether the action space contains anything at all, and
# a full CEM budget spent before that question is answered is a full CEM budget
# at risk.
if [ "$STAGE" = "all" ] || [ "$STAGE" = "gate1" ]; then
    echo "=== stage 3: Gate 1 (random + controls)"
    LAST=""
    for BOX in "${_BOXES[@]}"; do
        for SPEC in "random:none" "random:random_particles" \
                    "random:shuffled_host" "random:near_subhalo"; do
            ARM="${SPEC%%:*}"; CTL="${SPEC##*:}"
            SUB_OVERRIDES=("BOX=$BOX" "ARM=$ARM" "CONTROL=$CTL"
                           "CANDIDATES=$GATE1_CANDIDATES")
            LAST=$(sub "gate1: $BOX $ARM/$CTL (CPU array)" \
                $(dep) --array=0-"$SHARDS" \
                scripts/slurm/local_editor_candidates_cpu.sbatch)
            SUB_OVERRIDES=()
            die_if_aborted
        done
    done
    SUB_OVERRIDES=("STAGE=aggregate" "ROUND=0" "BOXES=${BOXES// /,}")
    sub "gate1: verdict (CPU)" --dependency=afterok:"$LAST" \
        scripts/slurm/local_editor_cem_cpu.sbatch >/dev/null
    SUB_OVERRIDES=()
    die_if_aborted
    echo "  verdict -> $ROOT/runs/$RUN_NAME/gate1.json"
fi

# --- cem: propose -> score -> aggregate, per round -------------------------
# The whole chain goes in now because ROUNDS is known at submit time. Each
# round's propose job depends on the previous round's aggregate, because the
# distribution it samples from is that job's output.
if [ "$STAGE" = "all" ] || [ "$STAGE" = "cem" ]; then
    echo "=== stage 4: CEM ($ROUNDS rounds x $CANDIDATES candidates x $N_BOXES boxes)"
    PREV="${JID_SETUP:-$AFTER}"
    for ((r = 0; r < ROUNDS; r++)); do
        ACTIONS="$ROOT/runs/$RUN_NAME/cem/round_$(printf '%03d' "$r")_actions.json"

        SUB_OVERRIDES=("STAGE=propose" "ROUND=$r" "CANDIDATES=$CANDIDATES")
        if [ -n "$PREV" ]; then
            JID_P=$(sub "cem r$r: propose actions (CPU)" \
                --dependency=afterok:"$PREV" scripts/slurm/local_editor_cem_cpu.sbatch)
        else
            JID_P=$(sub "cem r$r: propose actions (CPU)" \
                scripts/slurm/local_editor_cem_cpu.sbatch)
        fi
        SUB_OVERRIDES=()
        die_if_aborted

        SCORE=()
        for BOX in "${_BOXES[@]}"; do
            SUB_OVERRIDES=("BOX=$BOX" "ARM=cem" "ROUND=$r"
                           "ACTIONS_JSON=$ACTIONS" "CANDIDATES=$CANDIDATES")
            J=$(sub "cem r$r: score $BOX (CPU array)" \
                --dependency=afterok:"$JID_P" --array=0-"$SHARDS" \
                scripts/slurm/local_editor_candidates_cpu.sbatch)
            SUB_OVERRIDES=()
            die_if_aborted
            SCORE+=("$J")
        done

        DEPS=$(IFS=:; echo "${SCORE[*]}")
        SUB_OVERRIDES=("STAGE=aggregate" "ROUND=$r" "CANDIDATES=$CANDIDATES"
                       "BOXES=${BOXES// /,}")
        PREV=$(sub "cem r$r: aggregate + Gate 1 (CPU)" \
            --dependency=afterok:"$DEPS" scripts/slurm/local_editor_cem_cpu.sbatch)
        SUB_OVERRIDES=()
        die_if_aborted
    done
    echo "  rounds -> $ROOT/runs/$RUN_NAME/round_*_summary.json"
fi

# --- distill: flow + GMM, then their candidate arms ------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "distill" ]; then
    echo "=== stage 5: distillation"
    SUB_OVERRIDES=("BOXES=${BOXES// /,}")
    JID_F=$(sub "stage 5: train flow + GMM baseline (GPU)" \
        $(dep) scripts/slurm/local_editor_flow_gpu.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted

    for POLICY in flow gmm; do
        SUB_OVERRIDES=("POLICY=$POLICY" "N_CANDIDATES=$CANDIDATES"
                       "BOXES=${BOXES// /,}")
        JID_S=$(sub "stage 5b: sample $POLICY actions (CPU)" \
            --dependency=afterok:"$JID_F" scripts/slurm/local_editor_sample_cpu.sbatch)
        SUB_OVERRIDES=()
        die_if_aborted
        for BOX in "${_BOXES[@]}"; do
            SUB_OVERRIDES=("BOX=$BOX" "ARM=$POLICY" "ROUND=99"
                           "CANDIDATES=$CANDIDATES"
                           "ACTIONS_JSON=$ROOT/runs/$RUN_NAME/actions/${POLICY}_actions.json")
            sub "stage 5b: score $POLICY on $BOX (CPU array)" \
                --dependency=afterok:"$JID_S" --array=0-"$SHARDS" \
                scripts/slurm/local_editor_candidates_cpu.sbatch >/dev/null
            SUB_OVERRIDES=()
            die_if_aborted
        done
    done
fi

# --- eval ------------------------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "eval" ]; then
    echo "=== final comparison"
    SUB_OVERRIDES=("BOXES=${BOXES// /,}")
    sub "eval: six-arm table (CPU)" scripts/slurm/local_editor_eval_cpu.sbatch \
        >/dev/null
    SUB_OVERRIDES=()
    die_if_aborted
    echo "  table -> $ROOT/runs/$RUN_NAME/final_comparison.json"
    echo "  set13/14/15 stay closed until you run this with FINAL=1 explicitly."
fi

echo "=== submitted. watch with: squeue -u \$USER -o '%.10i %.20j %.10T %R'"
