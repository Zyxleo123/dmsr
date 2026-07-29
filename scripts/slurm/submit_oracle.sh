#!/bin/bash
# Submitter for Experiments 0 and 1, plus the Gate-A prior training that
# Experiments 2-4 are blocked behind. It ONLY calls sbatch -- no work of its own.
#
#   bash scripts/slurm/submit_oracle.sh              # everything, in order
#   bash scripts/slurm/submit_oracle.sh exp1         # exp1 prereqs + chain
#   bash scripts/slurm/submit_oracle.sh exp0         # exp0 prereqs + chain
#   bash scripts/slurm/submit_oracle.sh gate_a       # just the prior training
#   SKIP_SMOKE=1 bash scripts/slurm/submit_oracle.sh gate_a   # skip the smoke gate
#   DRY=1 bash scripts/slurm/submit_oracle.sh        # print the commands, submit nothing
#
# If sbatch itself fails partway (e.g. QOSMaxSubmitJobPerUserLimit), the
# submitter aborts immediately instead of chaining a broken --dependency onto
# an empty job id. Resume the stage that was cut off once queue space frees up,
# passing the already-submitted member-id array's job id through DEPEND (find
# it with `sacct` or the printed "-> job N" lines above the failure):
#
#   DEPEND=23159 bash scripts/slurm/submit_oracle.sh exp1_run
#   DEPEND=23159 bash scripts/slurm/submit_oracle.sh exp0_analyse
#   DEPEND=none  bash scripts/slurm/submit_oracle.sh exp1_run   # if 23159 already finished
#
# Configuration goes into ONE timestamped env file passed as a POSITIONAL
# argument. Never `sbatch --export`: on this cluster any explicit export list
# makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login
# environment on the compute node, and the job is requeued and HELD with
# "(user env retrieval failed requeued held)", stranding every dependent job.
# Arguments travel through the job record untouched.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
STAGE="${1:-all}"
DRY="${DRY:-0}"

# Boxes: the val split. set12-set15 stay untouched -- set12 is the declared dev
# box but spending it here buys nothing, because the 16-box audit shows the
# upper host bins vary by only +/-8% between boxes, so box choice is a split
# hygiene decision rather than a statistical one.
BOXES="${BOXES:-set8 set9}"
ALPHAS="${ALPHAS:-0 0.25 0.5 1.0}"
MODES="${MODES:-disp vel both}"
KINDS="${KINDS:-targeted control exact_particles}"
N_VERIFY="${N_VERIFY:-12}"
# The tile-geometry reward model needs a box-level covariance, so it is fitted
# over every box that has HR tile summaries, not just the two dev boxes.
MODEL_BOXES="${MODEL_BOXES:-set0 set1 set2 set3 set4 set5 set6 set7 set8 set9 set10 set11}"

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/oracle_$(date +%Y%m%d_%H%M%S).env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_oracle.sh at $(date '+%F %T'); sourced by the
# job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$ROOT
REWARD_CFG=configs/reward/reward.yaml
BASE_SEED=0
BOXES="$BOXES"
ALPHAS="$ALPHAS"
MODES="$MODES"
KINDS="$KINDS"
N_VERIFY=$N_VERIFY
MODEL_BOXES="$MODEL_BOXES"
EOF
echo "env file: $ENVFILE"

# Submitting from inside an allocation would leak SLURM_* into the child jobs.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

# Per-job config overrides (e.g. one box/source pair) go in SUB_OVERRIDES and
# MUST be applied after ENVFILE, never before: _reward_common.sh processes its
# positional args strictly in order, so ENVFILE -- which sets BOXES to the full
# multi-box list for every OTHER stage -- would silently clobber an earlier
# "BOXES=set8" back to "set8 set9" if it came first. That exact bug made two
# supposedly-independent member-id jobs both resolve to the same (box, source)
# and race on one output directory, corrupting it -- caught by comparing their
# logs, not by any check in the pipeline itself. Set SUB_OVERRIDES right before
# the sub() call it applies to and reset it to () after; it is never inherited
# between calls.
#
# Declared here, globally, and always reset to () rather than unset: under
# `set -u`, "${SUB_OVERRIDES[@]:-}" on a genuinely unset array still expands to
# ONE empty-string argument (bash quirk, confirmed empirically), and that empty
# string fails _reward_common.sh's `[[ -r "$_arg" ]]` config-file check and
# aborts the job -- for every sub() call that has no override at all. A
# declared-but-empty array does not have this problem: "${arr[@]}" on it
# expands to zero words, so the plain (no ":-") form below is required, not
# optional.
SUB_OVERRIDES=()
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
    # A submit failure must ABORT, not return an empty id. Letting it through
    # produced `--dependency=afterok:` on the next job ("Job dependency
    # problem") and a half-submitted chain whose most important stage was the
    # one missing. `$(...)` runs sub in a subshell, so `set -e` in the parent
    # does not see the failure -- the trap below is what actually stops us.
    jid=$(sbatch --parsable "$@" "$ENVFILE" "${SUB_OVERRIDES[@]}") || rc=$?
    if [ -n "${rc:-}" ] || [ -z "$jid" ]; then
        echo "" >&2
        echo "!!! sbatch FAILED for: $label" >&2
        echo "!!! nothing further has been submitted." >&2
        echo "!!! If this was QOSMaxSubmitJobPerUserLimit, wait for jobs to" >&2
        echo "!!! drain and resume with the stage that failed, e.g." >&2
        echo "!!!     DEPEND=<particles-jobid> bash $0 exp1_run" >&2
        echo "!!!     DEPEND=<particles-jobid> bash $0 exp0_analyse" >&2
        echo "!!!     bash $0 gate_a" >&2
        rm -f "$ABORT_FLAG"
        touch "$ABORT_FLAG"
        echo ""
        return 1
    fi
    echo "  $label -> job $jid" >&2
    echo "$jid"
}

ABORT_FLAG="$ROOT/.submit_aborted.$$"
trap 'rm -f "$ABORT_FLAG"' EXIT

die_if_aborted() {
    if [ -e "$ABORT_FLAG" ]; then
        echo "aborting the submitter; see the message above." >&2
        exit 1
    fi
}

n_boxes=$(wc -w <<< "$BOXES")

# Number of Experiment-1 cells, computed the same way the job script does.
N_CELLS=$(BOXES="$BOXES" ALPHAS="$ALPHAS" MODES="$MODES" KINDS="$KINDS" python - <<'PY'
import os
boxes = os.environ["BOXES"].split()
alphas = [float(a) for a in os.environ["ALPHAS"].split()]
modes = os.environ["MODES"].split()
kinds = os.environ["KINDS"].split()
n = 0
for b in boxes:
    if 0.0 in alphas:
        n += 1
    for k in kinds:
        for m in modes:
            for a in alphas:
                if a == 0.0 or (k != "targeted" and a != 1.0):
                    continue
                n += 1
print(n)
PY
)
echo "Experiment 1 grid: $N_CELLS cells over $n_boxes box(es)"

# Submission ORDER is priority order, because a QOS submit cap truncates the
# tail. Experiment 1 is the headline experiment; Experiment 0's 24-task credit
# verification is a nice-to-have bound on an assumption. So Experiment 1 goes
# first. An earlier version had this backwards, and a
# QOSMaxSubmitJobPerUserLimit cost exactly the stage that mattered.
JID_TARGETS=""; JID_PARTS="${DEPEND:-}"; JID_DECOMP=""

# --- prerequisites, shared by both experiments -----------------------------
# Target selection must precede the member-id extraction: it writes the halo id
# list that pass reads, so the two experiments share one Rockstar run per box.
#
# The member-id stage is submitted as ONE JOB PER (box, source) PAIR, not one
# array of 4. Observed in practice: an array can be cancelled in its entirety
# with a single `scancel <array-id>`, and that is exactly what took out every
# pair simultaneously (all 4 tasks killed within 200ms of each other) the one
# time an external SIGTERM hit this stage -- cause unconfirmed (sacct/sacctmgr
# could not reach the accounting daemon to check), but four independent job ids
# confine any repeat to the one pair actually affected, and everything else in
# the chain still needs "did every pair succeed" rather than "did the array
# survive", so this costs nothing downstream.
if [[ "$STAGE" =~ ^(all|exp0|exp1)$ ]]; then
    JID_TARGETS=$(sub "exp1 select targets" scripts/slurm/exp1_select_targets_cpu.sbatch)
    die_if_aborted
    PART_IDS=()
    for BOX in $BOXES; do
        for SRC in hr base; do
            SUB_OVERRIDES=("BOXES=$BOX" "SOURCES=$SRC")
            JP=$(sub "exp0 rockstar+particles ($BOX/$SRC)" \
                --dependency=afterok:"$JID_TARGETS" \
                scripts/slurm/exp0_rockstar_particles_cpu.sbatch)
            SUB_OVERRIDES=()
            die_if_aborted
            PART_IDS+=("$JP")
        done
    done
    JID_PARTS=$(IFS=:; echo "${PART_IDS[*]}")
fi

# Resuming after a submit cap: the prerequisite jobs already exist, so their
# colon-separated ids come in through DEPEND instead of being submitted again.
if [[ "$STAGE" =~ ^(exp1_run|exp0_analyse)$ ]] && [ -z "$JID_PARTS" ]; then
    echo "!! $STAGE resumes a partly-submitted chain and needs the job id(s) of" >&2
    echo "!! the member-id stage, colon-separated if more than one:" >&2
    echo "!!     DEPEND=23166:23167:23168:23159 bash $0 $STAGE" >&2
    echo "!! Use DEPEND=none if those jobs have already finished." >&2
    exit 1
fi
DEP=()
if [ -n "$JID_PARTS" ] && [ "$JID_PARTS" != "none" ]; then
    DEP=(--dependency=afterok:"$JID_PARTS")
fi

# --- Experiment 1: the interventions and the report ------------------------
if [[ "$STAGE" =~ ^(all|exp1|exp1_run)$ ]]; then
    JID_INT=$(sub "exp1 interventions" "${DEP[@]}" --array=0-$((N_CELLS - 1)) \
        scripts/slurm/exp1_intervene_cpu.sbatch)
    die_if_aborted
    sub "exp1 report" --dependency=afterany:"$JID_INT" \
        scripts/slurm/exp1_report_cpu.sbatch >/dev/null
    die_if_aborted
fi

# --- Experiment 0: the table, the credit, and the verification plan --------
# Sibling of the Experiment-1 chain, not a child of it: both need only the
# member-id job, so they queue concurrently.
if [[ "$STAGE" =~ ^(all|exp0|exp0_analyse)$ ]]; then
    JID_DECOMP=$(sub "exp0 decompose + credit" "${DEP[@]}" \
        scripts/slurm/exp0_decompose_cpu.sbatch)
    die_if_aborted

    for BOX in $BOXES; do
        JV=$(sub "exp0 verify splices ($BOX)" \
            --dependency=afterok:"$JID_DECOMP" --array=0-$((N_VERIFY - 1)) \
            scripts/slurm/exp0_credit_verify_cpu.sbatch "BOX=$BOX")
        die_if_aborted
        sub "exp0 verify aggregate ($BOX)" \
            --dependency=afterok:"$JV" \
            scripts/slurm/exp0_credit_verify_cpu.sbatch "BOX=$BOX" AGGREGATE=1 \
            >/dev/null
        die_if_aborted
    done
fi

# --- Gate A: the residual prior Experiments 2-4 are blocked behind ---------
# Independent of everything above, so it starts training immediately. If
# Experiment 1 comes back negative, scancel it -- that is the whole reason it is
# a sibling rather than a dependent.
if [[ "$STAGE" =~ ^(all|gate_a)$ ]]; then
    if [ -r scripts/slurm/smoke_residual_prior_gpu.sbatch ]; then
        if [ "${SKIP_SMOKE:-0}" = "1" ]; then
            echo "  !! SKIP_SMOKE=1: submitting gate A training with no smoke gate" >&2
            sub "gate A prior training (GPU)" \
                scripts/slurm/train_residual_prior.sbatch >/dev/null
        else
            JS=$(sub "gate A smoke (GPU)" scripts/slurm/smoke_residual_prior_gpu.sbatch)
            sub "gate A prior training (GPU)" --dependency=afterok:"$JS" \
                scripts/slurm/train_residual_prior.sbatch >/dev/null
        fi
    else
        echo "  !! no residual-prior job scripts found; skipping Gate A" >&2
    fi
fi

echo
echo "queue:   squeue -u \$USER -o '%.9i %.14j %.2t %.10M %R'"
echo "logs:    $ROOT/logs"
echo "results: $ROOT/audits/tile_decomposition  and  $ROOT/oracle_hr"
