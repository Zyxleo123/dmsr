#!/bin/bash
# Submitter for rung 2 (CEM search over the diffusion noise) and rung 3
# (fixed-host reward overfitting) of docs/reward_residual_diffusion.md §5.
# It ONLY calls sbatch -- no work of its own.
#
#   bash scripts/slurm/submit_cem.sh              # both rungs
#   bash scripts/slurm/submit_cem.sh cem          # just the CEM chain
#   bash scripts/slurm/submit_cem.sh overfit      # just the fixed-host rung
#   ITERS=3 CEM_RUN=cem_b bash scripts/slurm/submit_cem.sh cem
#   DRY=1 bash scripts/slurm/submit_cem.sh        # print the commands, submit nothing
#
# Both rungs are independent of the in-flight Gate B job -- they answer
# different questions and read only the trained prior, so nothing here depends
# on Gate B's verdict. The scheduler handles GPU contention.
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
ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
STAGE="${1:-all}"
DRY="${DRY:-0}"

CEM_RUN="${CEM_RUN:-cem_a}"
ITERS="${ITERS:-5}"
CEM_CFG="${CEM_CFG:-configs/reward/cem.yaml}"
SCORE_SHARDS="${SCORE_SHARDS:-7}"     # array is 0-$SCORE_SHARDS
OVERFIT_RUN="${OVERFIT_RUN:-overfit_hosts}"
OVERFIT_CFG="${OVERFIT_CFG:-configs/reward/fixed_host_overfit.yaml}"
# A TRAIN box: these chunks become training crops, so a val box here would put
# validation data in the training set. select_fixed_hosts.py refuses anyway.
OVERFIT_BOXES="${OVERFIT_BOXES:-set0}"

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/cem_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_cem.sh at $(date '+%F %T'); sourced by the
# job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$ROOT
REWARD_CFG=configs/reward/reward.yaml
CEM_CFG=$CEM_CFG
CEM_RUN=$CEM_RUN
BASE_SEED=0
EOF
echo "env file: $ENVFILE"

# Submitting from inside an allocation would leak SLURM_* into the child jobs.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

# Per-job overrides MUST come after ENVFILE: _reward_common.sh processes its
# positional args strictly in order, so ENVFILE would clobber an earlier
# setting. Declared globally and reset to () -- never unset, because under
# `set -u` an unset array still expands to one empty argument, which fails the
# preamble's config-file check.
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
    jid=$(sbatch --parsable "$@" "$ENVFILE" "${SUB_OVERRIDES[@]}") || rc=$?
    if [ -n "${rc:-}" ] || [ -z "$jid" ]; then
        echo "" >&2
        echo "!!! sbatch FAILED for: $label" >&2
        echo "!!! nothing further has been submitted." >&2
        echo "!!! If this was QOSMaxSubmitJobPerUserLimit, wait for jobs to" >&2
        echo "!!! drain and resubmit; already-generated iterations are kept," >&2
        echo "!!! so re-running this submitter resumes rather than restarts." >&2
        rm -f "$ABORT_FLAG"
        touch "$ABORT_FLAG"
        echo ""
        return 1
    fi
    echo "  $label -> job $jid" >&2
    echo "$jid"
}

ABORT_FLAG="$ROOT/.submit_cem_aborted.$$"
trap 'rm -f "$ABORT_FLAG"' EXIT

die_if_aborted() {
    if [ -e "$ABORT_FLAG" ]; then
        echo "aborting the submitter; see the message above." >&2
        exit 1
    fi
}

# --- Rung 2: CEM search over the diffusion noise ---------------------------
# ITERS is known at submit time, so the whole chain goes in now:
#   gen_0 -> score_0 -> select_0 -> gen_1 -> ...
# Each iteration's generate job depends on the previous SELECT job, because the
# elites it perturbs are that job's output. A gate that cannot proceed exits 0
# with an explanation rather than stranding the rest on DependencyNeverSatisfied.
if [ "$STAGE" = "all" ] || [ "$STAGE" = "cem" ]; then
    echo "=== rung 2: CEM ($CEM_RUN, $ITERS iterations)"
    PREV=""
    for ((i = 0; i < ITERS; i++)); do
        RUN_IT="${CEM_RUN}_it${i}"

        SUB_OVERRIDES=("ITER=$i")
        if [ -z "$PREV" ]; then
            JID_GEN=$(sub "cem it$i: generate population (GPU)" \
                scripts/slurm/cem_generate_gpu.sbatch)
        else
            JID_GEN=$(sub "cem it$i: generate population (GPU)" \
                --dependency=afterok:"$PREV" \
                scripts/slurm/cem_generate_gpu.sbatch)
        fi
        SUB_OVERRIDES=()
        die_if_aborted

        # Scoring is the ordinary oracle CPU stage, unmodified -- it is already
        # driven entirely by RUN_NAME.
        SUB_OVERRIDES=("RUN_NAME=$RUN_IT")
        JID_SCORE=$(sub "cem it$i: score population (CPU array)" \
            --dependency=afterok:"$JID_GEN" --array=0-"$SCORE_SHARDS" \
            scripts/slurm/catalog_reward_oracle_cpu.sbatch)
        SUB_OVERRIDES=()
        die_if_aborted

        SUB_OVERRIDES=("ITER=$i")
        PREV=$(sub "cem it$i: rank and select elites (CPU)" \
            --dependency=afterok:"$JID_SCORE" \
            scripts/slurm/cem_select_elites_cpu.sbatch)
        SUB_OVERRIDES=()
        die_if_aborted
    done
    echo "  elites per iteration -> $ROOT/oracle/${CEM_RUN}_it<i>/elites.json"
fi

# --- Rung 3: fixed-host reward overfitting ---------------------------------
# Deliberately overfit a handful of massive hosts. If reward cannot be raised
# even here the problem is representational, not statistical -- which is a
# different conclusion from "search did not find it", and the reason this runs
# alongside the CEM chain rather than after it.
if [ "$STAGE" = "all" ] || [ "$STAGE" = "overfit" ]; then
    echo "=== rung 3: fixed-host overfit ($OVERFIT_RUN)"
    SUB_OVERRIDES=("OVERFIT_BOXES=$OVERFIT_BOXES")
    JID_SEL=$(sub "overfit: pick fixed hosts (CPU)" \
        scripts/slurm/select_fixed_hosts_cpu.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted

    SUB_OVERRIDES=("OVERFIT_RUN=$OVERFIT_RUN" "OVERFIT_CFG=$OVERFIT_CFG")
    sub "overfit: train on fixed hosts (GPU)" \
        --dependency=afterok:"$JID_SEL" \
        scripts/slurm/train_residual_overfit_gpu.sbatch >/dev/null
    SUB_OVERRIDES=()
    die_if_aborted
fi

echo "=== submitted. watch with: squeue -u \$USER -o '%.10i %.20j %.10T %R'"
