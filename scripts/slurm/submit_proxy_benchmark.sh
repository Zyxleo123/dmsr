#!/bin/bash
# Phases 4-6: fit both arms, gate both arms, verify with real splices, decide.
# This submitter ONLY calls sbatch. It stops at proxy_benchmark.json and has no
# path into SR2 fine-tuning: that decision is what the benchmark is for.
#
#   DRY=1 bash scripts/slurm/submit_proxy_benchmark.sh all
#   bash scripts/slurm/submit_proxy_benchmark.sh fit       # train + gate both arms
#   bash scripts/slurm/submit_proxy_benchmark.sh splice    # select + 12 runs per arm
#   bash scripts/slurm/submit_proxy_benchmark.sh decide    # re-gate + benchmark
#   bash scripts/slurm/submit_proxy_benchmark.sh all ARMS="a"
#
# WHY IT IS THREE STAGES AND NOT ONE CHAIN
# ----------------------------------------
# `all` does chain them, because the ordering is mechanical. But `splice` is
# separable on purpose: its plan is twelve full-box Rockstar runs per arm chosen
# by the proxy, and it is worth reading splice_plan_<arm>.json before spending
# them. Nothing is lost by looking -- the plan is written by a job that takes
# seconds.
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
ARMS="${ARMS:-a b}"
N_SPLICES="${N_SPLICES:-12}"
AFTER="${AFTER:-}"

for kv in "$@"; do
    case "$kv" in *=*) export "$kv" ;; esac
done
read -r -a _ARMS <<< "${ARMS//,/ }"

MARK="$ROOT/proxy_data/labels_complete.json"
if [ ! -r "$MARK" ] && [ "$DRY" != "1" ]; then
    echo "!!! $MARK does not exist, so labelling is not complete and any table" >&2
    echo "!!! on disk is partial. Nothing has been submitted." >&2
    echo "!!! Run the label workflow first:" >&2
    echo "!!!   bash scripts/slurm/submit_proxy_labels.sh all" >&2
    echo "!!! and read $ROOT/proxy_data/index_report.json for what is missing." >&2
    exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/bench_$(date +%Y%m%d_%H%M%S).env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_proxy_benchmark.sh at $(date '+%F %T'); sourced
# by the job preamble as a positional argument.
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

for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

SUB_OVERRIDES=()
ABORT_FLAG="$ROOT/.submit_proxy_benchmark_aborted.$$"
trap 'rm -f "$ABORT_FLAG"' EXIT

sub() {   # sub <human label> <sbatch args...>
    local label="$1"; shift
    if [ "$DRY" = "1" ]; then
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

# --- fit: both arms and both gates, in one job -----------------------------
JID_FIT=""
if [ "$STAGE" = "all" ] || [ "$STAGE" = "fit" ]; then
    echo "=== fit + gate both arms ($RUN_NAME)"
    SUB_OVERRIDES=("ARMS=${ARMS// /,}")
    JID_FIT=$(sub "proxy: CV, both arms, both gates (GPU)" $(dep_of "$AFTER") \
        scripts/slurm/train_catalog_proxy_gpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()
    echo "  ensembles -> $ROOT/runs/$RUN_NAME/proxy_<arm>/"
    echo "  verdicts  -> $ROOT/runs/$RUN_NAME/proxy_gate_<arm>.json"
    echo "  !! the splice criterion is EXPECTED to be unmet at this point."
fi

# --- splice: select, then one array task per splice, per arm ---------------
SPLICE_JIDS=()
if [ "$STAGE" = "all" ] || [ "$STAGE" = "splice" ]; then
    echo "=== real splice verification ($N_SPLICES per arm)"
    for ARM in "${_ARMS[@]}"; do
        SUB_OVERRIDES=("ARM=$ARM")
        JS=$(sub "splice $ARM: choose the plan (CPU)" \
            $(dep_of "${JID_FIT:-$AFTER}") \
            scripts/slurm/splice_select_cpu.sbatch); die_if_aborted
        # One array, not N jobs: the array index IS the plan index, and a task
        # beyond the plan prints so and exits 0 rather than failing.
        SPLICE_JIDS+=("$(sub "splice $ARM: $N_SPLICES real Rockstar runs (CPU array)" \
            --array=0-$((N_SPLICES - 1)) --dependency=afterok:"$JS" \
            scripts/slurm/splice_verify_cpu.sbatch)"); die_if_aborted
        SUB_OVERRIDES=()
    done
    echo "  plan    -> $ROOT/runs/$RUN_NAME/splice_plan_<arm>.json"
    echo "  results -> $ROOT/runs/$RUN_NAME/splice_verification_<arm>.jsonl"
fi

# --- decide: re-gate with the splices in place, write the benchmark --------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "decide" ]; then
    DEP=""
    if [ "${#SPLICE_JIDS[@]}" -gt 0 ]; then
        # afterany: a splice that failed is missing evidence, and the gate
        # reports missing evidence as an unmet criterion. afterok would strand
        # the benchmark instead and say nothing.
        DEP="--dependency=afterany:$(IFS=:; echo "${SPLICE_JIDS[*]}")"
    elif [ -n "${JID_FIT:-$AFTER}" ]; then
        DEP="--dependency=afterany:${JID_FIT:-$AFTER}"
    fi
    echo "=== decision"
    SUB_OVERRIDES=("ARMS=${ARMS// /,}")
    sub "benchmark: re-gate both arms, write proxy_benchmark.json (CPU)" \
        $DEP scripts/slurm/proxy_benchmark_cpu.sbatch >/dev/null; die_if_aborted
    SUB_OVERRIDES=()
    echo "  record -> $ROOT/runs/$RUN_NAME/proxy_benchmark.json"
fi

echo "=== submitted. watch with: squeue -u \$USER -o '%.10i %.20j %.10T %R'"
echo "=== This milestone ENDS at proxy_benchmark.json. Read its decision before"
echo "=== any SR2 parameter is allowed to move."
