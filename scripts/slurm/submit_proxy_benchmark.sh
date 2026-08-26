#!/bin/bash
# Phases 4-6: fit the arms, gate them offline, verify the passing arms with the
# ACTOR-LIKE Rockstar gate (probe fine-tune -> real splices), decide.
# This submitter ONLY calls sbatch. It stops at proxy_benchmark.json and has no
# path into production SR2 fine-tuning: that decision is what the benchmark is
# for. The probe runs it submits are capped, stamped probe_only, and refused as
# a resume point by the production trainer.
#
#   DRY=1 bash scripts/slurm/submit_proxy_benchmark.sh all
#   bash scripts/slurm/submit_proxy_benchmark.sh fit     # train + gate all arms
#   bash scripts/slurm/submit_proxy_benchmark.sh probe   # probe + 12 checks/arm
#   bash scripts/slurm/submit_proxy_benchmark.sh decide  # re-gate + benchmark
#   bash scripts/slurm/submit_proxy_benchmark.sh probe ARMS="b"
#
# WHY IT IS THREE STAGES AND NOT ONE CHAIN
# ----------------------------------------
# `all` does chain them, because the ordering is mechanical. But `probe` is
# separable on purpose, twice over: it should be pointed at the arms whose
# OFFLINE gate passed (read proxy_gate_<arm>.json after `fit`, then run `probe`
# with ARMS set to the survivors), and its plan is twelve full-box Rockstar
# runs per arm chosen by the proxy, worth reading (actor_plan_<arm>.json)
# before spending them. Nothing is lost by looking -- the plan job is minutes.
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
ARMS="${ARMS:-a b c d e f}"
N_CHECKS="${N_CHECKS:-12}"
# Field metrics for the probe gate are measured where the splices happen: the
# held-out verify boxes (actor_gate.verify_boxes in the config).
VERIFY_BOXES="${VERIFY_BOXES:-set8,set9,set10}"
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
# Schema must match the code. A stale marker (e.g. schema 2 while the trainer
# expects 3) used to let fit jobs start and die in seconds on every GPU; refuse
# up front so the backfill/reindex is the obvious next step.
if [ -r "$MARK" ] && [ "$DRY" != "1" ]; then
    _schema=$(python - "$MARK" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("src")))
from cosmo_sr.reward.arms import FEATURE_SCHEMA_VERSION
doc = json.loads(Path(sys.argv[1]).read_text())
got = int(doc.get("feature_schema_version", 1))
print(got)
raise SystemExit(0 if got == int(FEATURE_SCHEMA_VERSION) else 2)
PY
) || {
        echo "!!! $MARK is feature schema ${_schema:-?}, but the code expects a" >&2
        echo "!!! newer schema (grids_e + field_changed). Fit would fail instantly." >&2
        echo "!!! Finish the features-only backfill and re-index first:" >&2
        echo "!!!   bash scripts/slurm/submit_proxy_labels.sh all" >&2
        echo "!!! Then re-run this submitter once labels_complete.json shows the" >&2
        echo "!!! current feature_schema_version and sidecars include grids_e.npy." >&2
        exit 1
    }
fi

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/bench_$(date +%Y%m%d_%H%M%S)_$$.env"
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

# Optional node/partition targeting, following the myfree discipline: run myfree,
# pick a node whose NOTE says AVAILABLE, and pin to it and its partition/GPU.
#   NODE=gpu30 bash scripts/slurm/submit_proxy_benchmark.sh fit
# These become sbatch CLI args (which override the script's #SBATCH lines) and
# travel outside the env file, so they never trigger the user-env-retrieval bug.
# PARTITION/QOS default to the sbatch script's own general/qos_general.
NODE_ARGS=()
[ -n "${NODE:-}" ]      && NODE_ARGS+=("--nodelist=$NODE")
[ -n "${PARTITION:-}" ] && NODE_ARGS+=("--partition=$PARTITION")
[ -n "${QOS:-}" ]       && NODE_ARGS+=("--qos=$QOS")

sub() {   # sub <human label> <sbatch args...>
    local label="$1"; shift
    if [ "$DRY" = "1" ]; then
        echo "DRY  $label:" >&2
        echo "       sbatch ${NODE_ARGS[*]} $* $ENVFILE ${SUB_OVERRIDES[*]}" >&2
        echo "000000"
        return 0
    fi
    local jid rc
    jid=$(sbatch --parsable "${NODE_ARGS[@]}" "$@" "$ENVFILE" "${SUB_OVERRIDES[@]}") || rc=$?
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

# --- fit: one GPU job per arm (train + offline gate), all siblings ----------
# Arms used to share one process so bootstrap draws and the table were
# identical by construction. Parallel jobs keep that: the trainer stamps
# bootstrap_draws.json under a flock on first write, and every later arm reuses
# it; baselines/train_report merges are flocked too. Wall-clock then scales with
# the slowest arm (E/F), not the sum of A..F.
JID_FIT=""
FIT_JIDS=()
declare -A FIT_BY_ARM=()
if [ "$STAGE" = "all" ] || [ "$STAGE" = "fit" ]; then
    echo "=== fit + offline-gate arms (1 GPU job each): $ARMS ($RUN_NAME)"
    for ARM in "${_ARMS[@]}"; do
        SUB_OVERRIDES=("ARMS=$ARM")
        jid=$(sub "proxy: fit + offline gate arm $ARM (GPU)" \
            $(dep_of "$AFTER") --job-name="sd_proxy_${ARM}" \
            scripts/slurm/train_catalog_proxy_gpu.sbatch)
        die_if_aborted
        FIT_JIDS+=("$jid")
        FIT_BY_ARM[$ARM]="$jid"
        SUB_OVERRIDES=()
    done
    JID_FIT=$(IFS=:; echo "${FIT_JIDS[*]}")
    echo "  ensembles -> $ROOT/runs/$RUN_NAME/proxy_<arm>/"
    echo "  verdicts  -> $ROOT/runs/$RUN_NAME/proxy_gate_<arm>.json"
    echo "  fit jids  -> $JID_FIT"
    echo "  !! the actor criterion is EXPECTED to be unmet at this point;"
    echo "  !! read offline_passed, then run the probe stage on the survivors."
fi

# --- probe: probe fine-tune, field gates, select, then the Rockstar array ---
# Per arm: train a capped probe_only checkpoint on the FIT boxes, measure its
# fields on the verify boxes against the frozen baseline, gate them, generate
# the verify boxes and pick the stratified plan, then one Rockstar run per
# check. The probe trainer itself refuses arms whose offline gate failed, so
# pointing this at every arm wastes at most a queued job, never a Rockstar run.
# Each arm's probe chain waits only on THAT arm's fit job (not the slowest
# sibling), so a finished A can probe while E is still training.
ACTOR_JIDS=()
if [ "$STAGE" = "all" ] || [ "$STAGE" = "probe" ]; then
    echo "=== actor-like Rockstar gate ($N_CHECKS checks per arm)"
    # The frozen baseline rows are shared by every arm's field gate; the eval
    # job skips (box, seed) pairs already measured.
    SUB_OVERRIDES=("BOXES=$VERIFY_BOXES")
    JEF=$(sub "eval frozen fields on verify boxes (GPU)" \
        $(dep_of "$AFTER") \
        scripts/slurm/evaluate_sr2_direct_gpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()
    for ARM in "${_ARMS[@]}"; do
        # Prefer this arm's own fit jid; fall back to the full fit set / AFTER
        # when probe is run as a separate stage after fits already finished.
        ARM_FIT="${FIT_BY_ARM[$ARM]:-${JID_FIT:-$AFTER}}"
        SUB_OVERRIDES=("ARM=$ARM" "PROBE=1")
        JP=$(sub "probe $ARM: capped probe_only fine-tune (GPU)" \
            $(dep_of "$ARM_FIT") \
            scripts/slurm/train_sr2_direct_gpu.sbatch); die_if_aborted
        SUB_OVERRIDES=("CHECKPOINT=$ROOT/runs/$RUN_NAME/probe_$ARM/ema_generator.pt"
                       "TAG=probe_$ARM" "BOXES=$VERIFY_BOXES")
        JEP=$(sub "probe $ARM: field metrics on verify boxes (GPU)" \
            --dependency=afterok:"$JP":"$JEF" \
            scripts/slurm/evaluate_sr2_direct_gpu.sbatch); die_if_aborted
        SUB_OVERRIDES=("STAGE=gate" "TAG=probe_$ARM")
        JG=$(sub "probe $ARM: field gates (CPU)" \
            --dependency=afterok:"$JEP" \
            scripts/slurm/score_sr2_direct_cpu.sbatch); die_if_aborted
        SUB_OVERRIDES=("ARM=$ARM")
        JS=$(sub "actor $ARM: generate verify boxes, choose the plan (GPU)" \
            --dependency=afterok:"$JG" \
            scripts/slurm/actor_select_gpu.sbatch); die_if_aborted
        # One array, not N jobs: the array index IS the plan index, and a task
        # beyond the plan prints so and exits 0 rather than failing.
        ACTOR_JIDS+=("$(sub "actor $ARM: $N_CHECKS real Rockstar runs (CPU array)" \
            --array=0-$((N_CHECKS - 1)) --dependency=afterok:"$JS" \
            scripts/slurm/actor_verify_cpu.sbatch)"); die_if_aborted
        SUB_OVERRIDES=()
    done
    echo "  probe   -> $ROOT/runs/$RUN_NAME/probe_<arm>/ema_generator.pt"
    echo "  gates   -> $ROOT/runs/$RUN_NAME/gate_probe_<arm>.json"
    echo "  plan    -> $ROOT/runs/$RUN_NAME/actor_plan_<arm>.json"
    echo "  results -> $ROOT/runs/$RUN_NAME/actor_verification_<arm>.jsonl"
fi

# --- decide: re-gate with the actor checks in place, write the benchmark ---
if [ "$STAGE" = "all" ] || [ "$STAGE" = "decide" ]; then
    DEP=""
    if [ "${#ACTOR_JIDS[@]}" -gt 0 ]; then
        # afterany: a check that failed is missing evidence, and the gate
        # reports missing evidence as an unmet criterion. afterok would strand
        # the benchmark instead and say nothing.
        DEP="--dependency=afterany:$(IFS=:; echo "${ACTOR_JIDS[*]}")"
    elif [ -n "${JID_FIT:-$AFTER}" ]; then
        DEP="--dependency=afterany:${JID_FIT:-$AFTER}"
    fi
    echo "=== decision"
    SUB_OVERRIDES=("ARMS=${ARMS// /,}")
    sub "benchmark: re-gate all arms, write proxy_benchmark.json (CPU)" \
        $DEP scripts/slurm/proxy_benchmark_cpu.sbatch >/dev/null; die_if_aborted
    SUB_OVERRIDES=()
    echo "  record -> $ROOT/runs/$RUN_NAME/proxy_benchmark.json"
fi

echo "=== submitted. watch with: squeue -u \$USER -o '%.10i %.20j %.10T %R'"
echo "=== This milestone ENDS at proxy_benchmark.json. Read its decision before"
echo "=== any SR2 parameter is allowed to move."
