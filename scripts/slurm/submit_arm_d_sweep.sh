#!/bin/bash
# Arm-D diagnostic sweep: loss type FIRST, then hyperparameters, both under
# leave-one-box-out validation inside set0-7. This submitter ONLY calls sbatch.
#
#   DRY=1 bash scripts/slurm/submit_arm_d_sweep.sh all
#   bash scripts/slurm/submit_arm_d_sweep.sh loss     # 7 loss variants x 8 folds
#   bash scripts/slurm/submit_arm_d_sweep.sh agg      # aggregate a finished stage
#   bash scripts/slurm/submit_arm_d_sweep.sh hparam   # sweep the winning loss
#   bash scripts/slurm/submit_arm_d_sweep.sh loss EPOCHS=500 VAL_EVERY=100
#
# WHY TWO STAGES AND NOT ONE GRID
# -------------------------------
# `all` chains them (loss -> aggregate -> hparam -> aggregate) because the
# ordering is mechanical, but the hparam job READS arm_d_loss_decision.json and
# exits 0 with an explanation if it is absent. So a loss stage that has not
# finished cannot silently seed the hyperparameter stage with a stale winner --
# and the chain reports that rather than stranding on Dependency.
#
# WHAT THIS IS NOT
# ----------------
# Not part of the pre-registered A-F benchmark. The production trainer still
# fits every arm once with its predeclared configuration; nothing here writes
# into a run directory or into proxy_benchmark.json. set8-11 are never read.
#
# Configuration goes into ONE timestamped env file passed as a POSITIONAL
# argument -- never `sbatch --export`, which on this cluster makes sbatch set
# SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
ROOT="${DMSR_SR2_DIRECT_ROOT:-$REWARD_ROOT/sr2_direct}"
STAGE="${1:-all}"; shift || true
DRY="${DRY:-0}"

ARM="${ARM:-d}"
DIRECT_CFG="${DIRECT_CFG:-configs/reward/sr2_direct_finetune.yaml}"
EPOCHS="${EPOCHS:-500}"
VAL_EVERY="${VAL_EVERY:-100}"
LRS="${LRS:-1e-3}"
FOLDS="${FOLDS:-}"
AFTER="${AFTER:-}"

# Plan sizes. Loss stage: one task per variant in LOSS_VARIANTS. Hparam stage:
# lr x weight_decay x dropout = 2 x 2 x 2. A task past the end exits 0, so an
# over-sized array is harmless; an under-sized one silently drops work, which is
# why these are stated here and asserted by the worker's banner.
N_LOSS="${N_LOSS:-7}"
N_HPARAM="${N_HPARAM:-8}"

for kv in "$@"; do
    case "$kv" in *=*) export "$kv" ;; esac
done

MARK="$ROOT/proxy_data/labels_complete.json"
if [ ! -r "$MARK" ] && [ "$DRY" != "1" ]; then
    echo "!!! $MARK does not exist, so labelling is not complete and any table" >&2
    echo "!!! on disk is partial. Nothing has been submitted." >&2
    exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/env" "$ROOT/sweeps"
ENVFILE="$ROOT/env/sweepd_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_arm_d_sweep.sh at $(date '+%F %T'); sourced by
# the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
DMSR_SR2_DIRECT_ROOT=$ROOT
DIRECT_CFG=$DIRECT_CFG
REWARD_CFG=configs/reward/reward.yaml
RUN_NAME=arm_${ARM}_sweep
SWEEP_ARM=$ARM
SWEEP_EPOCHS=$EPOCHS
SWEEP_VAL_EVERY=$VAL_EVERY
SWEEP_LRS=$LRS
SWEEP_FOLDS=$FOLDS
EOF
echo "env file: $ENVFILE"

for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

SUB_OVERRIDES=()
ABORT_FLAG="$ROOT/.submit_arm_d_sweep_aborted.$$"
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

JID_LOSS=""; JID_LOSS_AGG=""; JID_HP=""

# --- stage 1: loss type -----------------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "loss" ]; then
    echo "=== loss sweep: $N_LOSS variants x LOBO folds, $EPOCHS epochs, val every $VAL_EVERY"
    SUB_OVERRIDES=("SWEEP_STAGE=loss")
    JID_LOSS=$(sub "arm $ARM: loss sweep (GPU array 0-$((N_LOSS - 1)))" \
        --array=0-$((N_LOSS - 1)) $(dep_of "$AFTER") \
        scripts/slurm/sweep_arm_d_gpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()
    echo "  rows -> $ROOT/sweeps/arm_${ARM}_loss.jsonl"
fi

# --- aggregate the loss stage ----------------------------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "loss" ] || [ "$STAGE" = "agg" ]; then
    DEP=""
    # afterany, not afterok: a variant that died still leaves usable rows for the
    # others, and the aggregator refuses to pick a winner from incomplete folds.
    [ -n "${JID_LOSS:-$AFTER}" ] && DEP="--dependency=afterany:${JID_LOSS:-$AFTER}"
    SUB_OVERRIDES=("SWEEP_STAGE=${AGG_STAGE:-loss}")
    JID_LOSS_AGG=$(sub "arm $ARM: aggregate ${AGG_STAGE:-loss} stage (CPU)" \
        $DEP scripts/slurm/aggregate_arm_d_sweep_cpu.sbatch); die_if_aborted
    SUB_OVERRIDES=()
    echo "  decision -> $ROOT/sweeps/arm_${ARM}_${AGG_STAGE:-loss}_decision.json"
    echo "  curves   -> $ROOT/sweeps/arm_${ARM}_${AGG_STAGE:-loss}_curves.png"
fi

# --- stage 2: hyperparameters on the winning loss ---------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "hparam" ]; then
    echo "=== hyperparameter sweep on the winning loss ($N_HPARAM configurations)"
    SUB_OVERRIDES=("SWEEP_STAGE=hparam")
    JID_HP=$(sub "arm $ARM: hparam sweep (GPU array 0-$((N_HPARAM - 1)))" \
        --array=0-$((N_HPARAM - 1)) $(dep_of "${JID_LOSS_AGG:-$AFTER}") \
        scripts/slurm/sweep_arm_d_gpu.sbatch); die_if_aborted
    SUB_OVERRIDES=("SWEEP_STAGE=hparam")
    sub "arm $ARM: aggregate hparam stage (CPU)" \
        --dependency=afterany:"$JID_HP" \
        scripts/slurm/aggregate_arm_d_sweep_cpu.sbatch >/dev/null; die_if_aborted
    SUB_OVERRIDES=()
    echo "  decision -> $ROOT/sweeps/arm_${ARM}_hparam_decision.json"
fi

echo "=== submitted. watch with: squeue -u \$USER -o '%.10i %.20j %.10T %R'"
echo "=== DIAGNOSTIC ONLY: set8-11 are never read, and a winner here is not an"
echo "=== arm-comparison result. To make one, re-fit ALL arms via"
echo "=== scripts/slurm/submit_proxy_benchmark.sh fit."
