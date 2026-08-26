#!/bin/bash
# Submitter for the reward-only residual line: Phase 1 (calibration + projection
# oracle) and Phase 2 (Gaussian policy, support gate, reward-only training).
# It ONLY calls sbatch -- no work of its own.
#
#   bash scripts/slurm/submit_reward_only.sh              # phase1 then phase2
#   bash scripts/slurm/submit_reward_only.sh phase1       # calibration + oracle
#   bash scripts/slurm/submit_reward_only.sh phase2       # gaussian support arm
#   bash scripts/slurm/submit_reward_only.sh train        # replay + training
#   DRY=1 bash scripts/slurm/submit_reward_only.sh        # print, submit nothing
#
# Phase 1 and Phase 2's sampling stage are SIBLINGS, not a chain: the projection
# oracle chooses alpha, and the support gate asks whether the reward has any
# support at all. Neither answer depends on the other, so queueing them
# sequentially would only add latency. What DOES depend on the oracle is the
# alpha you eventually put in configs/reward/gaussian_policy.yaml, and that is a
# deliberate human step -- the submitter never pastes a measured constraint into
# a config on your behalf.
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

PROJ_RUN="${PROJ_RUN:-proj0}"
PROJ_CFG="${PROJ_CFG:-configs/reward/projection_oracle.yaml}"
PROJ_SEEDS="${PROJ_SEEDS:-0,1,2}"
PROJ_REF_SHARDS="${PROJ_REF_SHARDS:-3}"     # array is 0-N
PROJ_SHARDS="${PROJ_SHARDS:-15}"

GAUSS_CFG="${GAUSS_CFG:-configs/reward/gaussian_policy.yaml}"
GAUSS_RUN="${GAUSS_RUN:-gauss_ref_k16}"
GAUSS_CKPT="${GAUSS_CKPT:-}"                # empty = untrained reference policy
SCORE_BASE_SHARDS="${SCORE_BASE_SHARDS:-3}"
SCORE_SHARDS="${SCORE_SHARDS:-15}"
ROUND="${ROUND:-round_000}"
TRAIN_RUN="${TRAIN_RUN:-gaussian_r0}"
SCALES="${SCALES:-$ROOT/audits/correction_scales/correction_scales.json}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/reward_only_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_reward_only.sh at $(date '+%F %T'); sourced by
# the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$ROOT
REWARD_CFG=configs/reward/reward.yaml
PROJ_CFG=$PROJ_CFG
GAUSS_CFG=$GAUSS_CFG
SCALES=$SCALES
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
ABORT_FLAG="$ROOT/.submit_reward_only_aborted.$$"
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
        echo "!!! nothing further has been submitted." >&2
        echo "!!! Every stage here is resumable (rows and fields are skipped when" >&2
        echo "!!! they already exist), so re-running this submitter resumes." >&2
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

JID_CALIB=""

# --- Phase 1: amplitude calibration, then the projection oracle -------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "phase1" ] || [ "$STAGE" = "calib" ]; then
    echo "=== phase 1a: correction-scale calibration (CPU)"
    SUB_OVERRIDES=("SPLIT=$CALIB_SPLIT" "OUT=$SCALES")
    JID_CALIB=$(sub "calibrate correction scales" \
        scripts/slurm/calibrate_correction_scales_cpu.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "phase1" ]; then
    echo "=== phase 1b: projection oracle ($PROJ_RUN)"
    # The reference arms (`hr`, `sr2`) run first and alone: every sweep row
    # matches against the HR catalog, and two shards running Rockstar in one work
    # directory corrupt each other's catalog.
    SUB_OVERRIDES=("RUN_NAME=$PROJ_RUN" "BASE_SEEDS=$PROJ_SEEDS" "REFERENCES=only")
    JID_REF=$(sub "projection oracle: reference arms (CPU array)" \
        --array=0-"$PROJ_REF_SHARDS" scripts/slurm/audit_projection_oracle.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted

    SUB_OVERRIDES=("RUN_NAME=$PROJ_RUN" "BASE_SEEDS=$PROJ_SEEDS" "REFERENCES=skip")
    JID_SWEEP=$(sub "projection oracle: alpha sweep (CPU array)" \
        --dependency=afterok:"$JID_REF" --array=0-"$PROJ_SHARDS" \
        scripts/slurm/audit_projection_oracle.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted

    SUB_OVERRIDES=("RUN_NAME=$PROJ_RUN")
    sub "projection oracle: report + recommendation (CPU)" \
        --dependency=afterok:"$JID_SWEEP" \
        scripts/slurm/projection_oracle_report_cpu.sbatch >/dev/null
    SUB_OVERRIDES=()
    die_if_aborted
fi

# --- Phase 2: the Gaussian policy's support arm -----------------------------
if [ "$STAGE" = "all" ] || [ "$STAGE" = "phase2" ]; then
    echo "=== phase 2: Gaussian policy support arm ($GAUSS_RUN)"
    if [ -n "$GAUSS_CKPT" ]; then
        echo "    sampling FROM A CHECKPOINT: $GAUSS_CKPT"
    else
        echo "    sampling the UNTRAINED REFERENCE policy (the support-gate arm)"
    fi

    # Measure the receptive field on CPU first. The sampler re-measures and
    # hard-exits on too small a margin, but discovering that costs a GPU
    # allocation; this costs a few CPU minutes. NOT a dependency of the sampler:
    # the margin is a config value a person has to set from the measurement, so
    # chaining them would only hide a stale margin behind a green job.
    if [ "${SKIP_RF:-0}" != "1" ]; then
        sub "gaussian: measure policy receptive field (CPU)" \
            scripts/slurm/measure_policy_rf_cpu.sbatch >/dev/null
        die_if_aborted
    fi

    SUB_OVERRIDES=("RUN_NAME=$GAUSS_RUN" ${GAUSS_CKPT:+"CKPT=$GAUSS_CKPT"})
    if [ -n "$JID_CALIB" ]; then
        JID_GEN=$(sub "gaussian: sample candidates (GPU)" \
            --dependency=afterok:"$JID_CALIB" scripts/slurm/gaussian_sample_gpu.sbatch)
    else
        JID_GEN=$(sub "gaussian: sample candidates (GPU)" \
            scripts/slurm/gaussian_sample_gpu.sbatch)
    fi
    SUB_OVERRIDES=()
    die_if_aborted

    SUB_OVERRIDES=("RUN_NAME=$GAUSS_RUN" "BASELINES=1")
    JID_BASE=$(sub "gaussian: score per-box a=0 baselines (CPU array)" \
        --dependency=afterok:"$JID_GEN" --array=0-"$SCORE_BASE_SHARDS" \
        scripts/slurm/gaussian_score_cpu.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted

    SUB_OVERRIDES=("RUN_NAME=$GAUSS_RUN")
    JID_SCORE=$(sub "gaussian: score candidates (CPU array)" \
        --dependency=afterok:"$JID_BASE" --array=0-"$SCORE_SHARDS" \
        scripts/slurm/gaussian_score_cpu.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted

    SUB_OVERRIDES=("RUN_NAME=$GAUSS_RUN" "AGGREGATE=1")
    JID_AGG=$(sub "gaussian: aggregate (CPU)" \
        --dependency=afterok:"$JID_SCORE" --array=0-0 \
        scripts/slurm/gaussian_score_cpu.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted

    SUB_OVERRIDES=("RUN_NAME=$GAUSS_RUN")
    sub "gaussian: SUPPORT GATE (CPU)" \
        --dependency=afterok:"$JID_AGG" \
        scripts/slurm/gaussian_support_gate_cpu.sbatch >/dev/null
    SUB_OVERRIDES=()
    die_if_aborted

    echo ""
    echo "    Training is NOT queued behind the gate on purpose. Read"
    echo "    $ROOT/oracle/$GAUSS_RUN/support_gate.md first, then:"
    echo "      bash scripts/slurm/submit_reward_only.sh train"
fi

# --- Replay + reward-only training -----------------------------------------
# Deliberately a separate invocation. The support gate is a decision point for a
# person, not a dependency edge: a chain that trains automatically after a failed
# gate produces a checkpoint fitted to noise, and the job that did so exits 0.
if [ "$STAGE" = "train" ]; then
    echo "=== replay round $ROUND from $GAUSS_RUN, then reward-only training"
    SUB_OVERRIDES=("RUN_NAME=$GAUSS_RUN" "ROUND=$ROUND")
    JID_REPLAY=$(sub "gaussian: build elite replay (CPU)" \
        scripts/slurm/gaussian_replay_cpu.sbatch)
    SUB_OVERRIDES=()
    die_if_aborted

    SUB_OVERRIDES=("ROUND=$ROUND" "ORACLE_RUN=$GAUSS_RUN" "RUN_NAME=$TRAIN_RUN"
                   ${GAUSS_CKPT:+"BEHAVIOR_CKPT=$GAUSS_CKPT"})
    sub "gaussian: reward-only training (GPU)" \
        --dependency=afterok:"$JID_REPLAY" \
        scripts/slurm/gaussian_train_gpu.sbatch >/dev/null
    SUB_OVERRIDES=()
    die_if_aborted
fi

echo ""
echo "submitted. logs: $ROOT/logs"
echo "reminder: the projection oracle RECOMMENDS alpha_disp/alpha_vel; paste them"
echo "into configs/reward/gaussian_policy.yaml yourself before the next round."
