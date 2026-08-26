#!/bin/bash
# Fine-tune SR2 against true HR subhalos with the auxiliary gather loss:
#
#   shakeout (GPU, targets only) --afterok--> train (GPU) --afterok--> redraw (CPU)
#
# The shakeout builds the targets, prints how many true HR subhalos the loss will
# actually supervise in the cluster's tiles, reports the frozen generator's
# compact-mass ratio at them, and stops. It is ~5 minutes and it is what stands
# between a misconfigured selection and eight GPU hours: a run with three live
# targets is not worth training.
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster sets SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
#
#   bash scripts/slurm/submit_host_gather.sh
#   HG_RUNG=middle_fine HG_STEPS=6000 bash scripts/slurm/submit_host_gather.sh
#   SHAKEOUT_ONLY=1 bash scripts/slurm/submit_host_gather.sh   # just check targets
#   SKIP_SHAKEOUT=1 bash scripts/slurm/submit_host_gather.sh   # straight to training
#   RENDER_ONLY=1 HG_RUN_DIR=<dir> bash scripts/slurm/submit_host_gather.sh
#   DRY=1 bash scripts/slurm/submit_host_gather.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
SHAKEOUT_ONLY="${SHAKEOUT_ONLY:-0}"
SKIP_SHAKEOUT="${SKIP_SHAKEOUT:-0}"
RENDER_ONLY="${RENDER_ONLY:-0}"

# --- the experiment ----------------------------------------------------------
HG_BOX="${HG_BOX:-set8}"
HG_HOST_ID="${HG_HOST_ID:--1}"        # -1 = the most massive host, as step 4 used
HG_RUNG="${HG_RUNG:-fine}"
HG_STEPS="${HG_STEPS:-3000}"
HG_BATCH="${HG_BATCH:-2}"
HG_N_TILES="${HG_N_TILES:-4}"
HG_LR_SCALE="${HG_LR_SCALE:-10.0}"
HG_EVAL_EVERY="${HG_EVAL_EVERY:-100}"
HG_PNG_EVERY="${HG_PNG_EVERY:-500}"
HG_SLAB="${HG_SLAB:-4}"
HG_SEED="${HG_SEED:-0}"
HG_LABEL="${HG_LABEL:-}"
# --- the objective -----------------------------------------------------------
HG_W_GATHER="${HG_W_GATHER:-1.0}"
HG_W_CONTRAST="${HG_W_CONTRAST:-1.0}"
HG_W_VDISP="${HG_W_VDISP:-1.0}"      # velocity dispersion match
HG_W_VBULK="${HG_W_VBULK:-1.0}"      # bulk velocity match
HG_W_PRESERVE="${HG_W_PRESERVE:-1.0}"  # defend structure outside the windows
HG_W_LOW="${HG_W_LOW:-1.0}"
HG_W_ANCHOR="${HG_W_ANCHOR:-0.1}"
HG_W_MSE="${HG_W_MSE:-0.0}"
HG_SIGMA_FLOOR="${HG_SIGMA_FLOOR:-1.0}"
HG_MIN_NUM_P="${HG_MIN_NUM_P:-50}"
HG_MIN_PURITY="${HG_MIN_PURITY:-0.5}"
HG_MIN_HR_COMPACT="${HG_MIN_HR_COMPACT:-5.0}"
HG_K_SPLIT="${HG_K_SPLIT:-4.0}"
HG_RUN_DIR="${HG_RUN_DIR:-}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/host_gather_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_host_gather.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
HG_BOX=$HG_BOX
HG_HOST_ID=$HG_HOST_ID
HG_RUNG=$HG_RUNG
HG_STEPS=$HG_STEPS
HG_BATCH=$HG_BATCH
HG_N_TILES=$HG_N_TILES
HG_LR_SCALE=$HG_LR_SCALE
HG_EVAL_EVERY=$HG_EVAL_EVERY
HG_PNG_EVERY=$HG_PNG_EVERY
HG_SLAB=$HG_SLAB
HG_SEED=$HG_SEED
HG_LABEL=$HG_LABEL
HG_W_GATHER=$HG_W_GATHER
HG_W_CONTRAST=$HG_W_CONTRAST
HG_W_VDISP=$HG_W_VDISP
HG_W_VBULK=$HG_W_VBULK
HG_W_PRESERVE=$HG_W_PRESERVE
HG_W_LOW=$HG_W_LOW
HG_W_ANCHOR=$HG_W_ANCHOR
HG_W_MSE=$HG_W_MSE
HG_SIGMA_FLOOR=$HG_SIGMA_FLOOR
HG_MIN_NUM_P=$HG_MIN_NUM_P
HG_MIN_PURITY=$HG_MIN_PURITY
HG_MIN_HR_COMPACT=$HG_MIN_HR_COMPACT
HG_K_SPLIT=$HG_K_SPLIT
HG_RUN_DIR=$HG_RUN_DIR
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch jobs.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

sub() {
    if [ "$DRY" = "1" ]; then
        echo "DRY: sbatch $*" >&2
        echo "DRYID"
    else
        sbatch --parsable "$@"
    fi
}

SID=""; TID=""; RID=""

if [ "$RENDER_ONLY" != "1" ]; then
    if [ "$SKIP_SHAKEOUT" != "1" ]; then
        SID=$(sub scripts/slurm/finetune_host_gather_gpu.sbatch "$ENVFILE" \
              HG_TARGETS_ONLY=1 HG_STEPS=0)
        echo "submitted shakeout (targets only) -> job $SID"
    fi
    if [ "$SHAKEOUT_ONLY" != "1" ]; then
        DEP=(); [ -n "$SID" ] && DEP=(--dependency=afterok:"$SID")
        TID=$(sub "${DEP[@]}" scripts/slurm/finetune_host_gather_gpu.sbatch "$ENVFILE")
        echo "submitted train                   -> job $TID${SID:+ (afterok:$SID)}"
    fi
fi

if [ "$SHAKEOUT_ONLY" != "1" ]; then
    DEP=(); [ -n "$TID" ] && DEP=(--dependency=afterok:"$TID")
    RID=$(sub "${DEP[@]}" scripts/slurm/render_gather_slices_cpu.sbatch "$ENVFILE")
    echo "submitted redraw (CPU)            -> job $RID${TID:+ (afterok:$TID)}"
fi

echo
[ -n "$SID" ] && echo "watch shakeout: tail -F $REWARD_ROOT/logs/slurm-hgather-$SID.out"
[ -n "$TID" ] && echo "watch train:    tail -F $REWARD_ROOT/logs/slurm-hgather-$TID.out"
[ -n "$RID" ] && echo "watch redraw:   tail -F $REWARD_ROOT/logs/slurm-hgrender-$RID.out"
echo
echo "results in: $REWARD_ROOT/host_gather/${HG_BOX}_h<host>_${HG_RUNG}${HG_LABEL}/"
echo "  targets.json    the shakeout: how many true HR subhalos are supervised"
echo "  summary.json    weights, target report, full history, VERDICT"
echo "  metrics.jsonl   one row per eval step (compact_ratio, low_k_change, ...)"
echo "  subhalos.json   per-subhalo: HR asked for X, frozen had Y, tuned has Z"
echo "  slices/*.png    frozen SR2 | fine-tuned | HR, true subhalos ringed"
echo "  eval/step*.npz  the redraw source -- figures never need the GPU again"
