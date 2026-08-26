#!/bin/bash
# Does the id-gathered member-set objective admit bound halos? One chain:
#
#   shakeout (GPU, selection only) --afterok--> free-field optimise (GPU)
#     --afterok--> splice (CPU) --afterok--> Rockstar (CPU) --afterok--> compare (CPU)
#
# The shakeout builds the member sets, prints how many true HR subhalos this
# tiling actually supervises and what reference each one can reach, and stops.
# It is minutes and it stands between a misconfigured selection and the rest of
# the chain -- exactly as submit_host_gather.sh's does, and for the same reason.
#
# Read the two questions in order:
#   1. does the FREE FIELD reach HR-like bound_frac?   -> summary.json VERDICT
#   2. does Rockstar agree?                            -> the compare json
# Only (2) settles anything. docs/sr2_gather_finetune.md section 6 is the
# standing reason not to believe a differentiable surrogate about its own
# success, and section 8.1 supplies the calibration that makes (2) readable:
# ceiling 227 subhalos in R_vir, noise +-9, per-target 42/43 against 0/43.
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster sets SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
#
#   bash scripts/slurm/submit_free_field_gather.sh
#   SHAKEOUT_ONLY=1 bash scripts/slurm/submit_free_field_gather.sh
#   FF_STEPS=6000 FF_LABEL=_long bash scripts/slurm/submit_free_field_gather.sh
#   FF_GATE=0 bash scripts/slurm/submit_free_field_gather.sh    # skip Rockstar
#   # the CEILING at a wider tiling: no optimisation, splice the true HR tiles
#   FF_WHICH=hr FF_STEPS=0 FF_N_TILES=16 FF_LABEL=_ceil16 \
#     bash scripts/slurm/submit_free_field_gather.sh
#   DRY=1 bash scripts/slurm/submit_free_field_gather.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
SHAKEOUT_ONLY="${SHAKEOUT_ONLY:-0}"
SKIP_SHAKEOUT="${SKIP_SHAKEOUT:-0}"
FF_GATE="${FF_GATE:-1}"

# --- the experiment ----------------------------------------------------------
FF_BOX="${FF_BOX:-set8}"
# Explicit rather than -1 on purpose: the run directory carries the host id, and
# the splice/compare jobs are chained BEFORE the optimiser has resolved it. A -1
# here would leave the gate pointing at a directory nobody can name yet, so the
# chain would have to be split. 271800 is set8's most massive host, the one every
# result in docs/sr2_gather_finetune.md is measured on.
FF_HOST_ID="${FF_HOST_ID:-271800}"
# The R_vir ceiling knob. 4 tiles hold 42.4% of the host's Lagrangian sites and
# the measured ceiling is 227 of HR's 506; raising this is the only thing that
# raises that bound. scripts/slurm/submit_gather_coverage.sh measures the
# function without a halo finder, so a rung can be chosen before it is spent on.
FF_N_TILES="${FF_N_TILES:-4}"
FF_FORWARD_CHUNK="${FF_FORWARD_CHUNK:-4}"
FF_TILES="${FF_TILES:-}"          # explicit tile ids; empty = the host ranking
FF_STEPS="${FF_STEPS:-2000}"
FF_LR="${FF_LR:-3e-3}"
FF_EVAL_EVERY="${FF_EVAL_EVERY:-100}"
FF_SEED="${FF_SEED:-0}"
FF_LABEL="${FF_LABEL:-}"
# --- the objective -----------------------------------------------------------
FF_W_VIRIAL="${FF_W_VIRIAL:-1.0}"    # the driver: a pair sum, 64x from HR
FF_W_BOUND="${FF_W_BOUND:-1.0}"      # the target: Rockstar's own decision rule
FF_W_D6="${FF_W_D6:-1.0}"            # 6-D contrast against real local non-members
FF_W_RRMS="${FF_W_RRMS:-0.3}"
FF_W_SIGMAV="${FF_W_SIGMAV:-0.3}"
# Added after the 2026-08-21 run: without a centroid term the loss constrains
# only internal moments, and the objects it built landed a median 0.414 Mpc/h
# from their targets (8/154 recovered against a 0.150 Mpc/h search radius).
FF_W_CENTRE="${FF_W_CENTRE:-1.0}"
# The three arms of docs/sr2_member_gather_training.md section 9.4, all off by
# default. Section 10.5 has their first generator numbers.
# `full` + dead 0 + huber 0 IS the 72/154 objective.
FF_BOUND_PENALTY="${FF_BOUND_PENALTY:-hinge}"
FF_CENTRE_MODE="${FF_CENTRE_MODE:-full}"
FF_CENTRE_DEAD_ZONE="${FF_CENTRE_DEAD_ZONE:-0.0}"
FF_CENTRE_HUBER="${FF_CENTRE_HUBER:-0.0}"
# Also added: the LR guard's block-averaged gradient plus Adam rewrote 99.56% of
# the tile, so nothing in that run was attributable to the objective.
FF_MASK_GRAD="${FF_MASK_GRAD:-1}"
FF_W_LOW="${FF_W_LOW:-100.0}"
FF_MIN_NUM_P="${FF_MIN_NUM_P:-200}"
# Binds from ~8 tiles up: at 16 the selection is 625 sets and this keeps 256.
FF_MAX_SETS="${FF_MAX_SETS:-256}"
FF_MIN_PURITY="${FF_MIN_PURITY:-0.5}"
FF_MIN_LIVE_FRAC="${FF_MIN_LIVE_FRAC:-0.5}"
FF_SOFTENING="${FF_SOFTENING:-0.01}"
FF_BOUND_TAU="${FF_BOUND_TAU:-0.5}"
FF_BG_K="${FF_BG_K:-4096}"
FF_BG_RADIUS="${FF_BG_RADIUS:-4.0}"
FF_POT_CHUNK="${FF_POT_CHUNK:-2048}"
FF_GAIN_FRAC="${FF_GAIN_FRAC:-0.5}"
FF_LOW_K_MAX="${FF_LOW_K_MAX:-0.02}"
FF_VEL_RMS_TOL="${FF_VEL_RMS_TOL:-0.10}"
# --- the gate ----------------------------------------------------------------
FF_MIN_P="${FF_MIN_P:-50}"
FF_WHICH="${FF_WHICH:-out}"

RUN_NAME="${FF_BOX}_h${FF_HOST_ID}${FF_LABEL}"
RUN_DIR="$REWARD_ROOT/free_field_gather/$RUN_NAME"
_SUFFIX=""; [ "$FF_WHICH" != "out" ] && _SUFFIX="_$FF_WHICH"
RS_TAG="freefield_${RUN_NAME}${_SUFFIX}"
FIELD="$REWARD_ROOT/flow_rockstar/fields/${FF_BOX}__${RS_TAG}__seed${FF_SEED}.npy"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" \
         "$REWARD_ROOT/flow_rockstar/fields" "$REWARD_ROOT/flow_rockstar/logs"
ENVFILE="$REWARD_ROOT/env/free_field_gather_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_free_field_gather.sh at $(date '+%F %T');
# sourced by the job preambles as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
FF_BOX=$FF_BOX
FF_HOST_ID=$FF_HOST_ID
FF_N_TILES=$FF_N_TILES
FF_FORWARD_CHUNK=$FF_FORWARD_CHUNK
FF_TILES=$FF_TILES
FF_STEPS=$FF_STEPS
FF_LR=$FF_LR
FF_EVAL_EVERY=$FF_EVAL_EVERY
FF_SEED=$FF_SEED
FF_LABEL=$FF_LABEL
FF_W_VIRIAL=$FF_W_VIRIAL
FF_W_BOUND=$FF_W_BOUND
FF_W_D6=$FF_W_D6
FF_W_RRMS=$FF_W_RRMS
FF_W_SIGMAV=$FF_W_SIGMAV
FF_W_CENTRE=$FF_W_CENTRE
FF_BOUND_PENALTY=$FF_BOUND_PENALTY
FF_CENTRE_MODE=$FF_CENTRE_MODE
FF_CENTRE_DEAD_ZONE=$FF_CENTRE_DEAD_ZONE
FF_CENTRE_HUBER=$FF_CENTRE_HUBER
FF_MASK_GRAD=$FF_MASK_GRAD
FF_W_LOW=$FF_W_LOW
FF_MIN_NUM_P=$FF_MIN_NUM_P
FF_MAX_SETS=$FF_MAX_SETS
FF_MIN_PURITY=$FF_MIN_PURITY
FF_MIN_LIVE_FRAC=$FF_MIN_LIVE_FRAC
FF_SOFTENING=$FF_SOFTENING
FF_BOUND_TAU=$FF_BOUND_TAU
FF_BG_K=$FF_BG_K
FF_BG_RADIUS=$FF_BG_RADIUS
FF_POT_CHUNK=$FF_POT_CHUNK
FF_GAIN_FRAC=$FF_GAIN_FRAC
FF_LOW_K_MAX=$FF_LOW_K_MAX
FF_VEL_RMS_TOL=$FF_VEL_RMS_TOL
# --- read by the shared gather gate jobs, which use the HG_* names -----------
HG_BOX=$FF_BOX
HG_SEED=$FF_SEED
HG_RUN_DIR=$RUN_DIR
HG_WHICH=$FF_WHICH
HG_RS_TAG=$RS_TAG
HG_HOST_ID=$FF_HOST_ID
HG_MIN_P=$FF_MIN_P
HG_SWEEP=1
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo
echo "run dir: $RUN_DIR"
echo "field:   $FIELD"
echo

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch jobs.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

sub() {
    if [ "$DRY" = "1" ]; then echo "DRY: sbatch $*" >&2; echo "DRYID"
    else sbatch --parsable "$@"; fi
}

SID=""; OID=""; SPLICE=""; RS=""; CMP=""

if [ "$SKIP_SHAKEOUT" != "1" ]; then
    SID=$(sub scripts/slurm/free_field_gather_gpu.sbatch "$ENVFILE" \
          FF_TARGETS_ONLY=1 FF_STEPS=0)
    echo "submitted shakeout (sets only) -> job $SID (GPU)"
fi

if [ "$SHAKEOUT_ONLY" = "1" ]; then
    echo
    [ -n "$SID" ] && echo "watch: tail -F $REWARD_ROOT/logs/slurm-ffgather-$SID.out"
    echo "then: $RUN_DIR/targets.json"
    exit 0
fi

DEP=(); [ -n "$SID" ] && DEP=(--dependency=afterok:"$SID")
OID=$(sub "${DEP[@]}" scripts/slurm/free_field_gather_gpu.sbatch "$ENVFILE")
echo "submitted free-field optimise  -> job $OID (GPU${SID:+, afterok:$SID})"

if [ "$FF_GATE" = "1" ]; then
    SPLICE=$(sub --dependency=afterok:"$OID" \
             scripts/slurm/gather_splice_cpu.sbatch "$ENVFILE")
    echo "submitted splice               -> job $SPLICE (CPU, afterok:$OID)"

    # The generic Rockstar job takes its own variable names, so they go as
    # positional VAR=value arguments rather than through the env file.
    RS=$(sub --dependency=afterok:"$SPLICE" \
         scripts/slurm/flow_rockstar_catalog_cpu.sbatch \
         "BOX=$FF_BOX" "TAG=$RS_TAG" "SEED=$FF_SEED" "FIELD_OUT=$FIELD")
    echo "submitted rockstar             -> job $RS (CPU, afterok:$SPLICE)"

    CMP=$(sub --dependency=afterok:"$RS" \
          scripts/slurm/gather_compare_cpu.sbatch "$ENVFILE")
    echo "submitted compare              -> job $CMP (CPU, afterok:$RS)"
fi

echo
[ -n "$SID" ]    && echo "watch shakeout: tail -F $REWARD_ROOT/logs/slurm-ffgather-$SID.out"
[ -n "$OID" ]    && echo "watch optimise: tail -F $REWARD_ROOT/logs/slurm-ffgather-$OID.out"
[ -n "$SPLICE" ] && echo "watch splice:   tail -F $REWARD_ROOT/logs/slurm-hg_splice-$SPLICE.out"
[ -n "$RS" ]     && echo "watch rockstar: tail -F $REWARD_ROOT/flow_rockstar/logs/slurm-flow_rs_cat-$RS.out"
[ -n "$CMP" ]    && echo "watch compare:  tail -F $REWARD_ROOT/logs/slurm-hg_compare-$CMP.out"
echo
echo "results:"
echo "  $RUN_DIR/targets.json   the shakeout: how many sets, and their reference"
echo "  $RUN_DIR/summary.json   config, full history, VERDICT (feasibility only)"
echo "  $RUN_DIR/metrics.jsonl  one row per eval step (bound_frac, 2T/|W|, low_k)"
echo "  $RUN_DIR/subhalos.json  per set: what HR reaches, what the free field did"
echo "  $RUN_DIR/tiles.npz      the splice source"
echo "  $REWARD_ROOT/flow_rockstar/compare/${FF_BOX}__${RS_TAG}.json"
echo "      THE ANSWER. Read subhalos-in-R_vir against the section 8.1 numbers:"
echo "      HR 506, ceiling 227, null 20 (+-9 noise), base 11, window run 5."
echo "      And the per-target row: the ceiling scores 42/43, the window run 0/43."
