#!/bin/bash
# Submitter for the proxy-gradient diagnostic and the (diagnostic-only) reward
# fine-tune. ONLY calls sbatch. Configuration goes into ONE timestamped env file
# passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster makes slurmd try to rebuild the login env and requeue-holds the job.
#
#   DRY=1 bash scripts/slurm/submit_proxy_gradient.sh grad ARM=c   # print only
#   bash scripts/slurm/submit_proxy_gradient.sh grad ARM=c         # gradient at frozen G_z0 -> plot
#   bash scripts/slurm/submit_proxy_gradient.sh finetune ARM=c     # ignore-gate overfit -> gradient@ckpt -> plot
#   bash scripts/slurm/submit_proxy_gradient.sh all ARM=c          # both chains
#
# WHY IGNORE_GATE: every arm fails the offline proxy gate (within-tile
# spearman ~0.26 < 0.5; candidate collapse). The production actor path therefore
# refuses to train. The fine-tune here runs with --ignore-gate purely to SEE
# what the reward signal does to the field; its checkpoint is stamped not-
# evidence and must never advance a rung. The gradient diagnostic is the point:
# it shows whether the proxy even gives the generator a usable gradient.
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
ROOT="${DMSR_SR2_DIRECT_ROOT:-$REWARD_ROOT/sr2_direct}"
STAGE="${1:-grad}"; shift || true
DRY="${DRY:-0}"

RUN_NAME="${RUN_NAME:-direct_a}"
ARM="${ARM:-c}"
DIRECT_CFG="${DIRECT_CFG:-configs/reward/sr2_direct_finetune.yaml}"

# Per-invocation overrides after the stage (ARM=c, N_SEEDS=4, CPU=1, ...).
for kv in "$@"; do case "$kv" in *=*) export "$kv" ;; esac; done
RUN_NAME="${RUN_NAME}"; ARM="${ARM}"

# CPU=1 runs the GRADIENT job on the cpu partition (no GPU queue wait; slower per
# tile). The fine-tune, when requested, still needs the GPU. Default: GPU.
CPU="${CPU:-0}"
if [ "$CPU" = "1" ]; then
    GRAD_SBATCH=scripts/slurm/proxy_gradient_cpu.sbatch
    GRAD_WHERE=CPU; GRAD_LOG=sd_grad_cpu
else
    GRAD_SBATCH=scripts/slurm/proxy_gradient_gpu.sbatch
    GRAD_WHERE=GPU; GRAD_LOG=sd_grad
fi
LOGS="$ROOT/logs"
RUN_DIR="$ROOT/runs/$RUN_NAME"

mkdir -p "$ROOT/logs" "$ROOT/env"
ENVFILE="$ROOT/env/grad_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_proxy_gradient.sh at $(date '+%F %T').
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
DMSR_SR2_DIRECT_ROOT=$ROOT
DIRECT_CFG=$DIRECT_CFG
REWARD_CFG=configs/reward/reward.yaml
RUN_NAME=$RUN_NAME
BASE_SEED=0
ARM=$ARM
EOF
echo "env file: $ENVFILE  (run=$RUN_NAME arm=$ARM stage=$STAGE)"

# Never leak SLURM_* into child jobs when submitting from inside an allocation.
for v in $(env | grep -o '^SLURM_[A-Z_]*' || true); do unset "$v"; done

# sub() runs in a $(...) subshell so it cannot append to a parent array, but it
# CAN append to a file -- that side effect survives. The end-of-run summary reads
# it back to print the tail commands. `logname` is the sbatch's #SBATCH job-name,
# so the log path slurm-<logname>-<jobid>.out is known without scontrol.
JOBS_FILE="$ENVFILE.jobs"; : > "$JOBS_FILE"
RESULTS=()      # parent-scope; result files this run will produce

SUB_OVERRIDES=()
sub() {   # sub <label> <logname> <sbatch args...>  -> prints job id
    local label="$1" logname="$2"; shift 2
    if [ "$DRY" = "1" ]; then
        echo "DRY  $label:" >&2
        echo "       sbatch $* $ENVFILE ${SUB_OVERRIDES[*]}" >&2
        echo "DRY000|$logname|$label" >> "$JOBS_FILE"
        echo "000000"; return 0
    fi
    local jid
    jid=$(sbatch --parsable "$@" "$ENVFILE" "${SUB_OVERRIDES[@]}") || {
        echo "!!! sbatch FAILED for: $label" >&2; exit 1; }
    echo "  $label -> job $jid" >&2
    echo "$jid|$logname|$label" >> "$JOBS_FILE"
    echo "$jid"
}

# --- gradient at the frozen generator, then plot ---------------------------
if [ "$STAGE" = "grad" ] || [ "$STAGE" = "all" ]; then
    JG=$(sub "gradient @ frozen G_z0 ($GRAD_WHERE)" "$GRAD_LOG" "$GRAD_SBATCH")
    SUB_OVERRIDES=("TAG=frozen")
    sub "plot gradient @ frozen (CPU)" sd_grad_plot --dependency=afterok:"$JG" \
        scripts/slurm/plot_proxy_gradient_cpu.sbatch >/dev/null
    SUB_OVERRIDES=()
    RESULTS+=("$RUN_DIR/gradient/proxy_grad_${ARM}_frozen_scatter.png"
              "$RUN_DIR/gradient/proxy_grad_${ARM}_frozen_convergence.png"
              "$RUN_DIR/gradient/proxy_grad_${ARM}_frozen.json")
fi

# --- ignore-gate overfit fine-tune -> gradient @ ckpt -> plot --------------
if [ "$STAGE" = "finetune" ] || [ "$STAGE" = "all" ]; then
    SUB_OVERRIDES=("OVERFIT=1" "IGNORE_GATE=1")
    JF=$(sub "overfit fine-tune (ignore-gate, GPU)" sd_actor \
        scripts/slurm/train_sr2_direct_gpu.sbatch)
    SUB_OVERRIDES=()

    CK="$ROOT/runs/$RUN_NAME/overfit/ema_generator.pt"
    SUB_OVERRIDES=("CHECKPOINT=$CK")
    JG=$(sub "gradient @ overfit ckpt ($GRAD_WHERE)" "$GRAD_LOG" \
        --dependency=afterok:"$JF" "$GRAD_SBATCH")
    SUB_OVERRIDES=()

    SUB_OVERRIDES=("TAG=overfit")
    sub "plot gradient @ overfit (CPU)" sd_grad_plot --dependency=afterok:"$JG" \
        scripts/slurm/plot_proxy_gradient_cpu.sbatch >/dev/null
    SUB_OVERRIDES=()
    RESULTS+=("$CK"
              "$RUN_DIR/gradient/proxy_grad_${ARM}_overfit_scatter.png"
              "$RUN_DIR/gradient/proxy_grad_${ARM}_overfit_convergence.png")
    echo "  !! the overfit checkpoint is NOT evidence (ignore-gate); it exists"
    echo "  !! only to compare the gradient field before/after a reward step."
fi

# --- overfit ONE tile to the proxy -> render GIF + before/after -------------
# Field-space gradient ascent on a single SR2 tile until the reward converges,
# then a GIF of the process. BOX/TILE default to set0's most host-rich tile so
# the tag is deterministic and both jobs agree on it.
if [ "$STAGE" = "tile" ] || [ "$STAGE" = "all" ]; then
    B="${BOX:-set0}"; T="${TILE:-486}"; SEED="${SEED:-0}"
    ITERS="${ITERS:-800}"; FIELD_LR="${FIELD_LR:-1e-3}"
    SAVE_EVERY="${SAVE_EVERY:-5}"
    # host/low-k control knobs (pass through to the optimiser; -1 = config default)
    W_REWARD="${W_REWARD:--1}"
    W_LOWK="${W_LOWK:--1}"; W_DENSITY="${W_DENSITY:--1}"; W_PROX="${W_PROX:--1}"
    W_HOST="${W_HOST:-0}"; HOST_QUANTILE="${HOST_QUANTILE:-0.99}"
    TOL="${TOL:-1e-4}"; PATIENCE="${PATIENCE:-20}"
    # ROCKSTAR=1 saves snapshots and re-runs the real halo finder on each.
    ROCKSTAR="${ROCKSTAR:-0}"
    CKPT_EVERY="${CKPT_EVERY:-100}"
    [ "$ROCKSTAR" = "1" ] || CKPT_EVERY=0
    # LABEL keeps a new experiment's outputs separate from older ones.
    LABEL="${LABEL:-}"
    TAG="${B}_t${T}_${ARM}${LABEL:+__$LABEL}"
    SUB_OVERRIDES=("ARM=$ARM" "BOX=$B" "TILE=$T" "SEED=$SEED" "ITERS=$ITERS"
                   "FIELD_LR=$FIELD_LR" "SAVE_EVERY=$SAVE_EVERY"
                   "CKPT_EVERY=$CKPT_EVERY" "W_REWARD=$W_REWARD" "W_LOWK=$W_LOWK"
                   "W_DENSITY=$W_DENSITY" "W_PROX=$W_PROX" "W_HOST=$W_HOST"
                   "HOST_QUANTILE=$HOST_QUANTILE" "LABEL=$LABEL"
                   "TOL=$TOL" "PATIENCE=$PATIENCE")
    # ROCK_ONLY=1 resumes from existing snapshots: skip the optimiser and run
    # only the Rockstar monitor + plots (e.g. after a quota-truncated attempt).
    ROCK_ONLY="${ROCK_ONLY:-0}"
    if [ "$ROCK_ONLY" = "1" ]; then
        JO=""; DEP_JO=""
        echo "  (ROCK_ONLY: reusing existing snapshots, optimiser skipped)"
    else
        JO=$(sub "overfit tile $B/t$T -> proxy arm $ARM (CPU)" sd_tile_of \
            scripts/slurm/overfit_tile_cpu.sbatch)
        SUB_OVERRIDES=()
        DEP_JO="--dependency=afterok:$JO"
    fi

    # The GIF/before-after renderer overlays the REAL Rockstar dR_occ when the
    # monitor rows exist, so with ROCKSTAR=1 it must wait for the array; without
    # it, render straight after the optimiser.
    RENDER_DEP="$JO"
    if [ "$ROCKSTAR" = "1" ]; then
        # over-provision array tasks: iter0 + floor(ITERS/CKPT_EVERY) + final.
        NMAX=$(( ITERS / CKPT_EVERY + 2 ))
        # THROTTLE: each Rockstar task writes a ~3.76 GB gadget2 before the halo
        # finder runs and only deletes it afterwards. Too many at once blew the
        # per-user disk QUOTA -> truncated gadget2 + empty rockstar.log + silent
        # kill (see memory rockstar-label-fails-are-quota). %ROCK_CONC caps how
        # many run simultaneously so peak disk stays bounded. The ladder runs
        # several rungs at once, so keep this small.
        ROCK_CONC="${ROCK_CONC:-2}"
        SUB_OVERRIDES=("TAG=$TAG")
        JR=$(sub "REAL Rockstar on snapshots [0-$NMAX] max $ROCK_CONC/node (CPU array)" \
            sd_tile_rock \
            --array=0-"$NMAX"%"$ROCK_CONC" $DEP_JO \
            scripts/slurm/rockstar_monitor_tile_cpu.sbatch)
        SUB_OVERRIDES=()
        RENDER_DEP="$JR"
        SUB_OVERRIDES=("TAG=$TAG")
        sub "overlay proxy vs real reward for $TAG (CPU)" sd_tile_rock_plot \
            --dependency=afterok:"$JR" \
            scripts/slurm/plot_tile_rockstar_cpu.sbatch >/dev/null
        SUB_OVERRIDES=()
        RESULTS+=("$RUN_DIR/tile_overfit/rockstar_monitor_${TAG}/tile_rockstar_${TAG}.png")
        echo "  !! ROCKSTAR=1: real halo finder on each snapshot; iter0 measured"
        echo "  !! dR_occ ~0 (splice == frozen) is the sanity check. The GIF and"
        echo "  !! before/after PNG carry the REAL dR_occ + host counts too."
    fi

    DEP_RENDER=""; [ -n "$RENDER_DEP" ] && DEP_RENDER="--dependency=afterok:$RENDER_DEP"
    SUB_OVERRIDES=("TAG=$TAG")
    sub "render GIF + before/after for $TAG (CPU)" sd_tile_gif \
        $DEP_RENDER \
        scripts/slurm/render_tile_overfit_cpu.sbatch >/dev/null
    SUB_OVERRIDES=()
    RESULTS+=("$RUN_DIR/tile_overfit/tile_overfit_${TAG}.gif"
              "$RUN_DIR/tile_overfit/tile_overfit_${TAG}_beforeafter.png"
              "$RUN_DIR/tile_overfit/tile_overfit_${TAG}.json")
    echo "  !! surrogate fixed point, not a physical one: the field is optimised"
    echo "  !! directly against the proxy and will exploit it; the density/low-k"
    echo "  !! guards bound that but do not remove it."
fi

# --- monitor + result-file summary -----------------------------------------
# Read the jobs file sub() wrote (survives its subshell). Print one tail -F per
# job and a combined tail across all of them; -F (not -f) follows by NAME and
# retries, so it works even before a queued job's log file exists.
echo
echo "=== MONITOR ==="
LOGPATHS=()
while IFS='|' read -r jid logname label; do
    [ -z "${jid:-}" ] && continue
    lp="$LOGS/slurm-${logname}-${jid}.out"
    LOGPATHS+=("$lp")
    printf '  # %s (job %s)\n  tail -F %s\n' "$label" "$jid" "$lp"
done < "$JOBS_FILE"
if [ "${#LOGPATHS[@]}" -gt 1 ]; then
    echo "  # all jobs of this submission at once:"
    echo "  tail -F ${LOGPATHS[*]}"
fi
echo
echo "=== RESULT FILES (appear when the chain finishes) ==="
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "  (dir) $RUN_DIR"
echo
echo "=== live queue state: squeue -u \$USER -o '%.10i %.18j %.10T %R'"
