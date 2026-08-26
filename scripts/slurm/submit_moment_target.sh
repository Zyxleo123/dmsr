#!/bin/bash
# Build the step-5 projected target x1 = Pi(Psi_HR - Psi_SR2) for one or more
# boxes and draw the figure that says whether each is correct.
# docs/sr2_moment_constraint.md 5.1.
#
#   features (CPU)  LR-Rockstar host features, ONLY for boxes that lack them --
#                   one job for all such boxes.          lagrangian_host_build_cpu
#   build   (GPU)   generate SR2, form the residual, project it, write target +
#                   diagnostics -- one job per box.       build_moment_target_gpu
#   render  (CPU)   redraw the verification figure, afterok the box's build.
#                                                        render_moment_target_cpu
#
# Per box it is a chain (features? -> build -> render); across boxes the chains
# are SIBLINGS, so all boxes queue concurrently. The features job is submitted
# once for every box still missing its npz, and each such box's build waits on
# it (afterok); boxes that already have features get no such dependency. Boxes
# whose features never appear skip themselves (the build gates to exit 0).
#
# This submitter ONLY calls sbatch. Configuration goes into timestamped env
# files passed as POSITIONAL arguments -- never `sbatch --export`, which on this
# cluster sets SLURM_GET_USER_ENV=1 and gets the job requeued and HELD.
#
#   MT_BOXES=set0,set1,...,set15 bash scripts/slurm/submit_moment_target.sh
#   bash scripts/slurm/submit_moment_target.sh                 # MT_BOXES=set8
#   SKIP_FEATURES=1 bash scripts/slurm/submit_moment_target.sh # assume features exist
#   ONLY=render bash scripts/slurm/submit_moment_target.sh     # redraw only
#   DRY=1 MT_BOXES=set0,set8 bash scripts/slurm/submit_moment_target.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
ONLY="${ONLY:-both}"                 # both | build | render
SKIP_FEATURES="${SKIP_FEATURES:-0}"  # 1 = never build features, assume present

MT_BOXES="${MT_BOXES:-${MT_BOX:-set8}}"
MT_MODE="${MT_MODE:-affine}"
MT_SEED="${MT_SEED:-0}"
MT_BATCH="${MT_BATCH:-8}"
MT_DTYPE="${MT_DTYPE:-float16}"
MT_TOP_HOSTS="${MT_TOP_HOSTS:-24}"
MT_CROP="${MT_CROP:-128}"
MT_FORCE="${MT_FORCE:-0}"
REDSHIFT="${REDSHIFT:-0.0}"

IFS=',' read -r -a BOXES <<< "$MT_BOXES"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"

# --- which boxes need features? -------------------------------------------
FEAT_NEEDED=()
for b in "${BOXES[@]}"; do
    npz="$REWARD_ROOT/lagrangian_host/$b/${b}_lagrangian_host.npz"
    [[ -r "$npz" ]] || FEAT_NEEDED+=("$b")
done
FEAT_LIST="$(IFS=,; echo "${FEAT_NEEDED[*]:-}")"

echo "boxes: $MT_BOXES"
echo "features present:  $(comm -23 <(printf '%s\n' "${BOXES[@]}" | sort) \
    <(printf '%s\n' "${FEAT_NEEDED[@]:-}" | sort) | tr '\n' ' ')"
echo "features to build: ${FEAT_LIST:-<none>}"
echo

# --- env files (one shared for the moment target, one for features) --------
MT_ENV="$REWARD_ROOT/env/moment_target_${STAMP}.env"
cat > "$MT_ENV" <<EOT
# Written by scripts/slurm/submit_moment_target.sh at $(date '+%F %T').
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
MT_MODE=$MT_MODE
MT_SEED=$MT_SEED
MT_BATCH=$MT_BATCH
MT_DTYPE=$MT_DTYPE
MT_TOP_HOSTS=$MT_TOP_HOSTS
MT_CROP=$MT_CROP
MT_FORCE=$MT_FORCE
EOT

FEAT_ENV=""
if [ -n "$FEAT_LIST" ] && [ "$SKIP_FEATURES" != "1" ] && [ "$ONLY" != "render" ]; then
    FEAT_ENV="$REWARD_ROOT/env/moment_target_feat_${STAMP}.env"
    cat > "$FEAT_ENV" <<EOT
# Written by scripts/slurm/submit_moment_target.sh at $(date '+%F %T').
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
BOXES=$FEAT_LIST
REDSHIFT=$REDSHIFT
RERUN_ROCKSTAR=0
EOT
fi

echo "moment-target env: $MT_ENV"
[ -n "$FEAT_ENV" ] && echo "features env:      $FEAT_ENV"
echo

if [ "$DRY" = "1" ]; then
    [ -n "$FEAT_ENV" ] && echo "DRY: sbatch scripts/slurm/lagrangian_host_build_cpu.sbatch $FEAT_ENV"
    for b in "${BOXES[@]}"; do
        dep=""
        [[ -n "$FEAT_ENV" ]] && printf '%s\n' "${FEAT_NEEDED[@]}" | grep -qx "$b" \
            && dep="--dependency=afterok:<feat> "
        echo "DRY: sbatch ${dep}scripts/slurm/build_moment_target_gpu.sbatch $MT_ENV MT_BOX=$b"
        echo "DRY: sbatch --dependency=afterok:<build> scripts/slurm/render_moment_target_cpu.sbatch $MT_ENV MT_BOX=$b"
    done
    exit 0
fi

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch jobs.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

# --- features (once, for the boxes that lack them) -------------------------
FID=""
if [ -n "$FEAT_ENV" ]; then
    FID=$(sbatch scripts/slurm/lagrangian_host_build_cpu.sbatch "$FEAT_ENV" | awk '{print $NF}')
    echo "submitted features (CPU) job $FID for: $FEAT_LIST"
fi

# --- per box: build (GPU) -> render (CPU) ----------------------------------
declare -a BIDS RIDS
for b in "${BOXES[@]}"; do
    BID=""; RID=""
    dep=()
    if [ -n "$FID" ] && printf '%s\n' "${FEAT_NEEDED[@]}" | grep -qx "$b"; then
        dep=(--dependency=afterok:"$FID")
    fi
    if [ "$ONLY" != "render" ]; then
        BID=$(sbatch "${dep[@]}" scripts/slurm/build_moment_target_gpu.sbatch \
              "$MT_ENV" "MT_BOX=$b" | awk '{print $NF}')
        echo "submitted build  (GPU) job $BID  $b ${dep[*]:-}"
    fi
    if [ "$ONLY" != "build" ]; then
        rdep=()
        [ -n "$BID" ] && rdep=(--dependency=afterok:"$BID")
        RID=$(sbatch "${rdep[@]}" scripts/slurm/render_moment_target_cpu.sbatch \
              "$MT_ENV" "MT_BOX=$b" | awk '{print $NF}')
        echo "submitted render (CPU) job $RID  $b ${rdep[*]:-}"
    fi
    [ -n "$BID" ] && BIDS+=("$b:$BID")
    [ -n "$RID" ] && RIDS+=("$b:$RID")
done

echo
echo "--- watch ---"
[ -n "$FID" ] && echo "tail -F $REWARD_ROOT/logs/slurm-lag_host-$FID.out   # features"
for x in "${BIDS[@]:-}"; do echo "tail -F $REWARD_ROOT/logs/slurm-momtgt-${x##*:}.out   # ${x%%:*} build"; done
for x in "${RIDS[@]:-}"; do echo "tail -F $REWARD_ROOT/logs/slurm-momtgtfig-${x##*:}.out   # ${x%%:*} render"; done
echo
echo "--- results (per box) ---"
echo "$REWARD_ROOT/moment_target/<box>/<box>_moment_target_summary.json   # the verdict"
echo "$REWARD_ROOT/moment_target/<box>/<box>_moment_target.png            # scp + look"
echo "$REWARD_ROOT/moment_target/<box>/<box>_moment_target.npy            # the training cache"
echo
echo "each build's verdict is the last PASS/CHECK line of its .out and the"
echo "'verdict' key of its summary json."
