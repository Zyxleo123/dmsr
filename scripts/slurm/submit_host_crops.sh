#!/bin/bash
# Collect Option A's host-frame crops, then render the page.
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster makes sbatch set SLURM_GET_USER_ENV=1 and gets the job requeued+held.
#
#   bash scripts/slurm/submit_host_crops.sh
#   BOXES=set8,set9 CROP_N_HOSTS=24 bash scripts/slurm/submit_host_crops.sh
#   RENDER_ONLY=1 bash scripts/slurm/submit_host_crops.sh   # redraw only
#   DRY=1 bash scripts/slurm/submit_host_crops.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
RENDER_ONLY="${RENDER_ONLY:-0}"

BOXES="${BOXES:-set8}"
CROP_N_HOSTS="${CROP_N_HOSTS:-16}"
CROP_N_CLUSTERS="${CROP_N_CLUSTERS:-6}"
CROP_PER_BIN="${CROP_PER_BIN:-2}"
CROP_SCALE="${CROP_SCALE:-1.0}"
CROP_GRID="${CROP_GRID:-48}"
CROP_MIN_P="${CROP_MIN_P:-200}"
CROP_SEED="${CROP_SEED:-0}"
CROP_VIEW_HOSTS="${CROP_VIEW_HOSTS:-0}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/host_crops_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_host_crops.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
BOXES=$BOXES
CROP_N_HOSTS=$CROP_N_HOSTS
CROP_N_CLUSTERS=$CROP_N_CLUSTERS
CROP_PER_BIN=$CROP_PER_BIN
CROP_SCALE=$CROP_SCALE
CROP_GRID=$CROP_GRID
CROP_MIN_P=$CROP_MIN_P
CROP_SEED=$CROP_SEED
CROP_VIEW_HOSTS=$CROP_VIEW_HOSTS
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch job.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/host_crops_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch --dependency=afterok:<collect> scripts/slurm/host_crops_render_cpu.sbatch $ENVFILE"
    exit 0
fi

CID=""
if [ "$RENDER_ONLY" != "1" ]; then
    CID=$(sbatch scripts/slurm/host_crops_cpu.sbatch "$ENVFILE" | awk '{print $NF}')
    echo "submitted crops (collect) job $CID"
fi

DEP=()
if [ -n "$CID" ]; then DEP=(--dependency=afterok:"$CID"); fi
RID=$(sbatch "${DEP[@]}" scripts/slurm/host_crops_render_cpu.sbatch \
      "$ENVFILE" | awk '{print $NF}')
echo "submitted crops_view (render) job $RID"
echo
if [ -n "$CID" ]; then
    echo "watch collect: tail -F $REWARD_ROOT/logs/slurm-crops-$CID.out"
fi
echo "watch render:  tail -F $REWARD_ROOT/logs/slurm-crops_view-$RID.out"
echo
echo "results in: $REWARD_ROOT/lagrangian_host/<box>/"
for b in ${BOXES//,/ }; do
    echo "  $b:"
    echo "    ${b}_host_crops.json   learnability table (redraw source)"
    echo "    ${b}_host_crops.npz    quantised crop volumes + subhalo tables"
    echo "    host_crops_${b}.html   the page -- scp it off and open it"
done
