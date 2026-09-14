#!/bin/bash
# The held-out host figure, zoomed to R_vir: same three panels, same layout,
# rebuilt at full tile resolution instead of cropped.
#
# The stage-1 panels are 0.10 Mpc/h a pixel, so R_vir spans ~30 of them and a
# crop would just enlarge a thumbnail. The export's tiles.npz keeps every
# spliced tile for all three arms at 64^3, so the window can be rebinned as
# finely as the particles allow -- and it is the only surviving source for the
# tuned panel, whose 3.2 GB field was purged.
#
# zoom data (CPU, minutes) --afterok--> render fig1 (CPU, seconds).
# The data job re-derives stage 1's own panels from the tiles first and refuses
# to write unless they match exactly.
#
#   bash scripts/slurm/submit_gather_zoom.sh
#   GF_ZOOM_RVIR=1.5 GF_ZOOM_BINS=320 bash scripts/slurm/submit_gather_zoom.sh
#   GF_ZOOM_DEPTH_RVIR=-1 bash scripts/slurm/submit_gather_zoom.sh   # no depth cut
#   PLOT_ONLY=1 bash scripts/slurm/submit_gather_zoom.sh             # re-render only
#   DRY=1 bash scripts/slurm/submit_gather_zoom.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"

GF_BOX="${GF_BOX:-set9}"
GF_HOST_ID="${GF_HOST_ID:-168880}"
SELF_ARM="${SELF_ARM:-all_blocks_self}"
GF_RUN_DIR="${GF_RUN_DIR:-$REWARD_ROOT/member_gather/$SELF_ARM/holdout_$GF_BOX}"
GF_OUT="${GF_OUT:-$GF_RUN_DIR/figures/data_h${GF_HOST_ID}}"

# window half-width in R_vir; 1.15 puts the circle just inside the panel
GF_ZOOM_RVIR="${GF_ZOOM_RVIR:-1.15}"
# slab half-thickness in R_vir: 0 = same as the window, <0 = the full 25 Mpc/h
# Lagrangian column stage 1 projects
GF_ZOOM_DEPTH_RVIR="${GF_ZOOM_DEPTH_RVIR:-0}"
GF_ZOOM_BINS="${GF_ZOOM_BINS:-256}"
GF_ZOOM_FIGURES="${GF_ZOOM_FIGURES:-fig1}"
GF_ZOOM_TAG="${GF_ZOOM_TAG:-rvir${GF_ZOOM_RVIR}}"
GF_ZOOM_OUT="${GF_ZOOM_OUT:-$GF_RUN_DIR/figures/zoom_${GF_ZOOM_TAG}_h${GF_HOST_ID}}"
GF_ZOOM_FIG_DIR="${GF_ZOOM_FIG_DIR:-$GF_RUN_DIR/figures/h${GF_HOST_ID}_zoom_${GF_ZOOM_TAG}}"

for f in "$GF_RUN_DIR/tiles.npz" "$GF_OUT.npz" "$GF_OUT.json"; do
    [ -f "$f" ] || { echo "missing input: $f" >&2; exit 1; }
done
mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$GF_ZOOM_FIG_DIR"

ENVFILE="$REWARD_ROOT/env/gather_zoom_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_gather_zoom.sh at $(date '+%F %T');
# sourced by the job preambles as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
GF_BOX=$GF_BOX
GF_HOST_ID=$GF_HOST_ID
GF_RUN_DIR=$GF_RUN_DIR
GF_OUT=$GF_OUT
GF_ZOOM_OUT=$GF_ZOOM_OUT
GF_ZOOM_FIG_DIR=$GF_ZOOM_FIG_DIR
GF_ZOOM_RVIR=$GF_ZOOM_RVIR
GF_ZOOM_DEPTH_RVIR=$GF_ZOOM_DEPTH_RVIR
GF_ZOOM_BINS=$GF_ZOOM_BINS
GF_ZOOM_FIGURES=$GF_ZOOM_FIGURES
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
sub() { if [ "$DRY" = "1" ]; then echo "DRY: sbatch $*" >&2; echo DRYID
        else sbatch --parsable "$@"; fi; }

DEP=()
if [ "$PLOT_ONLY" != "1" ]; then
    DATA=$(sub scripts/slurm/gather_zoom_data_cpu.sbatch "$ENVFILE")
    echo "submitted zoom data -> job $DATA (CPU)"
    DEP=(--dependency=afterok:"$DATA")
    echo "watch data: tail -F $REWARD_ROOT/logs/slurm-gfig_zoom-$DATA.out"
fi
PLOT=$(sub ${DEP[@]+"${DEP[@]}"} scripts/slurm/gather_zoom_plot_cpu.sbatch "$ENVFILE")
echo "submitted zoom plot -> job $PLOT (CPU)"

echo
echo "watch plot: tail -F $REWARD_ROOT/logs/slurm-gfig_zplot-$PLOT.out"
echo
echo "data file:  $GF_ZOOM_OUT.npz"
echo "figure:     $GF_ZOOM_FIG_DIR/fig1_host_density.png"
