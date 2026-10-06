#!/bin/bash
# Why is there no visible overdensity where Rockstar says there are subhalos?
#
# The host figures project the host's whole ~25 Mpc/h Lagrangian column. A
# 200-particle subhalo is ~0.1 Mpc/h across, so that column stacks ~250
# subhalo-diameters of unrelated material over it and even a perfectly real
# object shows up as a percent-level tint. This chain replaces that picture with
# one panel per subhalo, in a slab only as deep as the subhalo -- and, because a
# picture cannot settle it either, with the excess measured at five slab depths
# from ±1 R_vir out to the full column.
#
# READ THE CONTROL FIRST. The `hrtruth` target runs the identical measurement on
# HR's own subhalos in the HR field. Whatever that scores is what "real" looks
# like for this box, this resolution and this aperture; an arm is judged against
# it, not against 1.
#
#   bash scripts/slurm/submit_subhalo_zoom.sh
#   SZ_ARMS="self soft knn hrtruth" bash scripts/slurm/submit_subhalo_zoom.sh
#   SZ_USE_FIELDS=0 bash scripts/slurm/submit_subhalo_zoom.sh   # tiles only
#   PLOT_ONLY=1 bash scripts/slurm/submit_subhalo_zoom.sh       # re-render
#   DRY=1 bash scripts/slurm/submit_subhalo_zoom.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
MG="$REWARD_ROOT/member_gather"
FIELDS="$REWARD_ROOT/flow_rockstar/fields"
DRY="${DRY:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"

SZ_BOX="${SZ_BOX:-set9}"
SZ_HOST_ID="${SZ_HOST_ID:-168880}"
SZ_SEED="${SZ_SEED:-0}"
SZ_N="${SZ_N:-9}"
SZ_MIN_P="${SZ_MIN_P:-50}"
SZ_DEPTHS="${SZ_DEPTHS:-1,2,5,10,-1}"
SZ_HALF_XY_RVIR="${SZ_HALF_XY_RVIR:-4.0}"
SZ_BINS="${SZ_BINS:-96}"
SZ_AXIS="${SZ_AXIS:-1}"
# Bin the complete box rather than only the host's Lagrangian tiles when the
# spliced field is on disk. The tile set omits material that drifted in from
# other tiles, which biases the local background low and every contrast high.
SZ_USE_FIELDS="${SZ_USE_FIELDS:-1}"
# Where the paired dataset lives; matches configs/reward/reward.yaml data.root.
SZ_DATA_ROOT="${SZ_DATA_ROOT:-/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"

# self  the baseline arm; soft  all_blocks_sm_coarse_soft, the soft-membership
# arm that wins the held-out gate; knn  all_blocks_knn6d_w5, the kNN-CDF term on
# top of self; hrtruth  the control, HR's own subhalos in the HR field.
SZ_ARMS="${SZ_ARMS:-hrtruth self soft knn}"

declare -A ARM_OF=(
    [self]=all_blocks_self
    [soft]=all_blocks_sm_coarse_soft
    [knn]=all_blocks_knn6d_w5
    [hrtruth]=all_blocks_self
)

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
# tag -> splice job id, so the two targets that share an arm (self and the
# hrtruth control both read all_blocks_self) queue ONE splice between them
# rather than two jobs racing to write the same 3.2 GB file.
declare -A SPLICE_JOB=()
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

sub() {
    local dep="$1"; shift
    if [ "$DRY" = "1" ]; then echo "DRY: sbatch ${dep:+--dependency=afterok:$dep} $*" >&2; echo 0; return; fi
    local out; out=$(sbatch ${dep:+--dependency=afterok:$dep} --parsable "$@")
    echo "${out%%;*}"
}

echo "=== per-subhalo slab zoom -- box $SZ_BOX, host $SZ_HOST_ID"
echo "    $SZ_N subhalos each, >= ${SZ_MIN_P}p, depths $SZ_DEPTHS (R_vir of the SUBHALO)"
echo

declare -a WATCH=()
for t in $SZ_ARMS; do
    ARM="${ARM_OF[$t]:-}"
    [ -n "$ARM" ] || { echo "unknown arm '$t'" >&2; exit 1; }
    RUN_DIR="$MG/$ARM/holdout_$SZ_BOX"
    TAG="mgho_${ARM}_${SZ_BOX}"
    SELECT=cand
    [ "$t" = "hrtruth" ] && SELECT=hr

    if [ ! -f "$RUN_DIR/tiles.npz" ]; then
        echo "──────── $t: no $RUN_DIR/tiles.npz -- SKIPPED"; echo; continue
    fi

    OUT="$RUN_DIR/figures/subzoom_${t}"
    FIG_DIR="$RUN_DIR/figures/subzoom_${t}_h${SZ_HOST_ID}"

    # Full-box fields, when they survive. A purged field falls back to tiles;
    # the data file records which was used, so a number can never be read
    # without knowing what it was measured on.
    F_OUT=""; F_FROZEN=""; F_HR=""; DEP=""
    if [ "$SZ_USE_FIELDS" = "1" ]; then
        _c="$FIELDS/${SZ_BOX}__${TAG}__seed${SZ_SEED}.npy"
        _f="$FIELDS/${SZ_BOX}__mgho_frozen_${SZ_BOX}__seed${SZ_SEED}.npy"
        # The HR field is the paired dataset's own box, and it is never purged.
        _h="$SZ_DATA_ROOT/hr/${SZ_BOX}.npy"

        # A purged candidate field is rebuilt from the arm's tiles.npz, which is
        # never purged -- a CPU numpy copy, no GPU. It is worth queueing rather
        # than falling back, because the CONTROL must be measured exactly the
        # way the arms are: an hrtruth row binned from tiles while the arms are
        # binned from the full box calibrates nothing.
        if [ ! -f "$_c" ] && [ -f "$_f" ] && [ -f "$_h" ]; then
            if [ -n "${SPLICE_JOB[$TAG]:-}" ]; then
                DEP="${SPLICE_JOB[$TAG]}"
                echo "──────── $t  (arm $ARM, select $SELECT)"
                echo "    field purged; reusing the splice queued for $TAG (job $DEP)"
            else
                echo "──────── $t  (arm $ARM, select $SELECT)"
                echo "    field purged; re-splicing from $RUN_DIR/tiles.npz"
                SPENV="$REWARD_ROOT/env/subzoom_splice_${t}_${STAMP}.env"
                cat > "$SPENV" <<EOS
# Written by scripts/slurm/submit_subhalo_zoom.sh at $(date '+%F %T').
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
HG_RUN_DIR=$RUN_DIR
HG_BOX=$SZ_BOX
HG_SEED=$SZ_SEED
HG_WHICH=out
HG_RS_TAG=$TAG
EOS
                DEP=$(sub "" scripts/slurm/gather_splice_cpu.sbatch "$SPENV")
                SPLICE_JOB[$TAG]="$DEP"
                WATCH+=("hg_splice:$DEP")
                echo "    splice -> job $DEP"
            fi
            F_OUT="$_c"; F_FROZEN="$_f"; F_HR="$_h"
            SPLICED=1
        elif [ -f "$_c" ] && [ -f "$_f" ] && [ -f "$_h" ]; then
            F_OUT="$_c"; F_FROZEN="$_f"; F_HR="$_h"
        fi
    fi

    [ -n "${SPLICED:-}" ] || echo "──────── $t  (arm $ARM, select $SELECT)"
    unset SPLICED || true
    [ -n "$F_OUT" ] && echo "    particles: full box" || echo "    particles: host tiles (no field, --no-fields)"

    ENVFILE="$REWARD_ROOT/env/subzoom_${t}_${STAMP}.env"
    cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_subhalo_zoom.sh at $(date '+%F %T');
# sourced by the job preambles as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
SZ_RUN_DIR=$RUN_DIR
SZ_BOX=$SZ_BOX
SZ_TAG=$TAG
SZ_HOST_ID=$SZ_HOST_ID
SZ_SELECT=$SELECT
SZ_N=$SZ_N
SZ_MIN_P=$SZ_MIN_P
SZ_HALF_XY_RVIR=$SZ_HALF_XY_RVIR
SZ_DEPTHS=$SZ_DEPTHS
SZ_AXIS=$SZ_AXIS
SZ_BINS=$SZ_BINS
SZ_OUT=$OUT
SZ_FIG_DIR=$FIG_DIR
SZ_FIELD_OUT=$F_OUT
SZ_FIELD_FROZEN=$F_FROZEN
SZ_FIELD_HR=$F_HR
EOT

    if [ "$PLOT_ONLY" != "1" ]; then
        J1=$(sub "$DEP" scripts/slurm/subhalo_zoom_data_cpu.sbatch "$ENVFILE")
        echo "    data -> job $J1"
        WATCH+=("sz_data:$J1")
        DEP="$J1"
    fi
    J2=$(sub "$DEP" scripts/slurm/subhalo_zoom_plot_cpu.sbatch "$ENVFILE")
    echo "    plot -> job $J2"
    WATCH+=("sz_plot:$J2")
    echo
done

echo "watch:"
for w in "${WATCH[@]}"; do
    echo "  tail -F $REWARD_ROOT/logs/slurm-${w%%:*}-${w##*:}.out"
done
echo
echo "results:"
echo "  numbers  $MG/<arm>/holdout_$SZ_BOX/figures/subzoom_<t>.json"
echo "           per subhalo: contrast_3d, then the ladder from ±1 R_vir to the"
echo "           full column, each with ratio and snr"
echo "  figures  $MG/<arm>/holdout_$SZ_BOX/figures/subzoom_<t>_h${SZ_HOST_ID}/"
echo "           fig1_subhalo_grid.png  tuned | frozen | HR, thinnest slab"
echo "           fig2_depth_ladder.png  one subhalo across five slab depths"
echo "           fig3_contrast.png      the same as numbers -- quote this one"
echo
echo "read hrtruth first: it is what a REAL subhalo scores on this measurement."
