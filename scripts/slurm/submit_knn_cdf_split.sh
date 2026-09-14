#!/bin/bash
# kNN go/no-go round 2 on set9: membership split + neighbour-tile boundary.
# CPU diagnostic -> CPU plot (afterok). Two cpu slots; the cpu partition caps a
# user at 50 queued jobs (cpu-partition-50-job-cap).
#
#   bash scripts/slurm/submit_knn_cdf_split.sh              # box-wide (round 2)
#   REGION=1 bash scripts/slurm/submit_knn_cdf_split.sh     # the 32 spliced tiles only
#   NO_PLOT=1 bash scripts/slurm/submit_knn_cdf_split.sh     # one slot only
#
# REGION=1 is the round-2 follow-up: every arm field on disk replaces only 32
# tiles (6.25%) of the frozen box, so box-wide samples dilute the arms ~16x and
# random boundary tiles never test stale frozen neighbours.
#   DRY=1 bash scripts/slurm/submit_knn_cdf_split.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
NO_PLOT="${NO_PLOT:-0}"

F="$REWARD_ROOT/flow_rockstar/fields"
HR="$ZFS/DMSR/paired_catnorm/hr/set9.npy"
KS_FIELDS="${KS_FIELDS:-hr=$HR frozen=$F/set9__mgho_frozen_set9__seed0.npy selfhostguard=$F/set9__mgho_all_blocks_selfhostguard_set9__seed0.npy smallhalo=$F/set9__mgho_all_blocks_smallhalo_set9__seed0.npy condsmall_ab=$F/set9__mgho_all_blocks_condsmall_set9__seed0.npy condsmall_fine=$F/set9__mgho_fine_condsmall_set9__seed0.npy}"
KS_CATALOG="${KS_CATALOG:-$(ls $REWARD_ROOT/halos/set9__hr__hr/*_rockstar/halos_*.ascii 2>/dev/null | head -1)}"
KS_OWNER="${KS_OWNER:-$REWARD_ROOT/halos_particles/set9__hr__hr/set9_hr_owner.npy}"
REGION="${REGION:-0}"
if [ "$REGION" = "1" ]; then
    KS_REGION_JSON="${KS_REGION_JSON:-$F/set9__mgho_all_blocks_selfhostguard_set9__seed0.json}"
    [[ -r "$KS_REGION_JSON" ]] || { echo "no region json: $KS_REGION_JSON"; exit 1; }
    KS_OUT="${KS_OUT:-$REWARD_ROOT/member_gather/knn_cdf/knn_cdf_split_region_set9.json}"
    KS_BOUNDARY_TILES="${KS_BOUNDARY_TILES:-32}"
    KS_SIZES="${KS_SIZES:-0 4 8}"
else
    KS_REGION_JSON="${KS_REGION_JSON:-}"
    KS_OUT="${KS_OUT:-$REWARD_ROOT/member_gather/knn_cdf/knn_cdf_split_set9.json}"
fi
[[ -n "$KS_CATALOG" && -r "$KS_CATALOG" ]] || { echo "no HR catalog for set9"; exit 1; }
[[ -r "$KS_OWNER" ]] || { echo "no owner array: $KS_OWNER"; exit 1; }

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$(dirname "$KS_OUT")"
ENVFILE="$REWARD_ROOT/env/knnsplit_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_knn_cdf_split.sh at $(date '+%F %T').
# List values QUOTED (env-file-list-values-must-be-quoted).
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
KS_FIELDS="$KS_FIELDS"
KS_CATALOG=$KS_CATALOG
KS_OWNER=$KS_OWNER
KS_OUT=$KS_OUT
KS_TUNED_LABEL="${KS_TUNED_LABEL:-selfhostguard}"
KS_KS="${KS_KS:-8 32 64 128}"
KS_N_PER_CLASS=${KS_N_PER_CLASS:-100000}
KS_SIZES="${KS_SIZES:-0 2 4 8 16}"
KS_BOUNDARY_TILES=${KS_BOUNDARY_TILES:-16}
KS_REGION_JSON="$KS_REGION_JSON"
KS_N_UNIFORM=${KS_N_UNIFORM:-200000}
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/knn_cdf_split_cpu.sbatch $ENVFILE"
    [ "$NO_PLOT" = "1" ] || echo "DRY: sbatch --dependency=afterok:<id> scripts/slurm/knn_cdf_split_plot_cpu.sbatch $ENVFILE"
    exit 0
fi
J1=$(sbatch --parsable scripts/slurm/knn_cdf_split_cpu.sbatch "$ENVFILE")
echo "submitted knnsplit -> job $J1 (CPU)"
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-knnsplit-$J1.out"
if [ "$NO_PLOT" != "1" ]; then
    J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/slurm/knn_cdf_split_plot_cpu.sbatch "$ENVFILE")
    echo "submitted knnsplitplot -> job $J2 (CPU, afterok:$J1)"
    echo "watch:   tail -F $REWARD_ROOT/logs/slurm-knnsplitplot-$J2.out"
else
    echo "plot later: sbatch scripts/slurm/knn_cdf_split_plot_cpu.sbatch $ENVFILE"
fi
echo "result:  $KS_OUT"
echo "figure:  ${KS_OUT%.json}.png"
