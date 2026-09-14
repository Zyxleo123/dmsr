#!/bin/bash
# kNN-distance go/no-go on set9: HR vs frozen vs the arms whose fields are on
# disk, plus the tile-buffer measurement. CPU diagnostic -> CPU plot (afterok).
#
#   bash scripts/slurm/submit_knn_cdf_diagnostic.sh
#   KC_KS="8 16 32" bash scripts/slurm/submit_knn_cdf_diagnostic.sh   # the original script's ranks
#   DRY=1 bash scripts/slurm/submit_knn_cdf_diagnostic.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

F="$REWARD_ROOT/flow_rockstar/fields"
HR="${KC_HR:-$ZFS/DMSR/paired_catnorm/hr/set9.npy}"
# selfhostguard PASSED the held-out set9 Rockstar gate; smallhalo and the
# condsmall rungs have known Rockstar outcomes too, so the diagnostic can be
# read against an ordering we already trust. Missing paths are skipped.
KC_FIELDS="${KC_FIELDS:-hr=$HR frozen=$F/set9__mgho_frozen_set9__seed0.npy selfhostguard=$F/set9__mgho_all_blocks_selfhostguard_set9__seed0.npy smallhalo=$F/set9__mgho_all_blocks_smallhalo_set9__seed0.npy condsmall_ab=$F/set9__mgho_all_blocks_condsmall_set9__seed0.npy condsmall_fine=$F/set9__mgho_fine_condsmall_set9__seed0.npy}"
KC_KS="${KC_KS:-8 32 64 128}"
KC_N_SAMPLE="${KC_N_SAMPLE:-200000}"
KC_BUFFERS="${KC_BUFFERS:-0 2 4 6 8 12 16}"
KC_BOUNDARY_TILES="${KC_BOUNDARY_TILES:-24}"
KC_OUT="${KC_OUT:-$REWARD_ROOT/member_gather/knn_cdf/knn_cdf_set9.json}"
KC_ORDER="${KC_ORDER:-frozen selfhostguard smallhalo condsmall_ab condsmall_fine}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$(dirname "$KC_OUT")"

ENVFILE="$REWARD_ROOT/env/knncdf_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_knn_cdf_diagnostic.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument. List values are QUOTED
# (env-file-list-values-must-be-quoted: an unquoted list runs its 2nd word).
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
KC_FIELDS="$KC_FIELDS"
KC_KS="$KC_KS"
KC_N_SAMPLE=$KC_N_SAMPLE
KC_BUFFERS="$KC_BUFFERS"
KC_BOUNDARY_TILES=$KC_BOUNDARY_TILES
KC_OUT=$KC_OUT
KC_ORDER="$KC_ORDER"
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/knn_cdf_diagnostic_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch --dependency=afterok:<id> scripts/slurm/knn_cdf_plot_cpu.sbatch $ENVFILE"
    exit 0
fi
J1=$(sbatch --parsable scripts/slurm/knn_cdf_diagnostic_cpu.sbatch "$ENVFILE")
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/slurm/knn_cdf_plot_cpu.sbatch "$ENVFILE")
echo "submitted knncdf  -> job $J1 (CPU)"
echo "submitted knnplot -> job $J2 (CPU, afterok:$J1)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-knncdf-$J1.out"
echo "         tail -F $REWARD_ROOT/logs/slurm-knnplot-$J2.out"
echo "result:  $KC_OUT"
echo "figure:  ${KC_OUT%.json}.png"
