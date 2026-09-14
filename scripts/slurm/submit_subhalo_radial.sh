#!/bin/bash
# Collect subhalo mass vs. distance from host centre for HR and the tuned
# `self` arm, then plot. One CPU job (parse catalogs -> npz -> figures).
#
#   bash scripts/slurm/submit_subhalo_radial.sh
#   SR_RUN_DIR=$REWARD/member_gather/all_blocks_self/holdout_set9 \
#     bash scripts/slurm/submit_subhalo_radial.sh   # restrict to edited region
#   DRY=1 bash scripts/slurm/submit_subhalo_radial.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

SR_BOX="${SR_BOX:-set9}"
SR_KINDS="${SR_KINDS:-hr,self}"
SR_ARM_TAGS="${SR_ARM_TAGS:-self=mgho_all_blocks_self_${SR_BOX}}"
SR_RUN_DIR="${SR_RUN_DIR:-}"
SR_DILATE="${SR_DILATE:-2.0}"
SR_OUT_DIR="${SR_OUT_DIR:-$REWARD_ROOT/subhalo_radial/${SR_BOX}}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$SR_OUT_DIR"

ENVFILE="$REWARD_ROOT/env/subhalo_radial_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_subhalo_radial.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
SR_BOX=$SR_BOX
SR_KINDS=$SR_KINDS
SR_ARM_TAGS=$SR_ARM_TAGS
SR_RUN_DIR=$SR_RUN_DIR
SR_DILATE=$SR_DILATE
SR_OUT_DIR=$SR_OUT_DIR
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/subhalo_radial_cpu.sbatch $ENVFILE"; exit 0
fi
JID=$(sbatch --parsable scripts/slurm/subhalo_radial_cpu.sbatch "$ENVFILE")
echo "submitted subhalo_radial -> job $JID (CPU)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-subhalo_radial-$JID.out"
echo "results: $SR_OUT_DIR/summary.json"
echo "figures: $SR_OUT_DIR/subhalo_radial.png  $SR_OUT_DIR/subhalo_radial_profile.png"
