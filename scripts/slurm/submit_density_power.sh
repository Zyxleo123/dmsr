#!/bin/bash
# Density P(k)/P_HR(k) for the member-gather arms with fields on disk.
#
#   bash scripts/slurm/submit_density_power.sh
#   DP_GRID_MULT=1 bash scripts/slurm/submit_density_power.sh   # coarser/faster
#   DRY=1 bash scripts/slurm/submit_density_power.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
DP_GRID_MULT="${DP_GRID_MULT:-2}"

F="$REWARD_ROOT/flow_rockstar/fields"
HR="${DP_HR:-$ZFS/DMSR/paired_catnorm/hr/set9.npy}"
# Only arms whose spliced field survived on disk; the diagnostic skips any
# label whose path is missing, so extra entries are harmless.
DP_FIELDS="${DP_FIELDS:-hr=$HR frozen=$F/set9__mgho_frozen_set9__seed0.npy smallhalo=$F/set9__mgho_all_blocks_smallhalo_set9__seed0.npy selfhostguard=$F/set9__mgho_all_blocks_selfhostguard_set9__seed0.npy}"
DP_OUT="${DP_OUT:-$REWARD_ROOT/member_gather/all_hosts/density_power_set9.json}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$(dirname "$DP_OUT")"

ENVFILE="$REWARD_ROOT/env/dpower_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_density_power.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument. DP_FIELDS is a
# space-separated label=path list (a file, so spaces travel fine).
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
DP_FIELDS="$DP_FIELDS"
DP_GRID_MULT=$DP_GRID_MULT
DP_OUT=$DP_OUT
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/density_power_cpu.sbatch $ENVFILE"; exit 0
fi
JID=$(sbatch --parsable scripts/slurm/density_power_cpu.sbatch "$ENVFILE")
echo "submitted dpower -> job $JID (CPU)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-dpower-$JID.out"
echo "result:  $DP_OUT   (plot with gather_all_hosts_plot.py --density-power)"
