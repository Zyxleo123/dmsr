#!/bin/bash
# Decompose the member-gather high-k guard: which k carries `self`'s 1.70x, and
# is that excess correlated structure or grid-scale ringing?
#
#   bash scripts/slurm/submit_highk_spectrum.sh
#   HK_BOX=set9 bash scripts/slurm/submit_highk_spectrum.sh
#   DRY=1 bash scripts/slurm/submit_highk_spectrum.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

HK_BOX="${HK_BOX:-set9}"
HK_ARMS="${HK_ARMS:-all_blocks_self all_blocks_nocentre all_blocks_full all_blocks_radial}"
HK_BINS="${HK_BINS:-20}"
HK_OUT="${HK_OUT:-$REWARD_ROOT/member_gather/highk_spectrum_$HK_BOX}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$(dirname "$HK_OUT")"

ENVFILE="$REWARD_ROOT/env/highk_spectrum_$(date +%Y%m%d_%H%M%S)_$$.env"
# List-valued variables MUST be quoted in the env file: an unquoted
# `HK_ARMS=a b` runs `b` as a command when the preamble sources it, which has
# already killed a whole submission on this cluster.
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_highk_spectrum.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
HK_BOX=$HK_BOX
HK_ARMS="$HK_ARMS"
HK_BINS=$HK_BINS
HK_OUT=$HK_OUT
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
sub() { if [ "$DRY" = "1" ]; then echo "DRY: sbatch $*" >&2; echo DRYID
        else sbatch --parsable "$@"; fi; }

JOB=$(sub scripts/slurm/highk_spectrum_cpu.sbatch "$ENVFILE")
echo "submitted highk spectrum -> job $JOB (CPU)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-hk_spec-$JOB.out"
echo "result:  ${HK_OUT}.json"
