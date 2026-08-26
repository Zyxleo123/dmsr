#!/bin/bash
# Submit the peak-targeting diagnostic (can a DoG peak mark a missing subhalo?).
# ONLY calls sbatch. Configuration goes into ONE timestamped env file passed as a
# POSITIONAL argument -- never `sbatch --export` (requeued+held on this cluster).
#
#   bash scripts/slurm/submit_peak_targeting.sh
#   DRY=1 bash scripts/slurm/submit_peak_targeting.sh
#   DOG_FINE=1 DOG_COARSE=6 bash scripts/slurm/submit_peak_targeting.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

BOXES="${BOXES:-set8,set9}"
NG_PEAK="${NG_PEAK:-512}"
DOG_FINE="${DOG_FINE:-1.5}"
DOG_COARSE="${DOG_COARSE:-8.0}"
DOG_THRESHOLDS="${DOG_THRESHOLDS:-0.5,1,2,4}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-1.0}"
R_MULT="${R_MULT:-2.0}"
OUT_NAME="${OUT_NAME:-peak_targeting}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/peak_target_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_peak_targeting.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
BOXES=$BOXES
NG_PEAK=$NG_PEAK
DOG_FINE=$DOG_FINE
DOG_COARSE=$DOG_COARSE
DOG_THRESHOLDS=$DOG_THRESHOLDS
SCORE_THRESHOLD=$SCORE_THRESHOLD
R_MULT=$R_MULT
OUT_NAME=$OUT_NAME
EOF

echo "envfile: $ENVFILE"
cat "$ENVFILE"

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

CMD=(sbatch scripts/slurm/peak_targeting_cpu.sbatch "$ENVFILE")
if [ "$DRY" = "1" ]; then
    echo "DRY: ${CMD[*]}"
else
    "${CMD[@]}"
fi
