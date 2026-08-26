#!/bin/bash
# Submit the progress-signal diagnostic (does the multiscale score guide collapse?).
# ONLY calls sbatch. Configuration goes into ONE timestamped env file passed as a
# POSITIONAL argument -- never `sbatch --export` (requeued+held on this cluster).
#
#   bash scripts/slurm/submit_progress_signal.sh
#   DRY=1 bash scripts/slurm/submit_progress_signal.sh
#   BOXES=set8 N_T=21 bash scripts/slurm/submit_progress_signal.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

BOXES="${BOXES:-set8,set9}"
SCALE_MULTS="${SCALE_MULTS:-1,2,4}"
N_T="${N_T:-11}"
N_CONTROLS="${N_CONTROLS:-24}"
OUT_NAME="${OUT_NAME:-progress_signal}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/prog_sig_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_progress_signal.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
BOXES=$BOXES
SCALE_MULTS=$SCALE_MULTS
N_T=$N_T
N_CONTROLS=$N_CONTROLS
OUT_NAME=$OUT_NAME
EOF

echo "envfile: $ENVFILE"
cat "$ENVFILE"

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

CMD=(sbatch scripts/slurm/progress_signal_cpu.sbatch "$ENVFILE")
if [ "$DRY" = "1" ]; then
    echo "DRY: ${CMD[*]}"
else
    "${CMD[@]}"
fi
