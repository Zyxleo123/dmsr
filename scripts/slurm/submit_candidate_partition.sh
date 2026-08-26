#!/bin/bash
# Submit the candidate/partitioning diagnostic (Q1 retrieval + Q2 coverage).
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster makes sbatch set SLURM_GET_USER_ENV=1 and gets the job requeued+held.
#
#   bash scripts/slurm/submit_candidate_partition.sh
#   DRY=1 bash scripts/slurm/submit_candidate_partition.sh
#   BOXES=set8 N_CONTROLS=64 bash scripts/slurm/submit_candidate_partition.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

BOXES="${BOXES:-set8,set9}"
R_MULTS="${R_MULTS:-0.5,1,2,4}"
N_CONTROLS="${N_CONTROLS:-32}"
NG_PEAK="${NG_PEAK:-256}"
OUT_NAME="${OUT_NAME:-candidate_partition}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/cand_part_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_candidate_partition.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
BOXES=$BOXES
R_MULTS=$R_MULTS
N_CONTROLS=$N_CONTROLS
NG_PEAK=$NG_PEAK
OUT_NAME=$OUT_NAME
EOF

echo "envfile: $ENVFILE"
cat "$ENVFILE"

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch job.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

CMD=(sbatch scripts/slurm/candidate_partition_cpu.sbatch "$ENVFILE")
if [ "$DRY" = "1" ]; then
    echo "DRY: ${CMD[*]}"
else
    "${CMD[@]}"
fi
