#!/bin/bash
# Trace what happened to every base host the `self` arm lost: SHRUNK below the
# floor / DEMOTED to a subhalo / VANISHED, and whether HR agrees it was real.
#
#   bash scripts/slurm/submit_host_forensics.sh
#   HF_ARM_NAME=selfhostguard SELF_ARM=all_blocks_selfhostguard \
#     HF_ARM_TAG=mgho_all_blocks_selfhostguard_set9 \
#     bash scripts/slurm/submit_host_forensics.sh
#   DRY=1 bash scripts/slurm/submit_host_forensics.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

HF_BOX="${HF_BOX:-set9}"
SELF_ARM="${SELF_ARM:-all_blocks_self}"
HF_ARM_NAME="${HF_ARM_NAME:-self}"
HF_RUN_DIR="${HF_RUN_DIR:-$REWARD_ROOT/member_gather/$SELF_ARM/holdout_$HF_BOX}"
HF_ARM_TAG="${HF_ARM_TAG:-mgho_${SELF_ARM}_${HF_BOX}}"
HF_DILATE="${HF_DILATE:-2.0}"
HF_OUT="${HF_OUT:-$HF_RUN_DIR/forensics/host_forensics_${HF_BOX}}"

[ -f "$HF_RUN_DIR/export.json" ] \
    || { echo "missing input: $HF_RUN_DIR/export.json" >&2; exit 1; }
mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$(dirname "$HF_OUT")"

ENVFILE="$REWARD_ROOT/env/host_forensics_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_host_forensics.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
HF_BOX=$HF_BOX
HF_RUN_DIR=$HF_RUN_DIR
HF_ARM_TAG=$HF_ARM_TAG
HF_ARM_NAME=$HF_ARM_NAME
HF_DILATE=$HF_DILATE
HF_OUT=$HF_OUT
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
if [ "$DRY" = "1" ]; then
    echo "DRY: sbatch scripts/slurm/host_forensics_cpu.sbatch $ENVFILE"; exit 0
fi
JID=$(sbatch --parsable scripts/slurm/host_forensics_cpu.sbatch "$ENVFILE")
echo "submitted host_forensics -> job $JID (CPU)"
echo
echo "watch:   tail -F $REWARD_ROOT/logs/slurm-host_forensics-$JID.out"
echo "results: ${HF_OUT}.json   (per-host rows: ${HF_OUT}.jsonl)"
