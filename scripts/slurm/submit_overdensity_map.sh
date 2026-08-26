#!/bin/bash
# Submit the overdensity-slab visualization (projected 1+delta + centre markers).
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster makes sbatch set SLURM_GET_USER_ENV=1 and gets the job requeued+held.
#
#   bash scripts/slurm/submit_overdensity_map.sh
#   DRY=1 bash scripts/slurm/submit_overdensity_map.sh
#   BOXES=set8 SLAB_DZ_MPC=1.5 bash scripts/slurm/submit_overdensity_map.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

BOXES="${BOXES:-set8,set9}"
NG_IMG="${NG_IMG:-512}"
SLAB_Z_MPC="${SLAB_Z_MPC:--1}"
SLAB_DZ_MPC="${SLAB_DZ_MPC:-2.5}"
DOG_NG="${DOG_NG:-512}"
DOG_FINE="${DOG_FINE:-1.5}"
DOG_COARSE="${DOG_COARSE:-8.0}"
DOG_THRESHOLD="${DOG_THRESHOLD:-1.0}"
OUT_NAME="${OUT_NAME:-overdensity_map}"

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/over_map_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOF
# Written by scripts/slurm/submit_overdensity_map.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
BOXES=$BOXES
NG_IMG=$NG_IMG
SLAB_Z_MPC=$SLAB_Z_MPC
SLAB_DZ_MPC=$SLAB_DZ_MPC
DOG_NG=$DOG_NG
DOG_FINE=$DOG_FINE
DOG_COARSE=$DOG_COARSE
DOG_THRESHOLD=$DOG_THRESHOLD
OUT_NAME=$OUT_NAME
EOF

echo "envfile: $ENVFILE"
cat "$ENVFILE"

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch job.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

CMD=(sbatch scripts/slurm/overdensity_map_cpu.sbatch "$ENVFILE")
if [ "$DRY" = "1" ]; then
    echo "DRY: ${CMD[*]}"
else
    JID=$("${CMD[@]}" | awk '{print $NF}')
    echo "submitted over_map job $JID"
    echo
    echo "watch it:   tail -F $REWARD_ROOT/logs/slurm-over_map-$JID.out"
    echo "results in: $REWARD_ROOT/audits/$OUT_NAME/<box>/"
    echo "  - map.npz              (redraw source; rerun render_overdensity_html.py)"
    echo "  - overdensity_<box>.html   (open in a browser; scp it off the cluster)"
fi
