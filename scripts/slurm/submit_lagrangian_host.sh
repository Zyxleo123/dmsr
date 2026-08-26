#!/bin/bash
# Submit the LR-Rockstar Lagrangian host feature build, then the viewer page.
#
# This submitter ONLY calls sbatch. Configuration goes into ONE timestamped env
# file passed as a POSITIONAL argument -- never `sbatch --export`, which on this
# cluster makes sbatch set SLURM_GET_USER_ENV=1 and gets the job requeued+held.
#
#   bash scripts/slurm/submit_lagrangian_host.sh
#   BOXES=set8,set9 bash scripts/slurm/submit_lagrangian_host.sh
#   RENDER_ONLY=1 N_HOSTS=40 bash scripts/slurm/submit_lagrangian_host.sh  # cap to 40
#   SUBTILES=0 bash scripts/slurm/submit_lagrangian_host.sh   # skip panel 7's join
#   DEFICIT=0 bash scripts/slurm/submit_lagrangian_host.sh    # skip the deficit tables
#   DRY=1 bash scripts/slurm/submit_lagrangian_host.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

BOXES="${BOXES:-set8}"
REDSHIFT="${REDSHIFT:-0.0}"
RERUN_ROCKSTAR="${RERUN_ROCKSTAR:-0}"
N_HOSTS="${N_HOSTS:-0}"   # 0 = all hosts
N_SAMPLE="${N_SAMPLE:-2500}"
LR_FIELD="${LR_FIELD:-}"
TILE_ABUNDANCE="${TILE_ABUNDANCE:-1}"
MIN_SUB_PARTICLES="${MIN_SUB_PARTICLES:-0}"
RENDER_ONLY="${RENDER_ONLY:-0}"
SUBTILES="${SUBTILES:-1}"              # per-host HR/SR2 subhalo counts (panel 7)
SUBTILES_N_HOSTS="${SUBTILES_N_HOSTS:-0}"
DEFICIT="${DEFICIT:-1}"                # docs/sr2_subhalo_deficit.md's tables

mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env"
ENVFILE="$REWARD_ROOT/env/lag_host_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_lagrangian_host.sh at $(date '+%F %T');
# sourced by the job preamble as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
BOXES=$BOXES
REDSHIFT=$REDSHIFT
RERUN_ROCKSTAR=$RERUN_ROCKSTAR
N_HOSTS=$N_HOSTS
N_SAMPLE=$N_SAMPLE
LR_FIELD=$LR_FIELD
TILE_ABUNDANCE=$TILE_ABUNDANCE
MIN_SUB_PARTICLES=$MIN_SUB_PARTICLES
SUBTILES_N_HOSTS=$SUBTILES_N_HOSTS
EOT

echo "envfile: $ENVFILE"
cat "$ENVFILE"
echo

# Scrub SLURM_* so a submission from inside an allocation cannot leak an
# --export-like environment onto the batch job.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

BUILD=(sbatch scripts/slurm/lagrangian_host_build_cpu.sbatch "$ENVFILE")
if [ "$DRY" = "1" ]; then
    echo "DRY: ${BUILD[*]}"
    echo "DRY: sbatch [--dependency=afterok:<build>] scripts/slurm/host_subhalo_tiles_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch [--dependency=afterok:<build>] scripts/slurm/subhalo_deficit_cpu.sbatch $ENVFILE"
    echo "DRY: sbatch [--dependency=afterok:<subs>] scripts/slurm/lagrangian_host_render_cpu.sbatch $ENVFILE"
    exit 0
fi

# build -> per-host subhalo join -> render. The join is a separate job because
# it reads the 512 MB owner arrays the render job has no business loading, and
# it gates itself to exit 0 when they are absent so the render still happens.
BID=""
if [ "$RENDER_ONLY" != "1" ]; then
    BID=$("${BUILD[@]}" | awk '{print $NF}')
    echo "submitted lag_host (build) job $BID"
fi

SID=""
if [ "$SUBTILES" = "1" ]; then
    DEP=()
    if [ -n "$BID" ]; then DEP=(--dependency=afterok:"$BID"); fi
    SID=$(sbatch "${DEP[@]}" scripts/slurm/host_subhalo_tiles_cpu.sbatch \
          "$ENVFILE" | awk '{print $NF}')
    echo "submitted lag_subs (per-host subhalo join) job $SID"
fi

# Sibling of the join, not behind it: they read different inputs and neither
# feeds the other, so queueing them concurrently costs nothing.
FID=""
if [ "$DEFICIT" = "1" ]; then
    DEP=()
    if [ -n "$BID" ]; then DEP=(--dependency=afterok:"$BID"); fi
    FID=$(sbatch "${DEP[@]}" scripts/slurm/subhalo_deficit_cpu.sbatch \
          "$ENVFILE" | awk '{print $NF}')
    echo "submitted sub_defic (deficit tables) job $FID"
fi

DEP=()
if [ -n "$SID" ]; then
    DEP=(--dependency=afterok:"$SID")
elif [ -n "$BID" ]; then
    DEP=(--dependency=afterok:"$BID")
fi
RID=$(sbatch "${DEP[@]}" scripts/slurm/lagrangian_host_render_cpu.sbatch \
      "$ENVFILE" | awk '{print $NF}')
echo "submitted lag_view (render) job $RID"
echo
if [ -n "$BID" ]; then
    echo "watch build:  tail -F $REWARD_ROOT/logs/slurm-lag_host-$BID.out"
fi
if [ -n "$SID" ]; then
    echo "watch join:   tail -F $REWARD_ROOT/logs/slurm-lag_subs-$SID.out"
fi
if [ -n "$FID" ]; then
    echo "watch deficit:tail -F $REWARD_ROOT/logs/slurm-sub_defic-$FID.out"
fi
echo "watch render: tail -F $REWARD_ROOT/logs/slurm-lag_view-$RID.out"
echo
echo "results in: $REWARD_ROOT/lagrangian_host/<box>/"
for b in ${BOXES//,/ }; do
    echo "  $b:"
    echo "    ${b}_lagrangian_host.npz    features at 64^3 + host table (redraw source)"
    echo "    ${b}_lagrangian_host.json   normalisation report"
    echo "    ${b}_lr_rockstar/           LR catalog + member-particle table"
    echo "    ${b}_host_subhalo_tiles.json per-host per-tile HR/SR2 subhalo counts"
    echo "    ${b}_subhalo_deficit.json    the deficit tables in docs/sr2_subhalo_deficit.md"
    echo "    lagrangian_host_${b}.html   the viewer -- scp it off and open it"
done
