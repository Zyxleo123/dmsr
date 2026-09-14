#!/bin/bash
# Fill in a blank density panel on the held-out gather figures, then re-render.
#
# fig1_host_density draws each arm's projected density; the HR panel came out
# white because the stage-1 job ran without --hr-field. This selects the host's
# Lagrangian tiles out of the HR field -- the same material, the same window and
# resolution as the panels already in the npz -- and merges it back (CPU,
# minutes) --afterok--> re-renders the figures (CPU, seconds).
#
# The job first RE-DERIVES the frozen panel from the frozen field (which still
# exists) and compares it with the stored one. If that does not reproduce
# exactly, the new panel is not built like its neighbours and nothing is
# written -- a wrong panel would still render a convincing picture.
#
#   bash scripts/slurm/submit_gather_add_density.sh
#   GF_BOX=set10 bash scripts/slurm/submit_gather_add_density.sh
#   GF_ADD_NAME=base GF_ADD_FIELD=/path/base.npy bash scripts/slurm/submit_gather_add_density.sh
#   GF_ADD_OVERWRITE=1 bash scripts/slurm/submit_gather_add_density.sh  # recompute
#   DRY=1 bash scripts/slurm/submit_gather_add_density.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"

GF_BOX="${GF_BOX:-set9}"
GF_HOST_ID="${GF_HOST_ID:-168880}"
SELF_ARM="${SELF_ARM:-all_blocks_self}"
GF_RUN_DIR="${GF_RUN_DIR:-$REWARD_ROOT/member_gather/$SELF_ARM/holdout_$GF_BOX}"
GF_OUT="${GF_OUT:-$GF_RUN_DIR/figures/data_h${GF_HOST_ID}}"
GF_FIG_DIR="${GF_FIG_DIR:-$GF_RUN_DIR/figures/h${GF_HOST_ID}}"

GF_ADD_NAME="${GF_ADD_NAME:-hr}"
# Resolved by the repo's own loader so the job needs no hard-coded data path.
GF_HR_FIELD="${GF_HR_FIELD:-$(cd "$PROJECT" && /zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin/python -c "
import sys; sys.path[:0]=['src','scripts/reward']
from _sr2_direct import load_direct_config, data_root
class A: config='configs/reward/sr2_direct_finetune.yaml'; overrides=[]
print(data_root(load_direct_config(A())) / 'hr' / '${GF_BOX}.npy')" 2>/dev/null || true)}"
GF_ADD_FIELD="${GF_ADD_FIELD:-$GF_HR_FIELD}"
GF_ADD_TILES="${GF_ADD_TILES:-}"      # empty: the host's tiles from export.json
GF_ADD_OVERWRITE="${GF_ADD_OVERWRITE:-0}"
# The panel re-derived as the correctness check. frozen's field survives; the
# tuned one was purged, which is why the npz is being patched instead of rebuilt.
GF_VERIFY_NAME="${GF_VERIFY_NAME:-frozen}"
GF_VERIFY_FIELD="${GF_VERIFY_FIELD:-$REWARD_ROOT/flow_rockstar/fields/${GF_BOX}__mgho_frozen_${GF_BOX}__seed0.npy}"

for f in "$GF_OUT.npz" "$GF_ADD_FIELD" "$GF_RUN_DIR/export.json"; do
    [ -f "$f" ] || { echo "missing input: $f" >&2; exit 1; }
done
[ -f "$GF_VERIFY_FIELD" ] || echo "warning: no $GF_VERIFY_NAME field at " \
    "$GF_VERIFY_FIELD -- the new panel will not be checked" >&2
mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$GF_FIG_DIR"

ENVFILE="$REWARD_ROOT/env/gather_add_density_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_gather_add_density.sh at $(date '+%F %T');
# sourced by the job preambles as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
GF_BOX=$GF_BOX
GF_HOST_ID=$GF_HOST_ID
GF_RUN_DIR=$GF_RUN_DIR
GF_OUT=$GF_OUT
GF_FIG_DIR=$GF_FIG_DIR
GF_ADD_NAME=$GF_ADD_NAME
GF_ADD_FIELD=$GF_ADD_FIELD
GF_ADD_TILES=$GF_ADD_TILES
GF_ADD_OVERWRITE=$GF_ADD_OVERWRITE
GF_VERIFY_NAME=$GF_VERIFY_NAME
GF_VERIFY_FIELD=$GF_VERIFY_FIELD
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
sub() { if [ "$DRY" = "1" ]; then echo "DRY: sbatch $*" >&2; echo DRYID
        else sbatch --parsable "$@"; fi; }

DENS=$(sub scripts/slurm/gather_add_density_cpu.sbatch "$ENVFILE")
echo "submitted add-density -> job $DENS (CPU)"
PLOT=$(sub --dependency=afterok:"$DENS" scripts/slurm/gather_figures_plot_cpu.sbatch "$ENVFILE")
echo "submitted plot -> job $PLOT (CPU, afterok:$DENS)"

echo
echo "watch add-density: tail -F $REWARD_ROOT/logs/slurm-gfig_dens-$DENS.out"
echo "watch plot:        tail -F $REWARD_ROOT/logs/slurm-gfig_plot-$PLOT.out"
echo
echo "data file:  $GF_OUT.npz"
echo "figures in: $GF_FIG_DIR/  (fig1_host_density.png is the one with the HR panel)"
