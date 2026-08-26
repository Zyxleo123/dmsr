#!/bin/bash
# Held-out gather figures: extract data (CPU, reads fields once) --afterok-->
# render (CPU, seconds). The split makes the figures redrawable without
# re-reading a 3.2 GB field.
#
#   bash scripts/slurm/submit_gather_figures.sh
#   GF_HOST_ID=303060 bash scripts/slurm/submit_gather_figures.sh
#   PLOT_ONLY=1 bash scripts/slurm/submit_gather_figures.sh   # re-render only
#   DRY=1 bash scripts/slurm/submit_gather_figures.sh
set -euo pipefail

PROJECT="${PROJECT:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}"
cd "$PROJECT"
ZFS="${ZFS:-/zfsauton/scratch/yixiz}"
REWARD_ROOT="${DMSR_REWARD_ROOT:-$ZFS/DMSR/dmsr_reward}"
DRY="${DRY:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"

GF_BOX="${GF_BOX:-set9}"
GF_HOST_ID="${GF_HOST_ID:-168880}"
SELF_ARM="${SELF_ARM:-all_blocks_self}"
GF_RUN_DIR="${GF_RUN_DIR:-$REWARD_ROOT/member_gather/$SELF_ARM/holdout_$GF_BOX}"
GF_SELF_FIELD="${GF_SELF_FIELD:-$REWARD_ROOT/flow_rockstar/fields/${GF_BOX}__mgho_${SELF_ARM}_${GF_BOX}__seed0.npy}"
GF_FROZEN_FIELD="${GF_FROZEN_FIELD:-$REWARD_ROOT/flow_rockstar/fields/${GF_BOX}__mgho_frozen_${GF_BOX}__seed0.npy}"
# The cosmic-web "before" and the HR density panels. Resolved by the repo's own
# loaders so the figure job needs no hard-coded data paths.
GF_HR_FIELD="${GF_HR_FIELD:-$(cd "$PROJECT" && /zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin/python -c "
import sys; sys.path[:0]=['src','scripts/reward']
from _sr2_direct import load_direct_config, data_root
class A: config='configs/reward/sr2_direct_finetune.yaml'; overrides=[]
print(data_root(load_direct_config(A())) / 'hr' / '${GF_BOX}.npy')" 2>/dev/null || true)}"
GF_BASE_FIELD="${GF_BASE_FIELD:-$(cd "$PROJECT" && /zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin/python -c "
import sys; sys.path.insert(0,'src')
from cosmo_sr.reward.base import find_base_field
p=find_base_field('${GF_BOX}',0); print(p if p else '')" 2>/dev/null || true)}"
GF_OUT="${GF_OUT:-$REWARD_ROOT/member_gather/$SELF_ARM/holdout_$GF_BOX/figures/data_h${GF_HOST_ID}}"
GF_FIG_DIR="${GF_FIG_DIR:-$REWARD_ROOT/member_gather/$SELF_ARM/holdout_$GF_BOX/figures/h${GF_HOST_ID}}"

for f in "$GF_RUN_DIR/tiles.npz" "$GF_SELF_FIELD" "$GF_FROZEN_FIELD"; do
    [ -f "$f" ] || { echo "missing input: $f" >&2; exit 1; }
done
mkdir -p "$REWARD_ROOT/logs" "$REWARD_ROOT/env" "$(dirname "$GF_OUT")" "$GF_FIG_DIR"

ENVFILE="$REWARD_ROOT/env/gather_figures_$(date +%Y%m%d_%H%M%S)_$$.env"
cat > "$ENVFILE" <<EOT
# Written by scripts/slurm/submit_gather_figures.sh at $(date '+%F %T');
# sourced by the job preambles as a positional argument.
PROJECT=$PROJECT
ZFS=$ZFS
DMSR_REWARD_ROOT=$REWARD_ROOT
GF_BOX=$GF_BOX
GF_HOST_ID=$GF_HOST_ID
GF_RUN_DIR=$GF_RUN_DIR
GF_SELF_FIELD=$GF_SELF_FIELD
GF_FROZEN_FIELD=$GF_FROZEN_FIELD
GF_BASE_FIELD=$GF_BASE_FIELD
GF_HR_FIELD=$GF_HR_FIELD
GF_OUT=$GF_OUT
GF_FIG_DIR=$GF_FIG_DIR
EOT
echo "envfile: $ENVFILE"; cat "$ENVFILE"; echo

for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done
sub() { if [ "$DRY" = "1" ]; then echo "DRY: sbatch $*" >&2; echo DRYID
        else sbatch --parsable "$@"; fi; }

DATA=""
if [ "$PLOT_ONLY" != "1" ]; then
    DATA=$(sub scripts/slurm/gather_figures_data_cpu.sbatch "$ENVFILE")
    echo "submitted data -> job $DATA (CPU)"
fi
DEP=(); [ -n "$DATA" ] && DEP=(--dependency=afterok:"$DATA")
PLOT=$(sub "${DEP[@]}" scripts/slurm/gather_figures_plot_cpu.sbatch "$ENVFILE")
echo "submitted plot -> job $PLOT (CPU${DATA:+, afterok:$DATA})"

echo
[ -n "$DATA" ] && echo "watch data: tail -F $REWARD_ROOT/logs/slurm-gfig_data-$DATA.out"
echo "watch plot: tail -F $REWARD_ROOT/logs/slurm-gfig_plot-$PLOT.out"
echo
echo "figures land in: $GF_FIG_DIR/"
echo "  fig1_host_density.png   frozen | tuned | HR, zoomed to R_vir, subhalos marked"
echo "  fig2_mass_function.png  subhalo counts per mass bin within R_vir"
echo "  fig3_local_excess.png   hosts/subs restricted to the 32 spliced tiles vs HR"
echo "  fig4_cosmic_web.png     full-box density slab: SR2 before | tuned | HR"
echo "  fig5_host_scatter.png   hosts>=200p, base vs tuned, lost hosts marked"
