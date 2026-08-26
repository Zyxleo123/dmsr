#!/bin/bash
# Render the interactive host-feature page from the cached npz.
#
# A pure redraw: it reads only <box>_lagrangian_host.npz, so re-rendering with a
# different host cap or sample size never re-runs Rockstar. CPU, seconds.
#
#   sbatch scripts/slurm/lagrangian_host_render_cpu.sbatch <envfile>
#SBATCH --job-name=lag_view
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-lag_view-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8}"
: "${N_HOSTS:=0}"     # 0 = every host selectable (~240 kB); >0 caps to the N most massive
: "${N_SAMPLE:=2500}"
: "${LR_FIELD:=}"      # empty -> the reward config's lr/<box>.npy; "none" -> skip panel 6

# Gate: without the cached features there is nothing to draw. Say so and exit 0
# -- a non-zero exit here would only strand anything queued behind this job.
MISSING=0
for b in ${BOXES//,/ }; do
    f="$DMSR_REWARD_ROOT/lagrangian_host/$b/${b}_lagrangian_host.npz"
    [ -f "$f" ] || { echo "GATE: no cached features at $f"; MISSING=1; }
done
if [ "$MISSING" = "1" ]; then
    echo "GATE: run scripts/slurm/lagrangian_host_build_cpu.sbatch first."
    exit 0
fi

# Per-tile SR2-vs-HR abundance, when the SR2/HR catalogs exist. Gate rather
# than fail: a box without them should still get a page, minus panel 5.
: "${TILE_ABUNDANCE:=1}"
if [ "$TILE_ABUNDANCE" = "1" ]; then
    for b in ${BOXES//,/ }; do
        if [ -f "$DMSR_REWARD_ROOT/halos_particles/${b}__hr__hr/${b}_hr_tilew.npz" ] \
        && [ -f "$DMSR_REWARD_ROOT/halos_particles/${b}__base__base/${b}_base_tilew.npz" ]; then
            python -u scripts/features/collect_tile_abundance.py --boxes "$b" \
                --min-sub-particles "${MIN_SUB_PARTICLES:-0}"
        else
            echo "GATE: no SR2/HR tile weights for $b; panel 5 will be disabled."
        fi
    done
fi

EXTRA=()
[ -n "$LR_FIELD" ] && EXTRA+=(--lr-field "$LR_FIELD")

python -u scripts/features/render_lagrangian_host_app.py \
    --boxes "$BOXES" --n-hosts "$N_HOSTS" --n-sample "$N_SAMPLE" "${EXTRA[@]}"

echo "=== page: $DMSR_REWARD_ROOT/lagrangian_host/<box>/lagrangian_host_<box>.html"
echo "=== it is self-contained; scp it off the cluster and open it in a browser."
