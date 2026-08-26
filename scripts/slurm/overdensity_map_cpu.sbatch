#!/bin/bash
# Project the frozen-SR2 overdensity in a slab and draw the candidate + real
# subhalo centres on it, as a self-contained HTML page.
#
# CPU only. Reads one frozen-SR2 field (~4 GB in RAM) plus the base/HR catalogs
# and the oracle targets -- all already on disk -- writes map.npz + summary.json,
# then renders overdensity_<box>.html from the npz. No Rockstar, no GPU, no
# training. The DoG peak grids (ng^3 float32) are the extra memory, hence 48G.
#
#   sbatch scripts/slurm/overdensity_map_cpu.sbatch <envfile>
#SBATCH --job-name=over_map
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-over_map-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8,set9}"
: "${NG_IMG:=512}"
: "${SLAB_Z_MPC:=-1}"
: "${SLAB_DZ_MPC:=2.5}"
: "${DOG_NG:=512}"
: "${DOG_FINE:=1.5}"
: "${DOG_COARSE:=8.0}"
: "${DOG_THRESHOLD:=1.0}"
: "${OUT_NAME:=overdensity_map}"

python -u scripts/reward/collect_overdensity_map.py \
    --boxes "$BOXES" --ng-img "$NG_IMG" \
    --slab-z-mpc "$SLAB_Z_MPC" --slab-dz-mpc "$SLAB_DZ_MPC" \
    --dog-ng "$DOG_NG" --dog-fine "$DOG_FINE" --dog-coarse "$DOG_COARSE" \
    --dog-threshold "$DOG_THRESHOLD" --out-name "$OUT_NAME"

python -u scripts/reward/render_overdensity_html.py \
    --boxes "$BOXES" --out-name "$OUT_NAME"

echo "=== npz + html: $DMSR_REWARD_ROOT/audits/$OUT_NAME/<box>/"
echo "=== open overdensity_<box>.html in a browser (copy off the cluster with scp)."
