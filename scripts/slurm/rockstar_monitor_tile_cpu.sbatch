#!/bin/bash
# REAL halo finder on tile-overfit snapshots -- one array task per snapshot. Each
# splices its snapshot into the frozen box and re-runs Rockstar on the complete
# periodic box (an isolated tile has no host context), writing one iter_<k>.json.
# Out-of-range array indices exit 0, so the submitter can over-provision tasks
# without knowing the exact snapshot count. Full-box Rockstar -> same resources
# as sd_label. Particle tables are deleted in-job (disk quota).
#
#   sbatch --array=0-20 scripts/slurm/rockstar_monitor_tile_cpu.sbatch RUN_NAME=direct_a TAG=set0_t486_c
#SBATCH --job-name=sd_tile_rock
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_tile_rock-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${TAG:?set TAG=<box>_t<tile>_<arm>, e.g. TAG=set0_t486_c}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"
require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/tile_overfit/tile_overfit_${TAG}.json" \
    "scripts/slurm/overfit_tile_cpu.sbatch (with CKPT_EVERY>0)"

python -u scripts/reward/rockstar_monitor_tile.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME" --tag "$TAG" --index "$IDX"
