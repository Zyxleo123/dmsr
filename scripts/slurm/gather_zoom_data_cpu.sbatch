#!/bin/bash
# Rebuild the three host-density panels zoomed to R_vir, at full tile
# resolution, from the export's tiles.npz (CPU).
#
#   sbatch scripts/slurm/gather_zoom_data_cpu.sbatch <envfile>
#
# 24G: tiles.npz holds (32, 6, 64, 64, 64) float32 for each of three arms, and
# np.load decompresses a whole array on access -- ~200 MB each, plus the
# float64 reconstruction of the host's four tiles. Not the 64G gfig_data needs,
# because no 3.2 GB whole-box field is ever opened.
#SBATCH --job-name=gfig_zoom
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-gfig_zoom-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${GF_RUN_DIR:?GF_RUN_DIR must be the holdout export dir}"
: "${GF_OUT:?GF_OUT must be the stage-1 data stem}"
: "${GF_ZOOM_OUT:?GF_ZOOM_OUT must be the zoom data stem}"
: "${GF_HOST_ID:=168880}"
: "${GF_ZOOM_RVIR:=1.15}"
: "${GF_ZOOM_DEPTH_RVIR:=0}"
: "${GF_ZOOM_BINS:=256}"

require_input "$GF_RUN_DIR/tiles.npz" "scripts/features/export_gather_tiles.py"
require_input "$GF_OUT.npz" "scripts/slurm/gather_figures_data_cpu.sbatch"

python -u scripts/features/gather_holdout_zoom_data.py \
    --run-dir "$GF_RUN_DIR" \
    --source-npz "$GF_OUT.npz" \
    --host-id "$GF_HOST_ID" \
    --zoom-rvir "$GF_ZOOM_RVIR" \
    --depth-rvir "$GF_ZOOM_DEPTH_RVIR" \
    --bins "$GF_ZOOM_BINS" \
    --out "$GF_ZOOM_OUT"
