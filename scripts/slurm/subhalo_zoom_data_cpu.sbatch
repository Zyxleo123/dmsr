#!/bin/bash
# Per-subhalo slab panels + contrast ladder (CPU).
#
# 64G because --field-* converts a whole (6, 512^3) field to particles before
# cropping to the host, which is ~13 GB live; the tiles path needs a fraction of
# that but shares the job.
#
#   sbatch scripts/slurm/subhalo_zoom_data_cpu.sbatch <envfile>
#SBATCH --job-name=sz_data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-sz_data-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${SZ_RUN_DIR:?SZ_RUN_DIR must be the holdout export dir}"
: "${SZ_BOX:=set9}"
: "${SZ_TAG:?SZ_TAG must be the Rockstar candidate tag}"
: "${SZ_HOST_ID:=168880}"
: "${SZ_SELECT:=cand}"
: "${SZ_PICK:=spread}"
: "${SZ_N:=9}"
: "${SZ_MIN_P:=50}"
: "${SZ_HALF_XY_RVIR:=4.0}"
: "${SZ_DEPTHS:=1,2,5,10,-1}"
: "${SZ_AXIS:=1}"
: "${SZ_BINS:=96}"
: "${SZ_OUT:?SZ_OUT must be the output stem}"
: "${SZ_FIELD_OUT:=}"
: "${SZ_FIELD_FROZEN:=}"
: "${SZ_FIELD_HR:=}"

require_input "$SZ_RUN_DIR/tiles.npz" "scripts/features/export_gather_tiles.py"

echo "=== per-subhalo zoom: tag=$SZ_TAG host=$SZ_HOST_ID select=$SZ_SELECT"
python -u scripts/features/subhalo_zoom_data.py \
    --run-dir "$SZ_RUN_DIR" \
    --box "$SZ_BOX" --tag "$SZ_TAG" --host-id "$SZ_HOST_ID" \
    --select "$SZ_SELECT" --pick "$SZ_PICK" \
    --n-subhalos "$SZ_N" --min-p "$SZ_MIN_P" \
    --half-xy-rvir "$SZ_HALF_XY_RVIR" \
    --depths-rvir "$SZ_DEPTHS" \
    --axis "$SZ_AXIS" --bins "$SZ_BINS" \
    ${SZ_FIELD_OUT:+--field-out "$SZ_FIELD_OUT"} \
    ${SZ_FIELD_FROZEN:+--field-frozen "$SZ_FIELD_FROZEN"} \
    ${SZ_FIELD_HR:+--field-hr "$SZ_FIELD_HR"} \
    --out "$SZ_OUT"
