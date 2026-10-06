#!/bin/bash
# Why does SUBFIND refuse these Rockstar subhalos? (CPU)
#
# Measures compactness and boundness from the PARTICLES for every Rockstar
# subhalo in the 200-2000p band inside the trained host, split by whether
# SUBFIND confirmed it. That band is chosen because HR confirms 86-100% there,
# so a low rate is a statement about the objects and not about the finder.
#
# 64G: field_to_particles materialises positions and velocities for 512^3 in
# float64 before the crop.
#
#   sbatch scripts/slurm/subfind_forensics_cpu.sbatch <envfile>
#SBATCH --job-name=sf_forensic
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/subfind/logs/slurm-sf_forensic-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_subfind_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_subfind_common.sh" "$@"
set -euo pipefail

: "${SF_BOX:=set9}"
: "${SF_NAME:?SF_NAME must name the run dir under SUBFIND_ROOT}"
: "${SF_TAG:?SF_TAG must be the Rockstar tag}"
: "${SF_SOURCE:=cand}"
: "${SF_FIELD:?SF_FIELD must be the .npy the catalogues came from}"
: "${SF_HOST_ID:=168880}"
: "${SF_BAND_MIN:=200}"
: "${SF_BAND_MAX:=2000}"
: "${SF_FOR_OUT:=$SUBFIND_ROOT/forensics/${SF_BOX}__${SF_NAME}}"

use_env pjm

CAT="$SUBFIND_ROOT/$SF_NAME/subfind_catalog.npz"
require_input "$CAT" "sbatch scripts/slurm/subfind_run_cpu.sbatch"
require_input "$SF_FIELD" "scripts/slurm/gather_splice_cpu.sbatch"

python -u scripts/features/subfind_miss_forensics.py \
    --box "$SF_BOX" --tag "$SF_TAG" --source "$SF_SOURCE" \
    --subfind "$CAT" --field "$SF_FIELD" \
    --host-id "$SF_HOST_ID" \
    --min-p "$SF_BAND_MIN" --max-p "$SF_BAND_MAX" \
    --out "$SF_FOR_OUT"
