#!/bin/bash
# Channel-attribution control for the substructure field collapse (see
# scripts/reward/diagnose_substructure_field.py). Rockstars four crop variants
# around the cluster host to isolate whether displacement or velocity d does the
# damage. CPU (halo finder), reads only saved fields -- cheap, minutes.
#
#   sbatch scripts/slurm/diagnose_substructure_cpu.sbatch BOX=set8 TAG=substructure_set8
#SBATCH --job-name=sub_diag
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/flow_rockstar/logs/slurm-sub_diag-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOX:=set8}"
: "${TAG:=substructure_set8}"
: "${SEED:=0}"
: "${HOST_ID:=717}"
: "${REGION_SITES:=192}"

require_input "external/rockstar/rockstar" "make -C external/rockstar"
require_input "$DMSR_REWARD_ROOT/flow_rockstar/fields/${BOX}__${TAG}__seed${SEED}.npy" \
    "scripts/slurm/sample_substructure_gpu.sbatch"

python -u scripts/reward/diagnose_substructure_field.py \
    --box "$BOX" --tag "$TAG" --seed "$SEED" \
    --host-id "$HOST_ID" --region-sites "$REGION_SITES"
