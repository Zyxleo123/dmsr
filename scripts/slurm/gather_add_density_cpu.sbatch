#!/bin/bash
# Add one missing density panel to an existing stage-1 figure npz (CPU). Use
# when a panel came out blank because its field was not passed to the data job
# -- rebuilding the whole npz would need every arm's 3.2 GB field, and the
# purged ones are gone.
#
#   sbatch scripts/slurm/gather_add_density_cpu.sbatch <envfile>
#
# Only the host's Lagrangian tiles are read (~1M particles, not 134M), so this
# is minutes and 16G rather than the 64G gfig_data needs for whole-box
# field_to_particles. GF_VERIFY_FIELD re-derives a panel already in the npz and
# refuses to write on a mismatch: the check that the new panel is built like its
# neighbours rather than merely looking plausible.
#SBATCH --job-name=gfig_dens
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-gfig_dens-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${GF_OUT:?GF_OUT must be the stage-1 data stem}"
: "${GF_HOST_ID:=168880}"
: "${GF_RUN_DIR:=}"
: "${GF_ADD_NAME:=hr}"
: "${GF_ADD_FIELD:=${GF_HR_FIELD:-}}"
: "${GF_ADD_TILES:=}"
: "${GF_VERIFY_NAME:=frozen}"
: "${GF_VERIFY_FIELD:=${GF_FROZEN_FIELD:-}}"
: "${GF_ADD_OVERWRITE:=0}"

if [ -z "$GF_ADD_FIELD" ]; then
    gate_failed "no GF_ADD_FIELD (or GF_HR_FIELD) given -- nothing to add"
fi
require_input "$GF_OUT.npz" "scripts/slurm/gather_figures_data_cpu.sbatch"
require_input "$GF_ADD_FIELD" "the paired_catnorm field for this box"

OPT=()
if [ -n "$GF_ADD_TILES" ]; then OPT+=(--tiles "$GF_ADD_TILES"); fi
if [ -n "$GF_RUN_DIR" ]; then OPT+=(--run-dir "$GF_RUN_DIR" --host-id "$GF_HOST_ID"); fi
if [ "$GF_ADD_OVERWRITE" = "1" ]; then OPT+=(--overwrite); fi
if [ -n "$GF_VERIFY_FIELD" ] && [ -e "$GF_VERIFY_FIELD" ]; then
    OPT+=(--verify-name "$GF_VERIFY_NAME" --verify-field "$GF_VERIFY_FIELD")
else
    echo "note: no GF_VERIFY_FIELD on disk -- the new panel will NOT be checked"
    echo "      against an existing one. Pass the frozen field to check it."
fi

python -u scripts/features/gather_holdout_add_density.py \
    --npz "$GF_OUT.npz" \
    --field "$GF_ADD_FIELD" \
    --name "$GF_ADD_NAME" \
    ${OPT[@]+"${OPT[@]}"}
