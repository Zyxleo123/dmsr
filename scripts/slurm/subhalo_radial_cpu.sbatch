#!/bin/bash
# Subhalo mass vs. distance from its host centre, for HR and the tuned `self`
# arm. Reads the SAVED ASCII catalogs (no field read, no halo finder), writes
# redrawable per-subhalo arrays, then renders the figures. CPU and submitted
# because parsing the catalogs is real work the login node kills; it is minutes.
#
#   sbatch scripts/slurm/subhalo_radial_cpu.sbatch <envfile>
#SBATCH --job-name=subhalo_radial
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-subhalo_radial-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${SR_BOX:=set9}"
: "${SR_KINDS:=hr,self}"
: "${SR_ARM_TAGS:=self=mgho_all_blocks_self_set9}"
: "${SR_RUN_DIR:=}"          # optional: restrict to an export.json tile region
: "${SR_DILATE:=2.0}"
: "${SR_OUT_DIR:?SR_OUT_DIR must be the output directory}"

RUN_DIR_ARG=()
[[ -n "$SR_RUN_DIR" ]] && RUN_DIR_ARG=(--run-dir "$SR_RUN_DIR")

python -u scripts/features/subhalo_radial.py \
    --reward-root "$DMSR_REWARD_ROOT" \
    --box "$SR_BOX" \
    --kinds "$SR_KINDS" \
    --arm-tags "$SR_ARM_TAGS" \
    --dilate "$SR_DILATE" \
    "${RUN_DIR_ARG[@]}" \
    --out-dir "$SR_OUT_DIR"

python -u scripts/features/subhalo_radial_plot.py \
    --out-dir "$SR_OUT_DIR" \
    --kinds "$SR_KINDS"

echo "=== data + figures -> $SR_OUT_DIR"
ls -la "$SR_OUT_DIR"
