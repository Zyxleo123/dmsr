#!/bin/bash
# What happened to the hosts the `self` arm "destroyed"? Cross-matches every
# base host to the tuned catalog AND to HR, classifies the fate (SURVIVED /
# SHRUNK / DEMOTED / VANISHED), separates genuine damage from re-parenting
# bookkeeping, and reconciles the aggregate -320.
#
# Reads three SAVED ASCII catalogs (base, hr, arm) -- no field read, no halo
# finder -- so it is minutes. CPU and submitted because parsing the catalogs is
# real work the login node kills.
#
#   sbatch scripts/slurm/host_forensics_cpu.sbatch <envfile>
#SBATCH --job-name=host_forensics
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-host_forensics-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HF_BOX:=set9}"
: "${HF_RUN_DIR:?HF_RUN_DIR must name the arm's holdout export dir}"
: "${HF_ARM_TAG:?HF_ARM_TAG must be the arm's Rockstar tag}"
: "${HF_ARM_NAME:=self}"
: "${HF_DILATE:=2.0}"
: "${HF_OUT:?HF_OUT must be the output stem}"

require_input "$HF_RUN_DIR/export.json" "scripts/features/export_gather_tiles.py"

python -u scripts/features/host_forensics.py \
    --reward-root "$DMSR_REWARD_ROOT" \
    --box "$HF_BOX" \
    --run-dir "$HF_RUN_DIR" \
    --arm-tag "$HF_ARM_TAG" \
    --arm-name "$HF_ARM_NAME" \
    --dilate "$HF_DILATE" \
    --out "$HF_OUT"

echo "=== forensic summary -> ${HF_OUT}.json  (per-host rows -> ${HF_OUT}.jsonl)"
