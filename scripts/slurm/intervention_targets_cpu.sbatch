#!/bin/bash
# Pick the HR subhalos every box's interventions will aim at. Seconds of CPU on
# catalogs the residual line already has -- no fields, no halo finding.
#
# Runs BEFORE the HR candidates are labelled, because the labelling job reads
# halo_ids_<box>.json to decide which member-particle ids to extract from the
# .particles table it is about to delete.
#
#   sbatch scripts/slurm/intervention_targets_cpu.sbatch <envfile>
#SBATCH --job-name=sd_targets
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_targets-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${BOXES:=}"

ARGS=(--config "$DIRECT_CFG" --run-name "$RUN_NAME" --stage targets)
[ -n "$BOXES" ] && ARGS+=(--boxes ${BOXES//,/ })

python -u scripts/reward/build_intervention_masks.py "${ARGS[@]}"
