#!/bin/bash
# IC determinism, stage T (CPU, minutes): for each HR object >= 400 p at 0.1-3 R_vir
#   (A) the same matter -- where do its own core particles end up in the dmo-256
#       run and in frozen SR2, and how compact are they? (does LR alone already
#       know WHERE the matter goes?)
#   (B) the same object -- the candidate object built from the same IC patch,
#       wherever it sits: same radius? same direction? (orbit size vs phase)
# Reads the dmo-256 catalog written by icdet_rockstar_cpu.sbatch. Builds two
# 134M-particle KD-trees in turn (HR, then SR2), hence 64G.
#
#   sbatch scripts/slurm/icdet_track_cpu.sbatch <envfile> [VAR=value ...]
#
#SBATCH --job-name=icd_track
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-icd_track-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
source "$_D/_icdet_common.sh"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 64G  time 2:00:00  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
require_input "$ICD_DMO_CATALOG" "scripts/slurm/icdet_rockstar_cpu.sbatch"
require_input "${ICD_HR_CATALOG:-/nonexistent}" "the HR Rockstar catalog (halos/${BOX}__hr__hr)"
require_input "${ICD_SR2_CATALOG:-/nonexistent}" "the frozen SR2 catalog (halos/${BOX}__base__base)"
require_input "$ICD_HR_FIELD" "scripts/preprocess_paired.py"
require_input "${ICD_SR2_FIELD:-/nonexistent}" "scripts/slurm/cache_sr2_base.sbatch BOXES=$BOX"

python -u scripts/features/ic_determinism.py track \
    --hr-catalog "$ICD_HR_CATALOG" --hr-field "$ICD_HR_FIELD" \
    --cand "$ICD_TAG" "$ICD_DMO_CATALOG" "$ICD_DMO_DIR" raw \
    --cand sr2 "$ICD_SR2_CATALOG" "$ICD_SR2_FIELD" catnorm \
    --host-min-np "$ICD_HOST_MIN_NP" --obj-min-np "$ICD_OBJ_MIN_NP" \
    --mass-tol "$ICD_MASS_TOL" --match-rvir "$ICD_MATCH_RVIR" \
    --match-min-mpc "$ICD_MATCH_MIN_MPC" --lag-k "$ICD_LAG_K" --lag-tol "$ICD_LAG_TOL" \
    --radius-tol "$ICD_RADIUS_TOL" --dir-cos "$ICD_DIR_COS" --pool-rmax "$ICD_POOL_RMAX" \
    --out "$ICD_TRACK_JSON"
