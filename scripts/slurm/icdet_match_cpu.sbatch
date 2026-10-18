#!/bin/bash
# IC determinism, stage M (CPU, <1 h): around every massive HR host, is each HR
# object >= 400 p at 0.1-3 R_vir found in the dmo-256 catalog at the same
# host-relative place (vs an angular-shuffle null), and is it made of the same
# IC patch? Frozen SR2 (LR only) is run through the identical procedure as the
# control. Builds two 134M-particle KD-trees in turn (HR, then SR2), hence 64G.
#
#   sbatch --dependency=afterany:<ripples>,afterok:<rockstar> scripts/slurm/icdet_match_cpu.sbatch <envfile>
#
#SBATCH --job-name=icd_match
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-icd_match-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
source "$_D/_icdet_common.sh"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 64G  time 3:00:00  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
require_input "$ICD_DMO_CATALOG" "scripts/slurm/icdet_rockstar_cpu.sbatch"
require_input "${ICD_HR_CATALOG:-/nonexistent}" "the HR Rockstar catalog (halos/${BOX}__hr__hr)"
require_input "${ICD_SR2_CATALOG:-/nonexistent}" "the frozen SR2 catalog (halos/${BOX}__base__base)"
require_input "$ICD_HR_FIELD" "scripts/preprocess_paired.py"
require_input "${ICD_SR2_FIELD:-/nonexistent}" "scripts/slurm/cache_sr2_base.sbatch BOXES=$BOX"
[ -f "$ICD_RIPPLES_JSON" ] || echo ">>> no ripples gate at $ICD_RIPPLES_JSON (stage R failed?):" \
    "the match still runs, but its verdict cannot say whether the fine ripples are shared"

python -u scripts/features/ic_determinism.py match \
    --hr-catalog "$ICD_HR_CATALOG" --hr-field "$ICD_HR_FIELD" \
    --cand "$ICD_TAG" "$ICD_DMO_CATALOG" "$ICD_DMO_DIR" raw \
    --cand sr2 "$ICD_SR2_CATALOG" "$ICD_SR2_FIELD" catnorm \
    --gate-json "$ICD_RIPPLES_JSON" \
    --host-min-np "$ICD_HOST_MIN_NP" --obj-min-np "$ICD_OBJ_MIN_NP" \
    --mass-tol "$ICD_MASS_TOL" --match-rvir "$ICD_MATCH_RVIR" \
    --match-min-mpc "$ICD_MATCH_MIN_MPC" --n-null "$ICD_N_NULL" \
    --lag-k "$ICD_LAG_K" --lag-tol "$ICD_LAG_TOL" --out "$ICD_MATCH_JSON"
