#!/bin/bash
# Stage 3 (CPU): is the LR host channel non-null at the LAGRANGIAN sites the
# shell loss needs to move? Joins the LR membership table (halos_*.particles)
# against the HR Lagrangian owner array, and reports coverage against r/R_vir --
# the axis where HR's subhalos peak (1.0-1.5) and the profile is wrong.
#
# CPU: one chunked pass over 1.3e8 owner entries, no model and no halo finder.
#
#   sbatch scripts/slurm/lr_membership_coverage_cpu.sbatch <envfile>
#SBATCH --job-name=lr_memb_cov
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-lr_memb_cov-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${LHF_BOXES:=set9}"
: "${LHF_MATCH_RVIR:=1.0}"
: "${LMC_HR_MIN_HOST_NUM_P:=200}"
: "${LMC_SUB_MIN_NUM_P:=20}"
: "${LMC_CHUNK:=8000000}"
: "${LHF_LR_WORK_ROOT:=$DMSR_REWARD_ROOT/lagrangian_host}"
: "${LHF_OUT_DIR:?LHF_OUT_DIR must be the output directory}"

# The Lagrangian owner array is the input this stage cannot synthesise. Gate on
# the FIRST box rather than crashing, so a missing array reports why.
_FIRST="${LHF_BOXES%%,*}"
require_input \
    "$DMSR_REWARD_ROOT/halos_particles/${_FIRST}__hr__hr/${_FIRST}_hr_owner.npy" \
    "scripts/reward/rockstar_particles.py (HR Lagrangian owner array)"

python -u scripts/features/lr_membership_coverage.py \
    --boxes "$LHF_BOXES" \
    --reward-root "$DMSR_REWARD_ROOT" \
    --lr-work-root "$LHF_LR_WORK_ROOT" \
    --hr-min-host-num-p "$LMC_HR_MIN_HOST_NUM_P" \
    --sub-min-num-p "$LMC_SUB_MIN_NUM_P" \
    --match-rvir "$LHF_MATCH_RVIR" \
    --chunk "$LMC_CHUNK" \
    --out-dir "$LHF_OUT_DIR"

echo "=== membership coverage -> $LHF_OUT_DIR/membership_summary.json"
