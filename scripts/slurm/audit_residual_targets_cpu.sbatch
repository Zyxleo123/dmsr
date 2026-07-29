#!/bin/bash
# Stage 2 (CPU): audit the paired residual targets before any GPU training.
#
# This is the cheapest place to catch a normalization or periodic-coordinate
# mistake. It also emits sigma_res.json, the per-channel whitening scale the
# diffusion model needs; training without it falls back to a few-batch estimate.
#
#   sbatch scripts/slurm/audit_residual_targets_cpu.sbatch
#   sbatch scripts/slurm/audit_residual_targets_cpu.sbatch BOXES=set0,set1,set2
#
#SBATCH --job-name=rw_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_audit-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${BOXES:=}"

require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"

python -u scripts/reward/audit_residual_targets.py \
    --config "$REWARD_CFG" \
    --base-seed "$BASE_SEED" \
    ${BOXES:+--boxes "$BOXES"}

echo "=== audit -> $DMSR_REWARD_ROOT/audits/residual_target_audit"
