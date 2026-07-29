#!/bin/bash
# Experiment 1 (CPU): select the missing HR subhalos the oracle will try to restore.
#
# Minutes of arithmetic on catalogs that already exist -- no halo finding. It
# must run BEFORE exp0_rockstar_particles_cpu, because it writes the halo id
# list that job uses to extract Lagrangian member sites in the same pass.
#
#   sbatch scripts/slurm/exp1_select_targets_cpu.sbatch
#   sbatch scripts/slurm/exp1_select_targets_cpu.sbatch BOXES="set8 set9" MAX_TARGETS=24
#
#SBATCH --job-name=exp1_targets
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-exp1_targets-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8 set9}"
: "${MAX_TARGETS:=24}"
: "${MIN_SUB_PARTICLES:=50}"
: "${MIN_SEPARATION:=6.0}"
: "${HOST_BINS:=2,3}"

require_input "$DMSR_REWARD_ROOT/halos" "scripts/slurm/hr_catalog_summaries_cpu.sbatch"

python -u scripts/reward/oracle_select_targets.py \
    --config "$REWARD_CFG" \
    --boxes "${BOXES// /,}" \
    --max-targets "$MAX_TARGETS" \
    --min-sub-particles "$MIN_SUB_PARTICLES" \
    --min-separation "$MIN_SEPARATION" \
    --host-bins "$HOST_BINS"

echo "=== targets -> $DMSR_REWARD_ROOT/oracle_hr"
