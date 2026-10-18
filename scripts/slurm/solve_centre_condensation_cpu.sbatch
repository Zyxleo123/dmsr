#!/bin/bash
# Did the generator build an object where the soft membership solve put each
# candidate centre c_j? Radius vs HR (own parent) and arm-catalog objects near
# c_j, per radial bin. Reads a saved membership dump + two Rockstar asciis.
#
#   sbatch scripts/slurm/solve_centre_condensation_cpu.sbatch [envfile|VAR=value ...]
#SBATCH --job-name=solve_condense
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-solve_condense-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 16G"

: "${SC_BOX:=set9}"
: "${SC_ARM:=mgho_all_blocks_sm_coarse_soft_set9}"
: "${SC_NPZ:=$DMSR_REWARD_ROOT/member_gather/membership_viz/membership_coarse_lam1.0_set9_20260917_151014_3064864.npz}"
: "${SC_MIN_NUM_P:=50}"
: "${SC_OUT_DIR:=$DMSR_REWARD_ROOT/member_gather/membership_viz/condensation}"

python -u scripts/features/solve_centre_condensation.py \
    --npz "$SC_NPZ" \
    --hr-ascii "$DMSR_REWARD_ROOT/halos/${SC_BOX}__hr__hr/hr_rockstar/halos_0.0.ascii" \
    --arm-ascii "$DMSR_REWARD_ROOT/flow_rockstar/halos/${SC_BOX}__candidate__${SC_ARM}/${SC_ARM}_rockstar/halos_0.0.ascii" \
    --min-num-p "$SC_MIN_NUM_P" \
    --out-dir "$SC_OUT_DIR"
