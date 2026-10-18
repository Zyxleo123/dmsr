#!/bin/bash
# Frame check (CPU, catalogs only): which axis permutation/reflection + shift
# maps a candidate box's hosts onto HR's? Frozen SR2 is the control (identity).
#
#   sbatch scripts/slurm/frame_offset_check_cpu.sbatch <envfile> [VAR=value ...]
#
#SBATCH --job-name=framechk
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-framechk-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 16G  partition cpu"
: "${BOX:=set9}"
: "${FC_CANDS:?FC_CANDS (name=halos.ascii,...) is required}"
: "${FC_OUT:=$DMSR_REWARD_ROOT/placement_forensics/${BOX}_frame_check.json}"
HR_CAT=$(ls "$DMSR_REWARD_ROOT"/halos/"${BOX}"__hr__hr/*_rockstar/halos_*.ascii 2>/dev/null | head -1 || true)
require_input "${HR_CAT:-/nonexistent}" "the frozen HR Rockstar catalog"

python -u scripts/features/frame_offset_check.py --hr-catalog "$HR_CAT" \
    --cand "$FC_CANDS" --out "$FC_OUT"
