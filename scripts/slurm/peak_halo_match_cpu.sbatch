#!/bin/bash
# Peak -> halo conversion (CPU, saved peak records + catalogs only): which of a
# field's density peaks does its own halo finder turn into objects, split by
# Lagrangian compactness and kinematic distinctness.
#
#   sbatch scripts/slurm/peak_halo_match_cpu.sbatch [envfile] PM_FIELDS=... [PM_OUT=...]
#
#SBATCH --job-name=peakmatch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-peakmatch-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 16G  partition cpu"
: "${PM_FIELDS:?PM_FIELDS (name=peaks.npz:rockstar.ascii[:subfind.npz],...) is required}"
: "${PM_OUT:=$DMSR_REWARD_ROOT/placement_forensics/set9_peak_halo_match.json}"
python -u scripts/features/peak_halo_match.py --fields "$PM_FIELDS" --out "$PM_OUT"
