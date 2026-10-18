#!/bin/bash
# Per-peak properties vs halo-finder conversion (CPU, saved records + catalogs):
# an HR-fitted model of "is this peak a halo", applied to frozen / fine-tuned peaks.
#
#   sbatch scripts/slurm/peak_property_cpu.sbatch [envfile] PP_FIELDS=... [PP_OUT=...]
#
#SBATCH --job-name=peakprop
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-peakprop-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 16G  partition cpu"
: "${PP_FIELDS:?PP_FIELDS (name=peaks.npz:rockstar.ascii,... HR first) is required}"
: "${PP_OUT:=$DMSR_REWARD_ROOT/placement_forensics/set9_peak_properties.json}"
python -u scripts/features/peak_property_analysis.py --fields "$PP_FIELDS" --out "$PP_OUT"
