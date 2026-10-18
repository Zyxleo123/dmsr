#!/bin/bash
# IC determinism, stage S (CPU, minutes): Rockstar on the dmo-256 box, through the
# same field -> GADGET2 -> Rockstar path and config as HR's catalog. The particle
# mass (8x HR) reaches Rockstar through the snapshot header. Resumable: an
# existing catalog is reused.
#
#   sbatch scripts/slurm/icdet_rockstar_cpu.sbatch <envfile> [VAR=value ...]
#
#SBATCH --job-name=icd_rs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-icd_rs-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
source "$_D/_icdet_common.sh"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 16G  time 1:00:00  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
require_input "$ICD_DMO_DIR/disp.npy" "the multi-resolution suite under $ICD_SIMS"
require_input "$ICD_DMO_DIR/vel.npy" "the multi-resolution suite under $ICD_SIMS"

python -u scripts/features/ic_determinism.py rockstar \
    --dmo-dir "$ICD_DMO_DIR" --out-dir "$ICD_RS_DIR" --tag "$ICD_TAG"
ls -la "$ICD_DMO_CATALOG"
