#!/bin/bash
# IC determinism, stage R (CPU, ~10 min, fields only): do the 256^3 and 512^3 runs
# of the box share the IC ripples finer than the LR grid? r(k) of the Lagrangian
# displacement/velocity on the 256^3 grid, HR vs dmo-256 and HR vs frozen SR2.
# Writes the gate the match stage reads.
#
#   sbatch scripts/slurm/icdet_ripples_cpu.sbatch <envfile> [VAR=value ...]
#
#SBATCH --job-name=icd_rip
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-icd_rip-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
source "$_D/_icdet_common.sh"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 16G  time 1:00:00  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
require_input "$ICD_HR_FIELD" "scripts/preprocess_paired.py"
require_input "${ICD_SR2_FIELD:-/nonexistent}" "scripts/slurm/cache_sr2_base.sbatch BOXES=$BOX"
require_input "$ICD_DMO_DIR/disp.npy" "the multi-resolution suite under $ICD_SIMS"
require_input "$ICD_DMO_DIR/vel.npy" "the multi-resolution suite under $ICD_SIMS"

python -u scripts/features/ic_determinism.py ripples \
    --hr-field "$ICD_HR_FIELD" --sr2-field "$ICD_SR2_FIELD" --dmo-dir "$ICD_DMO_DIR" \
    --band "$ICD_GATE_LO" "$ICD_GATE_HI" --gate-min "$ICD_GATE_MIN" \
    --gate-margin "$ICD_GATE_MARGIN" --out "$ICD_RIPPLES_JSON"
