#!/bin/bash
# IC determinism, stage P (CPU, seconds): the figure, from the two JSONs only.
# Redraw any time without recomputing:
#
#   sbatch scripts/slurm/icdet_plot_cpu.sbatch <envfile> [VAR=value ...]
#
#SBATCH --job-name=icd_plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-icd_plot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
source "$_D/_icdet_common.sh"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 4G  time 0:15:00  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
python -u scripts/features/ic_determinism_plot.py \
    --ripples "$ICD_RIPPLES_JSON" --match "$ICD_MATCH_JSON" --out "$ICD_FIG" \
    --track "$ICD_TRACK_JSON" --track-out "$ICD_TRACK_FIG"
