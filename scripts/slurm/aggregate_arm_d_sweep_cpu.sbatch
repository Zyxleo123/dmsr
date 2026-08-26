#!/bin/bash
# Aggregate one arm-D sweep stage: pick the winner, draw the curves.
# CPU only -- it reads the sweep's JSONL rows and nothing else, so the figures
# are redrawable without refitting anything.
#
#   sbatch scripts/slurm/aggregate_arm_d_sweep_cpu.sbatch <envfile> SWEEP_STAGE=loss
#SBATCH --job-name=sd_sweep_agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_sweep_agg-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${SWEEP_ARM:=d}"
: "${SWEEP_STAGE:=loss}"

JSONL="$DMSR_SR2_DIRECT_ROOT/sweeps/arm_${SWEEP_ARM}_${SWEEP_STAGE}.jsonl"
require_input "$JSONL" "scripts/slurm/sweep_arm_d_gpu.sbatch (stage $SWEEP_STAGE)"

python -u scripts/reward/aggregate_arm_d_sweep.py \
    --arm "$SWEEP_ARM" --stage "$SWEEP_STAGE"
