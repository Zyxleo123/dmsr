#!/bin/bash
# CPU stage: table + figure from the residual diagnosis rows. No model, no GPU.
#
# Reads only the JSONL the GPU job wrote, so the figure is redrawable after a
# plotting change without resampling anything.
#
#   sbatch scripts/slurm/residual_diagnose_report_cpu.sbatch
#   sbatch scripts/slurm/residual_diagnose_report_cpu.sbatch DIAG_DIR=.../residual_diagnose_tmax095
#
#SBATCH --job-name=rw_res_diag_rep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_res_diag_rep-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${TAG:=}"
: "${DIAG_DIR:=$DMSR_REWARD_ROOT/audits/residual_diagnose${TAG:+_$TAG}}"

# A missing rows file is "the GPU job did not run", not "the diagnosis failed".
# The report script prints that and exits 0 itself; keep the require_input here
# too so the reason is in this job's log rather than only in python's.
require_input "$DIAG_DIR/residual_diagnose.jsonl" \
    "scripts/slurm/residual_diagnose_gpu.sbatch"

python -u scripts/reward/residual_diagnose_report.py \
    --config "$REWARD_CFG" \
    --rows "$DIAG_DIR/residual_diagnose.jsonl" \
    --out "$DIAG_DIR"

echo "=== summary  -> $DIAG_DIR/residual_diagnose_summary.json"
echo "=== figure   -> $DIAG_DIR/residual_diagnose.png"
