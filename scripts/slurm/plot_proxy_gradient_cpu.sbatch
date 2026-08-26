#!/bin/bash
# Redraw the proxy-gradient diagnostic from gradient/proxy_grad_<arm>_<tag>.{npz,
# json}. CPU only -- no model, no GPU -- so the figures are re-renderable cheaply.
#
#   sbatch scripts/slurm/plot_proxy_gradient_cpu.sbatch RUN_NAME=direct_a ARM=c TAG=frozen
#SBATCH --job-name=sd_grad_plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_grad_plot-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${ARM:=c}"
: "${TAG:=frozen}"

require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/gradient/proxy_grad_${ARM}_${TAG}.npz" \
    "scripts/slurm/proxy_gradient_gpu.sbatch"

python -u scripts/reward/plot_proxy_gradient.py \
    --run-name "$RUN_NAME" --arm "$ARM" --tag "$TAG"
