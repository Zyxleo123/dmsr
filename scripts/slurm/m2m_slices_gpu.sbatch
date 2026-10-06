#!/bin/bash
# Dump 2D slices of candidate / frozen SR2 / HR (GPU), then draw them (CPU-side
# script, run here only because the data is already in memory is NOT true --
# see below).
#
#   sbatch scripts/slurm/m2m_slices_gpu.sbatch <envfile>
#
# GPU because it needs one full-box realisation from the generator. It writes
# the SLICES ONLY and does not plot: drawing reads a few MB of npz and belongs
# on the cpu partition (scripts/slurm/m2m_slice_plot_cpu.sbatch), so the figure
# can be restyled without regenerating a 512^3 box.
#
# HOST RAM 48G: SR, HR and frozen boxes are 3.2 GB each as 6 x 512^3 float32.
#SBATCH --job-name=m2mslice
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2mslice-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

CKPT="${M2M_CKPT:-$M2M_OUT/checkpoint.pt}"
if [[ ! -r "$CKPT" ]]; then
    echo ">>> MISSING INPUT: $CKPT"
    echo ">>> produced by: sbatch scripts/slurm/m2m_train_gpu.sbatch"
    exit 0
fi

for BOX in ${M2M_REPORT_BOXES}; do
    echo
    echo "=== $CKPT on $BOX"
    python scripts/features/m2m_slice_dump.py \
        --ckpt "$CKPT" --repo "$M2M_REPO" --model "$M2M_MODEL" \
        --style-a "$M2M_STYLE_A" --box "$BOX" --seed "$M2M_REPORT_SEED" \
        --zoom "${M2M_SLICE_ZOOM:-128}"
done
echo
echo "=== $(date '+%F %T')  done -- now draw with m2m_slice_plot_cpu.sbatch"
