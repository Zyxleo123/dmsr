#!/bin/bash
# Full P(k) curves for a map2map checkpoint, candidate vs frozen SR2 vs HR.
#
#   sbatch scripts/slurm/m2m_power_curves_gpu.sbatch <envfile>
#
# WHY, when m2m_field_report already reports a number: that number is the mean
# of P_cand/P_HR over the TOP THIRD of k bins (k ~ 170-256 for a 512^3 box).
# It said displacement 1.2e-04 and velocity 1.1e-05 while density was 1.14 --
# and the explanation offered for that (a smooth large-scale displacement field
# manufacturing high-k density through nonlinear collapse) is an ARGUMENT. This
# measures it: the curves show WHERE the candidate stops tracking HR, and the
# prediction is a falloff at the LR Nyquist (k = 32, since LR is 64^3).
#
# It also shows P_HR itself, which the ratio hides -- `_highk` clips the
# denominator at 1e-30 and weights every bin equally, so a near-empty HR bin can
# dominate the average. That is worth ruling out before trusting either number.
#
# GPU: one full-box realisation through the generator, as in the field report.
# HOST RAM 48G for the same reason: SR, HR and frozen boxes are 3.2 GB each as
# 6 x 512^3 float32, plus FFT scratch for a 512^3 cube.
#SBATCH --job-name=m2mpow
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# --mem=32G is MEASURED, not estimated. Job 70308 reproduced this job's
# CPU-side work (load HR + frozen, a same-shaped stand-in for the SR box, then
# the same power_spectrum and cic_density calls) under /usr/bin/time -v and
# peaked at **24,755,312 KB = 23.6 GiB**.
#
# The arithmetic that preceded it said 14-16 GB and was ~50% LOW, which would
# have justified --mem=24G -- i.e. 24 GiB, leaving 1.6% headroom over a 23.6 GiB
# peak, almost certainly an OOM. Estimate, then measure, then size.
#
# 32G is the measured peak plus ~35%, which also covers what the probe could not
# see: the real job holds a CUDA context and the generator weights on the host
# side as well. Do not cut below this without re-measuring.
#
# The old 48G was 2x over. An unmeasured over-ask is not free -- it strands the
# job behind nodes with idle GPUs but insufficient RAM (2026-10-01: gpu24 had 3
# GPUs free and 8 GB). Same lesson as `mgtrain-mem-was-9x-overasked`.
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2mpow-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

CKPT="${M2M_CKPT:-$M2M_OUT/checkpoint.pt}"

# Gates are exit-0 skips: a missing upstream artifact is someone else's job not
# having run, and a non-zero exit strands dependents with no explanation.
if [[ ! -r "$CKPT" ]]; then
    echo ">>> MISSING INPUT: $CKPT"
    echo ">>> produced by: sbatch scripts/slurm/m2m_train_gpu.sbatch"
    exit 0
fi
if [[ ! -r "$M2M_REPO/m2m.py" ]]; then
    echo ">>> MISSING INPUT: $M2M_REPO/m2m.py  (the generator class lives there)"
    exit 0
fi

# M2M_CKPT_LIST scores SEVERAL checkpoints in one allocation. Six separate jobs
# would each re-queue for a contended GPU and reload the same HR/frozen boxes;
# one job walks them sequentially in ~9 min each.
for C in ${M2M_CKPT_LIST:-$CKPT}; do
    [[ -r "$C" ]] || { echo ">>> skip (missing): $C"; continue; }
    for BOX in ${M2M_REPORT_BOXES}; do
        echo
        echo "=== $C on $BOX"
        /usr/bin/time -v python scripts/features/m2m_power_curves.py \
            --ckpt "$C" --repo "$M2M_REPO" --model "$M2M_MODEL" \
            --style-a "$M2M_STYLE_A" --box "$BOX" --seed "$M2M_REPORT_SEED"
    done
done

echo
echo "=== $(date '+%F %T')  done"
echo "    curves are in $(dirname "$CKPT")/power_curves_*.npz"
echo "    Replot from those without regenerating a box."
