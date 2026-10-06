#!/bin/bash
# Score a map2map checkpoint on density P/P_HR, which is the verdict for this
# line (scripts/features/m2m_field_report.py).
#
#   sbatch scripts/slurm/m2m_field_report_gpu.sbatch <envfile>
#
# NOT the training loss. m2m.py logs MSE, and frozen SR2's own high-k r(k) is ~0
# on every channel, so a model that beats SR2 on MSE is most likely suppressing
# small-scale power rather than learning the map. MEASURED on set14: trilinear
# interpolation scores hr_mse 0.241 against SR2's 0.435 and LOSES on density.
# The bar is density P/P_HR: frozen SR2 0.977, trilinear 0.706.
#
# WHY A GPU. One full-box realisation per box -- 512 tiles through the generator
# -- because the reference table was computed on whole boxes; a tile spectrum
# would need a Hann window and would not be comparable to the numbers it is read
# against. Inference under no_grad, so any GPU class fits and the 11 GB
# rtx_2080_ti are not excluded here, unlike the training job.
#
# HOST RAM is the real ask and is why this is 48G: the SR box, the HR box and
# the cached frozen box are 3.2 GB each as 6 x 512^3 float32, plus complex128
# FFT scratch for a 512^3 cube. Same arithmetic as
# sr2_scratch_field_report_gpu.sbatch, which is the twin of this job.
#SBATCH --job-name=m2mfield
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
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2mfield-%j.out

# `set -euo pipefail` HERE as well as in the preamble: a failed source otherwise
# leaves the job on the system python and EXITING 0, which would release every
# afterok dependent behind a stage that did no work.
set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

CKPT="${M2M_CKPT:-$M2M_OUT/checkpoint.pt}"

# Gates are exit-0 skips: a missing upstream artifact is someone else's job not
# having run, and a non-zero exit would strand dependents in
# DependencyNeverSatisfied with no explanation.
if [[ ! -r "$CKPT" ]]; then
    echo ">>> MISSING INPUT: $CKPT"
    echo ">>> produced by: sbatch scripts/slurm/m2m_train_gpu.sbatch"
    echo ">>> A run that has not finished its first epoch has no checkpoint yet."
    echo ">>> exiting 0 so dependents report the same rather than stranding."
    exit 0
fi
if [[ ! -r "$M2M_REPO/m2m.py" ]]; then
    echo ">>> MISSING INPUT: $M2M_REPO/m2m.py  (the generator class lives there)"
    exit 0
fi

echo
echo "=== scoring $CKPT"
echo "    model=$M2M_MODEL  boxes=${M2M_REPORT_BOXES}  seed=${M2M_REPORT_SEED}"
echo "    bar: density P/P_HR -- frozen SR2 0.977, TRILINEAR 0.706"
echo

/usr/bin/time -v python scripts/features/m2m_field_report.py \
    --ckpt "$CKPT" \
    --repo "$M2M_REPO" \
    --model "$M2M_MODEL" \
    --style-a "$M2M_STYLE_A" \
    --boxes "$M2M_REPORT_BOXES" \
    --seed "$M2M_REPORT_SEED"
_rc=$?

echo
echo "=== $(date '+%F %T')  finished rc=$_rc"
echo "    Anything below 0.706 is worse than interpolation. Our best"
echo "    from-scratch run so far reached 0.26."
exit "$_rc"
