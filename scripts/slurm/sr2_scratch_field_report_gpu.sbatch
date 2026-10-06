#!/bin/bash
# Score a from-scratch SR2 checkpoint on the yardstick SR2 itself was scored on
# (scripts/features/sr2_scratch_field_report.py).
#
#   sbatch scripts/slurm/sr2_scratch_field_report_gpu.sbatch <envfile>
#
# WHY A GPU. The report needs a full-box realisation from the checkpoint --
# 512 tiles through the generator per box -- because the reference table
# (runs/sr2_baseline/field_table) was computed on whole boxes. A tile spectrum
# would need a Hann window and would NOT be comparable to the 0.396 / 0.989 /
# 0.977 it is being read against. The frozen side needs no generation at all:
# it is sliced from the cached SR2 box.
#
# NODE CHOICE, `myfree` at write time: general a6000 (48 GB) were contended and
# legacy v100 were free. Generation peaks at one tile's activations -- the
# shakeout MEASURED 4.2 GB for a training step, and this is inference under
# no_grad -- so any GPU class fits and the 11 GB rtx_2080_ti are NOT excluded
# here, unlike the training job.
#
# HOST RAM is the real ask, and it is why this is 48G: one full-box SR field is
# 6 x 512^3 float32 = 3.2 GB, the HR box is another 3.2 GB read contiguous (not
# memmapped -- the FFT needs it), the frozen box a third, plus complex128 FFT
# scratch of a 512^3 cube (2.1 GB per plan). Boxes are freed between iterations.
#SBATCH --job-name=sr2scr_field
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-sr2scrfield-%j.out

# `set -euo pipefail` HERE as well as in the preamble: a failed source otherwise
# leaves the job on the system python and EXITING 0.
set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_scratch_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_scratch_common.sh" "$@"
set -euo pipefail

: "${FR_CKPT:=}"
: "${FR_BOXES:=set9 set10}"
: "${FR_SEED:=0}"
: "${FR_OUT:=}"

# ---- gates, as exit-0 skips ------------------------------------------------
# A non-zero exit would strand any dependent job in DependencyNeverSatisfied
# with no explanation, so every missing input says why and exits 0.
if [[ -z "${FR_CKPT// }" ]]; then
    echo ">>> FR_CKPT is empty. Nothing to score."
    echo ">>> set it to a tuned_step*.pt (preferred) or latest.pt."
    exit 0
fi
if [[ ! -f "$FR_CKPT" ]]; then
    echo ">>> MISSING INPUT: $FR_CKPT"
    echo ">>> produced by: scripts/slurm/submit_sr2_scratch.sh (MG_GATE_AT_STEPS"
    echo ">>> writes tuned_step*.pt while training continues)."
    echo ">>> exiting 0 so dependents report the same rather than stranding."
    exit 0
fi
# The frozen cached box is the comparison. Without it the job would still run
# and quietly report only the candidate column, which reads like a result.
_missing=0
for _b in $FR_BOXES; do
    if ! compgen -G "$DMSR_REWARD_ROOT/cache/sr2_base/${_b}_seed${FR_SEED}_*.npy" >/dev/null; then
        echo ">>> MISSING INPUT: cached frozen SR2 box for $_b seed $FR_SEED"
        echo ">>> produced by: scripts/reward/cache_sr2_base.py"
        _missing=1
    fi
done
if (( _missing )); then
    echo ">>> every box lacking a frozen field is SKIPPED by the report; the"
    echo ">>> comparison column is the whole point, so check this before reading."
fi
unset _b _missing

echo
echo "=== scoring $FR_CKPT on $FR_BOXES (seed $FR_SEED)"
srun --unbuffered python scripts/features/sr2_scratch_field_report.py \
    --config "$SC_CFG" \
    --ckpt "$FR_CKPT" \
    --boxes "$FR_BOXES" \
    --seed "$FR_SEED" \
    ${FR_OUT:+--out "$FR_OUT"}
_rc=$?

echo
echo "=== $(date '+%F %T')  field report finished rc=$_rc"
echo "    the row that matters is P/P_HR against the frozen column, NOT mse:"
echo "    SR2's own high-k r(k) is ~0, so a LOWER mse than frozen means"
echo "    suppressed small-scale power, not a better map."
exit "$_rc"
