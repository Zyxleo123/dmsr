#!/bin/bash
# Write a map2map checkpoint's full 512^3 candidate box (scripts/features/m2m_box_dump.py).
#
#   sbatch scripts/slurm/m2m_box_dump_gpu.sbatch <envfile>
#
# This is the PREREQUISITE for the halo finders, not a result of its own. P(k)
# cannot see subhalos -- they are sub-grid above k = 32 -- so the field report
# ranks arms on power and says nothing about objects. Rockstar and SUBFIND read
# particles, and both want a (6, 512, 512, 512) .npy.
#
# GPU: 512 tiles through the generator under no_grad, so any class fits and the
# 11 GB 2080 Ti are not excluded.
#SBATCH --job-name=m2mbox
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# --mem=8G is MEASURED: jobs 73662/73664/73666 peaked at 3.8-3.9 GiB under
# /usr/bin/time -v. The first version of this file asked 24G, reasoning from the
# field report's measured 16.9 GiB on the same inference path -- but that job
# also holds HR, the frozen box and complex128 FFT scratch, and this one holds
# only LR and the output array. The estimate was 6x OVER, which is not free: an
# unmeasured over-ask strands the job behind nodes with free GPUs and no RAM
# (same lesson as mgtrain-mem-was-9x-overasked). Estimate, run once, then size.
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-m2mbox-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_m2m_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_m2m_common.sh" "$@"
set -euo pipefail

CKPT="${M2M_CKPT:-$M2M_OUT/checkpoint.pt}"
: "${M2M_BOX_BOX:=set9}"
: "${M2M_BOX_OUT:=$DMSR_REWARD_ROOT/m2m/boxes/$(basename "$M2M_OUT")__$(basename "${CKPT%.pt}")__${M2M_BOX_BOX}.npy}"

if [[ ! -r "$CKPT" ]]; then
    echo ">>> MISSING INPUT: $CKPT"
    echo ">>> produced by: sbatch scripts/slurm/m2m_train_gpu.sbatch"
    exit 0
fi

echo "=== dumping $CKPT on $M2M_BOX_BOX"
echo "    -> $M2M_BOX_OUT"
/usr/bin/time -v python scripts/features/m2m_box_dump.py \
    --ckpt "$CKPT" --repo "$M2M_REPO" --model "$M2M_MODEL" \
    --style-a "$M2M_STYLE_A" --box "$M2M_BOX_BOX" \
    --seed "$M2M_REPORT_SEED" --out "$M2M_BOX_OUT"

echo
echo "=== $(date '+%F %T')  done"
echo "    DELETE this box once its catalogs exist -- it is 3.2 GB and"
echo "    regenerable from the checkpoint in ~12 min with a fixed seed."
