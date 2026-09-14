#!/bin/bash
# Build the LR-host conditioning field for every box a conditioned member-gather
# run will touch. CPU, minutes per box.
#
# CPU because there is no model here: Rockstar on the 64^3 LR box (262k
# particles) plus one 262k x n_hosts scaled-distance argmin. Submitted rather
# than run inline because a halo finder is exactly the kind of work the login
# node SIGKILLs.
#
# Resumable by construction: a box whose <box>.npz already exists is skipped
# unless CH_OVERWRITE=1, so hitting the time limit means resubmitting, not
# restarting.
#
#   sbatch scripts/slurm/lr_host_channel_cpu.sbatch <envfile>
#SBATCH --job-name=lr_host_chan
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-lr_host_chan-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${CH_BOXES:=set3 set4 set5 set6 set7 set9 set10}"
: "${CH_CHANNELS:=r_over_rvir,log_mhost,valid}"
: "${CH_MODE:=derived}"
: "${CH_MIN_HR_NUM_P:=20000}"
: "${CH_R_MAX_RVIR:=3.0}"
: "${CH_LOG_MHOST_REF:=13.0}"
: "${CH_LOG_MHOST_SCALE:=1.0}"
: "${CH_RERUN_ROCKSTAR:=0}"
: "${CH_OVERWRITE:=0}"
: "${CH_WORK_ROOT:=$DMSR_REWARD_ROOT/lagrangian_host}"
: "${CH_OUT_DIR:=$DMSR_REWARD_ROOT/lr_host_channel}"

# The LR Rockstar config is resolution-specific and must never be the frozen HR
# one; fail loudly here rather than cache a quietly wrong catalog that every
# later training run would then condition on.
require_input "$PROJECT/configs/sr2_baseline/rockstar_lr_particles.cfg" \
    "the repository (LR-resolution Rockstar config)"
require_input "$PROJECT/external/rockstar/rockstar" \
    "make -C external/rockstar"

EXTRA=()
[[ "$CH_RERUN_ROCKSTAR" = "1" ]] && EXTRA+=(--rerun-rockstar)
[[ "$CH_OVERWRITE" = "1" ]] && EXTRA+=(--overwrite)

python -u scripts/features/build_lr_host_channel.py \
    --boxes "$CH_BOXES" \
    --reward-root "$DMSR_REWARD_ROOT" \
    --channels "$CH_CHANNELS" \
    --mode "$CH_MODE" \
    --min-hr-num-p "$CH_MIN_HR_NUM_P" \
    --r-max-rvir "$CH_R_MAX_RVIR" \
    --log-mhost-ref "$CH_LOG_MHOST_REF" \
    --log-mhost-scale "$CH_LOG_MHOST_SCALE" \
    --work-root "$CH_WORK_ROOT" \
    "${EXTRA[@]}" \
    --out-dir "$CH_OUT_DIR"

echo "=== conditioning fields -> $CH_OUT_DIR"
ls -la "$CH_OUT_DIR"
