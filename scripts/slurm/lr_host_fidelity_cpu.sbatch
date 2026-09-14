#!/bin/bash
# Stage 1 (CPU): how well can an LR-derived channel carry a host's centre and
# scale? Runs Rockstar on the LR (64^3) box, cross-matches to the HR catalog,
# and prices that against the smooth LR-potential alternative.
#
# CPU because there is no model here: a 262k-particle halo find plus an ASCII
# parse. Submitted rather than run inline because the HR parse is real work the
# login node kills.
#
#   sbatch scripts/slurm/lr_host_fidelity_cpu.sbatch <envfile>
#SBATCH --job-name=lr_host_fid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-lr_host_fid-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${LHF_BOXES:=set9}"
: "${LHF_HR_MIN_NUM_P:=200}"
: "${LHF_MATCH_RVIR:=1.0}"
: "${LHF_POT_NG:=64}"
: "${LHF_POT_SCALES:=0,1.5625,3.125,6.25}"
: "${LHF_RERUN_ROCKSTAR:=0}"
: "${LHF_LR_WORK_ROOT:=$DMSR_REWARD_ROOT/lagrangian_host}"
: "${LHF_OUT_DIR:?LHF_OUT_DIR must be the output directory}"

# The LR Rockstar config is resolution-specific and must never be the frozen HR
# one; fail loudly here rather than produce a quietly wrong catalog.
require_input "$PROJECT/configs/sr2_baseline/rockstar_lr_particles.cfg" \
    "the repository (LR-resolution Rockstar config)"
require_input "$PROJECT/external/rockstar/rockstar" \
    "make -C external/rockstar"

RERUN=()
[[ "$LHF_RERUN_ROCKSTAR" = "1" ]] && RERUN=(--rerun-rockstar)

python -u scripts/features/lr_host_fidelity.py \
    --boxes "$LHF_BOXES" \
    --reward-root "$DMSR_REWARD_ROOT" \
    --lr-work-root "$LHF_LR_WORK_ROOT" \
    --hr-min-num-p "$LHF_HR_MIN_NUM_P" \
    --match-rvir "$LHF_MATCH_RVIR" \
    --pot-ng "$LHF_POT_NG" \
    --pot-scales "$LHF_POT_SCALES" \
    "${RERUN[@]}" \
    --out-dir "$LHF_OUT_DIR"

echo "=== data -> $LHF_OUT_DIR"
ls -la "$LHF_OUT_DIR"
