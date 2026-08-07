#!/bin/bash
# Phase 2.2 (GPU): full-box candidates from the Gaussian residual policy.
#
# One forward pass per tile, no sampler loop, so this is much cheaper than the
# diffusion oracle. The job only generates and saves; halo finding and reward are
# the CPU stage, so a GPU is never held while Rockstar runs.
#
# CKPT is OPTIONAL. Omitting it samples the UNTRAINED REFERENCE policy, which is
# exactly what the support gate needs: the question is whether the reward has
# support before any training, and requiring a checkpoint for that would mean
# writing one out just to read it back.
#
#   sbatch scripts/slurm/gaussian_sample_gpu.sbatch RUN_NAME=gauss_ref_k16
#   sbatch scripts/slurm/gaussian_sample_gpu.sbatch RUN_NAME=gauss_r1_k16 \
#          CKPT=$ZFS/DMSR/dmsr_reward/checkpoints/gaussian_r0/ckpt.pt
#
#SBATCH --job-name=rw_gauss_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_gauss_gen-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${GAUSS_CFG:=configs/reward/gaussian_policy.yaml}"
: "${RUN_NAME:=gauss_ref_k16}"
: "${CKPT:=}"                  # empty = untrained reference policy
: "${BOXES:=}"
: "${SPLIT:=}"
: "${K:=}"
: "${SEED0:=}"
: "${AMPLITUDES:=}"            # empty = sampling.amplitude_curriculum
: "${ALPHA_DISP:=}"            # empty = correction.alpha_disp (the oracle's value)
: "${ALPHA_VEL:=}"
: "${MODE:=}"
: "${SCALES:=$DMSR_REWARD_ROOT/audits/correction_scales/correction_scales.json}"
: "${TILE_CORE:=}"
: "${TILE_MARGIN:=}"
: "${TILE_BATCH:=}"
: "${DTYPE:=float32}"

require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"
# The amplitude bound sets the size of every edit the policy can propose, so an
# uncalibrated one is refused by the script itself. Fail here with the fix
# instead of letting the traceback explain it.
require_input "$SCALES" "scripts/slurm/calibrate_correction_scales_cpu.sbatch"

# Reclaim the reserved-but-unallocated fragmentation that helped OOM jobs
# 25570/25586 (3.6 of 21.4 GB on a 24 GB A5000). Set here rather than in the
# shared preamble: it is a GPU-only knob.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -u scripts/reward/sample_gaussian_candidates.py \
    --config "$REWARD_CFG" --model-config "$GAUSS_CFG" \
    --run-name "$RUN_NAME" \
    ${CKPT:+--checkpoint "$CKPT"} \
    ${BOXES:+--boxes "$BOXES"} ${SPLIT:+--split "$SPLIT"} \
    ${K:+--k "$K"} ${SEED0:+--seed0 "$SEED0"} \
    --base-seed "$BASE_SEED" \
    ${AMPLITUDES:+--amplitudes "$AMPLITUDES"} \
    ${ALPHA_DISP:+--alpha-disp "$ALPHA_DISP"} \
    ${ALPHA_VEL:+--alpha-vel "$ALPHA_VEL"} \
    ${MODE:+--mode "$MODE"} \
    --scales-path "$SCALES" \
    ${TILE_CORE:+--tile-core "$TILE_CORE"} \
    ${TILE_MARGIN:+--tile-margin "$TILE_MARGIN"} \
    ${TILE_BATCH:+--tile-batch "$TILE_BATCH"} \
    --dtype "$DTYPE" --device "$DEVICE"

echo "=== candidates -> $DMSR_REWARD_ROOT/oracle/$RUN_NAME"
echo "=== next: sbatch scripts/slurm/gaussian_score_cpu.sbatch RUN_NAME=$RUN_NAME BASELINES=1"
