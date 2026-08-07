#!/bin/bash
# Phase 2.2 (GPU, diagnostic): how long is ONE reward-only training step?
#
# Times the exact step sequence of train_gaussian_reward.py -- behaviour forward
# (no_grad) + reference forward (no_grad) + policy forward/backward + AdamW --
# on synthetic tensors of the configured crop shape, so the per-step cost and
# the 20k-step wall clock are known before a replay buffer exists. Conv cost
# depends on shape, not contents; replay crop I/O is NOT included.
#
# Walks a descending ladder of shapes -- the margin is fixed by the measured
# receptive field, so a smaller core is the only way to shrink the tile -- and
# runs EACH ONE IN ITS OWN PROCESS. Job 25850 shows why: a shape that OOMs
# leaves its graph reachable from the traceback, and the next shape in the same
# process starts against an allocator that is already 46 of 47 GiB full, so
# every later row reports a failure that is an artifact of the earlier one.
#
#   sbatch scripts/slurm/bench_gaussian_step_gpu.sbatch
#   sbatch scripts/slurm/bench_gaussian_step_gpu.sbatch SHAPES=128:56:2 STEPS=20
#
#SBATCH --job-name=rw_gauss_bench
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_gauss_bench-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${GAUSS_CFG:=configs/reward/gaussian_policy.yaml}"
# Margin stays at the MEASURED 56 in every rung: below it, valid-core tiling
# leaks padding into the written core, so a cheaper margin is not a cheaper
# configuration, it is a different (wrong) one. Only the core moves.
: "${SHAPES:=128:56:2,128:56:1,96:56:1,64:56:1,32:56:1}"
: "${STEPS:=10}"
: "${WARMUP:=3}"
: "${OUT:=$DMSR_REWARD_ROOT/audits/gaussian_step_bench/bench_gaussian_step.json}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"     # --append merges across processes; start from an empty table

for shape in ${SHAPES//,/ }; do
    # || true: an OOM is a RESULT here, recorded by the script itself. One shape
    # failing must not abandon the ladder -- the point is to find where it stops.
    python -u scripts/reward/bench_gaussian_step.py \
        --config "$REWARD_CFG" --model-config "$GAUSS_CFG" \
        --shapes "$shape" --steps "$STEPS" --warmup "$WARMUP" \
        --out "$OUT" --append --device "$DEVICE" || \
        echo "!!! shape $shape died outside the script's own OOM handler"
done

echo "=== per-step timings -> $OUT"
