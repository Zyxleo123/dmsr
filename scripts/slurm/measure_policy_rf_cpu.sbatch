#!/bin/bash
# Phase 2.0 (CPU): measure the Gaussian policy's conditioning receptive field.
#
# Run this BEFORE the GPU sampler. The sampler re-measures and hard-exits on too
# small a margin, but discovering that costs a GPU allocation; this costs a few
# CPU minutes. It is a job rather than a shell one-liner because a converged
# probe needs a box comfortably larger than twice the reach, and at width 48 a
# 192^3 probe allocates several GB per stage -- large enough to OOM a login node.
#
# A probe that is too small does not fail loudly: the perturbation wraps through
# the circular padding and the reported reach saturates at the probe's own
# half-width. The script detects that and refuses to report it as a measurement.
#
#   sbatch scripts/slurm/measure_policy_rf_cpu.sbatch
#   sbatch scripts/slurm/measure_policy_rf_cpu.sbatch SIZES=192,256,320
#
#SBATCH --job-name=rw_policy_rf
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_policy_rf-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${GAUSS_CFG:=configs/reward/gaussian_policy.yaml}"
: "${SIZES:=128,192,256}"

python -u scripts/reward/measure_policy_receptive_field.py \
    --config "$REWARD_CFG" --model-config "$GAUSS_CFG" \
    --sizes "$SIZES" \
    --threads "${SLURM_CPUS_PER_TASK:-16}"

echo "=== set sampling.tile_margin and train.context_margin from min_tile_margin"
echo "=== then: sbatch scripts/slurm/gaussian_sample_gpu.sbatch RUN_NAME=gauss_ref_k16"
