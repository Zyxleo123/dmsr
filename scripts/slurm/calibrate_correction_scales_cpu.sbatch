#!/bin/bash
# Phase 1.1 (CPU): measure the correction transform's amplitude bounds.
#
# Reads paired HR and the cached SR2 base fields and writes the coarse/fine
# residual scales the bounded action `u = s * tanh(h)` uses. Pure array
# arithmetic over memory-mapped fields -- no GPU, no halo finder.
#
# Calibration only: no residual, crop or training example is written.
#
#   sbatch scripts/slurm/calibrate_correction_scales_cpu.sbatch
#   sbatch scripts/slurm/calibrate_correction_scales_cpu.sbatch BOXES=set0,set1 BOUND_SIGMA=3
#
#SBATCH --job-name=rw_corr_scales
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_corr_scales-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${BOXES:=}"
: "${SPLIT:=train}"
: "${BOUND_SIGMA:=3.0}"
: "${SLAB:=64}"
: "${OUT:=$DMSR_REWARD_ROOT/audits/correction_scales/correction_scales.json}"

require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"
require_input "$DATA/hr" "scripts/preprocess_paired.py"

python -u scripts/reward/calibrate_correction_scales.py \
    --config "$REWARD_CFG" \
    --split "$SPLIT" ${BOXES:+--boxes "$BOXES"} \
    --base-seed "$BASE_SEED" \
    --bound-sigma "$BOUND_SIGMA" \
    --slab "$SLAB" \
    --out "$OUT"

echo "=== correction scales -> $OUT"
echo "=== point configs/reward/gaussian_policy.yaml at it:"
echo "===   correction: { scales_path: $OUT }"
