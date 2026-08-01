#!/bin/bash
# Gate A criterion 6 (GPU): tile-size independence of the written core.
#
# The training job cannot answer this -- it never tiles -- so without this job
# criterion 6 is "not_evaluated", gate_a.json says passed: false, and
# sample_reward_oracle.sbatch correctly refuses to sample. That was the state of
# the chain before this script existed.
#
# GPU only: two full-box sampling passes. The verdict itself is decided by the
# CPU job behind this one (gate_a_check_cpu.sbatch), which only reads JSON.
#
#   sbatch scripts/slurm/gate_a_tile_scan_gpu.sbatch
#   sbatch scripts/slurm/gate_a_tile_scan_gpu.sbatch TILE_CORES=256,128 SCAN_STEPS=20
#
#SBATCH --job-name=rw_tile_scan
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_tile_scan-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${PRIOR_CKPT:=$DMSR_REWARD_ROOT/checkpoints/residual_prior/ckpt_best.pt}"
: "${RUN_DIR:=$(dirname "$PRIOR_CKPT")}"
: "${SCAN_BOX:=}"                # empty = the first val box
# Two cores that both divide ng_hr=512. 128 is the production core
# (sample_reward_oracle.sbatch); 64 is the cheap comparison, tile 160^3.
: "${TILE_CORES:=128,64}"
: "${TILE_MARGIN:=48}"
: "${TILE_BATCH:=1}"
# The tiling identity holds at any step count and each step costs two full-box
# passes, so the scan runs short by default. Raise it only to reproduce a
# borderline number.
: "${SCAN_STEPS:=5}"
: "${SCAN_SEED:=0}"
: "${TILE_TOLERANCE:=1e-4}"
: "${TILE_SCAN_JSON:=$RUN_DIR/tile_scan.json}"

require_input "$PRIOR_CKPT" "scripts/slurm/train_residual_prior.sbatch"
require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"

python -u scripts/reward/tile_scan.py \
    --config "$REWARD_CFG" --model-config "$MODEL_CFG" \
    --checkpoint "$PRIOR_CKPT" \
    ${SCAN_BOX:+--box "$SCAN_BOX"} \
    --seed "$SCAN_SEED" --base-seed "$BASE_SEED" \
    --tile-cores "$TILE_CORES" \
    --tile-margin "$TILE_MARGIN" \
    --tile-batch "$TILE_BATCH" \
    --n-steps "$SCAN_STEPS" \
    --tolerance "$TILE_TOLERANCE" \
    --out "$TILE_SCAN_JSON" \
    --device "$DEVICE"

echo "=== tile scan -> $TILE_SCAN_JSON"
echo "=== next: sbatch scripts/slurm/gate_a_check_cpu.sbatch"
