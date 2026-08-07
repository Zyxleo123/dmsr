#!/bin/bash
# Phase 2.3 (CPU): the support gate. Reads scored rows only -- no field, no GPU.
#
# Answers one question before any long training run: does the reward have usable
# support on samples from the untrained/reference policy? Feasible fraction,
# fraction with positive paired dR_occ, occupation and subhalo-count variation,
# tanh saturation, ESS, and whether two reliable bins improve including an upper
# one.
#
# Cheap and re-runnable: change a threshold, re-render, no recomputation.
#
#   sbatch scripts/slurm/gaussian_support_gate_cpu.sbatch RUN_NAME=gauss_ref_k16
#
#SBATCH --job-name=rw_gauss_gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_gauss_gate-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${GAUSS_CFG:=configs/reward/gaussian_policy.yaml}"
: "${RUN_NAME:=gauss_ref_k16}"
: "${REWARD_MODEL:=$DMSR_REWARD_ROOT/reward_model/reward_model.json}"

ORACLE="$DMSR_REWARD_ROOT/oracle/$RUN_NAME"
require_input "$ORACLE/scored" "scripts/slurm/gaussian_score_cpu.sbatch"
require_input "$REWARD_MODEL" "scripts/slurm/fit_reward_model_cpu.sbatch"

python -u scripts/reward/gaussian_support_gate.py \
    --config "$REWARD_CFG" --model-config "$GAUSS_CFG" \
    --run-name "$RUN_NAME" \
    --reward-model "$REWARD_MODEL"

echo "=== support gate -> $ORACLE/support_gate.{json,md}"
echo "=== If it PASSED: build a replay round and train."
echo "===   sbatch scripts/slurm/gaussian_replay_cpu.sbatch RUN_NAME=$RUN_NAME ROUND=round_000"
echo "=== If it FAILED: read next_steps in the JSON. Diffusion does not fix"
echo "===   zero reward variation -- a richer sampler over the same action space"
echo "===   with the same reward has the same support."
