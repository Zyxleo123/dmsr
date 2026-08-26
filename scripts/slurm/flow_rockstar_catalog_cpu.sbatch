#!/bin/bash
# Flow -> Rockstar, stage 2 (CPU): full-box halo finding + chunk attribution on
# the sampled flow field, written in the exact frozen catalog_cache format that
# produced set15__hr__hr and set15__base__base -- so the flow catalog is
# apples-to-apples with both references.
#
# This is just scripts/reward/catalog_summaries.py with --source candidate; the
# Rockstar config is the frozen one (freeze.yaml), identical to HR/base.
#
#   sbatch scripts/slurm/flow_rockstar_catalog_cpu.sbatch \
#       TAG=flow_cascade \
#       FIELD_OUT=.../flow_rockstar/fields/set15__flow_cascade__seed0.npy
#SBATCH --job-name=flow_rs_cat
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/flow_rockstar/logs/slurm-flow_rs_cat-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${TAG:=flow_cascade}"
: "${BOX:=set15}"
: "${FLOW_RS_ROOT:=$DMSR_REWARD_ROOT/flow_rockstar}"
: "${FIELD_OUT:=$FLOW_RS_ROOT/fields/${BOX}__${TAG}__seed${SEED}.npy}"

require_input "external/rockstar/rockstar" "make -C external/rockstar"
require_input "$FIELD_OUT" "scripts/slurm/flow_rockstar_sample_gpu.sbatch"

echo "=== catalog: box=$BOX tag=$TAG field=$FIELD_OUT"
python -u scripts/reward/catalog_summaries.py \
    --config "$REWARD_CFG" \
    --box "$BOX" --source candidate --tag "$TAG" \
    --field "$FIELD_OUT" \
    --work "$FLOW_RS_ROOT/halos/${BOX}__candidate__${TAG}"

echo "=== summary -> $DMSR_REWARD_ROOT/catalog_cache/${BOX}__candidate__${TAG}.json"
