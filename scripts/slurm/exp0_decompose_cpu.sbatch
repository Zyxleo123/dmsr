#!/bin/bash
# Experiment 0 (CPU): the deliverable table, and the per-tile credit + plan.
#
# Pure arithmetic on cached tile summaries -- no halo finding, no GPU. Produces
#   audits/tile_decomposition/exp0_table.json     direct vs tile-summed, with error
#   audits/tile_decomposition/credit_<box>.json   A_j and the swap predictions
#   audits/tile_decomposition/verify_plan_<box>.json  tiles to check with Rockstar
#
#   sbatch scripts/slurm/exp0_decompose_cpu.sbatch
#   sbatch scripts/slurm/exp0_decompose_cpu.sbatch BOXES="set8 set9" N_VERIFY=12
#
#SBATCH --job-name=exp0_decomp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-exp0_decomp-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8 set9}"
: "${SOURCES:=hr,base}"
: "${N_VERIFY:=12}"
: "${MODEL_BOXES:=}"

require_input "$DMSR_REWARD_ROOT/tile_cache" \
    "scripts/slurm/exp0_rockstar_particles_cpu.sbatch"

BOXES_CSV="${BOXES// /,}"
echo "=== Experiment 0 table: $BOXES_CSV"
python -u scripts/reward/tile_decompose.py \
    --config "$REWARD_CFG" --boxes "$BOXES_CSV" --sources "$SOURCES"

# The credit stage needs a reward model at the TILE geometry. It is fitted from
# every box that has HR tile summaries; with only the two dev boxes it will
# refuse (a box-level covariance from 2 boxes is not a covariance), which is a
# deliberate stop rather than a silent bad fit.
EXTRA=()
[ -n "$MODEL_BOXES" ] && EXTRA+=(--model-boxes "${MODEL_BOXES// /,}")

for BOX in ${BOXES//,/ }; do
    echo "=== per-tile credit: $BOX"
    python -u scripts/reward/tile_loo_credit.py \
        --config "$REWARD_CFG" --mode plan --box "$BOX" \
        --n-verify "$N_VERIFY" "${EXTRA[@]}"
done

echo "=== Experiment 0 tables -> $DMSR_REWARD_ROOT/audits/tile_decomposition"
