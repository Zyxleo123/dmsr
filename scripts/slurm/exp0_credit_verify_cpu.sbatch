#!/bin/bash
# Experiment 0 (CPU array): does predicted per-tile credit match a real tile swap?
#
# One array task per planned tile. Each splices HR content into that ONE
# Lagrangian tile of the frozen SR2 box, re-runs the halo finder on the complete
# periodic box, and caches the resulting tile summaries. `tile_loo_credit.py
# --mode verify` then compares the measured reward change with the prediction
# made from cached statistics alone.
#
# This is the only part of Experiment 0 that costs halo-finder time, and it is
# the part that bounds the cross-tile interaction that per-tile credit assumes
# away. 8-16 tiles is enough for that bound; more is waste.
#
#   sbatch --array=0-11 scripts/slurm/exp0_credit_verify_cpu.sbatch BOX=set8
#   sbatch scripts/slurm/exp0_credit_verify_cpu.sbatch BOX=set8 AGGREGATE=1
#
#SBATCH --job-name=exp0_verify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-exp0_verify-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOX:=set8}"
: "${AGGREGATE:=0}"

PLAN="$DMSR_REWARD_ROOT/audits/tile_decomposition/verify_plan_${BOX}.json"
require_input "$PLAN" "scripts/slurm/exp0_decompose_cpu.sbatch"

if [ "$AGGREGATE" = "1" ]; then
    echo "=== aggregating $BOX"
    python -u scripts/reward/tile_loo_credit.py \
        --config "$REWARD_CFG" --mode verify --box "$BOX"
    exit 0
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
TILE=$(python -c "
import json,sys
p=json.load(open('$PLAN'))['tiles']
i=$IDX
print(p[i] if i < len(p) else -1)
")
if [ "$TILE" -lt 0 ]; then
    echo ">>> array index $IDX beyond the $BOX plan; nothing to do."
    exit 0
fi

echo "=== $BOX: splicing HR into tile $TILE and re-running Rockstar"
python -u scripts/reward/rockstar_particles.py \
    --config "$REWARD_CFG" --box "$BOX" --source base \
    --tag "splice${TILE}" --splice-hr-tiles "$TILE" \
    --base-seed "$BASE_SEED" --overwrite

echo "=== $BOX tile $TILE done"
