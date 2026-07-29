#!/bin/bash
# Stage 1 (GPU): cache the frozen SR2 field for every box, once.
#
# Everything downstream conditions on the *same* Psi_base, so it is generated
# here and memory-mapped forever after. 3.2 GB per box x 16 boxes = 52 GB, on
# scratch only.
#
#   sbatch scripts/slurm/cache_sr2_base.sbatch
#   sbatch scripts/slurm/cache_sr2_base.sbatch BOXES=set0,set1 SEED=0
#
#SBATCH --job-name=rw_cache_base
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_cache_base-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${BOXES:=}"

require_input "$DATA/lr" "scripts/preprocess_paired.py"

python -u scripts/reward/cache_sr2_base.py \
    --config "$REWARD_CFG" \
    --seed "$BASE_SEED" \
    --device "$DEVICE" \
    ${BOXES:+--boxes "$BOXES"}

echo "=== cached -> $DMSR_REWARD_ROOT/cache/sr2_base"
