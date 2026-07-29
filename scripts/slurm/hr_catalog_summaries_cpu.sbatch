#!/bin/bash
# Stage 4a (CPU array): HR and frozen-SR2 chunk summaries for the reward target.
#
# One array task per box. The HR pass defines mu_HR and C_reg; the base pass is
# the counterfactual every per-chunk marginal contribution is measured against,
# so both are cached here once and never recomputed.
#
#   sbatch --array=0-15 scripts/slurm/hr_catalog_summaries_cpu.sbatch
#   sbatch --array=0-7  scripts/slurm/hr_catalog_summaries_cpu.sbatch SOURCES=hr
#
#SBATCH --job-name=rw_hr_cat
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_hr_cat-%A_%a.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${ALL_BOXES:=set0 set1 set2 set3 set4 set5 set6 set7 set8 set9 set10 set11 set12 set13 set14 set15}"
: "${SOURCES:=hr base}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
read -r -a _BOXES <<< "${ALL_BOXES//,/ }"
BOX="${BOX:-${_BOXES[$IDX]}}"

require_input "external/rockstar/rockstar" "make -C external/rockstar"

for SRC in ${SOURCES//,/ }; do
    echo "=== $BOX / $SRC"
    if [ "$SRC" = "base" ]; then
        require_input "$DMSR_REWARD_ROOT/cache/sr2_base" \
            "scripts/slurm/cache_sr2_base.sbatch"
    fi
    python -u scripts/reward/catalog_summaries.py \
        --config "$REWARD_CFG" \
        --box "$BOX" --source "$SRC" --base-seed "$BASE_SEED"
done

echo "=== summaries -> $DMSR_REWARD_ROOT/catalog_cache"
