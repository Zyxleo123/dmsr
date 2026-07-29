#!/bin/bash
# Stage 4b + 5 (CPU, minutes): fit the reward model and calibrate the constraints.
#
# Both are cheap and both must happen before any candidate is scored: the reward
# needs mu_HR and C_reg, and the feasibility filter needs thresholds derived from
# the frozen baseline rather than chosen to let everything pass.
#
#   sbatch scripts/slurm/fit_reward_model_cpu.sbatch
#   sbatch scripts/slurm/fit_reward_model_cpu.sbatch SHRINKAGE=0.2
#
#SBATCH --job-name=rw_fit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_fit-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${FIT_BOXES:=}"             # empty = train split
: "${ENSEMBLE_SIZE:=16}"
: "${DRAWS:=400}"
: "${SHRINKAGE:=0.1}"
: "${MARGIN:=1.5}"
: "${LOW_K_FRACTION:=0.1}"
: "${SKIP_CALIBRATION:=0}"

require_input "$DMSR_REWARD_ROOT/catalog_cache" \
    "scripts/slurm/hr_catalog_summaries_cpu.sbatch"

python -u scripts/reward/fit_reward_model.py \
    --config "$REWARD_CFG" \
    ${FIT_BOXES:+--boxes "$FIT_BOXES"} \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --draws "$DRAWS" \
    --shrinkage "$SHRINKAGE"

if [ "$SKIP_CALIBRATION" != "1" ]; then
    python -u scripts/reward/calibrate_constraints.py \
        --config "$REWARD_CFG" \
        ${FIT_BOXES:+--boxes "$FIT_BOXES"} \
        --base-seed "$BASE_SEED" \
        --margin "$MARGIN" \
        --low-k-fraction "$LOW_K_FRACTION"
    echo "=== paste the proposed block into configs/reward/reward.yaml:"
    echo "===   $DMSR_REWARD_ROOT/audits/constraint_calibration/constraints.yaml"
fi

echo "=== reward model -> $DMSR_REWARD_ROOT/reward_model/reward_model.json"
