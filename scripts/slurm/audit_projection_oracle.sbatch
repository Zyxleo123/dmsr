#!/bin/bash
# Phase 1.2 (CPU array): the projection oracle.
#
# Assembles B + T_alpha(r) for every arm, runs the real Rockstar pipeline on the
# complete periodic box, and scores occupation, abundance, host recovery, power
# spectra and the LR/coarse mismatch. No GPU: the SR2 base fields are already
# cached, so this stage only reads arrays and runs the halo finder.
#
# The REFERENCE arms (`hr`, `sr2`) run in their own prerequisite array. Every
# sweep row matches against the HR catalog, so it has to exist first -- and two
# shards running Rockstar in one work directory corrupt each other's catalog,
# the same failure catalog_reward_oracle_cpu.sbatch splits its baselines to avoid.
#
#   JR=$(sbatch --parsable --array=0-3 scripts/slurm/audit_projection_oracle.sbatch \
#          RUN_NAME=proj0 REFERENCES=only)
#   JS=$(sbatch --parsable --dependency=afterok:$JR --array=0-15 \
#          scripts/slurm/audit_projection_oracle.sbatch RUN_NAME=proj0)
#   sbatch --dependency=afterok:$JS scripts/slurm/projection_oracle_report_cpu.sbatch \
#          RUN_NAME=proj0
#
#SBATCH --job-name=rw_proj_oracle
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_proj_oracle-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${PROJ_CFG:=configs/reward/projection_oracle.yaml}"
: "${RUN_NAME:=proj0}"
: "${BOXES:=}"                 # empty = projection_oracle.boxes from the config
: "${BASE_SEEDS:=0,1,2}"
: "${ALPHAS:=}"                # empty = projection_oracle.alphas
: "${SWEEPS:=joint,disp_only,vel_only}"
: "${REFERENCES:=shard0}"      # only | shard0 | skip | all
: "${REWARD_MODEL:=$DMSR_REWARD_ROOT/reward_model/reward_model.json}"
: "${MHOST_MIN:=0.0}"
: "${SLAB:=64}"
: "${SKIP_HALO:=0}"
: "${SHARD:=${SLURM_ARRAY_TASK_ID:-0}}"
: "${NUM_SHARDS:=${SLURM_ARRAY_TASK_COUNT:-1}}"

require_input "$DMSR_REWARD_ROOT/cache/sr2_base" "scripts/slurm/cache_sr2_base.sbatch"
require_input "$DATA/hr" "scripts/preprocess_paired.py"
[ "$SKIP_HALO" = "1" ] || require_input "external/rockstar/rockstar" "make -C external/rockstar"

# Not a hard requirement: without it the occupation curves are still measured,
# but R_occ_reliable is absent and the decision rule loses its primary metric.
if [ ! -r "$REWARD_MODEL" ]; then
    echo ">>> NOTE: no reward model at $REWARD_MODEL."
    echo ">>> Occupation and abundance are still measured, but R_occ_reliable"
    echo ">>> will be missing and projection_oracle_report.py cannot apply the"
    echo ">>> primary decision metric. Produce it with:"
    echo ">>>   sbatch scripts/slurm/fit_reward_model_cpu.sbatch"
fi

python -u scripts/reward/audit_projection_oracle.py \
    --config "$PROJ_CFG" \
    --run-name "$RUN_NAME" \
    ${BOXES:+--boxes "$BOXES"} \
    --base-seeds "$BASE_SEEDS" \
    ${ALPHAS:+--alphas "$ALPHAS"} \
    --sweeps "$SWEEPS" \
    --references "$REFERENCES" \
    --mhost-min "$MHOST_MIN" \
    --slab "$SLAB" \
    $([ "$SKIP_HALO" = "1" ] && echo --skip-halo) \
    $([ -r "$REWARD_MODEL" ] && echo --reward-model "$REWARD_MODEL") \
    --shard "$SHARD" --num-shards "$NUM_SHARDS"

echo "=== shard $SHARD done (references=$REFERENCES)"
echo "=== rows -> $DMSR_REWARD_ROOT/audits/projection_oracle/$RUN_NAME/rows"
echo "=== report: sbatch --dependency=afterok:\$JID \\"
echo "===   scripts/slurm/projection_oracle_report_cpu.sbatch RUN_NAME=$RUN_NAME"
