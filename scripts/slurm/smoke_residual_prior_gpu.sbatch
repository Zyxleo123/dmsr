#!/bin/bash
# Gate A smoke test (GPU, minutes): a few optimizer steps and one sampling pass.
#
# It answers three questions and nothing else: does the model run on a GPU at the
# real crop size, does the loss stay finite, and does a sampled residual have
# nonzero spread. Any of those failing means the long job would fail too, just
# eight hours later.
#
#   sbatch scripts/slurm/smoke_residual_prior_gpu.sbatch
#
#SBATCH --job-name=rw_smoke_prior
#SBATCH --partition=legacy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_smoke_prior-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${RUN_DIR:=$DMSR_REWARD_ROOT/checkpoints/smoke_prior}"
: "${SIGMA_JSON:=$DMSR_REWARD_ROOT/audits/residual_target_audit/sigma_res.json}"

SIGMA_ARG=()
[[ -r "$SIGMA_JSON" ]] && SIGMA_ARG=(--sigma-res "$SIGMA_JSON")

python -u scripts/reward/train_residual_prior.py \
    --config "$MODEL_CFG" \
    --out "$RUN_DIR" \
    --smoke \
    "${SIGMA_ARG[@]}" \
    --set train.device="$DEVICE"

echo "=== smoke run -> $RUN_DIR"
echo "=== Gate A: check metrics.csv for finite loss and diag_sample_diversity > 0"
