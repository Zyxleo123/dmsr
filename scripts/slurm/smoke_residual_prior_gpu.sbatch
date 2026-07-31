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

# The smoke run is 6 steps on a tiny model: it cannot pass the real Gate A and
# is not meant to. What it CAN answer mechanically is "did anything produce a
# NaN, and does a sampled residual have spread", which is why the criteria are
# evaluated here rather than left to a human reading metrics.csv. Unevaluated
# criteria are allowed; a hard FAIL still stops the chain, because the long job
# would fail the same way eight hours later.
python -u scripts/reward/gate_a_check.py \
    --config "$REWARD_CFG" \
    --run-dir "$RUN_DIR" \
    --out "$RUN_DIR/gate_a_smoke.json" \
    --allow-unevaluated \
    --strict \
    --loss-drop -1e9 \
    || { echo "!!! Gate A smoke criteria FAILED; see $RUN_DIR/gate_a_smoke.json"
         echo "!!! The dependent training job will stay on DependencyNeverSatisfied."
         echo "!!! That is deliberate: it would fail the same way, eight hours later."
         exit 1; }

echo "=== Gate A smoke verdict -> $RUN_DIR/gate_a_smoke.json"
echo "=== the REAL Gate A runs after training:"
echo "===   python scripts/reward/gate_a_check.py --run-dir <prior run dir>"
