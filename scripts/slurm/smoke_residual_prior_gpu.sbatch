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
# NaN, and does a sampled residual have spread".
#
# So only those two criteria are ENFORCED here (--enforce 2,5). Everything else
# is still computed and written to gate_a_smoke.json as advisory, but cannot
# block: this job is --strict, its failure strands the real training job on
# DependencyNeverSatisfied, and a residual-RMS or power-spectrum criterion that
# a six-step model was never going to satisfy would do exactly that for no
# reason. The physical criteria are decided after real training, by
# gate_a_check_cpu.sbatch.
: "${SMOKE_ENFORCE:=2,5}"
python -u scripts/reward/gate_a_check.py \
    --config "$REWARD_CFG" \
    --run-dir "$RUN_DIR" \
    --out "$RUN_DIR/gate_a_smoke.json" \
    --enforce "$SMOKE_ENFORCE" \
    --allow-unevaluated \
    --strict \
    --loss-drop -1e9 \
    || { echo "!!! Gate A smoke criteria FAILED; see $RUN_DIR/gate_a_smoke.json"
         echo "!!! The dependent training job will stay on DependencyNeverSatisfied."
         echo "!!! That is deliberate: it would fail the same way, eight hours later."
         exit 1; }

echo "=== Gate A smoke verdict -> $RUN_DIR/gate_a_smoke.json (enforced: $SMOKE_ENFORCE)"
echo "=== the REAL Gate A runs after training, and needs the tile scan:"
echo "===   sbatch scripts/slurm/gate_a_tile_scan_gpu.sbatch"
echo "===   sbatch scripts/slurm/gate_a_check_cpu.sbatch"
