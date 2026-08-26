#!/bin/bash
# Step-5 target precompute: x1 = Pi(Psi_HR - Psi_SR2) for one box.
# docs/sr2_moment_constraint.md section 5.1 -- the whole-box point where the
# per-host affine-moment projector enters training.
#
# GPU, because it runs the frozen SR2 generator over the 512 tiles of the box.
# Everything after (residual, projection, diagnostics) is CPU arithmetic on the
# 3.2 GB whole-box field and is kept here only so the field is formed once. Peak
# host RAM ~15 GB (HR + SR2 + residual + target); any `general` GPU is enough for
# a 64^3 tile batch, so this is sized on RAM, not GPU memory. On an OOM lower
# MT_BATCH. Resumable: an existing target is skipped unless MT_FORCE=1.
#
#   sbatch scripts/slurm/build_moment_target_gpu.sbatch <envfile>
#SBATCH --job-name=momtgt
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-momtgt-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${DATA:=/zfsauton/scratch/yixiz/DMSR/paired_catnorm}"
: "${MT_BOX:=set8}"
: "${MT_MODE:=affine}"
: "${MT_SEED:=0}"
: "${MT_BATCH:=8}"
: "${MT_DTYPE:=float16}"
: "${MT_TOP_HOSTS:=24}"
: "${MT_CROP:=128}"
: "${MT_FORCE:=0}"

require_input "$DATA/hr/$MT_BOX.npy" "scripts/reward/preprocess (paired_catnorm)"
require_input "$DATA/lr/$MT_BOX.npy" "scripts/reward/preprocess (paired_catnorm)"
require_input "$PROJECT/external/SRS-map2map/SRmodel/G_z0.pt" \
    "the frozen SR2 checkpoint (configs/sr2_baseline/freeze.yaml)"

# The host features are an UPSTREAM GATE, not a hard input: if they are missing
# (their build failed or was never run) this box is skipped with exit 0, so it
# does not strand the render dependent in DependencyNeverSatisfied. Distinguish
# it from the hard inputs above, which really must exist.
FEAT="$DMSR_REWARD_ROOT/lagrangian_host/$MT_BOX/${MT_BOX}_lagrangian_host.npz"
if [[ ! -r "$FEAT" ]]; then
    echo "GATE: no host features at $FEAT for $MT_BOX -- upstream feature build "\
"did not produce them. Skipping this box (exit 0)." >&2
    exit 0
fi

python -u scripts/features/build_moment_target.py \
    --config "$DIRECT_CFG" \
    --box "$MT_BOX" \
    --mode "$MT_MODE" \
    --seed "$MT_SEED" \
    --batch "$MT_BATCH" \
    --dtype "$MT_DTYPE" \
    --top-hosts "$MT_TOP_HOSTS" \
    --crop "$MT_CROP" \
    $([ "$MT_FORCE" = "1" ] && echo --force) \
    ${DEVICE:+--device "$DEVICE"}

echo "=== result: $DMSR_REWARD_ROOT/moment_target/$MT_BOX/${MT_BOX}_moment_target.npy"
echo "===         + ${MT_BOX}_moment_target_summary.json  (the verdict)"
echo "===         + ${MT_BOX}_moment_target_diag.npz      (the figure's data)"
