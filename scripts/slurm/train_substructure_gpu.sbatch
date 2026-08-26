#!/bin/bash
# Pilot step 5 (GPU): train the moment-constrained substructure module on set8.
#   docs/sr2_substructure_module.md  section 4  (Option B, native Lagrangian tiles)
#   docs/sr2_moment_constraint.md    section 5  (target = Pi(Psi_HR - Psi_SR2))
#
# Node choice (checked with `myfree`): general/a5000 (gpu28, 24 GB, idle) is the
# clean non-preemptible option; a 64^3 x 6 U-Net (width 64, 3 levels) trains well
# under 24 GB with amp on. On an OOM, lower train.batch_size or set
# model.use_checkpoint: true in the config. The frozen SR2 box is generated once
# and cached to moment_target/<box>/<box>_sr2_box.npy, so a resubmit skips it.
#
# The in-loop Rockstar eval (config eval_rockstar) runs a halo-finder subprocess
# on the GPU node's CPUs every few thousand steps on a ~192^3 mini-box (~7M
# particles); that plus the SR2 box + target + scale fields in host RAM is why
# this asks for 16 CPUs and 80 GB (gpu28 has 48 free CPUs, 500 GB). Set
# eval_rockstar.enabled: false to drop back to a pure GPU job.
#
#   sbatch scripts/slurm/train_substructure_gpu.sbatch \
#       SUB_CONFIG=configs/substructure_set8.yaml RUN_NAME=substructure_set8
#SBATCH --job-name=sub_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sub_train-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${SUB_CONFIG:=configs/substructure_set8.yaml}"
: "${SUB_BOX:=set8}"
: "${RUN_NAME:=substructure_set8}"
# Checkpoints + logs land on scratch, not in home (tight quota) or the repo.
export SUBSTRUCTURE_RUN_DIR="${SUBSTRUCTURE_RUN_DIR:-$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME}"

MT="$DMSR_REWARD_ROOT/moment_target/$SUB_BOX/${SUB_BOX}_moment_target.npy"
FEAT="$DMSR_REWARD_ROOT/lagrangian_host/$SUB_BOX/${SUB_BOX}_lagrangian_host.npz"
require_input "$MT" "scripts/slurm/submit_moment_target.sh (build_moment_target)"
require_input "$FEAT" "scripts/slurm/submit_lagrangian_host.sh"
require_input "$PROJECT/external/SRS-map2map/SRmodel/G_z0.pt" \
    "the frozen SR2 checkpoint (configs/sr2_baseline/freeze.yaml)"

echo "=== train: config=$SUB_CONFIG box=$SUB_BOX run_dir=$SUBSTRUCTURE_RUN_DIR"
python -u -m cosmo_sr.train.train_substructure --config "$SUB_CONFIG"

echo "=== checkpoint -> $SUBSTRUCTURE_RUN_DIR/ckpt_last.pt"
