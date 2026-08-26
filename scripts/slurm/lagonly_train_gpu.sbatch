#!/bin/bash
# Lagrangian-only-critic sweep: train ONE arm (GPU). One arm per job, so a
# failure or a preemption costs one arm, not the sweep.
#
# The four arms are a one-lever-per-rung ladder off the current best density-fix
# arm (t13_fix_vc32); tests/dmsr/test_lagonly_arms.py asserts that property:
#   t13_lagonly_l003              density_mode "off"    (drop the r=0.08 channel)
#   t13_lagonly_bp4_l003          + bp_steps null       (adv gradient over the whole ODE)
#   t13_lagonly_bp4_l03           + lambda_adv 0.030    (SR2-like adversarial weight)
#   t13_lagonly_bp4_l03_flow01    + lambda_flow 0.1     (demote the mean-seeking term)
#
# Node choice (checked with `myfree`, 2026-08-21): general/a6000 (gpu29 free,
# 48 GB). NOT project/h200 despite 8 idle cards there: `project` only allows
# qos_project_preempt, and train_dmsr has NO resume path -- `model.init_from`
# loads weights only, not the optimizer or the step counter -- so a preemption
# at hour 7 of ~8 costs the whole run. `--exclude=gpu28` keeps the job off the
# 24 GB a5000 in the same partition: the bp4 arms retain activations for all 4
# Euler steps instead of 1, and a CUDA OOM here dies with no Python traceback.
#
# ~1.2 s/step at bp_steps=1 (measured, t13_fix_vc32_s0) x 20000 steps ~ 7 h;
# the bp4 arms are slower, hence the 24 h limit.
#
#   sbatch scripts/slurm/lagonly_train_gpu.sbatch ARM=t13_lagonly_l003
#SBATCH --job-name=lagonly
#SBATCH --partition=general
#SBATCH --exclude=gpu28
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/cosmo_sr_runs/dmsr/lagonly/logs/slurm-lagonly-%j.out

# `set -euo pipefail` lives HERE, not only in the preamble: if the source below
# silently fails, the job would otherwise run on the system python and still
# exit 0.
set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_lagonly_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_lagonly_common.sh" "$@"
set -euo pipefail

CFG="configs/dmsr/${ARM}.yaml"
require_input "$CFG" "configs/dmsr/${ARM}.yaml (this sweep)"

# Both arms warm-start from the phase-1 Stage B checkpoint on the 13-box split,
# same as every t13_* arm. Without it this trains from scratch and is not
# comparable to anything.
require_input "runs/dmsr/t13_stage_b/ckpt_best.pt" "the phase-1 t13 Stage B run"
require_input "runs/dmsr/lr_ssl/encoder_frozen.pt" "the LR masked-SSL pretraining run"

RUN_DIR="$(python -c "
import sys; sys.path.insert(0, 'src')
from cosmo_sr.utils.config import load_config
print(load_config('$CFG')['output']['run_dir'])
")"
echo "=== arm=$ARM  config=$CFG  run_dir=$RUN_DIR"

# --audit-compute is free and prints the update budget, so the log records what
# this run was supposed to cost before it spends it.
python -u -m cosmo_sr.train.train_dmsr --config "$CFG" --audit-compute

python -u -m cosmo_sr.train.train_dmsr --config "$CFG"

echo "=== $ARM done -> $RUN_DIR/{metrics.csv,ckpt_best.pt}"
echo "=== next: full-box ruler, one arm per job:"
echo "    ARMS=\"$ARM:$RUN_DIR\" sbatch scripts/slurm/dfix_stage0_ceiling.sbatch"
