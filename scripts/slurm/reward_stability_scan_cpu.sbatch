#!/bin/bash
# Which scalar catalog target is stable across SR2 seeds?
#
# Reads only the saved per-tile catalog summaries (~17 MB of JSONL) -- no
# Rockstar, no field, no GPU -- and scores a menu of candidate reward
# functionals for seed churn, effect size and cross-seed rank consistency.
# Seconds of work, but submitted because the login node kills real work and
# because every result in this line has to be reproducible from a job record.
#
#   sbatch scripts/slurm/reward_stability_scan_cpu.sbatch <envfile>
#SBATCH --job-name=sd_rstab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_rstab-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

require_input "$REWARD_MODEL_JSON" "scripts/slurm/fit_reward_model_cpu.sbatch"

python -u scripts/reward/reward_stability_scan.py \
    --config "$DIRECT_CFG" --run-name "$RUN_NAME"

echo "=== report: $DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/reward_stability_scan.json"
echo "=== read d_alpha1 (effect size in seed-noise units) and p_rank first;"
echo "=== rho_boxes is the model-free cross-seed ordering check."
