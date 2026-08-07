#!/bin/bash
# Phase 1.2 (CPU): render the projection-oracle report and the recommendation.
#
# Reads only the JSON rows the audit array wrote, so it is cheap and can be
# re-run whenever the bootstrap settings or the metric list change -- no field is
# rebuilt and no halo finder runs. That is the point of writing rows to disk.
#
#   sbatch scripts/slurm/projection_oracle_report_cpu.sbatch RUN_NAME=proj0
#
#SBATCH --job-name=rw_proj_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_proj_report-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"

: "${PROJ_CFG:=configs/reward/projection_oracle.yaml}"
: "${RUN_NAME:=proj0}"
: "${SWEEPS:=joint,disp_only,vel_only}"
: "${ALPHAS:=}"
: "${N_BOOT:=2000}"
: "${BOOTSTRAP_SEED:=0}"
: "${CI:=0.95}"

RUN_DIR="$DMSR_REWARD_ROOT/audits/projection_oracle/$RUN_NAME"
require_input "$RUN_DIR/rows" "scripts/slurm/audit_projection_oracle.sbatch"

python -u scripts/reward/projection_oracle_report.py \
    --config "$PROJ_CFG" \
    --run-name "$RUN_NAME" \
    --sweeps "$SWEEPS" \
    ${ALPHAS:+--alphas "$ALPHAS"} \
    --n-boot "$N_BOOT" --bootstrap-seed "$BOOTSTRAP_SEED" --ci "$CI"

echo "=== report -> $RUN_DIR/projection_oracle_report.{json,md}"
echo "=== rows csv -> $RUN_DIR/projection_oracle_rows.csv"
echo "=== paste the recommended alpha_disp/alpha_vel into"
echo "===   configs/reward/gaussian_policy.yaml under correction:"
