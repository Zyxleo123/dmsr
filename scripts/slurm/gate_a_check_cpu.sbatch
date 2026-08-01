#!/bin/bash
# Gate A verdict (CPU, seconds): all eight criteria of sec. 3a -> gate_a.json.
#
# Reads metrics.csv, the residual-target audit and the tile scan; runs no model
# and needs no GPU. This is the job sample_reward_oracle.sbatch's gate check is
# really waiting for, and the only one that can write a gate_a.json with
# criterion 6 decided.
#
#   sbatch scripts/slurm/gate_a_check_cpu.sbatch
#   sbatch scripts/slurm/gate_a_check_cpu.sbatch RUN_DIR=.../residual_prior
#
#SBATCH --job-name=rw_gate_a
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-rw_gate_a-%j.out

_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${RUN_DIR:=$DMSR_REWARD_ROOT/checkpoints/residual_prior}"
: "${TILE_SCAN_JSON:=$RUN_DIR/tile_scan.json}"
: "${AUDIT_JSON:=$DMSR_REWARD_ROOT/audits/residual_target_audit/residual_target_audit.json}"
: "${GATE_A_JSON:=$RUN_DIR/gate_a.json}"

require_input "$RUN_DIR/metrics.csv" "scripts/slurm/train_residual_prior.sbatch"

if [[ ! -r "$TILE_SCAN_JSON" ]]; then
    echo ">>> no $TILE_SCAN_JSON: criterion 6 (tile independence) cannot be"
    echo ">>> decided, so Gate A will report passed: false and Gate B will not"
    echo ">>> sample. Run scripts/slurm/gate_a_tile_scan_gpu.sbatch first."
fi

# Not strict: a FAILING Gate A must leave a readable verdict rather than strand
# every dependent job on DependencyNeverSatisfied. sample_reward_oracle.sbatch
# reads gate_a.json and explains itself.
python -u scripts/reward/gate_a_check.py \
    --config "$REWARD_CFG" \
    --run-dir "$RUN_DIR" \
    --out "$GATE_A_JSON" \
    ${TILE_SCAN_JSON:+--tile-scan "$TILE_SCAN_JSON"} \
    ${AUDIT_JSON:+--audit "$AUDIT_JSON"} \
    || echo ">>> gate_a_check.py itself errored; treat Gate A as UNDECIDED"

echo "=== Gate A verdict -> $GATE_A_JSON"
