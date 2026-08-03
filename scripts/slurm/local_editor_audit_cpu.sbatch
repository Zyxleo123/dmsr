#!/bin/bash
# Stage 0 (CPU, minutes): calibrate the feasibility thresholds from measured
# populations. Emits no reward, so it is the one stage exempt from the
# calibrated-constraints guard.
#
#   sbatch scripts/slurm/local_editor_audit_cpu.sbatch RUN_NAME=le_a
#SBATCH --job-name=le_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_local_editor/logs/slurm-le_audit-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_local_editor_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_local_editor_common.sh" "$@"
set -euo pipefail

: "${BOXES:=}"
: "${MARGIN:=2.0}"

ARGS=(--config "$LE_CFG" --run-name "$RUN_NAME" --margin "$MARGIN")
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")

python -u scripts/reward/audit_local_editor_constraints.py "${ARGS[@]}"
echo "=== paste the proposed block into configs/reward/local_editor.yaml,"
echo "=== set calibrated: true, and commit before any scoring stage runs."
