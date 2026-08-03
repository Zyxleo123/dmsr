#!/bin/bash
# Stage 1 (CPU, seconds): pick the frozen-SR2 hosts the editor will operate on.
# Pure arithmetic on the frozen catalogs -- no halo finding, no fields.
#
#   sbatch scripts/slurm/local_editor_select_hosts_cpu.sbatch RUN_NAME=le_a
#SBATCH --job-name=le_hosts
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_local_editor/logs/slurm-le_hosts-%A_%a.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_local_editor_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_local_editor_common.sh" "$@"
set -euo pipefail

: "${BOXES:=}"
: "${REPORT_BOUNDS:=1}"

ARGS=(--config "$LE_CFG" --run-name "$RUN_NAME")
[ -n "$BOXES" ] && ARGS+=(--boxes "$BOXES")

python -u scripts/reward/select_editor_hosts.py "${ARGS[@]}"

# The action bounds in the YAML should come from the training catalogs, not from
# the plan's round numbers. This writes the percentiles next to the selection so
# the two are read together.
if [ "$REPORT_BOUNDS" = "1" ]; then
    python -u scripts/reward/select_editor_hosts.py \
        --config "$LE_CFG" --run-name "$RUN_NAME" --report-bounds || true
fi
echo "=== hosts -> $DMSR_LOCAL_EDITOR_ROOT/runs/$RUN_NAME/hosts/"
