#!/bin/bash
# Diagnose the within-tile ranking ceiling: bug or genuine signal-to-noise?
# CPU only -- it reads ranking_arrays_{arm}.npz (saved by the gate) and nothing
# else, so the verdict and figure are redrawable without any GPU pass.
#
#   sbatch scripts/slurm/diagnose_rank_ceiling_cpu.sbatch <envfile> RUN_NAME=direct_a DIAG_ARMS=a,b
#SBATCH --job-name=sd_rank_diag
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/sr2_direct/logs/slurm-sd_rank_diag-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${DIAG_ARMS:=a,b}"

# Need at least one arm's ranking arrays; the script itself reports any missing.
FIRST_ARM="${DIAG_ARMS%%,*}"
require_input "$DMSR_SR2_DIRECT_ROOT/runs/$RUN_NAME/ranking_arrays_${FIRST_ARM}.npz" \
    "scripts/reward/gate_catalog_proxy.py --arm $FIRST_ARM"

python -u scripts/reward/diagnose_rank_ceiling.py \
    --run-name "$RUN_NAME" --arms "$DIAG_ARMS"
