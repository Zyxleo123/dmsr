#!/bin/bash
# Pool the per-box all-hosts results into the one cross-box table (CPU, ~5 s).
#
#   sbatch scripts/slurm/gather_all_hosts_report_cpu.sbatch <envfile>
#
# Reads only the JSONs the abundance stage wrote, so the table re-renders
# without re-parsing a single catalog.
#SBATCH --job-name=ah_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-ah_report-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${AH_BOXES:?set AH_BOXES to the boxes to pool}"
: "${AH_BASELINE:=frozen}"
: "${AH_OUT_DIR:=$DMSR_REWARD_ROOT/member_gather/all_hosts}"

INPUTS=()
for b in $AH_BOXES; do
    if [[ -f "$AH_OUT_DIR/$b.json" ]]; then
        INPUTS+=("$AH_OUT_DIR/$b.json")
    else
        echo "note: $AH_OUT_DIR/$b.json missing -- $b is left out of the pool."
    fi
done
if [[ ${#INPUTS[@]} -eq 0 ]]; then
    echo "GATE: no per-box result to pool. The abundance stage produced nothing;"
    echo "  read its log (slurm-ah_abund-*.out) for why."
    exit 0
fi

python -u scripts/features/gather_all_hosts_report.py \
    --inputs "${INPUTS[@]}" \
    --baseline "$AH_BASELINE" \
    --out "$AH_OUT_DIR/report"

echo "=== result: $AH_OUT_DIR/report.json"
echo "=== slide table: $AH_OUT_DIR/report.tex"
