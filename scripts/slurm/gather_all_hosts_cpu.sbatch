#!/bin/bash
# All-hosts subhalo abundance for ONE gated box (CPU, ~10 min).
#
#   sbatch scripts/slurm/gather_all_hosts_cpu.sbatch <envfile>
#
# Parses the box's HR, base and every gated candidate ASCII catalog (~100 MB
# each) and counts substructure inside every HR host aperture of the edited
# region. No field is read and no halo finder runs, so this is minutes and a
# little memory -- but it is far more than a login node tolerates.
#SBATCH --job-name=ah_abund
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-ah_abund-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${AH_BOX:?set AH_BOX to the box to report on}"
: "${AH_RUN_DIR:?set AH_RUN_DIR to a gated export dir of that box}"
: "${AH_ARMS:=}"
: "${AH_DILATE:=2.0}"
: "${AH_HOST_MIN_P:=200}"
: "${AH_SUB_MIN_P:=50}"
: "${AH_MATCH_RVIR:=1.0}"
: "${AH_MAX_HOSTS:=0}"
: "${AH_OUT_DIR:=$DMSR_REWARD_ROOT/member_gather/all_hosts}"

# The export dir is what fixes the spliced tile list. Without it the region is
# undefined, so say which stage produces it and exit 0 rather than stranding
# the report job in DependencyNeverSatisfied.
if [[ ! -f "$AH_RUN_DIR/export.json" ]]; then
    echo "GATE: no export.json under $AH_RUN_DIR"
    echo "  produced by: scripts/slurm/submit_gather_holdout_rockstar.sh"
    echo "  (the export stage of the held-out gate for $AH_BOX)"
    exit 0
fi

mkdir -p "$AH_OUT_DIR"
python -u scripts/features/gather_all_hosts_abundance.py \
    --reward-root "$DMSR_REWARD_ROOT" \
    --box "$AH_BOX" \
    --run-dir "$AH_RUN_DIR" \
    --arms "$AH_ARMS" \
    --dilate "$AH_DILATE" \
    --host-min-p "$AH_HOST_MIN_P" \
    --sub-min-p "$AH_SUB_MIN_P" \
    --match-rvir "$AH_MATCH_RVIR" \
    --max-hosts "$AH_MAX_HOSTS" \
    --out "$AH_OUT_DIR/$AH_BOX"

echo "=== result: $AH_OUT_DIR/$AH_BOX.json"
