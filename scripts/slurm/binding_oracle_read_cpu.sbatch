#!/bin/bash
# Read the binding-oracle Rockstar catalogs (CPU, catalogs only, minutes):
#   1. gate verdict of each bound field vs the field it was made from
#      (region abundance + structure_distribution core/outskirt)
#   2. peak_supply at 0.5-1 and 1-3 R_vir: per-band bump conversion and
#      objects-on-bumps of every field against HR's
#
#   sbatch --dependency=afterok:<oracle> scripts/slurm/binding_oracle_read_cpu.sbatch <envfile>
#
#SBATCH --job-name=bindread
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-bindread-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 32G  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"
: "${BOX:=set9}"
: "${BO_OUT_DIR:=$DMSR_REWARD_ROOT/binding_oracle}"
: "${BO_VERDICTS:?BO_VERDICTS (name:baseline_tag:baseline_name ...) is required}"
: "${BO_SUPPLY_FIELDS:?BO_SUPPLY_FIELDS (peak_supply --fields, HR first) is required}"
: "${BO_HR_SUBFIND:=}"
: "${BO_READ_TAG:=}"      # suffix on the peak_supply outputs (a second read must not clobber the first)
: "${BO_REGION_EXPORT:=$DMSR_REWARD_ROOT/member_gather/all_blocks_sm_coarse_soft/holdout_${BOX}/export.json}"
RUN_DIR="$(dirname "$BO_REGION_EXPORT")"
H="$DMSR_REWARD_ROOT/flow_rockstar/halos"

failed=0
read -r -a _V <<< "$BO_VERDICTS"
for v in "${_V[@]}"; do
    IFS=: read -r NAME BT BN <<< "$v"
    if [ ! -f "$BO_OUT_DIR/$NAME/${NAME}_rockstar/halos_0.0.ascii" ]; then
        echo ">>> $NAME: no catalog (oracle variant did not finish) -- verdict skipped"
        failed=1; continue
    fi
    mkdir -p "$H/${BOX}__candidate__bind_$NAME"
    ln -sfn "$BO_OUT_DIR/$NAME/${NAME}_rockstar" "$H/${BOX}__candidate__bind_$NAME/${NAME}_rockstar"
    echo "=== verdict bind_$NAME vs $BN"
    bash scripts/slurm/gate_verdict_cpu.sbatch HG_BOX="$BOX" HG_RUN_DIR="$RUN_DIR" \
        HG_RS_TAG="bind_$NAME" HG_BASELINE_TAG="$BT" HG_BASELINE_NAME="$BN" \
        HG_VERDICT_OUT="$BO_OUT_DIR/$NAME/verdict_vs_$BN" || { echo ">>> verdict $NAME FAILED"; failed=1; }
done

for xr in "0.5 1.0" "1.0 3.0"; do
    read -r lo hi <<< "$xr"
    echo "=== peak_supply x $lo-$hi"
    python -u scripts/features/peak_supply.py --fields "$BO_SUPPLY_FIELDS" \
        ${BO_HR_SUBFIND:+--hr-subfind "$BO_HR_SUBFIND"} --region-export "$BO_REGION_EXPORT" \
        --x-min "$lo" --x-max "$hi" --out "$BO_OUT_DIR/${BOX}_peak_supply${BO_READ_TAG:-}_${lo}_${hi}.json" \
        || { echo ">>> peak_supply $lo-$hi FAILED"; failed=1; }
done
echo "=== done; failed=$failed"
exit "$failed"
