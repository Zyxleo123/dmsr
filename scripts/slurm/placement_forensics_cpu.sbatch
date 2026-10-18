#!/bin/bash
# Placement forensics (CPU, catalogs only): what the inverted subhalo radial
# distribution is made of -- core fragments, real misplaced objects, or an
# outskirt mass shift. Reads every gated catalog on disk, the SUBFIND npz
# catalogs and each arm's summary.json; writes one JSON.
#
#   sbatch scripts/slurm/placement_forensics_cpu.sbatch <envfile> [VAR=value ...]
#
# Configured by PF_* (defaults below). PF_ARMS / PF_SUBFIND take the literal
# " " (a space) for "none": an empty value would trip ${VAR:=default}.
#
#SBATCH --job-name=placefor
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-placefor-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  mem 32G  partition cpu"
nvidia-smi -L 2>/dev/null || echo "(no GPU: CPU job)"

: "${BOX:=set9}"
: "${PF_REGION_EXPORT:=$DMSR_REWARD_ROOT/member_gather/all_blocks_sm_coarse_soft/holdout_${BOX}/export.json}"
: "${PF_CHECK_VERDICT:=$DMSR_REWARD_ROOT/member_gather/all_blocks_radparent/holdout_${BOX}/verdict_vs_coarse_soft.json}"
: "${PF_ARMS:=auto}"
: "${PF_EXTRA:=}"
: "${PF_SUBFIND:=auto}"
: "${PF_OUT:=$DMSR_REWARD_ROOT/placement_forensics/${BOX}_forensics.json}"

require_input "$PF_REGION_EXPORT" "a held-out gate's export stage"
CHECK=()
if [ -r "$PF_CHECK_VERDICT" ]; then CHECK=(--check-verdict "$PF_CHECK_VERDICT"); fi
mkdir -p "$(dirname "$PF_OUT")"

python -u scripts/features/placement_forensics.py \
    --reward-root "$DMSR_REWARD_ROOT" --box "$BOX" \
    --region-export "$PF_REGION_EXPORT" \
    --arms "$PF_ARMS" --extra "$PF_EXTRA" --subfind "$PF_SUBFIND" \
    "${CHECK[@]}" --out "$PF_OUT"

echo "=== wrote $PF_OUT"
