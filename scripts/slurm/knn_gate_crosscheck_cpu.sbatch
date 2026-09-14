#!/bin/bash
# Object forensics for every gated set9 arm (HR-matched vs unmatched 50-200p
# subhalos, lost >=200p frozen halos, fragment candidates) and Spearman against
# the kNN terms, then the figure. Catalogs + one JSON; no fields, no model.
#
#   sbatch scripts/slurm/knn_gate_crosscheck_cpu.sbatch <envfile>
#SBATCH --job-name=knngate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-knngate-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${KG_REGION_JSON:?KG_REGION_JSON must be an arm field .json}"
: "${KG_KNN_JSON:?KG_KNN_JSON must be the all-arms region split JSON}"
: "${KG_OUT:?KG_OUT must be the output json}"
: "${KG_BOX:=set9}"
if [[ ! -s "$KG_KNN_JSON" ]]; then
    echo "no kNN JSON at $KG_KNN_JSON -- running forensics without correlations"
    KNN_ARG=()
else
    KNN_ARG=(--knn-json "$KG_KNN_JSON")
fi
python -u scripts/features/knn_gate_crosscheck.py \
    --reward-root "$DMSR_REWARD_ROOT" --box "$KG_BOX" \
    --region-json "$KG_REGION_JSON" "${KNN_ARG[@]}" --out "$KG_OUT"
python -u scripts/features/knn_gate_crosscheck_plot.py --json "$KG_OUT" \
    --out "${KG_OUT%.json}.png"
