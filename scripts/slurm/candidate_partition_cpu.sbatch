#!/bin/bash
# Does the candidate/partitioning step of the shaping reward work? (Q1+Q2)
#
# CPU only. Reads the frozen-SR2 and HR fields, the HR/base catalogs, the
# extracted HR member ids and the oracle targets -- all already on disk -- and
# writes retrieval.jsonl / coverage.jsonl / summary.json per box, then draws the
# figures from those rows. No Rockstar, no GPU, no training. Two Ng^3 Eulerian
# position arrays live in RAM at once (~7 GB), hence the memory request.
#
#   sbatch scripts/slurm/candidate_partition_cpu.sbatch <envfile>
#SBATCH --job-name=cand_part
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-cand_part-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_reward_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_reward_common.sh" "$@"
set -euo pipefail

: "${BOXES:=set8,set9}"
: "${R_MULTS:=0.5,1,2,4}"
: "${N_CONTROLS:=32}"
: "${NG_PEAK:=256}"
: "${OUT_NAME:=candidate_partition}"

python -u scripts/reward/diagnose_candidate_partition.py \
    --boxes "$BOXES" --r-mults "$R_MULTS" --n-controls "$N_CONTROLS" \
    --ng-peak "$NG_PEAK" --out-name "$OUT_NAME"

python -u scripts/reward/plot_candidate_partition.py \
    --boxes "$BOXES" --out-name "$OUT_NAME"

echo "=== summaries + figures: $DMSR_REWARD_ROOT/audits/$OUT_NAME/<box>/"
echo "=== Q1: read completeness_sr2_median vs the HR ceiling in summary.json;"
echo "===     a large gap means the members are dispersed in frozen SR2."
echo "=== Q2: read cov_all_within_* for the 'missing' block -- that is whether"
echo "===     density peaks flag the subhalos frozen SR2 does not contain."
