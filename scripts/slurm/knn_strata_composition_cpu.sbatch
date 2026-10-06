#!/bin/bash
# Gate for the catalog-free kNN strata arm: is a self-defined DENSITY stratum a
# usable stand-in for --knn-classes sub_50_200? Reads fields and the HR owner
# array only. No model, no GPU.
#
# Sizing (estimated, one field at a time -- the script frees each field's points
# and tree before loading the next): float64 positions 3.2 GB + cKDTree ~5 GB +
# class vector 0.13 GB + query chunks => ~10 GB peak in 3D. 6D (KC_PHASE_LAMBDA
# > 0) doubles the points and the tree => ~20 GB; submit_knn_strata.sh asks 64G
# there. 48G is ~4.5x the 3D estimate.
#
#   sbatch scripts/slurm/knn_strata_composition_cpu.sbatch <envfile>
#SBATCH --job-name=knnstrata
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-knnstrata-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${KC_FIELDS:?KC_FIELDS must be the label=path list (hr, frozen, ...)}"
: "${KC_CATALOG:?KC_CATALOG must be the HR rockstar ascii}"
: "${KC_OWNER:?KC_OWNER must be the HR owner .npy}"
: "${KC_OUT:?KC_OUT must be the output json}"
: "${KC_GATE_LABEL:=frozen}"
: "${KC_KS:=8 32 64 128}"
: "${KC_STRATA_RANK:=0}"          # 0 = min(ks)
: "${KC_STRATA:=4 8 16}"
: "${KC_CHARGE:=0 0,1}"
: "${KC_TILES:=24}"
: "${KC_QUERIES:=4096}"
: "${KC_BUFFER_CELLS:=8.0}"
: "${KC_N_QUANTILES:=16}"
: "${KC_DEAD_ZONE:=0.02}"
: "${KC_PHASE_LAMBDA:=0}"         # >0 = 6D (x, v/lambda); round 5's pass was 4000
: "${KC_REGION_JSON:=}"
: "${KC_MIN_RECALL_ENRICHMENT:=2.0}"
: "${KC_MIN_RHO:=0.5}"
: "${KC_SEED:=0}"

python -u scripts/features/knn_strata_composition.py \
    --fields $KC_FIELDS \
    --gate-label "$KC_GATE_LABEL" \
    --catalog "$KC_CATALOG" --owner "$KC_OWNER" \
    --ks $KC_KS \
    --strata-rank "$KC_STRATA_RANK" \
    --strata $KC_STRATA \
    --charge $KC_CHARGE \
    --tiles "$KC_TILES" \
    --queries "$KC_QUERIES" \
    --buffer-cells "$KC_BUFFER_CELLS" \
    --n-quantiles "$KC_N_QUANTILES" \
    --dead-zone "$KC_DEAD_ZONE" \
    --phase-lambda "$KC_PHASE_LAMBDA" \
    ${KC_REGION_JSON:+--region-json "$KC_REGION_JSON"} \
    --min-recall-enrichment "$KC_MIN_RECALL_ENRICHMENT" \
    --min-rho "$KC_MIN_RHO" \
    --seed "$KC_SEED" \
    --out "$KC_OUT"
echo "=== kNN strata composition -> $KC_OUT"
echo "=== read verdict.pass: FAIL means a density cut is NOT a catalog-free"
echo "=== substitute for --knn-classes, and the GPU arm does not launch."
