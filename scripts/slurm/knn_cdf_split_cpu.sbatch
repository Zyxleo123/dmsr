#!/bin/bash
# kNN go/no-go round 2: r_k split by HR owner class (is the signal subhalos or
# host cores?) and tile-local estimator error with neighbour tiles / centroid
# centring / frozen neighbours. No model, no GPU.
#
# Sizing (estimated): HR float64 positions 3.2 GB + tree ~5 GB, the tuned field
# the same while the frozen box is held as float32 1.6 GB, class vector 0.13 GB,
# neighbour-scan chunks 0.4 GB => ~19 GB peak. 48G is 2.5x.
# 6D (KS_PHASE_LAMBDA>0, estimated): 6.4 GB points + a possible periodic copy in
# the tree + 3.2 GB float32 frozen => ~22 GB; submit_knn_phase.sh asks 64G.
#
#   sbatch scripts/slurm/knn_cdf_split_cpu.sbatch <envfile>
#SBATCH --job-name=knnsplit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-knnsplit-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${KS_FIELDS:?KS_FIELDS must be the label=path list (hr, frozen, ...)}"
: "${KS_CATALOG:?KS_CATALOG must be the HR rockstar ascii}"
: "${KS_OWNER:?KS_OWNER must be the HR owner .npy}"
: "${KS_OUT:?KS_OUT must be the output json}"
: "${KS_TUNED_LABEL:=selfhostguard}"
: "${KS_KS:=8 32 64 128}"
: "${KS_N_PER_CLASS:=100000}"
: "${KS_SIZES:=0 2 4 8 16}"
: "${KS_BOUNDARY_TILES:=16}"
: "${KS_REGION_JSON:=}"          # an arm field's .json: sample only its spliced tiles
: "${KS_N_UNIFORM:=200000}"
: "${KS_BOUNDARY_QUERIES:=1024}"
: "${KS_PHASE_LAMBDA:=0}"         # round 5: km/s per Mpc/h; >0 = 6D (x, v/lambda)
: "${KS_BOUNDARY_BY_CLASS:=0}"    # 1: stratify boundary queries by HR owner class
: "${KS_RESUME:=0}"               # 1: keep fields already in KS_OUT
EXTRA=()
[[ "$KS_BOUNDARY_BY_CLASS" == 1 ]] && EXTRA+=(--boundary-by-class)
[[ "$KS_RESUME" == 1 ]] && EXTRA+=(--resume)

python -u scripts/features/knn_cdf_split_diagnostic.py \
    --fields $KS_FIELDS \
    --tuned-label "$KS_TUNED_LABEL" \
    --catalog "$KS_CATALOG" --owner "$KS_OWNER" \
    --ks $KS_KS \
    --n-per-class "$KS_N_PER_CLASS" \
    --sizes $KS_SIZES \
    --boundary-tiles "$KS_BOUNDARY_TILES" \
    --boundary-queries "$KS_BOUNDARY_QUERIES" \
    --phase-lambda "$KS_PHASE_LAMBDA" \
    --n-uniform "$KS_N_UNIFORM" \
    ${KS_REGION_JSON:+--region-json "$KS_REGION_JSON"} \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --out "$KS_OUT"
echo "=== kNN split diagnostic -> $KS_OUT"
