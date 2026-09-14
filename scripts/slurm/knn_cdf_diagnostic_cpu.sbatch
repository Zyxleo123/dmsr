#!/bin/bash
# Go/no-go for the kNN-distance loss (src/cosmo_sr/features/knn_cdf.py): does
# r_k at k=8..128 separate frozen SR2 from HR on set9, does the arm that PASSED
# the held-out Rockstar gate move toward HR, and how many cells of buffer does a
# 64^3 training tile need? Full-box periodic cKDTree per field; no model, no GPU.
#
# Sizing, estimated not guessed (mgtrain-mem-was-9x-overasked): per field a
# float64 position table 3.2 GB + cKDTree copy 3.2 GB + index/nodes ~2 GB + load
# temporaries ~3.3 GB => ~12 GB peak, one field at a time. 40G is ~3x that.
# The tree build is single-threaded; the queries use every allocated core.
#
#   sbatch scripts/slurm/knn_cdf_diagnostic_cpu.sbatch <envfile>
#SBATCH --job-name=knncdf
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-knncdf-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${KC_FIELDS:?KC_FIELDS must be the space-separated label=path list (incl hr)}"
: "${KC_OUT:?KC_OUT must be the output json}"
: "${KC_KS:=8 32 64 128}"
: "${KC_N_SAMPLE:=200000}"
: "${KC_BUFFERS:=0 2 4 6 8 12 16}"
: "${KC_BOUNDARY_TILES:=24}"

python -u scripts/features/knn_cdf_diagnostic.py \
    --fields $KC_FIELDS \
    --ks $KC_KS \
    --n-sample "$KC_N_SAMPLE" \
    --buffers $KC_BUFFERS \
    --boundary-tiles "$KC_BOUNDARY_TILES" \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --out "$KC_OUT"

echo "=== kNN CDF diagnostic -> $KC_OUT"
