#!/bin/bash
# Stage 3 (CPU): read the three catalogs and answer "are the clumps bound?".
#
# Pure catalog arithmetic on ASCII files already on disk -- no field, no finder.
# Not instant despite that: each of the three catalogs is a ~160 MB ASCII table
# and `load_rockstar_ascii` parses it with np.loadtxt, which is minutes and a few
# GB per file. Hence 48 GB and two hours rather than the 16 GB/30 min a "just
# read some JSON" job would get.
#
#   sbatch scripts/slurm/gather_compare_cpu.sbatch <envfile>
#SBATCH --job-name=hg_cmp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hg_cmp-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HG_BOX:=set8}"
: "${HG_RS_TAG:?HG_RS_TAG must be set}"
: "${HG_HOST_ID:=271800}"
: "${HG_MIN_P:=50}"
: "${HG_SWEEP:=0}"

CAND="$DMSR_REWARD_ROOT/flow_rockstar/halos/${HG_BOX}__candidate__${HG_RS_TAG}"
[[ -d "$CAND" ]] || gate_failed "no candidate catalog at $CAND (did the Rockstar stage finish?)"

echo "=== compare: box=$HG_BOX tag=$HG_RS_TAG host=$HG_HOST_ID"
# The per-R_vir count and the per-target hit rate answer different questions;
# both are reported, and subhalos.json is what makes the second possible.
TGT=""
[[ -f "${HG_RUN_DIR:-}/subhalos.json" ]] && TGT="$HG_RUN_DIR/subhalos.json"

python -u scripts/reward/compare_gather_catalog.py \
    --box "$HG_BOX" --tag "$HG_RS_TAG" \
    --host-id "$HG_HOST_ID" --min-p "$HG_MIN_P" \
    ${TGT:+--targets-json "$TGT"} \
    $([[ "$HG_SWEEP" == "1" ]] && echo --sweep)

echo "=== result: $DMSR_REWARD_ROOT/flow_rockstar/compare/${HG_BOX}__${HG_RS_TAG}.json"
