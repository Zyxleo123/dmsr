#!/bin/bash
# kNN damage report for one gated arm (scripts/features/knn_gate_report.py),
# a sibling of the Rockstar stage: needs only the arm's and the frozen control's
# spliced fields. ADVISORY -- every path through this script exits 0, so it can
# never strand the gate chain; the Rockstar counts remain the verdict.
#
# Sizing: HR, frozen and arm fields one at a time (float64 positions 3.2 GB +
# cKDTree ~5 GB each, frozen kept as float32 1.6 GB), owner classification
# 0.13 GB => ~12 GB peak; 40G is ~3x. Measured on set9: ~1 min per field.
#
#   sbatch [--dependency=afterok:<arm splice>:<frozen splice>] \
#          scripts/slurm/gate_knn_cpu.sbatch <gate envfile>
#SBATCH --job-name=hg_knn
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-hg_knn-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${HG_BOX:?}" "${HG_SEED:?}" "${HG_RS_TAG:?}" "${HG_RUN_DIR:?}"
FIELDS_DIR="$DMSR_REWARD_ROOT/flow_rockstar/fields"
ARM_FIELD="$FIELDS_DIR/${HG_BOX}__${HG_RS_TAG}__seed${HG_SEED}.npy"
FROZEN_FIELD="$FIELDS_DIR/${HG_BOX}__mgho_frozen_${HG_BOX}__seed${HG_SEED}.npy"
HR_FIELD="$ZFS/DMSR/paired_catnorm/hr/${HG_BOX}.npy"
OWNER="$DMSR_REWARD_ROOT/halos_particles/${HG_BOX}__hr__hr/${HG_BOX}_hr_owner.npy"
CATALOG=$(ls "$DMSR_REWARD_ROOT/halos/${HG_BOX}__hr__hr/"*_rockstar/halos_*.ascii 2>/dev/null | head -1 || true)
OUT="$HG_RUN_DIR/knn_region.json"

skip() { echo "kNN report SKIPPED: $*"; exit 0; }
for f in "$ARM_FIELD" "${ARM_FIELD%.npy}.json" "$FROZEN_FIELD" "${FROZEN_FIELD%.npy}.json" \
         "$HR_FIELD" "$OWNER"; do
    [[ -s "$f" ]] || skip "missing $f"
done
[[ -n "$CATALOG" ]] || skip "no HR catalog for $HG_BOX"
# The frozen control must be spliced on the SAME tiles, or arm-vs-frozen compares
# two different regions.
python - "${ARM_FIELD%.npy}.json" "${FROZEN_FIELD%.npy}.json" <<'PY' || skip "arm and frozen control were spliced on different tiles"
import json, sys
a, b = (sorted(json.load(open(p))["tiles"]) for p in sys.argv[1:3])
sys.exit(0 if a == b else 1)
PY
mkdir -p "$HG_RUN_DIR"

# Settings must stay the calibration's (knn_gate_report.py docstring).
if ! python -u scripts/features/knn_cdf_split_diagnostic.py \
        --fields "hr=$HR_FIELD" "frozen=$FROZEN_FIELD" "arm=$ARM_FIELD" \
        --catalog "$CATALOG" --owner "$OWNER" \
        --region-json "${ARM_FIELD%.npy}.json" --boundary-tiles 0 \
        --ks 8 32 64 128 --n-per-class 100000 --n-uniform 200000 \
        --n-quantiles 32 --dead-zone 0.02 --seed 0 --out "$OUT"; then
    skip "diagnostic failed (see above)"
fi
python -u scripts/features/knn_gate_report.py --knn-json "$OUT" --arm arm \
    --box "$HG_BOX" --out "$HG_RUN_DIR/knn_report.json" || skip "report failed"
python -u scripts/features/knn_cdf_split_plot.py --json "$OUT" \
    --out "$HG_RUN_DIR/knn_region.png" || echo "figure failed (numbers are in $OUT)"
exit 0
