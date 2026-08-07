#!/bin/bash
# SR2 halo/subhalo abundance + occupancy + positional-fidelity report (CPU only).
#
# Reads the frozen Stage-1 Rockstar catalogs; runs no model and needs no GPU.
# Analysis and plotting are separate stages, so figures can be redrawn from
# metrics.npz without re-parsing the catalogs:
#
#   sbatch scripts/slurm/sr2_subhalo_report_cpu.sbatch
#   sbatch scripts/slurm/sr2_subhalo_report_cpu.sbatch SR2_STAGE=plot   # redraw only
#
# All overrides are SR2_-prefixed (SR2_BOX, SR2_SEEDS, SR2_MATCH_SEEDS, SR2_OUT,
# SR2_HALOS_ROOT, SR2_ROOT, SR2_STAGE) and are passed as positional args.
#
#SBATCH --job-name=sr2_subrep
#SBATCH --partition=cpu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /zfsauton/scratch/yixiz/DMSR/sr2_baseline/logs/slurm-sr2_subrep-%j.out

set -euo pipefail

REPO=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project
cd "${SLURM_SUBMIT_DIR:-$REPO}" 2>/dev/null || cd "$REPO"
export PATH=/zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin:$PATH
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

# Positional args of the form VAR=value override the defaults below
# (never use `sbatch --export` on this cluster -- it requeues the job held).
for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) [ -f "$arg" ] && . "$arg" ;;
  esac
done

# Every knob is SR2_-prefixed on purpose: the cluster login environment already
# exports generic names (SCRATCH=/home/scratch/$USER, and possibly OUT/STAGE), and
# a `${SCRATCH:-default}` silently inherits those and points at the wrong tree.
SR2_ROOT=${SR2_ROOT:-/zfsauton/scratch/yixiz/DMSR/sr2_baseline}
SR2_HALOS_ROOT=${SR2_HALOS_ROOT:-$SR2_ROOT/stage1/halos}
SR2_OUT=${SR2_OUT:-$SR2_ROOT/stage1/subhalo_report}
SR2_BOX=${SR2_BOX:-set12}
SR2_SEEDS=${SR2_SEEDS:-0,1,2,3,4,5,6,7}
SR2_MATCH_SEEDS=${SR2_MATCH_SEEDS:-0}
SR2_STAGE=${SR2_STAGE:-both}

echo "SR2_ROOT=$SR2_ROOT"
echo "SR2_HALOS_ROOT=$SR2_HALOS_ROOT"
echo "SR2_OUT=$SR2_OUT"

mkdir -p "$SR2_ROOT/logs" "$SR2_OUT"

# Gate: the catalogs must already exist. Exit 0 on a failed gate so any
# dependent job reports "gate failed" rather than DependencyNeverSatisfied.
if [ ! -d "$SR2_HALOS_ROOT/$SR2_BOX/hr/hr_rockstar" ]; then
  echo "GATE FAILED: no HR Rockstar catalog at $SR2_HALOS_ROOT/$SR2_BOX/hr/hr_rockstar"
  echo "Run scripts/slurm/sr2_stage1_boxes.sbatch first."
  exit 0
fi

echo "=== SR2 subhalo report: box=$SR2_BOX seeds=$SR2_SEEDS match=$SR2_MATCH_SEEDS stage=$SR2_STAGE ==="
python scripts/sr2/subhalo_report.py \
  --halos-root "$SR2_HALOS_ROOT" \
  --box "$SR2_BOX" \
  --seeds "$SR2_SEEDS" \
  --match-seeds "$SR2_MATCH_SEEDS" \
  --out "$SR2_OUT" \
  --stage "$SR2_STAGE"

echo "=== done ==="
ls -la "$SR2_OUT" "$SR2_OUT/figures" 2>/dev/null || true
python -c "import json; print(json.dumps(json.load(open('$SR2_OUT/summary.json')), indent=2))"
