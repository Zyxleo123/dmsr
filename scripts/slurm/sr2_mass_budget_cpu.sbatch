#!/bin/bash
# Where did the mass missing from SR2 halos go? (CPU only)
#
# Reads the cached SR2 base displacement fields and the paired HR fields,
# CIC-deposits both on a common mesh, and compares the density PDF, the
# per-particle HR->SR2 density migration matrix, and the collapsed-mass budget
# from the frozen Rockstar catalogs. No model, no GPU.
#
#   sbatch scripts/slurm/sr2_mass_budget_cpu.sbatch
#   sbatch scripts/slurm/sr2_mass_budget_cpu.sbatch SR2_SEEDS=0,1,2   # scatter band
#   sbatch scripts/slurm/sr2_mass_budget_cpu.sbatch SR2_STAGE=plot    # AFTER analyze
#
# The three lines above are sequential, not concurrent. Every invocation writes
# metrics.npz/summary.json into the same SR2_OUT, so two analyze jobs submitted
# together race and the loser's results are overwritten; pass a distinct
# SR2_OUT to run variants side by side.
#
# All overrides are SR2_-prefixed and passed as positional args (never
# `sbatch --export`: on this cluster it requeues the job held).
#
#SBATCH --job-name=sr2_massbud
#SBATCH --partition=cpu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /zfsauton/scratch/yixiz/DMSR/sr2_baseline/logs/slurm-sr2_massbud-%j.out

set -euo pipefail

REPO=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project
cd "${SLURM_SUBMIT_DIR:-$REPO}" 2>/dev/null || cd "$REPO"
export PATH=/zfsauton/scratch/yixiz/miniconda3/envs/pjm/bin:$PATH
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

for arg in "$@"; do
  case "$arg" in
    *=*) export "${arg?}" ;;
    *) [ -f "$arg" ] && . "$arg" ;;
  esac
done

# numpy would otherwise start one BLAS thread per core on an 88-core node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

SR2_ROOT=${SR2_ROOT:-/zfsauton/scratch/yixiz/DMSR/sr2_baseline}
SR2_HALOS_ROOT=${SR2_HALOS_ROOT:-$SR2_ROOT/stage1/halos}
SR2_HR_ROOT=${SR2_HR_ROOT:-/zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr}
SR2_SR_CACHE=${SR2_SR_CACHE:-/zfsauton/scratch/yixiz/DMSR/dmsr_reward/cache/sr2_base}
SR2_CAT_CACHE=${SR2_CAT_CACHE:-$SR2_ROOT/stage1/subhalo_report/cache}
SR2_OUT=${SR2_OUT:-$SR2_ROOT/stage1/mass_budget}
SR2_BOX=${SR2_BOX:-set12}
SR2_SEEDS=${SR2_SEEDS:-0}
SR2_CAT_SEEDS=${SR2_CAT_SEEDS:-0,1,2,3,4,5,6,7}
SR2_CHUNK=${SR2_CHUNK:-32}
SR2_STAGE=${SR2_STAGE:-both}

mkdir -p "$SR2_ROOT/logs" "$SR2_OUT"

# Gates. Exit 0 on a failed gate so any dependent job reports "gate failed"
# rather than stranding in DependencyNeverSatisfied.
if [ ! -f "$SR2_HR_ROOT/$SR2_BOX.npy" ]; then
  echo "GATE FAILED: no HR field at $SR2_HR_ROOT/$SR2_BOX.npy"
  echo "Run scripts/preprocess_paired.py first."
  exit 0
fi
_first_seed=${SR2_SEEDS%%,*}
if ! ls "$SR2_SR_CACHE/${SR2_BOX}_seed${_first_seed}_"*.npy >/dev/null 2>&1; then
  echo "GATE FAILED: no cached SR2 base field for $SR2_BOX seed $_first_seed in $SR2_SR_CACHE"
  echo "Run: sbatch scripts/slurm/cache_sr2_base.sbatch BOXES=$SR2_BOX"
  exit 0
fi
if [ ! -d "$SR2_HALOS_ROOT/$SR2_BOX/hr/hr_rockstar" ]; then
  echo "GATE FAILED: no HR Rockstar catalog at $SR2_HALOS_ROOT/$SR2_BOX/hr/hr_rockstar"
  exit 0
fi
# SR2_STAGE=plot only redraws; it needs a completed analyze pass to read.
if [ "$SR2_STAGE" = plot ] && [ ! -f "$SR2_OUT/metrics.npz" ]; then
  echo "GATE FAILED: SR2_STAGE=plot but no $SR2_OUT/metrics.npz"
  echo "Run the analyze stage first (submit with no SR2_STAGE override)."
  exit 0
fi

echo "=== SR2 mass budget: box=$SR2_BOX seeds=$SR2_SEEDS stage=$SR2_STAGE ==="
echo "    out=$SR2_OUT"
python -u scripts/sr2/mass_budget.py \
  --box "$SR2_BOX" \
  --seeds "$SR2_SEEDS" \
  --catalog-seeds "$SR2_CAT_SEEDS" \
  --hr-root "$SR2_HR_ROOT" \
  --sr-cache-root "$SR2_SR_CACHE" \
  --halos-root "$SR2_HALOS_ROOT" \
  --catalog-cache "$SR2_CAT_CACHE" \
  --out "$SR2_OUT" \
  --chunk "$SR2_CHUNK" \
  --stage "$SR2_STAGE"

echo "=== done ==="
ls -la "$SR2_OUT" "$SR2_OUT/figures" 2>/dev/null || true
python -c "
import json; s=json.load(open('$SR2_OUT/summary.json'))
print(json.dumps({'headline': s['headline'], 'migration': s.get('migration', {}),
                  'collapsed': {k: v for k, v in s['collapsed'].items() if k != 'sr'}},
                 indent=2))"
