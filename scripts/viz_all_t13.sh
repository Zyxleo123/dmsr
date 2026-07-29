#!/usr/bin/env bash
# Backfill per-run visualisations for every existing t13 run (both 64^3 and
# 128^3), writing figures directly under each run dir via dmsr_run_viz.py.
#
# New runs get these automatically at the end of training (train_dmsr.py); this
# script is only for the runs that finished before the hook existed.
#
# Usage:  bash scripts/viz_all_t13.sh [runs/dmsr]           # all t13_* runs
#         bash scripts/viz_all_t13.sh runs/dmsr t13_critic  # a name filter
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

RUNS_ROOT="${1:-runs/dmsr}"
FILTER="${2:-t13_}"

# map2map (SRS G_z0) baseline -- run ONCE into the shared figures dir (it does
# not depend on any t13 checkpoint). Skip with VIZ_SKIP_BASELINE=1.
if [[ "${VIZ_SKIP_BASELINE:-0}" != "1" ]]; then
  echo "===================================================================="
  echo "VIZ map2map baseline (SRS G_z0) -> runs/dmsr/figures/baseline_map2map"
  echo "===================================================================="
  python scripts/dmsr_baseline_viz.py || echo "!! baseline failed (continuing)"
fi

for d in "$RUNS_ROOT"/${FILTER}*/; do
  d="${d%/}"
  [[ -f "$d/config.yaml" && -f "$d/ckpt_best.pt" ]] || { echo "skip $d (no config/ckpt)"; continue; }
  echo "===================================================================="
  echo "VIZ $d"
  echo "===================================================================="
  python scripts/dmsr_run_viz.py --run "$d" || echo "!! $d failed (continuing)"
done
echo "all done."
