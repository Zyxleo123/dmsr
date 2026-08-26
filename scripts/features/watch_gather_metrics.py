#!/usr/bin/env python
"""Print a member-gather run's eval rows as a table. A LOG READER, not compute.

`metrics.jsonl` carries a full per-host block per eval, so the fields that decide
whether a run is working are unreadable by eye. This prints the ones section 11.6
says to watch, train and held out, with the FROZEN reference on the same axis:

    velhighk   velocity power above k_split, /HR. Frozen SR2 is 1.02x -- it has
               this right -- and all four earlier arms collapsed to 0.034-0.053.
               This is the number the velocity term was added to move.
    highk      displacement power above k_split, /HR (the historical scalar).
    bound      the gate's own criterion. The velocity term directly opposes the
               global-cooling shortcut `bound` was exploiting, so some cost here
               is the expected price, not a fault.

Usage:  python scripts/features/watch_gather_metrics.py <run_dir> [--last N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = (("bound_hard", "bound", 3), ("virial", "2T/|W|", 1),
        ("highk_ratio", "highk", 3), ("velhighk_ratio", "velhk", 4),
        ("vel_rms_ratio", "velrms", 3), ("low_k", "low_k", 4),
        ("centre_offset_radii", "dx_r", 2), ("r_rms_over_hr", "r/HR", 2))


def _fmt(row, key, nd):
    v = row.get(key)
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "--"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--last", type=int, default=0, help="only the last N rows")
    a = ap.parse_args()

    p = Path(a.run_dir) / "metrics.jsonl"
    if not p.exists():
        print(f"no metrics yet at {p}\n"
              f"  (the shakeout writes none -- it runs 0 steps and builds "
              f"pool.json only)")
        return 0
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if a.last and len(rows) > a.last + 1:
        # Keep step 0 as the frozen reference, then the tail. The guard matters:
        # `rows[:1] + rows[-N:]` prints step 0 twice while a run has only it.
        rows = rows[:1] + rows[-a.last:]

    for split in ("train", "holdout"):
        print(f"\n=== {split}")
        print(f"{'step':>6} " + " ".join(f"{lab:>7}" for _, lab, _ in COLS))
        for r in rows:
            d = r.get(split) or {}
            if not d.get("n_hosts"):
                continue
            print(f"{r['step']:>6} "
                  + " ".join(f"{_fmt(d, k, n):>7}" for k, _, n in COLS))

    last = rows[-1]
    for split in ("train", "holdout"):
        b = (last.get(split) or {}).get("highk_bands")
        if b:
            print(f"\n=== {split} high-k per band, step {last['step']} "
                  f"(median over hosts)")
            print("   k  " + " ".join(f"{k:>7.2f}" for k in b["k"]))
            print("ratio " + " ".join(f"{v:>7.3f}" for v in b["ratio"]))

    print("\nreference, measured on held-out set9 (job 36394):")
    print("  velhk  frozen SR2 1.023x HR  |  the four earlier arms 0.034-0.053x")
    print("  highk  frozen SR2 0.46x      |  self 1.70x (worst host 3.87x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
