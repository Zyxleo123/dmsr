#!/usr/bin/env python
"""Validate host matching: catalog vs itself after a known periodic translation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", required=True, help="Rockstar halos_*.ascii")
    ap.add_argument("--boxsize", type=float, default=100.0)
    ap.add_argument("--shift", default="0.05,-0.03,0.04",
                    help="comma-separated Mpc/h translation (≪ host spacing)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from cosmo_sr.eval.rockstar import load_rockstar_ascii
    from cosmo_sr.eval.halo_match import self_match_after_translation

    shift = tuple(float(x) for x in args.shift.split(","))
    cat = load_rockstar_ascii(args.catalog)
    res = self_match_after_translation(cat, args.boxsize, shift_mpc_h=shift)
    ok = res["correct_id_rate"] >= 0.99 and res["match_rate"] >= 0.99
    res["pass"] = bool(ok)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
    if not ok:
        raise SystemExit(
            f"FAIL: match_rate={res['match_rate']:.4f} "
            f"correct_id_rate={res['correct_id_rate']:.4f}"
        )
    print("PASS: self-translation host match ≥99%")


if __name__ == "__main__":
    main()
