#!/usr/bin/env python
"""Did the smoke run recover the clumps it was given? A yes/no on the chain.

The smoke snapshot plants a known number of self-bound clumps at known centres
(:mod:`subfind_snapshot`), so this is the one place in the SUBFIND pipeline
where the right answer is known in advance. That makes it the only check that
can separate "the chain runs" from "the chain works": a wrong header, a
mis-scaled velocity block, a parameter file that silently disables SUBFIND, or a
catalogue parsed with the wrong rank convention all produce a clean exit and an
empty or nonsense catalogue.

Runs in the GADGET-4 environment, like the parser it follows -- numpy only.

    python scripts/features/subfind_smoke_verdict.py --run-dir $SUBFIND_ROOT/smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tol-sigma", type=float, default=6.0,
                    help="a clump counts as recovered if a SUBFIND satellite "
                         "sits within this many of its own Gaussian widths")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="and keeps at least this fraction of its particles")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    planted = manifest.get("planted_clumps")
    if not planted:
        raise SystemExit(f"{run}/manifest.json has no planted_clumps: this is "
                         "not a smoke run")
    npz = run / "subfind_catalog.npz"
    if not npz.is_file():
        raise SystemExit(f"no {npz}; the SUBFIND run produced no catalogue")

    z = np.load(npz)
    # EVERY subhalo, central or satellite. In the smoke geometry the planted
    # clumps sit in a uniform background rather than inside a host, so each one
    # is its own FOF group and SUBFIND reports it as that group's CENTRAL
    # (SubhaloRankInGr == 0). Matching only satellites would score a perfect run
    # as zero recovered -- the filter that is right for a cluster is wrong here.
    sat = np.asarray(z["is_satellite"]).astype(bool)
    pos = np.asarray(z["SubhaloPos"]).astype(np.float64)
    length = np.asarray(z["SubhaloLen"]).astype(np.int64)
    box = float(manifest["boxsize_mpc_h"])

    rows, recovered = [], 0
    for c in planted:
        centre = np.asarray(c["centre_mpc_h"], dtype=np.float64)
        tol = float(args.tol_sigma) * float(c["sigma_mpc_h"])
        if pos.shape[0] == 0:
            rows.append({**c, "found": False, "reason": "no subhalos at all"})
            continue
        d = (pos - centre[None, :] + 0.5 * box) % box - 0.5 * box
        r = np.sqrt((d ** 2).sum(axis=1))
        j = int(np.argmin(r))
        frac = float(length[j]) / float(c["n_particles"])
        ok = bool(r[j] <= tol and frac >= float(args.min_frac))
        recovered += int(ok)
        rows.append({**c, "found": ok, "nearest_sep_mpc_h": float(r[j]),
                     "tol_mpc_h": tol, "subfind_len": int(length[j]),
                     "len_fraction": frac})

    verdict = {
        "run_dir": str(run),
        "n_planted": len(planted),
        "n_recovered": recovered,
        "n_subhalos_found": int(length.size),
        "n_centrals": int((~sat).sum()),
        "n_satellites": int(sat.sum()),
        "pass": bool(recovered == len(planted)),
        "clumps": rows,
    }
    (run / "smoke_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)
    print(flush=True)
    if verdict["pass"]:
        print(f"=== PASS: SUBFIND recovered all {len(planted)} planted clumps. "
              "The chain -- snapshot, parameters, finder, parser -- is sound.",
              flush=True)
        return 0
    print(f"=== FAIL: {recovered} of {len(planted)} planted clumps recovered. "
          "Fix this before running the 512^3 box; a real run cannot be "
          "interpreted through a chain that cannot find what it was handed.",
          flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
