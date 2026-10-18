#!/usr/bin/env python
"""Is a candidate box in HR's frame? Find the transform that maps its hosts onto HR's.

Every HR-referenced comparison -- the in-host SUBFIND / Rockstar counts at a
named HR host, host-centred profiles at HR's centres, splice seams -- assumes
the candidate box shares HR's coordinates. The placement forensics found the
m2m e77 boxes match ZERO of 57 massive HR hosts within 0.3 R_vir (offset 1.5-2.9
R_vir), while frozen SR2 matches 32%. A model that takes the LR box as input
cannot misplace every massive host by an LR cell unless its output frame is
shifted, permuted or flipped. This finds which.

For each of the 48 axis permutation x reflection transforms, the candidate's
host positions are transformed, each massive HR host is paired with its
nearest candidate host, and the pair offsets are histogrammed on the HR cell
grid. The peak is the frame offset; the score is the fraction of HR hosts
matched within ``--match-rvir`` R_vir after removing it. Catalog geometry
only, seconds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from placement_forensics import min_image, read_rockstar  # noqa: E402

BOX = 100.0
CELL = BOX / 512


def transforms():
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((False, True), repeat=3):
            yield perm, flips


def apply(pos: np.ndarray, perm, flips) -> np.ndarray:
    out = pos[:, list(perm)].copy()
    for a, f in enumerate(flips):
        if f:
            out[:, a] = BOX - out[:, a]
    return np.mod(out, BOX)


def best_offset(hr_pos, hr_rv, cand_pos, search: float, match_rvir: float):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.mod(cand_pos, BOX), boxsize=BOX)
    dist, k = tree.query(np.mod(hr_pos, BOX))
    raw = min_image(cand_pos[k] - hr_pos)
    near = np.linalg.norm(raw, axis=1) < search
    if near.sum() < 5:
        return None
    q = np.round(raw[near] / CELL).astype(np.int64)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    peak = keys[np.argmax(counts)] * CELL
    # score after shifting the candidate by -peak
    d2, k2 = tree.query(np.mod(hr_pos + peak, BOX))
    ok = d2 < match_rvir * hr_rv
    matched = float(np.mean(ok))
    # sub-cell refinement: the raw offset of the pairs the peak shift matched
    raw2 = min_image(cand_pos[k2[ok]] - hr_pos[ok])
    return {"offset_mpc_h": peak.tolist(), "offset_cells": (peak / CELL).round().tolist(),
            "peak_count": int(counts.max()), "n_near": int(near.sum()),
            "matched_frac": matched,
            "matched_frac_no_shift": float(np.mean(dist < match_rvir * hr_rv)),
            "matched_median_offset_mpc_h": (np.median(raw2, axis=0).tolist()
                                            if ok.any() else None),
            "matched_p16_p84_mpc_h": (np.percentile(raw2, [16, 84], axis=0).tolist()
                                      if ok.any() else None),
            "n_matched": int(ok.sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hr-catalog", required=True)
    ap.add_argument("--cand", required=True, help="comma list name=halos.ascii")
    ap.add_argument("--hr-min-num-p", type=int, default=20000)
    ap.add_argument("--cand-min-num-p", type=int, default=2000)
    ap.add_argument("--search", type=float, default=4.0, help="Mpc/h")
    ap.add_argument("--match-rvir", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    hr = read_rockstar(a.hr_catalog)
    hs = (hr["parent_id"] < 0) & (hr["num_p"] >= a.hr_min_num_p)
    hr_pos, hr_rv = hr["pos"][hs], hr["rvir"][hs] / 1000.0
    print(f"{hs.sum()} HR hosts >= {a.hr_min_num_p}p (whole box)")
    res = {}
    for item in a.cand.split(","):
        name, path = item.split("=", 1)
        if path.endswith(".npz"):
            # an LR host table (build_lr_host_channel): host centres only, already
            # gated on mass, so no num_p cut applies
            cpos = np.asarray(np.load(path)["host_pos_mpc_h"], dtype=np.float64)
        else:
            c = read_rockstar(path)
            cs = (c["parent_id"] < 0) & (c["num_p"] >= a.cand_min_num_p)
            cpos = c["pos"][cs]
        rows = []
        for perm, flips in transforms():
            r = best_offset(hr_pos, hr_rv, apply(cpos, perm, flips), a.search, a.match_rvir)
            if r is None:
                continue
            r.update({"perm": list(perm), "flips": list(flips)})
            rows.append(r)
        rows.sort(key=lambda r: -r["matched_frac"])
        ident = next(r for r in rows if r["perm"] == [0, 1, 2] and not any(r["flips"]))
        res[name] = {"identity": ident, "best": rows[0], "top5": rows[:5]}
        b = rows[0]
        print(f"{name:16s} identity: offset {ident['offset_cells']} cells, matched "
              f"{ident['matched_frac_no_shift']:.2f} -> {ident['matched_frac']:.2f} after shift | "
              f"best: perm {b['perm']} flips {b['flips']} offset {b['offset_cells']} "
              f"cells, matched {b['matched_frac']:.2f}", flush=True)
        mo = ident["matched_median_offset_mpc_h"]
        if mo is not None:
            print(f"{'':16s} identity, matched pairs (n {ident['n_matched']}): median offset "
                  f"cand-HR = ({mo[0]:+.3f}, {mo[1]:+.3f}, {mo[2]:+.3f}) Mpc/h = "
                  f"({mo[0] / CELL:+.2f}, {mo[1] / CELL:+.2f}, {mo[2] / CELL:+.2f}) cells",
                  flush=True)
    write_json(Path(a.out), {"hr_catalog": a.hr_catalog, "n_hr_hosts": int(hs.sum()),
                             "results": res})
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
