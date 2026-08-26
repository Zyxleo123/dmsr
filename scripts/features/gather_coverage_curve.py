#!/usr/bin/env python
"""How high can the ceiling go? Lagrangian coverage vs. recoverable subhalos.

``docs/sr2_member_gather.md`` section 6.1 measured two ceilings on the same four
tiles, and they are ceilings of two *different* kinds:

* **151 of 154** supervised targets -- essentially saturated. Nothing to raise.
* **227 of HR's 506** subhalos inside ``R_vir`` -- 0.449, and section 8.1 of
  ``docs/sr2_gather_finetune.md`` reads that number off the coverage: the four
  trained tiles hold **42.4%** of the host's Lagrangian sites, so the other
  subhalos are built from material the splice never replaces.

So the ``R_vir`` ceiling is not a property of the objective. It is a property of
**which tiles are trained**, and this script measures that function without
running Rockstar 8 more times. Two knobs, and they raise different bounds:

``--n-tiles``
    raises the **splice ceiling**: how many of HR's ``R_vir`` subhalos have their
    Lagrangian material inside the trained tiles at all.

``min_num_p``
    raises the **supervised** fraction of that ceiling. It does not move the
    ceiling itself. It matters because the two populations are badly mismatched
    today: the gate counts subhalos at ``>= 50p`` and the loss supervises at
    ``>= 200p``, and only **151 of the 506** are ``>= 200p``. Roughly seven in
    ten objects the gate scores are ones nothing in the loss ever asks for.

What is measured
----------------
One grouped pass over the box's subhalos gives, per subhalo, its full sparse
occupancy over Lagrangian tiles (:func:`subhalo_gather.subhalo_home_tiles` with
``return_occupancy``, so the home-tile and live-fraction definitions here are
the *same code* the loss selects with). Then for each rung of a tile ladder:

* ``host_site_coverage``  -- the 42.4% number, generalised.
* ``live_ge_*``           -- of HR's ``R_vir`` subhalos, how many have at least
  that fraction of their member particles inside the trained tiles. **These are
  the predicted splice ceiling.**
* ``supervised``          -- how many sets the loss would actually get, at each
  ``min_num_p``, under the real cuts (home tile trained, purity, live fraction).
* cost -- free parameters, member particles, and the ``sum N^2`` that the pair
  sums in the loss are linear in.

The predictors are calibrated, not asserted
-------------------------------------------
The 4-tile rung has a **measured** answer: 227. The script prints every
predictor against it, so the curve is only trusted at the rung where one of them
reproduces the number that was actually measured. Nothing here replaces the
Rockstar gate; it says which rung is worth spending one on.

    python scripts/features/gather_coverage_curve.py --box set8 --host-id 271800
    python scripts/features/gather_coverage_curve.py --from-json <coverage.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, paths, write_json  # noqa: E402
# One definition of "the R_vir population" and of the particle bins: this is the
# module the gate itself counts with.
from compare_gather_catalog import (  # noqa: E402
    PBINS, count_by_bin, subhalos_within,
)
from overfit_host_mse import NG_HR, TILE, _catalog, _owner, host_tiles  # noqa: E402

from cosmo_sr.eval.particle_identity import build_owner_index  # noqa: E402
from cosmo_sr.features.subhalo_gather import subhalo_home_tiles  # noqa: E402

LIVE_LEVELS = (0.9, 0.7, 0.5)
#: The one rung with a measured Rockstar answer, from
#: ``docs/sr2_gather_finetune.md`` section 8.1 / ``sr2_member_gather.md`` 6.1.
MEASURED = {"n_tiles": 4, "subhalos_in_rvir": 227, "host_site_coverage": 0.424}


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def tile_ranking(box: str, args) -> tuple[dict, list, np.ndarray]:
    """``(host meta, tiles ranked by host sites, their site counts)``.

    Ranking is :func:`overfit_host_mse.host_tiles` asked for *every* tile rather
    than the top four, so the ladder below walks the same ordering every run in
    this line was trained on -- rung ``n`` is exactly ``--n-tiles n``.
    """
    n_all = (NG_HR // TILE) ** 3
    ns = SimpleNamespace(host_id=int(args.host_id),
                         n_candidate_hosts=int(args.n_candidate_hosts),
                         n_tiles=n_all)
    meta, ranked, _ = host_tiles(box, ns)
    counts = np.asarray(meta["train_tile_member_sites"], dtype=np.int64)
    return meta, [int(t) for t in ranked], counts


def live_fractions(home: dict, tiles: np.ndarray) -> np.ndarray:
    """Per-subhalo fraction of member particles inside ``tiles``.

    Exactly the quantity :func:`member_gather.build_member_sets` computes as
    ``len(keep) / len(ids)`` and cuts on with ``min_live_frac`` -- here for the
    whole box at once, out of the occupancy the same grouped pass already built.
    """
    n = int(home["n_sites"].size)
    if n == 0:
        return np.zeros(0)
    m = np.isin(home["occ_tile"], tiles)
    inside = np.bincount(home["occ_row"][m],
                         weights=home["occ_count"][m].astype(np.float64),
                         minlength=n)
    return inside / np.maximum(home["n_sites"], 1).astype(np.float64)


def subhalo_ranking(home: dict, in_rvir: np.ndarray, n_all: int) -> list:
    """Tiles ranked by how much ``R_vir`` **subhalo** material they hold.

    The trained tiling is ranked by the *host's* Lagrangian sites, which is the
    right choice for supervising the host and is not obviously the right one for
    maximising recovered subhalos: a satellite's material sits where it sits.
    This ordering is the alternative, reported beside the default so the cost of
    the default is visible. Nothing trains on it unless ``--tiles`` is passed
    explicitly to ``free_field_gather.py``.
    """
    m = in_rvir[home["occ_row"]]
    w = np.bincount(home["occ_tile"][m],
                    weights=home["occ_count"][m].astype(np.float64),
                    minlength=n_all)
    return [int(t) for t in np.argsort(-w)]


def rung(home: dict, live: np.ndarray, in_rvir: np.ndarray, args,
         tiles: list, coverage: float) -> dict:
    """Every bound, and the cost of reaching for it, at one tile count."""
    trained = np.isin(home["tile"], np.asarray(tiles, dtype=np.int64))
    pure = home["purity"] >= float(args.min_purity)
    alive = live >= float(args.min_live_frac)
    num_p = home["num_p"]

    row = {"n_tiles": len(tiles), "tiles": tiles,
           "host_site_coverage": coverage,
           "delta_params": len(tiles) * 6 * TILE ** 3}

    # --- the ceiling: what the splice could possibly rebuild -----------------
    ceil = {"n_rvir_total": int(in_rvir.sum())}
    for lv in LIVE_LEVELS:
        ceil[f"live_ge_{lv:g}"] = int((in_rvir & (live >= lv)).sum())
    # The expectation-style reading: a subhalo half of whose material is
    # replaced is half a subhalo. Included because it is the predictor that does
    # not need a threshold chosen after seeing the answer.
    ceil["sum_live"] = float(live[in_rvir].sum())
    row["ceiling_rvir"] = ceil

    # --- the supervision: what the loss would actually be told about ---------
    sup = {}
    for mp in args.min_num_p_ladder:
        keep = trained & pure & alive & (num_p >= int(mp))
        n_sites = home["n_sites"][keep].astype(np.float64)
        sup[str(int(mp))] = {
            "n_sets": int(keep.sum()),
            "n_sets_in_rvir": int((keep & in_rvir).sum()),
            "member_particles": int(n_sites.sum()),
            # The loss's pair sums are O(N^2) per set; this is what a run's step
            # time is linear in, and it is why lowering min_num_p is cheaper
            # than it looks (the added sets are the small ones).
            "sum_n_squared": float((n_sites ** 2).sum()),
            "bg_particles": int(keep.sum()) * int(args.bg_k),
        }
    row["supervised"] = sup
    return row


def measure(args) -> dict:
    banner(f"gather coverage curve: {args.box}")
    cat = _catalog(args.box, "hr")
    oidx = build_owner_index(_owner(args.box, "hr"))

    meta, ranked, counts = tile_ranking(args.box, args)
    total_sites = int(meta["n_member_sites"])
    print(f"  host {meta['halo_id']} logM={meta['log_mvir']:.2f} "
          f"num_p={meta['num_p']} sites={total_sites}")
    cum = np.cumsum(counts) / max(total_sites, 1)

    hrow = np.flatnonzero(cat.ids == int(args.host_id))
    if hrow.size == 0:
        raise SystemExit(f"host {args.host_id} not in the {args.box} HR catalog")
    hrow = int(hrow[0])
    rvir = float(cat.rvir[hrow]) / 1000.0 * float(args.radius_factor)
    print(f"  R_vir {rvir:.3f} Mpc/h, counting subhalos >= {args.min_p}p")

    # One grouped pass, at the LOOSEST particle cut: every higher min_num_p is a
    # mask on `num_p` below, so the expensive member loop runs exactly once.
    print("  grouping subhalo members by Lagrangian tile ...", flush=True)
    home = subhalo_home_tiles(cat, oidx, ng_hr=NG_HR, tile_hr=TILE,
                              min_num_p=int(args.min_p), return_occupancy=True)
    home["num_p"] = cat.num_p[home["row"]]
    print(f"  {home['row'].size} resolved subhalos >= {args.min_p}p in the box")

    rvir_rows = subhalos_within(cat, cat.pos[hrow], rvir, min_p=int(args.min_p))
    in_rvir = np.isin(home["row"], rvir_rows)
    print(f"  {rvir_rows.size} of them inside R_vir "
          f"({int(in_rvir.sum())} carry member ids)")

    # The mass-cut mismatch, stated once: the gate counts this population, the
    # loss supervises only part of it.
    bins = count_by_bin(cat, rvir_rows)
    for mp in args.min_num_p_ladder:
        bins[f">= {int(mp)}p"] = int((cat.num_p[rvir_rows] >= int(mp)).sum())

    n_all = (NG_HR // TILE) ** 3
    host_counts = np.zeros(n_all, dtype=np.int64)
    host_counts[np.asarray(ranked)] = counts
    alt = subhalo_ranking(home, in_rvir, n_all)

    def ladder(order: list, name: str) -> list:
        print(f"  ladder, tiles ranked by {name}:", flush=True)
        rows = []
        for n in args.tile_ladder:
            n = int(n)
            if n > len(order):
                continue
            tiles = order[:n]
            cover = float(host_counts[np.asarray(tiles)].sum()
                          / max(total_sites, 1))
            rows.append(rung(home, live_fractions(home, np.asarray(tiles)),
                             in_rvir, args, tiles, cover))
            print(f"    n_tiles {n:3d}  host coverage {cover:.3f}  "
                  f"ceiling(live>=0.5) "
                  f"{rows[-1]['ceiling_rvir']['live_ge_0.5']:4d}  "
                  f"sets@200p {rows[-1]['supervised']['200']['n_sets']:5d}",
                  flush=True)
        return rows

    rows = ladder(ranked, "the host's Lagrangian sites (what --n-tiles trains)")
    alt_rows = ladder(alt, "R_vir subhalo material (an alternative, --tiles)")

    return {"box": args.box, "host": meta, "host_id": int(args.host_id),
            "rvir_mpc_h": rvir, "min_p": int(args.min_p),
            "min_purity": float(args.min_purity),
            "min_live_frac": float(args.min_live_frac),
            "n_subhalos_box": int(home["row"].size),
            "rvir_population": bins,
            "ranked_tiles": ranked[:max(args.tile_ladder)],
            "subhalo_ranked_tiles": alt[:max(args.tile_ladder)],
            "cumulative_coverage": cum[:max(args.tile_ladder)].tolist(),
            "measured_reference": MEASURED,
            "rungs": rows, "rungs_subhalo_ranked": alt_rows}


# --------------------------------------------------------------------------- #
# Reading it
# --------------------------------------------------------------------------- #
def report(res: dict) -> None:
    print("\n  HR subhalos within R_vir, by particle count "
          "(what the gate scores):")
    pop = res["rvir_population"]
    for lo, hi in PBINS:
        k = f"{lo}-{hi if hi < 10 ** 9 else 'inf'}p"
        print(f"    {k:>12s} {pop.get(k, 0):5d}")
    print(f"    {'total':>12s} {pop.get('total', 0):5d}")
    for k, v in pop.items():
        if k.startswith(">="):
            print(f"    {k:>12s} {v:5d}   <- supervised at this cut")

    def table(rows: list, title: str) -> None:
        print(f"\n  the ceiling by tile count, {title}:")
        print(f"    {'n_tiles':>7s} {'cover':>6s} {'>=0.9':>6s} {'>=0.7':>6s} "
              f"{'>=0.5':>6s} {'sum':>7s} {'sets@200p':>9s} {'sets@50p':>9s} "
              f"{'params':>9s}")
        for r in rows:
            c = r["ceiling_rvir"]
            sup = r["supervised"]
            print(f"    {r['n_tiles']:7d} {r['host_site_coverage']:6.3f} "
                  f"{c['live_ge_0.9']:6d} {c['live_ge_0.7']:6d} "
                  f"{c['live_ge_0.5']:6d} {c['sum_live']:7.1f} "
                  f"{sup.get('200', {}).get('n_sets', 0):9d} "
                  f"{sup.get('50', {}).get('n_sets', 0):9d} "
                  f"{r['delta_params'] / 1e6:8.1f}M")

    table(res["rungs"], "tiles ranked by the host's sites (what --n-tiles does)")
    alt = res.get("rungs_subhalo_ranked") or []
    if alt:
        table(alt, "tiles ranked by R_vir subhalo material (--tiles)")
        # The only reason the alternative is here: does the default ranking
        # leave ceiling on the table at equal cost?
        print("\n  same tile count, the two orderings (ceiling at live>=0.5):")
        by_n = {r["n_tiles"]: r for r in alt}
        for r in res["rungs"]:
            o = by_n.get(r["n_tiles"])
            if o is None:
                continue
            a, b = r["ceiling_rvir"]["live_ge_0.5"], o["ceiling_rvir"]["live_ge_0.5"]
            print(f"    {r['n_tiles']:7d} host-ranked {a:5d}   "
                  f"subhalo-ranked {b:5d}   {b - a:+5d}")

    # --- calibration: which predictor reproduces the number we measured? -----
    ref = res["measured_reference"]
    hit = [r for r in res["rungs"] if r["n_tiles"] == ref["n_tiles"]]
    print(f"\n  calibration at n_tiles={ref['n_tiles']}, where Rockstar "
          f"measured {ref['subhalos_in_rvir']} subhalos in R_vir:")
    if not hit:
        print("    (that rung is not on the ladder -- add it)")
        return
    c = hit[0]["ceiling_rvir"]
    for name, val in [(f"live>={lv:g}", c[f"live_ge_{lv:g}"])
                      for lv in LIVE_LEVELS] + [("sum live", c["sum_live"])]:
        err = val / max(ref["subhalos_in_rvir"], 1) - 1.0
        print(f"    {name:>10s} {val:8.1f}   {err:+6.1%} vs measured")
    print(f"    host site coverage {hit[0]['host_site_coverage']:.3f} "
          f"(docs say {ref['host_site_coverage']:.3f})")
    print("\n  Read the curve only through whichever predictor lands on 227 "
          "here; the\n  others are recorded so the choice is visible rather "
          "than fitted.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--box", default="set8")
    ap.add_argument("--host-id", type=int, default=271800)
    ap.add_argument("--n-candidate-hosts", type=int, default=8)
    ap.add_argument("--tile-ladder", default="1,2,4,6,8,12,16,24,32,48,64",
                    help="tile counts to evaluate, in the host-site ranking")
    ap.add_argument("--min-num-p-ladder", default="200,100,50",
                    help="supervision cuts to cost out at each rung")
    ap.add_argument("--min-p", type=int, default=50,
                    help="particle cut of the R_vir population the gate counts")
    ap.add_argument("--min-purity", type=float, default=0.5)
    ap.add_argument("--min-live-frac", type=float, default=0.5)
    ap.add_argument("--radius-factor", type=float, default=1.0)
    ap.add_argument("--bg-k", type=int, default=4096)
    ap.add_argument("--label", default="")
    ap.add_argument("--from-json", default="",
                    help="re-print the table from a finished run, no recompute")
    args = ap.parse_args()

    if args.from_json:
        report(json.loads(Path(args.from_json).read_text()))
        return 0

    args.tile_ladder = [int(x) for x in args.tile_ladder.split(",") if x]
    args.min_num_p_ladder = [int(x) for x in args.min_num_p_ladder.split(",")
                             if x]
    res = measure(args)
    out = paths.subdir("free_field_gather",
                       f"{args.box}_h{args.host_id}{args.label}", create=True)
    p = out / "coverage_curve.json"
    write_json(p, res)
    report(res)
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
