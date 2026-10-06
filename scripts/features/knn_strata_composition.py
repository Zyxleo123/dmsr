#!/usr/bin/env python
"""Is a self-defined DENSITY stratum a usable stand-in for the HR class mask?

Why this exists
---------------
``--knn-classes sub_50_200`` is the one place a Rockstar label reaches the kNN
term's gradient. It was not a preference: round 5b measured the pooled form
ranking the working arms BACKWARDS (rho +0.91 with subhalo count) because host
and unbound particles are ~95% of any tile interior, and masking the queries by
HR's owner class is what fixed it (rho -0.65).

``KnnConfig.strata`` proposes making the same split with no catalog: each field
sorts its own queries by its own ``r_{strata_rank}`` and the loss is charged on
the densest stratum. Two things about that are already MEASURED, in
``tests/features/test_knn_cdf.py``, and only one of them is good:

* charging every stratum equally is arithmetically the pooled loss in finer bins
  -- partitioning is not reweighting -- so the arm only means anything with
  ``strata_charge`` set (measured 0.726 dense-only against 0.0912 equal-weight);
* on the synthetic dilution fixture the densest stratum is **65-83% host
  particles**, and its subhalo fraction gets worse as the candidate's subhalo is
  made more diffuse.

The second is the problem. Host over-concentration is this line's characteristic
damage (`host-damage-is-mostly-reparenting`, `radial-profile-is-flat-in-host-mass`),
so a term charged on a host-dominated population could deepen exactly the defect
the gate keeps catching. Whether that synthetic number survives on the real box
-- where a 50-200p subhalo sits INSIDE a host and is barely denser than it in 3D,
which is why round 5 went to 6D at all -- is a property of the data, not of an
argument. This script measures it, and it gates the arm.

What it measures, per field and per tile
----------------------------------------
``composition``
    For each equal-count stratum of ``r_{strata_rank}``, the HR owner-class
    fractions of the particles in it. The headline numbers are the densest
    stratum's ``sub_50_200`` **purity** (its share of that stratum) and
    **recall** (the share of the region's ``sub_50_200`` particles it holds),
    both against the base rate, i.e. an ENRICHMENT. Enrichment ~1 means the
    density cut selects the deficit population no better than chance.

``agreement``
    The decision number. Per tile, the dense-only stratified loss and the
    class-conditioned ``sub_50_200`` loss are both computed against HR, in the
    same bins with the same hinge. If the density cut is a usable stand-in, the
    two must RANK TILES TOGETHER -- reported as a Spearman rho over tiles, per
    field. This is the direct question "would charging strata move the same
    tiles the label moves", and it does not care whether the cut is pure.

``sweep``
    Both of the above over ``--strata`` x ``--charge``, because "how many strata
    and which" is the arm's only real hyper-parameter and guessing it would be
    the ``k``-cap mistake again.

THE STOP RULE, fixed before the run
-----------------------------------
The arm gets GPU only if, on the frozen field, some (strata, charge) has BOTH

1. ``sub_50_200`` recall enrichment in the charged strata >= 2.0, and
2. Spearman rho between the dense-only and class-conditioned per-tile losses
   >= 0.5.

(1) alone is a cut that finds subhalos; (2) alone could be two statistics
agreeing because both are dominated by the same hosts. Neither is sufficient and
the pair is what the label was buying. If no configuration clears both, the
honest answer is that a density stratum is not a catalog-free substitute for the
class mask, the result is written anyway, and the GPU arm is not launched.

Positions only by default; ``--phase-lambda`` > 0 repeats the whole thing in 6D
``(x, v/lambda)``, which is the space round 5's pass was measured in and is
therefore the one the arm would actually train in.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import knn_cdf_diagnostic as v1  # noqa: E402
import knn_cdf_split_diagnostic as v2  # noqa: E402
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.features.knn_cdf import tile_rows_flat  # noqa: E402

CLASSES = v2.CLASSES
SUB = CLASSES.index("sub_50_200")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy.stats, ties averaged."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")

    def rank(v):
        order = v.argsort()
        r = np.empty(v.size, float)
        r[order] = np.arange(v.size, dtype=float)
        # Average ties so a constant column is not silently a perfect rank.
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def strata_groups(r_strat: np.ndarray, n_strata: int) -> list:
    """Equal-count row groups, dense first -- ``knn_cdf.strata_partition`` in numpy."""
    n = r_strat.size
    order = np.argsort(r_strat, kind="stable")
    edges = [(b * n) // n_strata for b in range(n_strata + 1)]
    return [order[a:b] for a, b in zip(edges[:-1], edges[1:])]


def unpaired_loss(zc: np.ndarray, zh: np.ndarray, n_q: int,
                  dead_zone: float) -> float:
    """``knn_cdf_loss`` in numpy: sorted, quantile-binned, squared dead-zone hinge."""
    if zc.shape[0] < n_q or zh.shape[0] < n_q:
        return float("nan")
    d = v1.quantile_curve(zc, n_q) - v1.quantile_curve(zh, n_q)
    return float(v2._hinge_sq(d, dead_zone).mean())


def tile_interior(pos: np.ndarray, rows: np.ndarray, tile_id: int, *,
                  buffer_cells: float, ng: int, tile: int, box: float
                  ) -> np.ndarray:
    """Rows of ``rows`` whose EULERIAN position is inside the buffered cube.

    The cube is centred on the tile's own particles (``periodic_centroid``'s
    argument: at z=0 a tile has moved about its own width), and the cut is in
    position, never on the Lagrangian lattice -- ``interior_queries``' rule, and
    this has to agree with it or the diagnostic measures a different population
    than the loss would.
    """
    cell = box / ng
    x = np.mod(pos[rows, :3], box)
    ang = x * (2 * np.pi / box)
    c = np.mod(np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
               * box / (2 * np.pi), box)
    d = x - c
    d -= box * np.round(d / box)
    half = 0.5 * tile * cell - float(buffer_cells) * cell
    if half <= 0:
        return rows[:0]
    return rows[np.abs(d).max(axis=1) <= half]


def measure_tile(z_field: np.ndarray, z_hr: np.ndarray, cls: np.ndarray,
                 strata: int, charge, n_q: int, dead_zone: float,
                 strat_col: int) -> dict:
    """Composition and the two competing losses for ONE tile of ONE field.

    ``z_field`` / ``z_hr`` are that tile's interior ``log r_k``, each on its own
    field's own interior draw; ``cls`` is the field's rows' HR owner class, used
    for the composition and the class-conditioned loss and NEVER for the strata.
    """
    gf = strata_groups(z_field[:, strat_col], strata)
    gh = strata_groups(z_hr[:, strat_col], strata)
    take = sorted(set(charge))
    comp = []
    for g in gf:
        c = cls[g]
        comp.append({name: float((c == i).mean()) for i, name in enumerate(CLASSES)})
    rows_f = np.concatenate([gf[j] for j in take])
    rows_h = np.concatenate([gh[j] for j in take])
    base = float((cls == SUB).mean())
    in_charged = float((cls[rows_f] == SUB).mean())
    n_sub_total = int((cls == SUB).sum())
    recall = (float((cls[rows_f] == SUB).sum()) / n_sub_total
              if n_sub_total else float("nan"))
    frac_charged = rows_f.size / max(z_field.shape[0], 1)
    sub_f = np.flatnonzero(cls == SUB)
    return {
        "n_interior": int(z_field.shape[0]),
        "composition": comp,
        "sub_50_200_base_rate": base,
        "sub_50_200_purity_charged": in_charged,
        # Enrichment, not the raw fraction: a rare class is rare everywhere and
        # only the ratio to its base rate says the cut did anything.
        "purity_enrichment": (in_charged / base) if base > 0 else float("nan"),
        "recall_charged": recall,
        "recall_enrichment": (recall / frac_charged) if frac_charged > 0
                             else float("nan"),
        "frac_charged": frac_charged,
        # The two losses being compared, same bins, same hinge, same tile.
        "loss_dense": unpaired_loss(z_field[rows_f], z_hr[rows_h], n_q, dead_zone),
        "loss_pooled": unpaired_loss(z_field, z_hr, n_q, dead_zone),
        "loss_class_sub": unpaired_loss(
            z_field[sub_f], z_hr[np.flatnonzero(cls == SUB)], n_q, dead_zone),
        "n_sub_interior": n_sub_total,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", required=True,
                    help="label=path; must include --hr-label and --gate-label")
    ap.add_argument("--hr-label", default="hr")
    ap.add_argument("--gate-label", default="frozen",
                    help="the field the stop rule is read on")
    ap.add_argument("--catalog", required=True, help="HR rockstar ascii")
    ap.add_argument("--owner", required=True, help="HR owner .npy")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 64, 128])
    ap.add_argument("--strata-rank", type=int, default=0,
                    help="0 = min(ks)")
    ap.add_argument("--strata", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--charge", nargs="+", default=["0", "0,1"],
                    help="comma-separated stratum indices per sweep point")
    ap.add_argument("--tiles", type=int, default=24)
    ap.add_argument("--queries", type=int, default=4096)
    ap.add_argument("--buffer-cells", type=float, default=8.0)
    ap.add_argument("--n-quantiles", type=int, default=16)
    ap.add_argument("--dead-zone", type=float, default=0.02)
    ap.add_argument("--phase-lambda", type=float, default=0.0)
    ap.add_argument("--region-json", default=None,
                    help="an arm field's .json: use only its spliced tiles")
    ap.add_argument("--min-recall-enrichment", type=float, default=2.0)
    ap.add_argument("--min-rho", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ks = np.asarray(sorted(set(a.ks)), dtype=int)
    k_max = int(ks.max())
    strat_k = int(a.strata_rank) or int(ks.min())
    if strat_k not in ks.tolist():
        raise SystemExit(f"--strata-rank {strat_k} not in --ks {ks.tolist()}")
    strat_col = int(np.flatnonzero(ks == strat_k)[0])
    charges = [tuple(int(j) for j in c.split(",")) for c in a.charge]
    ng, tile, box = v1.NG, v1.TILE, v1.BOX_MPC
    rng = np.random.default_rng(a.seed)
    fields = dict(f.partition("=")[::2] for f in a.fields)
    for need in (a.hr_label, a.gate_label):
        if need not in fields or not Path(fields[need]).exists():
            raise SystemExit(f"required field {need!r} missing: {fields.get(need)}")

    t0 = time.time()
    cls_all = v2.classify(np.load(a.owner, mmap_mode="r"),
                          load_rockstar_ascii(a.catalog))
    n_side = ng // tile
    if a.region_json:
        pool = sorted(int(t) for t in json.loads(
            Path(a.region_json).read_text())["tiles"])
    else:
        pool = list(range(n_side ** 3))
    tiles = sorted(rng.choice(pool, size=min(a.tiles, len(pool)),
                              replace=False).tolist())
    print(f"classes loaded, {len(tiles)} tiles of {len(pool)} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # Per field: one full-box tree, then every tile's interior queried against it.
    # The tree is the whole box, so there is NO boundary error to argue about --
    # this diagnostic must not inherit the estimator question round 2 settled.
    per_field = {}
    z_by_field = {}
    for label, path in fields.items():
        t1 = time.time()
        pts = v2.load_points(path, a.phase_lambda)
        tree = v2.make_tree(pts, box)
        rows_by_tile, z_by_tile = {}, {}
        for t in tiles:
            rows = tile_rows_flat(t, ng=ng, tile=tile)
            inside = tile_interior(pts, rows, t, buffer_cells=a.buffer_cells,
                                   ng=ng, tile=tile, box=box)
            if inside.size > a.queries:
                inside = np.sort(rng.choice(inside, a.queries, replace=False))
            if inside.size < max(a.n_quantiles, 1) * max(a.strata):
                continue
            d, _ = v2.rk_query(tree, pts[inside], inside, k_max)
            z_by_tile[t] = np.log(np.maximum(d[:, ks - 1], 1e-5))
            rows_by_tile[t] = inside
        z_by_field[label] = (z_by_tile, rows_by_tile)
        del pts, tree
        print(f"  {label}: {len(z_by_tile)} tiles measured "
              f"[{time.time() - t1:.0f}s]", flush=True)

    z_hr, rows_hr = z_by_field[a.hr_label]
    for label in fields:
        z_f, rows_f = z_by_field[label]
        sweep = {}
        for n_str in a.strata:
            for ch in charges:
                if max(ch) >= n_str:
                    continue
                key = f"strata{n_str}_charge{'-'.join(str(j) for j in ch)}"
                rows = []
                for t in sorted(set(z_f) & set(z_hr)):
                    m = measure_tile(z_f[t], z_hr[t], cls_all[rows_f[t]],
                                     n_str, ch, a.n_quantiles, a.dead_zone,
                                     strat_col)
                    m["tile"] = int(t)
                    rows.append(m)
                if not rows:
                    continue
                rho = spearman(np.array([r["loss_dense"] for r in rows]),
                               np.array([r["loss_class_sub"] for r in rows]))
                # A control: the POOLED loss against the same class loss. If the
                # dense cut's rho is no better than this, the agreement is not
                # coming from the density cut at all.
                rho_pooled = spearman(
                    np.array([r["loss_pooled"] for r in rows]),
                    np.array([r["loss_class_sub"] for r in rows]))

                def med(key_):
                    v = np.array([r[key_] for r in rows], float)
                    return float(np.nanmedian(v)) if v.size else float("nan")
                sweep[key] = {
                    "strata": n_str, "charge": list(ch), "n_tiles": len(rows),
                    "rho_dense_vs_class": rho,
                    "rho_pooled_vs_class_control": rho_pooled,
                    "median_recall_enrichment": med("recall_enrichment"),
                    "median_purity_enrichment": med("purity_enrichment"),
                    "median_frac_charged": med("frac_charged"),
                    "median_loss_dense": med("loss_dense"),
                    "median_loss_pooled": med("loss_pooled"),
                    "median_loss_class_sub": med("loss_class_sub"),
                    "median_composition_dense": {
                        c: float(np.median([r["composition"][0][c] for r in rows]))
                        for c in CLASSES},
                    "tiles": rows,
                }
        per_field[label] = sweep

    gate = per_field.get(a.gate_label, {})
    passes = {k: (v["median_recall_enrichment"] >= a.min_recall_enrichment
                  and v["rho_dense_vs_class"] >= a.min_rho)
              for k, v in gate.items()}
    verdict = {
        "gate_label": a.gate_label,
        "min_recall_enrichment": a.min_recall_enrichment,
        "min_rho": a.min_rho,
        "passing_configs": [k for k, ok in passes.items() if ok],
        "pass": any(passes.values()),
        "rule": "a density stratum stands in for --knn-classes only if it both "
                "ENRICHES sub_50_200 and RANKS TILES like the class loss; "
                "neither alone is sufficient",
    }
    out = {
        "ks": ks.tolist(), "strata_rank": strat_k, "tiles": tiles,
        "phase_lambda": a.phase_lambda, "n_quantiles": a.n_quantiles,
        "dead_zone": a.dead_zone, "buffer_cells": a.buffer_cells,
        "queries": a.queries, "region_json": a.region_json, "seed": a.seed,
        "fields": {k: str(v) for k, v in fields.items()},
        "verdict": verdict, "per_field": per_field,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n=== {a.gate_label} stop rule: "
          f"{'PASS' if verdict['pass'] else 'FAIL'}", flush=True)
    for k, v in sorted(gate.items()):
        print(f"  {k:26s} recall_enrich {v['median_recall_enrichment']:6.2f}  "
              f"rho {v['rho_dense_vs_class']:+.3f} "
              f"(pooled control {v['rho_pooled_vs_class_control']:+.3f})  "
              f"{'PASS' if passes[k] else 'fail'}", flush=True)
    print(f"-> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
