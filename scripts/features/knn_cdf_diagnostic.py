#!/usr/bin/env python
"""Go/no-go for the kNN-distance loss: does the statistic SEE the subhalo deficit?

This runs before any training, for the same reason ``density_power_spectrum.py``
did: ``density-power-cannot-see-subhalos`` measured that ``P(k)`` matched HR to
1-5% on a frozen field Rockstar says is missing most of HR's subhalos, so the
instrument was blind. :mod:`cosmo_sr.features.knn_cdf` claims to be the
instrument that is not. That claim is cheap to test on fields already on disk
and expensive to discover false after a 10 h train.

Two measurements, both against the **full-box** truth
------------------------------------------------------
``separation``
    For every field, ``log r_k`` at a fixed box-wide sample of Lagrangian ids --
    the SAME ids in every field, which is a uniform sample of MASS in each (the
    right weighting for a particle-centred statistic) -- with the neighbour
    search over all 512^3 particles on the periodic box. So nothing here has a
    tile boundary to be wrong about. Per field, the full quantile curve per ``k``
    and its ratio to HR.

    The verdict this is read for, per ``k``:

    * ``frozen`` vs ``hr`` in the LOW quantiles (``ratio_q10``; the median is
      background -- see ``knn_cdf_loss``'s docstring for the measurement). If
      frozen already sits within the loss's dead zone of HR there, the loss has
      no gradient to give and **the line stops here**.
    * whether the arm that won the held-out Rockstar gate moved TOWARD HR on
      this statistic. If it moved away, the statistic disagrees with the gate
      and is the wrong target whatever its gradient looks like.

    Paired, per-particle, reported because it is free and because it is the
    distinction `shell-target-is-real-not-noise` taught: ``clumped_recall`` is
    the fraction of particles HR has clumped (``r_k`` in HR's bottom decile) that
    the field also has clumped. A field can match the distribution with the
    clumping in different particles; the loss would score that as a win and the
    halo finder might not.

``boundary``
    ``KnnConfig.buffer_cells`` is the estimator's only bias and it biases in the
    direction of the signal, so it is measured, not chosen. For random tiles of
    the HR field, candidates are that tile's own 64^3 particles (what one tile of
    the training pool supplies) and queries are the tile's particles in the
    Eulerian cube shrunk by ``b`` cells. Against the full-box ``r_k`` of the same
    particles: the fraction of queries whose neighbour list is incomplete, and
    the median / p95 fractional overestimate of ``r_kmax``. The run should use
    the smallest ``b`` at which p95 falls inside the loss's dead zone.

No gradients anywhere: the search is scipy's periodic ``cKDTree`` on CPU, which
is exact and is the reference the torch search in ``knn_cdf`` is a
differentiable re-implementation of. Writes one JSON; the plotter reads that.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

BOX_MPC = 100.0
NG = 512
TILE = 64
CELL = BOX_MPC / NG
#: z=0 displacement undo, the same constant ``density_power_spectrum.py`` uses:
#: the on-disk channels are Psi / 6000 kpc/h, i.e. x6 in Mpc/h.
DIS_SCALE_MPC_H = 6.0


def load_positions(path: str) -> np.ndarray:
    """``(NG^3, 3)`` float64 Eulerian positions in Mpc/h, row = flat Lagrangian id.

    Row ``(i * NG + j) * NG + k`` is site ``(i, j, k)`` -- C order of the grid,
    which is the flat id every owner array in this line is indexed by.
    """
    x = np.load(path, mmap_mode="r")
    if x.shape != (6, NG, NG, NG):
        raise ValueError(f"{path}: expected (6,{NG},{NG},{NG}), got {x.shape}")
    ax = (np.arange(NG, dtype=np.float64) + 0.5) * CELL
    pos = np.empty((NG ** 3, 3), dtype=np.float64)
    for c in range(3):
        # One component at a time: a float64 copy of all three displacement
        # channels plus the lattice would be ~13 GB of temporaries at once.
        disp = np.asarray(x[c], dtype=np.float64).reshape(-1) * DIS_SCALE_MPC_H
        shape = [1, 1, 1]
        shape[c] = NG
        q = np.broadcast_to(ax.reshape(shape), (NG, NG, NG)).reshape(-1)
        pos[:, c] = np.mod(q + disp, BOX_MPC)
        del disp, q
    # cKDTree with boxsize requires coordinates strictly inside [0, L); mod can
    # return exactly L for a value a hair below 0 in float64.
    pos[pos >= BOX_MPC] -= BOX_MPC
    return pos


def knn_full_box(tree: cKDTree, pos: np.ndarray, ids: np.ndarray,
                 ks: np.ndarray, workers: int) -> np.ndarray:
    """``(len(ids), len(ks))`` ``r_k``, self excluded BY IDENTITY, periodic box."""
    k_max = int(ks.max())
    dist, nbr = tree.query(pos[ids], k=k_max + 1, workers=workers)
    keep = nbr != ids[:, None]
    bad = keep.sum(axis=1) != k_max
    if bad.any():
        # A query absent from its own top k+1 means more than k+1 particles
        # coincide with it exactly; every returned distance is then 0 and
        # dropping the last column is correct.
        keep[bad, -1] = False
    dist = dist[keep].reshape(len(ids), k_max)
    return dist[:, ks - 1]


def quantile_curve(z: np.ndarray, n_q: int) -> np.ndarray:
    """``(n_q, n_k)`` bin means of sorted ``log r_k``: ``knn_cdf._quantile_reduce``."""
    zs = np.sort(z, axis=0)
    n = zs.shape[0]
    edges = [(b * n) // n_q for b in range(n_q + 1)]
    return np.stack([zs[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])


def boundary_experiment(pos: np.ndarray, tree: cKDTree, ks: np.ndarray,
                        buffers, n_tiles: int, n_queries: int,
                        rng: np.random.Generator, workers: int):
    n_side = NG // TILE
    tiles = rng.choice(n_side ** 3, size=n_tiles, replace=False)
    side = TILE * CELL
    k_max = int(ks.max())
    per_b = {float(b): {"incomplete": [], "rel": []} for b in buffers}
    for tid in tiles:
        tx, ty, tz = tid // (n_side * n_side), (tid // n_side) % n_side, tid % n_side
        li = np.arange(TILE)
        gi, gj, gk = np.meshgrid(tx * TILE + li, ty * TILE + li, tz * TILE + li,
                                 indexing="ij")
        rows = ((gi * NG + gj) * NG + gk).reshape(-1)
        centre = (np.array([tx, ty, tz]) + 0.5) * side
        local = pos[rows]
        d = local - centre
        d -= BOX_MPC * np.round(d / BOX_MPC)
        cheb = np.abs(d).max(axis=1)
        # Tile-local tree, on the periodic box so a tile touching a face is
        # handled exactly as the torch minimum image handles it.
        ltree = cKDTree(local, boxsize=BOX_MPC)
        member = np.zeros(NG ** 3, dtype=bool)
        member[rows] = True
        for b in buffers:
            half = 0.5 * side - float(b) * CELL
            inside = np.flatnonzero(cheb <= half) if half > 0 else np.empty(0, int)
            if inside.size < 8:
                continue
            take = inside if inside.size <= n_queries else rng.choice(
                inside, n_queries, replace=False)
            q_rows = rows[take]
            # Truth: full box.
            full_d, full_n = tree.query(pos[q_rows], k=k_max + 1, workers=workers)
            # Tile-local candidates only.
            loc_d, _ = ltree.query(local[take], k=k_max + 1, workers=workers)
            # Column 0 is self in both (a particle is its own nearest at 0) -- safe
            # HERE because this is a diagnostic on HR where exact coincidence has
            # measure zero; the loss module does not make that assumption.
            full_r = full_d[:, k_max]
            loc_r = loc_d[:, k_max]
            incomplete = ~member[full_n[:, 1:]].all(axis=1)
            per_b[float(b)]["incomplete"].append(incomplete)
            per_b[float(b)]["rel"].append(loc_r / full_r - 1.0)
        del member
    out = []
    for b in buffers:
        rec = per_b[float(b)]
        if not rec["rel"]:
            out.append({"buffer_cells": float(b), "n_queries": 0})
            continue
        inc = np.concatenate(rec["incomplete"])
        rel = np.concatenate(rec["rel"])
        out.append({
            "buffer_cells": float(b),
            "buffer_mpc_h": float(b) * CELL,
            "n_queries": int(rel.size),
            "incomplete_frac": float(inc.mean()),
            "rel_err_median": float(np.median(rel)),
            "rel_err_p95": float(np.quantile(rel, 0.95)),
            "rel_err_p99": float(np.quantile(rel, 0.99)),
        })
    return {"k_max": k_max, "n_tiles": int(n_tiles),
            "tiles": [int(t) for t in tiles], "records": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", required=True,
                    help="label=path pairs; one label must be --hr-label")
    ap.add_argument("--hr-label", default="hr")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 64, 128])
    ap.add_argument("--n-sample", type=int, default=200_000,
                    help="box-wide Lagrangian ids measured in every field")
    ap.add_argument("--n-quantiles", type=int, default=64)
    ap.add_argument("--dead-zone", type=float, default=0.02,
                    help="the loss's dead zone, echoed into the verdict")
    ap.add_argument("--buffers", type=float, nargs="+",
                    default=[0, 2, 4, 6, 8, 12, 16])
    ap.add_argument("--boundary-tiles", type=int, default=24)
    ap.add_argument("--boundary-queries", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ks = np.asarray(sorted(set(a.ks)), dtype=int)
    rng = np.random.default_rng(a.seed)
    ids = np.sort(rng.choice(NG ** 3, size=a.n_sample, replace=False))
    fields = [f.partition("=")[::2] for f in a.fields]
    labels = [lab for lab, _ in fields]
    if a.hr_label not in labels:
        raise SystemExit(f"--hr-label {a.hr_label!r} is not among {labels}")
    hr_path = dict(fields)[a.hr_label]
    if not Path(hr_path).exists():
        # Every other field is a ratio to HR, so without it there is nothing to
        # report -- fail loudly rather than write a JSON of absolute curves.
        raise SystemExit(f"HR field missing: {hr_path}")
    # HR first: every other field's ratios need its curve, and the boundary
    # experiment runs on HR's tree while it is built.
    fields.sort(key=lambda lp: lp[0] != a.hr_label)

    out = {"box_mpc_h": BOX_MPC, "ng": NG, "tile": TILE, "ks": ks.tolist(),
           "n_sample": int(a.n_sample), "n_quantiles": int(a.n_quantiles),
           "dead_zone": float(a.dead_zone), "seed": int(a.seed),
           "fields": {}, "boundary": None}
    # Resumable: a previous JSON with the same sample (seed, n, ks) keeps its
    # finished NON-HR fields. HR is always recomputed -- every ratio needs its
    # per-particle values, which are not stored -- and it is one field of six.
    prev = {}
    if Path(a.out).exists():
        old = json.loads(Path(a.out).read_text())
        if (old.get("seed"), old.get("n_sample"), old.get("ks")) == (
                out["seed"], out["n_sample"], out["ks"]):
            prev = {k: v for k, v in old.get("fields", {}).items()
                    if k != a.hr_label and v.get("path") == dict(fields).get(k)}
            if prev:
                print(f"  resuming: keeping {sorted(prev)}", flush=True)
    hr_z = hr_curve = hr_low_mask = None
    for label, path in fields:
        if label in prev:
            out["fields"][label] = prev[label]
            continue
        if not Path(path).exists():
            print(f"  MISSING {label}: {path}", flush=True)
            continue
        t0 = time.time()
        pos = load_positions(path)
        tree = cKDTree(pos, boxsize=BOX_MPC, balanced_tree=False,
                       compact_nodes=False)
        print(f"  {label}: loaded + tree in {time.time() - t0:.0f}s", flush=True)
        r = knn_full_box(tree, pos, ids, ks, a.workers)
        z = np.log(r)
        curve = quantile_curve(z, a.n_quantiles)
        rec = {"path": path,
               "quantile_log_r": curve.tolist(),
               "median_r": np.exp(np.median(z, axis=0)).tolist()}
        if label == a.hr_label:
            hr_z, hr_curve = z, curve
            hr_low_mask = z <= np.quantile(z, 0.10, axis=0)
            if a.boundary_tiles > 0:
                t1 = time.time()
                out["boundary"] = boundary_experiment(
                    pos, tree, ks, a.buffers, a.boundary_tiles,
                    a.boundary_queries, rng, a.workers)
                print(f"  boundary experiment in {time.time() - t1:.0f}s",
                      flush=True)
                for br in out["boundary"]["records"]:
                    print("    b={buffer_cells:>4} incomplete={incomplete_frac:.3f} "
                          "p95={rel_err_p95:+.4f}".format(**br)
                          if br.get("n_queries") else f"    b={br['buffer_cells']} empty",
                          flush=True)
        else:
            dlog = curve - hr_curve
            n_low = max(1, a.n_quantiles // 10)
            low_mask = z <= np.quantile(z, 0.10, axis=0)
            rec.update({
                "ratio_bins": np.exp(dlog).tolist(),
                "ratio_med": np.exp(np.median(dlog, axis=0)).tolist(),
                "ratio_q10": np.exp(dlog[:n_low].mean(axis=0)).tolist(),
                "abs_dlog_mean": np.abs(dlog).mean(axis=0).tolist(),
                # The loss this field would pay at dead zone `dead_zone`,
                # two-sided -- directly comparable across fields.
                "loss_two_sided": float(np.square(np.clip(
                    np.abs(dlog) - np.log1p(a.dead_zone), 0, None)).mean()),
                "clumped_recall": ((low_mask & hr_low_mask).sum(axis=0)
                                   / hr_low_mask.sum(axis=0)).tolist(),
                "paired_dlog_median": np.median(z - hr_z, axis=0).tolist(),
            })
            print(f"  {label}: q10 ratio {np.round(rec['ratio_q10'], 3).tolist()} "
                  f"loss {rec['loss_two_sided']:.4g} recall "
                  f"{np.round(rec['clumped_recall'], 3).tolist()}", flush=True)
        out["fields"][label] = rec
        del pos, tree
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        # Written after every field: a time limit costs the remaining fields,
        # not the finished ones.
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"=== wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
