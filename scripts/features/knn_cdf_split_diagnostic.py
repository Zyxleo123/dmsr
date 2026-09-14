#!/usr/bin/env python
"""kNN-distance go/no-go, round 2: WHOSE clumping moved, and can a tile see it?

Round 1 (``knn_cdf_diagnostic.py``, job 49259) left two questions open.

``membership`` -- is the signal subhalos or host cores?
    Round 1's ``ratio_q10`` is the most-clustered 10% of MASS, which is mostly
    host centres; 50-200p subhalos are well under 1% of the mass. The arm that
    passed Rockstar went 1.028 -> 0.990 at k=128, which is exactly what the
    measured central over-concentration would also do. So here every sampled
    particle is classed by HR's OWNER array (the innermost halo that binds it)
    and the same Lagrangian ids are measured in every field:

    ========== =====================================================
    class      HR owner is ...
    ========== =====================================================
    sub_50_200 a subhalo (parent >= 0) with 50 <= num_p < 200 -- THE deficit
    sub_ge200  a subhalo with num_p >= 200
    sub_lt50   a subhalo below Rockstar's 50p floor
    host       a top-level halo (its own particles, not its subhalos')
    unbound    nothing (owner < 0)
    ========== =====================================================

    Up to ``--n-per-class`` ids per class, so the rare class is not a few
    hundred particles in a 200k draw. Per class and field: the median ``r_k``
    ratio to HR on the same ids (``paired_ratio_med``), the distribution ratio
    in quantile bins, and ``ratio_med`` of the two class distributions.

    **The stop rule.** If frozen's ``sub_50_200`` ratio is inside the 2% dead
    zone at every k, the statistic does not see the deficit on the particles
    that ARE the deficit, and the term is not built further.

``boundary`` -- does the neighbour-tile fix make a tile's estimate honest?
    Round 1 measured a single-tile candidate set failing at every buffer (p95
    overestimate of r_128 +62-80%), consistent with displacements of a whole
    tile (median |Psi| 11.2 Mpc/h on the one HR tile checked). Per random tile,
    against the full-box truth:

    ``lagrangian``       own particles, cube on the Lagrangian centre (round 1)
    ``centroid``         own particles, cube on their circular mean
    ``centroid_nbr``     + every other particle of the SAME field within the
                         cube grown by ``m`` cells (pure geometry: must go to 0)
    ``centroid_frozen``  live tile from the TUNED field + neighbours from the
                         FROZEN field -- what the trainer can afford. Its residual
                         is the cost of stale neighbours, read against
                         ``tuned_centroid_nbr`` (same, but tuned neighbours).

    Reported per mode and size: incomplete fraction, median/p95/p99 relative
    error of ``r_8`` and ``r_kmax``, the fraction of the tile's own particles
    eligible as queries, and the candidate-count multiplier (the trainer's cost).

``--region-json`` -- measure where the arms actually changed something
    Round 2 (job 51232) sampled the whole box, but every arm field on disk is the
    frozen box with only 32 tiles (6.25%) replaced -- the field ``.json``'s
    ``tiles``. So 94% of a box-wide sample is unedited and the arm-vs-frozen
    deltas were diluted ~16x, and none of the 16 random boundary tiles was
    edited, which made the frozen-neighbour check trivially exact. With this flag
    every sample is drawn from particles whose Lagrangian site is in those tiles,
    and the boundary tiles ARE those tiles.

``loss_forms`` -- which loss would actually see the gap?
    Round 2 found the frozen field's HR-subhalo particles 1.35-1.88x too far from
    their neighbours while its UNBOUND particles are 0.73-0.85x -- too crowded --
    so a pooled distribution match cancels most of the deficit. Per field, the
    three candidate forms are evaluated on the same particles with the loss's own
    squared dead-zone hinge, so their values and their arm ordering compare:

    ``unpaired``    one sorted, quantile-binned match over a uniform sample of
                    the region -- the form ``knn_cdf_loss`` implements now.
    ``class_cond``  the same match run separately per HR owner class, averaged
                    with the classes' mass fractions (``_mass``) or equally
                    (``_equal``). A distribution, but needs an owner array.
    ``paired``      per particle, candidate vs HR on the SAME Lagrangian id,
                    over the uniform sample. Catalog-free, but a per-particle
                    target -- the shape the shell family overfit with.

``--phase-lambda`` -- round 5: the same measurements in 6D phase space
    Rounds 1-4 measured POSITION kNN only. A 50-200p subhalo orbiting inside a
    host is barely denser than the host in 3D -- most of its 128 nearest
    neighbours are host particles -- so r_k on its particles mostly measures the
    host. Rockstar finds it in PHASE space, where its bulk velocity separates it.
    With ``--phase-lambda L`` (km/s per Mpc/h) every particle is the point
    ``(x, v / L)``: periodic in x, open in v. Virial scaling makes one L valid
    at every mass, ``V_vir / r_vir ~ 10 H0 = 1000 km/s per Mpc/h`` for Delta=200c.
    0 (the default) is the 3D statistic, unchanged.

``--boundary-by-class`` -- the tile-local error per HR owner class
    Boundary queries are drawn stratified by class (``--boundary-queries`` split
    evenly) and each mode also reports ``by_class``. The cost question of a
    trainer term is whether SUBHALO particles' neighbours are local, not the
    average particle's.

``--resume`` -- skip fields already in ``--out`` (6D is slower; a time limit
    then costs only the unfinished fields). HR's log r_k is kept next to the JSON
    as ``<out>.hr_z.npy`` so loss forms can be computed without HR's tree.

Reads saved fields, the HR catalog and owner array; no model. Writes one JSON,
rewritten after every stage.
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
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import knn_cdf_diagnostic as v1  # noqa: E402
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.features.knn_cdf import box_neighbours, tile_rows_flat  # noqa: E402

CLASSES = ("sub_50_200", "sub_ge200", "sub_lt50", "host", "unbound")
#: ``preprocess_srs.velnorm(1.0, 0.0, undo=True)``: on-disk velocity channels x
#: this = peculiar velocity in km/s (``PhaseSpaceConfig.vel_norm_km_s``).
VEL_NORM_KM_S = 313.42210244571896


def load_points(path: str, phase_lambda: float) -> np.ndarray:
    """``(NG^3, 3)`` positions, or ``(NG^3, 6)`` ``(x, v / phase_lambda)`` in Mpc/h."""
    if phase_lambda <= 0:
        return v1.load_positions(path)
    ng = v1.NG
    out = np.empty((ng ** 3, 6), dtype=np.float64)
    out[:, :3] = v1.load_positions(path)
    x = np.load(path, mmap_mode="r")
    for c in range(3):
        out[:, 3 + c] = (np.asarray(x[3 + c], dtype=np.float64).reshape(-1)
                         * (VEL_NORM_KM_S / float(phase_lambda)))
    return out


def tree_boxsize(ndim: int, box: float):
    """Periodic in position; 0 (open) in the velocity axes, which scipy supports."""
    return box if ndim == 3 else np.array([box] * 3 + [0.0] * (ndim - 3))


def make_tree(pts: np.ndarray, box: float) -> cKDTree:
    """A tile-local tree: a float64 copy with the position columns wrapped."""
    pts = np.array(pts, dtype=np.float64)
    x = pts[:, :3]
    np.mod(x, box, out=x)
    x[x >= box] -= box
    return cKDTree(pts, boxsize=tree_boxsize(pts.shape[1], box))


def classify(owner: np.ndarray, cat) -> np.ndarray:
    """``int8`` class per particle, indices into :data:`CLASSES`."""
    ids = np.asarray(cat.ids, dtype=np.int64)
    lut_parent = np.full(int(ids.max()) + 1, -2, dtype=np.int64)
    lut_np = np.zeros(int(ids.max()) + 1, dtype=np.int64)
    lut_parent[ids] = np.asarray(cat.parent_ids, dtype=np.int64)
    lut_np[ids] = np.asarray(cat.num_p, dtype=np.int64)
    out = np.full(owner.shape[0], CLASSES.index("unbound"), dtype=np.int8)
    for a in range(0, owner.shape[0], 1 << 24):
        o = np.asarray(owner[a:a + (1 << 24)], dtype=np.int64)
        b = o >= 0
        if (lut_parent[o[b]] == -2).any():
            raise ValueError("owner array names a halo id absent from the catalog; "
                             "catalog and owner array are from different runs")
        par, npart = lut_parent[o[b]], lut_np[o[b]]
        c = np.full(b.sum(), CLASSES.index("host"), dtype=np.int8)
        sub = par >= 0
        c[sub & (npart < 50)] = CLASSES.index("sub_lt50")
        c[sub & (npart >= 50) & (npart < 200)] = CLASSES.index("sub_50_200")
        c[sub & (npart >= 200)] = CLASSES.index("sub_ge200")
        seg = out[a:a + (1 << 24)]
        seg[b] = c
    return out


def sample_by_class(cls: np.ndarray, n_per: int, rng) -> dict:
    out = {}
    for i, name in enumerate(CLASSES):
        pool = np.flatnonzero(cls == i)
        take = pool if pool.size <= n_per else rng.choice(pool, n_per, replace=False)
        out[name] = np.sort(take)
    return out


def _hinge_sq(dlog: np.ndarray, dead_zone: float) -> np.ndarray:
    """``knn_cdf_loss``'s penalty, two-sided, elementwise."""
    return np.square(np.clip(np.abs(dlog) - np.log1p(dead_zone), 0.0, None))


def loss_forms(z_uni, h_uni, cls_uni, z_str, h_str, slices, class_mass,
               n_q, dead_zone) -> dict:
    """The three candidate forms on one field vs HR, per k and overall."""
    def unpaired(zc, zh):
        dq = v1.quantile_curve(zc, n_q) - v1.quantile_curve(zh, n_q)
        return _hinge_sq(dq, dead_zone).mean(axis=0)

    out = {"unpaired": unpaired(z_uni, h_uni).tolist(),
           "paired": _hinge_sq(z_uni - h_uni, dead_zone).mean(axis=0).tolist(),
           "paired_dlog_mean": (z_uni - h_uni).mean(axis=0).tolist()}
    per, w = {}, {}
    for c in CLASSES:
        zc, zh = z_str[slices[c]], h_str[slices[c]]
        if zc.shape[0] < n_q:
            continue
        per[c] = unpaired(zc, zh)
        w[c] = float(class_mass.get(c, 0))
        out[f"class_cond[{c}]"] = per[c].tolist()
        out[f"paired[{c}]"] = _hinge_sq(zc - zh, dead_zone).mean(axis=0).tolist()
    tot = sum(w.values()) or 1.0
    out["class_cond_mass"] = sum(per[c] * (w[c] / tot) for c in per).tolist()
    out["class_cond_equal"] = (sum(per.values()) / max(len(per), 1)).tolist()
    # Which classes the UNIFORM sample itself held, so the unpaired number can be
    # read against the mass it was computed over.
    out["uniform_class_frac"] = {c: float((cls_uni == i).mean())
                                 for i, c in enumerate(CLASSES)}
    for key in ("unpaired", "paired", "class_cond_mass", "class_cond_equal"):
        out[f"{key}_total"] = float(np.mean(out[key]))
    return out


def rk_query(tree, pts, ids_self, k_max):
    """``(n, k_max)`` distances with self removed BY ID (``ids_self`` in tree rows),
    plus neighbour ids. ``ids_self=None`` when the queries are not in the tree."""
    d, n = tree.query(pts, k=k_max + (0 if ids_self is None else 1), workers=-1)
    if ids_self is None:
        return d, n
    keep = n != ids_self[:, None]
    bad = keep.sum(axis=1) != k_max
    keep[bad, -1] = False
    return d[keep].reshape(len(pts), k_max), n[keep].reshape(len(pts), k_max)


def boundary(field_pos, tree, frozen_pos, ks, tiles, sizes, n_queries, rng,
             tag, ng, tile, cell, box, cls=None):
    """Per-mode error of tile-local r_k against ``tree`` (the field's full box).

    ``field_pos`` / ``frozen_pos`` are 3- or 6-column points (``load_points``);
    cubes, centroids and neighbour tiles use the position columns only, which is
    exact for 6D too: a 6D neighbour within r is within r in position. ``cls``
    (the per-particle class vector) stratifies queries and adds ``by_class``.
    """
    k_max = int(max(ks))
    side = tile * cell
    col8 = int(np.argmin(np.abs(np.asarray(ks) - 8)))
    rec = {}

    def pick(inside, rows):
        if cls is None:
            return inside if inside.size <= n_queries else rng.choice(
                inside, n_queries, replace=False)
        per = max(1, n_queries // len(CLASSES))
        c_in = cls[rows[inside]]
        out = []
        for i in range(len(CLASSES)):
            p = inside[c_in == i]
            out.append(p if p.size <= per else rng.choice(p, per, replace=False))
        return np.sort(np.concatenate(out))

    def add(mode, size, rel, inc, pool_frac, mult, qcls):
        r = rec.setdefault(mode, {}).setdefault(float(size), {
            "rel_small": [], "rel_kmax": [], "inc": [], "pool": [], "mult": [],
            "cls": []})
        r["rel_small"].append(rel[:, 0])
        r["rel_kmax"].append(rel[:, 1])
        r["inc"].append(inc)
        r["pool"].append(pool_frac)
        r["mult"].append(mult)
        r["cls"].append(qcls)

    def circ_mean(p):
        ang = p[:, :3] * (2 * np.pi / box)
        return np.mod(np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
                      * box / (2 * np.pi), box)

    def cheb_to(p, c):
        d = p[:, :3] - c
        d -= box * np.round(d / box)
        return np.abs(d).max(axis=1)

    for tid in tiles:
        rows = tile_rows_flat(int(tid), ng=ng, tile=tile)
        own = field_pos[rows].astype(np.float64)
        lag_c = (np.array([tid // (ng // tile) ** 2, (tid // (ng // tile)) % (ng // tile),
                           tid % (ng // tile)]) + 0.5) * side
        cen = circ_mean(own)
        member = np.zeros(field_pos.shape[0], dtype=bool)
        member[rows] = True
        big_m = max(sizes) * cell
        nbr_ids, _ = box_neighbours(field_pos[:, :3], int(tid), cen, 0.5 * side + big_m,
                                    ng=ng, tile=tile, boxsize_mpc_h=box)
        ltree = make_tree(own, box)
        for centre_name, c, sizes_here in (("lagrangian", lag_c, sizes),
                                           ("centroid", cen, sizes)):
            cheb = cheb_to(own, c)
            for b in sizes_here:
                half = 0.5 * side - b * cell
                inside = np.flatnonzero(cheb <= half) if half > 0 else np.zeros(0, int)
                if inside.size < 16:
                    continue
                take = pick(inside, rows)
                truth, tn = rk_query(tree, own[take], rows[take], k_max)
                t_sel = truth[:, [col8, len(ks) - 1]]
                loc, _ = rk_query(ltree, own[take], take, k_max)
                rel = loc[:, [col8, len(ks) - 1]] / t_sel - 1
                inc = ~member[tn].all(axis=1)
                add(f"{tag}{centre_name}", b, rel, inc, inside.size / rows.size, 1.0,
                    None if cls is None else cls[rows[take]])

        # Neighbours offered, cube on the centroid, NO buffer: the margin does it.
        inside = np.flatnonzero(cheb_to(own, cen) <= 0.5 * side)
        if inside.size < 16:
            continue
        take = pick(inside, rows)
        qcls = None if cls is None else cls[rows[take]]
        truth, tn = rk_query(tree, own[take], rows[take], k_max)
        t_sel = truth[:, [col8, len(ks) - 1]]

        # Each source picks its neighbours the way the trainer would: ids chosen
        # by THAT source's positions about THAT source's centroid of the tile.
        # For the frozen source this is the approximation being measured -- the
        # trainer only has the frozen box when it builds the pool.
        sources = [("centroid_nbr", field_pos, nbr_ids,
                    cheb_to(field_pos[nbr_ids], cen))]
        if frozen_pos is not None:
            fcen = circ_mean(frozen_pos[rows].astype(np.float64))
            f_ids, _ = box_neighbours(frozen_pos[:, :3], int(tid), fcen,
                                      0.5 * side + big_m,
                                      ng=ng, tile=tile, boxsize_mpc_h=box)
            sources.append(("centroid_frozen", frozen_pos, f_ids,
                            cheb_to(frozen_pos[f_ids].astype(np.float64), fcen)))
        for m in sizes:
            for name, src, ids_all, cheb_n in sources:
                sel = ids_all[cheb_n <= 0.5 * side + m * cell]
                cand_ids = np.concatenate([rows, sel])
                cand = np.concatenate([own, src[sel].astype(np.float64)])
                loc, ln = rk_query(make_tree(cand, box), own[take], take, k_max)
                rel = loc[:, [col8, len(ks) - 1]] / t_sel - 1
                inc_set = np.zeros(field_pos.shape[0], dtype=bool)
                inc_set[cand_ids] = True
                inc = ~inc_set[tn].all(axis=1)
                add(f"{tag}{name}", m, rel, inc, inside.size / rows.size,
                    cand.shape[0] / rows.size, qcls)
    out = {}
    for mode, per in rec.items():
        out[mode] = []
        for size in sorted(per):
            r = per[size]
            rs, rk = np.concatenate(r["rel_small"]), np.concatenate(r["rel_kmax"])
            inc_all = np.concatenate(r["inc"])
            by_class = {}
            if cls is not None:
                qc = np.concatenate(r["cls"])
                for i, c in enumerate(CLASSES):
                    s = qc == i
                    if s.sum() < 20:
                        continue
                    by_class[c] = {
                        "n_queries": int(s.sum()),
                        "incomplete_frac": float(inc_all[s].mean()),
                        "rel_kmax_median": float(np.median(rk[s])),
                        "rel_kmax_p95": float(np.quantile(np.abs(rk[s]), 0.95)),
                        "rel_small_p95": float(np.quantile(np.abs(rs[s]), 0.95)),
                    }
            out[mode].append({
                "by_class": by_class,
                "queries_stratified_by_class": cls is not None,
                "size_cells": size,
                "size_kind": "margin" if "nbr" in mode or "frozen" in mode else "buffer",
                "n_queries": int(rk.size),
                "incomplete_frac": float(np.concatenate(r["inc"]).mean()),
                "query_pool_frac": float(np.mean(r["pool"])),
                "candidate_multiplier": float(np.mean(r["mult"])),
                "rel_small_median": float(np.median(rs)),
                "rel_small_p95": float(np.quantile(np.abs(rs), 0.95)),
                "rel_kmax_median": float(np.median(rk)),
                "rel_kmax_p95": float(np.quantile(np.abs(rk), 0.95)),
                "rel_kmax_p99": float(np.quantile(np.abs(rk), 0.99)),
            })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", nargs="+", required=True,
                    help="label=path; must include --hr-label and --frozen-label")
    ap.add_argument("--hr-label", default="hr")
    ap.add_argument("--frozen-label", default="frozen")
    ap.add_argument("--tuned-label", default="selfhostguard",
                    help="field used for the stale-neighbour boundary mode")
    ap.add_argument("--catalog", required=True, help="HR rockstar ascii")
    ap.add_argument("--owner", required=True, help="HR owner .npy")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 64, 128])
    ap.add_argument("--n-per-class", type=int, default=100_000)
    ap.add_argument("--n-quantiles", type=int, default=32)
    ap.add_argument("--dead-zone", type=float, default=0.02)
    ap.add_argument("--sizes", type=int, nargs="+", default=[0, 2, 4, 8, 16],
                    help="buffer cells (no-neighbour modes) / margin cells (neighbour modes)")
    ap.add_argument("--boundary-tiles", type=int, default=16)
    ap.add_argument("--boundary-queries", type=int, default=1024)
    ap.add_argument("--region-json", default=None,
                    help="an arm field's .json; restrict every sample and the "
                         "boundary tiles to its spliced `tiles`")
    ap.add_argument("--n-uniform", type=int, default=200_000,
                    help="uniform sample (of the region, if given) for the "
                         "unpaired and paired loss forms")
    ap.add_argument("--phase-lambda", type=float, default=0.0,
                    help="km/s per Mpc/h; > 0 measures kNN on (x, v/lambda) in 6D. "
                         "0 = position only (rounds 1-4)")
    ap.add_argument("--boundary-by-class", action="store_true",
                    help="stratify boundary queries by HR owner class, report by_class")
    ap.add_argument("--resume", action="store_true",
                    help="keep fields already measured in --out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ks = np.asarray(sorted(set(a.ks)), dtype=int)
    k_max = int(ks.max())
    ng, tile, box = v1.NG, v1.TILE, v1.BOX_MPC
    cell = box / ng
    rng = np.random.default_rng(a.seed)
    fields = dict(f.partition("=")[::2] for f in a.fields)
    for need in (a.hr_label, a.frozen_label):
        if need not in fields or not Path(fields[need]).exists():
            raise SystemExit(f"required field {need!r} missing: {fields.get(need)}")
    order = [a.hr_label, a.frozen_label] + [l for l in fields
                                           if l not in (a.hr_label, a.frozen_label)]

    t0 = time.time()
    cat = load_rockstar_ascii(a.catalog)
    owner = np.load(a.owner, mmap_mode="r")
    if owner.shape[0] != ng ** 3:
        raise SystemExit(f"owner has {owner.shape[0]} rows, expected {ng ** 3}")
    cls = classify(owner, cat)
    counts = {c: int((cls == i).sum()) for i, c in enumerate(CLASSES)}
    n_side = ng // tile
    region_tiles = None
    if a.region_json:
        region_tiles = sorted(int(t) for t in json.loads(
            Path(a.region_json).read_text())["tiles"])
        in_region = np.zeros(ng ** 3, dtype=bool)
        for t in region_tiles:
            in_region[tile_rows_flat(t, ng=ng, tile=tile)] = True
        # -1 is no class, so sample_by_class never draws outside the region.
        cls = np.where(in_region, cls, np.int8(-1)).astype(np.int8)
        pool = np.flatnonzero(in_region)
        del in_region
    else:
        pool = np.arange(ng ** 3)
    region_counts = {c: int((cls == i).sum()) for i, c in enumerate(CLASSES)}
    samples = sample_by_class(cls, a.n_per_class, rng)
    uni = np.sort(pool if pool.size <= a.n_uniform
                  else rng.choice(pool, a.n_uniform, replace=False))
    cls_uni = cls[uni]
    cls_bnd = cls if (a.boundary_by_class and a.boundary_tiles) else None
    del cls, pool
    print(f"classes box {counts}\n        region {region_counts} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    all_ids = np.concatenate([samples[c] for c in CLASSES] + [uni])
    slices, at = {}, 0
    for c in CLASSES:
        slices[c] = slice(at, at + samples[c].size)
        at += samples[c].size
    uni_slice = slice(at, at + uni.size)

    if region_tiles is not None:
        tiles = np.asarray(region_tiles if a.boundary_tiles >= len(region_tiles)
                           else rng.choice(region_tiles, a.boundary_tiles,
                                           replace=False))
    else:
        tiles = rng.choice(n_side ** 3, size=a.boundary_tiles, replace=False)
    out = {"ks": ks.tolist(), "dead_zone": a.dead_zone, "seed": a.seed,
           "space": "6d" if a.phase_lambda > 0 else "3d",
           "phase_lambda_kms_per_mpch": float(a.phase_lambda),
           "vel_norm_km_s": VEL_NORM_KM_S,
           "region_json": a.region_json, "region_tiles": region_tiles,
           "class_particle_counts": counts,
           "class_particle_counts_region": region_counts,
           "class_sample_counts": {c: int(samples[c].size) for c in CLASSES},
           "n_uniform": int(uni.size),
           "boundary_tiles": [int(t) for t in tiles], "membership": {},
           "loss_forms": {}, "boundary": {}}

    def save():
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))

    hr_z = None
    frozen_pos32 = None
    hr_z_path = Path(a.out + ".hr_z.npy")
    if a.resume and Path(a.out).exists():
        prev = json.loads(Path(a.out).read_text())
        same = ("ks", "seed", "space", "phase_lambda_kms_per_mpch", "region_tiles",
                "class_sample_counts", "n_uniform", "boundary_tiles")
        bad = [k for k in same if prev.get(k) != out.get(k)]
        if bad:
            raise SystemExit(f"--resume: {a.out} was written with different {bad}; "
                             "move it aside or drop --resume")
        if a.hr_label in prev.get("membership", {}) and hr_z_path.exists():
            hr_z = np.load(hr_z_path)
            for key in ("membership", "loss_forms", "boundary"):
                out[key] = prev.get(key, {})
            print(f"  resume: {sorted(out['membership'])} already measured", flush=True)
    for label in order:
        path = fields[label]
        if not Path(path).exists():
            print(f"  MISSING {label}: {path}", flush=True)
            continue
        if label in out["membership"]:
            tuned_pending = (a.boundary_tiles and a.tuned_label in fields
                             and a.tuned_label not in out["membership"])
            if label == a.frozen_label and tuned_pending:
                frozen_pos32 = load_points(path, a.phase_lambda).astype(np.float32)
            continue
        t1 = time.time()
        pos = load_points(path, a.phase_lambda)
        tree = cKDTree(pos, boxsize=tree_boxsize(pos.shape[1], box),
                       balanced_tree=False, compact_nodes=False)
        print(f"  {label}: loaded + tree in {time.time() - t1:.0f}s", flush=True)
        d, _ = rk_query(tree, pos[all_ids], all_ids, k_max)
        z = np.log(d[:, ks - 1])
        rec = {"path": path, "classes": {}}
        for c in CLASSES:
            zc = z[slices[c]]
            if zc.shape[0] < a.n_quantiles:
                continue
            entry = {"median_r": np.exp(np.median(zc, axis=0)).tolist(),
                     "quantile_log_r": v1.quantile_curve(zc, a.n_quantiles).tolist()}
            if hr_z is not None:
                hz = hr_z[slices[c]]
                dq = (v1.quantile_curve(zc, a.n_quantiles)
                      - v1.quantile_curve(hz, a.n_quantiles))
                entry.update({
                    "paired_ratio_med": np.exp(np.median(zc - hz, axis=0)).tolist(),
                    "ratio_med": np.exp(np.median(zc, 0) - np.median(hz, 0)).tolist(),
                    "ratio_q10": np.exp(dq[:max(1, a.n_quantiles // 10)].mean(0)).tolist(),
                    "ratio_bins": np.exp(dq).tolist(),
                    "outside_dead_zone": bool(np.any(
                        np.abs(np.median(zc - hz, axis=0)) > np.log1p(a.dead_zone))),
                })
            rec["classes"][c] = entry
        if label == a.hr_label:
            hr_z = z
            np.save(hr_z_path, hr_z)
        else:
            lf = loss_forms(z[uni_slice], hr_z[uni_slice], cls_uni, z, hr_z,
                            slices, region_counts, a.n_quantiles, a.dead_zone)
            out["loss_forms"][label] = lf
            print(f"  {label}: loss unpaired {lf['unpaired_total']:.4g} "
                  f"class_cond(mass) {lf['class_cond_mass_total']:.4g} "
                  f"class_cond(equal) {lf['class_cond_equal_total']:.4g} "
                  f"paired {lf['paired_total']:.4g}", flush=True)
            s = rec["classes"].get("sub_50_200", {})
            print(f"  {label}: sub_50_200 paired ratio "
                  f"{np.round(s.get('paired_ratio_med', []), 3).tolist()} host "
                  f"{np.round(rec['classes'].get('host', {}).get('paired_ratio_med', []), 3).tolist()}",
                  flush=True)
        out["membership"][label] = rec
        save()

        if label == a.hr_label and a.boundary_tiles:
            t2 = time.time()
            out["boundary"].update(boundary(pos, tree, None, ks, tiles, a.sizes,
                                            a.boundary_queries, rng, "hr_", ng,
                                            tile, cell, box, cls=cls_bnd))
            print(f"  boundary (hr) in {time.time() - t2:.0f}s", flush=True)
            save()
        if label == a.frozen_label:
            frozen_pos32 = pos.astype(np.float32)
        if label == a.tuned_label and a.boundary_tiles and frozen_pos32 is not None:
            t2 = time.time()
            res = boundary(pos, tree, frozen_pos32, ks, tiles, a.sizes,
                           a.boundary_queries, rng, "tuned_", ng, tile, cell, box,
                           cls=cls_bnd)
            out["boundary"].update(res)
            print(f"  boundary (tuned + frozen neighbours) in {time.time() - t2:.0f}s",
                  flush=True)
            save()
        del pos, tree

    for mode, recs in out["boundary"].items():
        for r in recs:
            print(f"    {mode:<24} {r['size_kind']}={r['size_cells']:>4} "
                  f"inc={r['incomplete_frac']:.3f} p95(r_kmax)={r['rel_kmax_p95']:.4f} "
                  f"p95(r_8)={r['rel_small_p95']:.4f} x{r['candidate_multiplier']:.2f}",
                  flush=True)
    lfs = out["loss_forms"]
    if a.frozen_label in lfs:
        print("\n  loss form, change vs frozen (negative = closer to HR):", flush=True)
        for key in ("unpaired", "class_cond_mass", "class_cond_equal", "paired"):
            f0 = lfs[a.frozen_label][f"{key}_total"]
            row = "  ".join(
                f"{lab} {100 * (lfs[lab][f'{key}_total'] / f0 - 1):+.1f}%"
                for lab in lfs if lab != a.frozen_label and f0 > 0)
            print(f"    {key:<17} frozen={f0:.4g}  {row}", flush=True)
    save()
    print(f"=== wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
