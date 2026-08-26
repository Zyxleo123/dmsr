#!/usr/bin/env python
"""Where SR2 loses subhalos, and what kind of failure it is.

Reads only the committed ``*_tilew.npz`` weights and the Rockstar catalogs that
are already on disk -- no new halo finding, no owner arrays, no GPU. Emits one
JSON per box under ``$DMSR_REWARD_ROOT/lagrangian_host/<box>/`` so the tables in
``docs/sr2_subhalo_deficit.md`` are redrawable rather than transcribed.

It answers, in order:

1. **Abundance vs host mass, both sides.** HR puts a fixed number of subhalos per
   unit host mass (``sub_per_1e12_msun`` flat, log-log slope 1.0): substructure
   is spread evenly over host mass. Whether SR2 does the same is *the* question,
   and it does not -- its subhalos per unit mass fall by an order of magnitude
   toward cluster mass. This is the cleanest statement of the failure and it is
   measured per host, with no tiles and no correlations in the way.
2. **Host and subhalo abundance functions, over the whole mass range.** Counting
   only hosts above 1e11.5 covers 4% of them and hides that SR2 also loses ~23%
   of the small ones; both are reported by ``num_p`` so hosts and subhalos are
   compared at equal particle count.
3. **Particle-floor sweep.** Which subhalos are lost (the small ones).
4. **Local host fraction vs total host mass.** Per tile, ``L`` (bound-particle
   occupancy) and ``logM`` (mass-weighted mean total host mass) are 0.75
   correlated, so a *marginal* correlation with either cannot say which one
   drives the count. Both partials are reported, in both directions, with the
   2D-bin cell counts -- because the marginal "SR2 makes fewer subhalos where it
   sees more host material" is entirely mediated by mass: at fixed ``logM`` the
   dependence on ``L`` is ~0.
5. **Per-tile object counts.** Hosts, subhalos and all objects per tile by ``L``,
   which shows whether SR2 loses objects in dense regions or merely reclassifies
   them (it loses them: hosts too, not only interiors).
6. **Mean subhalo size, with a survivorship control.** SR2's surviving subhalos
   look "too massive" -- but so would HR's if you kept only its largest N. The
   control compares SR2's mean size against the mean of HR's N largest, which is
   what distinguishes *merging* from plain *attrition*.
7. **Matched clusters.** Every HR host above 1e14 matched to its nearest SR2 host
   periodically, with masses, radii and per-object subhalo counts, so "the hosts
   are present at matching masses" is checked rather than asserted.

One trap this script is built around: ``stream_particle_tile_counts`` groups by
``external_haloid`` and keeps every recursion row, so member weights are NOT a
partition of the particles (summing them over all objects over-counts ~4.6x).
Occupancy is therefore summed over top-level hosts only; subhalo *counts* are
safe because each subhalo owns its weight row once.

    python scripts/features/analyze_sr2_subhalo_deficit.py --boxes set8
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (  # noqa: E402
    DEFAULT_CONFIG, banner, load_reward_config, paths, write_json,
)
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.features import LagrangianHostFeatures  # noqa: E402

MASS_EDGES = np.array([9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
                       14.0, 15.5])
NP_EDGES = np.array([20, 50, 100, 200, 500, 10 ** 9])


def _spearman(a, b):
    """Rank correlation without a scipy dependency in the reward path."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _partial(y, a, b):
    """corr(y, a | b): both residualised on b by least squares, then correlated.

    Reported in *both* directions below. With corr(a, b) ~ 0.75 a single partial
    is not enough to name the driver: the variable whose partial survives is the
    one that matters, and the other's marginal correlation is its shadow.
    """
    A = np.vstack([b, np.ones_like(b)]).T

    def res(v):
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]

    return float(np.corrcoef(res(y), res(a))[0, 1])


def _roots(cat) -> np.ndarray:
    """Top-level ancestor of every catalog row (itself, for a host)."""
    parent = {int(i): int(p) for i, p in zip(cat.ids, cat.parent_ids)}
    out = np.empty(cat.n, dtype=np.int64)
    for k, i in enumerate(cat.ids):
        node = int(i)
        for _ in range(64):
            p = parent.get(node, -1)
            if p < 0:
                break
            node = p
        out[k] = node
    return out


def _load(root, box, tag):
    z = np.load(root / f"{box}__{tag}__{tag}" / f"{box}_{tag}_tilew.npz")
    hits = sorted(glob.glob(str(root / f"{box}__{tag}__{tag}" / f"{tag}_rockstar"
                                 / "halos*.ascii")))
    cat = load_rockstar_ascii(hits[0])
    parent = dict(zip(cat.ids.tolist(), cat.parent_ids.tolist()))
    mass = dict(zip(cat.ids.tolist(), cat.mvir.tolist()))
    num_p = dict(zip(cat.ids.tolist(), cat.num_p.tolist()))
    rec = dict(zip(z["member_halo_id"].tolist(), z["member_count"].tolist()))
    return z, cat, parent, mass, num_p, rec


def _tile_counts(z, parent, num_p, n_tiles, min_p, *, subs: bool):
    """Fractional per-tile count of subhalos (``subs``) or top-level hosts."""
    hid, tid, w = z["halo_id"], z["tile_id"], z["weight"]
    keep = np.array([(parent.get(int(h), -1) >= 0) == subs
                     and num_p.get(int(h), 0) >= min_p for h in hid])
    return np.bincount(tid[keep], weights=w[keep], minlength=n_tiles)


def _descendant_counts(cat, roots) -> np.ndarray:
    """Descendants of every catalog row, aligned with the rows themselves.

    Non-zero only for rows that are somebody's top-level ancestor. Built through
    an explicit argsort rather than assuming the catalog's ids arrive sorted --
    a wrong assumption here would silently attribute substructure to the wrong
    hosts and every abundance number below would be quietly wrong.
    """
    order = np.argsort(cat.ids)
    sid = cat.ids[order]
    sub_roots = roots[cat.parent_ids >= 0]
    pos = np.searchsorted(sid, sub_roots)
    if sub_roots.size and not np.array_equal(sid[np.minimum(pos, sid.size - 1)],
                                             sub_roots):
        raise ValueError("a subhalo's top-level ancestor is not in the catalog")
    out = np.zeros(cat.n, dtype=float)
    out[order] = np.bincount(pos, minlength=sid.size).astype(float)
    return out


def _abundance(cat, roots, edges):
    """Subhalos per host, and per unit host mass, by host mass bin."""
    is_host = cat.parent_ids < 0
    hmass = cat.mvir[is_host]
    ndesc = _descendant_counts(cat, roots)[is_host]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (np.log10(np.maximum(hmass, 1.0)) >= a) & (
            np.log10(np.maximum(hmass, 1.0)) < b)
        if m.sum() < 3:
            continue
        out.append({"lo": float(a), "hi": float(b), "n_hosts": int(m.sum()),
                    "mean_n_sub": float(ndesc[m].mean()),
                    "sub_per_1e12_msun": float(ndesc[m].sum()
                                               / (hmass[m].sum() / 1e12))})
    big = hmass > 1e13
    slope = (float(np.polyfit(np.log10(hmass[big]),
                              np.log10(np.maximum(ndesc[big], 0.5)), 1)[0])
             if big.sum() >= 3 else None)
    return out, slope, hmass, ndesc


def analyze(box, cfg, args):
    banner(f"SR2 subhalo deficit {box}")
    d = cfg.get("data", {})
    ng_hr = int(d.get("ng_hr", 512))
    box_l = float(d.get("boxsize_mpc_h", 100.0))
    per_tile = int(cfg.get("tiles", {}).get("tile_hr", 64)) ** 3
    n_tiles = (ng_hr ** 3) // per_tile
    root = paths.subdir("halos_particles")

    zh, cath, parh, massh, nph, rech = _load(root, box, args.hr_tag)
    zs, cats, pars, masss, nps, recs = _load(root, box, args.sr2_tag)
    rooth, roots_ = _roots(cath), _roots(cats)

    # --- 1. abundance vs host mass, BOTH sides ----------------------------
    # HR: flat subs per unit mass = substructure spread evenly over host mass.
    # SR2: the same measurement, which is where the failure actually shows.
    abund = {}
    for name, cat, rts in (("hr", cath, rooth), ("sr2", cats, roots_)):
        bins, slope, _, _ = _abundance(cat, rts, MASS_EDGES)
        abund[name] = {"bins": bins, "loglog_slope_resolved": slope}

    # --- 2. abundance functions over the whole mass range ------------------
    def counts_by_mass(cat):
        m = np.log10(np.maximum(cat.mvir[cat.parent_ids < 0], 1.0))
        return [int(((m >= a) & (m < b)).sum())
                for a, b in zip(MASS_EDGES[:-1], MASS_EDGES[1:])]

    hmf = [{"lo": float(a), "hi": float(b), "hr": nh, "sr2": ns,
            "ratio": float(ns / max(nh, 1))}
           for a, b, nh, ns in zip(MASS_EDGES[:-1], MASS_EDGES[1:],
                                   counts_by_mass(cath), counts_by_mass(cats))]

    def by_num_p(cat, subs: bool):
        sel = (cat.parent_ids >= 0) if subs else (cat.parent_ids < 0)
        n = cat.num_p[sel]
        return [int(((n >= a) & (n < b)).sum())
                for a, b in zip(NP_EDGES[:-1], NP_EDGES[1:])]

    npf = []
    for k, (a, b) in enumerate(zip(NP_EDGES[:-1], NP_EDGES[1:])):
        row = {"lo": int(a), "hi": int(b)}
        for what, subs in (("host", False), ("sub", True)):
            nh = by_num_p(cath, subs)[k]
            ns = by_num_p(cats, subs)[k]
            row[f"hr_{what}"] = nh
            row[f"sr2_{what}"] = ns
            row[f"ratio_{what}"] = float(ns / max(nh, 1))
        npf.append(row)

    def split_at(cat, edge=11.5):
        m = np.log10(np.maximum(cat.mvir[cat.parent_ids < 0], 1.0))
        return int((m < edge).sum()), int((m >= edge).sum())

    lo_h, hi_h = split_at(cath)
    lo_s, hi_s = split_at(cats)
    coverage = {
        "edge": 11.5,
        "hr_hosts_below": lo_h, "sr2_hosts_below": lo_s,
        "ratio_below": float(lo_s / max(lo_h, 1)),
        "hr_hosts_above": hi_h, "sr2_hosts_above": hi_s,
        "ratio_above": float(hi_s / max(hi_h, 1)),
        "frac_of_hosts_below": float(lo_h / max(lo_h + hi_h, 1)),
    }

    # --- 3. particle-floor sweep ------------------------------------------
    feat = LagrangianHostFeatures.from_npz(
        paths.subdir("lagrangian_host", box) / f"{box}_lagrangian_host.npz")
    g = feat.grid
    st = g.tile_of_lr_site(np.arange(g.n_lr))
    lr_occ = (np.bincount(st, weights=(feat.host_member > 0).reshape(-1).astype(float),
                          minlength=n_tiles) / (g.tile_lr ** 3))
    sweep = []
    for mp in (0, 20, 50, 100, 200, 500):
        hr = _tile_counts(zh, parh, nph, n_tiles, mp, subs=True)
        sr = _tile_counts(zs, pars, nps, n_tiles, mp, subs=True)
        ok = hr >= 1
        rel = (sr[ok] - hr[ok]) / np.maximum(hr[ok], 1e-9)
        sweep.append({
            "min_particles": mp,
            "hr_total": float(hr.sum()), "sr2_total": float(sr.sum()),
            "ratio": float(sr.sum() / max(hr.sum(), 1e-9)),
            "spearman_lrocc_hr": _spearman(lr_occ, hr),
            "spearman_lrocc_sr2": _spearman(lr_occ, sr),
            # Not independent evidence: rel = sr/hr - 1, so with the SR2 count
            # flat and the HR count rising in occupancy this is negative by
            # arithmetic. Kept for continuity, read with that in mind.
            "spearman_lrocc_deficit": _spearman(lr_occ[ok], rel),
        })

    # --- 4. local fraction vs total host mass ------------------------------
    hid, tid, w = zh["halo_id"], zh["tile_id"], zh["weight"]
    isroot = np.array([parh.get(int(h), -1) < 0 for h in hid])
    rec = np.array([rech.get(int(h), 0) for h in hid], float)
    hm = np.array([massh.get(int(h), 0) for h in hid], float)
    L = np.bincount(tid[isroot], weights=(w * rec)[isroot], minlength=n_tiles) / per_tile
    num = np.bincount(tid[isroot], weights=(w * rec * hm)[isroot], minlength=n_tiles)
    den = np.bincount(tid[isroot], weights=(w * rec)[isroot], minlength=n_tiles)
    logM = np.log10(np.maximum(num / np.maximum(den, 1e-9), 1.0))
    hr_sub = _tile_counts(zh, parh, nph, n_tiles, 0, subs=True)
    sr_sub = _tile_counts(zs, pars, nps, n_tiles, 0, subs=True)
    hr_host = _tile_counts(zh, parh, nph, n_tiles, 0, subs=False)
    sr_host = _tile_counts(zs, pars, nps, n_tiles, 0, subs=False)

    def controlled(y):
        return {"corr_L": float(np.corrcoef(y, L)[0, 1]),
                "corr_logM": float(np.corrcoef(y, logM)[0, 1]),
                "partial_logM_given_L": _partial(y, logM, L),
                "partial_L_given_logM": _partial(y, L, logM)}

    context = {"corr_L_logM": float(np.corrcoef(L, logM)[0, 1]),
               "hr_sub": controlled(hr_sub), "sr2_sub": controlled(sr_sub),
               "hr_host": controlled(hr_host), "sr2_host": controlled(sr_host)}

    # 2D bin table (L rows x logM cols) with the cell counts: the tertiles of two
    # variables correlated at 0.75 leave corner cells with a handful of tiles,
    # and a mean over 7 tiles should not be read like a mean over 116.
    Le, Me = np.quantile(L, [0, 1/3, 2/3, 1]), np.quantile(logM, [0, 1/3, 2/3, 1])

    def masks():
        for i in range(3):
            lm = (L >= Le[i]) & (L <= Le[i+1] if i == 2 else L < Le[i+1])
            row = []
            for j in range(3):
                mm = (logM >= Me[j]) & (logM <= Me[j+1] if j == 2 else logM < Me[j+1])
                row.append(lm & mm)
            yield row

    def grid_of(y):
        return [[float(y[s].mean()) if s.any() else None for s in row]
                for row in masks()]

    twod = {"L_edges": [float(x) for x in Le], "logM_edges": [float(x) for x in Me],
            "hr": grid_of(hr_sub), "sr2": grid_of(sr_sub),
            "n_tiles": [[int(s.sum()) for s in row] for row in masks()]}

    # --- 5. per-tile object counts by L ------------------------------------
    q5 = np.quantile(L, np.linspace(0, 1, 6))
    per_tile_rows = []
    for i in range(5):
        m = (L >= q5[i]) & (L <= q5[i+1] if i == 4 else L < q5[i+1])
        row = {"lo": float(q5[i]), "hi": float(q5[i+1]), "n_tiles": int(m.sum()),
               "L_mean": float(L[m].mean()), "logM_mean": float(logM[m].mean())}
        for name, y in (("hr_host", hr_host), ("hr_sub", hr_sub),
                        ("sr2_host", sr_host), ("sr2_sub", sr_sub)):
            row[name] = float(y[m].mean())
        row["ratio_host"] = row["sr2_host"] / max(row["hr_host"], 1e-9)
        row["ratio_sub"] = row["sr2_sub"] / max(row["hr_sub"], 1e-9)
        row["ratio_all"] = ((row["sr2_host"] + row["sr2_sub"])
                            / max(row["hr_host"] + row["hr_sub"], 1e-9))
        per_tile_rows.append(row)

    # --- 6. mean subhalo size, with the survivorship control ---------------
    def sub_rows(z, parent, num_p):
        hid, tid, w = z["halo_id"], z["tile_id"], z["weight"]
        k = np.array([parent.get(int(h), -1) >= 0 for h in hid])
        cnt = np.array([num_p.get(int(h), 0) for h in hid], float)
        return tid[k], w[k], cnt[k]

    th, wh, nh_ = sub_rows(zh, parh, nph)
    ts, ws, ns_ = sub_rows(zs, pars, nps)
    q4 = np.quantile(lr_occ, [0, 0.25, 0.5, 0.75, 1.0])
    size = []
    for i in range(4):
        tiles = np.flatnonzero((lr_occ >= q4[i])
                               & (lr_occ <= q4[i+1] if i == 3 else lr_occ < q4[i+1]))
        mh, ms = np.isin(th, tiles), np.isin(ts, tiles)
        Nh, Ns = float(wh[mh].sum()), float(ws[ms].sum())
        mean_h = float((wh[mh] * nh_[mh]).sum() / max(Nh, 1e-9))
        mean_s = float((ws[ms] * ns_[ms]).sum() / max(Ns, 1e-9))
        # The control: HR's own mean if you keep only its Ns largest subhalos.
        # Merging would put SR2 *above* that line; attrition lands on it.
        o = np.argsort(-nh_[mh])
        cw = np.cumsum(wh[mh][o])
        cut = min(int(np.searchsorted(cw, Ns)), cw.size - 1)
        top_h = float((wh[mh][o][:cut+1] * nh_[mh][o][:cut+1]).sum()
                      / max(cw[cut], 1e-9))
        size.append({
            "lrocc_lo": float(q4[i]), "lrocc_hi": float(q4[i+1]),
            "hr_n_sub": Nh, "sr2_n_sub": Ns,
            "hr_mean_size": mean_h, "sr2_mean_size": mean_s,
            "ratio": float(mean_s / max(mean_h, 1e-9)),
            "hr_mean_size_top_n": top_h,
            "ratio_vs_top_n": float(mean_s / max(top_h, 1e-9)),
        })

    # --- 6b. subhalo size function inside massive hosts --------------------
    # The other half of the merging test: merging would put SR2 *above* HR at
    # large num_p. It is below at every size, and holds less subhalo mass, so
    # the missing substructure went into the smooth host body, not into bigger
    # satellites.
    inside = []
    for thr in (1e13, 1e14):
        row = {"host_mvir_above": float(thr)}
        for name, cat, rts in (("hr", cath, rooth), ("sr2", cats, roots_)):
            sub = cat.parent_ids >= 0
            m_root = np.zeros(cat.n)
            mass_of = dict(zip(cat.ids.tolist(), cat.mvir.tolist()))
            m_root[sub] = [mass_of.get(int(r), 0.0) for r in rts[sub]]
            sel = sub & (m_root > thr)
            npv = cat.num_p[sel]
            row[f"{name}_n_sub"] = int(sel.sum())
            row[f"{name}_hist"] = [
                int(((npv >= a) & (npv < b)).sum())
                for a, b in zip([20, 50, 100, 200, 500, 2000],
                                [50, 100, 200, 500, 2000, 10 ** 9])]
            row[f"{name}_sub_particles"] = int(npv.sum())
        row["ratio_sub_particles"] = float(
            row["sr2_sub_particles"] / max(row["hr_sub_particles"], 1))
        row["hist_bins"] = [20, 50, 100, 200, 500, 2000]
        inside.append(row)

    # --- 7. matched clusters ----------------------------------------------
    hh = np.flatnonzero(cath.parent_ids < 0)
    hb = np.flatnonzero(cats.parent_ids < 0)
    big = hh[cath.mvir[hh] > float(args.match_mass)]
    big = big[np.argsort(-cath.mvir[big])]
    nsub_h = _descendant_counts(cath, rooth)
    nsub_s = _descendant_counts(cats, roots_)
    matched = []
    for k in big:
        dx = np.abs(cats.pos[hb] - cath.pos[k])
        dx = np.minimum(dx, box_l - dx)
        r = np.sqrt((dx ** 2).sum(1))
        j = hb[int(np.argmin(r))]
        matched.append({
            "hr_id": int(cath.ids[k]), "sr2_id": int(cats.ids[j]),
            "hr_log_mvir": float(np.log10(cath.mvir[k])),
            "sr2_log_mvir": float(np.log10(max(cats.mvir[j], 1.0))),
            "dist_mpc_h": float(r.min()),
            "hr_rvir": float(cath.rvir[k]), "sr2_rvir": float(cats.rvir[j]),
            "hr_n_sub": int(nsub_h[k]),
            "sr2_n_sub": int(nsub_s[j]),
        })
    match_summary = None
    if matched:
        a = np.array([[m["hr_log_mvir"], m["sr2_log_mvir"], m["dist_mpc_h"],
                       m["hr_n_sub"], m["sr2_n_sub"]] for m in matched])
        ok = a[:, 2] < float(args.match_radius)
        match_summary = {
            "mass_threshold": float(args.match_mass),
            "n": len(matched), "n_within_radius": int(ok.sum()),
            "match_radius_mpc_h": float(args.match_radius),
            "median_dist_mpc_h": float(np.median(a[:, 2])),
            "median_mass_ratio": float(np.median(10 ** (a[:, 1] - a[:, 0]))),
            "median_rvir_ratio": float(np.median(
                [m["sr2_rvir"] / max(m["hr_rvir"], 1e-9) for m in matched])),
            "hr_n_sub": int(a[:, 3].sum()), "sr2_n_sub": int(a[:, 4].sum()),
            "ratio": float(a[:, 4].sum() / max(a[:, 3].sum(), 1)),
        }

    out = {
        "box": box, "n_tiles": n_tiles,
        "n_hosts_hr": int((cath.parent_ids < 0).sum()),
        "n_hosts_sr2": int((cats.parent_ids < 0).sum()),
        "n_subs_hr": int((cath.parent_ids >= 0).sum()),
        "n_subs_sr2": int((cats.parent_ids >= 0).sum()),
        "abundance_vs_mass": abund,
        "host_mass_function": hmf,
        "num_p_function": npf,
        "mass_coverage": coverage,
        "particle_floor_sweep": sweep,
        "local_vs_global": context,
        "twod_bins": twod,
        "per_tile_by_L": per_tile_rows,
        "size_by_occupancy_quartile": size,
        "sub_size_inside_massive_hosts": inside,
        "matched_massive_hosts": {"summary": match_summary, "rows": matched},
    }
    dest = paths.subdir("lagrangian_host", box, create=True) \
        / f"{box}_subhalo_deficit.json"
    write_json(dest, out)

    print(f"    hosts   HR {out['n_hosts_hr']}  SR2 {out['n_hosts_sr2']}"
          f"  ({out['n_hosts_sr2']/out['n_hosts_hr']:.2f})")
    print(f"    subs    HR {out['n_subs_hr']}  SR2 {out['n_subs_sr2']}"
          f"  ({out['n_subs_sr2']/out['n_subs_hr']:.2f})")
    for name in ("hr", "sr2"):
        b = abund[name]["bins"]
        print(f"    {name:3s} subs per 1e12 Msun by host mass: "
              + " ".join(f"{r['sub_per_1e12_msun']:.2f}" for r in b)
              + f"   log-log slope (>1e13) {abund[name]['loglog_slope_resolved']:.2f}")
    print(f"    hosts below 1e11.5 ({100*coverage['frac_of_hosts_below']:.0f}% of "
          f"all): SR2/HR {coverage['ratio_below']:.2f}; above: "
          f"{coverage['ratio_above']:.2f}")
    print("    floor sweep SR2/HR: "
          + " ".join(f"{s['min_particles']}:{s['ratio']:.2f}" for s in sweep))
    c = context
    print(f"    corr(L, logM) = {c['corr_L_logM']:+.2f}; SR2 subs: corr_L "
          f"{c['sr2_sub']['corr_L']:+.2f} splits into partial(logM|L) "
          f"{c['sr2_sub']['partial_logM_given_L']:+.2f} and partial(L|logM) "
          f"{c['sr2_sub']['partial_L_given_logM']:+.2f}")
    print(f"    HR subs:  partial(logM|L) {c['hr_sub']['partial_logM_given_L']:+.2f}"
          f"  partial(L|logM) {c['hr_sub']['partial_L_given_logM']:+.2f}")
    print("    per-tile SR2/HR by L quintile: hosts "
          + " ".join(f"{r['ratio_host']:.2f}" for r in per_tile_rows)
          + " | subs " + " ".join(f"{r['ratio_sub']:.2f}" for r in per_tile_rows))
    print("    subhalo size SR2/HR by density quartile: "
          + " ".join(f"{s['ratio']:.2f}" for s in size)
          + "  | vs HR's own top-N: "
          + " ".join(f"{s['ratio_vs_top_n']:.2f}" for s in size))
    for r in inside:
        print(f"    subs inside hosts >{r['host_mvir_above']:.0e} by num_p "
              f"{r['hist_bins']}: HR {r['hr_hist']} SR2 {r['sr2_hist']}; "
              f"subhalo particles SR2/HR {r['ratio_sub_particles']:.2f}")
    if match_summary:
        m = match_summary
        print(f"    matched hosts >{m['mass_threshold']:.0e}: {m['n_within_radius']}"
              f"/{m['n']} within {m['match_radius_mpc_h']} Mpc/h, median mass "
              f"ratio {m['median_mass_ratio']:.2f}, subhalos HR {m['hr_n_sub']} "
              f"vs SR2 {m['sr2_n_sub']} (ratio {m['ratio']:.3f})")
    print(f"    wrote {dest}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    ap.add_argument("--hr-tag", default="hr")
    ap.add_argument("--sr2-tag", default="base")
    ap.add_argument("--match-mass", type=float, default=1e14,
                    help="match HR hosts above this Mvir to their nearest SR2 "
                         "host, object by object (default 1e14)")
    ap.add_argument("--match-radius", type=float, default=0.5,
                    help="Mpc/h; a match beyond this is reported but counted as "
                         "unmatched in the summary (default 0.5)")
    args = ap.parse_args(argv)
    cfg = load_reward_config(args)
    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        analyze(box, cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
