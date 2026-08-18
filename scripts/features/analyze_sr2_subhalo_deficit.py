#!/usr/bin/env python
"""Where SR2 loses subhalos, and what kind of failure it is.

Reads only the committed ``*_tilew.npz`` weights and the Rockstar catalogs that
are already on disk -- no new halo finding, no owner arrays, no GPU. Emits one
JSON per box under ``$DMSR_REWARD_ROOT/lagrangian_host/<box>/`` so the tables in
``docs/sr2_subhalo_deficit.md`` are redrawable rather than transcribed.

It answers three questions, each separating a candidate mechanism from a
competing one:

1. **Host mass function** -- does SR2 turn big hosts into small ones? (No: the
   host counts match HR at every mass bin. The hosts are right; only their
   interiors are missing.)
2. **Subhalo particle-floor sweep** -- is the deficit small subhalos or large
   ones? (Small: the SR2/HR ratio rises from 0.46 to 0.86 as the floor rises,
   and the anticorrelation with density flips sign above ~200 particles.)
3. **Local fraction vs total host mass** -- is SR2 a context-blind local model,
   or does it fail *worse* on fragments of big hosts? (The latter: at fixed
   local host fraction, the SR2 count still falls with total host mass -- a
   resolution/merging signature, not a context one.)

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


def _spearman(a, b):
    """Rank correlation without a scipy dependency in the reward path."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _partial(y, a, b):
    """corr(y, a | b): both residualised on b by least squares, then correlated."""
    A = np.vstack([b, np.ones_like(b)]).T
    def res(v):
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return float(np.corrcoef(res(y), res(a))[0, 1])


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


def _tile_subhalos(z, parent, num_p, n_tiles, min_p):
    hid, tid, w = z["halo_id"], z["tile_id"], z["weight"]
    keep = np.array([parent.get(int(h), -1) >= 0 and num_p.get(int(h), 0) >= min_p
                     for h in hid])
    return np.bincount(tid[keep], weights=w[keep], minlength=n_tiles)


def analyze(box, cfg, args):
    banner(f"SR2 subhalo deficit {box}")
    d = cfg.get("data", {})
    ng_hr = int(d.get("ng_hr", 512))
    per_tile = int(cfg.get("tiles", {}).get("tile_hr", 64)) ** 3
    n_tiles = (ng_hr ** 3) // per_tile
    root = paths.subdir("halos_particles")

    zh, cath, parh, massh, nph, rech = _load(root, box, args.hr_tag)
    zs, cats, pars, masss, nps, recs = _load(root, box, args.sr2_tag)

    # --- 1. host mass function -------------------------------------------
    edges = np.array([11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.0])
    def host_mvir(cat):
        return cat.mvir[cat.parent_ids < 0]
    hmf = []
    for a, b in zip(edges[:-1], edges[1:]):
        nh = int(((np.log10(host_mvir(cath)) >= a)
                  & (np.log10(host_mvir(cath)) < b)).sum())
        ns = int(((np.log10(host_mvir(cats)) >= a)
                  & (np.log10(host_mvir(cats)) < b)).sum())
        hmf.append({"lo": float(a), "hi": float(b), "hr": nh, "sr2": ns,
                    "ratio": float(ns / max(nh, 1))})

    # --- 2. particle-floor sweep -----------------------------------------
    feat = LagrangianHostFeatures.from_npz(
        paths.subdir("lagrangian_host", box) / f"{box}_lagrangian_host.npz")
    g = feat.grid
    st = g.tile_of_lr_site(np.arange(g.n_lr))
    lr_occ = (np.bincount(st, weights=(feat.host_member > 0).reshape(-1).astype(float),
                          minlength=n_tiles) / (g.tile_lr ** 3))
    sweep = []
    for mp in (0, 20, 50, 100, 200, 500):
        hr = _tile_subhalos(zh, parh, nph, n_tiles, mp)
        sr = _tile_subhalos(zs, pars, nps, n_tiles, mp)
        ok = hr >= 1
        rel = (sr[ok] - hr[ok]) / np.maximum(hr[ok], 1e-9)
        sweep.append({
            "min_particles": mp,
            "hr_total": float(hr.sum()), "sr2_total": float(sr.sum()),
            "ratio": float(sr.sum() / max(hr.sum(), 1e-9)),
            "spearman_lrocc_hr": _spearman(lr_occ, hr),
            "spearman_lrocc_sr2": _spearman(lr_occ, sr),
            "spearman_lrocc_deficit": _spearman(lr_occ[ok], rel),
        })

    # --- 3. local fraction vs total host mass ----------------------------
    hid, tid, w = zh["halo_id"], zh["tile_id"], zh["weight"]
    isroot = np.array([parh.get(int(h), -1) < 0 for h in hid])
    rec = np.array([rech.get(int(h), 0) for h in hid], float)
    hm = np.array([massh.get(int(h), 0) for h in hid], float)
    L = np.bincount(tid[isroot], weights=(w * rec)[isroot], minlength=n_tiles) / per_tile
    num = np.bincount(tid[isroot], weights=(w * rec * hm)[isroot], minlength=n_tiles)
    den = np.bincount(tid[isroot], weights=(w * rec)[isroot], minlength=n_tiles)
    logM = np.log10(np.maximum(num / np.maximum(den, 1e-9), 1.0))
    hr_sub = _tile_subhalos(zh, parh, nph, n_tiles, 0)
    sr_sub = _tile_subhalos(zs, pars, nps, n_tiles, 0)

    def controlled(y):
        return {"corr_L": float(np.corrcoef(y, L)[0, 1]),
                "corr_logM": float(np.corrcoef(y, logM)[0, 1]),
                "partial_logM_given_L": _partial(y, logM, L)}
    context = {"hr": controlled(hr_sub), "sr2": controlled(sr_sub)}

    # 2D bin table (L rows x logM cols), and mean subhalo size by L quartile
    Le, Me = np.quantile(L, [0, 1/3, 2/3, 1]), np.quantile(logM, [0, 1/3, 2/3, 1])
    def grid_of(y):
        out = []
        for i in range(3):
            lm = (L >= Le[i]) & (L <= Le[i+1] if i == 2 else L < Le[i+1])
            row = []
            for j in range(3):
                mm = (logM >= Me[j]) & (logM <= Me[j+1] if j == 2 else logM < Me[j+1])
                sel = lm & mm
                row.append(float(y[sel].mean()) if sel.any() else None)
            out.append(row)
        return out
    twod = {"L_edges": [float(x) for x in Le], "logM_edges": [float(x) for x in Me],
            "hr": grid_of(hr_sub), "sr2": grid_of(sr_sub)}

    def mean_sub_size(z, parent, num_p):
        hid, tid, w = z["halo_id"], z["tile_id"], z["weight"]
        k = np.array([parent.get(int(h), -1) >= 0 for h in hid])
        cnt = np.array([num_p.get(int(h), 0) for h in hid], float)
        n = np.bincount(tid[k], weights=w[k], minlength=n_tiles)
        m = np.bincount(tid[k], weights=(w * cnt)[k], minlength=n_tiles)
        return m / np.maximum(n, 1e-9)
    mh, ms = mean_sub_size(zh, parh, nph), mean_sub_size(zs, pars, nps)
    q = np.quantile(lr_occ, [0, 0.25, 0.5, 0.75, 1.0])
    size = []
    for i in range(4):
        m = (lr_occ >= q[i]) & (lr_occ <= q[i+1] if i == 3 else lr_occ < q[i+1])
        size.append({"lrocc_lo": float(q[i]), "lrocc_hi": float(q[i+1]),
                     "hr_mean_size": float(mh[m].mean()),
                     "sr2_mean_size": float(ms[m].mean()),
                     "ratio": float(ms[m].mean() / mh[m].mean())})

    # --- subhalo abundance vs host mass (is N_sub ~ linear in M_host?) ----
    from cosmo_sr.eval.particle_identity import child_map, descendants_of
    kids = child_map(cath)
    is_host = cath.parent_ids < 0
    hmass = cath.mvir[is_host]
    ndesc = np.array([len(descendants_of(cath, int(h), children=kids))
                      for h in cath.ids[is_host]], dtype=float)
    abund = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (np.log10(hmass) >= a) & (np.log10(hmass) < b)
        if m.sum() < 3:
            continue
        abund.append({"lo": float(a), "hi": float(b), "n_hosts": int(m.sum()),
                      "mean_n_sub": float(ndesc[m].mean()),
                      "sub_per_1e12_msun": float(ndesc[m].sum()
                                                 / (hmass[m].sum() / 1e12))})
    ok = (ndesc >= 1) & (hmass > 0)
    slope_all = float(np.polyfit(np.log10(hmass[ok]), np.log10(ndesc[ok]), 1)[0])
    big = ok & (hmass > 1e13)
    slope_resolved = (float(np.polyfit(np.log10(hmass[big]),
                                       np.log10(ndesc[big]), 1)[0])
                      if big.sum() >= 3 else None)

    out = {
        "box": box, "n_tiles": n_tiles,
        "n_hosts_hr": int((cath.parent_ids < 0).sum()),
        "n_hosts_sr2": int((cats.parent_ids < 0).sum()),
        "n_subs_hr": int((cath.parent_ids >= 0).sum()),
        "n_subs_sr2": int((cats.parent_ids >= 0).sum()),
        "host_mass_function": hmf,
        "particle_floor_sweep": sweep,
        "local_vs_global": context,
        "twod_bins": twod,
        "size_by_occupancy_quartile": size,
        "abundance_vs_mass": {
            "bins": abund,
            "loglog_slope_all_hosts": slope_all,
            "loglog_slope_resolved": slope_resolved,
        },
    }
    dest = paths.subdir("lagrangian_host", box, create=True) \
        / f"{box}_subhalo_deficit.json"
    write_json(dest, out)

    hh = out; print(f"    hosts   HR {hh['n_hosts_hr']}  SR2 {hh['n_hosts_sr2']}"
                    f"  ({hh['n_hosts_sr2']/hh['n_hosts_hr']:.2f})")
    print(f"    subs    HR {hh['n_subs_hr']}  SR2 {hh['n_subs_sr2']}"
          f"  ({hh['n_subs_sr2']/hh['n_subs_hr']:.2f})")
    print(f"    host mass function SR2/HR by bin: "
          + " ".join(f"{r['ratio']:.2f}" for r in hmf))
    print(f"    floor sweep SR2/HR: "
          + " ".join(f"{s['min_particles']}:{s['ratio']:.2f}" for s in sweep))
    print(f"    partial(logM|L): HR {context['hr']['partial_logM_given_L']:+.2f}"
          f"  SR2 {context['sr2']['partial_logM_given_L']:+.2f}")
    print(f"    N_sub/M_host slope: all {slope_all:.2f}  resolved(>1e13) "
          f"{slope_resolved:.2f}  (1.0=linear); subs/1e12Msun flat "
          f"{abund[0]['sub_per_1e12_msun']:.1f}->{abund[-1]['sub_per_1e12_msun']:.1f}")
    print(f"    subhalo size SR2/HR by density quartile: "
          + " ".join(f"{s['ratio']:.2f}" for s in size))
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
    args = ap.parse_args(argv)
    cfg = load_reward_config(args)
    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        analyze(box, cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
