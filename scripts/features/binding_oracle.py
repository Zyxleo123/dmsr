#!/usr/bin/env python
"""Binding oracle: make each outskirt bump self-bound by its OWN gravity, then run the halo finder.

The bump census (``host_phase_profile.py --peaks --bind``, ``peak_supply.py``)
found that frozen and fine-tuned fields hold HR's number of overdensity bumps
at 0.5-3 R_vir, and that HR's satellites sit on them, but SR bumps rarely
become objects. The knot oracle cooled each bump to 0.3x the dispersion of
the annulus around it and recovered only part of the gap; faint bumps stayed
unbound. Its target is relative to the HOST's local temperature: next to a
massive host that is still ~100-250 km/s, while a 32-128 particle clump's own
escape speed is tens of km/s. So the knot oracle could never bind small
clumps, which are most of HR's outskirt objects.

This oracle sets the target from the bump's own gravity instead. Members are
the bump's ``k_obj`` nearest particles, ``k_obj`` set per contrast band to the
size of the HR objects that sit on such bumps (``--aperture band``, default
64/64/128/256 against HR medians 47/65/81/196 p), and the frame v_b is the mean
velocity of the ``--core`` particles nearest the peak. The first version
(2026-10-15, ``--aperture density --core 0``: largest k still 3x the shell
mean, aperture-mean frame) FAILED its HR check -- k came out 32 or 512, and
"needs KE removed" predicted HR's own conversion at AUC 0.45-0.56 in every
band. With phi_i the softened pairwise potential of the members on each
other:

    v_i' = v_b + (v_i - v_b) * min(1, f * sqrt(2 |phi_i|) / |v_i - v_b|)

i.e. every member faster than f x its escape speed is slowed to it, direction
kept; already-bound members are untouched; positions are never touched. (A few
passes, because the clip moves v_b.) That is the projection onto "this bump is
self-bound", so the training cost it maps onto is the distance to it
(``escape_excess``).

``--rate all`` binds every bump. ``--rate hr`` binds, per contrast band, only
the fraction HR itself converts into Rockstar objects (same x-range, same
match as peak_supply), choosing the coldest relative to their surroundings
first (``--select cold``: the census ``coldbg``, which is HR's own conversion
rule -- AUC 0.66/0.75/0.83 in the 10-30/30-100/>100x bands -- while SR's
conversion ignores it), the least kinetic energy removed (``easy``) or at
random. ``all`` is
the ceiling; ``hr`` asks whether binding at HR's own rate gives HR's objects.
``--stats-only`` measures the per-bump criterion and stops (no field, no
finder): run on HR it says whether "unbound by own gravity" separates HR's
converting bumps from the rest, i.e. whether the cost's criterion is the right
one.

Peaks come from ``host_phase_profile.py --peaks`` on the SAME field with the
same host selection (``host`` indexes it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from host_phase_profile import (  # noqa: E402
    BOX, G_MPC, K_NN, SOFT_MPC, CellIndex, select_region_hosts,
)
from peak_halo_match import match  # noqa: E402
from peak_supply import BANDS, band_of  # noqa: E402
from placement_forensics import min_image, read_rockstar  # noqa: E402
from cosmo_sr.eval.particles import ParticleBox, field_to_particles  # noqa: E402

FROZEN_CFG = PROJECT_ROOT / "configs" / "sr2_baseline" / "rockstar.cfg"


def self_potential(dx: np.ndarray, m_p: float, soft: float = SOFT_MPC) -> np.ndarray:
    """Softened pairwise potential (km/s)^2 of each member on the others; <= 0."""
    dd = dx[:, None, :] - dx[None, :, :]
    inv = 1.0 / np.sqrt(np.einsum("ijk,ijk->ij", dd, dd) + soft * soft)
    np.fill_diagonal(inv, 0.0)
    return -G_MPC * m_p * inv.sum(axis=1)


def escape_clip(dx: np.ndarray, v: np.ndarray, m_p: float, f: float = 0.9,
                passes: int = 3, soft: float = SOFT_MPC, core: int = 0):
    """Slow every member above ``f`` x its escape speed to it, about the bulk velocity.

    ``core > 0``: the frame is the mean velocity of the first ``core`` rows (the
    particles nearest the peak) and stays fixed, so host material inside the
    aperture cannot drag the object's frame onto the local flow (the failure the
    first HR check showed: AUC ~0.5 with the aperture-mean frame). One pass is
    then exact. ``core = 0``: the frame is the members' mean, re-taken per pass.

    Returns ``(v_new, stats)``. ``stats``: ``f_unbound`` (members above escape
    speed before), ``ke_removed`` (fraction of the internal kinetic energy the
    clip removes; the per-bump 'how far from bound' score), ``tu_before`` /
    ``tu_after`` (internal T/|U|, the census's ``tu`` on these members).
    """
    phi = self_potential(dx, m_p, soft)
    vesc = np.sqrt(2.0 * np.abs(phi))
    v0 = np.asarray(v, dtype=np.float64)
    frame = v0[:core].mean(axis=0) if core > 0 else v0.mean(axis=0)
    u0 = v0 - frame
    ke0 = 0.5 * float(np.einsum("ij,ij->", u0, u0))
    w = v0.copy()
    for _ in range(1 if core > 0 else passes):
        vb = frame if core > 0 else w.mean(axis=0)
        u = w - vb
        s = np.sqrt(np.einsum("ij,ij->i", u, u))
        lim = f * vesc
        scale = np.where(s > lim, lim / np.maximum(s, 1e-30), 1.0)
        w = vb + u * scale[:, None]
    u1 = w - (frame if core > 0 else w.mean(axis=0))
    ke1 = 0.5 * float(np.einsum("ij,ij->", u1, u1))
    pot = 0.5 * float(np.abs(phi).sum())                 # |U|, each pair once
    s0 = np.sqrt(np.einsum("ij,ij->i", u0, u0))
    stats = {"f_unbound": float(np.mean(s0 > vesc)),
             "ke_removed": 1.0 - ke1 / ke0 if ke0 > 0 else 0.0,
             "tu_before": ke0 / pot if pot > 0 else np.nan,
             "tu_after": ke1 / pot if pot > 0 else np.nan}
    return w, stats


def aperture_k(dist: np.ndarray, n_shell: float, ladder, ap_contrast: float) -> int:
    """Largest ladder k whose enclosed mean density is >= ap_contrast x the shell mean.

    ``dist`` is a query from the peak particle itself (index 0 = self), so the
    k-th member including the peak sits at ``dist[k-1]`` (census convention).
    """
    best = int(ladder[0])
    for k in ladder:
        r = max(float(dist[k - 1]), 1e-9)
        if k / (4.0 / 3.0 * np.pi * r ** 3) >= ap_contrast * n_shell:
            best = int(k)
    return best


def select_bumps(band: np.ndarray, score: np.ndarray, rate: np.ndarray,
                 how: str, rng: np.random.Generator) -> np.ndarray:
    """Per band, choose round(rate[b] * n_b) bumps: lowest ``score`` first (NaN last), or at random."""
    chosen = np.zeros(band.size, dtype=bool)
    for b in range(rate.size):
        idx = np.where(band == b)[0]
        if idx.size == 0 or not np.isfinite(rate[b]):
            continue
        n = int(round(float(rate[b]) * idx.size))
        if n <= 0:
            continue
        if how in ("easy", "cold"):
            sc = np.where(np.isfinite(score[idx]), score[idx], np.inf)
            pick = idx[np.argsort(sc, kind="stable")[:n]]
        else:
            pick = rng.choice(idx, size=n, replace=False)
        chosen[pick] = True
    return chosen


def hr_rates(hr_peaks: str, hr_cat: dict, x_min: float, x_max: float,
             min_match: float, match_rvir: float) -> np.ndarray:
    """HR's per-band bump -> Rockstar (>=20p) conversion over the x-range (peak_supply's)."""
    pk = dict(np.load(hr_peaks))
    keep = hr_cat["num_p"] >= 20
    hit = match(pk["pos"], hr_cat["pos"][keep],
                hr_cat["rvir"][keep] / 1000.0 * match_rvir, min_match) >= 0
    sel = (pk["x"] >= x_min) & (pk["x"] < x_max) & (pk["ratio"] > BANDS[0])
    b = band_of(pk["ratio"][sel])
    n = np.bincount(b, minlength=4)[:4]
    nc = np.bincount(b, weights=hit[sel].astype(float), minlength=4)[:4]
    return np.where(n > 0, nc / np.maximum(n, 1), np.nan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--peaks", required=True, help="<field>_peaks.npz from the census")
    ap.add_argument("--hr-catalog", required=True)
    ap.add_argument("--region-export", required=True)
    ap.add_argument("--catalog", default="",
                    help="this field's own Rockstar catalog: each bump is flagged "
                         "converted/not, so the criterion can be scored against it")
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--min-host-num-p", type=int, default=10000)
    ap.add_argument("--max-hosts", type=int, default=60)
    ap.add_argument("--x-min", type=float, default=0.5)
    ap.add_argument("--x-max", type=float, default=3.0)
    ap.add_argument("--min-ratio", type=float, default=3.0)
    ap.add_argument("--k-ladder", default="32,64,128,256,512")
    ap.add_argument("--ap-contrast", type=float, default=3.0)
    ap.add_argument("--aperture", choices=("density", "band"), default="band",
                    help="'band': k_obj per contrast band from --band-k (HR's matched "
                         "object sizes: medians 47/65/81/196 p). 'density': largest "
                         "ladder k still --ap-contrast x the shell mean -- FAILED the HR "
                         "check (k is 32 or 512, AUC 0.45-0.56)")
    ap.add_argument("--band-k", default="64,64,128,256")
    ap.add_argument("--core", type=int, default=16,
                    help="frame = mean velocity of the core nearest the peak (0 = "
                         "aperture mean, the setting that failed the HR check)")
    ap.add_argument("--f-esc", type=float, default=0.9)
    ap.add_argument("--rate", choices=("all", "hr"), default="all")
    ap.add_argument("--select", choices=("cold", "bound", "easy", "random"), default="cold",
                    help="which bumps --rate hr binds per band: 'cold' = lowest coldbg "
                         "first (HR's own conversion rule, AUC 0.66-0.83 on HR); 'bound' "
                         "= highest bound fraction first (the rule BOTH fields follow, HR "
                         "AUC 0.69-0.88 / coarse_soft 0.60-0.82 -- what a hinge cost on it "
                         "would push); 'easy' "
                         "= least KE removed; 'random'")
    ap.add_argument("--hr-peaks", default="", help="HR's census peaks (needed for --rate hr)")
    ap.add_argument("--min-match", type=float, default=0.03)
    ap.add_argument("--match-rvir", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--save-field", default="",
                    help="also write the modified (6,N,N,N) field here (for SUBFIND)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    ladder = [int(k) for k in a.k_ladder.split(",")]
    band_k = [int(k) for k in a.band_k.split(",")]
    kmax = max(max(ladder), max(band_k), K_NN + 1)
    hr = read_rockstar(a.hr_catalog)
    sel = select_region_hosts(hr, a.region_export, a.dilate, a.min_host_num_p, a.max_hosts)
    rvir = hr["rvir"][sel] / 1000.0
    pk = dict(np.load(a.peaks))
    census = Path(a.peaks.replace("_peaks.npz", ".json"))
    if not census.is_file():
        raise SystemExit(f"no census json next to {a.peaks} to verify the host list")
    ids = json.loads(census.read_text())["per_host_scalars"]["host_id"]
    if [int(i) for i in hr["id"][sel]] != [int(i) for i in ids]:
        raise SystemExit(f"host list differs from the census that wrote {a.peaks}; "
                         "the peaks' host index would point at the wrong hosts")
    use = np.where((pk["x"] >= a.x_min) & (pk["x"] < a.x_max)
                   & (pk["ratio"] > a.min_ratio))[0]
    print(f"{sel.size} hosts; {use.size} of {pk['x'].size} bumps >{a.min_ratio:g}x at "
          f"{a.x_min}-{a.x_max} R_vir; ladder {ladder}, aperture >= {a.ap_contrast:g}x shell, "
          f"f_esc {a.f_esc}, rate {a.rate}/{a.select}", flush=True)

    conv = np.full(use.size, np.nan)
    if a.catalog:
        cat = read_rockstar(a.catalog)
        keep = cat["num_p"] >= 20
        conv = (match(pk["pos"][use], cat["pos"][keep],
                      cat["rvir"][keep] / 1000.0 * a.match_rvir, a.min_match) >= 0
                ).astype(float)

    from scipy.spatial import cKDTree
    field = np.load(a.field, mmap_mode="r")
    pb = field_to_particles(field, boxsize_kpc_h=BOX * 1000.0)
    pos, vel = pb.pos_mpc_h, pb.vel_kms
    m_p = float(pb.particle_mass_msun_h)
    ci = CellIndex(pos)
    print(f"particles + cell index in {time.time() - t0:.0f}s (m_p {m_p:.3g})", flush=True)

    # --- pass 1: every bump's members, clip and score from the ORIGINAL velocities
    n_u = use.size
    rec = {k: np.full(n_u, np.nan) for k in ("k_obj", "f_unbound", "ke_removed",
                                              "tu_before", "tu_after")}
    members, clipped = [None] * n_u, [None] * n_u
    for j, h in enumerate(sel):
        mine = np.where(pk["host"][use] == j)[0]
        if mine.size == 0:
            continue
        rv = float(rvir[j])
        reach = (a.x_max + 0.3) * rv + 0.5
        idx = ci.gather(hr["pos"][h], reach)
        d = min_image(pos[idx].astype(np.float64) - hr["pos"][h])
        keep = np.einsum("ij,ij->i", d, d) < reach * reach
        idx, d = idx[keep], d[keep]
        tree = cKDTree(d)
        p_rel = min_image(pk["pos"][use[mine]].astype(np.float64) - hr["pos"][h])
        dist, nbr = tree.query(p_rel, k=kmax, workers=-1)
        for q, m in enumerate(mine):
            r32 = max(float(dist[q, K_NN]), 1e-9)               # 32nd excl. the peak
            n_shell = K_NN / (4.0 / 3.0 * np.pi * r32 ** 3) / float(pk["ratio"][use[m]])
            if a.aperture == "band":
                k = band_k[int(band_of(np.asarray([pk["ratio"][use[m]]]))[0])]
            else:
                k = aperture_k(dist[q], n_shell, ladder, a.ap_contrast)
            row = nbr[q, :k]
            w, st = escape_clip(d[row] - d[row[0]], vel[idx[row]], m_p, a.f_esc,
                                core=a.core)
            rec["k_obj"][m] = k
            for key, val in st.items():
                rec[key][m] = val
            members[m] = idx[row]
            clipped[m] = w.astype(np.float32)
        print(f"  host {j + 1}/{sel.size}: {mine.size} bumps, {time.time() - t0:.0f}s",
              flush=True)

    band = band_of(pk["ratio"][use])
    rate = np.ones(4)
    if a.rate == "hr":
        if not a.hr_peaks:
            raise SystemExit("--rate hr needs --hr-peaks")
        rate = hr_rates(a.hr_peaks, hr, a.x_min, a.x_max, a.min_match, a.match_rvir)
    coldbg = pk["coldbg"][use].astype(float) if "coldbg" in pk else np.full(n_u, np.nan)
    score = {"cold": coldbg, "bound": rec["f_unbound"]}.get(a.select, rec["ke_removed"])
    how = "easy" if a.select in ("cold", "bound") else a.select
    chosen = select_bumps(band, score, rate, how, np.random.default_rng(a.seed))
    chosen &= np.array([m is not None for m in members])

    per_band = {}
    for b, lab in enumerate(f"{lo:g}-{hi:g}" for lo, hi in zip(BANDS[:-1], BANDS[1:])):
        mb = band == b
        row = {"n": int(mb.sum()), "rate_target": float(rate[b]),
               "n_bound": int((chosen & mb).sum())}
        for key in ("k_obj", "f_unbound", "ke_removed", "tu_before"):
            row[f"{key}_median"] = float(np.nanmedian(rec[key][mb])) if mb.any() else None
        if a.catalog and mb.any():
            c = conv[mb] > 0.5
            row["conversion"] = float(np.mean(c))
            for key in ("f_unbound", "ke_removed", "tu_before"):
                vals = rec[key][mb]
                row[f"{key}_median_converted"] = (float(np.nanmedian(vals[c]))
                                                  if c.any() else None)
                row[f"{key}_median_unconverted"] = (float(np.nanmedian(vals[~c]))
                                                    if (~c).any() else None)
            row["auc_bound_predicts_conversion"] = _auc(-rec["ke_removed"][mb], c)
            row["auc_coldbg_predicts_conversion"] = _auc(-coldbg[mb], c)
            row["auc_boundfrac_predicts_conversion"] = _auc(-rec["f_unbound"][mb], c)
        per_band[lab] = row
        print(f"  band {lab:>8s}: {json.dumps(row)}", flush=True)

    summary = {"name": a.name, "field": a.field, "peaks": a.peaks, "catalog": a.catalog,
               "x_range": [a.x_min, a.x_max], "min_ratio": a.min_ratio,
               "k_ladder": ladder, "ap_contrast": a.ap_contrast, "f_esc": a.f_esc,
               "aperture": a.aperture, "band_k": band_k, "core": a.core,
               "rate": a.rate, "select": a.select, "seed": a.seed,
               "n_bumps": int(n_u), "n_bound": int(chosen.sum()), "per_band": per_band}
    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{a.name}_bumps.npz", row=use, band=band, chosen=chosen,
             converted=conv, **rec)
    write_json(out / f"{a.name}_binding_oracle.json", summary)
    if a.stats_only:
        print(f"stats only; wrote {out}", flush=True)
        return 0

    # --- pass 2: apply, faintest first so the denser of two overlapping bumps wins
    v_new = vel.copy()
    changed = np.zeros(pos.shape[0], dtype=bool)
    for m in np.argsort(pk["ratio"][use], kind="stable"):
        if chosen[m]:
            v_new[members[m]] = clipped[m]
            changed[members[m]] = True
    dv = np.linalg.norm(v_new[changed] - vel[changed], axis=1) if changed.any() else np.zeros(0)
    summary["n_particles_changed"] = int(changed.sum())
    summary["dv_median_kms"] = float(np.median(dv)) if dv.size else None
    summary["frac_changed_moved"] = float(np.mean(dv > 1e-3)) if dv.size else None
    print(f"bound {int(chosen.sum())} bumps; {int(changed.sum())} particles in them, "
          f"median |dv| {summary['dv_median_kms']} km/s", flush=True)
    write_json(out / f"{a.name}_binding_oracle.json", summary)
    if a.save_field:
        from velocity_profile_oracle import field_with_velocities
        f2 = field_with_velocities(field, v_new)
        np.save(a.save_field, f2)
        del f2
        print(f"wrote modified field {a.save_field}", flush=True)
    del field
    from cosmo_sr.eval.rockstar import run_rockstar_on_particles
    pb2 = ParticleBox(pos_mpc_h=pos, vel_kms=v_new, ids=pb.ids,
                      boxsize_mpc_h=pb.boxsize_mpc_h, redshift=pb.redshift,
                      particle_mass_msun_h=pb.particle_mass_msun_h)
    t1 = time.time()
    cat = run_rockstar_on_particles(pb2, out, cfg=FROZEN_CFG, tag=a.name,
                                    overwrite=a.overwrite)
    snap = out / f"{a.name}.gadget2"
    if snap.is_file():
        snap.unlink()
    summary["result_catalog"] = str(cat.path)
    summary["rockstar_min"] = (time.time() - t1) / 60.0
    write_json(out / f"{a.name}_binding_oracle.json", summary)
    print(f"Rockstar {summary['rockstar_min']:.1f} min -> {cat.n} objects; {cat.path}")
    return 0


def _auc(score: np.ndarray, y: np.ndarray):
    """P(score of a random positive > a random negative); None if a class is empty."""
    ok = np.isfinite(score)
    s, y = score[ok], y[ok]
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


if __name__ == "__main__":
    raise SystemExit(main())
