#!/usr/bin/env python
"""SR2 halo/subhalo abundance, occupancy and positional-fidelity report.

Reads the frozen Stage-1 Rockstar catalogs (HR + one catalog per SR noise seed)
and answers three questions separately:

1. **Abundance** -- does SR2 make the right *number* of halos/subhalos?
   (host HMF, subhalo SHMF, Vmax function, all as HR/SR ratios)
2. **Occupancy** -- conditioned on a host of mass M, does SR2 put the right
   number of subhalos inside it?  (<N_sub|M_host>, P(N_sub>=1|M_host))
3. **Position** -- are the SR2 halos *where the HR ones are*?  (host-match
   completeness vs mass, HR->SR nearest-neighbour distance, displacement of
   matched pairs, mass/Vmax bias of matched pairs, and the reverse test:
   what fraction of SR halos are spurious.)

Two stages so figures are redrawable without recomputation:

    --stage analyze   parse catalogs -> metrics.npz + summary.json   (minutes)
    --stage plot      metrics.npz -> figures/*.png                   (seconds)
    --stage both      (default)

Catalog parsing is cached per catalog as ``<out>/cache/<tag>.npz`` so repeated
analyze runs are cheap.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Frozen Stage-1 constants (configs/sr2_baseline/freeze.yaml + rockstar.cfg).
BOXSIZE = 100.0          # Mpc/h
PARTICLE_MASS = 5.81881e8  # Msun/h
MIN_HALO_OUTPUT = 20     # particles

# Shared bins: identical for HR and every SR seed so ratios are meaningful.
MASS_BINS = np.logspace(9.5, 15.0, 34)
SUBMASS_BINS = np.logspace(9.5, 14.5, 31)
VMAX_BINS = np.logspace(1.3, 3.2, 26)
# Must reach down to the 20-particle floor (1.16e10): most HR hosts live
# between 1e10 and 1e11, and a range starting at 1e11 hides them entirely.
HOSTM_BINS = np.logspace(10.0, 15.0, 21)   # occupancy / completeness stratification
# "Recovered in place" tolerance for the strict, matcher-free positional test.
PLACE_TOL_MPC = 0.2
DX_BINS = np.logspace(-2.3, 1.3, 55)       # Mpc/h, for NN-distance CDFs


# --------------------------------------------------------------------------
# catalog loading
# --------------------------------------------------------------------------

def _find_ascii(d: Path) -> Path:
    for pat in ("halos*.ascii", "halos*.list"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no Rockstar ascii catalog in {d}")


def _header_cols(path: Path) -> Dict[str, int]:
    colmap: Dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            toks = re.split(r"\s+", line.lstrip("#").strip().lower())
            if toks and toks[0] == "id" and "id" not in colmap:
                for i, t in enumerate(toks):
                    colmap[t.strip("()")] = i
                break
    return colmap


def load_catalog(path: Path, cache: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Parse a Rockstar ASCII catalog into flat arrays (pandas fast path).

    ``parent`` follows :func:`cosmo_sr.eval.rockstar.load_rockstar_ascii`: the
    internal ``i_so`` sub-of index remapped to the printed halo id, ``-1`` for
    hosts.
    """
    if cache is not None and cache.is_file():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}

    import pandas as pd

    colmap = _header_cols(path)
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                     engine="c", dtype=np.float64)
    data = df.to_numpy()
    ncols = data.shape[1]

    def col(name: str, default: int) -> np.ndarray:
        i = colmap.get(name, default if default >= 0 else ncols + default)
        return data[:, i]

    ids = col("id", 0).astype(np.int64)
    idx = col("idx", -5).astype(np.int64)
    i_so = col("i_so", -4).astype(np.int64)
    # Vectorised idx -> id remap.
    order = np.argsort(idx)
    parent = np.full(ids.shape, -1, dtype=np.int64)
    valid = i_so >= 0
    if valid.any():
        loc = np.searchsorted(idx[order], i_so[valid])
        loc = np.clip(loc, 0, order.size - 1)
        cand = order[loc]
        ok = idx[cand] == i_so[valid]
        vi = np.flatnonzero(valid)
        parent[vi[ok]] = ids[cand[ok]]

    out = {
        "id": ids,
        "parent": parent,
        "num_p": col("num_p", 1).astype(np.int64),
        "mvir": col("mvir", 2),
        "rvir": col("rvir", 4),          # kpc/h
        "vmax": col("vmax", 5),
        "pos": np.stack([col("x", 8), col("y", 9), col("z", 10)], axis=1),
        "vel": np.stack([col("vx", 11), col("vy", 12), col("vz", 13)], axis=1),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


def _sel(cat: Dict[str, np.ndarray], m: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[m] for k, v in cat.items()}


def hosts_of(cat):
    return _sel(cat, cat["parent"] < 0)


def subs_of(cat):
    return _sel(cat, cat["parent"] >= 0)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def mass_function(mass: np.ndarray, bins: np.ndarray):
    """dn/dlnM in (h/Mpc)^3, plus raw counts (for Poisson errors)."""
    m = mass[np.isfinite(mass) & (mass > 0)]
    cnt, edges = np.histogram(m, bins=bins)
    dlnm = np.log(edges[1:]) - np.log(edges[:-1])
    return cnt.astype(np.float64), cnt / (BOXSIZE ** 3 * dlnm)


def occupancy(cat: Dict[str, np.ndarray], bins: np.ndarray):
    """<N_sub | M_host> and P(N_sub>=1 | M_host) in host-mass bins."""
    h = hosts_of(cat)
    s = subs_of(cat)
    # subs per host id
    order = np.argsort(h["id"])
    hid_sorted = h["id"][order]
    loc = np.searchsorted(hid_sorted, s["parent"])
    loc = np.clip(loc, 0, hid_sorted.size - 1)
    ok = hid_sorted[loc] == s["parent"]
    nsub = np.zeros(h["id"].size, dtype=np.int64)
    np.add.at(nsub, order[loc[ok]], 1)

    ib = np.digitize(h["mvir"], bins) - 1
    n_b = bins.size - 1
    nhost = np.zeros(n_b)
    mean = np.full(n_b, np.nan)
    frac1 = np.full(n_b, np.nan)
    for b in range(n_b):
        m = ib == b
        nhost[b] = m.sum()
        if nhost[b] > 0:
            mean[b] = nsub[m].mean()
            frac1[b] = (nsub[m] >= 1).mean()
    return nhost, mean, frac1


def nn_distance(src_pos: np.ndarray, dst_pos: np.ndarray) -> np.ndarray:
    """Periodic nearest-neighbour distance from each src point to dst set."""
    from scipy.spatial import cKDTree
    if dst_pos.shape[0] == 0 or src_pos.shape[0] == 0:
        return np.full(src_pos.shape[0], np.inf)
    tree = cKDTree(np.mod(dst_pos, BOXSIZE), boxsize=BOXSIZE)
    d, _ = tree.query(np.mod(src_pos, BOXSIZE), k=1)
    return d


def nn_cdf_by_mass(src: Dict[str, np.ndarray], dst_pos: np.ndarray,
                   mass_bins: np.ndarray):
    """NN-distance histogram of src->dst, stratified by src mass.

    Answers "is there *any* SR halo where this HR halo is", independent of any
    matching policy.
    """
    d = nn_distance(src["pos"], dst_pos)
    ib = np.digitize(src["mvir"], mass_bins) - 1
    n_b = mass_bins.size - 1
    hist = np.zeros((n_b, DX_BINS.size - 1))
    n_src = np.zeros(n_b)
    med = np.full(n_b, np.nan)
    for b in range(n_b):
        m = ib == b
        n_src[b] = m.sum()
        if n_src[b] > 0:
            hist[b] = np.histogram(d[m], bins=DX_BINS)[0]
            med[b] = np.median(d[m])
    return d, hist, n_src, med


# --------------------------------------------------------------------------
# analyze stage
# --------------------------------------------------------------------------

def analyze(halos_root: Path, box: str, seeds: List[int], match_seeds: List[int],
            out: Path, slab_thickness: float) -> None:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cosmo_sr.eval.rockstar import HaloCatalog
    from cosmo_sr.eval.halo_match import match_hosts

    cache = out / "cache"
    res: Dict[str, np.ndarray] = {}
    summary: Dict[str, object] = {
        "box": box, "seeds": seeds, "match_seeds": match_seeds,
        "boxsize_mpc_h": BOXSIZE, "particle_mass": PARTICLE_MASS,
        "min_halo_output_particles": MIN_HALO_OUTPUT,
        "mass_resolution_msun_h": MIN_HALO_OUTPUT * PARTICLE_MASS,
    }

    hr = load_catalog(_find_ascii(halos_root / box / "hr" / "hr_rockstar"),
                      cache / f"{box}_hr.npz")
    print(f"[hr] {hr['id'].size} halos", flush=True)

    def to_hc(c):
        return HaloCatalog(ids=c["id"], parent_ids=c["parent"], mvir=c["mvir"],
                           rvir=c["rvir"], vmax=c["vmax"], pos=c["pos"],
                           vel=c["vel"], num_p=c["num_p"])

    hr_h, hr_s = hosts_of(hr), subs_of(hr)

    res["mass_bins"] = MASS_BINS
    res["submass_bins"] = SUBMASS_BINS
    res["vmax_bins"] = VMAX_BINS
    res["hostm_bins"] = HOSTM_BINS
    res["dx_bins"] = DX_BINS

    res["hr_hmf_cnt"], res["hr_hmf"] = mass_function(hr_h["mvir"], MASS_BINS)
    res["hr_shmf_cnt"], res["hr_shmf"] = mass_function(hr_s["mvir"], SUBMASS_BINS)
    res["hr_vmaxf_host"] = mass_function(hr_h["vmax"], VMAX_BINS)[1]
    res["hr_vmaxf_sub"] = mass_function(hr_s["vmax"], VMAX_BINS)[1]
    res["hr_nhost_b"], res["hr_nsub_mean"], res["hr_frac_occ"] = \
        occupancy(hr, HOSTM_BINS)
    res["hr_counts"] = np.array([hr["id"].size, hr_h["id"].size, hr_s["id"].size],
                                dtype=np.float64)

    sr_hmf, sr_shmf, sr_vh, sr_vs = [], [], [], []
    sr_nsub_mean, sr_frac_occ, sr_nhost_b, sr_counts = [], [], [], []

    for seed in seeds:
        d = list((halos_root / box / f"sr_seed{seed}").glob("sr*_rockstar"))
        if not d:
            print(f"SKIP seed {seed}: no catalog")
            continue
        sr = load_catalog(_find_ascii(d[0]), cache / f"{box}_sr{seed}.npz")
        sh, ss = hosts_of(sr), subs_of(sr)
        print(f"[sr{seed}] {sr['id'].size} halos "
              f"({sh['id'].size} hosts, {ss['id'].size} subs)", flush=True)
        sr_hmf.append(mass_function(sh["mvir"], MASS_BINS)[1])
        sr_shmf.append(mass_function(ss["mvir"], SUBMASS_BINS)[1])
        sr_vh.append(mass_function(sh["vmax"], VMAX_BINS)[1])
        sr_vs.append(mass_function(ss["vmax"], VMAX_BINS)[1])
        nb, mean, f1 = occupancy(sr, HOSTM_BINS)
        sr_nhost_b.append(nb)
        sr_nsub_mean.append(mean)
        sr_frac_occ.append(f1)
        sr_counts.append([sr["id"].size, sh["id"].size, ss["id"].size])

        if seed in match_seeds:
            _positional(hr, sr, hr_h, hr_s, sh, ss, to_hc, match_hosts, res,
                        summary, seed, slab_thickness)

    for name, arr in [("sr_hmf", sr_hmf), ("sr_shmf", sr_shmf),
                      ("sr_vmaxf_host", sr_vh), ("sr_vmaxf_sub", sr_vs),
                      ("sr_nsub_mean", sr_nsub_mean),
                      ("sr_frac_occ", sr_frac_occ),
                      ("sr_nhost_b", sr_nhost_b), ("sr_counts", sr_counts)]:
        res[name] = np.asarray(arr, dtype=np.float64)
    res["seeds"] = np.asarray(seeds, dtype=np.float64)

    summary["counts"] = {
        "hr": {"all": int(hr["id"].size), "hosts": int(hr_h["id"].size),
               "subs": int(hr_s["id"].size)},
        "sr_mean": {"all": float(np.mean([c[0] for c in sr_counts])),
                    "hosts": float(np.mean([c[1] for c in sr_counts])),
                    "subs": float(np.mean([c[2] for c in sr_counts]))},
        "sr_over_hr": {
            "all": float(np.mean([c[0] for c in sr_counts]) / hr["id"].size),
            "hosts": float(np.mean([c[1] for c in sr_counts]) / hr_h["id"].size),
            "subs": float(np.mean([c[2] for c in sr_counts]) / hr_s["id"].size),
        },
    }

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "metrics.npz", **res)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out/'metrics.npz'} and {out/'summary.json'}")


def _positional(hr, sr, hr_h, hr_s, sh, ss, to_hc, match_hosts, res, summary,
                seed: int, slab: float) -> None:
    """Positional fidelity for one seed (the expensive part)."""
    print(f"[sr{seed}] positional analysis ...", flush=True)

    # --- (a) matching-free: nearest SR halo to each HR halo -----------------
    n_b = HOSTM_BINS.size - 1

    def _place_frac(src, d):
        """Fraction with an SR halo within PLACE_TOL_MPC, by src mass bin."""
        ib = np.digitize(src["mvir"], HOSTM_BINS) - 1
        f = np.full(n_b, np.nan)
        for b in range(n_b):
            m = ib == b
            if m.any():
                f[b] = (d[m] < PLACE_TOL_MPC).mean()
        return f

    d_h, hist_h, n_h, med_h = nn_cdf_by_mass(hr_h, sr["pos"], HOSTM_BINS)
    res[f"nn_hosthr_hist_s{seed}"] = hist_h
    res[f"nn_hosthr_n_s{seed}"] = n_h
    res[f"nn_hosthr_med_s{seed}"] = med_h
    res[f"place_host_s{seed}"] = _place_frac(hr_h, d_h)

    d_s, hist_s, n_s, med_s = nn_cdf_by_mass(hr_s, sr["pos"], HOSTM_BINS)
    res[f"nn_subhr_hist_s{seed}"] = hist_s
    res[f"nn_subhr_n_s{seed}"] = n_s
    res[f"nn_subhr_med_s{seed}"] = med_s
    res[f"place_sub_s{seed}"] = _place_frac(hr_s, d_s)

    # Null baseline: SR halos scattered uniformly at the *same number density*.
    # Without this, "median NN = 0.64 Mpc/h" has no reference scale -- halo
    # clustering alone puts a random catalog closer than Poisson would suggest.
    rng = np.random.default_rng(12345)
    rand_pos = rng.uniform(0.0, BOXSIZE, size=sr["pos"].shape)
    d_null = nn_distance(hr_h["pos"], rand_pos)
    ib = np.digitize(hr_h["mvir"], HOSTM_BINS) - 1
    med_null = np.full(n_b, np.nan)
    for b in range(n_b):
        m = ib == b
        if m.any():
            med_null[b] = np.median(d_null[m])
    res[f"nn_null_med_s{seed}"] = med_null
    res[f"nn_null_hist_s{seed}"] = np.histogram(d_null, bins=DX_BINS)[0]
    res[f"place_null_s{seed}"] = _place_frac(hr_h, d_null)

    # reverse: are SR halos spurious (nothing in HR nearby)?
    d_rev = nn_distance(sh["pos"], hr["pos"])
    ib = np.digitize(sh["mvir"], HOSTM_BINS) - 1
    spur = np.full(n_b, np.nan)
    n_sr_b = np.zeros(n_b)
    for b in range(n_b):
        m = ib == b
        n_sr_b[b] = m.sum()
        if n_sr_b[b] > 0:
            spur[b] = (d_rev[m] > 0.5).mean()   # >0.5 Mpc/h from any HR halo
    res[f"spurious_frac_s{seed}"] = spur
    res[f"spurious_n_s{seed}"] = n_sr_b

    # --- (b) policy matching: HR host -> SR host ----------------------------
    hm = match_hosts(to_hc(hr), to_hc(sr), boxsize_mpc_h=BOXSIZE)
    matched = hm.sr_ids >= 0
    # hm.hr_ids is ordered like hr.hosts()
    ib = np.digitize(hr_h["mvir"], HOSTM_BINS) - 1
    comp = np.full(n_b, np.nan)
    n_hr_b = np.zeros(n_b)
    for b in range(n_b):
        m = ib == b
        n_hr_b[b] = m.sum()
        if n_hr_b[b] > 0:
            comp[b] = matched[m].mean()
    res[f"host_completeness_s{seed}"] = comp
    res[f"host_completeness_n_s{seed}"] = n_hr_b

    # displacement + mass/Vmax bias of matched pairs
    sid_order = np.argsort(sh["id"])
    loc = np.searchsorted(sh["id"][sid_order], hm.sr_ids[matched])
    loc = np.clip(loc, 0, sid_order.size - 1)
    j = sid_order[loc]
    good = sh["id"][j] == hm.sr_ids[matched]
    j = j[good]
    i = np.flatnonzero(matched)[good]

    dvec = hr_h["pos"][i] - sh["pos"][j]
    dvec -= BOXSIZE * np.round(dvec / BOXSIZE)
    dx = np.linalg.norm(dvec, axis=1)
    rvir_mpc = np.maximum(hr_h["rvir"][i] / 1000.0, 1e-6)

    res[f"pair_mhr_s{seed}"] = hr_h["mvir"][i]
    res[f"pair_msr_s{seed}"] = sh["mvir"][j]
    res[f"pair_vhr_s{seed}"] = hr_h["vmax"][i]
    res[f"pair_vsr_s{seed}"] = sh["vmax"][j]
    res[f"pair_dx_s{seed}"] = dx
    res[f"pair_dx_rvir_s{seed}"] = dx / rvir_mpc

    ib = np.digitize(hr_h["mvir"][i], HOSTM_BINS) - 1
    for name, q in [("dx", dx), ("dxrv", dx / rvir_mpc),
                    ("mrat", sh["mvir"][j] / np.maximum(hr_h["mvir"][i], 1e-30)),
                    ("vrat", sh["vmax"][j] / np.maximum(hr_h["vmax"][i], 1e-30))]:
        med = np.full(n_b, np.nan)
        lo = np.full(n_b, np.nan)
        hi = np.full(n_b, np.nan)
        for b in range(n_b):
            m = ib == b
            if m.sum() >= 5:
                med[b], lo[b], hi[b] = np.percentile(q[m], [50, 16, 84])
        res[f"pair_{name}_med_s{seed}"] = med
        res[f"pair_{name}_lo_s{seed}"] = lo
        res[f"pair_{name}_hi_s{seed}"] = hi

    # --- (c) slab for the visual map ---------------------------------------
    # Store the slab UNCUT. A mass cut applied here would bake one choice into
    # the figure, and a cut at 1e11 hides the whole abundance deficit (which
    # lives below it) while the panels still look like a fair comparison.
    z0 = BOXSIZE / 2.0
    for tag, cat in [("hr", hr), ("sr", sr)]:
        m = np.abs(cat["pos"][:, 2] - z0) < slab / 2.0
        res[f"slab_{tag}_pos_s{seed}"] = cat["pos"][m][:, :2]
        res[f"slab_{tag}_m_s{seed}"] = cat["mvir"][m]
        res[f"slab_{tag}_issub_s{seed}"] = (cat["parent"][m] >= 0).astype(np.float64)
    res["slab_thickness"] = np.array([slab])

    # Whole-box count ratio as a function of mass cut: quantifies how strongly
    # any "SR2 looks fine" claim depends on where the cut is placed.
    cuts = np.logspace(9.5, 13.5, 33)
    for tag, cat in [("hr", hr), ("sr", sr)]:
        issub = cat["parent"] >= 0
        res[f"cutcurve_{tag}_all_s{seed}"] = np.array(
            [(cat["mvir"] > c).sum() for c in cuts], dtype=np.float64)
        res[f"cutcurve_{tag}_sub_s{seed}"] = np.array(
            [((cat["mvir"] > c) & issub).sum() for c in cuts], dtype=np.float64)
    res["cutcurve_cuts"] = cuts

    # --- (d) zoom on the most massive HR host ------------------------------
    k = int(np.argmax(hr_h["mvir"]))
    c = hr_h["pos"][k]
    rv = hr_h["rvir"][k] / 1000.0
    summary[f"zoom_host_seed{seed}"] = {
        "mvir": float(hr_h["mvir"][k]), "rvir_mpc_h": float(rv),
        "center": [float(x) for x in c],
    }
    for tag, cat in [("hr", hr), ("sr", sr)]:
        dv = cat["pos"] - c
        dv -= BOXSIZE * np.round(dv / BOXSIZE)
        m = np.linalg.norm(dv, axis=1) < 3.0 * rv
        res[f"zoom_{tag}_d_s{seed}"] = dv[m]
        res[f"zoom_{tag}_m_s{seed}"] = cat["mvir"][m]
    res[f"zoom_rvir_s{seed}"] = np.array([rv])

    summary[f"positional_seed{seed}"] = {
        "host_match_rate": float(matched.mean()),
        "n_hr_hosts": int(matched.size),
        "median_pair_dx_mpc_h": float(np.median(dx)),
        "median_pair_dx_rvir": float(np.median(dx / rvir_mpc)),
        "frac_pairs_within_0p2_rvir": float((dx / rvir_mpc < 0.2).mean()),
        "median_matched_mass_ratio": float(np.median(
            sh["mvir"][j] / np.maximum(hr_h["mvir"][i], 1e-30))),
        "frac_sr_hosts_spurious_gt_0p5mpc": float((d_rev > 0.5).mean()),
        "median_nn_hr_host_to_any_sr_mpc_h": float(np.median(d_h)),
        "median_nn_hr_sub_to_any_sr_mpc_h": float(np.median(d_s)),
        "median_nn_null_mpc_h": float(np.median(d_null)),
        f"frac_hr_hosts_with_sr_within_{PLACE_TOL_MPC}mpc": float(
            (d_h < PLACE_TOL_MPC).mean()),
        f"frac_hr_subs_with_sr_within_{PLACE_TOL_MPC}mpc": float(
            (d_s < PLACE_TOL_MPC).mean()),
        f"frac_null_within_{PLACE_TOL_MPC}mpc": float(
            (d_null < PLACE_TOL_MPC).mean()),
    }
    print(f"[sr{seed}] {summary[f'positional_seed{seed}']}", flush=True)


# --------------------------------------------------------------------------
# plot stage
# --------------------------------------------------------------------------

C_HR = "#1b1b1b"
C_SR = "#c2410c"
C_BAND = "#fb923c"


def _cent(b):
    return np.sqrt(b[:-1] * b[1:])


def _ratio_panel(ax, x, hr, sr_stack, xlabel):
    hr = np.asarray(hr, dtype=float)
    r = np.asarray(sr_stack, dtype=float) / np.where(hr > 0, hr, np.nan)
    ax.axhline(1.0, color=C_HR, lw=1, ls="--")
    ax.fill_between(x, np.nanmin(r, axis=0), np.nanmax(r, axis=0),
                    color=C_BAND, alpha=0.35, lw=0)
    ax.plot(x, np.nanmedian(r, axis=0), color=C_SR, lw=2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SR / HR")
    ax.grid(alpha=0.25)


def plot(out: Path, figdir: Path) -> None:
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Empty mass bins are expected (nothing above the most massive halo);
    # nanmin/nanmedian over an all-empty bin is not an error here.
    warnings.filterwarnings("ignore", message="All-NaN", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Mean of empty slice",
                            category=RuntimeWarning)

    z = np.load(out / "metrics.npz")
    summary = json.loads((out / "summary.json").read_text())
    figdir.mkdir(parents=True, exist_ok=True)
    seed = int(summary["match_seeds"][0])
    mres = MIN_HALO_OUTPUT * PARTICLE_MASS
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9,
                         "axes.titlesize": 10})

    def resline(ax, x=mres, label="20-particle limit"):
        ax.axvline(x, color="#64748b", lw=1, ls=":")
        ax.text(x * 1.08, 0.02, label, rotation=90, va="bottom", fontsize=6.5,
                color="#64748b", transform=ax.get_xaxis_transform())

    # ---- Fig 1: abundance --------------------------------------------------
    mc, sc = _cent(z["mass_bins"]), _cent(z["submass_bins"])
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.2),
                            gridspec_kw={"height_ratios": [2, 1]})
    for col, (x, hr, sr, name) in enumerate([
            (mc, z["hr_hmf"], z["sr_hmf"], "Host halos (parent < 0)"),
            (sc, z["hr_shmf"], z["sr_shmf"], "Subhalos (parent >= 0)")]):
        ax = axs[0, col]
        ax.plot(x, hr, color=C_HR, lw=2, label="HR (ground truth)")
        ax.fill_between(x, sr.min(0), sr.max(0), color=C_BAND, alpha=0.35, lw=0)
        ax.plot(x, np.median(sr, 0), color=C_SR, lw=2,
                label=f"SR2 (median of {sr.shape[0]} noise seeds)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(r"$dn/d\ln M\ [(h/{\rm Mpc})^3]$")
        ax.set_title(name)
        ax.grid(alpha=0.25); ax.legend(fontsize=7.5)
        resline(ax)
        _ratio_panel(axs[1, col], x, hr, sr, r"$M_{\rm vir}\ [M_\odot/h]$")
        resline(axs[1, col], label="")
    fig.suptitle(f"Halo and subhalo mass functions — {summary['box']}, z=0",
                 y=0.98)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_abundance_mass_functions.png",
                bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: Vmax function ---------------------------------------------
    vc = _cent(z["vmax_bins"])
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.2),
                            gridspec_kw={"height_ratios": [2, 1]})
    for col, (hr, sr, name) in enumerate([
            (z["hr_vmaxf_host"], z["sr_vmaxf_host"], "Hosts"),
            (z["hr_vmaxf_sub"], z["sr_vmaxf_sub"], "Subhalos")]):
        ax = axs[0, col]
        ax.plot(vc, hr, color=C_HR, lw=2, label="HR")
        ax.fill_between(vc, sr.min(0), sr.max(0), color=C_BAND, alpha=0.35, lw=0)
        ax.plot(vc, np.median(sr, 0), color=C_SR, lw=2, label="SR2")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(r"$dn/d\ln V_{\max}$")
        ax.set_title(f"{name}: $V_{{\\max}}$ function")
        ax.grid(alpha=0.25); ax.legend(fontsize=7.5)
        _ratio_panel(axs[1, col], vc, hr, sr, r"$V_{\max}$ [km/s]")
    fig.suptitle(f"$V_{{\\max}}$ function — {summary['box']}, z=0", y=0.98)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_vmax_function.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: occupancy --------------------------------------------------
    hc = _cent(z["hostm_bins"])
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axs[0]
    ax.plot(hc, z["hr_nsub_mean"], "o-", color=C_HR, lw=2, ms=3.5, label="HR")
    ax.fill_between(hc, np.nanmin(z["sr_nsub_mean"], 0),
                    np.nanmax(z["sr_nsub_mean"], 0), color=C_BAND, alpha=0.35, lw=0)
    ax.plot(hc, np.nanmedian(z["sr_nsub_mean"], 0), "o-", color=C_SR, lw=2,
            ms=3.5, label="SR2")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm host}\ [M_\odot/h]$")
    ax.set_ylabel(r"$\langle N_{\rm sub}\,|\,M_{\rm host}\rangle$")
    ax.set_title("Occupancy"); ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = axs[1]
    r = np.nanmedian(z["sr_nsub_mean"], 0) / np.where(
        z["hr_nsub_mean"] > 0, z["hr_nsub_mean"], np.nan)
    ax.axhline(1.0, color=C_HR, ls="--", lw=1)
    ax.plot(hc, r, "o-", color=C_SR, lw=2, ms=3.5)
    ax.set_xscale("log"); ax.set_ylim(0, 1.3)
    ax.set_xlabel(r"$M_{\rm host}\ [M_\odot/h]$")
    ax.set_ylabel(r"$\langle N_{\rm sub}\rangle_{\rm SR}/\langle N_{\rm sub}\rangle_{\rm HR}$")
    ax.set_title("Occupancy ratio vs host mass")
    ax.grid(alpha=0.25)

    ax = axs[2]
    ax.plot(hc, z["hr_frac_occ"], "o-", color=C_HR, lw=2, ms=3.5, label="HR")
    ax.plot(hc, np.nanmedian(z["sr_frac_occ"], 0), "o-", color=C_SR, lw=2,
            ms=3.5, label="SR2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$M_{\rm host}\ [M_\odot/h]$")
    ax.set_ylabel(r"$P(N_{\rm sub}\geq 1\,|\,M_{\rm host})$")
    ax.set_title("Fraction of hosts with any subhalo")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_occupancy.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 4: positional fidelity ---------------------------------------
    dxc = _cent(z["dx_bins"])
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.8))

    ax = axs[0, 0]
    ax.plot(hc, z[f"host_completeness_s{seed}"], "o-", color="#94a3b8", lw=2,
            ms=4, label="matcher (1 Mpc/h linking floor)")
    ax.plot(hc, z[f"place_host_s{seed}"], "o-", color="#0369a1", lw=2, ms=4,
            label=f"SR halo within {PLACE_TOL_MPC} Mpc/h (hosts)")
    ax.plot(hc, z[f"place_sub_s{seed}"], "o-", color=C_SR, lw=2, ms=4,
            label=f"SR halo within {PLACE_TOL_MPC} Mpc/h (subs)")
    ax.plot(hc, z[f"place_null_s{seed}"], ls="--", color="#7c3aed", lw=1.5,
            label="null: uniform-random SR")
    ax.axhline(1.0, color=C_HR, ls="--", lw=1)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$M_{\rm vir,HR}\ [M_\odot/h]$")
    ax.set_ylabel("fraction with a counterpart")
    ax.set_title("Does an SR2 halo exist at the HR location?")
    ax.grid(alpha=0.25); ax.legend(fontsize=6.8)

    ax = axs[0, 1]
    for tag, lab, color in [("hosthr", "HR hosts", "#0369a1"),
                            ("subhr", "HR subhalos", C_SR)]:
        h = z[f"nn_{tag}_hist_s{seed}"].sum(0)
        c = np.cumsum(h) / max(h.sum(), 1)
        ax.plot(dxc, c, lw=2, color=color, label=lab)
    hn = z[f"nn_null_hist_s{seed}"]
    ax.plot(dxc, np.cumsum(hn) / max(hn.sum(), 1), lw=1.5, ls="--",
            color="#7c3aed", label="null: uniform-random SR")
    ax.axvline(PLACE_TOL_MPC, color="#64748b", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"distance to nearest SR2 halo [Mpc/h]")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Is there *anything* at the HR location?")
    ax.grid(alpha=0.25); ax.legend(fontsize=7.5)

    ax = axs[1, 0]
    ax.plot(hc, z[f"nn_hosthr_med_s{seed}"], "o-", color="#0369a1", lw=2, ms=4,
            label="HR hosts")
    ax.plot(hc, z[f"nn_subhr_med_s{seed}"], "o-", color=C_SR, lw=2, ms=4,
            label="HR subhalos")
    ax.plot(hc, z[f"nn_null_med_s{seed}"], ls="--", color="#7c3aed", lw=1.5,
            label="null: uniform-random SR")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$M_{\rm vir,HR}\ [M_\odot/h]$")
    ax.set_ylabel("median NN distance [Mpc/h]")
    ax.set_title("Median NN distance vs HR mass")
    ax.grid(alpha=0.25); ax.legend(fontsize=7.5)

    ax = axs[1, 1]
    dxr = z[f"pair_dx_rvir_s{seed}"]
    dxr = dxr[np.isfinite(dxr)]
    ax.hist(np.clip(dxr, 1e-3, 100), bins=np.logspace(-3, 2, 70),
            color=C_SR, alpha=0.8)
    ax.axvline(np.median(dxr), color=C_HR, ls="--", lw=1.2,
               label=f"median = {np.median(dxr):.3f} $R_{{\\rm vir}}$")
    ax.axvline(1.0, color="#0369a1", ls=":", lw=1.2, label=r"$1\,R_{\rm vir}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\Delta x| / R_{\rm vir,HR}$ of matched host pairs")
    ax.set_ylabel("count")
    ax.set_title("Displacement of matched host pairs")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    fig.suptitle("Where SR2 halos are, relative to ground truth", y=0.99)
    fig.tight_layout()
    fig.savefig(figdir / "fig4_positional_fidelity.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 5: mass / Vmax bias + spurious --------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axs[0]
    mh = z[f"pair_mhr_s{seed}"]
    ms = z[f"pair_msr_s{seed}"]
    ok = (mh > 0) & (ms > 0)
    ax.hist2d(np.log10(mh[ok]), np.log10(ms[ok]), bins=90,
              norm=matplotlib.colors.LogNorm(), cmap="magma")
    lim = [10.0, 15.0]
    ax.plot(lim, lim, color="w", lw=1, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\log_{10} M_{\rm vir,HR}$")
    ax.set_ylabel(r"$\log_{10} M_{\rm vir,SR}$")
    ax.set_title("Matched-host mass")

    ax = axs[1]
    for key, lab, color in [("mrat", r"$M_{\rm SR}/M_{\rm HR}$", C_SR),
                            ("vrat", r"$V_{\max,\rm SR}/V_{\max,\rm HR}$", "#0369a1")]:
        m = z[f"pair_{key}_med_s{seed}"]
        lo = z[f"pair_{key}_lo_s{seed}"]
        hi = z[f"pair_{key}_hi_s{seed}"]
        ax.plot(hc, m, "o-", color=color, lw=2, ms=3.5, label=lab)
        ax.fill_between(hc, lo, hi, color=color, alpha=0.18, lw=0)
    ax.axhline(1.0, color=C_HR, ls="--", lw=1)
    ax.set_xscale("log"); ax.set_ylim(0, 2.0)
    ax.set_xlabel(r"$M_{\rm host,HR}\ [M_\odot/h]$")
    ax.set_ylabel("SR / HR (median, 16-84%)")
    ax.set_title("Bias of matched pairs")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = axs[2]
    ax.plot(hc, z[f"spurious_frac_s{seed}"], "o-", color="#7c3aed", lw=2, ms=4)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$M_{\rm vir,SR}\ [M_\odot/h]$")
    ax.set_ylabel("fraction > 0.5 Mpc/h from any HR halo")
    ax.set_title("Spurious SR2 hosts (false positives)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figdir / "fig5_mass_bias_and_spurious.png", bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 6: slab map (two mass cuts) + zoom + cut dependence ----------
    # Row 1 uses a 1e11 cut and shows *positional* fidelity of massive halos.
    # Row 2 uses every resolved halo and shows the *abundance* deficit, which
    # lives entirely below 1e11 -- a 1e11-cut panel alone reads as "perfect".
    thick = float(z["slab_thickness"][0]) if "slab_thickness" in z.files else 8.0
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6))
    for row, cut in enumerate([1e11, 0.0]):
        cut_lab = (r"$>10^{11}M_\odot/h$" if cut > 0
                   else "all resolved ($>20$ particles)")
        n_ref = None
        for k, tag in enumerate(["hr", "sr"]):
            p = z[f"slab_{tag}_pos_s{seed}"]
            m = z[f"slab_{tag}_m_s{seed}"]
            keep = m > cut
            p, m = p[keep], m[keep]
            if n_ref is None:
                n_ref = p.shape[0]
            ax = axs[row, k]
            ax.scatter(p[:, 0], p[:, 1], s=np.clip(m / 4e12, 0.25, 60),
                       c=C_HR if tag == "hr" else C_SR, alpha=0.5, lw=0)
            frac = "" if tag == "hr" else f"  ({p.shape[0]/max(n_ref,1):.2f}× HR)"
            ax.set_title(f"{'HR (truth)' if tag=='hr' else 'SR2'}, {cut_lab}\n"
                         f"{p.shape[0]} halos in slab{frac}", fontsize=9)
            ax.set_xlim(0, BOXSIZE); ax.set_ylim(0, BOXSIZE)
            ax.set_xlabel("x [Mpc/h]"); ax.set_ylabel("y [Mpc/h]")
            ax.set_aspect("equal")

    # Count ratio vs mass cut -- where a "looks fine" claim comes from.
    ax = axs[1, 2]
    cuts = z["cutcurve_cuts"]
    for key, lab, color in [("all", "all halos", "#0369a1"),
                            ("sub", "subhalos only", C_SR)]:
        a = z[f"cutcurve_hr_{key}_s{seed}"]
        b = z[f"cutcurve_sr_{key}_s{seed}"]
        ax.plot(cuts, np.where(a > 0, b / np.maximum(a, 1), np.nan), lw=2,
                color=color, label=lab)
    ax.axhline(1.0, color=C_HR, ls="--", lw=1)
    ax.axvline(1e11, color="#64748b", ls=":", lw=1.2)
    ax.text(1.15e11, 0.05, "cut used in row 1", rotation=90, fontsize=6.5,
            color="#64748b", va="bottom")
    ax.axvline(MIN_HALO_OUTPUT * PARTICLE_MASS, color="#7c3aed", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_ylim(0, 1.35)
    ax.set_xlabel(r"mass cut $[M_\odot/h]$")
    ax.set_ylabel("SR2 count / HR count (whole box)")
    ax.set_title("Any \"SR2 looks fine\" claim depends on the cut", fontsize=9)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = axs[0, 2]
    rv = float(z[f"zoom_rvir_s{seed}"][0])
    dhr, dsr = z[f"zoom_hr_d_s{seed}"], z[f"zoom_sr_d_s{seed}"]
    ax.scatter(dhr[:, 0], dhr[:, 1], s=14, facecolors="none", edgecolors=C_HR,
               lw=0.7, label=f"HR ({dhr.shape[0]})")
    ax.scatter(dsr[:, 0], dsr[:, 1], s=9, color=C_SR, alpha=0.85, lw=0,
               label=f"SR2 ({dsr.shape[0]})")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(rv * np.cos(th), rv * np.sin(th), color="#0369a1", lw=1.2, ls="--",
            label=r"$R_{\rm vir}$")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta x$ [Mpc/h]"); ax.set_ylabel(r"$\Delta y$ [Mpc/h]")
    ax.set_title("Zoom: most massive HR host (no mass cut)", fontsize=9)
    ax.legend(fontsize=7.5)
    fig.suptitle(f"Halo positions — {summary['box']}, "
                 f"{thick:g} Mpc/h slab through box centre "
                 f"(hosts and subhalos, no parent filter)", y=1.0)
    fig.tight_layout()
    fig.savefig(figdir / "fig6_position_map.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures to {figdir}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--halos-root", default="/zfsauton/scratch/yixiz/DMSR/"
                    "sr2_baseline/stage1/halos",
                    help="directory containing <box>/hr and <box>/sr_seed*")
    ap.add_argument("--box", default="set12")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7",
                    help="seeds used for abundance/occupancy (cheap)")
    ap.add_argument("--match-seeds", default="0",
                    help="seeds used for positional matching (expensive)")
    ap.add_argument("--slab", type=float, default=8.0,
                    help="slab thickness in Mpc/h for the position map")
    ap.add_argument("--out", default="/zfsauton/scratch/yixiz/DMSR/"
                    "sr2_baseline/stage1/subhalo_report")
    ap.add_argument("--figdir", default=None,
                    help="default: <out>/figures")
    ap.add_argument("--stage", choices=["analyze", "plot", "both"],
                    default="both")
    args = ap.parse_args()

    out = Path(args.out)
    figdir = Path(args.figdir) if args.figdir else out / "figures"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    match_seeds = [int(s) for s in args.match_seeds.split(",") if s.strip() != ""]

    if args.stage in ("analyze", "both"):
        analyze(Path(args.halos_root), args.box, seeds, match_seeds, out,
                args.slab)
    if args.stage in ("plot", "both"):
        plot(out, figdir)


if __name__ == "__main__":
    main()
