#!/usr/bin/env python
"""The decisive test: is ``p(HR | SR2)`` broad at subhalo scale, or determined?

``docs/sr2_substructure_module.md`` section 9 step 2 -- "not done and now the
critical path" -- and open risk 3. The entire "sample, do not regress" design
rests on the fine modes of an HR tile being **unpredictable** from its SR2 tile.
If they are predictable, an L2 regressor would do and the flow-matching module is
overbuilt. This script measures it, on set8 and set9, with no model trained.

It also closes section 9 step 3 (and ``docs/host_crop_learnability.md`` section 4
limit 3) in the same pass, because it is the same FFT: the **measured** spectrum
of ``Psi_HR - Psi_SR2`` in real ``h/Mpc``, which is what the high-pass cut has to
be re-chosen on.

The three measurements
----------------------
1. **Spectra**, per 64^3 tile (the Option B unit), stratified by whether the tile
   holds a cluster: ``P_HR(k)``, ``P_SR2(k)``, ``P_diff(k)`` and ``r(k)``. Exact.
   ``1 - r(k)^2`` is the variance an isotropic linear predictor cannot reach.
   The band split at ``--k-split`` (default 8 h/Mpc, the design's cut) reports
   how much of the residual the module must emit would be **filtered away** by
   skeleton item 4.

2. **The local conditional mean.** Ridge from SR2's displacement in a
   ``(2R+1)^3`` neighbourhood of a site to HR's *high-pass* displacement at that
   site, both divided by the local scale ``s`` of section 4.2. Fitted on one box,
   scored on another, so nothing here is an in-sample number. This is precisely
   the functional section 6.1 says an L2 regressor converges to -- ``E[HR|SR2]``
   -- and its held-out ``R^2`` is how full that functional is.

3. **The controls, without which neither number means anything.**
   * ``identity``: score of the trivial predictor "HR's fine field = SR2's fine
     field". The bar any regressor has to clear.
   * ``lowpass``: the same pipeline with HR's *smoothed* field as the target.
     SR2 is known to be right at those scales, so this must come back high. A
     pipeline that fails it is measuring its own blind spot, not the physics.
   * ``nonlinear``: the linear features plus random Fourier features of their
     leading principal components. Separates "a linear map cannot" from
     "nothing can".

What can be concluded
---------------------
Residual variance of *any* predictor is ``>= Var(HR | SR2)``. So a **high**
``R^2`` on the high-pass target would kill the "must sample" premise outright,
while a **low** one is evidence for it and not a proof -- it says this class
cannot find the conditional mean. The summary states the verdict in exactly
those terms; do not upgrade it when writing it up.

    python scripts/features/measure_conditional_spread.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from typing import Optional
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (  # noqa: E402
    DEFAULT_CONFIG, banner, hr_path, load_reward_config, paths, write_json,
)

from cosmo_sr.data.preprocess_srs import disnorm  # noqa: E402
from cosmo_sr.eval.particle_identity import root_lookup  # noqa: E402
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.features import (  # noqa: E402
    RidgeSolver, apply_random_features, band_power_fraction, gaussian_lowpass,
    hann_window, local_scale, neighbourhood_matrix, r2_uncentred,
    radial_cross_spectra, random_feature_map, ridge_predict, wavenumbers,
)

NG_HR = 512
BOXSIZE = 100.0                 # Mpc/h
DX = BOXSIZE / NG_HR            # 0.1953 Mpc/h
TILE = 64


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _sr2_field(root: Path, box: str, seed: int = 0) -> Path:
    hits = sorted((root / "cache" / "sr2_base").glob(f"{box}_seed{seed}_*.npy"))
    if not hits:
        raise FileNotFoundError(
            f"no cached SR2 field for {box} seed {seed}; "
            f"run scripts/slurm/cache_sr2_base.sbatch BOXES={box}")
    return hits[0]


def _catalog(root: Path, box: str, tag: str):
    hits = sorted(glob.glob(str(root / "halos" / f"{box}__{tag}__{tag}"
                                 / "*_rockstar" / "halos_*.ascii")))
    if not hits:
        raise FileNotFoundError(f"no {tag} catalog under {root/'halos'} for {box}")
    return load_rockstar_ascii(hits[0])


def _owner(root: Path, box: str, tag: str):
    """The per-site owner array, or ``None`` if this box does not have one.

    Optional rather than required, because it is the most expensive artifact in
    the pipeline: it is streamed from Rockstar's raw particle dumps, and those
    are deleted after extraction (``particles_deleted`` in
    ``particles_report.json``), so a box without one cannot get one back without
    re-running the halo finder. Only the host *stratification* depends on it --
    the spectra and the predictor both work without it, on uniform sites and a
    single pooled stratum, and the run says so in its output rather than
    quietly reporting a different measurement under the same name.
    """
    p = (root / "halos_particles" / f"{box}__{tag}__{tag}" / f"{box}_{tag}_owner.npy")
    return np.load(p, mmap_mode="r") if p.is_file() else None


def displacement(path, *, redshift: float = 0.0) -> np.ndarray:
    """``(3, 512, 512, 512)`` displacement in **Mpc/h** from an on-disk field.

    The catnorm files store ``disnorm``-ed kpc/h; every k here is quoted in
    h/Mpc, so the units have to be undone once, at the boundary, and never
    again.
    """
    f = np.load(str(path), mmap_mode="r")
    d = np.asarray(f[0:3], dtype=np.float32)
    return (disnorm(d.astype(np.float64), z=redshift, undo=True)
            * 1e-3).astype(np.float32)


# --------------------------------------------------------------------------- #
# 1. Spectra, per tile
# --------------------------------------------------------------------------- #
def tile_host_mass(root: Path, box: str):
    """``(512,)`` log10 of the largest HR host mass per tile, or ``None``.

    Stratifies the spectra: SR2's failure is a function of host mass, so a
    spectrum averaged over 512 mostly-empty tiles would be dominated by the
    regime that has no deficit.
    """
    raw = _owner(root, box, "hr")
    if raw is None:
        return None
    cat = _catalog(root, box, "hr")
    owner = np.asarray(raw, dtype=np.int64)
    roots = root_lookup(cat)
    mvir_by_id = np.zeros(int(roots.size) + 1, dtype=np.float64)
    keep = cat.ids < roots.size
    mvir_by_id[cat.ids[keep].astype(np.int64)] = cat.mvir[keep]

    bound = owner >= 0
    site_root = np.full(owner.shape, -1, dtype=np.int64)
    site_root[bound] = roots[np.clip(owner[bound], 0, roots.size - 1)]
    mass = np.zeros(owner.shape, dtype=np.float32)
    ok = site_root >= 0
    mass[ok] = mvir_by_id[site_root[ok]]

    n = NG_HR // TILE
    cube = mass.reshape(NG_HR, NG_HR, NG_HR)
    per_tile = cube.reshape(n, TILE, n, TILE, n, TILE).max(axis=(1, 3, 5))
    return np.log10(np.maximum(per_tile.reshape(-1), 1.0))


def tile_slice(t: int):
    n = NG_HR // TILE
    ix, iy, iz = t // (n * n), (t // n) % n, t % n
    return (slice(ix * TILE, (ix + 1) * TILE),
            slice(iy * TILE, (iy + 1) * TILE),
            slice(iz * TILE, (iz + 1) * TILE))


def spectra_for_box(hr: np.ndarray, sr2: np.ndarray, logm: np.ndarray,
                    args) -> dict:
    kmag = wavenumbers(TILE, DX)
    # A tile is a sub-cube of a periodic box, not a periodic cube. Without this
    # window its FFT leaks the tile's bulk flow into every k bin, and because HR
    # and SR2 share that flow the leak lands in both and fakes r(k) ~ 0.83 at
    # subhalo scale where the true value is ~0. Measured, not hypothetical --
    # see hann_window's docstring.
    win = hann_window(TILE)
    n_tiles = (NG_HR // TILE) ** 3
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_tiles)[:min(args.n_tiles, n_tiles)]

    if logm is None:
        groups = {"unstratified": np.ones(n_tiles, dtype=bool)}
    else:
        groups = {"cluster": logm >= args.cluster_log_mvir,
                  "group": (logm >= args.group_log_mvir)
                  & (logm < args.cluster_log_mvir),
                  "field": logm < args.group_log_mvir}
    acc = {g: None for g in groups}
    n_in = {g: 0 for g in groups}
    for t in order:
        sl = tile_slice(int(t))
        a = hr[(slice(None),) + sl]
        b = sr2[(slice(None),) + sl]
        s = radial_cross_spectra(a, b, DX, n_bins=args.n_bins, kmag=kmag,
                                 window=win)
        for g, m in groups.items():
            if not m[int(t)]:
                continue
            n_in[g] += 1
            if acc[g] is None:
                acc[g] = {k: np.array(v, dtype=np.float64) for k, v in s.items()}
            else:
                for k in ("P_a", "P_b", "P_diff", "P_cross"):
                    acc[g][k] += s[k]
        # `counts`, `k` and `k_edges` are geometry and identical for every tile.

    out = {}
    for g, a in acc.items():
        if a is None or n_in[g] == 0:
            continue
        for k in ("P_a", "P_b", "P_diff", "P_cross"):
            a[k] /= n_in[g]
        r = a["P_cross"] / np.sqrt(np.maximum(a["P_a"] * a["P_b"], 1e-300))
        lo_d, hi_d = band_power_fraction(a["k"], a["P_diff"], a["counts"],
                                         args.k_split)
        lo_h, hi_h = band_power_fraction(a["k"], a["P_a"], a["counts"], args.k_split)
        good = np.isfinite(r) & (a["counts"] > 0)
        out[g] = {
            "n_tiles": int(n_in[g]),
            "k": a["k"].tolist(),
            "counts": a["counts"].tolist(),
            "P_hr": a["P_a"].tolist(),
            "P_sr2": a["P_b"].tolist(),
            "P_diff": a["P_diff"].tolist(),
            "r": r.tolist(),
            "one_minus_r2": (1.0 - r ** 2).tolist(),
            "k_split": float(args.k_split),
            "resid_power_below_split": float(lo_d),
            "resid_power_above_split": float(hi_d),
            "hr_power_below_split": float(lo_h),
            "hr_power_above_split": float(hi_h),
            # Where SR2 stops tracking HR at all -- the honest place to look for
            # a cut, against the design's 8 h/Mpc guess.
            "k_at_r_0p9": _k_cross(a["k"][good], r[good], 0.9),
            "k_at_r_0p5": _k_cross(a["k"][good], r[good], 0.5),
            "k_at_r_0p1": _k_cross(a["k"][good], r[good], 0.1),
        }
    return out


def full_box_spectrum(hr: np.ndarray, sr2: np.ndarray, args) -> dict:
    """The exact, leakage-free spectrum: the whole box, which really is periodic.

    This is what the high-pass decision (section 9 step 3) must be read off.
    The per-tile spectra stay for stratification -- they are the only way to
    separate clusters from voids -- but they are windowed and therefore
    approximate, whereas this one has no window and no assumption to violate.
    Done one component at a time so two 512^3 complex transforms are the peak
    memory rather than six.
    """
    n = int(hr.shape[-1])
    kmag = wavenumbers(n, DX)
    k_f, k_ny = 2.0 * np.pi / (n * DX), np.pi / DX
    edges = np.logspace(np.log10(k_f * 0.999), np.log10(k_ny * 1.001),
                        int(args.n_bins) + 1)
    which = np.digitize(kmag.reshape(-1), edges) - 1
    flat = kmag.reshape(-1)
    ok = (which >= 0) & (which < int(args.n_bins)) & (flat > 0)
    w = which[ok]
    cnt = np.bincount(w, minlength=int(args.n_bins)).astype(np.float64)
    acc = {n_: np.zeros(int(args.n_bins)) for n_ in ("P_a", "P_b", "P_diff", "P_cross")}
    norm = DX ** 3 / float(n ** 3)
    for c in range(3):
        fa = np.fft.fftn(hr[c])
        fb = np.fft.fftn(sr2[c])
        for name, arr in (("P_a", np.abs(fa) ** 2), ("P_b", np.abs(fb) ** 2),
                          ("P_diff", np.abs(fa - fb) ** 2),
                          ("P_cross", np.real(fa * np.conj(fb)))):
            acc[name] += np.bincount(w, weights=(arr.reshape(-1)[ok] * norm),
                                     minlength=int(args.n_bins))
        del fa, fb
    for name in acc:
        acc[name] = np.where(cnt > 0, acc[name] / np.maximum(cnt, 1), np.nan)
    kbar = np.where(cnt > 0, np.bincount(w, weights=flat[ok],
                                         minlength=int(args.n_bins))
                    / np.maximum(cnt, 1), np.nan)
    r = acc["P_cross"] / np.sqrt(np.maximum(acc["P_a"] * acc["P_b"], 1e-300))
    lo_d, hi_d = band_power_fraction(kbar, acc["P_diff"], cnt, args.k_split)
    good = np.isfinite(r) & (cnt > 0)
    return {
        "n_tiles": 1, "windowed": False, "exact": True,
        "k": kbar.tolist(), "counts": cnt.tolist(),
        "P_hr": acc["P_a"].tolist(), "P_sr2": acc["P_b"].tolist(),
        "P_diff": acc["P_diff"].tolist(), "r": r.tolist(),
        "one_minus_r2": (1.0 - r ** 2).tolist(),
        "k_split": float(args.k_split),
        "resid_power_below_split": float(lo_d),
        "resid_power_above_split": float(hi_d),
        "k_at_r_0p9": _k_cross(kbar[good], r[good], 0.9),
        "k_at_r_0p5": _k_cross(kbar[good], r[good], 0.5),
        "k_at_r_0p1": _k_cross(kbar[good], r[good], 0.1),
    }


def _k_cross(k: np.ndarray, r: np.ndarray, level: float) -> float:
    """The smallest ``k`` at which ``r`` has fallen below ``level``, interpolated."""
    k = np.asarray(k, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    below = np.flatnonzero(r < float(level))
    if below.size == 0:
        return float("inf")
    j = int(below[0])
    if j == 0:
        return float(k[0])
    r0, r1, k0, k1 = r[j - 1], r[j], k[j - 1], k[j]
    if r0 == r1:
        return float(k1)
    return float(k0 + (k1 - k0) * (r0 - level) / (r0 - r1))


# --------------------------------------------------------------------------- #
# 2/3. The local conditional mean
# --------------------------------------------------------------------------- #
def select_sites(root: Path, box: str, args, rng) -> tuple:
    """``(sites, is_host)`` -- half inside massive-host footprints, half uniform.

    Reported separately rather than pooled. The deficit is a host-mass effect,
    so a score averaged over a box that is mostly void would answer a different
    question from the one section 9 asks.
    """
    raw = _owner(root, box, "hr")
    if raw is None:
        return None
    cat = _catalog(root, box, "hr")
    owner = np.asarray(raw, dtype=np.int64)
    roots = root_lookup(cat)
    mvir_by_id = np.zeros(int(roots.size) + 1, dtype=np.float64)
    keep = cat.ids < roots.size
    mvir_by_id[cat.ids[keep].astype(np.int64)] = cat.mvir[keep]

    bound = owner >= 0
    sr = np.full(owner.shape, -1, dtype=np.int64)
    sr[bound] = roots[np.clip(owner[bound], 0, roots.size - 1)]
    ok = sr >= 0
    host_mass = np.zeros(owner.shape, dtype=np.float32)
    host_mass[ok] = mvir_by_id[sr[ok]]
    pool = np.flatnonzero(host_mass >= 10.0 ** args.site_log_mvir)
    del owner, sr, host_mass

    n_half = int(args.n_sites) // 2
    if pool.size == 0:
        raise SystemExit(f"{box}: no sites bound to hosts above 1e{args.site_log_mvir}")
    hid = rng.choice(pool, size=min(n_half, pool.size), replace=False)
    uid = rng.integers(0, NG_HR ** 3, size=n_half)
    flat = np.concatenate([hid, uid])
    is_host = np.concatenate([np.ones(hid.size, bool), np.zeros(uid.size, bool)])
    # Shuffle before returning. The rows are built host-block-then-uniform-block
    # and `Fitter` splits off its validation quarter by position, so leaving the
    # order alone would validate the pooled fit on uniform sites only -- a
    # different population from the one it trained on, and alpha would be chosen
    # for the wrong regime.
    perm = rng.permutation(flat.size)
    flat, is_host = flat[perm], is_host[perm]
    sites = np.stack([flat // (NG_HR * NG_HR), (flat // NG_HR) % NG_HR,
                      flat % NG_HR], axis=1)
    return sites.astype(np.int64), is_host


def build_examples(hr: np.ndarray, sr2: np.ndarray, sites: np.ndarray, args):
    """Features from SR2, targets from HR, both in units of the local scale ``s``.

    One feature matrix serves every band: the features are SR2's *raw*
    neighbourhood, so any band a linear map wants is inside its span. Only the
    targets are re-filtered per sigma, which is why sigma is cheap to sweep.
    """
    s = local_scale(sr2, sigma_sites=args.scale_sigma)
    s_at = s[sites[:, 0], sites[:, 1], sites[:, 2]][:, None].astype(np.float32)

    x = neighbourhood_matrix(sr2, sites, args.radius)
    x /= s_at

    targets = {}
    for sig in args.sigmas:
        hr_lo = gaussian_lowpass(hr, sig)
        sr_lo = gaussian_lowpass(sr2, sig)
        idx = (slice(None), sites[:, 0], sites[:, 1], sites[:, 2])
        y_hi = ((hr - hr_lo)[idx].T / s_at).astype(np.float32)
        y_lo = (hr_lo[idx].T / s_at).astype(np.float32)
        ident = ((sr2 - sr_lo)[idx].T / s_at).astype(np.float32)
        targets[f"sigma{sig:g}"] = {"high": y_hi, "low": y_lo, "identity": ident}
        del hr_lo, sr_lo
    return x, targets, s_at.reshape(-1)



def _pca(x: np.ndarray, dim: int) -> dict:
    """Whitening projection onto the leading ``dim`` principal components."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    z = (x - mu) / sd
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    w, v = np.linalg.eigh(cov.astype(np.float64))
    take = np.argsort(w)[::-1][:int(dim)]
    return {"mean": mu, "scale": sd, "vec": v[:, take].astype(np.float32),
            "sqrt_eig": np.sqrt(np.maximum(w[take], 1e-12)).astype(np.float32),
            "explained": float(np.maximum(w[take], 0).sum() / max(w.sum(), 1e-30))}


def _pca_apply(p: dict, x: np.ndarray) -> np.ndarray:
    return (((x - p["mean"]) / p["scale"]) @ p["vec"]) / p["sqrt_eig"]


class Fitter:
    """One feature matrix, split once, with the Gram computed once.

    ``alpha`` is chosen on a held-out quarter of the **fit box** and the score
    that gets reported is on the **other box** entirely, so no number here has
    seen its own test set. Only the training portion is ever fitted, which is
    why a single Gram suffices.
    """

    def __init__(self, x_fit: np.ndarray, x_test: np.ndarray, val_frac: float,
                 s_test: np.ndarray = None):
        self.w_test = None if s_test is None else np.asarray(s_test) ** 2
        n = int(x_fit.shape[0])
        self.cut = int(n * (1.0 - float(val_frac)))
        self.x_val = x_fit[self.cut:]
        self.x_test = x_test
        self.solver = RidgeSolver(x_fit[:self.cut])
        self.n_features = int(x_fit.shape[1])

    def score(self, y_fit: np.ndarray, y_test: np.ndarray, alphas) -> dict:
        y_tr, y_val = y_fit[:self.cut], y_fit[self.cut:]
        rows, best, best_a, best_fit = [], -np.inf, None, None
        for a in alphas:
            fit = self.solver.fit(y_tr, a)
            v = r2_uncentred(y_val, ridge_predict(fit, self.x_val))
            rows.append({"alpha": float(a), "r2_val_in_fit_box": float(v)})
            if np.isfinite(v) and v > best:
                best, best_a, best_fit = v, float(a), fit
        if best_fit is None:
            raise SystemExit("every alpha gave a non-finite validation score")
        return {
            "alpha": best_a,
            "r2_val_in_fit_box": float(best),
            "r2_heldout_box": float(r2_uncentred(y_test,
                                                 ridge_predict(best_fit, self.x_test))),
            # Same predictor, scored with the 1/s normalisation undone, so it is
            # directly comparable to P_diff/P_HR from the spectrum of the same
            # box. Disagreement between the two means the scale-normalised
            # number is describing a subpopulation, not the field.
            "r2_heldout_unweighted": float(r2_uncentred(
                y_test, ridge_predict(best_fit, self.x_test), self.w_test)),
            "n_fit": int(self.cut),
            "n_test": int(self.x_test.shape[0]),
            "n_features": self.n_features,
            "alpha_sweep": rows,
        }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    cfg = load_reward_config(args)
    root = paths.reward_root()
    boxes = [b for b in args.boxes.split(",") if b]
    if len(boxes) not in (1, 2):
        raise SystemExit("--boxes takes one box (split spatially) or two "
                         "(fit box, held-out box)")
    fit_box, test_box = (boxes[0], boxes[0]) if len(boxes) == 1 else boxes
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    spectra, examples, site_flags, site_x = {}, {}, {}, {}
    site_scale = {}
    for box in boxes:
        banner(f"conditional spread: {box}")
        hr = displacement(hr_path(cfg, box))
        sr2 = displacement(_sr2_field(root, box, args.seed_sr2))
        print(f"  fields loaded ({hr.nbytes / 2**30:.2f} GiB each)", flush=True)

        logm = tile_host_mass(root, box)
        spectra[box] = spectra_for_box(hr, sr2, logm, args)
        spectra[box]["whole_box"] = full_box_spectrum(hr, sr2, args)
        for g, d in spectra[box].items():
            print(f"  {g:13s} n={d['n_tiles']:3d}  resid power above "
                  f"{args.k_split:g} h/Mpc = {d['resid_power_above_split']:.3f}"
                  f"   r=0.5 at k={d['k_at_r_0p5']:.2f}", flush=True)

        sites, is_host = select_sites(root, box, args, rng)
        x, tgt, s_site = build_examples(hr, sr2, sites, args)
        examples[box] = (x, tgt)
        site_scale[box] = s_site
        site_flags[box] = is_host
        site_x[box] = sites[:, 0]
        print(f"  {x.shape[0]} sites ({is_host.sum()} in >=1e"
              f"{args.site_log_mvir:g} host footprints), {x.shape[1]} features "
              f"({x.nbytes / 2**30:.2f} GiB)  [{time.time() - t0:.0f}s]", flush=True)
        del hr, sr2

    if len(boxes) == 1:
        # Spatial split of one box: a slab either side of a buffer wide enough
        # that no training site's receptive field touches a test site's. Same
        # realisation, so the large-scale modes are shared -- but every target
        # here is high-pass, and the buffer is what keeps the *local* windows
        # disjoint, which is what a local predictor could otherwise exploit.
        b = boxes[0]
        x, tgt = examples[b]
        lo = int(args.split_site) - int(args.split_buffer)
        hi = int(args.split_site) + int(args.split_buffer)
        mf, mt = site_x[b] < lo, site_x[b] >= hi
        split = {"kind": "spatial", "box": b, "split_site": int(args.split_site),
                 "buffer_sites": int(args.split_buffer),
                 "buffer_mpc_h": float(args.split_buffer * DX)}
        x_fit, x_test = x[mf], x[mt]
        t_fit = {k: {n: v[mf] for n, v in d.items()} for k, d in tgt.items()}
        t_test = {k: {n: v[mt] for n, v in d.items()} for k, d in tgt.items()}
        flag_fit, flag_test = site_flags[b][mf], site_flags[b][mt]
        s_fit, s_test = site_scale[b][mf], site_scale[b][mt]
        print(f"  spatial split of {b}: {int(mf.sum())} fit / {int(mt.sum())} "
              f"test sites, {args.split_buffer} sites "
              f"({args.split_buffer * DX:.2f} Mpc/h) of buffer", flush=True)
    else:
        split = {"kind": "cross_box", "fit_box": fit_box, "heldout_box": test_box}
        x_fit, t_fit = examples[fit_box]
        x_test, t_test = examples[test_box]
        flag_fit, flag_test = site_flags[fit_box], site_flags[test_box]
        s_fit, s_test = site_scale[fit_box], site_scale[test_box]

    pca = _pca(x_fit[:min(args.pca_rows, x_fit.shape[0])], args.nl_dim)
    rf = random_feature_map(args.nl_dim, args.rff_dim, gamma=1.0 / args.nl_dim,
                            seed=args.seed)
    print(f"  PCA {args.nl_dim} components explain {pca['explained']:.3f} "
          f"of the feature variance", flush=True)
    nl_fit = np.hstack([x_fit, apply_random_features(rf, _pca_apply(pca, x_fit))])
    nl_test = np.hstack([x_test, apply_random_features(rf, _pca_apply(pca, x_test))])

    subsets = (("all", slice(None), slice(None)),
               ("host_sites", flag_fit, flag_test),
               ("uniform_sites", ~flag_fit, ~flag_test))
    pred = {key: {} for key in t_fit}
    for subset, mf, mt in subsets:
        n_f = x_fit.shape[0] if isinstance(mf, slice) else int(mf.sum())
        n_t = x_test.shape[0] if isinstance(mt, slice) else int(mt.sum())
        if min(n_f, n_t) < args.min_subset_rows:
            print(f"  [{subset}] skipped: {n_f} fit / {n_t} test rows, under "
                  f"--min-subset-rows {args.min_subset_rows}", flush=True)
            continue
        lin = Fitter(x_fit[mf], x_test[mt], args.val_frac, s_test[mt])
        nl = Fitter(nl_fit[mf], nl_test[mt], args.val_frac, s_test[mt])
        print(f"  [{subset}] {lin.solver.n} fit rows, "
              f"{lin.n_features} linear / {nl.n_features} nonlinear features "
              f"[{time.time() - t0:.0f}s]", flush=True)
        for key in t_fit:
            row = {}
            for band in ("high", "low"):
                row[band] = lin.score(t_fit[key][band][mf], t_test[key][band][mt],
                                      args.alphas)
            row["high"]["r2_identity_baseline"] = float(
                r2_uncentred(t_test[key]["high"][mt], t_test[key]["identity"][mt]))
            row["high"]["r2_identity_unweighted"] = float(
                r2_uncentred(t_test[key]["high"][mt], t_test[key]["identity"][mt],
                             np.asarray(s_test[mt]) ** 2))
            row["high_nonlinear"] = nl.score(t_fit[key]["high"][mf],
                                             t_test[key]["high"][mt], args.alphas)
            pred[key][subset] = row
            print(f"  {key} {subset:14s} high R2={row['high']['r2_heldout_box']:+.4f}"
                  f"  (+RFF {row['high_nonlinear']['r2_heldout_box']:+.4f})"
                  f"  identity {row['high']['r2_identity_baseline']:+.4f}"
                  f"  low-pass control R2={row['low']['r2_heldout_box']:+.4f}\n"
                  f"    {'':>{len(key)}} unweighted (spectrum-comparable): "
                  f"high R2={row['high']['r2_heldout_unweighted']:+.4f}  "
                  f"identity {row['high']['r2_identity_unweighted']:+.4f}",
                  flush=True)
        del lin, nl

    summary = {
        "ok": True,
        "fit_box": fit_box,
        "heldout_box": test_box,
        "boxsize_mpc_h": BOXSIZE,
        "ng_hr": NG_HR,
        "dx_mpc_h": DX,
        "tile": TILE,
        "k_split_design": float(args.k_split),
        "sigmas_sites": list(args.sigmas),
        "sigma_k_edge_h_per_mpc": {f"sigma{s:g}": float(1.0 / (s * DX))
                                   for s in args.sigmas},
        "radius_sites": int(args.radius),
        "receptive_field_mpc_h": float((2 * args.radius + 1) * DX),
        "scale_sigma_sites": float(args.scale_sigma),
        "n_sites_per_box": int(args.n_sites),
        "site_log_mvir": float(args.site_log_mvir),
        "pca_explained": pca["explained"],
        "split": split,
        "spectra": spectra,
        "predictability": pred,
        "verdict": verdict(pred, args, spectra),
        "seconds": round(time.time() - t0, 1),
    }
    out = paths.subdir("cond_spread", create=True)
    write_json(out / f"cond_spread_{fit_box}_{test_box}.json", summary)
    print(f"  wrote {out / f'cond_spread_{fit_box}_{test_box}.json'}")
    return summary


def spectrum_identity_r2(strata: dict, *, sigma_sites: float, dx: float,
                         k_edge: Optional[float] = None) -> float:
    """Uncentred ``R^2`` of the identity predictor "HR fine = SR2 fine", read
    off the exact spectrum, weighted by the SAME high-pass the site-space route
    applies -- a soft Gaussian, not a sharp ``k`` cut.

    The site target is ``x - gaussian_filter(x, sigma)`` (``cond_spread.
    gaussian_lowpass``), whose Fourier transfer function is

        H(k) = 1 - exp(-(k * dx * sigma)^2 / 2)        (k in h/Mpc, sigma sites)

    so by Parseval the site-space identity ``R^2`` over the whole box equals

        1 - sum_k H(k)^2 P_diff(k) / sum_k H(k)^2 P_HR(k),

    mode-count weighted. The old cross-check used a *hard* mask ``k >= 1/(sigma*
    dx)`` instead, which keeps only the fully-decorrelated tail and scores the
    identical predictor far lower: that operator mismatch (a soft rolloff that
    is still 0.61 at the quoted edge versus a brick wall) was the bulk of the
    0.42 site-vs-spectrum gap the gate tripped on. Matching the transfer
    function removes it and leaves only the population/sampling difference,
    which is what the gate tolerance is for.

    ``k_edge`` is retained as an optional *additional* hard floor for the old
    diagnostic; left ``None`` (the reconciled default) the Gaussian weight alone
    selects the band.
    """
    num = den = 0.0
    for d in strata.values():
        k = np.asarray(d["k"], dtype=np.float64)
        n = np.asarray(d["counts"], dtype=np.float64) * int(d["n_tiles"])
        pd_ = np.asarray(d["P_diff"], dtype=np.float64)
        ph = np.asarray(d["P_hr"], dtype=np.float64)
        h2 = (1.0 - np.exp(-0.5 * (k * float(dx) * float(sigma_sites)) ** 2)) ** 2
        m = np.isfinite(k) & (n > 0) & np.isfinite(pd_) & np.isfinite(ph)
        if k_edge is not None:
            m &= k >= float(k_edge)
        w = n[m] * h2[m]
        num += float((pd_[m] * w).sum())
        den += float((ph[m] * w).sum())
    return 1.0 - num / den if den > 0 else float("nan")


def verdict(pred: dict, args, spectra: dict = None) -> dict:
    """The one-line reading, stated with the asymmetry it actually has."""
    key = f"sigma{args.sigmas[0]:g}"
    # host_sites is the subset the deficit lives in and is preferred, but a box
    # without an owner array has none; fall back rather than crash, and name the
    # subset in the result so the reading is never ambiguous about its own scope.
    subset = "host_sites" if "host_sites" in pred[key] else "all"
    row = pred[key][subset]
    high = row["high"]["r2_heldout_box"]
    nl = row["high_nonlinear"]["r2_heldout_box"]
    low = row["low"]["r2_heldout_box"]
    control_ok = low >= args.control_min
    best = max(high, nl)

    # Cross-check: the identity predictor's score, computed two ways that must
    # agree by Parseval. Two things have to match for that agreement to hold,
    # and the pre-fix version got both wrong, which is where the 0.42 gap came
    # from: (1) the high-pass OPERATOR -- the site route is a soft Gaussian, so
    # the spectrum route must weight by its transfer function, not apply a sharp
    # k cut; done inside spectrum_identity_r2 now. (2) the POPULATION -- the
    # exact spectrum is the whole box, so it must be compared against the
    # whole-box-sampling `uniform_sites`, not the host-enriched `host_sites`
    # subset the physics reading uses. We report the host_sites R2 for the
    # verdict but gate consistency on the matched population.
    sigma0 = float(args.sigmas[0])
    k_edge = 1.0 / (sigma0 * DX)
    gate_subset = ("uniform_sites" if "uniform_sites" in pred[key]
                   else "all" if "all" in pred[key] else subset)
    ident_sites = pred[key][gate_subset]["high"].get(
        "r2_identity_unweighted", float("nan"))
    strata = next(iter(spectra.values())) if spectra else {}
    exact = {k_: v for k_, v in strata.items() if v.get("exact")}
    ident_spec = (spectrum_identity_r2(exact or strata,
                                       sigma_sites=sigma0, dx=DX)
                  if strata else float("nan"))
    gap = abs(ident_sites - ident_spec)
    consistent = not np.isfinite(gap) or gap <= args.identity_gap_max

    if not consistent:
        return {"band": key, "subset": subset, "consistent": False,
                "gate_subset": gate_subset,
                "r2_high_linear": high, "r2_high_nonlinear": nl,
                "r2_low_control": low, "control_passed": bool(control_ok),
                "identity_from_sites": float(ident_sites),
                "identity_from_spectrum": float(ident_spec),
                "text": (
                    f"INCONSISTENT: the identity predictor scores "
                    f"{ident_sites:+.3f} in site space ({gate_subset}) and "
                    f"{ident_spec:+.3f} from the Gaussian-weighted exact "
                    f"spectrum, a gap of {gap:.2f}. The two routes measure the "
                    "same quantity, so one of them is not describing the field. "
                    "Do not read the R2 numbers until this is closed.")}

    if not control_ok:
        text = ("INCONCLUSIVE: the low-pass control did not come back "
                f"({low:.3f} < {args.control_min}), so the pipeline cannot see "
                "determinism it should be able to see. Fix the control first.")
    elif best >= args.determined_min:
        text = (f"DETERMINED at R2={best:.3f}: the conditional mean of HR's fine "
                "modes is recoverable from a local SR2 neighbourhood. Section "
                "6.1's 'L2 converges to an empty functional' does not hold here "
                "and the flow-matching design is overbuilt -- revisit before "
                "training.")
    else:
        text = (f"BROAD (evidence, not proof) at R2={best:.3f} against a "
                f"low-pass control of {low:.3f}: no local linear or "
                "random-feature map finds a conditional mean in the fine modes, "
                "while the same pipeline recovers the smoothed field. Residual "
                "variance of any predictor bounds Var(HR|SR2) from ABOVE only, "
                "so this supports 'sample, do not regress' without proving it.")
    return {"band": key, "subset": subset, "consistent": True,
            "gate_subset": gate_subset,
            "r2_high_linear": high, "r2_high_nonlinear": nl,
            "r2_low_control": low, "control_passed": bool(control_ok),
            "identity_from_sites": float(ident_sites),
            "identity_from_spectrum": float(ident_spec),
            "band_edge_h_per_mpc": float(k_edge),
            "text": text}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--boxes", default="set8",
                    help="one box (fit and test on disjoint slabs of it) or two "
                         "(fit on the first, score on the second). Two boxes is "
                         "the cleaner held-out set; one box is what runs when "
                         "the second box has no owner array")
    ap.add_argument("--split-site", type=int, default=256,
                    help="one-box mode: the x site the slabs straddle")
    ap.add_argument("--split-buffer", type=int, default=32,
                    help="one-box mode: half-gap in sites (32 -> 6.25 Mpc/h), "
                         "which must exceed --radius by a wide margin")
    ap.add_argument("--min-subset-rows", type=int, default=5000,
                    help="a subset with fewer rows than this is skipped rather "
                         "than reported at a sample size that cannot carry it")
    ap.add_argument("--n-tiles", type=int, default=512, help="tiles per box for spectra")
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--k-split", type=float, default=8.0,
                    help="the design's high-pass cut, in h/Mpc")
    ap.add_argument("--cluster-log-mvir", type=float, default=14.0)
    ap.add_argument("--group-log-mvir", type=float, default=13.0)
    ap.add_argument("--sigmas", default="0.7,1,2",
                    help="high-pass smoothing scales in native sites. The pass "
                         "edge is k ~ 1/(sigma*dx): 0.7, 1 and 2 sites are 7.3, "
                         "5.1 and 2.6 h/Mpc, which brackets the 2-7 h/Mpc band "
                         "every resolvable subhalo occupies. Every sigma must "
                         "stay well inside --radius or the low-pass control "
                         "fails on receptive field rather than on physics")
    ap.add_argument("--radius", type=int, default=5,
                    help="neighbourhood half-width in sites; 5 -> 11^3, "
                         "2.15 Mpc/h, and >= 2.5x the largest sigma")
    ap.add_argument("--scale-sigma", type=float, default=3.0)
    ap.add_argument("--n-sites", type=int, default=480000,
                    help="total fitted sites, half host-footprint half uniform. "
                         "The uniform half is the gate's whole-box sample, so "
                         "this is bumped from 120k to shrink the sampling noise "
                         "on the site-vs-spectrum identity comparison")
    ap.add_argument("--site-log-mvir", type=float, default=13.0)
    ap.add_argument("--alphas", default="1e-6,1e-4,1e-2,1e0")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--nl-dim", type=int, default=64, help="PCA dim before the RFF")
    ap.add_argument("--rff-dim", type=int, default=1024)
    ap.add_argument("--pca-rows", type=int, default=60000)
    ap.add_argument("--determined-min", type=float, default=0.30,
                    help="held-out R2 above which the conditional is called narrow")
    ap.add_argument("--identity-gap-max", type=float, default=0.25,
                    help="largest tolerable disagreement between the identity "
                         "predictor scored in site space and off the spectrum; "
                         "beyond it the run reports INCONSISTENT and no R2")
    ap.add_argument("--control-min", type=float, default=0.80,
                    help="low-pass R2 below which the run is inconclusive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-sr2", type=int, default=0)
    args = ap.parse_args(argv)
    args.sigmas = [float(s) for s in str(args.sigmas).split(",") if s]
    args.alphas = [float(a) for a in str(args.alphas).split(",") if a]

    s = run(args)
    print()
    print("VERDICT:", s["verdict"]["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
