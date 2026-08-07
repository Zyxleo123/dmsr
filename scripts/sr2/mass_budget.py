#!/usr/bin/env python
"""Where did the mass that is missing from SR2 halos actually go?

SR2 cannot lose mass. HR and SR2 are displacement fields on the *same*
Lagrangian lattice (``cosmo_sr.eval.particles.field_to_particles`` sets
``id = arange(Ng**3)``), so both boxes contain exactly ``512**3`` equal-mass
particles and the total is conserved to machine precision. Every particle that
SR2 fails to put in a halo is therefore somewhere else, and this script asks
where, with three measurements that need no halo matching at all:

1. **Density PDF** -- CIC-deposit both boxes on the same mesh and compare the
   volume-weighted PDF of ``delta`` and, more to the point, the *mass*-weighted
   one: what fraction of the box mass sits at ``delta < -0.8`` (the standard
   void criterion), and is it larger in SR2 than in HR?

2. **Migration matrix** -- the decisive test, and the reason particle identity
   matters. For every particle ``i`` sample ``delta_HR`` at its HR position and
   ``delta_SR`` at its SR2 position and histogram the pair. This says where the
   mass that HR put in dense regions ended up in SR2, per particle, rather than
   comparing two PDFs that could differ for unrelated reasons.

3. **Collapsed-mass budget** -- from the frozen Rockstar catalogs, the fraction
   of box mass bound in host halos, HR vs SR2. The halo *count* deficit is
   concentrated at the 20-particle floor (see ``docs/sr2_subhalo_results.md``),
   where the halos hold very little mass, so the count deficit and the mass
   deficit are different sizes and only the latter has to be accounted for.

The hypothesis under test is "the missing particles are in the void". The
competing hypothesis, which ``sr2_subhalo_results.md`` §2 makes the more likely
one, is that they never left the cluster: dissolving a subhalo leaves its
particles in the host's smooth component, i.e. at *high* density but unbound.
Measurement 2 separates these -- void dumping shows up as off-diagonal mass at
high ``delta_HR`` / low ``delta_SR``, smooth-component dumping as a spread that
stays at high ``delta_SR``.

Provenance note: the SR2 particle snapshots for set12 were deleted after
Rockstar ran; this reads the cached SR2 base displacement fields
(``dmsr_reward/cache/sr2_base``), which carry the same frozen model sha,
``nsplit=8``, ``pad=3``, ``noise_mode=per_tile``. The noise realisation for a
given seed index is not guaranteed to be the same draw as the ``sr_seed*``
catalog of the same index, so §1/§2 (fields) and §3 (catalogs) are consistent
in configuration but not necessarily particle-for-particle in noise. Seed-to-
seed scatter is ~0.3% (``docs/sr2_subhalo_results.md``), and running several
seeds gives the band directly.

Two stages so figures are redrawable without recomputation:

    --stage analyze   fields + catalogs -> metrics.npz + summary.json
    --stage plot      metrics.npz -> figures/*.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "sr2"))

# Frozen Stage-1 constants (configs/sr2_baseline/freeze.yaml + rockstar.cfg).
BOXSIZE_MPC = 100.0
BOX_KPC = 100_000.0
PARTICLE_MASS = 5.81881e8        # Msun/h
DIS_NORM_KPC = 6000.0            # disnorm at z=0: D(0) = 1

# Meshes. 512^3 particles on a 128^3 mesh is 64 particles/cell (0.78 Mpc/h),
# which is what the void statistics want: coarse enough that empty cells mean
# "void" and not "Poisson". 256^3 (0.39 Mpc/h, 8/cell) is carried as a
# scale-dependence check. A 512^3 mesh would be 1 particle/cell -- pure shot
# noise -- and is deliberately absent.
MESHES = (128, 256)
PRIMARY_MESH = 128

# log10(1+delta) bins, shared by the PDFs and the migration matrix so the
# marginals of the matrix reproduce the PDFs exactly.
LD_BINS = np.linspace(-2.0, 4.0, 121)

# Void / collapse thresholds reported in summary.json.
VOID_CUTS = (-0.9, -0.8, -0.7, -0.5, 0.0)
DENSE_CUTS = (10.0, 100.0, 1000.0)


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------

def _open_disp(path: Path) -> np.ndarray:
    """Memory-map a catnorm ``(6, Ng, Ng, Ng)`` field and return the disp view."""
    a = np.load(path, mmap_mode="r")
    if a.ndim != 4 or a.shape[0] < 3 or not (a.shape[1] == a.shape[2] == a.shape[3]):
        raise ValueError(f"expected (>=3, Ng, Ng, Ng), got {a.shape} in {path}")
    return a[0:3]


def _slab_positions(disp: np.ndarray, lo: int, hi: int, ng: int) -> np.ndarray:
    """Eulerian positions in Mpc/h for Lagrangian planes ``[lo, hi)``.

    Mirrors :func:`cosmo_sr.eval.particles.field_to_particles`: the lattice is
    ``(i + 0.5) * cell`` and displacements undo ``disnorm`` (x6000 kpc/h at
    z=0). Returns ``(3, n)`` float32 in Mpc/h, wrapped into the box.
    """
    cell = BOX_KPC / ng
    d = np.asarray(disp[:, lo:hi], dtype=np.float32) * np.float32(DIS_NORM_KPC)
    q = (np.arange(ng, dtype=np.float32) + np.float32(0.5)) * np.float32(cell)
    d[0] += q[lo:hi].reshape(-1, 1, 1)
    d[1] += q.reshape(1, -1, 1)
    d[2] += q.reshape(1, 1, -1)
    d %= np.float32(BOX_KPC)
    return d.reshape(3, -1) * np.float32(1e-3)


def _cic_indices(pos: np.ndarray, ng: int) -> Tuple[np.ndarray, np.ndarray]:
    """Textbook CIC: cell ``i`` spans ``[i, i+1)`` with centre ``i + 0.5``.

    Returns ``(idx8, w8)``, each ``(8, n)``: flat mesh indices and weights.
    Deposit and interpolation share this helper so a particle sitting in a
    clump reads back the density it deposited.
    """
    g = pos * np.float32(ng / BOXSIZE_MPC) - np.float32(0.5)
    i0 = np.floor(g).astype(np.int64)
    f = (g - i0).astype(np.float32)
    i0 %= ng
    i1 = (i0 + 1) % ng

    idx = np.empty((8, pos.shape[1]), dtype=np.int64)
    w = np.empty((8, pos.shape[1]), dtype=np.float32)
    one = np.float32(1.0)
    for k in range(8):
        bx, by, bz = (k >> 2) & 1, (k >> 1) & 1, k & 1
        ix = i1[0] if bx else i0[0]
        iy = i1[1] if by else i0[1]
        iz = i1[2] if bz else i0[2]
        idx[k] = (ix * ng + iy) * ng + iz
        w[k] = ((f[0] if bx else one - f[0])
                * (f[1] if by else one - f[1])
                * (f[2] if bz else one - f[2]))
    return idx, w


def cic_deposit(disp: np.ndarray, meshes=MESHES, chunk: int = 32,
                log=print) -> Dict[int, np.ndarray]:
    """CIC-deposit all ``Ng**3`` particles onto each mesh; return overdensities."""
    ng = disp.shape[1]
    grids = {m: np.zeros(m ** 3, dtype=np.float64) for m in meshes}
    t0 = time.time()
    for lo in range(0, ng, chunk):
        hi = min(lo + chunk, ng)
        pos = _slab_positions(disp, lo, hi, ng)
        for m in meshes:
            idx, w = _cic_indices(pos, m)
            for k in range(8):
                grids[m] += np.bincount(idx[k], weights=w[k], minlength=m ** 3)
        del pos
        log(f"    deposit planes {lo}-{hi} ({time.time() - t0:.0f}s)")
    out = {}
    for m, g in grids.items():
        mean = g.mean()
        out[m] = (g / mean - 1.0).reshape(m, m, m).astype(np.float32)
    return out


def sample_at_particles(disp: np.ndarray, delta: np.ndarray, chunk: int = 32,
                        log=print) -> np.ndarray:
    """Trilinear-interpolate ``delta`` at every particle's own position.

    Returns ``(Ng**3,)`` float32 in Lagrangian-id order, so the HR and SR2
    results are aligned particle-for-particle.
    """
    ng = disp.shape[1]
    m = delta.shape[0]
    flat = delta.reshape(-1)
    out = np.empty(ng ** 3, dtype=np.float32)
    per_plane = ng * ng
    t0 = time.time()
    for lo in range(0, ng, chunk):
        hi = min(lo + chunk, ng)
        pos = _slab_positions(disp, lo, hi, ng)
        idx, w = _cic_indices(pos, m)
        acc = np.zeros(pos.shape[1], dtype=np.float32)
        for k in range(8):
            acc += flat[idx[k]] * w[k]
        out[lo * per_plane:hi * per_plane] = acc
        del pos, idx, w, acc
        log(f"    sample planes {lo}-{hi} ({time.time() - t0:.0f}s)")
    return out


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def _ld(delta) -> np.ndarray:
    """log10(1 + delta), floored so empty cells land in the first bin."""
    return np.log10(np.maximum(np.asarray(delta, dtype=np.float64) + 1.0, 1e-3))


def pdf_from_grid(delta: np.ndarray) -> Dict[str, np.ndarray]:
    """Volume- and mass-weighted PDFs of ``delta`` over mesh cells.

    Mass weighting uses the cell mass ``1 + delta`` directly, which is exact:
    the CIC grid *is* the mass distribution, so no particle sampling is needed
    for the marginals.
    """
    ld = _ld(delta).ravel()
    w = (delta.ravel().astype(np.float64) + 1.0)
    vol, _ = np.histogram(ld, bins=LD_BINS)
    mass, _ = np.histogram(ld, bins=LD_BINS, weights=w)
    return {
        "pdf_vol": vol / max(vol.sum(), 1),
        "pdf_mass": mass / max(mass.sum(), 1e-30),
    }


def threshold_fractions(delta: np.ndarray) -> Dict[str, float]:
    """Volume and mass fractions below the void cuts / above the dense cuts."""
    d = delta.ravel().astype(np.float64)
    w = d + 1.0
    tot_w = w.sum()
    out: Dict[str, float] = {}
    for c in VOID_CUTS:
        m = d < c
        out[f"fvol_lt_{c}"] = float(m.mean())
        out[f"fmass_lt_{c}"] = float(w[m].sum() / tot_w)
    for c in DENSE_CUTS:
        m = d > c
        out[f"fvol_gt_{c}"] = float(m.mean())
        out[f"fmass_gt_{c}"] = float(w[m].sum() / tot_w)
    out["sigma_delta"] = float(d.std())
    return out


def migration_matrix(d_hr: np.ndarray, d_sr: np.ndarray,
                     chunk: int = 1 << 24) -> np.ndarray:
    """Joint histogram of per-particle ``(log10(1+delta_HR), log10(1+delta_SR))``.

    Rows are HR bins, columns SR2 bins; the entry is a particle *count*, i.e.
    a mass, since all particles are equal-mass. Row-normalising gives
    ``P(delta_SR | delta_HR)`` -- where HR's dense mass went.
    """
    nb = LD_BINS.size - 1
    h = np.zeros((nb, nb), dtype=np.int64)
    n = d_hr.size
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        a = np.digitize(_ld(d_hr[lo:hi]), LD_BINS) - 1
        b = np.digitize(_ld(d_sr[lo:hi]), LD_BINS) - 1
        np.clip(a, 0, nb - 1, out=a)
        np.clip(b, 0, nb - 1, out=b)
        h += np.bincount(a * nb + b, minlength=nb * nb).reshape(nb, nb)
    return h


# --------------------------------------------------------------------------
# catalogs
# --------------------------------------------------------------------------

def collapsed_budget(halos_root: Path, box: str, seeds: List[int],
                     cache: Path, log=print) -> Dict[str, object]:
    """Fraction of box mass bound in host halos, HR vs each SR2 seed.

    Subhalos are excluded from the sum because Rockstar's ``mvir`` for a host
    already contains its substructure; adding both double-counts. The same
    quantity is also reported above mass cuts, because the *count* deficit
    lives at the 20-particle floor where the mass is negligible.
    """
    from subhalo_report import _find_ascii, load_catalog  # noqa: E402

    box_mass = PARTICLE_MASS * (512 ** 3)
    cuts = [0.0, 1e10, 1e11, 1e12, 1e13]

    def one(d: Path, tag: str) -> Dict[str, object]:
        cat = load_catalog(_find_ascii(d), cache=cache / f"{tag}.npz")
        host = cat["parent"] < 0
        mv = cat["mvir"][host]
        row: Dict[str, object] = {
            "n_halos": int(cat["mvir"].size),
            "n_hosts": int(host.sum()),
            "f_collapsed": float(mv.sum() / box_mass),
        }
        row["f_collapsed_above"] = {
            f"{c:.0e}": float(mv[mv > c].sum() / box_mass) for c in cuts
        }
        return row

    res: Dict[str, object] = {}
    res["hr"] = one(halos_root / box / "hr" / "hr_rockstar", f"{box}_hr")
    log(f"  HR   f_collapsed = {res['hr']['f_collapsed']:.4f}")
    sr = {}
    for s in seeds:
        d = halos_root / box / f"sr_seed{s}" / f"sr{s}_rockstar"
        if not d.is_dir():
            log(f"  seed {s}: no catalog at {d}, skipped")
            continue
        sr[str(s)] = one(d, f"{box}_sr{s}")
        log(f"  sr{s}  f_collapsed = {sr[str(s)]['f_collapsed']:.4f}")
    res["sr"] = sr
    if sr:
        vals = [v["f_collapsed"] for v in sr.values()]
        res["sr_mean_f_collapsed"] = float(np.mean(vals))
        res["sr_std_f_collapsed"] = float(np.std(vals))
        res["delta_f_collapsed"] = float(np.mean(vals) - res["hr"]["f_collapsed"])
    return res


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def _find_sr_cache(cache_root: Path, box: str, seed: int) -> Path:
    hits = sorted(cache_root.glob(f"{box}_seed{seed}_*.npy"))
    if not hits:
        raise FileNotFoundError(
            f"no cached SR2 base field for {box} seed {seed} in {cache_root}; "
            f"run scripts/slurm/cache_sr2_base.sbatch BOXES={box}")
    return hits[0]


def analyze(hr_field: Path, cache_root: Path, halos_root: Path, box: str,
            seeds: List[int], out: Path, chunk: int, cat_cache: Path,
            cat_seeds: List[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log = print
    store: Dict[str, np.ndarray] = {"ld_bins": LD_BINS}
    summary: Dict[str, object] = {
        "box": box, "seeds": seeds, "meshes": list(MESHES),
        "primary_mesh": PRIMARY_MESH, "hr_field": str(hr_field),
        "n_particles": 512 ** 3, "particle_mass_msun_h": PARTICLE_MASS,
        "boxsize_mpc_h": BOXSIZE_MPC,
    }

    log("=== HR field ===")
    hr_disp = _open_disp(hr_field)
    hr_delta = cic_deposit(hr_disp, log=log)
    for m, d in hr_delta.items():
        p = pdf_from_grid(d)
        store[f"hr_pdf_vol_{m}"] = p["pdf_vol"]
        store[f"hr_pdf_mass_{m}"] = p["pdf_mass"]
        summary[f"hr_mesh{m}"] = threshold_fractions(d)
    log(f"  HR sigma_delta(mesh{PRIMARY_MESH}) = "
        f"{summary[f'hr_mesh{PRIMARY_MESH}']['sigma_delta']:.3f}")

    log("=== HR per-particle density ===")
    hr_at_p = sample_at_particles(hr_disp, hr_delta[PRIMARY_MESH], chunk, log)

    sr_thresh: Dict[str, List[float]] = {}
    for s in seeds:
        path = _find_sr_cache(cache_root, box, s)
        log(f"=== SR2 seed {s}: {path.name} ===")
        with open(path.with_suffix(".json")) as fh:
            summary[f"sr_seed{s}_provenance"] = json.load(fh)
        sr_disp = _open_disp(path)
        sr_delta = cic_deposit(sr_disp, log=log)
        for m, d in sr_delta.items():
            p = pdf_from_grid(d)
            store[f"sr{s}_pdf_vol_{m}"] = p["pdf_vol"]
            store[f"sr{s}_pdf_mass_{m}"] = p["pdf_mass"]
            tf = threshold_fractions(d)
            summary[f"sr_seed{s}_mesh{m}"] = tf
            for k, v in tf.items():
                sr_thresh.setdefault(f"mesh{m}_{k}", []).append(v)

        log("  per-particle density + migration matrix")
        sr_at_p = sample_at_particles(sr_disp, sr_delta[PRIMARY_MESH], chunk, log)
        store[f"sr{s}_migration"] = migration_matrix(hr_at_p, sr_at_p)
        del sr_at_p, sr_delta, sr_disp

    summary["sr_mean"] = {k: float(np.mean(v)) for k, v in sr_thresh.items()}
    summary["sr_std"] = {k: float(np.std(v)) for k, v in sr_thresh.items()}

    log("=== collapsed-mass budget from Rockstar catalogs ===")
    summary["collapsed"] = collapsed_budget(halos_root, box, cat_seeds,
                                            cat_cache, log)

    # Headline: the direct answer to "is there more mass in the void in SR2?"
    pm = f"mesh{PRIMARY_MESH}"
    head = {}
    for c in VOID_CUTS:
        hr_v = summary[f"hr_{pm}"][f"fmass_lt_{c}"]
        sr_v = summary["sr_mean"][f"{pm}_fmass_lt_{c}"]
        head[f"fmass_delta_lt_{c}"] = {
            "hr": hr_v, "sr2": sr_v, "sr2_minus_hr": sr_v - hr_v,
            "ratio": (sr_v / hr_v) if hr_v > 0 else float("nan"),
        }
    for c in DENSE_CUTS:
        hr_v = summary[f"hr_{pm}"][f"fmass_gt_{c}"]
        sr_v = summary["sr_mean"][f"{pm}_fmass_gt_{c}"]
        head[f"fmass_delta_gt_{c}"] = {
            "hr": hr_v, "sr2": sr_v, "sr2_minus_hr": sr_v - hr_v,
            "ratio": (sr_v / hr_v) if hr_v > 0 else float("nan"),
        }
    summary["headline"] = head

    # Conditional readout of the migration matrix for seed[0].
    s0 = seeds[0]
    h = store[f"sr{s0}_migration"].astype(np.float64)
    centres = 0.5 * (LD_BINS[1:] + LD_BINS[:-1])
    dcen = 10.0 ** centres - 1.0
    rows = h.sum(axis=1)
    cond = {}
    for lo_d, hi_d, name in ((-1.0, -0.8, "hr_void"),
                             (-0.8, 1.0, "hr_mean"),
                             (100.0, 1e3, "hr_dense"),
                             (1e3, 1e9, "hr_verydense")):
        sel = (dcen >= lo_d) & (dcen < hi_d)
        if not sel.any() or rows[sel].sum() == 0:
            continue
        sub = h[sel].sum(axis=0)
        cond[name] = {
            "mass_fraction_of_box": float(rows[sel].sum() / h.sum()),
            "p_lands_in_sr_void": float(sub[dcen < -0.8].sum() / sub.sum()),
            "p_lands_below_delta0": float(sub[dcen < 0.0].sum() / sub.sum()),
            "p_stays_above_delta100": float(sub[dcen > 100.0].sum() / sub.sum()),
            "median_sr_delta": float(
                dcen[np.searchsorted(np.cumsum(sub) / sub.sum(), 0.5)]),
        }
    summary["migration"] = cond
    store["dcen"] = dcen

    np.savez_compressed(out / "metrics.npz", **store)
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"wrote {out/'metrics.npz'} and {out/'summary.json'}")


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------

def plot(out: Path, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    figdir.mkdir(parents=True, exist_ok=True)
    z = np.load(out / "metrics.npz")
    with open(out / "summary.json") as fh:
        s = json.load(fh)
    seeds = s["seeds"]
    bins = z["ld_bins"]
    c = 0.5 * (bins[1:] + bins[:-1])
    dcen = z["dcen"]

    def sr_stack(key: str) -> np.ndarray:
        return np.stack([z[f"sr{k}_{key}"] for k in seeds])

    # fig1: volume- and mass-weighted delta PDFs, primary mesh, with ratios.
    m = s["primary_mesh"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for j, kind in enumerate(("vol", "mass")):
        hr = z[f"hr_pdf_{kind}_{m}"]
        sr = sr_stack(f"pdf_{kind}_{m}")
        ax[0, j].step(c, hr, where="mid", color="k", label="HR")
        ax[0, j].step(c, sr.mean(0), where="mid", color="C3", label="SR2")
        if sr.shape[0] > 1:
            ax[0, j].fill_between(c, sr.min(0), sr.max(0), color="C3", alpha=.3,
                                  step="mid")
        ax[0, j].set_yscale("log")
        ax[0, j].set_title(f"{kind}-weighted PDF of $\\delta$ "
                           f"({m}$^3$ mesh, {100.0/m:.2f} Mpc/h)")
        ax[0, j].legend()
        with np.errstate(divide="ignore", invalid="ignore"):
            r = sr.mean(0) / hr
        ax[1, j].step(c, r, where="mid", color="C3")
        ax[1, j].axhline(1.0, color="k", lw=.8)
        ax[1, j].set_ylim(0, 2)
        ax[1, j].set_ylabel("SR2 / HR")
        ax[1, j].set_xlabel(r"$\log_{10}(1+\delta)$")
        ax[1, j].axvline(np.log10(0.2), color="C0", ls=":", label=r"$\delta=-0.8$")
        ax[0, j].axvline(np.log10(0.2), color="C0", ls=":")
        ax[1, j].legend()
    fig.suptitle(f"{s['box']}: does SR2 put more mass in the void?")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_density_pdf.png", dpi=130)
    plt.close(fig)

    # fig2: cumulative mass fraction below delta, and the SR2-HR difference.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    hr = z[f"hr_pdf_mass_{m}"]
    sr = sr_stack(f"pdf_mass_{m}").mean(0)
    ax[0].plot(c, np.cumsum(hr), color="k", label="HR")
    ax[0].plot(c, np.cumsum(sr), color="C3", label="SR2")
    ax[0].set_ylabel(r"mass fraction below $\delta$")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[1].plot(c, np.cumsum(sr) - np.cumsum(hr), color="C3")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_ylabel("SR2 - HR cumulative mass fraction")
    for a in ax:
        a.set_xlabel(r"$\log_{10}(1+\delta)$")
        a.axvline(np.log10(0.2), color="C0", ls=":")
    hl = s["headline"]["fmass_delta_lt_-0.8"]
    fig.suptitle(f"void ($\\delta<-0.8$) mass fraction: HR {hl['hr']:.4f} vs "
                 f"SR2 {hl['sr2']:.4f}  (ratio {hl['ratio']:.3f})")
    fig.tight_layout()
    fig.savefig(figdir / "fig2_cumulative_mass.png", dpi=130)
    plt.close(fig)

    # fig3: migration matrix -- where HR's mass went, per particle.
    h = z[f"sr{seeds[0]}_migration"].astype(np.float64)
    rows = h.sum(axis=1, keepdims=True)
    cond = np.divide(h, rows, out=np.zeros_like(h), where=rows > 0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].pcolormesh(c, c, np.maximum(h.T, 0.5), norm=LogNorm(), cmap="magma")
    fig.colorbar(im, ax=ax[0], label="particles")
    ax[0].set_title("joint mass distribution")
    im = ax[1].pcolormesh(c, c, np.maximum(cond.T, 1e-5),
                          norm=LogNorm(vmin=1e-4, vmax=1), cmap="viridis")
    fig.colorbar(im, ax=ax[1], label=r"$P(\delta_{SR}\,|\,\delta_{HR})$")
    ax[1].set_title("row-normalised: where HR's mass went")
    for a in ax:
        a.plot(c, c, color="w", lw=.8, ls="--")
        a.axhline(np.log10(0.2), color="C0", ls=":")
        a.set_xlabel(r"$\log_{10}(1+\delta_{\rm HR})$ at the particle")
        a.set_ylabel(r"$\log_{10}(1+\delta_{\rm SR2})$ at the same particle")
    fig.suptitle("Same particle, both boxes: void dumping = mass below the "
                 "dotted line at high $\\delta_{HR}$")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_migration.png", dpi=130)
    plt.close(fig)

    # fig4: mesh dependence + collapsed-mass budget.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.35
    labels = [f"{cut}" for cut in VOID_CUTS]
    xs = np.arange(len(VOID_CUTS))
    for i, mesh in enumerate(s["meshes"]):
        hrv = [s[f"hr_mesh{mesh}"][f"fmass_lt_{cut}"] for cut in VOID_CUTS]
        srv = [s["sr_mean"][f"mesh{mesh}_fmass_lt_{cut}"] for cut in VOID_CUTS]
        ax[0].bar(xs + (i - 0.5) * width, np.array(srv) / np.array(hrv),
                  width, label=f"{mesh}$^3$ mesh")
    ax[0].axhline(1.0, color="k", lw=.8)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(labels)
    ax[0].set_xlabel(r"$\delta$ threshold")
    ax[0].set_ylabel(r"SR2/HR mass fraction below threshold")
    ax[0].set_title("void mass: SR2 relative to HR")
    ax[0].legend()

    cb = s["collapsed"]
    cuts = list(cb["hr"]["f_collapsed_above"].keys())
    hrv = [cb["hr"]["f_collapsed_above"][k] for k in cuts]
    srv = [np.mean([v["f_collapsed_above"][k] for v in cb["sr"].values()])
           for k in cuts]
    xs = np.arange(len(cuts))
    ax[1].bar(xs - width / 2, hrv, width, color="k", label="HR")
    ax[1].bar(xs + width / 2, srv, width, color="C3", label="SR2")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f">{k}" for k in cuts], rotation=20)
    ax[1].set_ylabel("box mass fraction in host halos")
    ax[1].set_title(f"collapsed budget: HR {cb['hr']['f_collapsed']:.3f} vs "
                    f"SR2 {cb.get('sr_mean_f_collapsed', float('nan')):.3f}")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig4_budget.png", dpi=130)
    plt.close(fig)
    print(f"wrote figures to {figdir}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", default="set12")
    ap.add_argument("--seeds", default="0",
                    help="SR2 base-cache seeds; several gives a scatter band")
    ap.add_argument("--hr-root", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr")
    ap.add_argument("--sr-cache-root",
                    default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward/cache/sr2_base")
    ap.add_argument("--halos-root",
                    default="/zfsauton/scratch/yixiz/DMSR/sr2_baseline/stage1/halos")
    ap.add_argument("--out", default="/zfsauton/scratch/yixiz/DMSR/"
                    "sr2_baseline/stage1/mass_budget")
    ap.add_argument("--figdir", default=None, help="default: <out>/figures")
    ap.add_argument("--catalog-seeds", default="0,1,2,3,4,5,6,7",
                    help="Rockstar seeds for the collapsed-mass budget (cheap)")
    ap.add_argument("--catalog-cache",
                    default="/zfsauton/scratch/yixiz/DMSR/sr2_baseline/stage1/"
                            "subhalo_report/cache",
                    help="reuses the parsed catalogs the subhalo report cached")
    ap.add_argument("--chunk", type=int, default=32,
                    help="Lagrangian planes per pass (memory knob)")
    ap.add_argument("--stage", choices=["analyze", "plot", "both"], default="both")
    args = ap.parse_args()

    out = Path(args.out)
    figdir = Path(args.figdir) if args.figdir else out / "figures"
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cat_seeds = [int(x) for x in args.catalog_seeds.split(",") if x.strip()]

    if args.stage in ("analyze", "both"):
        analyze(Path(args.hr_root) / f"{args.box}.npy", Path(args.sr_cache_root),
                Path(args.halos_root), args.box, seeds, out, args.chunk,
                Path(args.catalog_cache), cat_seeds)
    if args.stage in ("plot", "both"):
        plot(out, figdir)


if __name__ == "__main__":
    main()
