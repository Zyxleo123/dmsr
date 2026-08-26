#!/usr/bin/env python
"""Materialise Option A's host-frame crops and measure how learnable they are.

``docs/sr2_substructure_module.md`` section 3 defines the crop; section 7 risk 3
asks the question this script answers -- *is where HR puts its subhalos already
visible in the SR2 crop?* Both are settled on the same artifacts:

* ``paired_catnorm/hr/<box>.npy`` and ``cache/sr2_base/<box>_seed0_*.npy``,
  the two ``(6, 512, 512, 512)`` fields, indexed by the **same** Lagrangian
  sites (this is the exactness Option B advertises; Option A inherits it as long
  as the crop frame is defined once and used for both);
* ``halos_particles/<box>__{hr,base}__.../*_owner.npy``, which say what each
  Lagrangian site's particle is bound to on each side;
* the two Rockstar catalogs, for masses and for the host/subhalo tree.

No halo finder runs, no GPU, nothing is trained.

What comes out
--------------
``<box>_host_crops.npz``
    Per selected host: three quantised ``grid^3`` volumes (SR2 local log
    density, HR local log density, HR subhalo-membership fraction), the HR and
    SR2 subhalo tables in crop coordinates, and an Eulerian particle
    subsample. This is the redraw source for the viewer.
``<box>_host_crops.json``
    The learnability table: per host, the AUC with which an SR2-only scalar
    ranks Lagrangian sites by "an HR subhalo forms here", against the two
    baselines that matter -- distance from the host centre (pure geometry, no
    SR2 information) and the same statistic read off the HR field (the ceiling
    a perfect model would hit).

Why local density and not the displacement itself
-------------------------------------------------
A subhalo is a *contrast* feature. ``|Psi|`` on a cluster patch is dominated by
the bulk infall the whole patch shares, so it ranks sites by where they are in
the host, not by whether they collapsed. The k-th nearest neighbour distance in
**Eulerian** space, gathered back to the Lagrangian site the particle came from,
is the cheapest scalar that asks the right question: did the material at this
Lagrangian site end up in a dense clump? Both fields get the identical estimator,
so the HR number is a like-for-like ceiling rather than a different measurement.

    python scripts/features/collect_host_crops.py --boxes set8 --n-hosts 12
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
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
from cosmo_sr.eval.particle_identity import (  # noqa: E402
    build_owner_index, child_map, descendants_of, root_lookup,
)
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.features import (  # noqa: E402
    auc, block_reduce, crop_frame, flat_to_sites, resample_report, roc_curve,
    to_crop_coords,
)

NG_HR = 512
BOXSIZE = 100.0            # Mpc/h
K_NEIGHBOURS = 32          # for the local density estimate
SMOOTH_SIGMAS = (0.0, 1.0, 2.0, 4.0)   # native sites


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _wrap(d: np.ndarray, box: float) -> np.ndarray:
    """Periodic offset into ``[-box/2, box/2)``."""
    return ((np.asarray(d, dtype=np.float64) + box / 2.0) % box) - box / 2.0


def _catalog(root: Path, box: str, tag: str):
    hits = sorted(glob.glob(str(root / "halos" / f"{box}__{tag}__{tag}"
                                 / "*_rockstar" / "halos_*.ascii")))
    if not hits:
        raise FileNotFoundError(f"no {tag} catalog under {root/'halos'} for {box}")
    return load_rockstar_ascii(hits[0])


def _owner(root: Path, box: str, tag: str) -> np.ndarray:
    hits = sorted(glob.glob(str(root / "halos_particles" / f"{box}__{tag}__{tag}"
                                 / f"{box}_{tag}_owner.npy")))
    if not hits:
        raise FileNotFoundError(
            f"no owner array for {box}/{tag}; run scripts/reward/rockstar_particles.py")
    return np.load(hits[0], mmap_mode="r")


def _sr2_field(root: Path, box: str, seed: int = 0) -> Path:
    hits = sorted((root / "cache" / "sr2_base").glob(f"{box}_seed{seed}_*.npy"))
    if not hits:
        raise FileNotFoundError(
            f"no cached SR2 field for {box} seed {seed}; "
            f"run scripts/slurm/cache_sr2_base.sbatch BOXES={box}")
    return hits[0]


# --------------------------------------------------------------------------
# Per-crop physics
# --------------------------------------------------------------------------

def crop_positions(field, frame, *, pad: int, redshift: float = 0.0):
    """Eulerian positions of a padded crop, in Mpc/h, unwrapped about its centre.

    Returns ``(pos, inner)`` where ``pos`` is ``(M, 3)`` over the padded cube and
    ``inner`` is the boolean mask of the sites that belong to ``frame`` itself.
    The padding exists only so the neighbour search has real neighbours at the
    faces; densities are never reported on it.

    Coordinates are unwrapped -- expressed as an offset from the crop's own
    Lagrangian centre rather than modulo the box -- so a crop straddling a box
    face is a contiguous cloud and needs no periodic tree.
    """
    side, ng = int(frame.side), int(frame.ng)
    wide = side + 2 * pad
    r = np.arange(wide, dtype=np.int64) - pad
    ax = [(int(frame.start[a]) + r) % ng for a in range(3)]

    sub = np.asarray(field[(slice(0, 3),) + np.ix_(*ax)], dtype=np.float32)
    disp = disnorm(sub.astype(np.float64), z=redshift, undo=True) * 1e-3  # Mpc/h

    cell = BOXSIZE / ng
    q = [((np.asarray(a, dtype=np.float64) + 0.5) * cell) for a in ax]
    # Unwrap q about the crop centre before adding the displacement: the modulo
    # in ax is what makes a face-straddling crop discontiguous, and the offset
    # form removes it. Displacements are already offsets and need no unwrapping.
    centre = (frame.centre + 0.5) * cell
    q = [((qa - centre[a] + BOXSIZE / 2) % BOXSIZE) - BOXSIZE / 2
         for a, qa in enumerate(q)]

    pos = np.empty((3, wide, wide, wide), dtype=np.float32)
    pos[0] = q[0].reshape(-1, 1, 1) + disp[0]
    pos[1] = q[1].reshape(1, -1, 1) + disp[1]
    pos[2] = q[2].reshape(1, 1, -1) + disp[2]

    inner = np.zeros((wide, wide, wide), dtype=bool)
    inner[pad:pad + side, pad:pad + side, pad:pad + side] = True
    return pos.reshape(3, -1).T.copy(), inner.reshape(-1)


def local_log_density(pos: np.ndarray, inner: np.ndarray, side: int,
                      *, k: int = K_NEIGHBOURS) -> np.ndarray:
    """``log10`` of a kNN density estimate, gathered onto the crop's sites.

    ``rho ~ k / d_k^3`` up to a constant that cancels in every rank statistic
    below. Queried for the inner sites only, but against the padded point set,
    so a site at the crop face sees the same neighbourhood an interior one does.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(pos)
    d, _ = tree.query(pos[inner], k=k + 1, workers=-1)
    dk = np.maximum(d[:, -1], 1e-6)
    return (np.log10(k) - 3.0 * np.log10(dk)).astype(np.float32).reshape(
        side, side, side)


def smoothed_aucs(vol: np.ndarray, label: np.ndarray, domain: np.ndarray) -> dict:
    """AUC of ``vol`` for ``label``, over ``domain``, at several smoothings.

    A single-site density is a noisy read of "is there a clump here"; smoothing
    it asks the same question of a neighbourhood. Reporting the ladder rather
    than one number keeps the scale at which the information sits visible.
    """
    from scipy.ndimage import gaussian_filter

    out = {}
    y = label[domain]
    for s in SMOOTH_SIGMAS:
        v = vol if s == 0 else gaussian_filter(vol, sigma=s, mode="nearest")
        out[f"sigma{s:g}"] = auc(v[domain], y)
    return out


# --------------------------------------------------------------------------
# One host
# --------------------------------------------------------------------------

def collect_host(host_id, *, cat, sr2_cat, oidx, hr_owner, sr2_owner, children,
                 roots, hr_field, sr2_field, args):
    row = int(np.flatnonzero(cat.ids == host_id)[0])
    subs = descendants_of(cat, int(host_id), children=children)
    members = oidx.members_with_substructure(cat, int(host_id), children=children)
    if members.size < args.min_particles:
        return None

    sites = flat_to_sites(members, NG_HR)
    frame = crop_frame(sites, NG_HR, scale=args.crop_scale,
                       min_side=args.min_side, max_side=args.max_side)
    side = frame.side
    ids = frame.flat_ids()

    # -- labels on the crop -------------------------------------------------
    own_hr = np.asarray(hr_owner[ids], dtype=np.int64)
    root_of = roots
    hr_root = np.where((own_hr >= 0) & (own_hr < root_of.size),
                       root_of[np.clip(own_hr, 0, root_of.size - 1)], -1)
    in_host = (hr_root == int(host_id))
    is_sub = in_host & (own_hr != int(host_id))
    sub_vol = is_sub.reshape(side, side, side)
    host_vol = in_host.reshape(side, side, side)

    sr2_own = np.asarray(sr2_owner[ids], dtype=np.int64)
    sr2_bound = (sr2_own >= 0).reshape(side, side, side)

    # -- fields -------------------------------------------------------------
    t0 = time.time()
    pos_hr, inner = crop_positions(hr_field, frame, pad=args.pad)
    hr_rho = local_log_density(pos_hr, inner, side)
    pos_sr, _ = crop_positions(sr2_field, frame, pad=args.pad)
    sr2_rho = local_log_density(pos_sr, inner, side)
    secs = time.time() - t0

    # -- learnability -------------------------------------------------------
    # Distance from the host's Lagrangian centre, in R_L units: the baseline
    # that uses no SR2 information at all. If SR2's density cannot beat it,
    # SR2 is contributing geometry and nothing else.
    g = np.arange(side, dtype=np.float32) - (side - 1) / 2.0
    r_site = np.sqrt(g[:, None, None] ** 2 + g[None, :, None] ** 2
                     + g[None, None, :] ** 2)
    rl_sites = side / (2.0 * max(args.crop_scale, 1e-9))
    radius = (-r_site / rl_sites).astype(np.float32)   # negated: near centre = high

    # Two domains. "footprint" is the honest one -- the model is only ever
    # asked where inside the host's own material a subhalo goes; "crop" adds
    # the surrounding field, where the answer is trivially no.
    domains = {"footprint": host_vol, "crop": np.ones_like(host_vol)}
    learn = {}
    for name, dom in domains.items():
        y = sub_vol[dom]
        if y.sum() == 0 or (~y).sum() == 0:
            continue
        learn[name] = {
            "n_sites": int(dom.sum()),
            "base_rate": float(y.mean()),
            "auc_sr2_rho": smoothed_aucs(sr2_rho, sub_vol, dom),
            "auc_hr_rho": smoothed_aucs(hr_rho, sub_vol, dom),
            "auc_radius": auc(radius[dom], y),
            "auc_sr2_bound": auc(sr2_bound[dom].astype(np.float32), y),
            "roc_sr2_rho": roc_curve(sr2_rho[dom], y),
            "roc_hr_rho": roc_curve(hr_rho[dom], y),
        }

    # -- the subhalo tables -------------------------------------------------
    hr_subs = subhalo_table(subs, cat, oidx, frame, sr2_own_vol=sr2_own,
                            sr2_rho=sr2_rho, ids=ids, host_row=row)
    sr2_subs = sr2_subhalo_table(sr2_cat, frame, args)

    # -- payload ------------------------------------------------------------
    vols, extent, g_out = {}, side, side
    for name, arr, how in (("sr2_rho", sr2_rho, "max"),
                           ("hr_rho", hr_rho, "max"),
                           ("hr_sub", sub_vol.astype(np.float32), "mean"),
                           ("hr_host", host_vol.astype(np.float32), "mean")):
        red, extent = block_reduce(arr, args.grid, how)
        vols[name] = red
        g_out = int(red.shape[0])

    cell = BOXSIZE / NG_HR
    meta = {
        "halo_id": int(host_id),
        "mvir": float(cat.mvir[row]),
        "log_mvir": float(np.log10(max(cat.mvir[row], 1.0))),
        "num_p": int(cat.num_p[row]),
        "rvir_mpc_h": float(cat.rvir[row]) * 1e-3,
        "n_member_sites": int(members.size),
        "n_subhalos": int(len(subs)),
        "n_subhalos_20p": int(sum(1 for s in hr_subs if s["num_p"] >= 20)),
        "crop_side_sites": int(side),
        "crop_side_mpc_h": float(side * cell),
        "crop_centre_site": frame.centre.round(3).tolist(),
        "rl_sites": float(rl_sites),
        "rl_mpc_h": float(rl_sites * cell),
        "resample": resample_report(frame, args.target_grid),
        "grid_out": int(g_out),
        # The cube spans `extent` native sites, not `side`: the overlay must
        # scale by this or the circles drift outward across the crop.
        "vol_extent_sites": int(extent),
        "block_factor": float(extent / g_out),
        "vol_pad_sites": int(extent - side),
        "host_fill_frac": float(host_vol.mean()),
        "sub_frac_of_host": float(sub_vol.sum() / max(host_vol.sum(), 1)),
        "sr2_bound_frac_in_host": float(sr2_bound[host_vol].mean()),
        "seconds": round(secs, 2),
        "learnability": learn,
    }
    return meta, vols, hr_subs, sr2_subs, frame, pos_hr, pos_sr, inner, side


def subhalo_table(subs, cat, oidx, frame, *, sr2_own_vol, sr2_rho, ids,
                  host_row, boxsize=BOXSIZE):
    """Every HR subhalo of the host, placed in the crop's coordinates.

    ``sr2_bound_frac`` is the number this table exists for: of the Lagrangian
    sites HR binds into *this* subhalo, what fraction does SR2 bind into any
    object at all. It is the per-object form of the AUC above and does not
    depend on a positional match.
    """
    if not subs:
        return []
    side = frame.side
    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]
    sr2_flat = sr2_own_vol
    rho_flat = sr2_rho.reshape(-1)

    id_to_row = {int(i): r for r, i in enumerate(cat.ids)}
    host_pos = cat.pos[host_row]
    host_rvir = max(float(cat.rvir[host_row]) * 1e-3, 1e-6)   # kpc/h -> Mpc/h
    out = []
    for sid in subs:
        r = id_to_row.get(int(sid))
        if r is None:
            continue
        mem = oidx.members(int(sid))
        if mem.size == 0:
            continue
        s = flat_to_sites(mem, frame.ng)
        u = to_crop_coords(s, frame)
        inside = np.all((u >= 0) & (u < side), axis=1)
        if not inside.any():
            continue
        # Locate the members inside the crop within the crop's own id list.
        k = np.searchsorted(ids_sorted, mem[inside])
        k = np.clip(k, 0, ids_sorted.size - 1)
        hit = ids_sorted[k] == mem[inside]
        idx = order[k[hit]]
        centre = u[inside].mean(axis=0)
        out.append({
            "halo_id": int(sid),
            "num_p": int(cat.num_p[r]),
            "mvir": float(cat.mvir[r]),
            "log_mvir": float(np.log10(max(cat.mvir[r], 1.0))),
            "rvir_mpc_h": float(cat.rvir[r]) * 1e-3,
            "u": centre.round(3).tolist(),
            "d_mpc_h": _wrap(cat.pos[r] - host_pos, boxsize).round(4).tolist(),
            "r_over_rvir": float(
                np.linalg.norm(_wrap(cat.pos[r] - host_pos, boxsize)) / host_rvir),
            "frac_in_crop": float(inside.mean()),
            "rl_sites": float((3.0 * mem.size / (4.0 * np.pi)) ** (1.0 / 3.0)),
            "sr2_bound_frac": float((sr2_flat[idx] >= 0).mean()) if idx.size else 0.0,
            "sr2_logrho": float(np.mean(rho_flat[idx])) if idx.size else float("nan"),
        })
    out.sort(key=lambda d: -d["num_p"])
    return out


def sr2_subhalo_table(sr2_cat, frame, args):
    """SR2's own catalog objects, for the same crop, by Eulerian position.

    This one *is* a positional match and is labelled as such: SR2 rows carry no
    Lagrangian identity here (the owner array does, and the AUC above uses it).
    It is drawn only so the page can show what SR2 believes is in the region.
    """
    cell = BOXSIZE / frame.ng
    centre = (frame.centre + 0.5) * cell
    d = ((sr2_cat.pos - centre[None, :] + BOXSIZE / 2) % BOXSIZE) - BOXSIZE / 2
    half = frame.side * cell / 2.0
    m = np.all(np.abs(d) <= half, axis=1) & (sr2_cat.num_p >= args.min_sub_p)
    idx = np.flatnonzero(m)
    if idx.size > args.max_sr2_rows:
        idx = idx[np.argsort(-sr2_cat.num_p[idx])[:args.max_sr2_rows]]
    return [{
        "halo_id": int(sr2_cat.ids[i]),
        "num_p": int(sr2_cat.num_p[i]),
        "log_mvir": float(np.log10(max(sr2_cat.mvir[i], 1.0))),
        "is_sub": bool(sr2_cat.parent_ids[i] >= 0),
        "d_mpc_h": d[i].round(4).tolist(),
    } for i in idx]


# --------------------------------------------------------------------------
# Host selection
# --------------------------------------------------------------------------

def select_hosts(cat, args):
    """A mass ladder: every cluster, then a fixed count per half-decade below.

    Sampling per bin rather than taking the top N is deliberate -- the deficit
    is a function of host mass, so a page that only shows clusters cannot show
    where it turns on.
    """
    host = np.flatnonzero(cat.parent_ids < 0)
    host = host[cat.num_p[host] >= args.min_particles]
    lm = np.log10(np.maximum(cat.mvir[host], 1.0))
    rng = np.random.default_rng(args.seed)

    keep = []
    top = host[lm >= args.cluster_log_mvir]
    keep.extend(top[np.argsort(-cat.mvir[top])][:args.n_clusters].tolist())
    edges = np.arange(args.min_log_mvir, args.cluster_log_mvir + 1e-9, 0.5)
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = host[(lm >= lo) & (lm < hi)]
        if pool.size == 0:
            continue
        pick = rng.choice(pool, size=min(args.per_bin, pool.size), replace=False)
        keep.extend(np.atleast_1d(pick).tolist())
    rows = np.array(sorted(set(int(k) for k in keep)), dtype=np.int64)
    rows = rows[np.argsort(-cat.mvir[rows])]
    return cat.ids[rows][:args.n_hosts]


# --------------------------------------------------------------------------
# Box driver
# --------------------------------------------------------------------------

def run_box(box: str, cfg, args) -> dict:
    root = paths.reward_root()
    banner(f"host crops: {box}")

    cat = _catalog(root, box, "hr")
    sr2_cat = _catalog(root, box, "base")
    hr_owner = _owner(root, box, "hr")
    sr2_owner = _owner(root, box, "base")
    oidx = build_owner_index(np.asarray(hr_owner))
    children = child_map(cat)
    roots = root_lookup(cat)

    hr_field = np.load(hr_path(cfg, box), mmap_mode="r")
    sr2_field = np.load(str(_sr2_field(root, box, args.seed_sr2)), mmap_mode="r")

    ids = select_hosts(cat, args)
    print(f"  {cat.n} HR rows, {int((cat.parent_ids < 0).sum())} hosts; "
          f"{ids.size} selected")

    metas, store, t0 = [], {}, time.time()
    for j, hid in enumerate(ids):
        got = collect_host(int(hid), cat=cat, sr2_cat=sr2_cat, oidx=oidx,
                           hr_owner=hr_owner, sr2_owner=sr2_owner,
                           children=children, roots=roots, hr_field=hr_field,
                           sr2_field=sr2_field, args=args)
        if got is None:
            continue
        meta, vols, hr_subs, sr2_subs, frame, pos_hr, pos_sr, inner, side = got
        key = f"h{int(hid)}"
        for name, v in vols.items():
            store[f"{key}__{name}"] = _quant(v, name)
        store[f"{key}__scatter_hr"] = _scatter(pos_hr, inner, side, frame, args)
        store[f"{key}__scatter_sr2"] = _scatter(pos_sr, inner, side, frame, args)
        meta["hr_subhalos"] = hr_subs
        meta["sr2_objects"] = sr2_subs
        metas.append(meta)
        print(f"  [{j + 1}/{ids.size}] id={int(hid)} "
              f"logM={meta['log_mvir']:.2f} side={side} "
              f"nsub={meta['n_subhalos']} "
              f"AUC(sr2)={_pick(meta):.3f} ({meta['seconds']}s)", flush=True)

    out_dir = paths.subdir("lagrangian_host", box, create=True)
    npz = out_dir / f"{box}_host_crops.npz"
    np.savez_compressed(npz, **store)
    summary = {
        "box": box,
        "ok": True,
        "n_hosts": len(metas),
        "grid": args.grid,
        "crop_scale": args.crop_scale,
        "k_neighbours": K_NEIGHBOURS,
        "pad_sites": args.pad,
        "boxsize_mpc_h": BOXSIZE,
        "ng_hr": NG_HR,
        "particle_mass_msun_h": float(args.particle_mass),
        "hr_field": str(hr_path(cfg, box)),
        "sr2_field": str(_sr2_field(root, box, args.seed_sr2)),
        "npz": str(npz),
        "seconds": round(time.time() - t0, 1),
        "hosts": metas,
    }
    write_json(out_dir / f"{box}_host_crops.json", summary)
    print(f"  wrote {npz}")
    print(f"  wrote {out_dir / f'{box}_host_crops.json'}")
    return summary


def _pick(meta) -> float:
    d = meta.get("learnability", {}).get("footprint")
    if not d:
        return float("nan")
    return d["auc_sr2_rho"]["sigma2"]


def _quant(v: np.ndarray, name: str) -> np.ndarray:
    """float cube -> uint8 with the range appended, so the page can invert it."""
    v = np.asarray(v, dtype=np.float32)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(lo) or hi <= lo:
        lo, hi = 0.0, 1.0
    v = np.nan_to_num(v, nan=lo)
    q = np.clip((v - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return np.concatenate([np.array([lo, hi], dtype=np.float32).view(np.uint8),
                           q.reshape(-1)])


def _scatter(pos, inner, side, frame, args):
    """A subsample of the crop's Eulerian cloud, as int16 in units of 1/64 Mpc/h."""
    p = pos[inner]
    n = min(args.n_scatter, p.shape[0])
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(p.shape[0], size=n, replace=False) if n < p.shape[0] \
        else np.arange(p.shape[0])
    q = np.clip(np.round(p[sel] * 64.0), -32768, 32767).astype(np.int16)
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--boxes", default="set8")
    ap.add_argument("--n-hosts", type=int, default=16)
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--per-bin", type=int, default=2)
    ap.add_argument("--cluster-log-mvir", type=float, default=14.0)
    ap.add_argument("--min-log-mvir", type=float, default=12.0)
    ap.add_argument("--min-particles", type=int, default=200)
    ap.add_argument("--min-sub-p", type=int, default=20)
    ap.add_argument("--crop-scale", type=float, default=1.0)
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--max-side", type=int, default=192)
    ap.add_argument("--grid", type=int, default=48, help="stored volume side")
    ap.add_argument("--target-grid", type=int, default=96,
                    help="Option A's fixed grid, reported not applied")
    ap.add_argument("--pad", type=int, default=8, help="halo of sites for the kNN")
    ap.add_argument("--n-scatter", type=int, default=6000)
    ap.add_argument("--max-sr2-rows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-sr2", type=int, default=0)
    ap.add_argument("--particle-mass", type=float, default=5.81881e8)
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    for box in [b for b in args.boxes.split(",") if b]:
        run_box(box, cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
