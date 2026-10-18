#!/usr/bin/env python
"""Are the density bumps real? Cutouts for viewing, and an angular-shuffle null.

The low-contrast census counts every local k=32 density maximum above 3x its
shell mean. A bump at that threshold can be a clump -- or Poisson scatter, the
density gradient across a wide radial bin, or an infall stream. This writes:

* a NULL per field: around ``--null-hosts`` hosts, every particle at
  ``--null-x`` R_vir keeps its distance from the host centre and gets a random
  direction -- the radial profile and particle count survive, clumps do not --
  and the census's own bump finder (``host_phase_profile.peak_profile``) is
  re-run. Bumps per contrast band, observed vs null, over the same hosts;
* CUTOUTS: ``--per-cell`` bumps per (contrast band x became-a-Rockstar-object)
  at ``--x-min..--x-max`` R_vir, each the particles within a ``--cut`` Mpc/h
  cube, positions about the bump and velocities about its 32-nearest mean;
* CONTEXT: one host's slab (|dz| < ``--slab`` R_vir) as a density image, with its
  bumps and Rockstar objects, in the host frame.

Everything a figure needs goes to ``<out-dir>/<name>_bumps.npz`` + ``bump_null.json``
so ``bump_viz.py`` redraws without recomputing.
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
    BOX, CellIndex, peak_profile, select_region_hosts, shrinking_centre,
)
from peak_halo_match import match  # noqa: E402
from placement_forensics import min_image, read_rockstar  # noqa: E402
from cosmo_sr.eval.particles import field_to_particles  # noqa: E402

BANDS = (3.0, 10.0, 30.0, 100.0, np.inf)
BAND_LABELS = ["3-10x", "10-30x", "30-100x", ">100x"]


def band_of(ratio: np.ndarray) -> np.ndarray:
    return np.digitize(ratio, BANDS[1:-1], right=True)


def shuffle_directions(d: np.ndarray, rvir: float, x_lo: float, x_hi: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Particles at x_lo..x_hi R_vir keep |d| and get a uniformly random direction."""
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    m = (r >= x_lo * rvir) & (r < x_hi * rvir)
    u = rng.normal(size=(int(m.sum()), 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    out = d.copy()
    out[m] = u * r[m, None]
    return out


def host_frame(pos, ci, c_hr, rv):
    reach = 3.0 * rv + max(0.3 * rv, 0.3)
    idx = ci.gather(c_hr, reach)
    d = min_image(pos[idx].astype(np.float64) - c_hr)
    keep = np.einsum("ij,ij->i", d, d) < reach * reach
    idx, d = idx[keep], d[keep]
    c = shrinking_centre(d, 0.5 * rv, 0.05 * rv)
    return idx, d - c, c + c_hr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--peaks", required=True, help="census peaks.npz of THIS field")
    ap.add_argument("--catalog", required=True, help="this field's Rockstar halos ascii")
    ap.add_argument("--hr-catalog", required=True)
    ap.add_argument("--region-export", required=True)
    ap.add_argument("--x-min", type=float, default=0.5)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--cut", type=float, default=0.25, help="cutout half-size, Mpc/h")
    ap.add_argument("--null-hosts", type=int, default=12)
    ap.add_argument("--null-x", default="0.35,1.15")
    ap.add_argument("--context-num-p", type=int, default=150000,
                    help="context host = the region host with num_p nearest this")
    ap.add_argument("--slab", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    hr = read_rockstar(a.hr_catalog)
    sel = select_region_hosts(hr, a.region_export, 2.0, 10000, 60)
    rvir = hr["rvir"][sel] / 1000.0
    pk = dict(np.load(a.peaks))
    census = json.loads(Path(a.peaks.replace("_peaks.npz", ".json")).read_text())
    if [int(i) for i in hr["id"][sel]] != [int(i) for i in census["per_host_scalars"]["host_id"]]:
        raise SystemExit("host list differs from the census that wrote the peaks")
    cat = read_rockstar(a.catalog)
    keep = cat["num_p"] >= 20
    m = match(pk["pos"], cat["pos"][keep], cat["rvir"][keep] / 1000.0 * 0.5, 0.03)
    hit = m >= 0
    obj_np = np.where(hit, cat["num_p"][keep][np.maximum(m, 0)], 0)

    field = np.load(a.field, mmap_mode="r")
    pb = field_to_particles(field, boxsize_kpc_h=BOX * 1000.0)
    del field
    pos, vel = pb.pos_mpc_h, pb.vel_kms
    ci = CellIndex(pos)
    print(f"{a.name}: particles + cell index {time.time() - t0:.0f}s", flush=True)
    from scipy.spatial import cKDTree

    # --- cutouts ---------------------------------------------------------------
    inr = (pk["x"] >= a.x_min) & (pk["x"] < a.x_max) & (pk["ratio"] > BANDS[0])
    b = band_of(pk["ratio"])
    chosen = []
    for bi in range(4):
        for h in (1, 0):
            cand = np.where(inr & (b == bi) & (hit == bool(h)))[0]
            if cand.size:
                chosen.extend(rng.choice(cand, min(a.per_cell, cand.size), replace=False).tolist())
    cut_pos, cut_vel, offs, meta = [], [], [0], []
    for i in chosen:
        p0 = pk["pos"][i].astype(np.float64)
        idx = ci.gather(p0, a.cut * np.sqrt(3.0))
        d = min_image(pos[idx].astype(np.float64) - p0)
        inside = np.all(np.abs(d) < a.cut, axis=1)
        idx, d = idx[inside], d[inside]
        if idx.size < 40:
            continue
        tree = cKDTree(d)
        _, nn = tree.query(np.zeros(3), k=32)
        vc = vel[idx][nn].astype(np.float64).mean(axis=0)
        cut_pos.append(d.astype(np.float32))
        cut_vel.append((vel[idx].astype(np.float64) - vc).astype(np.float32))
        offs.append(offs[-1] + idx.size)
        meta.append([i, int(b[i]), int(hit[i]), float(pk["ratio"][i]), float(pk["x"][i]),
                     int(obj_np[i])])
    print(f"  {len(meta)} cutouts, {time.time() - t0:.0f}s", flush=True)

    # --- context host slab -----------------------------------------------------
    jc = int(np.argmin(np.abs(hr["num_p"][sel].astype(float) - a.context_num_p)))
    rv = float(rvir[jc])
    idx, d, centre = host_frame(pos, ci, hr["pos"][sel[jc]], rv)
    slab = np.abs(d[:, 2]) < a.slab * rv
    ext = 1.2 * rv
    img, _, _ = np.histogram2d(d[slab, 0], d[slab, 1], bins=400, range=[[-ext, ext], [-ext, ext]])
    pc = min_image(pk["pos"].astype(np.float64) - centre)
    in_slab = (np.abs(pc[:, 2]) < a.slab * rv) & (np.abs(pc[:, 0]) < ext) & (np.abs(pc[:, 1]) < ext) \
        & (pk["ratio"] > BANDS[0])
    oc = min_image(cat["pos"][keep].astype(np.float64) - centre)
    o_slab = (np.abs(oc[:, 2]) < a.slab * rv) & (np.abs(oc[:, 0]) < ext) & (np.abs(oc[:, 1]) < ext)

    # --- angular-shuffle null over the first null-hosts hosts -------------------
    xl, xh = (float(t) for t in a.null_x.split(","))
    obs = np.zeros(4, dtype=np.int64)
    nul = np.zeros(4, dtype=np.int64)
    for j in range(min(a.null_hosts, sel.size)):
        rvj = float(rvir[j])
        idx_j, d_j, _ = host_frame(pos, ci, hr["pos"][sel[j]], rvj)
        ds = shuffle_directions(d_j, rvj, xl, xh, rng)
        pr = peak_profile(ds, vel[idx_j].astype(np.float64), rvj, pb.particle_mass_msun_h,
                          records=True, min_ratio=BANDS[0])["_records"]
        rows = pr["row"].astype(np.int64)
        xn = np.sqrt(np.einsum("ij,ij->i", ds[rows], ds[rows])) / rvj
        sn = (xn >= a.x_min) & (xn < a.x_max)
        nul += np.bincount(band_of(pr["ratio"][sn]), minlength=4)[:4]
        so = (pk["host"] == j) & inr
        obs += np.bincount(b[so], minlength=4)[:4]
        print(f"  null host {j + 1}: observed {obs.tolist()} null {nul.tolist()}, "
              f"{time.time() - t0:.0f}s", flush=True)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{a.name}_bumps.npz",
        cut_pos=np.concatenate(cut_pos) if cut_pos else np.zeros((0, 3), np.float32),
        cut_vel=np.concatenate(cut_vel) if cut_vel else np.zeros((0, 3), np.float32),
        cut_offsets=np.asarray(offs), cut_meta=np.asarray(meta, dtype=np.float64),
        ctx_img=img.astype(np.float32), ctx_extent=np.asarray([ext]), ctx_rvir=np.asarray([rv]),
        ctx_bumps=pc[in_slab, :2].astype(np.float32), ctx_bump_band=b[in_slab],
        ctx_bump_hit=hit[in_slab], ctx_objects=oc[o_slab, :2].astype(np.float32),
        ctx_object_np=cat["num_p"][keep][o_slab], ctx_host_id=np.asarray([int(hr["id"][sel[jc]])]))
    nj = out / "bump_null.json"
    allr = json.loads(nj.read_text()) if nj.is_file() else {}
    allr[a.name] = {"bands": BAND_LABELS, "x_range": [a.x_min, a.x_max],
                    "null_hosts": int(min(a.null_hosts, sel.size)), "shuffled_x": [xl, xh],
                    "observed": obs.tolist(), "null": nul.tolist(),
                    "real_fraction": [float(1 - n / o) if o else float("nan")
                                      for o, n in zip(obs, nul)]}
    write_json(nj, allr)
    print(f"wrote {out / (a.name + '_bumps.npz')} and {nj}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
