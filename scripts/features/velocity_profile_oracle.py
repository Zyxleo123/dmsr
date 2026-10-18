#!/usr/bin/env python
"""Velocity-temperature oracle: give a field HR's host-centric local temperature, then run the halo finder.

``host_phase_profile.py`` measured, around the 60 most massive HR hosts of the
held-out edited region, that every SR field (frozen and tuned) has HR's
DENSITY profile to ~10% but the wrong LOCAL velocity dispersion (k=32
neighbours): 0.68-0.75x HR inside 0.1 R_vir (0.34x for the tuned arm's
>=100k-particle hosts) and 1.3-1.75x HR beyond 0.75 R_vir, where HR's own
host-to-host scatter is 3-12%. The halo-finder reading of the same fields is
an inverted subhalo radial distribution: the core fragments 13x per
structure, the outskirts hold 15-23% of HR's self-bound satellites.

If the temperature profile is the CAUSE, then changing only the velocities --
each particle's deviation from its local (k-NN) mean velocity rescaled so the
local dispersion of every radial bin matches HR's stacked profile, positions
untouched -- should (a) collapse the core fragments and (b) bind the outskirt
clumps. This script performs exactly that intervention and runs the frozen
Rockstar config on the result. It is not a model and is not deployable as is
(it uses HR's host catalog for centres and R_vir); it asks whether a loss that
reached this target would fix the radial distribution, before any loss is
trained.

Per host (each particle is owned by the host with the smallest r/R_vir among
those whose 3 R_vir sphere contains it):

    v_loc  = mean velocity of the particle's k nearest neighbours (original v)
    s_bin  = sigma_HR(bin) * V_vir / median_bin |v - v_loc|_rms   (clipped)
    v'     = v_loc + (v - v_loc) * s_bin          only in the bins --mode selects

``--mode all`` rescales every bin, ``core`` only r < --split R_vir (heating the
core), ``out`` only r >= --split (cooling the outskirts). The target is HR's
stacked ``sigma_loc`` median for the host's mass class (>=100k or <100k
particles), in units of the host's V_vir. A verification pass re-measures the
achieved local dispersion with the new velocities.

    python scripts/features/velocity_profile_oracle.py --field <npy> --name cs_vfix_all \\
        --mode all --hr-profile host_profiles/set9__hr.json --hr-catalog <halos.ascii> \\
        --region-export <export.json> --out-dir <reward>/velocity_oracle
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from gather_holdout_figures_data import in_tile_region, tile_boxes  # noqa: E402
from host_phase_profile import (  # noqa: E402
    BOX, G_MPC, K_NN, CellIndex, shrinking_centre,
)
from placement_forensics import min_image, read_rockstar  # noqa: E402
from cosmo_sr.eval.particles import ParticleBox, field_to_particles  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

FROZEN_CFG = PROJECT_ROOT / "configs" / "sr2_baseline" / "rockstar.cfg"


def local_mean_and_dispersion(tree, d_all: np.ndarray, v_all: np.ndarray,
                              q: np.ndarray, k: int = K_NN, chunk: int = 400000
                              ) -> tuple:
    """Per query row: mean velocity of its k neighbours (self excluded) and the
    rms dispersion of those neighbours about that mean.

    The dispersion is host_phase_profile's ``sigma_loc`` estimator exactly (the
    target is HR's median of it), so a scale computed from it reaches the
    target instead of a fixed multiple of it. ``q`` indexes rows of
    ``d_all``/``v_all`` (the tree was built on d_all).
    """
    vloc = np.empty((q.size, 3), dtype=np.float64)
    sig = np.empty(q.size, dtype=np.float64)
    for s in range(0, q.size, chunk):
        qq = q[s:s + chunk]
        _, nbr = tree.query(d_all[qq], k=k + 1, workers=-1)
        vn = v_all[nbr[:, 1:]]                         # (Q, K, 3), self dropped
        vm = vn.mean(axis=1)
        vn = vn - vm[:, None, :]
        vloc[s:s + qq.size] = vm
        sig[s:s + qq.size] = np.sqrt(np.mean(np.einsum("qkj,qkj->qk", vn, vn), axis=1))
    return vloc, sig


def field_with_velocities(field, v_new: np.ndarray) -> np.ndarray:
    """The (6,N,N,N) field with its velocity channels replaced by ``v_new``.

    Displacement is copied verbatim; ``v_new`` ((N^3, 3) km/s, flat lattice order,
    as ``field_to_particles`` returns it) is re-normalised by ``velnorm`` so the
    same ``field_to_particles`` reads it back (pinned by a round-trip test).
    """
    from cosmo_sr.data.preprocess_srs import velnorm
    ng = int(field.shape[1])
    out = np.empty(field.shape, dtype=np.float32)
    out[0:3] = field[0:3]
    out[3:6] = velnorm(np.asarray(v_new, dtype=np.float64).T.reshape(3, ng, ng, ng),
                       z=0.0).astype(np.float32)
    return out


def bin_scale(x: np.ndarray, sig: np.ndarray, edges: np.ndarray,
              target: np.ndarray, vvir: float, mode: str, split: float,
              s_lo: float, s_hi: float) -> np.ndarray:
    """Scale factor per radial bin; 1 where the mode leaves a bin alone.

    The field's local dispersion per bin is the median of the per-particle
    neighbourhood dispersion -- host_phase_profile's ``sigma_loc`` statistic,
    which is what the HR target is the median of.
    """
    nb = edges.size - 1
    s = np.ones(nb)
    b = np.digitize(x, edges) - 1
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        if mode == "core" and lo >= split:
            continue
        if mode == "out" and hi <= split:
            continue
        m = b == i
        if m.sum() < 50 or not np.isfinite(target[i]):
            continue
        cur = float(np.median(sig[m]))
        if cur > 0:
            s[i] = float(np.clip(target[i] * vvir / cur, s_lo, s_hi))
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=("all", "core", "out"), default="all")
    ap.add_argument("--split", type=float, default=0.5,
                    help="r/R_vir boundary between 'core' and 'out' (default 0.5)")
    ap.add_argument("--hr-profile", required=True,
                    help="host_phase_profile.py JSON of the HR field (the target)")
    ap.add_argument("--hr-catalog", required=True)
    ap.add_argument("--region-export", required=True)
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--min-host-num-p", type=int, default=10000)
    ap.add_argument("--max-hosts", type=int, default=60)
    ap.add_argument("--reach", type=float, default=3.0,
                    help="particles are rescaled out to this many R_vir")
    ap.add_argument("--s-lo", type=float, default=0.25)
    ap.add_argument("--s-hi", type=float, default=4.0)
    ap.add_argument("--verify-per-bin", type=int, default=4000)
    ap.add_argument("--save-field", default="",
                    help="also write the modified (6,N,N,N) field here (for SUBFIND)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    t0 = time.time()
    prof = json.loads(Path(a.hr_profile).read_text())
    edges = np.asarray(prof["edges_x"], dtype=np.float64)
    tgt_big = np.asarray(prof["stack_ge100k"]["sigma_loc"]["median"], dtype=np.float64)
    tgt_small = np.asarray(prof["stack_lt100k"]["sigma_loc"]["median"], dtype=np.float64)

    export = json.loads(Path(a.region_export).read_text())
    tiles = [int(t) for t in export["tiles"]] if "tiles" in export else sorted(
        {int(t) for h in export["per_host"] for t in h["tiles"]})
    boxes = tile_boxes(tiles, TileGrid())
    hr = read_rockstar(a.hr_catalog)
    sel = np.where((hr["parent_id"] < 0) & (hr["num_p"] >= a.min_host_num_p)
                   & in_tile_region(hr["pos"], boxes, dilate=a.dilate))[0]
    sel = sel[np.argsort(-hr["num_p"][sel])][: a.max_hosts]
    rvir = hr["rvir"][sel] / 1000.0
    vvir = np.sqrt(G_MPC * hr["mvir"][sel] / rvir)
    print(f"{sel.size} HR hosts (num_p {hr['num_p'][sel].min()}-"
          f"{hr['num_p'][sel].max()}), mode {a.mode}, split {a.split}", flush=True)

    field = np.load(a.field, mmap_mode="r")
    pb = field_to_particles(field, boxsize_kpc_h=BOX * 1000.0)
    pos, vel = pb.pos_mpc_h, pb.vel_kms
    ci = CellIndex(pos)
    print(f"particles + cell index in {time.time() - t0:.0f}s", flush=True)

    # --- ownership: smallest r/R_vir among the hosts whose sphere holds it ---
    n = pos.shape[0]
    own = np.full(n, -1, dtype=np.int16)
    own_x = np.full(n, np.inf, dtype=np.float32)
    gathered, centres = [], []
    for j, h in enumerate(sel):
        rv = float(rvir[j])
        reach = a.reach * rv + max(0.3 * rv, 0.3)
        idx = ci.gather(hr["pos"][h], reach)
        d = min_image(pos[idx].astype(np.float64) - hr["pos"][h])
        keep = np.einsum("ij,ij->i", d, d) < reach * reach
        idx, d = idx[keep], d[keep]
        c = shrinking_centre(d, 0.5 * rv, 0.05 * rv)
        d = d - c
        x = np.sqrt(np.einsum("ij,ij->i", d, d)) / rv
        better = (x < a.reach) & (x < own_x[idx])
        own[idx[better]] = j
        own_x[idx[better]] = x[better]
        gathered.append(idx)
        centres.append(c + hr["pos"][h])
    print(f"ownership: {int(np.sum(own >= 0))} particles in {sel.size} hosts, "
          f"{time.time() - t0:.0f}s", flush=True)

    from scipy.spatial import cKDTree
    v_new = vel.copy()
    per_host: List[Dict] = []
    trees = []
    for j, h in enumerate(sel):
        idx = gathered[j]
        rv, vv = float(rvir[j]), float(vvir[j])
        d_all = min_image(pos[idx].astype(np.float64) - centres[j])
        v_all = vel[idx].astype(np.float64)
        tree = cKDTree(d_all)
        mine = np.where(own[idx] == j)[0]
        x = np.sqrt(np.einsum("ij,ij->i", d_all[mine], d_all[mine])) / rv
        vloc, sig = local_mean_and_dispersion(tree, d_all, v_all, mine)
        target = tgt_big if hr["num_p"][h] >= 100000 else tgt_small
        s = bin_scale(x, sig, edges, target, vv, a.mode, a.split, a.s_lo, a.s_hi)
        b = np.digitize(x, edges) - 1
        sb = s[np.clip(b, 0, s.size - 1)]
        v_new[idx[mine]] = (vloc + (v_all[mine] - vloc) * sb[:, None]).astype(np.float32)
        per_host.append({"host_id": int(hr["id"][h]), "num_p": int(hr["num_p"][h]),
                         "rvir": rv, "vvir": vv, "n_owned": int(mine.size),
                         "scale": s.tolist()})
        trees.append((tree, d_all))
        if j % 10 == 0:
            print(f"  host {j + 1}/{sel.size} id {hr['id'][h]}: {mine.size} owned, "
                  f"scale core {s[0]:.2f} / R_vir {s[7]:.2f} / 2R_vir {s[10]:.2f}, "
                  f"{time.time() - t0:.0f}s", flush=True)

    # --- verification: achieved local dispersion with the NEW velocities ------
    rng = np.random.default_rng(a.seed)
    for j, h in enumerate(sel):
        idx = gathered[j]
        tree, d_all = trees[j]
        rv, vv = float(rvir[j]), float(vvir[j])
        mine = np.where(own[idx] == j)[0]
        x = np.sqrt(np.einsum("ij,ij->i", d_all[mine], d_all[mine])) / rv
        b = np.digitize(x, edges) - 1
        pick = []
        for i in range(edges.size - 1):
            m = np.where(b == i)[0]
            if m.size:
                pick.append(rng.choice(m, min(m.size, a.verify_per_bin), replace=False))
        pick = mine[np.concatenate(pick)] if pick else mine[:0]
        target = tgt_big if hr["num_p"][h] >= 100000 else tgt_small
        for label, vsrc in (("before", vel), ("after", v_new)):
            _, sig = local_mean_and_dispersion(tree, d_all, vsrc[idx].astype(np.float64), pick)
            xb = np.digitize(np.sqrt(np.einsum("ij,ij->i", d_all[pick], d_all[pick])) / rv,
                             edges) - 1
            ratio = [float(np.median(sig[xb == i]) / (target[i] * vv))
                     if np.any(xb == i) and np.isfinite(target[i]) else float("nan")
                     for i in range(edges.size - 1)]
            per_host[j][f"sigma_loc_over_target_{label}"] = ratio
    del trees

    def stack(key):
        arr = np.asarray([p[key] for p in per_host], dtype=np.float64)
        return np.nanmedian(arr, axis=0).tolist()

    summary = {"name": a.name, "field": a.field, "mode": a.mode, "split": a.split,
               "edges_x": edges.tolist(), "n_hosts": int(sel.size),
               "n_particles_rescaled": int(np.sum(own >= 0)),
               "median_scale": stack("scale"),
               "median_sigma_loc_over_target_before": stack("sigma_loc_over_target_before"),
               "median_sigma_loc_over_target_after": stack("sigma_loc_over_target_after"),
               "per_host": per_host}
    print("median scale per bin:", np.round(summary["median_scale"], 2).tolist())
    print("sigma_loc / target before:",
          np.round(summary["median_sigma_loc_over_target_before"], 2).tolist())
    print("sigma_loc / target after: ",
          np.round(summary["median_sigma_loc_over_target_after"], 2).tolist(), flush=True)

    out = Path(a.out_dir) / f"{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{a.name}_oracle.json", summary)

    if a.save_field:
        f2 = field_with_velocities(field, v_new)
        np.save(a.save_field, f2)
        del f2
        print(f"wrote modified field {a.save_field}", flush=True)
    del field

    from cosmo_sr.eval.rockstar import run_rockstar_on_particles
    pb2 = ParticleBox(pos_mpc_h=pos, vel_kms=v_new, ids=pb.ids,
                      boxsize_mpc_h=pb.boxsize_mpc_h, redshift=pb.redshift,
                      particle_mass_msun_h=pb.particle_mass_msun_h)
    del vel
    t1 = time.time()
    cat = run_rockstar_on_particles(pb2, out, cfg=FROZEN_CFG, tag=a.name,
                                    overwrite=a.overwrite)
    snap = out / f"{a.name}.gadget2"
    if snap.is_file():
        snap.unlink()
    summary["catalog"] = str(cat.path)
    summary["rockstar_min"] = (time.time() - t1) / 60.0
    write_json(out / f"{a.name}_oracle.json", summary)
    print(f"Rockstar {summary['rockstar_min']:.1f} min -> {cat.n} objects; "
          f"catalog {cat.path}; total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
