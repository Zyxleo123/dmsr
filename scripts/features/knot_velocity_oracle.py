#!/usr/bin/env python
"""Knot oracle: make every dense outskirt peak kinematically cold, then run the halo finder.

The peak census found that frozen and fine-tuned fields hold HR's number of
dense peaks in host outskirts, with HR's shapes, but only 32-47% of them carry
a Rockstar object (HR 76%); small isolated knots fail most (16-27% vs HR
86-91%). Rescaling the host-scale temperature profile (velocity_profile_oracle)
barely helped. This asks the knot-scale question directly: if each peak's own
particles moved together -- their mean velocity kept, their internal dispersion
shrunk to ``--target-coldbg`` times the dispersion of the annulus around them
(HR's median is 0.36, frozen/fine-tuned 0.64-0.72) -- would they become halos?

    v' = v_core + (v - v_core) * s,   s = min(1, target * sigma_bg / sigma_core)

over the ``--k-core`` nearest particles of each peak at --x-min..--x-max R_vir
(positions untouched; only ever cools). Overlapping knots are resolved in
favour of the denser peak. If conversion then approaches HR's, a boundness term
on the field's own outskirt knots is the lever; if it does not, the deficit is
in density structure velocities cannot fix.

Peaks come from ``host_phase_profile.py --peaks`` run on the SAME field with the
same host selection (the records' ``host`` indexes it).
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
from host_phase_profile import BOX, CellIndex, select_region_hosts  # noqa: E402
from placement_forensics import min_image, read_rockstar  # noqa: E402
from cosmo_sr.eval.particles import ParticleBox, field_to_particles  # noqa: E402

FROZEN_CFG = PROJECT_ROOT / "configs" / "sr2_baseline" / "rockstar.cfg"


def knot_scale(v_core: np.ndarray, v_bg: np.ndarray, target: float) -> float:
    """Shrink factor for a knot's internal dispersion; 1 when already that cold."""
    sc = float(np.sqrt(np.mean(np.sum((v_core - v_core.mean(axis=0)) ** 2, axis=1))))
    sb = float(np.sqrt(np.mean(np.sum((v_bg - v_bg.mean(axis=0)) ** 2, axis=1))))
    if sc <= 0 or sb <= 0:
        return 1.0
    return float(min(1.0, target * sb / sc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--peaks", required=True, help="<field>_peaks.npz from the census")
    ap.add_argument("--hr-catalog", required=True)
    ap.add_argument("--region-export", required=True)
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--min-host-num-p", type=int, default=10000)
    ap.add_argument("--max-hosts", type=int, default=60)
    ap.add_argument("--x-min", type=float, default=0.75)
    ap.add_argument("--x-max", type=float, default=3.0)
    ap.add_argument("--k-core", type=int, default=64)
    ap.add_argument("--target-coldbg", type=float, default=0.3)
    ap.add_argument("--min-ratio", type=float, default=0.0,
                    help="only cool peaks above this contrast (0 = every peak in the file)")
    ap.add_argument("--save-field", default="",
                    help="also write the modified (6,N,N,N) field here (for SUBFIND)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    hr = read_rockstar(a.hr_catalog)
    sel = select_region_hosts(hr, a.region_export, a.dilate, a.min_host_num_p, a.max_hosts)
    rvir = hr["rvir"][sel] / 1000.0
    pk = dict(np.load(a.peaks))
    census = Path(a.peaks.replace("_peaks.npz", ".json"))
    if census.is_file():
        ids = json.loads(census.read_text())["per_host_scalars"]["host_id"]
        if [int(i) for i in hr["id"][sel]] != [int(i) for i in ids]:
            raise SystemExit(f"host list differs from the census that wrote {a.peaks}; "
                             "the peaks' host index would point at the wrong hosts")
    else:
        raise SystemExit(f"no census json next to {a.peaks} to verify the host list")
    use = (pk["x"] >= a.x_min) & (pk["x"] < a.x_max) & (pk["ratio"] > a.min_ratio)
    print(f"{sel.size} hosts; {int(use.sum())} of {use.size} peaks at "
          f"{a.x_min}-{a.x_max} R_vir; k_core {a.k_core}, target coldbg "
          f"{a.target_coldbg}", flush=True)

    field = np.load(a.field, mmap_mode="r")
    pb = field_to_particles(field, boxsize_kpc_h=BOX * 1000.0)
    pos, vel = pb.pos_mpc_h, pb.vel_kms
    ci = CellIndex(pos)
    v_new = vel.copy()
    changed = np.zeros(pos.shape[0], dtype=bool)
    from scipy.spatial import cKDTree
    scales, n_done = [], 0
    for j, h in enumerate(sel):
        mine = np.where(use & (pk["host"] == j))[0]
        if mine.size == 0:
            continue
        rv = float(rvir[j])
        reach = 3.3 * rv + 0.5
        idx = ci.gather(hr["pos"][h], reach)
        d = min_image(pos[idx].astype(np.float64) - hr["pos"][h])
        keep = np.einsum("ij,ij->i", d, d) < reach * reach
        idx, d = idx[keep], d[keep]
        tree = cKDTree(d)
        v_loc = vel[idx].astype(np.float64)
        p_rel = min_image(pk["pos"][mine].astype(np.float64) - hr["pos"][h])
        order = np.argsort(pk["ratio"][mine])          # densest last: it wins overlaps
        for m in order:
            dist, core = tree.query(p_rel[m], k=a.k_core)
            r_in = max(4.0 * float(dist[-1]), 0.1)
            shell = np.asarray(tree.query_ball_point(p_rel[m], r_in + 0.15), dtype=np.int64)
            dd = d[shell] - p_rel[m]
            bg = shell[np.einsum("ij,ij->i", dd, dd) > r_in * r_in]
            if bg.size < 16:
                continue
            s = knot_scale(v_loc[core], v_loc[bg], a.target_coldbg)
            vc = v_loc[core].mean(axis=0)
            gi = idx[core]
            v_new[gi] = (vc + (v_loc[core] - vc) * s).astype(np.float32)
            changed[gi] = True
            scales.append(s)
            n_done += 1
        print(f"  host {j + 1}/{sel.size}: {mine.size} knots, {time.time() - t0:.0f}s",
              flush=True)
    sc = np.asarray(scales)
    summary = {"name": a.name, "field": a.field, "peaks": a.peaks, "k_core": a.k_core,
               "target_coldbg": a.target_coldbg, "x_range": [a.x_min, a.x_max],
               "n_knots_cooled": int(n_done), "n_particles_changed": int(changed.sum()),
               "scale_median": float(np.median(sc)) if sc.size else None,
               "frac_already_cold": float(np.mean(sc >= 1.0)) if sc.size else None}
    print(json.dumps({k: v for k, v in summary.items() if k != "field"}), flush=True)

    out = Path(a.out_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{a.name}_knot_oracle.json", summary)
    if a.save_field:
        # displacement channels copied verbatim, velocity channels re-normalised:
        # the same inverse field_to_particles applies (velocity_profile_oracle)
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
    summary["catalog"] = str(cat.path)
    summary["rockstar_min"] = (time.time() - t1) / 60.0
    write_json(out / f"{a.name}_knot_oracle.json", summary)
    print(f"Rockstar {summary['rockstar_min']:.1f} min -> {cat.n} objects; {cat.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
