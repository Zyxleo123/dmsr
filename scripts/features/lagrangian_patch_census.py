#!/usr/bin/env python
"""How much Lagrangian mass has collapsed into compact objects, by host-centric radius?

A halo is a Lagrangian patch mapped onto a small Eulerian volume. On the shared
512^3 lattice, cut the particles into s^3 blocks of lattice neighbours and
measure each block's Eulerian rms size against its Lagrangian one:

    C = R_E / R_L,   R_L = cell * sqrt(3 (s^2 - 1) / 12)

C ~ 1 is a block that only moved, C << 1 one that collapsed into something
compact. The fraction of blocks with C below a threshold, per r/R_vir bin of
the block's Eulerian centroid about the nearest massive host, is a finder-free,
address-free census of collapsed objects of ~s^3 particles -- and it is a
function of the displacement field alone, so it is differentiable as it
stands (a candidate objective, which is why it is measured here first).

Hosts and R_vir come from the HR catalog; each field's host centre is re-found
in that field by shrinking spheres (as in host_phase_profile.py), so the SR2
frame offset does not enter. Writes ``<out>.json`` and ``<out>.npz``.
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
from gather_holdout_figures_data import in_tile_region, tile_boxes  # noqa: E402
from host_phase_profile import BOX, EDGES_X, CellIndex, shrinking_centre  # noqa: E402
from placement_forensics import min_image, read_rockstar  # noqa: E402
from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

C_THRESH = (0.1, 0.2, 0.3, 0.5)


def block_sizes(pos: np.ndarray, ng: int, s: int, box: float = BOX):
    """Eulerian centroid and rms size of every s^3 block of lattice neighbours.

    ``pos`` is (ng^3, 3) in flat lattice order. Each block is unwrapped about its
    first particle, so a block straddling the box edge is measured whole.
    """
    p = pos.reshape(ng // s, s, ng // s, s, ng // s, s, 3)
    p = p.transpose(0, 2, 4, 1, 3, 5, 6).reshape(-1, s ** 3, 3).astype(np.float64)
    rel = p - p[:, :1, :]
    rel -= box * np.round(rel / box)
    cen = rel.mean(axis=1)
    dev = rel - cen[:, None, :]
    size = np.sqrt(np.mean(np.einsum("pij,pij->pi", dev, dev), axis=1))
    centroid = np.mod(p[:, 0, :] + cen, box)
    return centroid, size


def lagrangian_rms(s: int, cell: float) -> float:
    return cell * np.sqrt(3.0 * (s * s - 1) / 12.0)


def nearest_host_x(cen: np.ndarray, centres: np.ndarray, rvir: np.ndarray,
                   chunk: int = 4_000_000, box: float = BOX) -> np.ndarray:
    """min over hosts of |cen - centre_j| / R_vir,j (periodic), vectorised per host.

    The nearest host in units of ITS OWN R_vir, so a block 2 Mpc/h from a cluster
    and 0.5 Mpc/h from a group is filed under whichever it sits deeper inside.
    """
    x = np.full(cen.shape[0], np.inf)
    for s in range(0, cen.shape[0], chunk):
        c = cen[s:s + chunk]
        xs = x[s:s + chunk]
        for cj, rj in zip(centres, rvir):
            dd = c - cj
            dd -= box * np.round(dd / box)
            np.minimum(xs, np.sqrt(np.einsum("ij,ij->i", dd, dd)) / rj, out=xs)
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hr-catalog", required=True)
    ap.add_argument("--region-export", default="")
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--min-host-num-p", type=int, default=10000)
    ap.add_argument("--max-hosts", type=int, default=60)
    ap.add_argument("--sizes", default="2,4,8")
    ap.add_argument("--reach", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    t0 = time.time()
    hr = read_rockstar(a.hr_catalog)
    if a.region_export.strip():
        ex = json.loads(Path(a.region_export).read_text())
        tiles = [int(t) for t in ex["tiles"]] if "tiles" in ex else sorted(
            {int(t) for h in ex["per_host"] for t in h["tiles"]})
        inreg = in_tile_region(hr["pos"], tile_boxes(tiles, TileGrid()), dilate=a.dilate)
    else:
        inreg = np.ones(hr["id"].size, dtype=bool)
    sel = np.where((hr["parent_id"] < 0) & (hr["num_p"] >= a.min_host_num_p) & inreg)[0]
    sel = sel[np.argsort(-hr["num_p"][sel])][: a.max_hosts]
    rvir = hr["rvir"][sel] / 1000.0

    field = np.load(a.field, mmap_mode="r")
    pb = field_to_particles(field, boxsize_kpc_h=BOX * 1000.0)
    pos = pb.pos_mpc_h
    del field
    ng = int(round(pos.shape[0] ** (1 / 3)))
    cell = BOX / ng
    ci = CellIndex(pos)
    centres = []
    for j, h in enumerate(sel):
        rv = float(rvir[j])
        idx = ci.gather(hr["pos"][h], 0.6 * rv + 0.3)
        d = min_image(pos[idx].astype(np.float64) - hr["pos"][h])
        c = shrinking_centre(d, 0.5 * rv, 0.05 * rv)
        centres.append(np.mod(hr["pos"][h] + c, BOX))
    centres = np.asarray(centres)
    print(f"{sel.size} hosts, centres re-found in {time.time() - t0:.0f}s", flush=True)

    nb = EDGES_X.size - 1
    res = {"name": a.name, "field": a.field, "n_hosts": int(sel.size),
           "edges_x": EDGES_X.tolist(), "c_thresh": list(C_THRESH), "sizes": {}}
    arrays = {}
    for s in [int(t) for t in a.sizes.split(",")]:
        cen, size = block_sizes(pos, ng, s)
        C = size / lagrangian_rms(s, cell)
        x = nearest_host_x(cen, centres, rvir)
        b = np.digitize(x, EDGES_X) - 1
        frac = np.full((nb, len(C_THRESH)), np.nan)
        med = np.full(nb, np.nan)
        cnt = np.zeros(nb, dtype=np.int64)
        for i in range(nb):
            m = b == i
            cnt[i] = int(m.sum())
            if cnt[i]:
                med[i] = float(np.median(C[m]))
                for t, th in enumerate(C_THRESH):
                    frac[i, t] = float(np.mean(C[m] < th))
        res["sizes"][str(s)] = {"n_blocks": cnt.tolist(), "median_C": med.tolist(),
                                "frac_below": frac.tolist(),
                                "global_frac_below": [float(np.mean(C < th)) for th in C_THRESH]}
        arrays[f"s{s}_C_hist"] = np.histogram(np.log10(np.maximum(C, 1e-4)), bins=80,
                                              range=(-4, 1))[0]
        print(f"  s={s}: {C.size} blocks, global frac C<0.3 {np.mean(C < 0.3):.4f}; "
              f"{time.time() - t0:.0f}s", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), **arrays)
    write_json(out.with_suffix(".json"), res)
    print(f"wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
