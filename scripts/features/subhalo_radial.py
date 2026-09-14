#!/usr/bin/env python
"""Subhalo mass vs. distance from its host centre, per Rockstar catalog.

For every subhalo (``parent_ids >= 0``) we look up its host row by halo id,
take the minimum-image distance to the host centre, and pair it with the
subhalo's mass. Distances are reported both in Mpc/h and in units of the host
R_vir (the physically meaningful radius -- a 50p clump at 0.2 Mpc means very
different things around a group and around a cluster).

Runs over any number of catalog kinds (``base``, ``hr``, and candidate arm
tags) so the same radial relation can be overlaid across them. CPU-only and
submitted: parsing the ASCII catalogs is real work the login node kills, but
there is no field read and no halo finder -- it is minutes.

Writes, under ``--out-dir``:
  * ``<kind>.npz``      -- per-subhalo arrays (mass, r_mpc, r_over_rvir, ...)
                          for cheap re-plotting.
  * ``<kind>.jsonl``    -- one row per subhalo (all fields).
  * ``summary.json``    -- per-kind counts and a binned median-mass profile.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from gather_holdout_figures_data import (  # noqa: E402
    BOX, catalog_path, in_tile_region, load_rockstar_ascii, tile_boxes,
)
from cosmo_sr.eval.rockstar import HaloCatalog  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402


def _periodic_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum-image displacement ``a - b`` (Mpc/h) in the periodic box."""
    d = a - b
    return d - BOX * np.round(d / BOX)


# radial bins in units of host R_vir, for the median-mass profile
RVIR_BINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])


def collect(cat: HaloCatalog, region_boxes=None, dilate: float = 0.0) -> Dict:
    """Per-subhalo mass and distance-to-host for one catalog.

    ``region_boxes`` (optional) restricts to subhalos whose *host* centre lies
    in the dilated tile region, so an arm is compared to base/HR on the same
    edited patch rather than the whole box.
    """
    id_to_row = {int(i): r for r, i in enumerate(cat.ids)}
    sub_rows = np.where(cat.parent_ids >= 0)[0]

    r_mpc: List[float] = []
    r_over_rvir: List[float] = []
    num_p: List[int] = []
    log_mvir: List[float] = []
    host_num_p: List[int] = []
    host_id: List[int] = []
    n_orphan = 0  # parent id not present in this catalog (should be ~0)

    for s in sub_rows:
        pid = int(cat.parent_ids[s])
        hrow = id_to_row.get(pid, -1)
        if hrow < 0:
            n_orphan += 1
            continue
        if region_boxes is not None and not in_tile_region(
                cat.pos[hrow][None, :], region_boxes, dilate=dilate)[0]:
            continue
        d = _periodic_delta(cat.pos[s], cat.pos[hrow])
        dist = float(np.sqrt(np.dot(d, d)))
        rvir_mpc = float(cat.rvir[hrow]) / 1000.0  # kpc/h -> Mpc/h
        r_mpc.append(dist)
        r_over_rvir.append(dist / rvir_mpc if rvir_mpc > 0 else np.nan)
        num_p.append(int(cat.num_p[s]))
        log_mvir.append(float(np.log10(max(cat.mvir[s], 1.0))))
        host_num_p.append(int(cat.num_p[hrow]))
        host_id.append(pid)

    arrays = {
        "r_mpc": np.asarray(r_mpc, dtype=np.float64),
        "r_over_rvir": np.asarray(r_over_rvir, dtype=np.float64),
        "num_p": np.asarray(num_p, dtype=np.int64),
        "log_mvir": np.asarray(log_mvir, dtype=np.float64),
        "host_num_p": np.asarray(host_num_p, dtype=np.int64),
        "host_id": np.asarray(host_id, dtype=np.int64),
    }
    return {"arrays": arrays, "n_sub": len(num_p), "n_orphan": n_orphan}


def profile(arrays: Dict[str, np.ndarray]) -> Dict:
    """Median (and quartile) subhalo mass in each R/R_vir bin."""
    x = arrays["r_over_rvir"]
    m = arrays["num_p"].astype(np.float64)
    good = np.isfinite(x)
    x, m = x[good], m[good]
    out = {"rvir_bin_edges": RVIR_BINS.tolist(), "bins": []}
    for lo, hi in zip(RVIR_BINS[:-1], RVIR_BINS[1:]):
        sel = (x >= lo) & (x < hi)
        n = int(sel.sum())
        if n == 0:
            out["bins"].append({"lo": float(lo), "hi": float(hi), "n": 0})
            continue
        q = np.percentile(m[sel], [25, 50, 75])
        out["bins"].append({
            "lo": float(lo), "hi": float(hi), "n": n,
            "num_p_p25": float(q[0]), "num_p_median": float(q[1]),
            "num_p_p75": float(q[2]),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward-root",
                    default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--kinds", default="base,hr,self",
                    help="comma list of catalog kinds; 'base','hr', or an arm "
                         "name (mapped to its candidate tag via --arm-tags)")
    ap.add_argument("--arm-tags", default="self=mgho_all_blocks_self_set9",
                    help="comma list of name=tag for candidate arms")
    ap.add_argument("--run-dir", default="",
                    help="optional export.json dir; if set, restrict to its "
                         "tile region (dilated) so arms match base/hr on the "
                         "same patch")
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    R = Path(a.reward_root)
    arm_tags = dict(kv.split("=", 1) for kv in a.arm_tags.split(",") if "=" in kv)

    region_boxes = None
    if a.run_dir:
        export = json.loads((Path(a.run_dir) / "export.json").read_text())
        tiles = ([int(t) for t in export["tiles"]] if "tiles" in export
                 else sorted({int(t) for h in export["per_host"]
                              for t in h["tiles"]}))
        region_boxes = tile_boxes(tiles, TileGrid())
        print(f"restricting to {len(tiles)} tiles (dilate {a.dilate} Mpc/h)",
              flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_kind: Dict[str, Dict] = {}
    for kind in [k.strip() for k in a.kinds.split(",") if k.strip()]:
        if kind in ("base", "hr"):
            path = catalog_path(R, kind, a.box)
        else:
            if kind not in arm_tags:
                raise SystemExit(f"no tag for arm '{kind}' in --arm-tags")
            path = catalog_path(R, "cand", a.box, arm_tags[kind])
        cat = load_rockstar_ascii(path)
        res = collect(cat, region_boxes=region_boxes, dilate=a.dilate)
        arrays = res["arrays"]
        np.savez_compressed(out_dir / f"{kind}.npz", **arrays)
        with open(out_dir / f"{kind}.jsonl", "w") as fh:
            for i in range(res["n_sub"]):
                fh.write(json.dumps({
                    "r_mpc": float(arrays["r_mpc"][i]),
                    "r_over_rvir": float(arrays["r_over_rvir"][i]),
                    "num_p": int(arrays["num_p"][i]),
                    "log_mvir": float(arrays["log_mvir"][i]),
                    "host_num_p": int(arrays["host_num_p"][i]),
                    "host_id": int(arrays["host_id"][i]),
                }) + "\n")
        per_kind[kind] = {
            "n_halos": cat.n, "n_sub": res["n_sub"],
            "n_orphan": res["n_orphan"],
            "median_r_over_rvir": float(np.nanmedian(arrays["r_over_rvir"]))
            if res["n_sub"] else 0.0,
            "median_num_p": float(np.median(arrays["num_p"]))
            if res["n_sub"] else 0.0,
            "profile": profile(arrays),
        }
        print(f"  {kind:<10} halos {cat.n:>7}  subhalos {res['n_sub']:>7}  "
              f"median r/Rvir {per_kind[kind]['median_r_over_rvir']:.2f}  "
              f"median num_p {per_kind[kind]['median_num_p']:.0f}", flush=True)

    summary = {
        "ok": True, "box": a.box, "dilate": a.dilate,
        "region_restricted": bool(a.run_dir),
        "kinds": list(per_kind.keys()), "per_kind": per_kind,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"\nwrote {out_dir}/<kind>.npz, <kind>.jsonl and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
