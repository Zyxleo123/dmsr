#!/usr/bin/env python
"""Per-tile subhalo abundance for SR2 vs HR, as a small redraw-ready JSON.

The host features say where structure *should* form. This says where SR2
actually put it. Both are indexed by the same 512 Lagrangian tiles, so the two
can be read side by side without any matching.

Counts come from the existing ``*_tilew.npz`` weights (a few MB), not from the
537 MB owner arrays and not from a new Rockstar run: a subhalo is attributed
fractionally over the tiles its member particles came from, so
``sum_j w[s, j] == 1`` and a tile's abundance is ``sum_s w[s, j]`` over subhalos.

One trap this script exists to avoid: ``stream_particle_tile_counts`` groups by
``external_haloid`` and keeps **every** recursion row, so a particle appears once
per ancestor and those weights are not a partition of the particles. Summing
them over all objects over-counts (measured: 4.6x the box). Subhalo *counts* are
safe because each subhalo contributes its own weight row once; the occupancy
figures below are therefore summed over **top-level hosts only**, whose recursive
counts cover each bound particle exactly once.

    python scripts/features/collect_tile_abundance.py --boxes set8
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (  # noqa: E402
    DEFAULT_CONFIG, banner, load_reward_config, paths, write_json,
)

from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402


def _catalog(work: Path, tag: str):
    hits = sorted(glob.glob(str(work / f"{tag}_rockstar" / "halos*.ascii"))
                  + glob.glob(str(work / f"{tag}_rockstar" / "halos*.list")))
    if not hits:
        raise SystemExit(f"no Rockstar catalog under {work / (tag + '_rockstar')}")
    return load_rockstar_ascii(hits[0])


def _weights(work: Path, box: str, tag: str):
    p = work / f"{box}_{tag}_tilew.npz"
    if not p.is_file():
        raise SystemExit(
            f"no tile weights at {p}; run scripts/reward/rockstar_particles.py "
            f"--box {box} --source {'hr' if tag == 'hr' else 'base'} first")
    return np.load(p)


def one_source(root: Path, box: str, tag: str, n_tiles: int, min_sub_p: int):
    work = root / f"{box}__{tag}__{tag}"
    z = _weights(work, box, tag)
    cat = _catalog(work, tag)
    num_p = dict(zip(cat.ids.tolist(), cat.num_p.tolist()))
    parent = dict(zip(cat.ids.tolist(), cat.parent_ids.tolist()))
    hid = z["halo_id"]
    tid = z["tile_id"]
    w = z["weight"]

    is_sub = np.array([parent.get(int(h), -1) >= 0
                       and num_p.get(int(h), 0) >= min_sub_p for h in hid])
    n_sub = np.bincount(tid[is_sub], weights=w[is_sub], minlength=n_tiles)

    # Occupancy: top-level hosts only (see the module docstring).
    counts = dict(zip(z["member_halo_id"].tolist(), z["member_count"].tolist()))
    is_root = np.array([parent.get(int(h), -1) < 0 for h in hid])
    rec = np.array([counts.get(int(h), 0) for h in hid], dtype=np.float64)
    bound = np.bincount(tid[is_root], weights=(w * rec)[is_root], minlength=n_tiles)
    return n_sub, bound, cat


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    ap.add_argument("--min-sub-particles", type=int, default=0,
                    help="ignore subhalos below this particle count (default 0)")
    ap.add_argument("--hr-tag", default="hr")
    ap.add_argument("--sr2-tag", default="base")
    args = ap.parse_args(argv)
    cfg = load_reward_config(args)

    d = cfg.get("data", {})
    ng_hr = int(d.get("ng_hr", 512))
    tile_hr = int(cfg.get("tiles", {}).get("tile_hr", 64))
    n_tiles = (ng_hr // tile_hr) ** 3
    per_tile = tile_hr ** 3
    root = paths.subdir("halos_particles")

    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        banner(f"tile abundance {box}")
        hr_sub, hr_bound, hr_cat = one_source(
            root, box, args.hr_tag, n_tiles, args.min_sub_particles)
        sr_sub, sr_bound, sr_cat = one_source(
            root, box, args.sr2_tag, n_tiles, args.min_sub_particles)

        ok = hr_sub >= 1.0
        rel = np.where(ok, (sr_sub - hr_sub) / np.maximum(hr_sub, 1e-9), np.nan)
        occ_hr, occ_sr = hr_bound / per_tile, sr_bound / per_tile

        out = {
            "box": box, "n_tiles": n_tiles, "particles_per_tile": per_tile,
            "min_sub_particles": int(args.min_sub_particles),
            "n_sub_hr": [float(v) for v in hr_sub],
            "n_sub_sr2": [float(v) for v in sr_sub],
            "rel_deficit": [None if not o else float(r) for o, r in zip(ok, rel)],
            "occupancy_hr": [float(v) for v in occ_hr],
            "occupancy_sr2": [float(v) for v in occ_sr],
            "totals": {
                "hr_subhalos": float(hr_sub.sum()),
                "sr2_subhalos": float(sr_sub.sum()),
                "ratio": float(sr_sub.sum() / max(hr_sub.sum(), 1e-9)),
                "hr_occupancy": float(hr_bound.sum() / (ng_hr ** 3)),
                "sr2_occupancy": float(sr_bound.sum() / (ng_hr ** 3)),
                "tiles_short": int(np.count_nonzero(rel[ok] < 0)),
                "tiles_scored": int(ok.sum()),
                "median_rel_deficit": float(np.median(rel[ok])),
                "p10_rel_deficit": float(np.quantile(rel[ok], 0.1)),
                "p90_rel_deficit": float(np.quantile(rel[ok], 0.9)),
                "worst_rel_deficit": float(np.min(rel[ok])),
            },
        }
        dest = paths.subdir("lagrangian_host", box, create=True) \
            / f"{box}_tile_abundance.json"
        write_json(dest, out)
        t = out["totals"]
        print(f"    HR  {t['hr_subhalos']:.0f} subhalos, occupancy {t['hr_occupancy']:.4f}")
        print(f"    SR2 {t['sr2_subhalos']:.0f} subhalos, occupancy {t['sr2_occupancy']:.4f}"
              f"  -> ratio {t['ratio']:.3f}")
        print(f"    relative deficit: median {t['median_rel_deficit']:+.3f}  "
              f"p10 {t['p10_rel_deficit']:+.3f}  p90 {t['p90_rel_deficit']:+.3f}  "
              f"worst {t['worst_rel_deficit']:+.3f}")
        print(f"    tiles short of HR: {t['tiles_short']}/{t['tiles_scored']}")
        print(f"    wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
