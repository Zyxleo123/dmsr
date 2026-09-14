#!/usr/bin/env python
"""Subhalo abundance over EVERY host in the edited region, on every gated box.

`gate_verdict_vs_baseline` reports two things that are not the same claim:

  * a REGION total (`subs_ge_50p` inside the dilated tile boxes), which counts
    substructure that belongs to no host of interest as readily as substructure
    that does, and
  * a mass function within R_vir of **one** host (168880 on set9, 216377 on
    set10), which is a picture, not a population.

The talk needs the population: how many subhalos HR puts inside its hosts, how
many the frozen SR2 baseline puts there, how many the tuned arm puts there --
summed over *all* the hosts of the edited region, on *both* held-out boxes, with
the host count stated so "118%" can never be read as one lucky halo.

DEFINITIONS, all fixed by HR
    hosts       HR top-level halos (`parent_ids < 0`) with `num_p >= 200`
                whose centre is inside the union of the spliced Lagrangian tile
                boxes, dilated by `--dilate` Mpc/h. HR fixes the sample so the
                same apertures are applied to every catalog; an arm cannot
                enlarge its own denominator.
    subhalos    halos of the catalog under test lying within R_vir of that HR
                centre with `num_p >= 50`, minus the single most massive object
                in the sphere (that is the host, not its substructure). This is
                `within_radius_mass_function`'s rule, applied to every host.

Counting inside an HR aperture rather than by Rockstar's parent pointers is
deliberate: `host-damage-is-mostly-reparenting` measured ~88% of the `self`
arm's host loss to be a `parent_id` flip, so a linkage count would attribute an
arm's substructure to a host that Rockstar has re-filed and read as loss where
none happened. The aperture does not move.

Halos outside every host aperture are not counted at all -- the deficit this
work attacks is a subhalo-abundance deficit, and field halos are a different
claim with a different denominator.

Also reported per arm: how many of those HR hosts still have a top-level
`>=200p` halo in the arm's own catalog (nearest-most-massive within
`--match-rvir` R_vir). That is the honest host-count column; it is a survival
rate against HR, not the region total the earlier write-up quoted.

CPU-only and submitted: it parses ~100 MB ASCII catalogs, which the login node
kills, but reads no field and runs no halo finder -- minutes.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from gather_holdout_figures_data import (  # noqa: E402
    BIN_LABELS, BOX, MASS_BINS, catalog_path, counts, in_tile_region,
    load_rockstar_ascii, tile_boxes,
)
from cosmo_sr.eval.rockstar import HaloCatalog  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

# Half-dex host-mass bins, the tail of `analyze_sr2_subhalo_deficit.MASS_EDGES`
# so the occupation table in the talk keeps one binning across the whole story.
# The last edge is open: a 1e15 host must not fall out of the sample silently.
HOST_MASS_EDGES = np.array([11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 15.5])


def _wrap(pos: np.ndarray) -> np.ndarray:
    """Positions folded into ``[0, BOX)`` for a periodic ``cKDTree``.

    ``np.mod`` can return exactly ``BOX`` for a small negative input through
    rounding, which cKDTree rejects, so the upper edge is folded to 0.
    """
    p = np.mod(np.asarray(pos, dtype=np.float64), BOX)
    return np.where(p >= BOX, 0.0, p)


def aperture_counts(cat: HaloCatalog, centres: np.ndarray, radii: np.ndarray,
                    sub_min_p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Subhalos of ``cat`` inside each ``(centre, radius)`` sphere.

    Returns ``(n_sub, hist)``: the count at ``>= sub_min_p`` and the
    ``MASS_BINS`` histogram, both per host. The most massive halo in each sphere
    is dropped as the host itself -- exactly
    ``gather_holdout_figures_data.within_radius_mass_function``, vectorised over
    every host instead of one.
    """
    n_sub = np.zeros(len(centres), dtype=np.int64)
    hist = np.zeros((len(centres), len(MASS_BINS)), dtype=np.int64)
    if cat.n == 0:
        return n_sub, hist
    tree = cKDTree(_wrap(cat.pos), boxsize=BOX)
    for k, idx in enumerate(tree.query_ball_point(_wrap(centres), radii)):
        if not idx:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        npart = cat.num_p[idx]
        # The central is the most massive object in the sphere. With ties
        # argmax takes the first, which drops exactly one row either way.
        keep = np.ones(idx.size, dtype=bool)
        keep[int(np.argmax(npart))] = False
        npart = npart[keep]
        n_sub[k] = int(np.sum(npart >= sub_min_p))
        for j, (lo, hi) in enumerate(MASS_BINS):
            hist[k, j] = int(np.sum((npart >= lo) & (npart < hi)))
    return n_sub, hist


def match_hosts(cat: HaloCatalog, centres: np.ndarray, radii: np.ndarray,
                host_min_p: int) -> np.ndarray:
    """Per HR host: does ``cat`` still resolve a top-level ``>=host_min_p`` halo?

    Nearest-most-massive within ``radii``, the rule `host_forensics` uses: a
    host that fragments leaves one dominant remnant and several small clumps,
    and the remnant is the physical continuation.
    """
    out = np.zeros(len(centres), dtype=bool)
    if cat.n == 0:
        return out
    tree = cKDTree(_wrap(cat.pos), boxsize=BOX)
    for k, idx in enumerate(tree.query_ball_point(_wrap(centres), radii)):
        if not idx:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        row = int(idx[np.argmax(cat.num_p[idx])])
        out[k] = bool(cat.parent_ids[row] < 0 and cat.num_p[row] >= host_min_p)
    return out


def discover_arms(reward_root: Path, box: str) -> List[Tuple[str, str]]:
    """``(name, rockstar tag)`` for every gated candidate catalog of ``box``.

    The name strips the ``mgho_``/``_<box>`` scaffolding the gate adds, so the
    report reads `all_blocks_self`, not the tag.
    """
    out = []
    pat = f"{reward_root}/flow_rockstar/halos/{box}__candidate__*"
    for d in sorted(glob.glob(pat)):
        tag = Path(d).name.split("__candidate__", 1)[1]
        name = tag
        if name.startswith("mgho_"):
            name = name[len("mgho_"):]
        if name.endswith(f"_{box}"):
            name = name[: -len(box) - 1]
        out.append((name, tag))
    return out


def summarise(n_sub: np.ndarray, hist: np.ndarray, host_bin: np.ndarray,
              n_bins: int) -> Dict[str, object]:
    """Totals, per-host mean, and the occupation profile over host mass."""
    per_bin = []
    for b in range(n_bins):
        m = host_bin == b
        per_bin.append({
            "n_hosts": int(np.sum(m)),
            "subs_total": int(np.sum(n_sub[m])),
            "subs_per_host": float(np.mean(n_sub[m])) if np.any(m) else 0.0,
        })
    return {
        "subs_total": int(np.sum(n_sub)),
        "subs_per_host": float(np.mean(n_sub)) if n_sub.size else 0.0,
        "mass_function": {lab: int(hist[:, j].sum())
                          for j, lab in enumerate(BIN_LABELS)},
        "by_host_mass": per_bin,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward-root",
                    default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--run-dir", required=True,
                    help="any gated export dir of this box (export.json gives "
                         "the spliced tile list -- the pool selects the same "
                         "tiles for every arm of a box, so one is enough)")
    ap.add_argument("--arms", default="",
                    help="space-separated name=tag pairs. Empty discovers "
                         "every gated candidate catalog of the box.")
    ap.add_argument("--dilate", type=float, default=2.0,
                    help="tile dilation (Mpc/h); matches the gate's region")
    ap.add_argument("--host-min-p", type=int, default=200)
    ap.add_argument("--sub-min-p", type=int, default=50)
    ap.add_argument("--match-rvir", type=float, default=1.0,
                    help="host-survival search radius in units of R_vir")
    ap.add_argument("--match-floor", type=float, default=0.3,
                    help="minimum host-survival search radius (Mpc/h)")
    ap.add_argument("--max-hosts", type=int, default=0,
                    help="shakeout: keep only the N most massive hosts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    R = Path(a.reward_root)
    export = json.loads((Path(a.run_dir) / "export.json").read_text())
    tiles = ([int(t) for t in export["tiles"]] if "tiles" in export
             else sorted({int(t) for h in export["per_host"] for t in h["tiles"]}))
    boxes = tile_boxes(tiles, TileGrid())

    if a.arms.strip():
        arms = [tuple(s.split("=", 1)) for s in a.arms.split()]
    else:
        arms = discover_arms(R, a.box)
    print(f"box {a.box}: {len(tiles)} spliced tiles, dilate {a.dilate} Mpc/h")
    print(f"arms: {', '.join(n for n, _ in arms) or '(none)'}", flush=True)

    hr = load_rockstar_ascii(catalog_path(R, "hr", a.box))
    base = load_rockstar_ascii(catalog_path(R, "base", a.box))

    # --- the host sample, fixed by HR ---------------------------------------
    sel = (hr.parent_ids < 0) & (hr.num_p >= a.host_min_p) & \
        in_tile_region(hr.pos, boxes, dilate=a.dilate)
    rows = np.where(sel)[0]
    if a.max_hosts > 0 and rows.size > a.max_hosts:
        rows = rows[np.argsort(-hr.num_p[rows])[:a.max_hosts]]
    centres = hr.pos[rows]
    rvir = hr.rvir[rows] / 1000.0            # Rockstar writes kpc/h
    mvir = hr.mvir[rows]
    match_r = np.maximum(a.match_rvir * rvir, a.match_floor)
    logm = np.log10(np.maximum(mvir, 1.0))
    host_bin = np.clip(np.digitize(logm, HOST_MASS_EDGES) - 1,
                       0, len(HOST_MASS_EDGES) - 2)
    n_bins = len(HOST_MASS_EDGES) - 1
    print(f"host sample: {rows.size} HR hosts >= {a.host_min_p}p in the region "
          f"(box-wide HR hosts >= {a.host_min_p}p: "
          f"{int(np.sum((hr.parent_ids < 0) & (hr.num_p >= a.host_min_p)))})",
          flush=True)
    if rows.size == 0:
        print("GATE: no hosts in the region -- nothing to report.")
        write_json(Path(a.out).with_suffix(".json"),
                   {"ok": False, "box": a.box, "reason": "no hosts in region"})
        return 0

    cats: List[Tuple[str, HaloCatalog]] = [("hr", hr), ("base", base)]
    for name, tag in arms:
        try:
            cats.append((name, load_rockstar_ascii(
                catalog_path(R, "cand", a.box, tag))))
        except SystemExit as e:
            print(f"    (skipping {name}: {e})", flush=True)

    per_arm: Dict[str, Dict[str, object]] = {}
    per_host: Dict[str, np.ndarray] = {}
    for name, cat in cats:
        n_sub, hist = aperture_counts(cat, centres, rvir, a.sub_min_p)
        alive = match_hosts(cat, centres, match_r, a.host_min_p)
        s = summarise(n_sub, hist, host_bin, n_bins)
        s["hosts_alive"] = int(np.sum(alive))
        s["hosts_alive_frac"] = float(np.mean(alive))
        s["region_counts"] = counts(cat, in_tile_region(cat.pos, boxes,
                                                        dilate=a.dilate))
        s["box_counts"] = counts(cat)
        per_arm[name] = s
        per_host[name] = n_sub
        print(f"  {name:<28} subs {s['subs_total']:>7}  "
              f"per host {s['subs_per_host']:>6.2f}  "
              f"hosts alive {s['hosts_alive']:>5}/{rows.size}", flush=True)

    hr_tot = max(per_arm["hr"]["subs_total"], 1)
    base_tot = per_arm["base"]["subs_total"]
    for name, s in per_arm.items():
        s["frac_of_hr"] = s["subs_total"] / hr_tot
        s["deficit_recovered"] = ((s["subs_total"] - base_tot)
                                  / max(hr_tot - base_tot, 1))

    print(f"\n=== {a.box}: {rows.size} hosts, subhalos within R_vir "
          f"(>= {a.sub_min_p}p), summed over ALL of them")
    print(f"{'':<28} {'subs':>8} {'/host':>7} {'/HR':>7} {'hosts alive':>12}")
    for name in ["hr", "base"] + [n for n, _ in arms if n in per_arm]:
        s = per_arm[name]
        print(f"{name:<28} {s['subs_total']:>8} {s['subs_per_host']:>7.2f} "
              f"{100 * s['frac_of_hr']:>6.1f}% {s['hosts_alive']:>6}"
              f" ({100 * s['hosts_alive_frac']:>4.1f}%)")

    out = Path(a.out)
    write_json(out.with_suffix(".json"), {
        "ok": True,
        "box": a.box,
        "n_tiles": len(tiles),
        "dilate": a.dilate,
        "host_min_p": a.host_min_p,
        "sub_min_p": a.sub_min_p,
        "match_rvir": a.match_rvir,
        "match_floor": a.match_floor,
        "n_hosts": int(rows.size),
        "n_hosts_box_wide": int(np.sum((hr.parent_ids < 0)
                                       & (hr.num_p >= a.host_min_p))),
        "host_mass_edges_log10": HOST_MASS_EDGES.tolist(),
        "host_mass_mean_log10": [
            float(np.mean(logm[host_bin == b])) if np.any(host_bin == b) else 0.0
            for b in range(n_bins)],
        "mass_bin_labels": BIN_LABELS,
        "arms": per_arm,
    })
    # Per-host rows, so a figure redraws without re-parsing 100 MB of ASCII.
    with open(out.with_suffix(".jsonl"), "w") as fh:
        for k in range(rows.size):
            fh.write(json.dumps({
                "box": a.box,
                "host_id": int(hr.ids[rows[k]]),
                "num_p": int(hr.num_p[rows[k]]),
                "mvir": float(mvir[k]),
                "rvir_mpc": float(rvir[k]),
                "pos": [float(v) for v in centres[k]],
                "host_bin": int(host_bin[k]),
                "subs": {name: int(v[k]) for name, v in per_host.items()},
            }) + "\n")
    print(f"\nwrote {out.with_suffix('.json')}")
    print(f"      {out.with_suffix('.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
