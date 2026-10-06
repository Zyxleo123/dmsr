#!/usr/bin/env python
"""Why does Rockstar report the inner supervised sets and not the outskirt ones?

CPU, minutes. Reads only artefacts already on disk -- the HR catalog, each
arm's ``holdout_<box>/subhalos.json``, its ``flow_rockstar/compare/*.json`` and
its ``pool.json`` -- and writes the numbers a figure can be redrawn from.

Read the output in this order
-----------------------------
1. **The radial profile.** Hit rate per host-centric radial bin, with
   ``median_num_p`` beside it. The gap is only a radial result if the mass is
   flat across the bins; if it is not, stop here.
2. **The address decomposition.** ``P(addressed)`` -- the gathered members'
   centroid landing within a tolerance of the HR subhalo -- and
   ``P(hit | addressed)``. These multiply to the rate in (1), and which of the
   two carries the radial trend is what decides the next experiment: a
   placement failure and a construction failure need different fixes.
3. **The hit/miss contrast.** Which per-set statistic actually separates a
   recovered target from a lost one. ``bound_hard`` is the term the loss
   charges for and the gate is named after; whether it is the discriminator is
   a measurement, not an assumption.
4. **The reference inflation.** How far the *reachable* reference the loss
   optimises against sits from HR's own object, and whether the gap tracks the
   out-of-tile straggler fraction.

Radii are signed against each set's **own parent**, never the task host
(``host-profile-gate-passed``: a trained tile is 12.5 Mpc/h a side, and signing
every set against the task host inverted that gate's verdict).

    python scripts/features/outskirt_detection_probe.py \\
        --box set9 --arms all_blocks_self all_blocks_sm_coarse_soft
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reward"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from _common import banner, write_json  # noqa: E402

from cosmo_sr.features.outskirt_detection import (  # noqa: E402
    ADDRESS_TOLERANCES, RADIAL_BINS, address_decomposition, build_set_table,
    hit_miss_contrast, radial_profile, reference_inflation)

BOXSIZE = 100.0


# --------------------------------------------------------------------------- #
# the HR catalog, cached
# --------------------------------------------------------------------------- #
def load_hr_catalog(reward_root: Path, box: str, cache_dir: Path) -> Dict[str, np.ndarray]:
    """``id / num_p / rvir / pos / parent_id`` for the HR catalog of ``box``.

    Cached as an npz because the ASCII is ~160 MB and every arm needs the same
    join; the cache is keyed on the catalog's mtime so a re-run of Rockstar
    invalidates it rather than being silently read from a stale file.
    """
    hits = sorted(glob.glob(str(reward_root / "halos" / f"{box}__hr__hr"
                                / "*_rockstar" / "halos_*.ascii")))
    if not hits:
        raise SystemExit(f"no HR catalog for {box} under {reward_root}/halos")
    src = Path(hits[0])
    cache = cache_dir / f"{box}_hr_cols.npz"
    if cache.is_file() and cache.stat().st_mtime >= src.stat().st_mtime:
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    from cosmo_sr.eval.rockstar import load_rockstar_ascii
    t0 = time.time()
    cat = load_rockstar_ascii(src)
    out = {"ids": cat.ids, "num_p": cat.num_p, "rvir_kpc_h": cat.rvir,
           "pos": cat.pos, "parent_ids": cat.parent_ids}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    print(f"  parsed {src.name}: {cat.n} halos [{time.time() - t0:.0f}s] "
          f"-> {cache}", flush=True)
    return out


def _compare_rows(reward_root: Path, box: str, tag: str) -> Optional[List[Dict]]:
    p = reward_root / "flow_rockstar" / "compare" / f"{box}__{tag}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())["supervised_targets"]["cand"]["rows"]


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def _f(x: float, w: int = 6, p: int = 3) -> str:
    return " " * (w - 3) + "n/a" if not np.isfinite(x) else f"{x:{w}.{p}f}"


def print_radial(prof: List[Dict]) -> None:
    print("  radial profile -- hit rate and the object, by host-centric radius")
    print("    r/Rvir       n   hit%   num_p  bound_hard  2T/|W|  r_rms/Rvir_HR"
          "  sigma_v  |off|Mpc/h  live  | miss: nearest  %any")
    for r in prof:
        lab = "   ALL   " if r["all_radii"] else f"{r['r_lo']:4.2f}-{min(r['r_hi'], 9.99):4.2f}"
        print(f"    {lab} {r['n']:5d}  {100 * r['hit_rate']:5.1f} "
              f"{r['median_num_p']:7.0f}    {_f(r['median_bound_hard'], 6)}  "
              f"{_f(r['median_virial'], 6, 1)}     {_f(r['median_r_rms_over_rvir_hr'], 6, 2)}   "
              f"{_f(r['median_sigma_v'], 6, 0)}     {_f(r['median_offset_mpc_h'], 6)}  "
              f"{_f(r['median_live_frac'], 5, 3)} |  {_f(r['miss_nearest_mpc_h'], 6)}  "
              f"{100 * r['miss_frac_something_in_radius']:5.1f}")


def print_address(dec: List[Dict]) -> None:
    if not dec:
        print("  address decomposition: UNAVAILABLE -- this export carried no "
              "`centre_offset_radii`.\n    Re-export the arm to get it; the "
              "radial hit rate above cannot be split without it.")
        return
    print("\n  address decomposition -- P(addressed) x P(hit | addressed)")
    for tol in sorted({r["tol_mpc_h"] for r in dec}):
        print(f"    -- |centroid - HR centre| < {tol:.2f} Mpc/h"
              f"{'   (the gate own search radius)' if abs(tol - 0.15) < 1e-9 else ''}")
        print("       r/Rvir       n   P(addr)   hit%|addr    n     hit%|NOT addr    n")
        for r in [x for x in dec if x["tol_mpc_h"] == tol]:
            lab = "   ALL   " if r["all_radii"] else f"{r['r_lo']:4.2f}-{min(r['r_hi'], 9.99):4.2f}"
            ga, gu = r["hit_rate_given_addressed"], r["hit_rate_given_unaddressed"]
            print(f"       {lab} {r['n']:5d}   {100 * r['p_addressed']:5.1f}%   "
                  f"{('%6.1f%%' % (100 * ga)) if np.isfinite(ga) else '   n/a '}  "
                  f"{r['n_addressed']:5d}      "
                  f"{('%6.1f%%' % (100 * gu)) if np.isfinite(gu) else '   n/a '}  "
                  f"{r['n_unaddressed']:5d}")


def print_contrast(con: Dict) -> None:
    print("\n  what separates a recovered target from a lost one (pooled)")
    print("             n   r/Rvir  |off|Mpc/h  r_rms/Rvir_HR  bound_hard  2T/|W|"
          "  sigma_v  num_p")
    for name in ("hit", "miss"):
        c = con[name]
        print(f"    {name:<5} {c['n']:5d}  {_f(c['median_r_over_rvir'], 6, 2)}  "
              f"{_f(c['median_offset_mpc_h'], 8)}   {_f(c['median_r_rms_over_rvir_hr'], 10, 2)}  "
              f"{_f(c['median_bound_hard'], 10)}  {_f(c['median_virial'], 6, 1)}  "
              f"{_f(c['median_sigma_v'], 7, 0)}  {c['median_num_p']:5.0f}")


def print_reference(ref: Dict) -> None:
    q = ref.get("quantities") or {}
    if not q:
        print("\n  reference inflation: no pool.json for this arm")
        return
    print(f"\n  the reference the loss optimises against, vs HR's own object "
          f"({ref['n_hosts']} {ref['split'].split('_')[0]} hosts, "
          f"median live_frac {ref.get('median_live_frac', float('nan')):.4f})")
    print(f"    policy in use: stats_members={ref.get('stats_members', '?')}"
          f"   (comparing {ref.get('compared_against', 'reference_median')})")
    print("    quantity        pure HR    hybrid ref   hybrid/HR")
    for k in ("r_rms", "sigma_v", "virial", "bound_hard"):
        if k not in q:
            continue
        v = q[k]
        print(f"    {k:<14} {_f(v['pure_hr'], 8, 3)}   {_f(v['reference'], 11, 3)}"
              f"   {_f(v['ratio'], 6, 2)}")
    c = ref.get("corr_straggler_frac_vs_r_rms_inflation", float("nan"))
    print(f"    corr(1 - live_frac, r_rms_ref / r_rms_HR) = {_f(c, 5, 2)}"
          "   <- near 1 means the inflation IS the straggler tail")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reward-root",
                    default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="member_gather run directory names, e.g. "
                         "all_blocks_sm_coarse_soft")
    ap.add_argument("--tag-format", default="mgho_{arm}_{box}",
                    help="how an arm maps to its compare/ tag")
    ap.add_argument("--out", default="",
                    help="output dir (default <reward_root>/outskirt_probe)")
    args = ap.parse_args(argv)

    R = Path(args.reward_root)
    MG = R / "member_gather"
    out_dir = Path(args.out) if args.out else (R / "outskirt_probe" / args.box)
    out_dir.mkdir(parents=True, exist_ok=True)

    banner(f"outskirt detection probe -- box {args.box}, {len(args.arms)} arm(s)")
    cat = load_hr_catalog(R, args.box, out_dir.parent / "cache")

    summary: Dict[str, Dict] = {"box": args.box, "arms": {}}
    for arm in args.arms:
        run = MG / arm
        subs = run / f"holdout_{args.box}" / "subhalos.json"
        tag = args.tag_format.format(arm=arm, box=args.box)
        if not subs.is_file():
            print(f"\n### {arm}: no {subs}, skipped", flush=True)
            continue
        crows = _compare_rows(R, args.box, tag)
        if crows is None:
            print(f"\n### {arm}: no compare/{args.box}__{tag}.json, skipped",
                  flush=True)
            continue

        rows = json.loads(subs.read_text())["rows"]
        t = build_set_table(rows, crows, boxsize=BOXSIZE, **cat)
        prof = radial_profile(t)
        dec = address_decomposition(t)
        con = hit_miss_contrast(t)
        pool_p = run / "pool.json"
        ref = (reference_inflation(json.loads(pool_p.read_text()))
               if pool_p.is_file() else {})

        print(f"\n########## {arm}   (tag {tag}, searched at {t.searched_at})")
        print(f"  {t.n} supervised sets joined, {t.n_unresolved} unresolved"
              + (f"; rows missing {t.missing}" if t.missing else ""))
        if t.searched_at == "relocated":
            print("  NOTE: this arm was scored where it ASKED for the object, "
                  "not at HR's centre.\n        Its rates are not comparable "
                  "with an arm-A row without saying so.")
        print_radial(prof)
        print_address(dec)
        print_contrast(con)
        print_reference(ref)

        rec = {"arm": arm, "tag": tag, "searched_at": t.searched_at,
               "n_sets": t.n, "n_unresolved": t.n_unresolved,
               "missing_row_fields": t.missing,
               "radial_profile": prof, "address_decomposition": dec,
               "hit_miss_contrast": con, "reference_inflation": ref,
               "radial_bins": [list(b) for b in RADIAL_BINS],
               "address_tolerances": list(ADDRESS_TOLERANCES)}
        write_json(out_dir / f"{arm}.json", rec)
        summary["arms"][arm] = {
            "hit_rate": prof[-1]["hit_rate"],
            "hit_rate_inner": prof[0]["hit_rate"],
            "hit_rate_1_to_1p5": next(
                (r["hit_rate"] for r in prof
                 if r["r_lo"] == 1.0 and not r["all_radii"]), float("nan")),
            "searched_at": t.searched_at,
        }

    write_json(out_dir / "summary.json", summary)
    print(f"\n=== wrote {out_dir}/<arm>.json and summary.json")
    print("=== redraw the figures with scripts/features/outskirt_detection_plot.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
