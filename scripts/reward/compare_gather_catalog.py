#!/usr/bin/env python
"""Did the gathered clumps survive a real halo finder? CPU, minutes.

Reads three Rockstar catalogs of the same box -- HR (truth), ``base`` (frozen
SR2) and the spliced candidate -- and asks the one question no field statistic
can answer: **is the material the gather loss concentrated actually bound?**
Rockstar links particles in 6-D, so an overdensity that is too hot or too
diffuse is simply never reported, however good its density profile looks.

The measurement, in the order it should be read
-----------------------------------------------
1. **Subhalos inside the trained host.** The headline. Counted inside the host's
   own R_vir in each catalog, binned by particle count, against the HR truth and
   the frozen baseline. ``docs/sr2_substructure_module.md`` section 9 step 6 puts
   the target at 0.07 -> 0.4+ of HR.
2. **The host itself.** Its mass and position must not move: a "gain" bought by
   turning one cluster into three is not a gain. Reported as the change in
   ``M_vir`` and the centre offset in Mpc/h.
3. **Distance profile of the change.** New and lost halos as a function of
   distance from the host centre. The splice has hard tile faces where two
   different generators meet, so an artifact appears here as a shell of new
   objects far from the host rather than as substructure near it. Read this
   before believing item 1.
4. **The whole box.** Host counts above 200 particles, which the gate requires to
   be unchanged. Only a small fraction of the box is spliced -- 0.78% at four
   tiles, 3.1% at sixteen -- so a large change here is a red flag about the
   splice, not a result about the objective. The fraction is read from the
   splice's own metadata rather than assumed, because the tile count is a knob
   (``docs/sr2_member_gather.md`` section 6.2).

What it cannot say
------------------
The spliced box keeps the frozen field outside the four trained tiles, so the
collateral damage the fine-tune does everywhere else is invisible here by
construction. That needs a whole-box regeneration.

    python scripts/reward/compare_gather_catalog.py --tag gather_set8_h271800_fine_anchored
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, write_json  # noqa: E402

from cosmo_sr.eval.rockstar import HaloCatalog, load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward import paths  # noqa: E402

BOXSIZE = 100.0
PBINS = [(50, 100), (100, 200), (200, 500), (500, 2000), (2000, 10 ** 9)]


def _load(pattern: str) -> Optional[HaloCatalog]:
    hits = sorted(glob.glob(pattern))
    for h in hits:
        cat = load_rockstar_ascii(h)
        if cat.n:
            return cat
    return None


def find_catalogs(box: str, tag: str) -> Dict[str, HaloCatalog]:
    root = paths.reward_root()
    out: Dict[str, HaloCatalog] = {}
    for name, pat in (
        ("hr", str(root / "halos" / f"{box}__hr__hr" / "*_rockstar" / "halos_*.ascii")),
        ("base", str(root / "halos" / f"{box}__base__base" / "*_rockstar" / "halos_*.ascii")),
        ("cand", str(root / "flow_rockstar" / "halos" / f"{box}__candidate__{tag}"
                     / "**" / "halos_*.ascii")),
    ):
        cat = _load(pat)
        if cat is None and name == "cand":
            cat = _load(str(root / "flow_rockstar" / "halos"
                            / f"{box}__candidate__{tag}" / "*" / "halos_*.ascii"))
        if cat is not None:
            out[name] = cat
    return out


def spliced_fraction(box: str, tag: str) -> Optional[float]:
    """How much of the box this candidate replaced, from the splice's own meta.

    ``splice_gather_field.py`` writes it beside the field it wrote. Recomputing
    it here would need the tile list, which this script does not otherwise read.
    """
    hits = sorted(glob.glob(str(paths.reward_root() / "flow_rockstar" / "fields"
                                / f"{box}__{tag}__seed*.json")))
    for h in hits:
        try:
            v = json.loads(Path(h).read_text()).get("spliced_volume_fraction")
        except (OSError, ValueError):
            continue
        if v is not None:
            return float(v)
    return None


def periodic_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(a) - np.asarray(b)
    return (d + 0.5 * BOXSIZE) % BOXSIZE - 0.5 * BOXSIZE


def match_host(cat: HaloCatalog, centre: np.ndarray, mvir: float,
               max_sep: float = 1.0) -> Optional[int]:
    """The most massive host within ``max_sep`` Mpc/h of ``centre``.

    Position, not id: the candidate catalog is a fresh Rockstar run and its ids
    have nothing to do with HR's.
    """
    hosts = np.flatnonzero(cat.parent_ids < 0)
    if hosts.size == 0:
        return None
    d = np.linalg.norm(periodic_delta(cat.pos[hosts], centre[None, :]), axis=1)
    near = hosts[d <= float(max_sep)]
    if near.size == 0:
        return None
    return int(near[np.argmax(cat.mvir[near])])


def subhalos_within(cat: HaloCatalog, centre: np.ndarray, radius_mpc: float,
                    min_p: int = 50) -> np.ndarray:
    """Rows of subhalos inside a sphere. Catalog-independent of the host tree.

    Counting by ``parent_id`` would compare three different halo trees; a sphere
    of a fixed physical radius asks the same question of each catalog.
    """
    sub = np.flatnonzero((cat.parent_ids >= 0) & (cat.num_p >= int(min_p)))
    if sub.size == 0:
        return sub
    d = np.linalg.norm(periodic_delta(cat.pos[sub], centre[None, :]), axis=1)
    return sub[d <= float(radius_mpc)]


def count_by_bin(cat: HaloCatalog, rows: np.ndarray) -> Dict[str, int]:
    out = {}
    for lo, hi in PBINS:
        m = (cat.num_p[rows] >= lo) & (cat.num_p[rows] < hi)
        out[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}p"] = int(m.sum())
    out["total"] = int(rows.size)
    return out


def shell_profile(a: HaloCatalog, b: HaloCatalog, centre: np.ndarray,
                  edges: List[float], min_p: int = 50) -> List[Dict]:
    """Halo counts of two catalogs in radial shells about the host.

    The edge-artifact check: the fine-tune's effect should be concentrated where
    the cluster is, and a splice-boundary artifact appears as objects appearing
    at a radius set by the tile faces instead.
    """
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        cnt = {}
        for name, cat in (("base", a), ("cand", b)):
            sel = np.flatnonzero(cat.num_p >= int(min_p))
            d = np.linalg.norm(periodic_delta(cat.pos[sel], centre[None, :]), axis=1)
            cnt[name] = int(((d >= lo) & (d < hi)).sum())
        rows.append({"r_lo": lo, "r_hi": hi, **cnt,
                     "delta": cnt["cand"] - cnt["base"]})
    return rows


def target_hit_rate(cat: HaloCatalog, hr: HaloCatalog, halo_ids: List[int],
                    *, radius_factor: float = 1.0, min_radius: float = 0.15,
                    mass_frac: float = 0.25) -> Dict:
    """Did each SUPERVISED subhalo come back as a bound halo?

    The per-R_vir count answers a different question than this experiment asked.
    Targets are selected by *Lagrangian* home tile -- every subhalo whose material
    originates in one of the trained tiles -- while the R_vir count is an
    *Eulerian* sphere around the host. The two populations only partly overlap,
    so the sphere under-credits the run: it contains supervised targets and
    unsupervised ones alike, and leaves out supervised targets that live outside
    it. This asks the direct question instead, one target at a time.

    A target counts as recovered if the catalog holds any halo within
    ``max(radius_factor * r_vir, min_radius)`` Mpc/h of the HR subhalo's centre
    carrying at least ``mass_frac`` of its particles -- position and mass, so a
    passing wisp near the right place does not count.
    """
    rows, hits = [], {"base": 0}
    by_id = {int(h): i for i, h in enumerate(hr.ids)}
    for hid in halo_ids:
        i = by_id.get(int(hid))
        if i is None:
            continue
        r = max(float(radius_factor) * float(hr.rvir[i]) / 1000.0, float(min_radius))
        d = np.linalg.norm(periodic_delta(cat.pos, hr.pos[i][None, :]), axis=1)
        near = np.flatnonzero((d <= r) & (cat.num_p >= mass_frac * hr.num_p[i]))
        # A miss has two very different causes and the hit flag cannot tell them
        # apart: nothing is THERE, or something is there and is too light. Record
        # both so a failed run can be read without re-running the comparison.
        inside = np.flatnonzero(d <= r)
        rows.append({
            "halo_id": int(hid), "hr_num_p": int(hr.num_p[i]),
            "search_radius_mpc_h": r,
            "hit": bool(near.size),
            "best_num_p": int(cat.num_p[near].max()) if near.size else 0,
            "nearest_mpc_h": float(d.min()),
            "n_within_radius": int(inside.size),
            "best_frac_within_radius": (
                float(cat.num_p[inside].max()) / max(int(hr.num_p[i]), 1)
                if inside.size else 0.0),
        })
    n = len(rows)
    return {"n": n, "hits": int(sum(r["hit"] for r in rows)),
            "rate": (sum(r["hit"] for r in rows) / n) if n else float("nan"),
            "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--box", default="set8")
    ap.add_argument("--tag", required=True, help="candidate catalog tag")
    ap.add_argument("--host-id", type=int, default=271800,
                    help="the HR host the fine-tune was trained on")
    ap.add_argument("--radius-factor", type=float, default=1.0,
                    help="count subhalos inside this many R_vir of the host")
    ap.add_argument("--min-p", type=int, default=50)
    ap.add_argument("--targets-json", default="",
                    help="a run's subhalos.json: score the SUPERVISED targets "
                         "one by one, which is the question this run asked")
    ap.add_argument("--match-radius-factor", type=float, default=1.0)
    ap.add_argument("--match-mass-frac", type=float, default=0.25)
    ap.add_argument("--sweep", action="store_true",
                    help="also run section 5's four-rung threshold ladder and "
                         "profile where the missed targets' material went")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    banner(f"gather Rockstar comparison: {args.box}, tag {args.tag}")
    cats = find_catalogs(args.box, args.tag)
    for need in ("hr", "base", "cand"):
        if need not in cats:
            raise SystemExit(
                f"missing the {need} catalog for {args.box} / {args.tag}. "
                "hr+base come from catalog_summaries.py --source {hr,base}; "
                "cand from scripts/slurm/flow_rockstar_catalog_cpu.sbatch")
    hr = cats["hr"]

    row = np.flatnonzero(hr.ids == int(args.host_id))
    if row.size == 0:
        raise SystemExit(f"host {args.host_id} not in the {args.box} HR catalog")
    row = int(row[0])
    centre = hr.pos[row]
    rvir = float(hr.rvir[row]) / 1000.0 * float(args.radius_factor)   # kpc/h -> Mpc/h
    print(f"  HR host {args.host_id}: log10 Mvir {np.log10(hr.mvir[row]):.2f}, "
          f"num_p {hr.num_p[row]}, R_vir {rvir:.3f} Mpc/h")

    result = {"box": args.box, "tag": args.tag, "host_id": int(args.host_id),
              "host_log_mvir": float(np.log10(hr.mvir[row])),
              "radius_mpc_h": rvir, "min_p": int(args.min_p)}

    # 1. subhalos inside the host
    counts = {}
    for name in ("hr", "base", "cand"):
        rows = subhalos_within(cats[name], centre, rvir, min_p=args.min_p)
        counts[name] = count_by_bin(cats[name], rows)
    result["subhalos_within_rvir"] = counts
    print("\n  subhalos within R_vir of the host (Rockstar, 6-D bound):")
    print(f"    {'bin':>12s} {'HR':>7s} {'frozen':>7s} {'tuned':>7s} "
          f"{'frozen/HR':>10s} {'tuned/HR':>9s}")
    for k in list(counts["hr"]):
        h, b, c = counts["hr"][k], counts["base"][k], counts["cand"][k]
        print(f"    {k:>12s} {h:7d} {b:7d} {c:7d} "
              f"{(b / h if h else float('nan')):10.3f} "
              f"{(c / h if h else float('nan')):9.3f}")
    result["ratio_total_frozen"] = (counts["base"]["total"] / counts["hr"]["total"]
                                    if counts["hr"]["total"] else float("nan"))
    result["ratio_total_tuned"] = (counts["cand"]["total"] / counts["hr"]["total"]
                                   if counts["hr"]["total"] else float("nan"))

    # 2. the host itself must not have moved or changed mass
    host = {}
    for name in ("base", "cand"):
        r = match_host(cats[name], centre, float(hr.mvir[row]))
        if r is None:
            host[name] = {"found": False}
            continue
        host[name] = {
            "found": True,
            "log_mvir": float(np.log10(max(cats[name].mvir[r], 1.0))),
            "num_p": int(cats[name].num_p[r]),
            "offset_mpc_h": float(np.linalg.norm(
                periodic_delta(cats[name].pos[r], centre))),
        }
    result["host"] = host
    print("\n  the host itself (a gain bought by fragmenting it is not a gain):")
    print(f"    HR       log10 Mvir {np.log10(hr.mvir[row]):.3f}")
    for name in ("base", "cand"):
        h = host[name]
        print(f"    {name:8s} " + ("NOT FOUND" if not h["found"] else
              f"log10 Mvir {h['log_mvir']:.3f}  offset {h['offset_mpc_h']:.3f} Mpc/h"))

    # 3. is the change local to the cluster, or a splice-edge artifact?
    edges = [0.0, rvir, 2 * rvir, 4 * rvir, 8 * rvir, 16 * rvir]
    prof = shell_profile(cats["base"], cats["cand"], centre, edges,
                         min_p=args.min_p)
    result["shell_profile"] = prof
    print("\n  halo count vs distance from the host (edge-artifact check):")
    for p in prof:
        print(f"    {p['r_lo']:6.2f}-{p['r_hi']:6.2f} Mpc/h   base {p['base']:6d}"
              f"   tuned {p['cand']:6d}   delta {p['delta']:+6d}")

    # 3b. the supervised targets, one at a time -- the question this run asked
    if args.targets_json:
        import json as _json
        ids = [int(r["halo_id"]) for r in
               _json.loads(Path(args.targets_json).read_text())["rows"]]
        tgt = {name: target_hit_rate(
                   cats[name], hr, ids,
                   radius_factor=float(args.match_radius_factor),
                   mass_frac=float(args.match_mass_frac))
               for name in ("base", "cand")}
        result["supervised_targets"] = tgt
        print(f"\n  the {tgt['base']['n']} SUPERVISED targets, recovered as bound "
              "halos (position AND mass):")
        print(f"    frozen {tgt['base']['hits']:3d}/{tgt['base']['n']} "
              f"= {tgt['base']['rate']:.3f}")
        print(f"    tuned  {tgt['cand']['hits']:3d}/{tgt['cand']['n']} "
              f"= {tgt['cand']['rate']:.3f}")

        if args.sweep:
            # The strict row alone cannot distinguish "the objects are not there"
            # from "the test is too strict", and a run that fails both readings
            # has failed for a reason worth naming. This is the same four-rung
            # ladder docs/sr2_gather_finetune.md section 5 reports.
            ladder = [(1.0, 0.25), (2.0, 0.10), (3.0, 0.05), (4.0, 0.02)]
            sweep = []
            for rf, mf in ladder:
                row = {"radius_factor": rf, "mass_frac": mf}
                for name in ("base", "cand"):
                    h = target_hit_rate(cats[name], hr, ids,
                                        radius_factor=rf, mass_frac=mf)
                    row[name] = h["hits"]
                    row[f"{name}_rate"] = h["rate"]
                sweep.append(row)
            result["supervised_sweep"] = sweep
            print("\n  threshold sweep (closes the 'too strict' loophole):")
            print("    radius  mass_frac   frozen    tuned")
            for row in sweep:
                print(f"    {row['radius_factor']:.0f} x r_vir   >= "
                      f"{row['mass_frac']:.0%}     {row['base']:4d}/{tgt['base']['n']}"
                      f"  {row['cand']:4d}/{tgt['cand']['n']}")

            # Where the objects actually are, for the targets that missed.
            miss = [r for r in tgt["cand"]["rows"] if not r["hit"]]
            if miss:
                nd = np.array([r["nearest_mpc_h"] for r in miss])
                bf = np.array([r["best_frac_within_radius"] for r in miss])
                result["supervised_miss_profile"] = {
                    "n_miss": len(miss),
                    "nearest_mpc_h_median": float(np.median(nd)),
                    "search_radius_median": float(np.median(
                        [r["search_radius_mpc_h"] for r in miss])),
                    "frac_with_something_in_radius": float(np.mean(bf > 0)),
                    "best_frac_within_radius_median": float(np.median(bf)),
                }
                m = result["supervised_miss_profile"]
                print(f"\n  the {m['n_miss']} misses: nearest halo of ANY mass is "
                      f"{m['nearest_mpc_h_median']:.3f} Mpc/h away (median), "
                      f"search radius {m['search_radius_median']:.3f};")
                print(f"    {m['frac_with_something_in_radius']:.0%} have SOMETHING "
                      f"inside the radius, carrying "
                      f"{m['best_frac_within_radius_median']:.1%} of HR's particles "
                      f"(median).")

    # 4. the whole box: the gate requires this to be unchanged
    box = {}
    for name in ("hr", "base", "cand"):
        c = cats[name]
        box[name] = {
            "hosts_ge_200p": int(((c.parent_ids < 0) & (c.num_p >= 200)).sum()),
            "subhalos_ge_50p": int(((c.parent_ids >= 0) & (c.num_p >= 50)).sum()),
            # The 50p floor is where the abundance defect lives (memory
            # `subhalo-count-is-hr-bound-is-not`), but it is also where Rockstar
            # is noisiest. The 100p and 200p counts say whether a whole-box gain
            # is real objects or marginal ones.
            "subhalos_ge_100p": int(((c.parent_ids >= 0) & (c.num_p >= 100)).sum()),
            "subhalos_ge_200p": int(((c.parent_ids >= 0) & (c.num_p >= 200)).sum()),
            "n_halos": int(c.n),
        }
    result["whole_box"] = box
    # The spliced fraction is a knob, not a constant: it is 0.78% at four tiles
    # and 3.1% at sixteen, and reading a whole-box change against the wrong one
    # is how a proportionate change gets called a red flag.
    frac = spliced_fraction(args.box, args.tag)
    result["spliced_volume_fraction"] = frac
    how_much = ("an unknown fraction of" if frac is None
                else f"{100.0 * frac:.2f}% of")
    print(f"\n  whole box ({how_much} it was spliced):")
    for name in ("hr", "base", "cand"):
        print(f"    {name:6s} hosts>=200p {box[name]['hosts_ge_200p']:6d}   "
              f"subhalos>=50p {box[name]['subhalos_ge_50p']:7d}"
              f"  >=100p {box[name]['subhalos_ge_100p']:7d}"
              f"  >=200p {box[name]['subhalos_ge_200p']:7d}")
    dh = box["cand"]["hosts_ge_200p"] - box["base"]["hosts_ge_200p"]
    result["hosts_ge_200p_change"] = int(dh)

    out = Path(args.out) if args.out else (
        paths.subdir("flow_rockstar", "compare", create=True)
        / f"{args.box}__{args.tag}.json")
    write_json(out, result)
    print(f"\n  wrote {out}")

    if "supervised_targets" in result:
        st = result["supervised_targets"]
        print(f"\nSUPERVISED TARGETS: {st['base']['hits']} -> {st['cand']['hits']} "
              f"of {st['base']['n']} recovered as bound halos.")
    t, b = result["ratio_total_tuned"], result["ratio_total_frozen"]
    print(f"\nVERDICT: subhalos inside the host went {b:.3f} -> {t:.3f} of HR's; "
          f"hosts>=200p in the whole box changed by {dh:+d}. "
          + ("The clumps are bound." if t > b + 0.05 else
             "No bound-subhalo gain: the density statistic moved and Rockstar "
             "did not agree."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
