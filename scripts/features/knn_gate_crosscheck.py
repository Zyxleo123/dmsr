#!/usr/bin/env python
"""Which is telling the truth where the kNN statistic and Rockstar counts disagree?

Round 3 of the kNN go/no-go (``knn_cdf_split_region_set9.json``, job 51399)
found the statistic scoring every arm WORSE than frozen on HR's own 50-200p
subhalo particles -- ``smallhalo`` and both ``condsmall`` runs included -- while
the Rockstar gate credits all of them with MORE 50-200p subhalos. Two readings,
and they point in opposite directions for the objective:

* the counts are right and the statistic is blind (it measures the particles HR
  put in subhalos; an arm can build real subhalos out of other particles), or
* the statistic is right and the counts are inflated: objects in the wrong
  places (``occupation-ratio-is-gameable``; ``smallhalo`` overshot HR in every
  field sub bin), or big halos broken into 50-200p pieces (``condsmall`` lost
  >=200p objects and its whole region reads ~2x diffuse).

Counts cannot separate those; MATCHING can. Per gated arm catalog, inside the
gate's own region (the spliced tile boxes dilated by ``--dilate`` Mpc/h,
``gather_holdout_figures_data``'s definition):

``sub_50_200``           arm subhalos with 50 <= num_p < 200 (the raw count)
``sub_50_200_matched``   ... with an HR halo within max(r_vir, 0.15) Mpc/h whose
                         num_p is within ``--mass-factor`` (the gate's radius)
``hr_recall_50_200``     fraction of HR's region 50-200p subhalos with such an
                         arm counterpart -- abundance of HR's OWN objects
``lost_ge200``           frozen halos >=200p with no arm halo of >= half their
                         num_p within their radius; ``_hr_backed`` = of those,
                         the ones HR also has
``fragment_candidates``  arm 50-200p objects with NO HR match sitting inside a
                         lost frozen >=200p halo's radius

Then, for the arms whose fields were measured by ``knn_cdf_split_diagnostic.py
--region-json``, the Spearman rank correlation across arms between each kNN
term's change vs frozen and each gate metric's change vs frozen. The question
is which gate metric the statistic tracks: if it tracks ``hr_recall`` and
``fragment_candidates`` but not the raw count, it is measuring the harder-to-
game quantity and the disagreement is the counts' fault.

Catalogs only for the forensics (seconds per arm); the kNN terms are read from
the split diagnostic's JSON. No fields, no model.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402
from gather_holdout_figures_data import (  # noqa: E402
    BOX, in_tile_region, tile_boxes,
)

MIN_RADIUS = 0.15   # Mpc/h -- compare_gather_catalog's floor on the search radius


def _radius(cat) -> np.ndarray:
    return np.maximum(np.asarray(cat.rvir, dtype=float) / 1000.0, MIN_RADIUS)


def has_counterpart(src_pos, src_np, src_rad, dst_pos, dst_np, *,
                    lo: float, hi: float) -> np.ndarray:
    """``src[i]`` has a ``dst`` halo within ``src_rad[i]`` with num_p ratio in [lo, hi]."""
    out = np.zeros(len(src_pos), dtype=bool)
    if len(src_pos) == 0 or len(dst_pos) == 0:
        return out
    tree = cKDTree(np.mod(dst_pos, BOX), boxsize=BOX)
    hits = tree.query_ball_point(np.mod(src_pos, BOX), r=src_rad)
    for i, idx in enumerate(hits):
        if idx:
            ratio = dst_np[idx] / float(src_np[i])
            out[i] = bool(np.any((ratio >= lo) & (ratio <= hi)))
    return out


def forensics(arm, hr, frozen, region_boxes, dilate: float,
              mass_factor: float) -> Dict[str, float]:
    lo, hi = 1.0 / mass_factor, mass_factor

    def sel(cat, sub: Optional[bool], a: int, b: int):
        m = in_tile_region(np.asarray(cat.pos), region_boxes, dilate=dilate)
        m &= (cat.num_p >= a) & (cat.num_p < b)
        if sub is True:
            m &= cat.parent_ids >= 0
        elif sub is False:
            m &= cat.parent_ids < 0
        return np.flatnonzero(m)

    row: Dict[str, float] = {}
    bins = [(50, 100), (100, 200), (200, 500), (500, 2000), (2000, 10 ** 12)]
    for a, b in bins:
        row[f"sub_{a}_{b}"] = int(sel(arm, True, a, b).size)
        row[f"host_{a}_{b}"] = int(sel(arm, False, a, b).size)

    # Arm's 50-200p subhalos: which are HR objects?
    s = sel(arm, True, 50, 200)
    matched = has_counterpart(arm.pos[s], arm.num_p[s], _radius(arm)[s],
                              hr.pos, hr.num_p, lo=lo, hi=hi)
    row["sub_50_200"] = int(s.size)
    row["sub_50_200_matched"] = int(matched.sum())
    row["sub_50_200_precision"] = float(matched.mean()) if s.size else float("nan")

    # HR's 50-200p subhalos: which does the arm have?
    h = sel(hr, True, 50, 200)
    rec = has_counterpart(hr.pos[h], hr.num_p[h], _radius(hr)[h],
                          arm.pos, arm.num_p, lo=lo, hi=hi)
    row["hr_sub_50_200"] = int(h.size)
    row["hr_recall_50_200"] = float(rec.mean()) if h.size else float("nan")

    # Frozen's >=200p objects the arm lost (no arm halo of >= half their size).
    f = sel(frozen, None, 200, 10 ** 12)
    kept = has_counterpart(frozen.pos[f], frozen.num_p[f], _radius(frozen)[f],
                           arm.pos, arm.num_p, lo=0.5, hi=np.inf)
    lost = f[~kept]
    row["frozen_ge200"] = int(f.size)
    row["lost_ge200"] = int(lost.size)
    backed = has_counterpart(frozen.pos[lost], frozen.num_p[lost],
                             _radius(frozen)[lost], hr.pos, hr.num_p, lo=lo, hi=hi)
    row["lost_ge200_hr_backed"] = int(backed.sum())

    # Unmatched arm 50-200p objects inside a lost frozen halo's radius.
    unmatched = s[~matched]
    if lost.size and unmatched.size:
        tree = cKDTree(np.mod(arm.pos[unmatched], BOX), boxsize=BOX)
        hits = tree.query_ball_point(np.mod(frozen.pos[lost], BOX),
                                     r=_radius(frozen)[lost])
        inside = set(i for idx in hits for i in idx)
        row["fragment_candidates"] = int(len(inside))
    else:
        row["fragment_candidates"] = 0
    return row


def knn_terms(knn: dict, label: str) -> Optional[Dict[str, float]]:
    lf = (knn.get("loss_forms") or {}).get(label)
    if lf is None:
        return None
    out = {f"knn_{k}": float(lf[f"{k}_total"])
           for k in ("unpaired", "class_cond_mass", "class_cond_equal", "paired")}
    for key, v in lf.items():
        m = re.fullmatch(r"(class_cond|paired)\[(\w+)\]", key)
        if m:
            out[f"knn_{m.group(1)}[{m.group(2)}]"] = float(np.mean(v))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward-root", required=True)
    ap.add_argument("--box", default="set9")
    ap.add_argument("--region-json", required=True,
                    help="an arm field .json whose `tiles` is the spliced region")
    ap.add_argument("--frozen-tag", default=None,
                    help="default mgho_frozen_<box>")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="candidate tags; default every flow_rockstar candidate "
                         "catalog for the box")
    ap.add_argument("--knn-json", default=None,
                    help="knn_cdf_split_diagnostic --region-json output; field "
                         "labels must be the tag with mgho_ and _<box> stripped")
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--mass-factor", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    root = Path(a.reward_root)
    frozen_tag = a.frozen_tag or f"mgho_frozen_{a.box}"

    def cat_of(kind, tag=""):
        pat = (f"{root}/halos/{a.box}__hr__hr/*_rockstar/halos_*.ascii" if kind == "hr"
               else f"{root}/flow_rockstar/halos/{a.box}__candidate__{tag}/*/halos_*.ascii")
        hits = sorted(glob.glob(pat))
        return load_rockstar_ascii(hits[0]) if hits else None

    tiles = [int(t) for t in json.loads(Path(a.region_json).read_text())["tiles"]]
    boxes = tile_boxes(tiles, TileGrid(ng_hr=512, tile_hr=64, boxsize_mpc_h=BOX))
    hr = cat_of("hr")
    frozen = cat_of("candidate", frozen_tag)
    if hr is None or frozen is None:
        raise SystemExit("HR or frozen catalog missing")
    if a.arms:
        arms = list(a.arms)
    else:
        arms = sorted(Path(p).name.split("__candidate__", 1)[1] for p in glob.glob(
            f"{root}/flow_rockstar/halos/{a.box}__candidate__*"))
    knn = json.loads(Path(a.knn_json).read_text()) if a.knn_json else {}

    rows: List[Dict] = []
    for tag in arms:
        cat = frozen if tag == frozen_tag else cat_of("candidate", tag)
        if cat is None:
            print(f"  MISSING catalog {tag}", flush=True)
            continue
        row = {"tag": tag,
               "label": re.sub(rf"^mgho_|_{a.box}$", "", tag)}
        row.update(forensics(cat, hr, frozen, boxes, a.dilate, a.mass_factor))
        kt = knn_terms(knn, row["label"])
        if kt:
            row.update(kt)
        rows.append(row)
        print(f"  {tag:<48} sub50-200 {row['sub_50_200']:>5} matched "
              f"{row['sub_50_200_matched']:>5} recall {row['hr_recall_50_200']:.3f} "
              f"lost>=200 {row['lost_ge200']:>4} frag {row['fragment_candidates']:>4}",
              flush=True)
    hr_row = {"tag": "hr", "label": "hr"}
    hr_row.update(forensics(hr, hr, frozen, boxes, a.dilate, a.mass_factor))
    rows.append(hr_row)

    # ---- rank correlations across arms, as changes vs frozen ------------------
    base = next((r for r in rows if r["tag"] == frozen_tag), None)
    gate_keys = ["sub_50_200", "sub_50_200_matched", "sub_50_200_precision",
                 "hr_recall_50_200", "lost_ge200", "lost_ge200_hr_backed",
                 "fragment_candidates", "host_200_500", "host_500_2000"]
    knn_keys = sorted({k for r in rows for k in r if k.startswith("knn_")})
    corr = {}
    have = [r for r in rows if r["tag"] not in (frozen_tag, "hr")
            and any(k in r for k in knn_keys)]
    if base is not None and len(have) >= 5 and all(k in base for k in knn_keys):
        for kk in knn_keys:
            x = np.array([r[kk] - base[kk] for r in have])
            corr[kk] = {}
            for gk in gate_keys:
                y = np.array([r[gk] - base[gk] for r in have], dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 5 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
                    rho, p = spearmanr(x[ok], y[ok])
                    corr[kk][gk] = {"rho": float(rho), "p": float(p),
                                    "n": int(ok.sum())}
        print("\n  Spearman rho across arms (kNN change vs gate change; kNN is a "
              "LOSS, so tracking a GOOD gate metric means rho < 0):", flush=True)
        head = "".join(f"{g[:14]:>16}" for g in gate_keys)
        print(f"    {'':<28}{head}", flush=True)
        for kk in knn_keys:
            cells = "".join(
                f"{corr[kk][g]['rho']:>+16.2f}" if g in corr[kk] else f"{'':>16}"
                for g in gate_keys)
            print(f"    {kk[:28]:<28}{cells}", flush=True)
    else:
        print(f"\n  correlations skipped: {len(have)} arms with kNN terms "
              "(need >= 5 and a frozen row)", flush=True)

    out = {"box": a.box, "region_tiles": tiles, "dilate": a.dilate,
           "mass_factor": a.mass_factor, "frozen_tag": frozen_tag,
           "knn_json": a.knn_json, "rows": rows, "spearman": corr,
           "n_arms_with_knn": len(have)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"=== wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
