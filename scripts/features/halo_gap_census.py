#!/usr/bin/env python
"""WHERE is SR2's halo-count deficit -- in small halos, or in all of them?

The question this answers, and why it is not already answered
-------------------------------------------------------------
Every arm in ``docs/sr2_member_gather_training.md`` supervised ``parent_ids >=
0``: HR's subhalos. The held-out gate (section 11.4) then measured, in the
edited region, that this was aimed at only half the problem:

    in the edited region      base SR2    tuned `self`    HR truth
      subhalos >=50p             4,541          9,187      12,367
      hosts    >=200p            3,028          2,708       3,775

HR wants **747 more top-level halos than base has**, and not one supervised set
in any run has ever pointed at that deficit. But "top-level halo" spans four
decades of mass, and the two readings of the 747 lead to different runs:

  SMALL-HALO deficit   the gap is concentrated near the resolution floor, and
                       ``base/HR`` climbs toward 1 with mass. Then the fix is a
                       ``num_p`` WINDOW -- supervise small halos of either
                       parentage (``--population all --max-num-p ...``) -- and
                       the existing objective is the right one, pointed wider.
  FLAT deficit         ``base/HR`` is roughly constant across every mass bin.
                       Then SR2 is losing halos at all scales, a mass window
                       buys nothing over the current subhalo cut, and the
                       missing thing is not "small objects" at all.

This job decides between them, from catalogs already on disk. No field is read
and no halo finder runs; it is minutes on a CPU.

The third axis, which is the one that decides the TILING
--------------------------------------------------------
A count split by mass alone cannot say whether the missing top-level halos are
*reachable*. The fine-tune's supervised tiles are the top four Lagrangian tiles
of massive clusters -- proto-cluster patches -- so a deficit sitting in genuinely
isolated field halos is out of reach of a wider POPULATION and needs a wider
TILING (``--n-tiles``, or a field-tile pool), which is a different change.

So every top-level halo is also labelled by its distance to the nearest **more
massive resolved host** (``>= --host-floor`` particles, top-level), in units of
that host's ``R_vir``:

    inside   < 1 R_vir     top-level in the catalog, but sitting inside a bigger
                           halo's virial radius. `host-damage-is-mostly-
                           reparenting` measured ~88% of the `self` arm's host
                           loss to be objects moving across exactly this line,
                           so this bin is where a count difference is most
                           likely to be bookkeeping rather than physics.
    near     1-3 R_vir     the infall region: a cluster's own Lagrangian tiles
                           plausibly hold this material.
    field    > 3 R_vir     genuinely isolated. Reachable only by changing which
                           tiles are trained.

What is a count and what is not
-------------------------------
These are **counts in bins**, never matched objects. Two Rockstar catalogs of
two different fields share no halo ids, so "the same halo is missing" is not a
statement this job can make -- ``scripts/features/host_forensics.py`` is the
matched-descendant tool and answers a different question (what happened to a
*given* base host). A bin count is exactly the right estimator for a mass
function and exactly the wrong one for a fate, and the two are not mixed here.

Writes, under ``--out-dir``:
  * ``census.json``   -- every count, every gap, every ratio, plus a VERDICT.
  * ``<kind>.npz``    -- per-halo (num_p, population, environment) for replots.

    python scripts/features/halo_gap_census.py --box set9 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
from cosmo_sr.features.subhalo_gather import (  # noqa: E402
    ENV_EDGES, ENV_LABELS, halo_environment,
)
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

#: Mass bins, in Rockstar ``num_p``. The first edge is 50 because that is the
#: cut every subhalo number in this line is quoted at, and `subhalo-count-is-
#: HR's-bound-is-not` measured 46.5% of HR's subhalos to live in 50-100p -- a
#: census that started at 200p would bin away the population most likely to
#: carry the answer.
MASS_EDGES = [50, 100, 200, 500, 1000, 2000, 5000, 10 ** 12]
MASS_LABELS = ["50-100p", "100-200p", "200-500p", "500-1000p", "1000-2000p",
               "2000-5000p", "5000+p"]

#: Distance to the nearest more-massive resolved host, in units of ITS
#: R_vir. The bins, the labels and the estimator all come from
#: ``cosmo_sr.features.subhalo_gather``, which is also what
#: ``member_gather.build_member_sets`` selects on -- so the environment
#: this census measures a deficit in and the environment an arm is pointed
#: at are the same quantity by construction, not by agreement.
def environment(cat: HaloCatalog, host_floor: int) -> np.ndarray:
    """``(N,)`` index into :data:`ENV_LABELS`; see ``halo_environment``."""
    return halo_environment(cat, host_floor=int(host_floor),
                            boxsize_mpc_h=float(BOX))


def classify(cat: HaloCatalog, host_floor: int,
             region_boxes: Optional[np.ndarray] = None,
             dilate: float = 0.0) -> Dict[str, np.ndarray]:
    """Per-halo (num_p, population, environment, in-region) for one catalog."""
    inside = (np.ones(cat.n, dtype=bool) if region_boxes is None
              else in_tile_region(np.asarray(cat.pos, dtype=np.float64),
                                  region_boxes, dilate=dilate))
    return {
        "num_p": np.asarray(cat.num_p, dtype=np.int64),
        "mass_bin": np.clip(np.digitize(cat.num_p, MASS_EDGES[1:], right=False),
                            0, len(MASS_LABELS) - 1).astype(np.int64),
        "is_top": (cat.parent_ids < 0).astype(np.int64),
        "env": environment(cat, host_floor),
        "in_region": inside.astype(np.int64),
    }


def table(c: Dict[str, np.ndarray], *, top: Optional[bool] = None,
          region_only: bool = True) -> Dict[str, List[int]]:
    """Counts per mass bin, and per (mass bin x environment).

    ``top=None`` counts both populations; ``True``/``False`` restricts to
    top-level / subhalos. Below the first mass edge nothing is counted at all --
    Rockstar's own floor varies with the field, and a bin that straddles it
    would report the finder's threshold as a physical deficit.
    """
    sel = c["num_p"] >= MASS_EDGES[0]
    if region_only:
        sel = sel & (c["in_region"] > 0)
    if top is not None:
        sel = sel & ((c["is_top"] > 0) == bool(top))
    mb, env = c["mass_bin"][sel], c["env"][sel]
    by_mass = [int(np.sum(mb == i)) for i in range(len(MASS_LABELS))]
    by_env = {ENV_LABELS[e]: [int(np.sum((mb == i) & (env == e)))
                              for i in range(len(MASS_LABELS))]
              for e in range(len(ENV_LABELS))}
    return {"by_mass": by_mass, "total": int(sel.sum()), "by_mass_env": by_env}


def _gap(hr: List[int], other: List[int]) -> Dict[str, List]:
    """HR minus a candidate, per bin, with the ratio that carries the shape."""
    h = np.asarray(hr, dtype=np.float64)
    o = np.asarray(other, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(h > 0, o / h, np.nan)
    return {"gap": (h - o).astype(np.int64).tolist(),
            "ratio": [None if not np.isfinite(r) else round(float(r), 4)
                      for r in ratio]}


def verdict(hr_top: Dict, base_top: Dict, hr_sub: Dict, base_sub: Dict) -> Dict:
    """Small-halo deficit, flat deficit, or neither -- stated, with its evidence.

    The discriminator is the SHAPE of ``base/HR`` across mass, not its level. A
    run that is uniformly at 0.7 of HR is losing 30% of halos at every scale and
    a mass window does nothing for it; a run at 0.4 in the smallest bin and 0.95
    in the largest is losing small objects specifically, which is what a window
    is for. The threshold is deliberately coarse -- this decides which of two
    experiments to run, not a number anyone quotes.
    """
    r = np.asarray([np.nan if x is None else x
                    for x in _gap(hr_top["by_mass"], base_top["by_mass"])["ratio"]],
                   dtype=np.float64)
    ok = np.isfinite(r) & (np.asarray(hr_top["by_mass"]) >= 20)
    out: Dict = {"top_level_ratio_by_mass": r.round(4).tolist(),
                 "bins_used": [MASS_LABELS[i] for i in np.flatnonzero(ok)]}
    if ok.sum() < 2:
        out["text"] = ("INCONCLUSIVE: fewer than two mass bins have >=20 HR "
                       "top-level halos in this region. Widen the region or "
                       "run box-wide.")
        return out
    lo, hi = float(r[ok][0]), float(r[ok][-1])
    span = float(np.nanmax(r[ok]) - np.nanmin(r[ok]))
    gap_tot = int(sum(_gap(hr_top["by_mass"], base_top["by_mass"])["gap"]))
    small = int(sum(_gap(hr_top["by_mass"], base_top["by_mass"])["gap"][:2]))
    out.update({
        "top_level_gap_total": gap_tot,
        "top_level_gap_below_200p": small,
        "frac_of_gap_below_200p": (round(small / gap_tot, 4) if gap_tot else None),
        "ratio_smallest_bin": round(lo, 4),
        "ratio_largest_bin": round(hi, 4),
        "ratio_span": round(span, 4),
        "subhalo_gap_total": int(sum(
            _gap(hr_sub["by_mass"], base_sub["by_mass"])["gap"])),
    })
    if span < 0.15:
        out["text"] = (
            f"FLAT: base/HR varies by only {span:.2f} across the mass bins "
            f"({lo:.2f} smallest, {hi:.2f} largest). SR2 loses top-level halos "
            "at every scale, so a num_p WINDOW buys nothing over the current "
            "subhalo cut -- widening the population to `all` with no ceiling "
            "restriction is the honest reading, and the deficit is not about "
            "object size.")
    elif hi - lo > 0.15:
        out["text"] = (
            f"SMALL-HALO: base/HR climbs from {lo:.2f} in {MASS_LABELS[0]} to "
            f"{hi:.2f} in the largest populated bin, and {small} of the {gap_tot} "
            "missing top-level halos sit below 200p. The deficit IS about size, "
            "and `--population all` with a `--max-num-p` ceiling near the top of "
            "the deficit is the arm this supports.")
    else:
        out["text"] = (
            f"INVERTED: base/HR is HIGHER in the small bins ({lo:.2f}) than in "
            f"the large ({hi:.2f}). SR2 is losing the BIG top-level halos, not "
            "the small ones -- a small-halo window is aimed the wrong way, and "
            "the host-preservation guard is the term this points at.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward-root",
                    default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--kinds", default="base,hr",
                    help="comma list; 'base', 'hr', or an arm name mapped by "
                         "--arm-tags. base and hr are both REQUIRED -- the gap "
                         "is defined between them.")
    ap.add_argument("--arm-tags", default="",
                    help="comma list of name=tag for candidate arms, e.g. "
                         "self=mgho_all_blocks_self_set9")
    ap.add_argument("--run-dir", default="",
                    help="an arm's holdout export dir; its export.json gives "
                         "the spliced tiles, and the census is then reported "
                         "for the EDITED REGION as well as box-wide. Without "
                         "it only the box-wide census is available, and for an "
                         "arm that is 93.75%% untouched frozen SR2.")
    ap.add_argument("--dilate", type=float, default=2.0)
    ap.add_argument("--host-floor", type=int, default=200,
                    help="particles for a top-level halo to count as a HOST "
                         "when labelling other halos' environment")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    R = Path(a.reward_root)
    arm_tags = dict(kv.split("=", 1) for kv in a.arm_tags.split(",") if "=" in kv)
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    for need in ("base", "hr"):
        if need not in kinds:
            raise SystemExit(f"--kinds must include '{need}': the gap is "
                             "defined as HR minus base, and a census of one "
                             "catalog is not a gap.")

    region_boxes = None
    tiles: List[int] = []
    if a.run_dir:
        export = json.loads((Path(a.run_dir) / "export.json").read_text())
        tiles = ([int(t) for t in export["tiles"]] if "tiles" in export
                 else sorted({int(t) for h in export["per_host"]
                              for t in h["tiles"]}))
        region_boxes = tile_boxes(tiles, TileGrid())
        print(f"edited region: {len(tiles)} tiles, dilated {a.dilate} Mpc/h",
              flush=True)
    else:
        print("no --run-dir: box-wide only. For an ARM that is 93.75% untouched "
              "frozen SR2 and the comparison says little.", flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cls: Dict[str, Dict[str, np.ndarray]] = {}
    for kind in kinds:
        path = (catalog_path(R, kind, a.box) if kind in ("base", "hr")
                else catalog_path(R, "cand", a.box, arm_tags[kind]))
        if kind not in ("base", "hr") and kind not in arm_tags:
            raise SystemExit(f"no tag for arm '{kind}' in --arm-tags")
        cat = load_rockstar_ascii(path)
        c = classify(cat, a.host_floor, region_boxes, a.dilate)
        np.savez_compressed(out_dir / f"{kind}.npz", **c)
        cls[kind] = c
        print(f"  {kind:<8} {cat.n:>8} halos, "
              f"{int(np.sum(c['is_top'] > 0)):>7} top-level, "
              f"{int(np.sum(c['in_region'] > 0)):>8} in region", flush=True)

    scopes = [("box", False)] + ([("region", True)] if region_boxes is not None
                                 else [])
    out: Dict = {"ok": True, "box": a.box, "kinds": kinds,
                 "mass_labels": MASS_LABELS, "env_labels": ENV_LABELS,
                 "host_floor": int(a.host_floor), "dilate": float(a.dilate),
                 "n_tiles": len(tiles), "scopes": {}}

    for scope, region_only in scopes:
        per: Dict = {}
        for kind in kinds:
            per[kind] = {
                "top": table(cls[kind], top=True, region_only=region_only),
                "sub": table(cls[kind], top=False, region_only=region_only),
                "all": table(cls[kind], top=None, region_only=region_only),
            }
        gaps: Dict = {}
        for kind in kinds:
            if kind == "hr":
                continue
            gaps[kind] = {
                pop: _gap(per["hr"][pop]["by_mass"], per[kind][pop]["by_mass"])
                for pop in ("top", "sub", "all")}
            gaps[kind]["top_by_env"] = {
                e: _gap(per["hr"]["top"]["by_mass_env"][e],
                        per[kind]["top"]["by_mass_env"][e])
                for e in ENV_LABELS}
        out["scopes"][scope] = {
            "counts": per, "gaps": gaps,
            "verdict": verdict(per["hr"]["top"], per["base"]["top"],
                               per["hr"]["sub"], per["base"]["sub"]),
        }

        # --- the readable form, in the log, for the scope that decides -----
        print(f"\n=== {scope} ===")
        # The data rows are `4 spaces + a 7-wide kind`, so the header carries
        # the same 11-character indent or the columns do not line up and the
        # table is read wrong at a glance.
        hdr = " " * 11 + "".join(f"{m:>12}" for m in MASS_LABELS)
        for pop, name in (("top", "TOP-LEVEL"), ("sub", "SUBHALO")):
            print(f"\n  {name} counts by num_p")
            print(hdr)
            for kind in kinds:
                row = "".join(f"{v:>12,}" for v in per[kind][pop]["by_mass"])
                print(f"    {kind:<7}{row}")
            g = gaps["base"][pop]
            print(f"    {'HR-base':<7}" + "".join(f"{v:>12,}" for v in g["gap"]))
            print(f"    {'base/HR':<7}"
                  + "".join(f"{'--' if r is None else f'{r:.3f}':>12}"
                            for r in g["ratio"]))
        print("\n  TOP-LEVEL gap (HR - base) by environment "
              "(distance to the nearest bigger resolved host, in its R_vir)")
        print(hdr)
        for e in ENV_LABELS:
            row = "".join(f"{v:>12,}" for v in gaps["base"]["top_by_env"][e]["gap"])
            print(f"    {e:<7}{row}")
        print("\n  VERDICT: " + out["scopes"][scope]["verdict"]["text"])

    write_json(out_dir / "census.json", out)
    print(f"\nwrote {out_dir}/census.json and <kind>.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
