#!/usr/bin/env python
"""Do SUBFIND and Rockstar agree about the subhalos a fine-tune produced?

The question this exists to settle
----------------------------------
Rockstar reports that the tuned arms add hundreds of subhalos inside the trained
host, and the projected density panels show nothing. Two stories fit:

* **the picture is wrong** -- the objects are real, and projecting a 25 Mpc/h
  column over a 0.1 Mpc/h subhalo buries them (that is what
  ``scripts/features/subhalo_zoom_data.py`` measures directly);
* **the catalogue is wrong** -- Rockstar links in six dimensions and is reporting
  phase-space features with no density peak behind them.

SUBFIND can separate the two because it defines substructure in *configuration
space*: a density peak bounded by a saddle, kept only if self-bound. So the
number that matters here is not "how many subhalos does SUBFIND find" but
**what fraction of Rockstar's subhalos SUBFIND confirms**, measured the same way
on HR as a control.

Reading the output
------------------
``confirmed_frac`` for HR is the calibration: two finders never agree perfectly,
and whatever HR scores is what "as good as real" looks like for this box, this
resolution and this matching tolerance. A tuned arm scoring near HR's rate has
made real objects. An arm scoring far below it has made catalogue entries.

The census numbers are reported alongside, per particle-count bin, because the
confirmation rate alone can be gamed in both directions: an arm that produces
almost nothing can confirm 100% of its handful, and an arm that shatters one
real clump into five reports five subhalos of which SUBFIND confirms one.

What is deliberately NOT done
-----------------------------
No attempt is made to reconcile the two mass definitions. Rockstar's ``mvir``
and SUBFIND's bound mass differ by construction, so every count here is on
**bound particle number**, which both finders mean the same thing by, and the
matching is on position with a tolerance that scales with the object's own size.

    python scripts/features/subfind_compare.py \\
        --box set9 --tag mgho_all_blocks_self_set9 \\
        --subfind $SUBFIND_ROOT/set9__mgho_all_blocks_self_set9/subfind_catalog.npz \\
        --hr-subfind $SUBFIND_ROOT/set9__hr/subfind_catalog.npz \\
        --host-id 168880 --out $SUBFIND_ROOT/compare/set9__self
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.eval.rockstar import HaloCatalog  # noqa: E402

from compare_gather_catalog import BOXSIZE, PBINS, find_catalogs  # noqa: E402


def periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (d + 0.5 * box) % box - 0.5 * box


class SubfindCat:
    """The satellite table of a parsed SUBFIND catalogue.

    Only satellites: ``SubhaloRankInGr == 0`` is the central of its FOF group,
    which is the *host*, not substructure. Comparing Rockstar's subhalos against
    a table that still contains centrals would confirm every host as a subhalo
    and inflate the agreement rate on both sides.
    """

    def __init__(self, path: str):
        z = np.load(path)
        self.path = str(path)
        sat = np.asarray(z["is_satellite"]) if "is_satellite" in z.files else None
        if sat is None:
            rank = np.asarray(z["SubhaloRankInGr"])
            sat = rank > 0
        self.all_len = np.asarray(z["SubhaloLen"]).astype(np.int64)
        self.all_pos = np.asarray(z["SubhaloPos"]).astype(np.float64)
        self.all_sat = np.asarray(sat).astype(bool)
        self.len = self.all_len[self.all_sat]
        self.pos = self.all_pos[self.all_sat]
        self.mass = (np.asarray(z["SubhaloMass"]).astype(np.float64)[self.all_sat]
                     if "SubhaloMass" in z.files else self.len.astype(np.float64))
        self.vmax = (np.asarray(z["SubhaloVmax"]).astype(np.float64)[self.all_sat]
                     if "SubhaloVmax" in z.files else np.zeros(self.len.size))
        self.rhalf = (np.asarray(z["SubhaloHalfmassRad"]).astype(np.float64)[self.all_sat]
                      if "SubhaloHalfmassRad" in z.files else np.zeros(self.len.size))

    @property
    def n(self) -> int:
        return int(self.len.size)

    def within(self, centre: Sequence[float], radius: float, min_p: int,
               box: float) -> np.ndarray:
        d = periodic_delta(self.pos, np.asarray(centre, float)[None, :], box)
        r = np.sqrt((d ** 2).sum(axis=1))
        return np.flatnonzero((r <= float(radius)) & (self.len >= int(min_p)))

    def centrals_near(self, centre: Sequence[float], box: float) -> Optional[int]:
        """Index (into the full table) of the central closest to ``centre``."""
        idx = np.flatnonzero(~self.all_sat)
        if idx.size == 0:
            return None
        d = periodic_delta(self.all_pos[idx], np.asarray(centre, float)[None, :], box)
        return int(idx[int(np.argmin((d ** 2).sum(axis=1)))])


def rockstar_subs_within(cat: HaloCatalog, centre: Sequence[float], radius: float,
                         min_p: int, box: float) -> np.ndarray:
    sub = cat.subhalos()
    d = periodic_delta(sub.pos, np.asarray(centre, float)[None, :], box)
    r = np.sqrt((d ** 2).sum(axis=1))
    keep = np.flatnonzero((r <= float(radius)) & (sub.num_p >= int(min_p)))
    return keep


def match(rs_pos: np.ndarray, rs_np: np.ndarray, rs_rvir_mpc: np.ndarray,
          sf_pos: np.ndarray, sf_np: np.ndarray, box: float, *,
          tol_mpc: float, rvir_mult: float) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one match, heaviest Rockstar object first.

    Heaviest-first matters: a small phantom sitting next to a real massive
    subhalo would otherwise be allowed to claim it and report the pair as
    confirmed, leaving the real object to fail. Each SUBFIND object can be
    claimed once.

    The tolerance scales with the Rockstar object's own ``R_vir`` (floored at
    ``tol_mpc``), because a 2000-particle subhalo and a 50-particle one are not
    located to the same precision by either finder.

    Returns ``(match_index, separation)`` per Rockstar object; ``-1`` where
    nothing matched.
    """
    n_rs = rs_pos.shape[0]
    out = np.full(n_rs, -1, dtype=np.int64)
    sep = np.full(n_rs, np.nan, dtype=np.float64)
    if n_rs == 0 or sf_pos.shape[0] == 0:
        return out, sep
    taken = np.zeros(sf_pos.shape[0], dtype=bool)
    order = np.argsort(-np.asarray(rs_np))
    for i in order.tolist():
        tol = max(float(tol_mpc), float(rvir_mult) * float(rs_rvir_mpc[i]))
        d = periodic_delta(sf_pos, rs_pos[i][None, :], box)
        r = np.sqrt((d ** 2).sum(axis=1))
        r[taken] = np.inf
        j = int(np.argmin(r))
        if r[j] <= tol:
            out[i] = j
            sep[i] = float(r[j])
            taken[j] = True
    return out, sep


def bin_counts(num_p: np.ndarray) -> Dict[str, int]:
    """Counts in the gate's own particle-number bins, so numbers line up."""
    num_p = np.asarray(num_p)
    out: Dict[str, int] = {}
    for lo, hi in PBINS:
        label = f"{lo}-{'inf' if hi >= 10 ** 9 else hi}p"
        out[label] = int(((num_p >= lo) & (num_p < hi)).sum())
    return out


def compare_one(name: str, rs: HaloCatalog, sf: SubfindCat, centre: np.ndarray,
                radius: float, min_p: int, box: float, *, tol_mpc: float,
                rvir_mult: float) -> Dict:
    """Both censuses and the cross-match, inside one host aperture."""
    rs_sub = rs.subhalos()
    keep = rockstar_subs_within(rs, centre, radius, min_p, box)
    sf_keep = sf.within(centre, radius, min_p, box)

    idx, sep = match(
        rs_sub.pos[keep], rs_sub.num_p[keep], rs_sub.rvir[keep] * 1e-3,
        sf.pos[sf_keep], sf.len[sf_keep], box,
        tol_mpc=tol_mpc, rvir_mult=rvir_mult)
    confirmed = idx >= 0
    n_rs, n_sf = int(keep.size), int(sf_keep.size)

    # Confirmation resolved by size: a finder disagreement concentrated at the
    # resolution limit means something different from one spread over every bin.
    per_bin: Dict[str, Dict[str, int]] = {}
    for lo, hi in PBINS:
        label = f"{lo}-{'inf' if hi >= 10 ** 9 else hi}p"
        m = (rs_sub.num_p[keep] >= lo) & (rs_sub.num_p[keep] < hi)
        # NOTE both entries are binned by the ROCKSTAR object's particle
        # count, so "confirmed" is a subset of "rockstar" and the two are
        # comparable. Do NOT read them against ``subfind_bins``, which is
        # binned by the SUBFIND object's own count: a matched pair routinely
        # lands in different bins there, which makes confirmed look larger
        # than the SUBFIND count in the same row.
        per_bin[label] = {"rockstar": int(m.sum()),
                          "confirmed": int((m & confirmed).sum()),
                          "binned_by": "rockstar num_p"}

    return {
        "name": name,
        "aperture_mpc_h": float(radius),
        "min_particles": int(min_p),
        "rockstar_subhalos": n_rs,
        "subfind_subhalos": n_sf,
        "subfind_over_rockstar": (n_sf / n_rs) if n_rs else None,
        "confirmed": int(confirmed.sum()),
        "confirmed_frac": float(confirmed.mean()) if n_rs else None,
        "unmatched_rockstar": int((~confirmed).sum()),
        "unmatched_subfind": int(n_sf - confirmed.sum()),
        "median_separation_mpc_h": (float(np.nanmedian(sep[confirmed]))
                                    if confirmed.any() else None),
        "rockstar_bins": bin_counts(rs_sub.num_p[keep]),
        "subfind_bins": bin_counts(sf.len[sf_keep]),
        "bins_note": ("rockstar_bins and confirmed_by_bin are binned by the "
                      "Rockstar object's particle count; subfind_bins by the "
                      "SUBFIND object's own. They are not row-comparable."),
        "confirmed_by_bin": per_bin,
    }


def whole_box(name: str, rs: HaloCatalog, sf: SubfindCat, min_p: int) -> Dict:
    """The box-wide census. A sanity check on the run, not the headline.

    Only a few percent of the box is spliced, so a large whole-box difference
    between the two finders is a statement about the finders (or about the
    snapshot), not about the fine-tune.
    """
    rs_sub = rs.subhalos()
    rs_keep = rs_sub.num_p >= int(min_p)
    sf_keep = sf.len >= int(min_p)
    return {
        "name": name,
        "rockstar_subhalos": int(rs_keep.sum()),
        "subfind_subhalos": int(sf_keep.sum()),
        "rockstar_bins": bin_counts(rs_sub.num_p[rs_keep]),
        "subfind_bins": bin_counts(sf.len[sf_keep]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", default="set9")
    ap.add_argument("--tag", required=True,
                    help="Rockstar candidate tag, e.g. mgho_all_blocks_self_set9")
    ap.add_argument("--subfind", required=True,
                    help="subfind_catalog.npz of the candidate field")
    ap.add_argument("--hr-subfind", default="",
                    help="subfind_catalog.npz of HR -- the control that calibrates "
                         "what 'as good as real' means")
    ap.add_argument("--base-subfind", default="",
                    help="subfind_catalog.npz of the frozen SR2 field")
    ap.add_argument("--host-id", type=int, default=168880,
                    help="HR halo id of the trained host")
    ap.add_argument("--radius-mult", type=float, default=1.0,
                    help="aperture in units of the host's HR R_vir")
    ap.add_argument("--min-p", type=int, default=50)
    ap.add_argument("--match-tol-mpc", type=float, default=0.05,
                    help="floor on the match tolerance, Mpc/h")
    ap.add_argument("--match-rvir-mult", type=float, default=1.0,
                    help="match tolerance in units of the Rockstar object's R_vir")
    ap.add_argument("--out", required=True, help="output stem (.json)")
    args = ap.parse_args(argv)

    cats = find_catalogs(args.box, args.tag)
    for need in ("hr", "cand"):
        if need not in cats:
            raise SystemExit(
                f"no Rockstar '{need}' catalog for box {args.box} tag {args.tag}; "
                "the SUBFIND cross-check compares against it, so it must exist")

    hr = cats["hr"]
    host = np.flatnonzero(hr.ids == int(args.host_id))
    if host.size == 0:
        raise SystemExit(f"host {args.host_id} is not in the HR catalog")
    h = int(host[0])
    centre = hr.pos[h].astype(np.float64)
    rvir = float(hr.rvir[h]) * 1e-3          # Rockstar writes R_vir in kpc/h
    radius = float(args.radius_mult) * rvir
    print(f"=== host {args.host_id}: log M = {np.log10(hr.mvir[h]):.3f}, "
          f"R_vir = {rvir:.3f} Mpc/h, aperture {radius:.3f} Mpc/h at {centre}",
          flush=True)

    entries: List[Tuple[str, str, str]] = [("cand", args.tag, args.subfind)]
    if args.hr_subfind:
        entries.append(("hr", "hr", args.hr_subfind))
    if args.base_subfind and "base" in cats:
        entries.append(("base", "base", args.base_subfind))

    host_rows, box_rows = [], []
    for key, label, npz in entries:
        if not Path(npz).is_file():
            print(f"!!! {label}: {npz} missing, skipped", flush=True)
            continue
        sf = SubfindCat(npz)
        rs = cats[key]
        print(f"=== {label}: Rockstar {rs.subhalos().n} subhalos, "
              f"SUBFIND {sf.n} satellites (whole box)", flush=True)
        host_rows.append(compare_one(label, rs, sf, centre, radius, args.min_p,
                                     BOXSIZE, tol_mpc=args.match_tol_mpc,
                                     rvir_mult=args.match_rvir_mult))
        box_rows.append(whole_box(label, rs, sf, args.min_p))

    result = {
        "box": args.box, "tag": args.tag, "host_id": int(args.host_id),
        "host_centre_mpc_h": centre.tolist(),
        "host_rvir_mpc_h": rvir, "aperture_mpc_h": radius,
        "min_particles": int(args.min_p),
        "match": {"tol_mpc_h": args.match_tol_mpc,
                  "rvir_mult": args.match_rvir_mult},
        "within_host": host_rows,
        "whole_box": box_rows,
    }

    # The verdict, stated rather than left to be read out of the table: does the
    # candidate confirm at HR's rate?
    by_name = {r["name"]: r for r in host_rows}
    if "hr" in by_name and by_name["hr"]["confirmed_frac"] is not None \
            and by_name[args.tag]["confirmed_frac"] is not None:
        hr_frac = by_name["hr"]["confirmed_frac"]
        cand_frac = by_name[args.tag]["confirmed_frac"]
        result["verdict"] = {
            "hr_confirmed_frac": hr_frac,
            "cand_confirmed_frac": cand_frac,
            "relative": (cand_frac / hr_frac) if hr_frac else None,
            "reading": ("candidate subhalos are confirmed at HR's rate: the "
                        "objects are real and the projection is what hides them"
                        if hr_frac and cand_frac >= 0.8 * hr_frac else
                        "candidate subhalos are confirmed far below HR's rate: "
                        "Rockstar is reporting objects SUBFIND does not see"),
        }

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = stem.with_name(stem.name + ".json") if not str(stem).endswith(".json") \
        else stem
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"=== wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
