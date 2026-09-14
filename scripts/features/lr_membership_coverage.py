#!/usr/bin/env python
"""Is the LR host channel non-null where the loss needs to move material?

The question this settles, and why it is the last one owed. Conditioning the SR
operator on a pre-run LR halo catalog can only help at Lagrangian sites where
that catalog says something. Rockstar membership is defined where a halo's
particles ALREADY are -- but HR's subhalos peak at **1.0-1.5 R_vir**, at or past
the edge of the LR halo's member set. If the map reads null out there, the
channel is present at the host centre and absent in the very shell the radial
profile is wrong in, and the fix is a DERIVED ``r/R_vir`` field over 2-3 R_vir
rather than strict membership. Same information, no hole.

Done in the right space. The SR operator consumes a Lagrangian tile, so the
channel is indexed by ``particle_id`` -- the Lagrangian lattice site -- not by
Eulerian position. So:

  * ``<box>_hr_owner.npy`` (512^3 int32) gives, per HR Lagrangian site, the HR
    halo that owns it.
  * the LR ``halos_*.particles`` table gives, per LR ``particle_id`` (a 64^3
    Lagrangian site), the LR halo it was assigned to; resolved here to that
    halo's TOP-LEVEL ancestor, because "which host does this material belong
    to" is the channel's question and a particle inside an LR subhalo still
    belongs to the cluster.
  * an HR site maps to its LR site by integer division (512/64 = 8) on each
    axis.

For every HR subhalo we then ask what fraction of its own Lagrangian sites land
on an LR site that (a) has any assignment at all, and (b) has the assignment
matching its true HR parent's LR counterpart. Reported against **r/R_vir**,
which is the axis the verdict lives on.

CPU-only and submitted: one pass over 1.3e8 owner entries, chunked.

Writes, under ``--out-dir``:
  * ``<box>_membership.jsonl`` -- one row per HR subhalo
  * ``<box>_membership.npz``   -- the same as arrays, for re-plots
  * ``membership_summary.json``-- the coverage-vs-radius verdict table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from gather_holdout_figures_data import BOX, catalog_path  # noqa: E402
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from lr_host_fidelity import nearest, periodic_delta  # noqa: E402

NG_HR = 512
NG_LR = 64
#: The axis the verdict lives on: HR's subhalos peak at 1.0-1.5.
RVIR_BINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf])


def rvir_bin_labels() -> List[str]:
    out = []
    for lo, hi in zip(RVIR_BINS[:-1], RVIR_BINS[1:]):
        out.append(f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}")
    return out


# --------------------------------------------------------------------------- #
# The LR side: Lagrangian site -> top-level LR host
# --------------------------------------------------------------------------- #
def top_level_of(cat) -> Dict[int, int]:
    """``halo id -> its top-level ancestor's id``, walking ``parent_ids``.

    A particle inside an LR subhalo still belongs to the cluster as far as the
    channel is concerned, so substructure is resolved upward rather than
    dropped. Cycles cannot occur in a Rockstar catalog but the walk is capped
    anyway -- a malformed parent chain should not hang the job.
    """
    row_of = {int(i): r for r, i in enumerate(cat.ids)}
    out: Dict[int, int] = {}
    for r, hid in enumerate(cat.ids):
        cur, seen = r, 0
        while int(cat.parent_ids[cur]) >= 0 and seen < 64:
            nxt = row_of.get(int(cat.parent_ids[cur]))
            if nxt is None or nxt == cur:
                break
            cur, seen = nxt, seen + 1
        out[int(hid)] = int(cat.ids[cur])
    return out


def lr_site_host(particles_path: Path, cat) -> Tuple[np.ndarray, Dict]:
    """``(int64[64^3] of top-level LR host id or -1, stats)``.

    Parses the interleaved halo/particle tables by field count: a particle row
    has 10 fields, a halo row 20. Only ``external_haloid`` is usable against
    the ASCII catalog's ids, and the file's own header warns it can be absent
    (-1) for halos that were not printed; those rows are counted and dropped
    rather than silently mapped to halo -1.
    """
    site = np.full(NG_LR ** 3, -1, dtype=np.int64)
    tl = top_level_of(cat)
    known = set(int(i) for i in cat.ids)
    n_rows = n_noext = n_unknown = 0
    with open(particles_path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.split()
            if len(f) != 10:
                continue                      # a halo row, not a particle row
            n_rows += 1
            ext = int(f[9])
            if ext < 0:
                n_noext += 1
                continue
            if ext not in known:
                n_unknown += 1
                continue
            pid = int(f[6])
            if 0 <= pid < site.size:
                site[pid] = tl.get(ext, ext)
    stats = {
        "particle_rows": n_rows,
        "rows_without_external_id": n_noext,
        "rows_with_unknown_external_id": n_unknown,
        "assigned_lr_sites": int((site >= 0).sum()),
        "total_lr_sites": int(site.size),
    }
    stats["assigned_fraction"] = stats["assigned_lr_sites"] / site.size
    return site, stats


def hr_site_to_lr_site(idx: np.ndarray) -> np.ndarray:
    """Flat HR Lagrangian index -> flat LR Lagrangian index (C order, /8)."""
    f = NG_HR // NG_LR
    cz = idx % NG_HR
    cy = (idx // NG_HR) % NG_HR
    cx = idx // (NG_HR * NG_HR)
    return (cx // f) * (NG_LR * NG_LR) + (cy // f) * NG_LR + (cz // f)


# --------------------------------------------------------------------------- #
# The pass over the owner array
# --------------------------------------------------------------------------- #
def accumulate(owner_path: Path, site_host: np.ndarray, hr_ids: np.ndarray,
               lr_host_of_row: np.ndarray, chunk: int = 8_000_000,
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per HR halo row: ``(n_sites, n_any_assignment, n_correct_host)``.

    One chunked pass over the 512^3 owner array. ``hr_ids`` must be sorted;
    rows are recovered with ``searchsorted`` and a validity check, so an owner
    id absent from the catalog is dropped rather than aliased onto a neighbour.
    """
    owner = np.load(owner_path, mmap_mode="r")
    n_row = hr_ids.size
    n_sites = np.zeros(n_row, dtype=np.int64)
    n_any = np.zeros(n_row, dtype=np.int64)
    n_ok = np.zeros(n_row, dtype=np.int64)
    for s in range(0, owner.shape[0], chunk):
        o = np.asarray(owner[s:s + chunk])
        m = o >= 0
        if not m.any():
            continue
        ids = o[m].astype(np.int64)
        pos = np.searchsorted(hr_ids, ids)
        pos = np.clip(pos, 0, n_row - 1)
        good = hr_ids[pos] == ids
        if not good.any():
            continue
        rows = pos[good]
        sites = (np.flatnonzero(m) + s)[good]
        lr = site_host[hr_site_to_lr_site(sites)]
        n_sites += np.bincount(rows, minlength=n_row)
        n_any += np.bincount(rows, weights=(lr >= 0).astype(np.float64),
                             minlength=n_row).astype(np.int64)
        want = lr_host_of_row[rows]
        n_ok += np.bincount(rows,
                            weights=((lr >= 0) & (want >= 0) & (lr == want)
                                     ).astype(np.float64),
                            minlength=n_row).astype(np.int64)
    return n_sites, n_any, n_ok


def run_box(box: str, args, out_dir: Path) -> Dict:
    print(f"\n=== {box}")
    hr = load_rockstar_ascii(catalog_path(Path(args.reward_root), "hr", box))
    lr_dir = Path(args.lr_work_root) / box / f"{box}_lr_rockstar"
    lr_ascii = sorted(lr_dir.glob("halos_*.ascii"))
    lr_parts = sorted(lr_dir.glob("halos_*.particles"))
    if not lr_ascii or not lr_parts:
        raise SystemExit(f"{box}: need both halos_*.ascii and halos_*.particles "
                         f"under {lr_dir}; re-run with LHF_RERUN_ROCKSTAR=1")
    lr = load_rockstar_ascii(lr_ascii[0])
    site_host, pstats = lr_site_host(lr_parts[0], lr)
    print(f"  LR membership: {pstats['assigned_lr_sites']} of "
          f"{pstats['total_lr_sites']} sites assigned "
          f"({pstats['assigned_fraction']*100:.1f}%), "
          f"{pstats['rows_without_external_id']} rows had no external id")

    # HR hosts -> their LR counterpart, only where the match is real.
    hr_host = np.flatnonzero((hr.parent_ids < 0)
                            & (hr.num_p >= args.hr_min_host_num_p))
    lr_top = np.flatnonzero(lr.parent_ids < 0)
    hpos = np.asarray(hr.pos[hr_host], dtype=np.float64)
    hrvir = np.asarray(hr.rvir[hr_host], dtype=np.float64) / 1000.0
    j, dr = nearest(hpos, np.asarray(lr.pos[lr_top], dtype=np.float64), BOX)
    matched = dr <= args.match_rvir * hrvir
    lr_id_of_host = np.where(matched, np.asarray(lr.ids[lr_top])[j], -1)
    print(f"  HR hosts >={args.hr_min_host_num_p}p: {hr_host.size}, "
          f"{int(matched.sum())} with an LR counterpart within "
          f"{args.match_rvir} R_vir")

    # Per HR halo ROW, the LR host id its material should be assigned to. A
    # subhalo inherits its PARENT's counterpart -- the channel's job is to say
    # "you belong to that cluster", not to resolve substructure LR cannot see.
    lr_want = np.full(hr.n, -1, dtype=np.int64)
    lr_want[hr_host] = lr_id_of_host
    sub = np.flatnonzero((hr.parent_ids >= 0) & (hr.num_p >= args.sub_min_num_p))
    row_of_id = {int(v): i for i, v in enumerate(hr.ids)}
    parent_row = np.array([row_of_id.get(int(p), -1)
                           for p in hr.parent_ids[sub]], dtype=np.int64)
    keep = parent_row >= 0
    sub, parent_row = sub[keep], parent_row[keep]
    lr_want[sub] = lr_want[parent_row]

    order = np.argsort(hr.ids)
    hr_ids_sorted = np.asarray(hr.ids)[order].astype(np.int64)
    lr_want_sorted = lr_want[order]
    print(f"  pass over {NG_HR**3} owner entries...")
    n_sites_s, n_any_s, n_ok_s = accumulate(
        Path(args.reward_root) / "halos_particles" / f"{box}__hr__hr"
        / f"{box}_hr_owner.npy",
        site_host, hr_ids_sorted, lr_want_sorted, chunk=args.chunk)
    # back to catalog row order
    n_sites = np.zeros(hr.n, dtype=np.int64); n_sites[order] = n_sites_s
    n_any = np.zeros(hr.n, dtype=np.int64); n_any[order] = n_any_s
    n_ok = np.zeros(hr.n, dtype=np.int64); n_ok[order] = n_ok_s

    # subhalo geometry
    r = np.sqrt((periodic_delta(np.asarray(hr.pos[sub], dtype=np.float64),
                                np.asarray(hr.pos[parent_row],
                                           dtype=np.float64), BOX) ** 2
                 ).sum(axis=1))
    prvir = np.asarray(hr.rvir[parent_row], dtype=np.float64) / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r_over = np.where(prvir > 0, r / prvir, np.nan)
    host_np = np.asarray(hr.num_p[parent_row], dtype=np.int64)
    ns, na, nk = n_sites[sub], n_any[sub], n_ok[sub]
    with np.errstate(divide="ignore", invalid="ignore"):
        f_any = np.where(ns > 0, na / np.maximum(ns, 1), np.nan)
        f_ok = np.where(ns > 0, nk / np.maximum(ns, 1), np.nan)
    # only subhalos whose host HAS a counterpart -- otherwise the question is moot
    live = lr_want[sub] >= 0
    print(f"  {sub.size} HR subhalos >={args.sub_min_num_p}p, "
          f"{int(live.sum())} in hosts with an LR counterpart")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{box}_membership.npz",
        num_p=np.asarray(hr.num_p[sub]), r_mpc=r, r_over_rvir=r_over,
        host_num_p=host_np, n_sites=ns, n_any=na, n_correct=nk,
        frac_any=f_any, frac_correct=f_ok, host_has_lr=live,
    )
    with open(out_dir / f"{box}_membership.jsonl", "w") as fh:
        for k in range(sub.size):
            fh.write(json.dumps({
                "hr_id": int(hr.ids[sub[k]]), "num_p": int(hr.num_p[sub[k]]),
                "r_mpc": float(r[k]), "r_over_rvir": float(r_over[k]),
                "host_num_p": int(host_np[k]), "n_sites": int(ns[k]),
                "frac_any": float(f_any[k]), "frac_correct": float(f_ok[k]),
                "host_has_lr": bool(live[k]),
            }) + "\n")

    # --- the verdict: coverage against r/R_vir ----------------------------- #
    labels = rvir_bin_labels()
    bins = []
    for k, (lo, hi) in enumerate(zip(RVIR_BINS[:-1], RVIR_BINS[1:])):
        m = live & np.isfinite(r_over) & (r_over >= lo) & (r_over < hi) & (ns > 0)
        if not m.any():
            bins.append({"bin": labels[k], "n_sub": 0})
            continue
        fa, fo = float(np.median(f_any[m])), float(np.median(f_ok[m]))
        bins.append({
            "bin": labels[k], "n_sub": int(m.sum()),
            "median_frac_any": fa, "median_frac_correct": fo,
            "frac_sub_mostly_covered": float(np.mean(f_ok[m] > 0.5)),
            "verdict": ("COVERED" if fo > 0.5 else
                        "PARTIAL" if fo > 0.1 else "NULL"),
        })
    print(f"\n  {'r/R_vir':10s} {'n_sub':>7s} {'any':>7s} {'correct':>8s} "
          f"{'>50% cov':>9s}  verdict")
    for b in bins:
        if not b.get("n_sub"):
            continue
        print(f"  {b['bin']:10s} {b['n_sub']:7d} {b['median_frac_any']:7.3f} "
              f"{b['median_frac_correct']:8.3f} "
              f"{b['frac_sub_mostly_covered']:9.3f}  {b['verdict']}")
    return {"box": box, "lr_membership": pstats,
            "n_hr_hosts": int(hr_host.size), "n_matched_hosts": int(matched.sum()),
            "n_subhalos": int(sub.size), "n_subhalos_live": int(live.sum()),
            "sub_min_num_p": int(args.sub_min_num_p),
            "rvir_bins": bins}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", default="set9")
    ap.add_argument("--reward-root", required=True)
    ap.add_argument("--lr-work-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hr-min-host-num-p", type=int, default=200)
    ap.add_argument("--sub-min-num-p", type=int, default=20,
                    help="include subhalos this small; 20 reaches below the "
                         "50p Rockstar floor and the 200p supervision floor")
    ap.add_argument("--match-rvir", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=8_000_000)
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    per_box = [run_box(b, args, out_dir) for b in boxes]
    write_json(out_dir / "membership_summary.json",
               {"ok": True, "boxes": boxes, "per_box": per_box})
    print(f"\n=== wrote {out_dir}/membership_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
