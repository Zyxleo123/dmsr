#!/usr/bin/env python
"""Audit: how many *independent hosts* does each host-mass bin actually have?

The occupation reward is a per-host-mass-bin mean, ``<N_sub | M_host,i>``. Its
denominator is a host count, so a bin with five hosts in an ensemble is not a
measurement -- it is noise wearing a bin label. This script answers, for every
box with an HR catalog and for every host-mass bin:

* hosts per independent box (whole box, and after chunk attribution);
* hosts per ``B``-chunk reward ensemble (the unit the reward actually scores);
* the percentage of ensembles in which the bin is empty;
* the box-bootstrap uncertainty of the occupation value in that bin;
* whether chunk attribution can double-count a host.

The last one is a structural check, not a statistic: Lagrangian chunks tile the
box disjointly and ``assign_halos_to_chunks`` returns at most one chunk id per
halo, so duplication is impossible by construction -- but the script verifies it
rather than asserting it, because "chunks might overlap" is exactly the kind of
assumption that is cheap to test and expensive to be wrong about.

**Multiple SR2 seeds of one box are not independent hosts** and are never read
here: only ``source == "hr"`` catalogs enter.

The gate this feeds: a host-mass bin is usable in the reward if it has roughly
20-30 *effective* hosts per reward estimate and few empty ensembles. Read the
verdict from the bootstrap width, not from the pooled total.

    python scripts/reward/audit_host_counts.py                 # cached summaries
    python scripts/reward/audit_host_counts.py --from-catalogs # attribute first
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from _common import (add_common_args, banner, bins_of, chunk_grid,
                     load_reward_config, write_json)

from cosmo_sr.eval.rockstar import load_rockstar_ascii
from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import ChunkSummary, pool, read_summaries, summarize_catalog
from cosmo_sr.reward.geometry import (ChunkGrid, assign_halos_to_chunks,
                                      chunk_purity_grid)

STAGE1 = Path("/zfsauton/scratch/yixiz/DMSR/sr2_baseline/stage1/halos")


def _hr_catalog_path(box: str) -> Path:
    """Where this box's HR Rockstar catalog lives.

    Prefer the one ``hr_catalog_summaries_cpu`` produced under the reward root
    (all 16 boxes); fall back to the old stage1 tree, which only ever held
    ``set12``. Returning the stage1 path unconditionally is what limited a
    16-box audit to a single box.
    """
    new = paths.subdir("halos") / f"{box}__hr__hr" / "hr_rockstar" / "halos_0.0.ascii"
    if new.is_file():
        return new
    return STAGE1 / box / "hr" / "hr_rockstar" / "halos_0.0.ascii"


def sweep_geometry(box: str, cfg: Dict, bins, specs) -> List[Dict]:
    """Host retention per mass bin as a function of ``(chunk_hr, min_purity)``.

    The core mask rejects a halo whose ``Rvir`` neighbourhood is not cleanly
    owned by one Lagrangian chunk. Massive hosts accrete from a large Lagrangian
    volume, so they are the *most* likely to straddle a chunk boundary -- the
    rejection is mass-dependent, and it bites hardest in exactly the host bins
    the occupation reward is about. This measures how hard, and whether a wider
    chunk or a looser purity threshold buys the bins back.
    """
    from dataclasses import replace

    d = cfg["data"]
    field = np.load(Path(d["root"]) / "hr" / f"{box}.npy", mmap_mode="r")
    disp = np.asarray(field[0:3])
    cat = load_rockstar_ascii(_hr_catalog_path(box))
    edges = np.asarray(bins.host_mass_edges)
    is_host = np.asarray(cat.parent_ids) < 0
    resolved = np.asarray(cat.num_p) >= bins.min_host_particles
    hostm = np.asarray(cat.mvir)[is_host & resolved]
    n_whole, _ = np.histogram(hostm, bins=edges)

    out = []
    by_chunk_hr: Dict[int, object] = {}
    for chunk_hr, min_purity, max_hw in specs:
        if chunk_hr not in by_chunk_hr:
            g = ChunkGrid(ng_hr=int(d.get("ng_hr", 512)), chunk_hr=int(chunk_hr),
                          boxsize_mpc_h=float(d["boxsize_mpc_h"]))
            by_chunk_hr[chunk_hr] = (g, chunk_purity_grid(
                disp, chunk_grid=g,
                grid=int(cfg.get("geometry", {}).get("purity_grid", 128)),
                dis_norm_kpc_h=float(d.get("dis_norm_kpc_h", 6000.0)),
                redshift=float(d.get("redshift", 0.0))))
        g, purity = by_chunk_hr[chunk_hr]
        assign = assign_halos_to_chunks(
            cat.pos, cat.rvir, purity, min_purity=float(min_purity),
            radius_mult=float(bins.radius_mult), max_half_width=int(max_hw))
        b2 = replace(bins, min_purity=float(min_purity))
        vols = purity.effective_volume_mpc3(float(min_purity))
        summ = summarize_catalog(cat, assign, b2, vols, box=box, source="hr",
                                 chunk_ids=g.all_ids())
        ens = pool([summ[c] for c in sorted(summ)])
        n_att = np.asarray(ens.n_host)
        out.append({
            "chunk_hr": int(chunk_hr),
            "chunk_mpc_h": float(g.chunk_mpc_h),
            "n_chunks": int(g.n_chunks),
            "min_purity": float(min_purity),
            "max_half_width": int(max_hw),
            "hosts_whole_box": n_whole.astype(int).tolist(),
            "hosts_attributed": n_att.astype(int).tolist(),
            "retention": [float(a / w) if w > 0 else None
                          for a, w in zip(n_att, n_whole)],
            "hosts_per_ensemble_B16": (n_att * 16.0 / g.n_chunks).tolist(),
            "core_volume_fraction": float(
                ens.volume_mpc3 / float(d["boxsize_mpc_h"]) ** 3),
            "occupation": [None if not np.isfinite(v) else float(v)
                           for v in ens.occupation()],
        })
        print(f"    chunk_hr={chunk_hr:4d} ({g.chunk_mpc_h:5.1f} Mpc/h, "
              f"{g.n_chunks:3d} chunks) purity={min_purity:.2f} hw={max_hw}: "
              f"retention="
              + " ".join("  -  " if r is None else f"{r:5.2f}"
                         for r in out[-1]["retention"]), flush=True)
    return out


def summaries_from_catalog(box: str, cfg: Dict, grid, bins) -> List[ChunkSummary]:
    """Attribute one HR box's halos to Lagrangian chunks, in-process.

    Only the HR displacement field and the existing HR catalog are read; the
    halo finder is never re-run, so this is minutes of CPU rather than hours.
    """
    cat_path = _hr_catalog_path(box)
    if not cat_path.is_file():
        raise FileNotFoundError(f"no HR catalog for {box} at {cat_path}")
    d = cfg["data"]
    field_path = Path(d["root"]) / "hr" / f"{box}.npy"
    field = np.load(field_path, mmap_mode="r")
    cat = load_rockstar_ascii(cat_path)
    purity = chunk_purity_grid(
        np.asarray(field[0:3]), chunk_grid=grid,
        grid=int(cfg.get("geometry", {}).get("purity_grid", 128)),
        dis_norm_kpc_h=float(d.get("dis_norm_kpc_h", 6000.0)),
        redshift=float(d.get("redshift", 0.0)),
    )
    assign = assign_halos_to_chunks(
        cat.pos, cat.rvir, purity,
        min_purity=float(bins.min_purity), radius_mult=float(bins.radius_mult),
        max_half_width=int(cfg.get("geometry", {}).get("max_half_width", 4)),
    )
    volumes = purity.effective_volume_mpc3(bins.min_purity)
    summ = summarize_catalog(cat, assign, bins, volumes, box=box, source="hr",
                             chunk_ids=grid.all_ids())
    # --- duplication check ---------------------------------------------------
    # assign is one id per halo, so a halo cannot land in two chunks. Confirm
    # that the per-chunk host counts sum to exactly the number of assigned,
    # resolved hosts: any excess would mean double counting.
    is_host = np.asarray(cat.parent_ids) < 0
    resolved = np.asarray(cat.num_p) >= bins.min_host_particles
    n_assigned_hosts = int(np.count_nonzero(is_host & resolved & (assign >= 0)))
    n_summed = int(sum(int(np.sum(s.n_host)) for s in summ.values()))
    n_in_bins = int(np.count_nonzero(
        is_host & resolved & (assign >= 0)
        & (np.asarray(cat.mvir) >= bins.host_mass_edges[0])
        & (np.asarray(cat.mvir) < bins.host_mass_edges[-1])
    ))
    dup = {
        "box": box,
        "n_halos_total": int(cat.n),
        "n_hosts_resolved": int(np.count_nonzero(is_host & resolved)),
        "n_hosts_assigned": n_assigned_hosts,
        "n_hosts_in_mass_range": n_in_bins,
        "n_hosts_summed_over_chunks": n_summed,
        "duplicated": int(n_summed - n_in_bins),
        "boundary_rejected_fraction": float(
            1.0 - n_assigned_hosts / max(np.count_nonzero(is_host & resolved), 1)
        ),
    }
    out = [summ[c] for c in sorted(summ)]
    for s in out:
        s.meta["duplication_check"] = dup
    return out


def _ensembles(chunks: List[ChunkSummary], B: int, n_draws: int, seed: int,
               box_bootstrap: bool) -> List[List[int]]:
    """Draw ``n_draws`` ensembles of ``B`` chunks.

    With ``box_bootstrap`` the boxes are resampled with replacement first --
    the independent cosmological unit is the box, never the chunk and never the
    SR2 seed.
    """
    rng = np.random.default_rng(seed)
    by_box: Dict[str, List[int]] = {}
    for i, c in enumerate(chunks):
        if c.volume_mpc3 > 0:
            by_box.setdefault(c.box, []).append(i)
    boxes = sorted(by_box)
    if not boxes:
        raise SystemExit("no chunks with positive core volume")
    draws = []
    for _ in range(n_draws):
        pick = list(rng.choice(boxes, size=len(boxes), replace=True)) \
            if box_bootstrap else boxes
        avail = [i for b in pick for i in by_box[b]]
        draws.append(list(rng.choice(avail, size=B, replace=True)))
    return draws


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--boxes", default=None,
                    help="comma-separated; default = every box with an HR catalog")
    ap.add_argument("--from-catalogs", action="store_true",
                    help="attribute halos to chunks now instead of reading "
                         "cached summaries from the Slurm array job")
    ap.add_argument("--ensemble-size", type=int, default=None)
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-hosts", type=float, default=20.0,
                    help="effective hosts per reward estimate required to use "
                         "a bin in the reward")
    ap.add_argument("--max-empty-frac", type=float, default=0.05)
    ap.add_argument("--sweep-geometry", default=None,
                    help="comma-separated chunk_hr:min_purity[:max_half_width] "
                         "specs, e.g. 128:0.8,256:0.8,256:0.6 -- reports host "
                         "retention per mass bin for each")
    ap.add_argument("--sweep-box", default="set12")
    ap.add_argument("--out", default="runs/reward/host_count_audit.json")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    grid = chunk_grid(cfg)
    B = int(args.ensemble_size or cfg.get("reward", {}).get("ensemble_size_B", 16))
    labels = [f"{bins.host_mass_edges[i]:.2e}" for i in range(bins.n_host_bins)]

    split = cfg.get("split", {})
    role = {}
    for k, v in split.items():
        for b in (v or []):
            role[b] = k.replace("_boxes", "")

    if args.boxes:
        boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    else:
        # A box is usable if EITHER its chunk summaries are already cached (the
        # normal path, produced by hr_catalog_summaries_cpu) OR a raw Rockstar
        # ASCII catalog exists to attribute on the fly. Looking only for the
        # ASCII silently reduced a 16-box audit to the one box that happens to
        # keep its catalog under the old stage1 tree.
        boxes = sorted({b for b in role}, key=lambda s: (len(s), s))
        boxes = [
            b for b in boxes
            if (paths.CATALOG_CACHE() / f"{b}__hr__hr.jsonl").is_file()
            or _hr_catalog_path(b).is_file()
        ]
    if not boxes:
        raise SystemExit(
            "no box has HR chunk summaries or an HR catalog. Submit "
            "`sbatch --array=0-15 scripts/slurm/hr_catalog_summaries_cpu.sbatch` "
            "first; this audit reads its output."
        )
    missing = [b for b in boxes
               if not (paths.CATALOG_CACHE() / f"{b}__hr__hr.jsonl").is_file()]
    if missing and not args.from_catalogs:
        print(f"  note: {missing} have no cached summaries; attributing on the fly",
              flush=True)

    sweep = None
    if args.sweep_geometry:
        specs = []
        for tok in args.sweep_geometry.split(","):
            parts = tok.strip().split(":")
            specs.append((int(parts[0]), float(parts[1]),
                          int(parts[2]) if len(parts) > 2
                          else int(cfg.get("geometry", {}).get("max_half_width", 4))))
        banner(f"Geometry sweep on {args.sweep_box}")
        sweep = sweep_geometry(args.sweep_box, cfg, bins, specs)

    chunks: List[ChunkSummary] = []
    per_box = {}
    dup_checks = []
    for b in boxes:
        cached = paths.CATALOG_CACHE() / f"{b}__hr__hr.jsonl"
        if cached.is_file() and not args.from_catalogs:
            cs = read_summaries(cached)
        else:
            print(f"  attributing {b} ...", flush=True)
            cs = summaries_from_catalog(b, cfg, grid, bins)
        chunks.extend(cs)
        ens = pool(cs)
        per_box[b] = {
            "role": role.get(b, "unknown"),
            "n_chunks": len(cs),
            "n_chunks_with_volume": int(sum(1 for c in cs if c.volume_mpc3 > 0)),
            "hosts_per_bin_attributed": np.asarray(ens.n_host).astype(int).tolist(),
            "occupation": [None if not np.isfinite(v) else float(v)
                           for v in ens.occupation()],
            "core_volume_mpc3": float(ens.volume_mpc3),
            "core_volume_fraction": float(
                ens.volume_mpc3 / float(cfg["data"]["boxsize_mpc_h"]) ** 3),
        }
        if cs and "duplication_check" in cs[0].meta:
            dup_checks.append(cs[0].meta["duplication_check"])
        # whole-box counts, unattributed: the ceiling the attribution starts from
        p = _hr_catalog_path(b)
        if p.is_file():
            cat = load_rockstar_ascii(p)
            hostm = np.asarray(cat.mvir)[
                (np.asarray(cat.parent_ids) < 0)
                & (np.asarray(cat.num_p) >= bins.min_host_particles)]
            n_whole, _ = np.histogram(hostm, bins=np.asarray(bins.host_mass_edges))
            per_box[b]["hosts_per_bin_whole_box"] = n_whole.astype(int).tolist()
            per_box[b]["attribution_retention"] = [
                float(a / w) if w > 0 else None
                for a, w in zip(per_box[b]["hosts_per_bin_attributed"], n_whole)
            ]

    n_boxes = len({c.box for c in chunks})
    draws = _ensembles(chunks, B, args.n_draws, args.seed, box_bootstrap=n_boxes > 1)

    n_host_draw = np.zeros((len(draws), bins.n_host_bins))
    occ_draw = np.full((len(draws), bins.n_host_bins), np.nan)
    for k, idx in enumerate(draws):
        e = pool([chunks[i] for i in idx])
        n_host_draw[k] = np.asarray(e.n_host)
        occ_draw[k] = np.asarray(e.occupation())

    report_bins = []
    for i, lab in enumerate(labels):
        h = n_host_draw[:, i]
        o = occ_draw[:, i]
        finite = np.isfinite(o)
        empty_frac = float(np.mean(h <= 0))
        # "Effective hosts" is the harmonic-style lower quantile rather than the
        # mean: the reward is evaluated on individual ensembles, so the bad
        # draws are what decides whether a bin is usable.
        eff = float(np.percentile(h, 10))
        boot = (float(np.nanpercentile(o, 2.5)), float(np.nanpercentile(o, 97.5))) \
            if finite.any() else (float("nan"), float("nan"))
        med = float(np.nanmedian(o)) if finite.any() else float("nan")
        rel_width = float((boot[1] - boot[0]) / med) if med and np.isfinite(med) else float("nan")
        usable = bool(eff >= args.target_hosts and empty_frac <= args.max_empty_frac)
        report_bins.append({
            "bin": i,
            "host_mass_lo": float(bins.host_mass_edges[i]),
            "host_mass_hi": float(bins.host_mass_edges[i + 1]),
            "label": lab,
            "hosts_per_ensemble_mean": float(np.mean(h)),
            "hosts_per_ensemble_p10": eff,
            "hosts_per_ensemble_median": float(np.median(h)),
            "hosts_per_ensemble_p90": float(np.percentile(h, 90)),
            "hosts_pooled_all_boxes": int(sum(
                int(np.asarray(c.n_host)[i]) for c in chunks)),
            "empty_ensemble_fraction": empty_frac,
            "occupation_median": med,
            "occupation_ci95": list(boot),
            "occupation_ci95_relative_width": rel_width,
            "usable_in_reward": usable,
        })

    verdict = {
        "n_boxes_with_hr_catalog": n_boxes,
        "boxes": boxes,
        "ensemble_size_B": B,
        "n_draws": len(draws),
        "box_bootstrap": bool(n_boxes > 1),
        "target_effective_hosts": args.target_hosts,
        "max_empty_ensemble_fraction": args.max_empty_frac,
        "usable_bins": [b["bin"] for b in report_bins if b["usable_in_reward"]],
        "evaluation_only_bins": [b["bin"] for b in report_bins
                                 if not b["usable_in_reward"]],
        "duplication_detected": bool(any(d["duplicated"] for d in dup_checks)),
    }
    if n_boxes < 2:
        verdict["warning"] = (
            f"only {n_boxes} box has an HR catalog, so the 'box bootstrap' above "
            "is a chunk bootstrap within one box and UNDERSTATES cosmic variance. "
            "Treat every interval as a lower bound until "
            "`sbatch --array=0-15 scripts/slurm/hr_catalog_summaries_cpu.sbatch` "
            "has run."
        )

    write_json(Path(args.out), {
        "verdict": verdict, "per_bin": report_bins, "per_box": per_box,
        "duplication_checks": dup_checks, "geometry_sweep": sweep,
    })

    banner("Host-count audit")
    print(f"  boxes with HR catalogs: {n_boxes}  ({', '.join(boxes)})")
    print(f"  {'bin':>10} {'/box':>7} {'/ens':>7} {'p10':>6} {'empty%':>7} "
          f"{'occ':>8} {'CI95 rel':>9}  usable")
    for r in report_bins:
        print(f"  {r['label']:>10} "
              f"{r['hosts_pooled_all_boxes'] / max(n_boxes, 1):7.1f} "
              f"{r['hosts_per_ensemble_mean']:7.1f} "
              f"{r['hosts_per_ensemble_p10']:6.1f} "
              f"{100 * r['empty_ensemble_fraction']:7.2f} "
              f"{r['occupation_median']:8.2f} "
              f"{r['occupation_ci95_relative_width']:9.3f}  "
              f"{'yes' if r['usable_in_reward'] else 'NO -- eval only'}")
    if verdict["duplication_detected"]:
        print("  ! duplication detected in chunk attribution -- investigate")
    if "warning" in verdict:
        print(f"  ! {verdict['warning']}")
    print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
