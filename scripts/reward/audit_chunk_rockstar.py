#!/usr/bin/env python
"""Diagnostic: run Rockstar on isolated Lagrangian chunk crops.

The production pipeline deliberately never does this (see
``cosmo_sr.reward.geometry``): a Lagrangian crop is not an Eulerian cube, and
Rockstar is only trustworthy on the full periodic box. This script exists to
*measure* how badly the naive alternative fails, by comparing:

* chunk-Rockstar on true HR vs frozen SR2 base (same chunk ids);
* the same chunks' *attributed* summaries from an existing full-box Rockstar
  cache, when available (geometry must match ``chunk_hr``).

Method under test
-----------------
1. Crop the ``(6, Ng, Ng, Ng)`` field to one Lagrangian ``chunk_hr`` cube.
2. Treat that crop as its own periodic universe of side
   ``L = boxsize * chunk_hr / Ng``.
3. Run Rockstar with ``BOX_SIZE = L`` (and ``PERIODIC=1`` by default).
4. Score the resulting catalog with the fitted whole-box reward model.

Caveat (printed in the report): ``mu_HR`` / ``C`` describe *whole-box* summary
vectors. ``R_*`` on a single chunk is therefore misspecified in absolute scale;
HR-vs-base *ranking* under the same procedure is still informative, and raw
counts / occupation are the primary readout.

    python scripts/reward/audit_chunk_rockstar.py \\
        --box set8 --chunks 0,1 --sources hr,base
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from _common import (PROJECT_ROOT, add_common_args, banner, bins_of, chunk_grid,
                     hr_path, load_reward_config, write_json)

from cosmo_sr.eval.particles import field_to_particles
from cosmo_sr.eval.rockstar import (default_rockstar_binary, default_rockstar_cfg,
                                    run_rockstar_on_particles)
from cosmo_sr.reward import paths
from cosmo_sr.reward.base import find_base_field
from cosmo_sr.reward.catalog import (ChunkSummary, read_summaries,
                                     summarize_full_box)
from cosmo_sr.reward.reward import RewardModel


def _load_field(box: str, source: str, cfg: Dict, base_seed: int) -> Path:
    if source == "hr":
        return Path(hr_path(cfg, box))
    if source == "base":
        p = find_base_field(box, seed=base_seed)
        if p is None:
            raise SystemExit(
                f"no SR2 base cache for {box} seed={base_seed}; "
                f"run scripts/reward/cache_sr2_base.py first")
        return Path(p)
    raise SystemExit(f"unknown source {source!r}")


def _crop_field(field: np.ndarray, grid, chunk_id: int) -> np.ndarray:
    sx, sy, sz = grid.slices(chunk_id)
    return np.ascontiguousarray(field[:, sx, sy, sz])


def _chunk_cfg(base_cfg: Path, box_mpc: float, periodic: int,
               out_path: Path) -> Path:
    text = base_cfg.read_text()
    text = re.sub(r"^\s*BOX_SIZE\s*=.*$", f"BOX_SIZE = {box_mpc}", text,
                  flags=re.M)
    text = re.sub(r"^\s*PERIODIC\s*=.*$", f"PERIODIC = {int(periodic)}", text,
                  flags=re.M)
    out_path.write_text(text)
    return out_path


def _score(model: RewardModel, summary: ChunkSummary,
           reliable: Sequence[int]) -> Dict:
    from cosmo_sr.reward.catalog import pool
    ens = pool([summary])
    scores = model.scores(ens, reliable_host_bins=list(reliable))
    occ = np.asarray(ens.occupation(), dtype=np.float64)
    return {
        "R_cat": float(scores["R_cat"]),
        "R_occ": float(scores["R_occ"]),
        "R_abund": float(scores["R_abund"]),
        "R_occ_reliable": float(scores.get("R_occ_reliable", scores["R_occ"])),
        "n_host_total": int(summary.n_host_total),
        "n_sub_total": int(summary.n_sub_total),
        "n_host": [float(x) for x in np.asarray(summary.n_host)],
        "n_sub": [float(x) for x in np.asarray(summary.n_sub)],
        "occupation": [float(x) if np.isfinite(x) else None for x in occ],
        "volume_mpc3": float(summary.volume_mpc3),
    }


def _attributed_lookup(cache: Path, box: str, source: str, tag: str,
                       chunk_id: int, expected_n: int
                       ) -> Optional[ChunkSummary]:
    stem = cache / f"{box}__{source}__{tag}.jsonl"
    if not stem.is_file():
        return None
    rows = read_summaries(stem)
    if len(rows) != expected_n:
        # Geometry mismatch (e.g. old chunk_hr=128 cache vs current 256).
        return None
    by_id = {int(s.chunk_id): s for s in rows}
    return by_id.get(int(chunk_id))


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--box", required=True)
    ap.add_argument("--chunks", default="0",
                    help="comma-separated Lagrangian chunk ids")
    ap.add_argument("--sources", default="hr,base",
                    help="comma-separated: hr, base")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--periodic", type=int, default=1, choices=[0, 1])
    ap.add_argument("--work", default=None,
                    help="scratch dir for GADGET+Rockstar (default under audits/)")
    ap.add_argument("--out", default=None, help="report JSON path")
    ap.add_argument("--reward-model", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-gadget", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    grid = chunk_grid(cfg)
    bins = bins_of(cfg)
    dcfg = cfg["data"]
    box_mpc_full = float(dcfg["boxsize_mpc_h"])
    chunk_mpc = float(grid.chunk_mpc_h)
    boxsize_kpc = chunk_mpc * 1e3
    reliable = list(cfg.get("reward", {}).get("occupation", {})
                    .get("reliable_host_bins", [0, 1, 2, 3]))

    chunk_ids = [int(x) for x in str(args.chunks).split(",") if x.strip() != ""]
    sources = [s.strip() for s in str(args.sources).split(",") if s.strip()]
    for c in chunk_ids:
        if not 0 <= c < grid.n_chunks:
            raise SystemExit(f"chunk {c} outside 0..{grid.n_chunks - 1}")

    rm_path = Path(args.reward_model) if args.reward_model else \
        paths.subdir("reward_model") / "reward_model.json"
    if not rm_path.is_file():
        raise SystemExit(f"no reward model at {rm_path}")
    model = RewardModel.from_dict(json.loads(rm_path.read_text()))

    out_dir = paths.AUDITS("chunk_rockstar", create=True)
    work = Path(args.work) if args.work else out_dir / "work" / args.box
    work.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.out) if args.out else \
        out_dir / f"{args.box}_chunks{'-'.join(map(str, chunk_ids))}.json"

    catalog_cache = paths.CATALOG_CACHE(create=False)
    binary = default_rockstar_binary()
    base_cfg = default_rockstar_cfg()

    banner(
        f"chunk-Rockstar audit: box={args.box} chunks={chunk_ids} "
        f"sources={sources} chunk_hr={grid.chunk_hr} L={chunk_mpc:g} Mpc/h "
        f"PERIODIC={args.periodic}"
    )
    print(
        "  NOTE: reward model was fit on whole-box vectors; R_* on a single "
        "chunk is misspecified in absolute scale. Prefer raw counts / "
        "occupation for the HR-vs-base comparison.",
        flush=True,
    )

    field_paths: Dict[str, str] = {}
    for src in sources:
        p = _load_field(args.box, src, cfg, args.base_seed)
        field_paths[src] = str(p)
        print(f"  {src}: {p}", flush=True)

    rows: List[Dict] = []
    for chunk_id in chunk_ids:
        for src in sources:
            tag = f"{src}_chunk{chunk_id}"
            run_dir = work / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()

            # One source at a time: never hold two 3.2 GB fields + Rockstar RSS.
            mm = np.load(field_paths[src], mmap_mode="r")
            crop = _crop_field(mm, grid, chunk_id)
            del mm
            particles = field_to_particles(
                crop,
                boxsize_kpc_h=boxsize_kpc,
                redshift=float(dcfg.get("redshift", 0.0)),
            )
            del crop
            cfg_path = _chunk_cfg(
                base_cfg, chunk_mpc, args.periodic, run_dir / "rockstar.cfg")
            cat = run_rockstar_on_particles(
                particles, run_dir, tag=tag, binary=binary, cfg=cfg_path,
                overwrite=bool(args.overwrite),
            )
            if not args.keep_gadget:
                snap = run_dir / f"{tag}.gadget2"
                if snap.is_file():
                    snap.unlink()

            summary = summarize_full_box(
                cat, bins, volume_mpc3=float(grid.chunk_volume_mpc3),
                box=args.box, source=f"chunk_rockstar_{src}",
            )
            # Re-tag so reports carry the chunk id.
            summary = ChunkSummary(
                box=summary.box, chunk_id=int(chunk_id),
                source=summary.source,
                n_sub=summary.n_sub, n_host=summary.n_host,
                occ_numerator=summary.occ_numerator,
                volume_mpc3=summary.volume_mpc3,
                n_sub_total=summary.n_sub_total,
                n_host_total=summary.n_host_total,
                n_excluded_boundary=0,
                n_excluded_resolution=summary.n_excluded_resolution,
                meta={**dict(summary.meta),
                      "method": "lagrangian_crop_rockstar",
                      "periodic": int(args.periodic),
                      "chunk_mpc_h": chunk_mpc,
                      "n_particles": int(particles.pos_mpc_h.shape[0]),
                      "particle_mass_msun_h": float(particles.particle_mass_msun_h)},
            )
            scored = _score(model, summary, reliable)
            elapsed = time.time() - t0

            attr = _attributed_lookup(
                catalog_cache, args.box, src, src, chunk_id, grid.n_chunks)
            attr_scored = _score(model, attr, reliable) if attr is not None else None

            row = {
                "box": args.box,
                "chunk_id": int(chunk_id),
                "source": src,
                "field": field_paths[src],
                "elapsed_s": elapsed,
                "n_halos_catalog": int(cat.n),
                "chunk_rockstar": scored,
                "fullbox_attributed": attr_scored,
                "work_dir": str(run_dir),
            }
            rows.append(row)
            print(
                f"  [{src} chunk {chunk_id}] hosts={scored['n_host_total']} "
                f"subs={scored['n_sub_total']} R_occ={scored['R_occ']:.3g} "
                f"({elapsed:.0f}s)"
                + ("" if attr_scored is None else
                   f"  | attributed hosts={attr_scored['n_host_total']} "
                   f"R_occ={attr_scored['R_occ']:.3g}"),
                flush=True,
            )

    # Pairwise HR vs base under chunk-Rockstar
    pairs: List[Dict] = []
    by_key = {(r["chunk_id"], r["source"]): r for r in rows}
    for cid in chunk_ids:
        if (cid, "hr") not in by_key or (cid, "base") not in by_key:
            continue
        hr = by_key[(cid, "hr")]["chunk_rockstar"]
        base = by_key[(cid, "base")]["chunk_rockstar"]
        pairs.append({
            "chunk_id": cid,
            "d_n_host": hr["n_host_total"] - base["n_host_total"],
            "d_n_sub": hr["n_sub_total"] - base["n_sub_total"],
            "dR_occ": hr["R_occ"] - base["R_occ"],
            "dR_cat": hr["R_cat"] - base["R_cat"],
            "hr_occupation": hr["occupation"],
            "base_occupation": base["occupation"],
            "hr_prefers_higher_R_occ": bool(hr["R_occ"] > base["R_occ"]),
            "hr_has_more_subs": bool(hr["n_sub_total"] > base["n_sub_total"]),
        })

    report = {
        "box": args.box,
        "chunk_hr": grid.chunk_hr,
        "chunk_mpc_h": chunk_mpc,
        "n_chunks_per_box": grid.n_chunks,
        "periodic": int(args.periodic),
        "boxsize_mpc_h_full": box_mpc_full,
        "reward_model": str(rm_path),
        "caveat": (
            "Reward model moments describe whole-box summary vectors. Absolute "
            "R_* on a Lagrangian crop is misspecified; use HR-vs-base deltas "
            "and raw catalog counts. A Lagrangian crop is not an Eulerian cube "
            "(|Psi| ~ 36 HR cells); this is the failure mode geometry.py warns "
            "about."
        ),
        "rows": rows,
        "hr_vs_base": pairs,
        "verdict_hints": {
            "n_pairs": len(pairs),
            "hr_wins_R_occ": sum(1 for p in pairs if p["hr_prefers_higher_R_occ"]),
            "hr_wins_n_sub": sum(1 for p in pairs if p["hr_has_more_subs"]),
        },
    }
    write_json(report_path, report)
    print(f"\n  -> {report_path}", flush=True)
    if pairs:
        w_r = report["verdict_hints"]["hr_wins_R_occ"]
        w_s = report["verdict_hints"]["hr_wins_n_sub"]
        print(
            f"  HR vs base on {len(pairs)} chunks: "
            f"HR higher R_occ in {w_r}/{len(pairs)}, "
            f"HR more subs in {w_s}/{len(pairs)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
