#!/usr/bin/env python
"""Stage 7 (CPU): turn scored oracle ensembles into an offline elite replay set.

Reads ``oracle_manifest.jsonl`` (one row per ensemble, already carrying the
per-chunk marginal contributions) and keeps a chunk only when its ensemble is
feasible and in the top quantile **and** its own marginal contribution is
positive. Nothing is copied: an entry points at the residual field the GPU job
already wrote, plus the Lagrangian origin of its chunk, so the trainer crops the
region it needs at load time.

    python scripts/reward/build_replay.py --run-name prior_k8 --round round_000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, chunk_grid, load_reward_config,
                     read_jsonl, split_boxes, write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.replay import (ReplayEntry, elite_weights, select_elites,
                                    sha256_file, write_replay)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-name", required=True, help="oracle run to harvest")
    ap.add_argument("--round", default="round_000")
    ap.add_argument("--elite-quantile", type=float, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--a-max", type=float, default=None)
    ap.add_argument("--w-max", type=float, default=None)
    ap.add_argument("--allow-any-marginal", action="store_true",
                    help="skip the A>0 requirement (diagnostic only)")
    ap.add_argument("--no-checksum", action="store_true",
                    help="skip sha256 of multi-GB residuals (fast, less safe)")
    ap.add_argument("--train-only", action="store_true", default=True)
    ap.add_argument("--allow-val", action="store_true",
                    help="permit val boxes in the buffer; this leaks -- diagnostic only")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    rcfg = cfg.get("replay", {})
    grid = chunk_grid(cfg)
    q = float(args.elite_quantile if args.elite_quantile is not None
              else rcfg.get("elite_quantile", 0.2))
    tau = float(args.tau if args.tau is not None else rcfg.get("tau", 1.0))
    a_max = float(args.a_max if args.a_max is not None else rcfg.get("a_max", 5.0))
    w_max = float(args.w_max if args.w_max is not None else rcfg.get("w_max", 10.0))

    src = paths.ORACLE(args.run_name)
    man = src / "oracle_manifest.jsonl"
    if not man.is_file():
        raise SystemExit(f"no {man}; run scripts/reward/oracle_report.py first")
    ensembles = read_jsonl(man)
    if not ensembles:
        raise SystemExit(f"{man} is empty")

    # Flatten to one row per (ensemble, chunk). Selection and credit both use
    # the score oracle_report.py decided on (occupation by default), not the
    # joint R_cat: weighting elites by a statistic that was not selected for is
    # how a distilled model ends up chasing abundance while occupation stays
    # exactly as flat as the frozen baseline left it.
    rows = []
    scores = {e.get("selection_score", "R_cat") for e in ensembles}
    banner(f"credit/selection score: {sorted(scores)}")
    for e in ensembles:
        for cid_s, a in (e.get("marginal_contributions") or {}).items():
            rows.append({
                "ensemble_id": e["ensemble_id"],
                "box": e["box"], "chunk_id": int(cid_s),
                "seed": e["seed"],
                "residual_path": e.get("residual_path"),
                "base_field": e.get("base_field"),
                "base_seed": int(e.get("base_seed", cfg.get("data", {}).get("base_seed", 0))),
                "residual_scale": float(e.get("residual_scale", 1.0)),
                "ensemble_reward": float(
                    e.get("selection_reward", e["catalog_reward"])),
                "selection_score": str(e.get("selection_score", "R_cat")),
                "marginal_contribution": float(a),
                "feasible": bool(e.get("feasible", False)),
                "constraint_values": e.get("constraint_values", {}),
                "checkpoint": e.get("residual_checkpoint", ""),
            })

    kept = select_elites(
        rows, elite_quantile=q,
        require_positive_marginal=not args.allow_any_marginal,
    )
    banner(f"{len(kept)} elite chunks from {len(rows)} scored chunks "
           f"({len(ensembles)} ensembles)")

    forbidden = set(split_boxes(cfg, "val")) | set(split_boxes(cfg, "test"))
    if args.allow_val:
        forbidden = set(split_boxes(cfg, "test"))
    leaked = sorted({r["box"] for r in kept} & forbidden)
    if leaked and not args.allow_val:
        # The oracle deliberately runs on val boxes, so this is the normal path,
        # not an accident: say so loudly instead of silently training on them.
        print(
            f"  ! {leaked} are held-out boxes. An offline replay buffer built "
            f"from them turns validation into training data. Re-run the oracle "
            f"on train boxes, or pass --allow-val and treat every later number "
            f"as in-sample.",
            flush=True,
        )
        raise SystemExit(
            "refusing to write a replay buffer containing held-out boxes"
        )

    if not kept:
        write_json(paths.REPLAY(args.round, create=True) / "replay_stats.json", {
            "n_entries": 0,
            "reason": "no chunk was both in a top-quantile feasible ensemble and "
                      "had a positive marginal contribution",
        })
        raise SystemExit(
            "empty elite set: either no candidate was feasible, or no chunk "
            "improved its ensemble. Check gate_b.json before loosening the cut."
        )

    w = elite_weights([r["marginal_contribution"] for r in kept],
                      tau=tau, a_max=a_max, w_max=w_max,
                      normalize=str(rcfg.get("weight_normalize", "mean")))

    dst = paths.REPLAY(args.round, create=True)
    sha_cache: dict[str, str] = {}
    entries = []
    for r, wi in zip(kept, w):
        rp = r["residual_path"]
        if not rp or not Path(rp).is_file():
            print(f"  ! missing residual for {r['ensemble_id']}, dropping", flush=True)
            continue
        if args.no_checksum:
            sha = ""
        else:
            sha = sha_cache.get(rp) or sha256_file(rp)
            sha_cache[rp] = sha
        entries.append(ReplayEntry(
            box=r["box"], chunk_id=int(r["chunk_id"]),
            ensemble_id=r["ensemble_id"],
            residual_path=str(rp),
            residual_sha256=sha,
            lr_id=r["box"], base_id=str(r.get("base_field") or ""),
            # The seed that actually produced the cached base field this elite
            # was composed against, not a literal 0. Training resolves the base
            # from base_id/base_seed, so a run at base_seed != 0 used to train
            # against the seed-0 field while claiming to be the elite's.
            base_seed=int(r["base_seed"]), residual_seed=int(r["seed"]),
            residual_scale=float(r["residual_scale"]),
            ensemble_reward=float(r["ensemble_reward"]),
            marginal_contribution=float(r["marginal_contribution"]),
            constraint_values={k: float(v) for k, v in
                               (r["constraint_values"] or {}).items()
                               if isinstance(v, (int, float))},
            feasible=True,
            generation_checkpoint=str(r.get("checkpoint", "")),
            hr_origin=grid.origin(int(r["chunk_id"])),
            chunk_hr=int(grid.chunk_hr),
            weight=float(wi),
        ))

    path = write_replay(dst / "replay.jsonl", entries)
    a = np.asarray([e.marginal_contribution for e in entries])
    ww = np.asarray([e.weight for e in entries])
    stats = {
        "round": args.round,
        "oracle_run": args.run_name,
        "n_scored_chunks": len(rows),
        "n_elite": len(entries),
        "elite_fraction": len(entries) / max(len(rows), 1),
        "elite_quantile": q, "tau": tau, "a_max": a_max, "w_max": w_max,
        "boxes": sorted({e.box for e in entries}),
        "unique_residual_fields": len({e.residual_path for e in entries}),
        "marginal": {"min": float(a.min()), "median": float(np.median(a)),
                     "max": float(a.max()), "mean": float(a.mean())},
        "weight": {"min": float(ww.min()), "median": float(np.median(ww)),
                   "max": float(ww.max()), "mean": float(ww.mean()),
                   "top1_share": float(ww.max() / max(ww.sum(), 1e-30))},
        "manifest": str(path),
        "checksums": not args.no_checksum,
    }
    write_json(dst / "replay_stats.json", stats)

    print(f"  A: median={stats['marginal']['median']:.4g} max={a.max():.4g}", flush=True)
    print(f"  w: median={stats['weight']['median']:.3f} max={ww.max():.3f} "
          f"top1 share={100 * stats['weight']['top1_share']:.1f}%", flush=True)
    if stats["weight"]["top1_share"] > 0.2:
        print("  ! one example carries >20% of the total weight; lower a_max or "
              "raise tau before training, or the elite loss is a single crop.",
              flush=True)
    print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
