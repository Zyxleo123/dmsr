#!/usr/bin/env python
"""Rung 2 (CPU): rank one CEM generation and carry the winners forward.

Ranks on ``R_occ``, not the joint ``R_cat``: §4 decides Gate B on occupation for
the same reason search must optimise it -- a candidate that fixes abundance and
leaves occupation flat is the informative failure, not a partial success, and
ranking on the joint reward would let abundance buy the win.

Each candidate is ranked on the pooled summaries of its **own** whole box. Gate
B's stratified groups exist to compare candidates against each other on paired
chunk subsets; here the population is generated and scored under identical
conditions already, so pooling everything is the lower-variance choice.

Infeasible candidates (field constraints violated) are ranked last regardless of
reward -- an infeasible field cannot be an elite at any reward.

    python scripts/reward/cem_select_elites.py --run-name cem_a --iteration 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import add_common_args, banner, load_reward_config, write_json
from cem_search import run_dir

from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import pool, read_summaries
from cosmo_sr.reward.reward import RewardModel


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--cem-config", default="configs/reward/cem.yaml")
    ap.add_argument("--reward-model", default=None)
    ap.add_argument("--elites", type=int, default=None)
    ap.add_argument("--keep-noise", action="store_true",
                    help="keep every candidate's noise, not just the elites'")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    from cosmo_sr.utils.config import load_config
    ccfg = dict(load_config(args.cem_config).get("cem", {}))
    E = int(args.elites or ccfg.get("elites", 2))
    it = int(args.iteration)
    out = run_dir(args.run_name, it)

    rm_path = Path(args.reward_model) if args.reward_model else \
        paths.subdir("reward_model") / "reward_model.json"
    if not Path(rm_path).is_file():
        # A gate whose input is missing must not strand its dependents.
        banner(f"no reward model at {rm_path}; run fit_reward_model first -- skipping")
        return
    model = RewardModel.from_dict(json.loads(Path(rm_path).read_text()))

    rows = [json.loads(p.read_text()) for p in sorted((out / "scored").glob("*.json"))]
    if not rows:
        banner(f"no scored candidates in {out / 'scored'} -- skipping")
        return
    manifest = json.loads((out / "candidates.json").read_text())
    noise_by_cid = {int(c["seed"]): c.get("noise") for c in manifest["candidates"]}

    def summaries_of(row):
        p = row.get("summaries")
        return [s for s in read_summaries(p)] if p else []

    base_occ = {}
    for r in (r for r in rows if r.get("seed") is None):
        s = summaries_of(r)
        if s:
            base_occ[r["box"]] = model.reward_occupation(pool(s))

    ranked = {}
    for r in (r for r in rows if r.get("seed") is not None):
        s = summaries_of(r)
        if not s:
            continue
        ens = pool(s)
        feasible = bool(r.get("feasible_field", True))
        ranked.setdefault(r["box"], []).append({
            "cem_iter": it,
            "seed": int(r["seed"]),
            "box": r["box"],
            "R_occ": float(model.reward_occupation(ens)),
            "R_abund": float(model.reward_abundance(ens)),
            "R_cat": float(model.reward(ens)),
            "feasible_field": feasible,
            "violations": r.get("violations", []),
            "noise": noise_by_cid.get(int(r["seed"])),
            "residual": r.get("residual_path"),
        })

    report, arrays = {}, {}
    for box, cands in sorted(ranked.items()):
        cands.sort(key=lambda c: (c["feasible_field"], c["R_occ"]), reverse=True)
        elites = [c for c in cands if c["feasible_field"]][:E]
        if not elites:
            banner(f"[{box}] every candidate is infeasible; no elites this iteration")
        for rank, c in enumerate(elites):
            if c["noise"] and Path(c["noise"]).is_file():
                arrays[f"{box}__noise{rank:02d}"] = np.load(c["noise"])
        report[box] = {
            "baseline_R_occ": base_occ.get(box),
            "best_R_occ": cands[0]["R_occ"] if cands else None,
            "elite_seeds": [c["seed"] for c in elites],
            "candidates": cands,
        }
        b = base_occ.get(box)
        best = cands[0]["R_occ"] if cands else float("nan")
        print(f"[{box}] best R_occ={best:.6g} baseline={b if b is None else f'{b:.6g}'} "
              f"elites={[c['seed'] for c in elites]}", flush=True)

    write_json(out / "elites.json", {
        "cem_run": args.run_name, "cem_iter": it, "elites_per_box": E,
        "reward_model": str(rm_path), "boxes": report,
    })
    np.savez_compressed(out / "elites.npz", **arrays)
    banner(f"{len(arrays)} elite noise vectors -> {out / 'elites.npz'}")

    if not args.keep_noise:
        keep = set()
        for box, rep in report.items():
            keep.update((box, s) for s in rep["elite_seeds"])
        for c in manifest["candidates"]:
            if (c["box"], int(c["seed"])) in keep:
                continue
            p = c.get("noise")
            if p and Path(p).is_file():
                Path(p).unlink()


if __name__ == "__main__":
    main()
