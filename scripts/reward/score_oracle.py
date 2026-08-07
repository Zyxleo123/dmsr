#!/usr/bin/env python
"""Stage 6b (CPU): constraints + halo finding + chunk summaries per candidate.

Reads the candidate residuals a GPU job produced, composes them with the cached
frozen SR2 base, measures the field-fidelity constraints, runs the halo finder on
the full periodic box, and writes per-chunk catalog summaries. No reward yet --
that is :mod:`oracle_report`, which needs every candidate to exist first.

Split across array tasks with ``--shard i --num-shards n``.

    python scripts/reward/score_oracle.py --run-name prior_k8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, bins_of, chunk_grid, constraints_of,
                     hr_path, load_freeze, load_reward_config, lr_path,
                     read_jsonl, write_json, PROJECT_ROOT)

from cosmo_sr.reward import paths
from cosmo_sr.reward.base import find_base_field
from cosmo_sr.reward.catalog import write_summaries
from cosmo_sr.reward.constraints import check_constraints, constraint_values
from cosmo_sr.reward.pipeline import field_to_chunk_summaries


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--baselines", default="shard0",
                    choices=["shard0", "only", "skip", "all"],
                    help="who scores the per-box a=0 baseline. Every shard holds "
                         "candidates from every box, so 'all' has each of them "
                         "writing the SAME {box}_base output through the SAME "
                         "Rockstar work directory, with an existence check and "
                         "no lock -- concurrent Rockstar runs then corrupt each "
                         "other's catalog. 'shard0' (default) keeps them on one "
                         "task; 'only' scores baselines and no candidates, for a "
                         "prerequisite array; 'skip' assumes that array ran.")
    ap.add_argument("--with-hr", action="store_true", default=True,
                    help="compute HR-referenced constraints (val boxes have HR)")
    ap.add_argument("--no-density", action="store_true",
                    help="skip the CIC density constraints (much faster)")
    ap.add_argument("--skip-halo", action="store_true",
                    help="constraints only; useful for a first pass")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    freeze = load_freeze(cfg)
    grid = chunk_grid(cfg)
    bins = bins_of(cfg)
    cons = constraints_of(cfg)
    geo = cfg.get("geometry", {})
    dcfg = cfg["data"]

    out = paths.ORACLE(args.run_name)
    manifest = json.loads((out / "candidates.json").read_text())
    cands = [c for c in manifest["candidates"]]
    shard, n_shards = int(args.shard), max(1, int(args.num_shards))
    mine = [] if args.baselines == "only" else cands[shard::n_shards]
    scored_dir = out / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    a = float(manifest.get("residual_scale", 1.0))

    banner(f"scoring {len(mine)}/{len(cands)} candidates (shard {shard}), "
           f"baselines={args.baselines}")

    # Baseline (a = 0) is scored once per box: it is the counterfactual every
    # marginal contribution is measured against. It must be scored by exactly ONE
    # task -- see the --baselines help.
    all_boxes = sorted({c["box"] for c in cands})
    if args.baselines == "only":
        base_boxes = all_boxes[shard::n_shards]
    elif args.baselines == "shard0":
        base_boxes = all_boxes if shard == 0 else []
    elif args.baselines == "all":
        base_boxes = sorted({c["box"] for c in mine})
    else:
        base_boxes = []
    for box in base_boxes:
        _score_one(cfg, freeze, grid, bins, cons, geo, dcfg, out, scored_dir,
                   summary_dir, box=box, seed=None, residual_path=None,
                   base_path=None, residual_scale=0.0, args=args, manifest=manifest)

    for c in mine:
        _score_one(cfg, freeze, grid, bins, cons, geo, dcfg, out, scored_dir,
                   summary_dir, box=c["box"], seed=int(c["seed"]),
                   residual_path=c["residual"], base_path=c.get("base"),
                   residual_scale=a, args=args, manifest=manifest)
    banner("done")


def _score_one(cfg, freeze, grid, bins, cons, geo, dcfg, out, scored_dir, summary_dir,
               *, box, seed, residual_path, base_path, residual_scale, args, manifest):
    tag = f"{box}_base" if seed is None else f"{box}_seed{seed}"
    row_path = scored_dir / f"{tag}.json"
    if row_path.is_file() and not args.overwrite:
        print(f"[{tag}] cached", flush=True)
        return

    base_seed = int(manifest.get("base_seed", 0))
    if base_path is None:
        found = find_base_field(box, base_seed)
        if found is None:
            raise SystemExit(f"no cached SR2 base for {box}")
        base_path = str(found)
    base = np.load(base_path, mmap_mode="r")

    if residual_path is None or residual_scale == 0.0:
        field = np.asarray(base, dtype=np.float32)
        resid = None
    else:
        resid = np.load(residual_path, mmap_mode="r")
        field = np.asarray(base, dtype=np.float32) + \
            np.float32(residual_scale) * np.asarray(resid, dtype=np.float32)

    lr = np.load(lr_path(cfg, box))
    hr = np.load(hr_path(cfg, box), mmap_mode="r") if args.with_hr else None

    t0 = time.time()
    vals = constraint_values(
        field, np.asarray(base), lr, hr=hr,
        scale_factor=int(dcfg["scale_factor"]),
        boxsize_mpc_h=float(dcfg["boxsize_mpc_h"]),
        dis_norm_kpc_h=float(dcfg["dis_norm_kpc_h"]),
        redshift=float(dcfg.get("redshift", 0.0)),
        compute_density=not args.no_density,
    )
    # Diversity is an ensemble-level quantity; oracle_report fills it in once all
    # candidates for a box exist. Leave it NaN here and do not judge it yet.
    chk = check_constraints(
        {**vals, "diversity": float("nan")}, _relax_diversity(cons)
    )
    t_cons = time.time() - t0

    row = {
        "box": box, "seed": seed, "tag": tag,
        "residual_path": residual_path, "base_path": base_path,
        "residual_scale": float(residual_scale),
        "constraints": vals, "feasible_field": bool(chk["feasible"]),
        "violations": chk["violations"],
        # Non-blocking breaches. A candidate can be feasible and still carry a
        # `critical` one -- that is the point of the severity map, and it is why
        # these travel on the row instead of being recomputed downstream.
        "constraint_warnings": chk["warnings"],
        "constraint_critical": chk["critical"],
        "seconds_constraints": t_cons,
        "checkpoint": manifest.get("checkpoint"),
    }

    if not args.skip_halo:
        res = field_to_chunk_summaries(
            field, box=box, source="base" if seed is None else "candidate",
            tag=tag, grid=grid, bins=bins,
            work_dir=out / "halos" / tag,
            rockstar_binary=PROJECT_ROOT / freeze["rockstar"]["binary"],
            rockstar_cfg=PROJECT_ROOT / freeze["rockstar"]["config"],
            boxsize_kpc_h=float(freeze["cosmology_sim"]["boxsize_kpc_h"]),
            redshift=float(freeze.get("redshift", 0.0)),
            dis_norm_kpc_h=float(dcfg["dis_norm_kpc_h"]),
            purity_grid=int(geo.get("purity_grid", 128)),
            max_half_width=int(geo.get("max_half_width", 4)),
            reuse_catalog=not args.overwrite,
        )
        write_summaries(summary_dir / f"{tag}.jsonl",
                        [res.summaries[c] for c in sorted(res.summaries)])
        # The whole-box summary goes in its OWN file: it is the reward statistic,
        # the chunk file is the credit-assignment statistic, and anything that
        # pools a summaries file must not accidentally add the box to its own
        # chunks. See cosmo_sr.reward.catalog.summarize_full_box.
        write_summaries(summary_dir / f"{tag}__fullbox.jsonl", [res.full_box])
        row.update({
            "full_box_summary": str(summary_dir / f"{tag}__fullbox.jsonl"),
            "n_hosts_full_box": int(res.full_box.n_host_total),
            "n_subs_full_box": int(res.full_box.n_sub_total),
            "n_halos": int(res.catalog.n),
            "n_hosts": int(res.catalog.hosts().n),
            "n_subs": int(res.catalog.subhalos().n),
            "assigned_fraction": res.assigned_fraction,
            "core_volume_fraction": res.core_volume_fraction,
            "summaries": str(summary_dir / f"{tag}.jsonl"),
            "seconds_halo": res.timings,
        })
        print(f"[{tag}] halos={res.catalog.n} subs={res.catalog.subhalos().n} "
              f"assigned={100 * res.assigned_fraction:.1f}% "
              f"low_k={vals['low_k_change']:.4g}", flush=True)
    else:
        print(f"[{tag}] constraints only: low_k={vals['low_k_change']:.4g}", flush=True)

    write_json(row_path, row)


def _relax_diversity(cons):
    """The same constraints with the ensemble-level bound switched off.

    `replace` rather than a field-by-field rebuild: a rebuild silently dropped
    `calibrated` and would now drop `severity` too, which is exactly the kind of
    provenance loss those flags exist to prevent.
    """
    from dataclasses import replace

    return replace(cons, diversity_min=None)


if __name__ == "__main__":
    main()
