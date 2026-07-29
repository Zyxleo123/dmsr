#!/usr/bin/env python
"""Stage 9b (CPU): halo finding and catalog metrics for every evaluation field.

Two families are reported separately and never mixed:

* **optimized** -- subhalo mass function, mean occupation, catalog reward. These
  are what the distillation was trained on; improving them is necessary, not
  sufficient.
* **held out** -- one- and two-halo correlation, radial profile, relative
  velocity, host mass function, isolated/subhalo split, count variance. Nothing
  in the pipeline ever optimized these, so they are the actual evidence.

Confidence intervals resample **boxes**, never chunks.

    python scripts/reward/eval_catalogs.py --run-name final
    python scripts/reward/eval_catalogs.py --run-name final --shard 0 --num-shards 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, bins_of, chunk_grid, hr_path,
                     load_freeze, load_reward_config, write_json, PROJECT_ROOT)

from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import pool, read_summaries, write_summaries
from cosmo_sr.reward.heldout import bootstrap_ci, held_out_metrics
from cosmo_sr.reward.pipeline import field_to_chunk_summaries
from cosmo_sr.reward.reward import RewardModel


def _compose(row, scale: float):
    base = np.load(row["base"] if row.get("base") else row["field"], mmap_mode="r")
    if not row.get("residual"):
        return np.asarray(base, dtype=np.float32)
    r = np.load(row["residual"], mmap_mode="r")
    return np.asarray(base, dtype=np.float32) + np.float32(scale) * np.asarray(
        r, dtype=np.float32)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-name", default="final")
    ap.add_argument("--reward-model", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--with-hr", action="store_true", default=True,
                    help="also process the paired HR box as the reference arm")
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip halo finding; aggregate what is already on disk")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    freeze = load_freeze(cfg)
    grid = chunk_grid(cfg)
    bins = bins_of(cfg)
    dcfg = cfg["data"]
    geo = cfg.get("geometry", {})

    out = paths.EVAL(args.run_name)
    manifest = json.loads((out / "eval_fields.json").read_text())
    scale = float(manifest.get("residual_scale", 1.0))
    rows = list(manifest["fields"])
    if args.with_hr:
        for b in sorted({r["box"] for r in rows}):
            rows.append({"arm": "hr", "box": b, "sample": 0, "seed": -1,
                         "field": str(hr_path(cfg, b)), "residual": None})

    per_dir = out / "per_field"
    per_dir.mkdir(parents=True, exist_ok=True)
    sum_dir = out / "summaries"
    sum_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        for row in rows[args.shard::max(1, args.num_shards)]:
            tag = f"{row['box']}_{row['arm']}_s{row['sample']}"
            rp = per_dir / f"{tag}.json"
            if rp.is_file() and not args.overwrite:
                print(f"[{tag}] cached", flush=True)
                continue
            field = _compose(row, scale)
            res = field_to_chunk_summaries(
                field, box=row["box"], source=row["arm"], tag=tag,
                grid=grid, bins=bins, work_dir=out / "halos" / tag,
                rockstar_binary=PROJECT_ROOT / freeze["rockstar"]["binary"],
                rockstar_cfg=PROJECT_ROOT / freeze["rockstar"]["config"],
                boxsize_kpc_h=float(freeze["cosmology_sim"]["boxsize_kpc_h"]),
                redshift=float(freeze.get("redshift", 0.0)),
                dis_norm_kpc_h=float(dcfg["dis_norm_kpc_h"]),
                purity_grid=int(geo.get("purity_grid", 128)),
                max_half_width=int(geo.get("max_half_width", 4)),
                reuse_catalog=not args.overwrite,
            )
            write_summaries(sum_dir / f"{tag}.jsonl",
                            [res.summaries[c] for c in sorted(res.summaries)])
            ho = held_out_metrics(
                res.catalog, boxsize_mpc_h=float(dcfg["boxsize_mpc_h"]),
                host_bins=np.asarray(bins.host_mass_edges),
                sub_bins=np.asarray(bins.sub_mass_edges),
                min_particles=int(bins.min_sub_particles),
            )
            write_json(rp, {
                "tag": tag, "arm": row["arm"], "box": row["box"],
                "sample": int(row["sample"]), "seed": int(row["seed"]),
                "residual": row.get("residual"), "base": row.get("base"),
                "n_halos": int(res.catalog.n),
                "n_hosts": int(res.catalog.hosts().n),
                "n_subs": int(res.catalog.subhalos().n),
                "assigned_fraction": res.assigned_fraction,
                "summaries": str(sum_dir / f"{tag}.jsonl"),
                "held_out": ho,
                "seconds": res.timings,
            })
            print(f"[{tag}] halos={res.catalog.n} subs={res.catalog.subhalos().n} "
                  f"sub/host={ho['counts']['sub_per_host']:.3f}", flush=True)

    if args.num_shards > 1 and not args.summarize_only:
        print(f"shard {args.shard} finished; rerun with --summarize-only once every "
              f"shard has completed", flush=True)
        return

    # ------------------------------------------------------------ aggregate --
    files = sorted(per_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"nothing to aggregate in {per_dir}")
    recs = [json.loads(p.read_text()) for p in files]

    rm_path = Path(args.reward_model) if args.reward_model else \
        paths.subdir("reward_model") / "reward_model.json"
    model = RewardModel.from_dict(json.loads(Path(rm_path).read_text())) \
        if Path(rm_path).is_file() else None
    if model is None:
        print(f"  ! no reward model at {rm_path}; reporting catalog metrics only",
              flush=True)

    for r in recs:
        s = read_summaries(r["summaries"])
        ens = pool(s)
        r["optimized"] = {
            "n_sub_by_bin": np.asarray(ens.n_sub).tolist(),
            "number_density": ens.number_density().tolist(),
            "occupation": np.nan_to_num(ens.occupation(), nan=0.0).tolist(),
            "n_host_by_bin": np.asarray(ens.n_host).tolist(),
            "volume_mpc3": float(ens.volume_mpc3),
        }
        if model is not None:
            r["optimized"]["catalog_reward"] = model.reward(ens)
            r["optimized"]["mahalanobis2"] = model.mahalanobis2(ens)

    arms = sorted({r["arm"] for r in recs})
    table = {}
    for arm in arms:
        sel = [r for r in recs if r["arm"] == arm]
        boxes = sorted({r["box"] for r in sel})

        def per_box(fn):
            # Average the stochastic samples inside a box first, then bootstrap
            # over boxes: samples of the same box are not independent draws of
            # the universe.
            return [float(np.mean([fn(r) for r in sel if r["box"] == b])) for b in boxes]

        entry = {
            "n_fields": len(sel), "n_boxes": len(boxes),
            "n_subs": bootstrap_ci(per_box(lambda r: r["n_subs"])),
            "n_hosts": bootstrap_ci(per_box(lambda r: r["n_hosts"])),
            "sub_per_host": bootstrap_ci(
                per_box(lambda r: r["held_out"]["counts"]["sub_per_host"])),
            "subhalo_fraction": bootstrap_ci(
                per_box(lambda r: r["held_out"]["counts"]["subhalo_fraction"])),
            "relative_velocity_median": bootstrap_ci(
                per_box(lambda r: r["held_out"]["relative_velocity"]["median"])),
        }
        if model is not None:
            entry["mahalanobis2"] = bootstrap_ci(
                per_box(lambda r: r["optimized"]["mahalanobis2"]))
            entry["catalog_reward"] = bootstrap_ci(
                per_box(lambda r: r["optimized"]["catalog_reward"]))
        # Count variance across stochastic samples, within a box.
        var = []
        for b in boxes:
            v = [r["n_subs"] for r in sel if r["box"] == b]
            if len(v) > 1:
                var.append(float(np.std(v, ddof=1) / max(np.mean(v), 1e-30)))
        entry["subhalo_count_scatter"] = float(np.mean(var)) if var else float("nan")

        for key, sub in (("subhalo_mass_function", "dn_dlnm"),
                         ("host_mass_function", "dn_dlnm"),
                         ("occupation", "mean_nsub"),
                         ("one_halo", "xi"), ("two_halo", "xi"),
                         ("radial_profile", "pdf")):
            stack = np.asarray([r["held_out"][key][sub] for r in sel], dtype=np.float64)
            entry[key] = {
                "mean": np.nanmean(stack, axis=0).tolist(),
                "std": np.nanstd(stack, axis=0).tolist(),
                "x": r_x(sel[0]["held_out"][key]),
            }
        table[arm] = entry

    verdict = _verdict(table)
    write_json(out / "catalog_eval.json", {
        "run_name": args.run_name, "arms": arms,
        "reward_model": str(rm_path) if model is not None else None,
        "table": table, "verdict": verdict,
        "note": ("Every arm is an average over ordinary random samples; no "
                 "selection was applied to the prior or distilled arms."),
    })
    write_json(out / "per_field_index.json",
               [{k: r[k] for k in ("tag", "arm", "box", "sample", "n_subs", "n_hosts")}
                for r in recs])

    banner("catalog evaluation")
    for arm in arms:
        t = table[arm]
        m = t.get("mahalanobis2", {}).get("mean", float("nan"))
        print(f"  {arm:8s} D2={m:>10.4g}  subs={t['n_subs']['mean']:>9.0f}  "
              f"sub/host={t['sub_per_host']['mean']:.3f}  "
              f"scatter={t['subhalo_count_scatter']:.3f}", flush=True)
    for line in verdict:
        print(f"  {line}", flush=True)
    print(f"  -> {out / 'catalog_eval.json'}", flush=True)


def r_x(d):
    for k in ("centers", "r", "r_over_rvir", "v"):
        if k in d:
            return list(d[k])
    return []


def _verdict(table):
    out = []
    if "hr" not in table:
        out.append("no HR arm: differences are relative only.")
    for arm in ("prior", "distill"):
        if arm not in table or "sr2" not in table:
            continue
        d0 = table["sr2"].get("mahalanobis2", {}).get("mean", float("nan"))
        d1 = table[arm].get("mahalanobis2", {}).get("mean", float("nan"))
        if np.isfinite(d0) and np.isfinite(d1) and d0 > 0:
            out.append(f"{arm}: catalog discrepancy {100 * (d0 - d1) / d0:+.1f}% "
                       f"vs frozen SR2 (average sample).")
        s0 = table["sr2"]["sub_per_host"]["mean"]
        s1 = table[arm]["sub_per_host"]["mean"]
        if np.isfinite(s0) and np.isfinite(s1):
            hr = table.get("hr", {}).get("sub_per_host", {}).get("mean", float("nan"))
            trend = ""
            if np.isfinite(hr):
                trend = (" toward HR" if abs(s1 - hr) < abs(s0 - hr) else " away from HR")
            out.append(f"{arm}: subhalos per host {s0:.3f} -> {s1:.3f}{trend}.")
    if "prior" in table and "distill" in table:
        sp = table["prior"]["subhalo_count_scatter"]
        sd = table["distill"]["subhalo_count_scatter"]
        if np.isfinite(sp) and np.isfinite(sd) and sd < 0.2 * sp:
            out.append(
                f"distill: sample-to-sample subhalo-count scatter collapsed "
                f"({sp:.3f} -> {sd:.3f}); the student is close to deterministic."
            )
    return out


if __name__ == "__main__":
    main()
