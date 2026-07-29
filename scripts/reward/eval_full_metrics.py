#!/usr/bin/env python
"""Stage 9c (CPU): full-box field metrics for every evaluation arm.

Displacement transfer and cross-correlation, density power, density PDF, an
equilateral bispectrum slice, LR consistency, sample diversity, and the
constraint values again so the final table can state plainly whether the
distilled arm stayed inside the box it was allowed to move in.

    python scripts/reward/eval_full_metrics.py --run-name final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, constraints_of, hr_path,
                     load_reward_config, lr_path, write_json)

from cosmo_sr.reward import fields, paths
from cosmo_sr.reward.constraints import check_feasible, constraint_values, diversity_value
from cosmo_sr.reward.heldout import bootstrap_ci


def _compose(row, scale):
    base_p = row.get("base") or row.get("field")
    base = np.load(base_p, mmap_mode="r")
    if not row.get("residual"):
        return np.asarray(base, dtype=np.float32), np.asarray(base, dtype=np.float32)
    r = np.load(row["residual"], mmap_mode="r")
    hat = np.asarray(base, dtype=np.float32) + np.float32(scale) * np.asarray(
        r, dtype=np.float32)
    return hat, np.asarray(base, dtype=np.float32)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-name", default="final")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--bispectrum-bins", type=int, default=8)
    ap.add_argument("--diversity-sub", type=int, default=128)
    ap.add_argument("--no-density", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    cons = constraints_of(cfg)
    d = cfg["data"]
    out = paths.EVAL(args.run_name)
    manifest = json.loads((out / "eval_fields.json").read_text())
    scale = float(manifest.get("residual_scale", 1.0))
    rows = list(manifest["fields"])
    per = out / "per_field_metrics"
    per.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        for row in rows[args.shard::max(1, args.num_shards)]:
            tag = f"{row['box']}_{row['arm']}_s{row['sample']}"
            rp = per / f"{tag}.json"
            if rp.is_file() and not args.overwrite:
                print(f"[{tag}] cached", flush=True)
                continue
            hat, base = _compose(row, scale)
            lr = np.load(lr_path(cfg, row["box"]))
            hp = hr_path(cfg, row["box"])
            hr = np.load(hp, mmap_mode="r") if hp.is_file() else None

            vals = constraint_values(
                hat, base, lr, hr=hr,
                scale_factor=int(d["scale_factor"]), n_bins=int(args.n_bins),
                boxsize_mpc_h=float(d["boxsize_mpc_h"]),
                dis_norm_kpc_h=float(d["dis_norm_kpc_h"]),
                redshift=float(d.get("redshift", 0.0)),
                compute_density=not args.no_density,
            )
            extra = _spectral(hat, hr, d, args)
            write_json(rp, {"tag": tag, "arm": row["arm"], "box": row["box"],
                            "sample": int(row["sample"]),
                            "residual": row.get("residual"),
                            "constraints": vals, "spectra": extra})
            print(f"[{tag}] low_k={vals['low_k_change']:.4g} "
                  f"T_err={vals['displacement_power_error']:.4g} "
                  f"dens_err={vals['density_power_error']:.4g}", flush=True)

    if args.num_shards > 1 and not args.summarize_only:
        print(f"shard {args.shard} finished; rerun with --summarize-only once every "
              f"shard has completed", flush=True)
        return

    recs = [json.loads(p.read_text()) for p in sorted(per.glob("*.json"))]
    if not recs:
        raise SystemExit(f"nothing to aggregate in {per}")

    # Diversity is per (box, arm): the spread across that arm's random samples.
    div = {}
    for arm in sorted({r["arm"] for r in recs}):
        for box in sorted({r["box"] for r in recs if r["arm"] == arm}):
            ps = [r["residual"] for r in recs
                  if r["arm"] == arm and r["box"] == box and r.get("residual")]
            if len(ps) < 2:
                continue
            sub = int(args.diversity_sub)
            cubes = []
            for p in ps:
                a = np.load(p, mmap_mode="r")
                o = (a.shape[-1] - sub) // 2
                cubes.append(np.asarray(a[0:3, o:o + sub, o:o + sub, o:o + sub],
                                        dtype=np.float32))
            div[f"{arm}/{box}"] = diversity_value(cubes)

    table = {}
    for arm in sorted({r["arm"] for r in recs}):
        sel = [r for r in recs if r["arm"] == arm]
        boxes = sorted({r["box"] for r in sel})

        def per_box(key, where="constraints"):
            v = []
            for b in boxes:
                x = [r[where].get(key, float("nan")) for r in sel if r["box"] == b]
                x = [q for q in x if q is not None and np.isfinite(q)]
                if x:
                    v.append(float(np.mean(x)))
            return v

        entry = {"n_fields": len(sel), "n_boxes": len(boxes)}
        for key in ("low_k_change", "displacement_power_error",
                    "displacement_power_error_low_k", "displacement_rk_low_k",
                    "density_power_error", "density_sigma_ratio", "density_pdf_l1",
                    "lr_consistency_error", "residual_rms"):
            entry[key] = bootstrap_ci(per_box(key))
        entry["bispectrum_equilateral_error"] = bootstrap_ci(
            per_box("bispectrum_equilateral_error", where="spectra"))
        dv = [v for k, v in div.items() if k.startswith(f"{arm}/")]
        entry["diversity"] = float(np.mean(dv)) if dv else float("nan")

        feas, viol = check_feasible(
            {k: entry[k]["mean"] for k in
             ("low_k_change", "displacement_power_error", "density_power_error",
              "lr_consistency_error")} | {"diversity": entry["diversity"]},
            cons,
        )
        entry["feasible_on_average"] = bool(feas)
        entry["violations"] = viol
        table[arm] = entry

    write_json(out / "field_eval.json", {
        "run_name": args.run_name,
        "constraints": cons.to_dict(),
        "diversity_per_arm_box": div,
        "table": table,
        "note": ("Feasibility here is evaluated on the *average* metric of each "
                 "arm, which is a summary. Per-candidate feasibility lives in "
                 "the oracle manifest."),
    })

    banner("field metrics")
    for arm, t in table.items():
        print(f"  {arm:8s} low_k={t['low_k_change']['mean']:.4g} "
              f"T_err={t['displacement_power_error']['mean']:.4g} "
              f"dens={t['density_power_error']['mean']:.4g} "
              f"div={t['diversity']:.4g} "
              f"{'OK' if t['feasible_on_average'] else 'INFEASIBLE ' + ','.join(t['violations'])}",
              flush=True)
    print(f"  -> {out / 'field_eval.json'}", flush=True)


def _spectral(hat, hr, d, args):
    """Displacement/density spectra plus an equilateral bispectrum slice."""
    out = {}
    n_bins = int(args.n_bins)
    ks, ps = [], []
    for ch in (0, 1, 2):
        k, p = fields.radial_power(np.asarray(hat[ch]), n_bins)
        ks.append(k)
        ps.append(p)
    out["k"] = ks[0].tolist()
    out["displacement_power"] = np.mean(ps, axis=0).tolist()

    if args.no_density:
        out["bispectrum_equilateral_error"] = float("nan")
        return out

    dh = fields.cic_density_box(
        np.asarray(hat[0:3]), boxsize_mpc_h=float(d["boxsize_mpc_h"]),
        dis_norm_kpc_h=float(d["dis_norm_kpc_h"]),
        redshift=float(d.get("redshift", 0.0)))
    k, p = fields.radial_power(dh, n_bins)
    out["density_power"] = p.tolist()
    kb, bh = fields.equilateral_bispectrum(dh, n_bins=int(args.bispectrum_bins))
    out["bispectrum_k"] = kb.tolist()
    out["bispectrum_equilateral"] = bh.tolist()
    _, pdf = fields.density_pdf(dh)
    out["density_pdf"] = pdf.tolist()
    del dh

    if hr is not None:
        dr = fields.cic_density_box(
            np.asarray(hr[0:3]), boxsize_mpc_h=float(d["boxsize_mpc_h"]),
            dis_norm_kpc_h=float(d["dis_norm_kpc_h"]),
            redshift=float(d.get("redshift", 0.0)))
        _, pr = fields.radial_power(dr, n_bins)
        out["density_power_hr"] = pr.tolist()
        _, br = fields.equilateral_bispectrum(dr, n_bins=int(args.bispectrum_bins))
        out["bispectrum_equilateral_hr"] = br.tolist()
        ok = np.isfinite(bh) & np.isfinite(br) & (np.abs(br) > 0)
        out["bispectrum_equilateral_error"] = float(
            np.mean(np.abs(bh[ok] / br[ok] - 1.0))) if ok.any() else float("nan")
        del dr
    else:
        out["bispectrum_equilateral_error"] = float("nan")
    return out


if __name__ == "__main__":
    main()
