#!/usr/bin/env python
"""Stage 5 (CPU): derive constraint thresholds from the frozen baseline itself.

Thresholds picked so that everything passes measure nothing. These are derived
from two references that exist before any residual is trained:

* **baseline scatter** -- the frozen SR2 field against its paired HR box, across
  the training boxes. This is how wrong SR2 already is, and it is the level a
  residual must not make worse.
* **box-to-box scatter** -- the same statistic between *different* HR boxes,
  which is the cosmic-variance floor. A threshold below this would reject a
  perfect model.

``low_k_change`` has no baseline (it is zero by construction when the residual is
zero), so it is set from the LR-visible scatter between HR boxes scaled by
``--low-k-fraction``: the residual may move the LR-visible scales by at most a
small fraction of the difference between two independent universes.

    python scripts/reward/calibrate_constraints.py --boxes set0,set1,set2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, hr_path, load_reward_config,
                     lr_path, split_boxes, write_json)

from cosmo_sr.reward import fields, paths
from cosmo_sr.reward.base import find_base_field


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--boxes", default=None, help="default = train split")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=1.5,
                    help="threshold = margin * baseline value (or quantile)")
    ap.add_argument("--low-k-fraction", type=float, default=0.1,
                    help="allowed low-k move as a fraction of HR box-to-box scatter")
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-density", action="store_true")
    ap.add_argument("--write-config", default=None,
                    help="also write a YAML fragment ready to paste/include")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    d = cfg["data"]
    boxes = [b.strip() for b in args.boxes.split(",")] if args.boxes \
        else split_boxes(cfg, "train")
    out = Path(args.out) if args.out else paths.AUDITS("constraint_calibration",
                                                       create=True)
    out.mkdir(parents=True, exist_ok=True)
    sf = int(d["scale_factor"])

    banner(f"calibrating on {len(boxes)} boxes: {boxes}")

    rows = []
    lr_of_hr = []
    for b in boxes:
        bp = find_base_field(b, args.base_seed)
        if bp is None:
            print(f"  [{b}] no cached SR2 base, skipping", flush=True)
            continue
        base = np.load(bp, mmap_mode="r")
        hr = np.load(hr_path(cfg, b), mmap_mode="r")
        lr = np.load(lr_path(cfg, b))

        a_base = fields.block_average(np.asarray(base), sf)
        a_hr = fields.block_average(np.asarray(hr), sf)
        lr_of_hr.append(a_hr)
        row = {
            "box": b,
            "lr_consistency_error_base": fields.rel_rms(a_base, lr),
            "lr_consistency_error_hr": fields.rel_rms(a_hr, lr),
        }
        del a_base, a_hr

        t_err = []
        for ch in (0, 1, 2):
            k, p_b, p_h, _ = fields.cross_power(np.asarray(base[ch]),
                                                np.asarray(hr[ch]), args.n_bins)
            tk = np.sqrt(np.maximum(p_b, 0.0) / np.maximum(p_h, 1e-30))
            t_err.append(float(np.mean(np.abs(tk - 1.0))))
        row["displacement_power_error_base"] = float(np.mean(t_err))

        if not args.no_density:
            db = fields.cic_density_box(
                np.asarray(base[0:3]), boxsize_mpc_h=float(d["boxsize_mpc_h"]),
                dis_norm_kpc_h=float(d["dis_norm_kpc_h"]),
                redshift=float(d.get("redshift", 0.0)))
            dh = fields.cic_density_box(
                np.asarray(hr[0:3]), boxsize_mpc_h=float(d["boxsize_mpc_h"]),
                dis_norm_kpc_h=float(d["dis_norm_kpc_h"]),
                redshift=float(d.get("redshift", 0.0)))
            k, p_b, p_h, _ = fields.cross_power(db, dh, args.n_bins)
            ratio = np.maximum(p_b, 1e-30) / np.maximum(p_h, 1e-30)
            row["density_power_error_base"] = float(np.mean(np.abs(np.log(ratio))))
            row["density_sigma_ratio_base"] = float(db.std() / max(dh.std(), 1e-30))
            del db, dh
        rows.append(row)
        print(f"  [{b}] " + "  ".join(
            f"{k}={v:.4g}" for k, v in row.items() if k != "box"), flush=True)

    if not rows:
        raise SystemExit(
            "no cached SR2 base fields found; run scripts/slurm/cache_sr2_base.sbatch "
            "before calibrating"
        )

    # HR box-to-box scatter on the LR-visible scales: the cosmic-variance floor.
    pair_low_k = []
    for i in range(len(lr_of_hr)):
        for j in range(i + 1, len(lr_of_hr)):
            pair_low_k.append(fields.rel_rms(lr_of_hr[i], lr_of_hr[j]))
    hr_scatter = float(np.median(pair_low_k)) if pair_low_k else float("nan")

    def agg(key):
        v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return (float(np.median(v)), float(np.max(v))) if v else (float("nan"),) * 2

    med_lr, max_lr = agg("lr_consistency_error_base")
    med_disp, max_disp = agg("displacement_power_error_base")
    med_dens, max_dens = agg("density_power_error_base")

    m = float(args.margin)
    low_k = (float(args.low_k_fraction) * hr_scatter
             if np.isfinite(hr_scatter) else 0.02)
    proposed = {
        "low_k_change_max": round(low_k, 6),
        "displacement_power_error_max": round(m * max_disp, 6)
        if np.isfinite(max_disp) else None,
        "density_power_error_max": round(m * max_dens, 6)
        if np.isfinite(max_dens) else None,
        "lr_consistency_error_max": round(m * max_lr, 6)
        if np.isfinite(max_lr) else None,
        "diversity_min": 0.05,
    }

    report = {
        "boxes": boxes,
        "n_boxes_used": len(rows),
        "margin": m,
        "low_k_fraction": float(args.low_k_fraction),
        "per_box": rows,
        "baseline": {
            "lr_consistency_error": {"median": med_lr, "max": max_lr},
            "displacement_power_error": {"median": med_disp, "max": max_disp},
            "density_power_error": {"median": med_dens, "max": max_dens},
        },
        "hr_box_to_box_low_k_scatter": hr_scatter,
        "proposed_constraints": proposed,
        "rationale": {
            "low_k_change_max": (
                f"{args.low_k_fraction:g} x the median LR-visible difference "
                f"between two independent HR boxes ({hr_scatter:.4g}). The "
                "residual may nudge the scales SR2 already gets right, not "
                "rewrite them."
            ),
            "others": (
                f"{m:g} x the worst frozen-SR2-vs-HR value over the calibration "
                "boxes. A candidate is infeasible when it is materially worse "
                "than the baseline it is supposed to improve, not when it is "
                "merely imperfect."
            ),
            "diversity_min": (
                "Not derived from data: it is a collapse detector. A distilled "
                "sampler that becomes deterministic scores well and is useless."
            ),
        },
    }
    write_json(out / "constraint_calibration.json", report)

    yaml_text = "constraints:\n  calibrated: true\n" + "".join(
        f"  {k}: {v}\n" for k, v in proposed.items()
    )
    (out / "constraints.yaml").write_text(yaml_text)
    if args.write_config:
        Path(args.write_config).write_text(yaml_text)

    banner("proposed thresholds")
    print(yaml_text)
    if np.isfinite(max_disp) and max_disp > 0.5:
        print("  ! baseline displacement transfer error is large (>0.5): the "
              "frozen SR2 is far from HR at some k, so this threshold is loose "
              "by construction. Do not read a pass as evidence of field fidelity.",
              flush=True)
    print(f"  -> {out / 'constraint_calibration.json'}", flush=True)


if __name__ == "__main__":
    main()
