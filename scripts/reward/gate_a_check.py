#!/usr/bin/env python
"""Gate A (CPU, seconds): decide the eight criteria of sec. 3a from a run's metrics.

Gate A used to be a line in a job log asking a human to "check metrics.csv for
finite loss and diag_sample_diversity > 0", while Gate B depended only on the
training job having exited 0. Two of the eight criteria were therefore load-
bearing and unchecked, and a prior that failed six of them would still have had
its catalog rewards computed -- which is the one thing sec. 3a says not to do.

This reads ``metrics.csv`` from a training run and writes ``gate_a.json`` with a
per-criterion verdict and an overall ``passed``. Criteria that need a second job
(tile-size dependence) are reported ``not_evaluated`` and, by default, block the
pass; ``--allow-unevaluated`` downgrades them to warnings for a smoke run.

    python scripts/reward/gate_a_check.py --run-dir $ZFS/.../residual_prior
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from _common import (add_common_args, banner, constraints_of, load_reward_config,
                     write_json)


def _read_metrics(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            row: Dict[str, float] = {}
            for k, v in r.items():
                if v is None or v == "":
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    continue
            rows.append(row)
    return rows


def _series(rows, key: str) -> List[float]:
    return [r[key] for r in rows if key in r and math.isfinite(r[key])]


def _verdict(ok: Optional[bool], detail: str) -> Dict:
    return {"status": "pass" if ok else ("fail" if ok is False else "not_evaluated"),
            "detail": detail}


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--metrics", default=None, help="default <run-dir>/metrics.csv")
    ap.add_argument("--out", default=None, help="default <run-dir>/gate_a.json")
    ap.add_argument("--allow-unevaluated", action="store_true",
                    help="criteria that need a second job do not block the pass")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when the gate does not pass. Use this only "
                         "where a FAILING gate must stop a chain outright (the "
                         "smoke job); elsewhere read gate_a.json, so a dependent "
                         "can explain itself instead of dying on Dependency")
    ap.add_argument("--audit", default=None,
                    help="residual_target_audit JSON, for criterion 3's reference RMS")
    ap.add_argument("--tile-scan", default=None,
                    help="JSON from a two-tile-size sampling run, for criterion 6")
    ap.add_argument("--loss-drop", type=float, default=0.10,
                    help="fractional fall in val_loss required by criterion 1")
    ap.add_argument("--rms-tolerance-dex", type=float, default=0.5,
                    help="criterion 3: |log10(pred/true) residual RMS| bound")
    ap.add_argument("--x0-clip-max", type=float, default=0.01,
                    help="criterion 5: largest tolerated x0_clip hit fraction")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    cons = constraints_of(cfg)
    run = Path(args.run_dir)
    metrics = Path(args.metrics) if args.metrics else run / "metrics.csv"
    if not metrics.is_file():
        raise SystemExit(f"no metrics at {metrics}; did the training job run?")
    rows = _read_metrics(metrics)
    if not rows:
        raise SystemExit(f"{metrics} has no parseable rows")

    c: Dict[str, Dict] = {}

    # 1 -- validation denoising loss falls.
    vl = _series(rows, "val_loss")
    if len(vl) < 2:
        c["1_val_loss_falls"] = _verdict(
            None, f"only {len(vl)} validation points; need at least 2")
    else:
        drop = (vl[0] - min(vl)) / abs(vl[0]) if vl[0] else float("nan")
        c["1_val_loss_falls"] = _verdict(
            bool(np.isfinite(drop) and drop >= args.loss_drop),
            f"val_loss {vl[0]:.4g} -> {min(vl):.4g} ({100 * drop:.1f}% fall, "
            f"need {100 * args.loss_drop:.0f}%)")

    # 2 -- residual diversity is not collapsed.
    dv = _series(rows, "diag_sample_diversity")
    floor = cons.diversity_min if cons.diversity_min is not None else 0.0
    c["2_diversity"] = _verdict(
        bool(dv) and dv[-1] >= floor,
        f"diag_sample_diversity={dv[-1]:.4g} vs floor {floor:.4g}" if dv
        else "diag_sample_diversity never logged")

    # 3 -- residual RMS is plausible against the paired audit.
    rp = _series(rows, "diag_residual_rms_pred")
    rt = _series(rows, "diag_residual_rms_true")
    if args.audit and Path(args.audit).is_file():
        a = json.loads(Path(args.audit).read_text())
        ref = float(np.mean(a.get("residual_rms", rt[-1:] or [float("nan")])))
    else:
        ref = rt[-1] if rt else float("nan")
    if rp and math.isfinite(ref) and ref > 0:
        dex = abs(math.log10(max(rp[-1], 1e-30) / ref))
        c["3_residual_rms"] = _verdict(
            dex <= args.rms_tolerance_dex,
            f"predicted RMS {rp[-1]:.4g} vs reference {ref:.4g} ({dex:.2f} dex, "
            f"tolerance {args.rms_tolerance_dex})")
    else:
        c["3_residual_rms"] = _verdict(None, "no residual RMS logged")

    # 4 -- residual power spectrum plausible: the composed field must not be
    # worse than the frozen baseline in the high band, which is where a model
    # emitting noise shows up first.
    hi, hib = _series(rows, "diag_disp_Tk_error_high"), _series(rows, "diag_base_Tk_error_high")
    if hi and hib:
        c["4_power_spectrum"] = _verdict(
            hi[-1] <= hib[-1] * 1.5,
            f"composed |T(k)-1| high band {hi[-1]:.4g} vs frozen base "
            f"{hib[-1]:.4g} (allowed 1.5x)")
    else:
        c["4_power_spectrum"] = _verdict(None, "no band metrics logged (crop too small?)")

    # 5 -- the x0 clip is not what sets the amplitude.
    cl = _series(rows, "diag_x0_clip_fraction_max")
    c["5_x0_clip"] = _verdict(
        bool(cl) and cl[-1] <= args.x0_clip_max,
        f"max x0_clip hit fraction {cl[-1]:.4g} vs {args.x0_clip_max}" if cl
        else "diag_x0_clip_fraction_max never logged (older run?)")

    # 6 -- no seam or tile-size dependence. Needs a sampling job at two tile
    # sizes; the training run cannot answer it.
    if args.tile_scan and Path(args.tile_scan).is_file():
        t = json.loads(Path(args.tile_scan).read_text())
        rel = float(t.get("max_rel_difference", float("nan")))
        c["6_tile_independence"] = _verdict(
            math.isfinite(rel) and rel <= float(t.get("tolerance", 1e-4)),
            f"max relative core difference across tile sizes {rel:.3g}")
    else:
        c["6_tile_independence"] = _verdict(
            None, "needs a two-tile-size sampling run; pass --tile-scan")

    # 7 -- no systematic low-k or LR-consistency damage.
    lk = _series(rows, "diag_low_k_change")
    lc = _series(rows, "diag_lr_consistency")
    lk_max = cons.low_k_change_max
    lc_max = cons.lr_consistency_error_max
    parts, ok7 = [], True
    if lk and lk_max is not None:
        parts.append(f"low_k_change={lk[-1]:.4g} vs {lk_max:.4g}")
        ok7 = ok7 and lk[-1] <= lk_max
    if lc and lc_max is not None:
        parts.append(f"lr_consistency={lc[-1]:.4g} vs {lc_max:.4g}")
        ok7 = ok7 and lc[-1] <= lc_max
    c["7_low_k_and_lr"] = _verdict(ok7 if parts else None,
                                   "; ".join(parts) or "not logged / thresholds disabled")

    # 8 -- density statistics physically plausible.
    ds = _series(rows, "diag_density_sigma_ratio")
    dr = _series(rows, "diag_dens_rk_high")
    parts, ok8 = [], True
    if ds:
        parts.append(f"sigma ratio {ds[-1]:.4g} (want 0.8-1.25)")
        ok8 = ok8 and 0.8 <= ds[-1] <= 1.25
    if dr:
        parts.append(f"density r(k) high band {dr[-1]:.4g} (want > 0.5)")
        ok8 = ok8 and dr[-1] > 0.5
    c["8_density"] = _verdict(ok8 if parts else None,
                              "; ".join(parts) or "no density diagnostics logged")

    failed = [k for k, v in c.items() if v["status"] == "fail"]
    unevaluated = [k for k, v in c.items() if v["status"] == "not_evaluated"]
    passed = not failed and (args.allow_unevaluated or not unevaluated)

    out = Path(args.out) if args.out else run / "gate_a.json"
    write_json(out, {
        "run_dir": str(run),
        "metrics": str(metrics),
        "criteria": c,
        "failed": failed,
        "not_evaluated": unevaluated,
        "allow_unevaluated": bool(args.allow_unevaluated),
        "passed": bool(passed),
        "note": (
            "All eight criteria of docs/reward_residual_diffusion.md sec. 3a. "
            "Gate B must not be submitted against a run whose gate_a.json says "
            "passed: false -- the reward numbers would be measuring the bug."
        ),
    })

    banner(f"Gate A: {'PASS' if passed else 'FAIL'}")
    for k in sorted(c):
        mark = {"pass": "ok ", "fail": "FAIL", "not_evaluated": "-- "}[c[k]["status"]]
        print(f"  [{mark}] {k}: {c[k]['detail']}", flush=True)
    print(f"  -> {out}", flush=True)
    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
