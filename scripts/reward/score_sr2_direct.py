#!/usr/bin/env python
"""Calibrate the density tolerance, then accept or reject a checkpoint.

Two stages, because they answer different questions with different data:

``calibrate``
    Measure the **frozen** generator's own seed-to-seed spread in
    ``density_power_error`` and propose the tolerances. This is the only honest
    source for "how much degradation is indistinguishable from noise", and it
    ends in a human step: the proposal is printed and must be pasted into
    ``configs/reward/sr2_direct_finetune.yaml`` with ``calibrated: true``. Until
    then every checkpoint is rejected, which is deliberate -- a gate whose
    thresholds are placeholders should not be able to pass anything.
``gate``
    Compare a candidate against frozen SR2 **paired on (box, seed)** and apply
    the bounds. Reports every breach rather than the first.

CPU only: both stages read the JSONL rows
``evaluate_sr2_direct.py`` wrote and compute nothing that needs a GPU.

    python scripts/reward/score_sr2_direct.py --stage calibrate --run-name direct_a
    python scripts/reward/score_sr2_direct.py --stage gate --run-name direct_a \
        --tag rung_proj_noise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from _sr2_direct import (  # noqa: E402
    add_direct_args, banner, gate_of, load_direct_config, read_jsonl, run_dir,
    write_json,
)

from cosmo_sr.reward.direct_gates import (  # noqa: E402
    calibrate_from_frozen_seeds, check_direct_gates,
)


def rows_for(run: Path, tag: str) -> List[Dict]:
    p = run / f"field_metrics_{tag}.jsonl"
    if not p.is_file():
        return []
    return read_jsonl(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--stage", required=True, choices=("calibrate", "gate"))
    ap.add_argument("--tag", default="", help="candidate tag (stage gate)")
    ap.add_argument("--frozen-tag", default="frozen")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    run = run_dir(args.run_name, create=True)
    frozen = rows_for(run, args.frozen_tag)

    if not frozen:
        print(f">>> MISSING INPUT: {run / f'field_metrics_{args.frozen_tag}.jsonl'}")
        print(">>> produced by: scripts/reward/evaluate_sr2_direct.py with no "
              ">>> --checkpoint (the frozen baseline)")
        print(">>> exiting 0 so dependents report the same rather than stranding.")
        return 0

    if args.stage == "calibrate":
        ccfg = dict(cfg.get("calibration", {}))
        try:
            report = calibrate_from_frozen_seeds(
                frozen,
                mean_margin=float(ccfg.get("mean_margin", 1.0)),
                single_margin=float(ccfg.get("single_margin", 2.0)))
        except ValueError as e:
            print(f">>> GATE FAILED: {e}")
            print(">>> Evaluate the frozen generator on at least two seeds per box "
                  ">>> before calibrating; a spread from one sample is zero, and a "
                  ">>> zero tolerance rejects the frozen generator itself.")
            return 0
        write_json(run / "gate_calibration.json", report)
        banner(json.dumps(report["proposal"], indent=2))
        print("  !! HUMAN STEP. Paste the proposal above into the `gates:` block of")
        print("  !! configs/reward/sr2_direct_finetune.yaml, set calibrated: true,")
        print("  !! commit, and only then run --stage gate. Nothing passes while")
        print("  !! calibrated is false, which is the point.")
        print(f"  report -> {run / 'gate_calibration.json'}", flush=True)
        return 0

    if not args.tag:
        raise SystemExit("--tag is required for --stage gate")
    cand = rows_for(run, args.tag)
    if not cand:
        print(f">>> MISSING INPUT: {run / f'field_metrics_{args.tag}.jsonl'}")
        print(">>> produced by: scripts/reward/evaluate_sr2_direct.py --checkpoint ...")
        return 0

    result = check_direct_gates(cand, frozen, gate_of(cfg))
    write_json(run / f"gate_{args.tag}.json", result.to_dict())
    banner(f"{args.tag}: {'PASS' if result.passed else 'REJECT'}")
    for k, v in result.values.items():
        print(f"    {k:32s} {v:+.6g}")
    for v in result.violations:
        print(f"    !! {v}")
    print("\n  per-box degradation vs frozen (same box, same seed):")
    for r in sorted(result.per_box, key=lambda r: -r["degradation"])[:12]:
        print(f"    {r['box']}/seed{r['seed']:<3d} cand {r['candidate']:.5f} "
              f"frozen {r['frozen']:.5f} delta {r['degradation']:+.5f}")
    print(f"\n  report -> {run / f'gate_{args.tag}.json'}", flush=True)
    if not result.passed:
        print(">>> This checkpoint is REJECTED. Do not advance to the next "
              ">>> unfreezing rung from it; the next rung starts from the last "
              ">>> FIELD-VALID checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
