#!/usr/bin/env python
"""Which from-scratch configuration survives? Reads the bracket probes' metrics.

Three long runs failed for three different reasons, each costing ~12 GPU-hours
before the problem was visible. Run 3's mode -- the generator diverging once the
adversary took over alone -- shows up within 500 steps of the handoff, so
``submit_sr2_scratch_bracket.sh`` asks the question in minutes instead. This
reads what those probes wrote and applies the stop rules.

The rules, fixed here rather than eyeballed per run:

``DIVERGED``
    ``base_mse`` exceeds ``--mse-max`` (default 10) at any eval. Run 3 hit
    1.6e8 within 500 steps of the handoff, so this is not a close call.

``SATURATED``
    the discriminator's score gap exceeds ``--gap-max`` (default 8) or its loss
    falls below ``--dloss-min`` (default 0.05). Run 3 ran at a gap of ~16.7 with
    ``d_loss`` 0.0005: a D that has won completely gives the generator nothing
    useful, which is the mechanism behind the divergence rather than a separate
    symptom. A healthy BCE discriminator sits near ``ln 2 ~ 0.69``.

``UNTESTED``
    the adversary never drove the generator (``adv.w_adv`` stayed 0), so the
    probe says nothing about the handoff. True of probe A **by design** -- it is
    the control for "is the MSE-only phase healthy at all?" -- and a warning for
    any other probe. This keys on ``w_adv`` and not on ``w_mse_now``: under
    ``--mse-mode sum`` the MSE term stays on for the whole run, so a
    ``w_mse_now`` test can never fire and would mark a fully exercised probe
    UNTESTED.

``OK``
    none of the above, over the whole run.

``SATURATED``'s two thresholds are calibrated for **BCE**, whose scores are
logits and whose loss sits near ``ln 2``. A ``wgan_gp`` arm's scores are
unbounded by construction, so a large gap there is not automatically a failure
and this rule does not apply to it -- validated on run 2, which reports
SATURATED at a gap of 284 under a WGAN critic where the real defect was the
missing input normaliser. Every bracket probe is BCE, so the rule is apt for
them.

An ``OK`` verdict means *stable*, not *correct*: 3000 steps cannot say whether
a configuration reproduces SR2's field. That is what the field report on a long
run is for (``submit_sr2_scratch_field_report.sh``), and its yardstick is
``P/P_HR``, not mse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.reward import paths  # noqa: E402


def read_rows(p: Path) -> List[Dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def verdict_of(rows: List[Dict], *, mse_max: float, gap_max: float,
               dloss_min: float) -> Dict:
    if not rows:
        return {"verdict": "NO DATA", "why": "metrics.jsonl is empty"}
    mses = [r.get("base_mse") for r in rows if r.get("base_mse") is not None]
    worst = max(mses) if mses else float("nan")
    last = rows[-1]

    gaps, dlosses = [], []
    for r in rows:
        a = r.get("adv") or {}
        if "d_real" in a and "d_fake" in a:
            gaps.append(float(a["d_real"]) - float(a["d_fake"]))
        if "d_loss" in a:
            dlosses.append(float(a["d_loss"]))
    # "did the adversary actually drive the generator", NOT "did MSE switch off".
    # Under --mse-mode sum, MSE stays on by construction, so a w_mse_now test
    # can never fire there and would mark a perfectly exercised probe UNTESTED.
    handed_over = any(float((r.get("adv") or {}).get("w_adv", 0.0)) > 0
                      for r in rows)

    out = {
        "step": int(last["step"]),
        "final_mse": float(last.get("base_mse", float("nan"))),
        "worst_mse": float(worst),
        "final_low_k": float(last.get("base_low_k", float("nan"))),
        "max_gap": (max(gaps) if gaps else None),
        "min_d_loss": (min(dlosses) if dlosses else None),
        "handed_over": bool(handed_over),
        "noise_std_mean": last.get("noise_std_mean"),
    }
    if worst == worst and worst > float(mse_max):
        out.update(verdict="DIVERGED",
                   why=f"base_mse reached {worst:.3g} (> {mse_max})")
    elif gaps and max(gaps) > float(gap_max):
        out.update(verdict="SATURATED",
                   why=f"D score gap reached {max(gaps):.1f} (> {gap_max})")
    elif dlosses and min(dlosses) < float(dloss_min):
        out.update(verdict="SATURATED",
                   why=f"d_loss fell to {min(dlosses):.4g} (< {dloss_min}); "
                       "a BCE discriminator sits near ln2 ~ 0.69")
    elif not handed_over:
        out.update(verdict="UNTESTED",
                   why="w_mse_now never reached 0: the adversary never took "
                       "over, so the handoff is not exercised")
    else:
        out.update(verdict="OK", why="stable through the handoff")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="",
                    help="sr2_scratch directory (default: the reward root's)")
    ap.add_argument("--probes", default="A B C D")
    ap.add_argument("--mse-max", type=float, default=10.0)
    ap.add_argument("--gap-max", type=float, default=8.0)
    ap.add_argument("--dloss-min", type=float, default=0.05)
    args = ap.parse_args(argv)

    root = Path(args.root) if args.root else paths.subdir("sr2_scratch")
    names = [f"probe_{p}" for p in str(args.probes).replace(",", " ").split()]

    print(f"bracket verdict  (root {root})\n")
    print(f"{'probe':<9} {'verdict':<10} {'step':>6} {'final mse':>11} "
          f"{'worst mse':>11} {'max gap':>8} {'min d_loss':>11} {'handoff':>8}")
    results: Dict[str, Dict] = {}
    for n in names:
        p = root / n / "metrics.jsonl"
        if not p.is_file():
            print(f"{n:<9} {'PENDING':<10}  (no metrics.jsonl yet)")
            continue
        v = verdict_of(read_rows(p), mse_max=args.mse_max,
                       gap_max=args.gap_max, dloss_min=args.dloss_min)
        results[n] = v
        if v["verdict"] == "NO DATA":
            print(f"{n:<9} {'NO DATA':<10}")
            continue
        gap = "-" if v["max_gap"] is None else f"{v['max_gap']:8.1f}"
        dl = "-" if v["min_d_loss"] is None else f"{v['min_d_loss']:11.4g}"
        print(f"{n:<9} {v['verdict']:<10} {v['step']:6d} {v['final_mse']:11.4g} "
              f"{v['worst_mse']:11.4g} {gap:>8} {dl:>11} "
              f"{'yes' if v['handed_over'] else 'no':>8}")

    print()
    for n, v in results.items():
        print(f"  {n}: {v['verdict']} -- {v['why']}")

    ok = [n for n, v in results.items() if v["verdict"] == "OK"]
    print()
    if ok:
        print(f"SURVIVORS: {', '.join(ok)}")
        print("  Stable is not correct. Commit ONE of these to a long run, then")
        print("  read the field report's P/P_HR against the frozen column --")
        print("  frozen SR2 is density 0.977 / velocity 0.989, and trilinear")
        print("  upsampling already reaches density 0.706, so anything below")
        print("  that is worse than interpolation.")
    elif results:
        print("NO SURVIVOR. Do not commit a long run on any of these.")
        print("  If every probe SATURATED, the discriminator is too strong:")
        print("  lower --sr2-critic-lr, raise --instance-noise, or cut")
        print("  --w-sr2-adv. If every probe DIVERGED while w_mse_now was 0,")
        print("  the generator has nothing holding its scale -- probe B's form")
        print("  (MSE kept alongside) is the fallback, and it is a deliberate")
        print("  deviation from map2map rather than a reproduction of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
