#!/usr/bin/env python
"""Experiment 1 deliverable: recovery and reward versus alpha, plus the verdict.

Reads only the JSONL rows the intervention jobs wrote, so the figures can be
redrawn without re-running a single halo finder.

The verdict follows the interpretation table in the brief, and it is decided on
three comparisons that are easy to conflate:

1. **targeted vs control at the same alpha** -- if the equal-particle-count
   random edit inside the same host recovers just as much, the targeted result
   is added fluctuation power, not structure. This gate comes first, because a
   positive result that fails it is worse than a null.
2. **shape of recovery vs alpha** -- smooth growth means the landscape has an
   accessible direction; a step at alpha = 1 means the representation works but
   exploration will be hard.
3. **disp vs vel vs both** -- whether phase-space coherence has to be modelled.

    python scripts/reward/oracle_hr_report.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from _common import (  # noqa: E402
    add_common_args, banner, bins_of, load_reward_config, paths, read_jsonl,
    write_json,
)


def collect(boxes: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for b in boxes:
        p = paths.subdir("oracle_hr", b) / "interventions.jsonl"
        if not p.is_file():
            print(f"!! no rows for {b}: {p}", flush=True)
            continue
        # Re-runs append, so keep the last row per (kind, mode, alpha).
        latest: Dict = {}
        for r in read_jsonl(p):
            latest[(r["kind"], r["mode"], float(r["alpha"]))] = r
        rows.extend(latest.values())
    return rows


def curve(rows, kind, mode):
    sel = [r for r in rows if r["kind"] == kind and r["mode"] == mode]
    by_alpha = defaultdict(list)
    for r in sel:
        by_alpha[float(r["alpha"])].append(r)
    out = []
    for a in sorted(by_alpha):
        rr = by_alpha[a]
        out.append({
            "alpha": a,
            "n_boxes": len(rr),
            "recovery_rate": float(np.mean([x["recovery_rate"] for x in rr])),
            "recovery_sem": (float(np.std([x["recovery_rate"] for x in rr], ddof=1)
                                   / np.sqrt(len(rr))) if len(rr) > 1 else float("nan")),
            "n_recovered": int(sum(x["n_recovered"] for x in rr)),
            "n_targets": int(sum(x["n_targets"] for x in rr)),
            "R_occ": float(np.mean([x.get("R_occ", np.nan) for x in rr])),
            "R_cat": float(np.mean([x.get("R_cat", np.nan) for x in rr])),
            "occupation": np.nanmean(
                np.array([x["occupation"] for x in rr], dtype=float), axis=0).tolist(),
            "feasible": all(x.get("feasible", True) for x in rr),
        })
    return out


def plot(curves, out_png, bins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, c in sorted(curves.items()):
        if not c:
            continue
        kind, mode = key
        a = [x["alpha"] for x in c]
        style = dict(marker="o", lw=1.8,
                     ls="--" if kind == "control" else "-",
                     alpha=0.65 if kind == "control" else 1.0)
        ax[0].errorbar(a, [x["recovery_rate"] for x in c],
                       yerr=[x["recovery_sem"] for x in c],
                       label=f"{kind}/{mode}", capsize=2, **style)
        ax[1].plot(a, [x["R_occ"] for x in c], label=f"{kind}/{mode}", **style)
    ax[0].set_xlabel(r"$\alpha$")
    ax[0].set_ylabel("fraction of targets recovered")
    ax[0].set_title("Recovery vs injected HR-residual fraction")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel(r"$\alpha$")
    ax[1].set_ylabel(r"$R_{\rm occ}$")
    ax[1].set_title("Occupation reward vs " + r"$\alpha$")
    ax[1].grid(alpha=0.3)

    centres = np.asarray(bins.host_mass_edges[:-1], dtype=float)
    for key, c in sorted(curves.items()):
        if key[0] != "targeted" or not c:
            continue
        for x in c:
            if x["alpha"] in (0.0, 1.0):
                ax[2].plot(centres, x["occupation"], marker="o",
                           label=f"{key[1]} " + r"$\alpha$=" + f"{x['alpha']:g}")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$M_{\rm host}\ [M_\odot/h]$")
    ax[2].set_ylabel(r"$\langle N_{\rm sub}\,|\,M_{\rm host}\rangle$")
    ax[2].set_title("Occupation function")
    ax[2].grid(alpha=0.3, which="both")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"figure -> {out_png}")


def verdict(curves) -> Dict:
    """The interpretation table, applied."""
    def rate(kind, mode, a):
        for x in curves.get((kind, mode), []):
            if abs(x["alpha"] - a) < 1e-9:
                return x["recovery_rate"]
        return float("nan")

    modes = sorted({m for (_, m) in curves})
    out = {"per_mode": {}}
    best_mode, best_gain = None, -np.inf
    for m in modes:
        c = curves.get(("targeted", m), [])
        if not c:
            continue
        alphas = [x["alpha"] for x in c]
        rates = [x["recovery_rate"] for x in c]
        r0 = rate("targeted", m, 0.0)
        r1 = rate("targeted", m, 1.0)
        ctrl1 = rate("control", m, 1.0)
        mid = [r for a, r in zip(alphas, rates) if 0.0 < a < 1.0]
        gain = (r1 - r0) if np.isfinite(r1) and np.isfinite(r0) else np.nan
        gradual = bool(mid and np.nanmax(mid) > r0 + 0.25 * max(gain, 1e-9))
        beats_control = bool(np.isfinite(ctrl1) and r1 > ctrl1 + 0.1) or \
            (not np.isfinite(ctrl1))
        out["per_mode"][m] = {
            "recovery_at_0": r0, "recovery_at_1": r1,
            "control_at_1": ctrl1, "gain": gain,
            "gradual": gradual, "beats_control": beats_control,
        }
        if np.isfinite(gain) and gain > best_gain:
            best_gain, best_mode = gain, m

    if best_mode is None or not np.isfinite(best_gain):
        out["verdict"] = "incomplete"
        out["next"] = "not enough cells scored to decide"
        return out

    b = out["per_mode"][best_mode]
    if best_gain <= 0.05:
        out["verdict"] = "localised_hr_correction_fails"
        out["next"] = ("Even the true HR correction, applied locally, does not "
                       "restore the missing subhalos. The residual/action "
                       "representation is insufficient -- move to the explicit "
                       "catalog-renderer oracle (decision table, last row). Do "
                       "NOT spend GPU on best-of-K.")
    elif not b["beats_control"]:
        out["verdict"] = "control_matches_targeted"
        out["next"] = ("The equal-particle-count random edit recovers as much as "
                       "the targeted one, so the gain is added fluctuation power, "
                       "not structure. Tighten the recovery criterion and re-run "
                       "before drawing any conclusion.")
    elif b["gradual"]:
        out["verdict"] = "accessible_direction"
        out["next"] = ("Recovery grows with alpha: the reward landscape has an "
                       "accessible direction. Continue to search (Experiment 2) "
                       "and reward training.")
    else:
        out["verdict"] = "representation_ok_exploration_hard"
        out["next"] = ("Recovery appears only near alpha = 1: the representation "
                       "works but ordinary sampling is unlikely to find it. "
                       "Prefer directed search (Experiment 3, CEM) over raw "
                       "best-of-K.")

    channels = out["per_mode"]
    if "disp" in channels and "vel" in channels:
        d, v = channels["disp"]["gain"], channels["vel"]["gain"]
        if np.isfinite(d) and np.isfinite(v):
            if d > 0.05 and v <= 0.05:
                out["channel_finding"] = ("position works, velocity does not: the "
                                          "correction is mainly structural")
            elif "both" in channels and channels["both"]["gain"] > max(d, v) + 0.1:
                out["channel_finding"] = ("position AND velocity are required: "
                                          "phase-space coherence must be modelled "
                                          "explicitly")
            else:
                out["channel_finding"] = "no clear channel separation"
    out["best_mode"] = best_mode
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    rows = collect(boxes)
    if not rows:
        raise SystemExit("no intervention rows found; run oracle_intervene.py first")

    kinds = sorted({r["kind"] for r in rows})
    modes = sorted({r["mode"] for r in rows})
    curves = {(k, m): curve(rows, k, m) for k in kinds for m in modes}
    curves = {k: v for k, v in curves.items() if v}

    banner("Experiment 1: recovery vs alpha")
    head = (f"{'kind':>16s} {'mode':>5s} {'alpha':>6s} {'boxes':>6s} "
            f"{'recovered':>11s} {'rate':>6s} {'R_occ':>9s} {'feasible':>9s}")
    print(head)
    print("-" * len(head))
    for (k, m), c in sorted(curves.items()):
        for x in c:
            print(f"{k:>16s} {m:>5s} {x['alpha']:6.2f} {x['n_boxes']:>6d} "
                  f"{x['n_recovered']:>5d}/{x['n_targets']:<5d} "
                  f"{x['recovery_rate']:6.3f} {x['R_occ']:9.3f} "
                  f"{str(x['feasible']):>9s}")

    banner("Occupation function, whole box")
    labels = [f"{e:.2e}" for e in bins.host_mass_edges[:-1]]
    print(f"{'kind/mode/alpha':>24s} " + " ".join(f"{l:>9s}" for l in labels))
    for (k, m), c in sorted(curves.items()):
        for x in c:
            key = f"{k}/{m}/{x['alpha']:g}"
            print(f"{key:>24s} " + " ".join(
                f"{v:9.2f}" if np.isfinite(v) else f"{'-':>9s}"
                for v in x["occupation"]))

    v = verdict(curves)
    banner(f"VERDICT: {v['verdict']}")
    for m, d in sorted(v.get("per_mode", {}).items()):
        print(f"  {m:>5s}: recovery {d['recovery_at_0']:.3f} -> {d['recovery_at_1']:.3f} "
              f"(control {d['control_at_1']:.3f}), "
              f"gradual={d['gradual']}, beats_control={d['beats_control']}")
    if "channel_finding" in v:
        print(f"  channels: {v['channel_finding']}")
    print(f"\n  next: {v['next']}")

    out_dir = Path(args.out) if args.out else paths.subdir("oracle_hr", create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "exp1_report.json",
               {"boxes": boxes, "rows": rows,
                "curves": {f"{k}/{m}": c for (k, m), c in curves.items()},
                "verdict": v})
    try:
        plot(curves, out_dir / "exp1_recovery_vs_alpha.png", bins)
    except Exception as e:                       # noqa: BLE001
        print(f"!! plotting failed ({e}); the JSON report is still complete")
    print(f"\nreport -> {out_dir / 'exp1_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
