#!/usr/bin/env python
"""Bug-or-genuine: why does within-tile ranking sit at ~0.25 across arms?

Reads ONLY ``ranking_arrays_{arm}.npz`` (saved by ``gate_catalog_proxy.py`` at
line 379: ``mean, std, true, box, tile_id, unit, source, alpha``), so it recomputes
nothing on a GPU and is a cheap CPU read. It answers two questions the gate's
pass/fail cannot:

1. **Is the true target rankable within a tile at all?** A numerator-count reward
   on a tile holding a fraction of one host is near-quantised: most candidate pairs
   in a unit differ by *zero* true reward, i.e. they are genuine ties, and no proxy
   can order a tie. We measure this directly as a variance decomposition of the
   TRUE dR (within-unit vs across-unit) and as the fraction of within-unit
   candidate pairs separated by more than the label-noise floor.

2. **Does the proxy collapse candidates?** The same variance decomposition of the
   PREDICTED dR. If the prediction has near-zero within-unit spread while the truth
   does not, the count->dR reduction -- not the features -- is the ceiling, and
   that would be the closest thing to a bug (it also explains A ~= B ~= ... ~= D:
   features cannot matter if dR washes them out).

Plus the honest ranking score as a function of tie margin (the gate's
``ranking_margin_sweep``, recomputed here so both arms sit in one figure), split
by ``source`` because the intervention alpha-ladder is monotone by construction
and the generated candidates are the real test.

    python scripts/reward/diagnose_rank_ceiling.py --run-name direct_a --arms a,b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from _proxy_data import group_indices, rank_metrics, robust_scale  # noqa: E402
from _sr2_direct import banner, run_dir, write_json_atomic  # noqa: E402


def selection_lift(pred: np.ndarray, true: np.ndarray,
                   top_fraction: float = 0.2) -> Dict[str, float]:
    """selected_positive_fraction next to its base rate -- the pass is only real
    as the gap between them (picking at random on a clean pool clears 0.6 alone)."""
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < 5:
        return {"selected_positive_fraction": float("nan"),
                "baseline_positive_fraction": float("nan"), "lift": float("nan")}
    p, t = pred[ok], true[ok]
    k = max(1, int(round(top_fraction * p.size)))
    top = np.argsort(-p)[:k]
    sel = float(np.mean(t[top] > 0))
    base = float(np.mean(t > 0))
    return {"selected_positive_fraction": sel, "baseline_positive_fraction": base,
            "lift": sel - base, "n": int(p.size), "n_selected": int(k)}


def variance_decomposition(value: np.ndarray, box: np.ndarray, unit: np.ndarray,
                           keep: np.ndarray) -> Dict[str, float]:
    """Split a quantity's variance into within-unit and across-unit parts.

    Only groups with >= 2 kept members contribute -- a singleton unit has no
    within-unit variance and no across-unit rank to place. ``within_fraction`` is
    the share of total variance that lives *inside* units, i.e. the share a
    within-tile ranker could ever explain. Near 0 for the TRUE dR means there is
    nothing to rank; near 0 for the PREDICTED dR means the proxy collapses a unit
    to one number.
    """
    within_vars, group_means, ns = [], [], []
    for idx in group_indices(box, unit).values():
        idx = [i for i in idx if keep[i]]
        if len(idx) < 2:
            continue
        v = value[np.asarray(idx)]
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        within_vars.append(float(np.var(v)))
        group_means.append(float(np.mean(v)))
        ns.append(v.size)
    if not within_vars:
        return {"within_var": float("nan"), "across_var": float("nan"),
                "within_fraction": float("nan"), "n_units": 0}
    w = float(np.average(within_vars, weights=ns))
    a = float(np.var(group_means))
    tot = w + a
    return {"within_var": w, "across_var": a,
            "within_fraction": (w / tot if tot > 0 else float("nan")),
            "within_std": float(np.sqrt(w)), "across_std": float(np.sqrt(a)),
            "n_units": len(within_vars)}


def tie_fraction(true: np.ndarray, box: np.ndarray, unit: np.ndarray,
                 keep: np.ndarray, margin: float) -> Dict[str, float]:
    """Of all within-unit candidate pairs, how many are true ties (|dTrue| <= margin)?

    This is the ceiling the ranking metric is really up against: a tied pair is
    unrankable by anything, so 1 - tie_fraction bounds the achievable pairwise
    accuracy above chance.
    """
    n_pairs = n_tied = 0
    for idx in group_indices(box, unit).values():
        idx = [i for i in idx if keep[i] and np.isfinite(true[i])]
        if len(idx) < 2:
            continue
        t = true[np.asarray(idx)]
        d = np.abs(t[:, None] - t[None, :])
        iu = np.triu_indices(len(idx), k=1)
        dd = d[iu]
        n_pairs += dd.size
        n_tied += int(np.sum(dd <= margin))
    return {"n_pairs": int(n_pairs), "tie_fraction": (n_tied / n_pairs if n_pairs
            else float("nan")), "rankable_fraction": (1 - n_tied / n_pairs
            if n_pairs else float("nan"))}


COUNT_KEYS = ("n_sub", "n_host", "occ_numerator")


def count_decomposition(counts_npz: Path, box: np.ndarray, unit: np.ndarray,
                        keep: np.ndarray) -> Dict[str, Dict]:
    """Within-tile variance of PREDICTED vs TRUE per-candidate counts.

    Sums each count over its host-mass bins to one scalar per candidate, then asks
    the same within-unit question as for dR. If the TRUE count varies within a
    tile but the PREDICTED count does not, the collapse is already in the counts --
    the proxy (or its features) is insensitive to the seed, and the count->dR
    reduction is not the culprit.
    """
    if not counts_npz.is_file():
        return {"note": f"no {counts_npz.name}; re-run the gate to save counts"}
    c = np.load(counts_npz, allow_pickle=True)
    out: Dict[str, Dict] = {}
    for k in COUNT_KEYS:
        pk, tk = f"pred_{k}", f"true_{k}"
        if pk not in c or tk not in c:
            continue
        pred = np.asarray(c[pk], dtype=np.float64)
        true = np.asarray(c[tk], dtype=np.float64)
        pred = pred.sum(axis=1) if pred.ndim > 1 else pred
        true = true.sum(axis=1) if true.ndim > 1 else true
        out[k] = {
            "pred_within_fraction": variance_decomposition(
                pred, box, unit, keep)["within_fraction"],
            "true_within_fraction": variance_decomposition(
                true, box, unit, keep)["within_fraction"],
        }
    return out


def analyse_arm(arm: str, npz: Path, sources: List[str]) -> Dict:
    d = np.load(npz, allow_pickle=True)
    mean, true = d["mean"], d["true"]
    box, unit, source = d["box"].astype(str), d["unit"], d["source"].astype(str)
    counts_npz = npz.with_name(f"ranking_counts_{arm}.npz")
    interv = source == "intervention"
    noise = float(robust_scale(true[interv & np.isfinite(true)])) if interv.any() \
        else float(robust_scale(true))
    margins = sorted({0.0, 0.5 * noise, noise, 2 * noise, 4 * noise})

    out: Dict = {"arm": arm, "n_rows": int(mean.size), "noise_floor": noise,
                 "source_counts": {s: int((source == s).sum())
                                   for s in sorted(set(source))}, "by_source": {}}
    for name in sources:
        keep = np.ones(mean.size, bool) if name == "all" else (source == name)
        if keep.sum() < 2:
            out["by_source"][name] = {"n": int(keep.sum()), "note": "too few rows"}
            continue
        sweep = {}
        for mm in margins:
            r = rank_metrics(mean[keep], true[keep], box[keep], unit[keep],
                             min_margin=mm)
            sweep[f"{mm:.4g}"] = {"within_tile_spearman": r["within_tile_spearman"],
                                  "pairwise_accuracy": r["pairwise_accuracy"],
                                  "n_groups": r["n_groups"], "n_pairs": r["n_pairs"]}
        out["by_source"][name] = {
            "n": int(keep.sum()),
            "ranking_margin_sweep": sweep,
            "true_variance_decomp": variance_decomposition(true, box, unit, keep),
            "pred_variance_decomp": variance_decomposition(mean, box, unit, keep),
            "count_decomp": count_decomposition(counts_npz, box, unit, keep),
            "ties": {f"{mm:.4g}": tie_fraction(true, box, unit, keep, mm)
                     for mm in (noise, 2 * noise)},
            "selection": selection_lift(mean[keep], true[keep]),
        }
    return out


def draw(results: List[Dict], out_png: Path, src: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  (figure skipped: {exc})", flush=True)
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, res in enumerate(results):
        c = cmap(i % 10)
        s = res["by_source"].get(src)
        if not s or "ranking_margin_sweep" not in s:
            continue
        sw = s["ranking_margin_sweep"]
        xs = sorted(float(k) for k in sw)
        nf = res["noise_floor"] or 1.0
        xr = [x / nf for x in xs]
        axes[0].plot(xr, [sw[f"{x:.4g}"]["within_tile_spearman"] for x in xs],
                     "-o", color=c, label=f"arm {res['arm']}", ms=4)
        axes[1].plot(xr, [sw[f"{x:.4g}"]["pairwise_accuracy"] for x in xs],
                     "-o", color=c, label=f"arm {res['arm']}", ms=4)
        tv = s["true_variance_decomp"]["within_fraction"]
        pv = s["pred_variance_decomp"]["within_fraction"]
        axes[2].bar(i - 0.2, tv, width=0.38, color=c)
        axes[2].bar(i + 0.2, pv, width=0.38, color=c, alpha=0.45)
    axes[0].axhline(0.5, color="k", ls=":", lw=1); axes[0].set_ylabel("within-tile Spearman")
    axes[1].axhline(0.65, color="k", ls=":", lw=1); axes[1].axhline(0.5, color="gray", ls=":", lw=0.8)
    axes[1].set_ylabel("pairwise accuracy")
    for ax in axes[:2]:
        ax.set_xlabel("tie margin (units of noise floor)"); ax.legend(fontsize=8)
    axes[2].set_ylabel("within-unit fraction of variance")
    axes[2].set_xticks(range(len(results)))
    axes[2].set_xticklabels([f"arm {r['arm']}\n(L=true R=pred)" for r in results], fontsize=8)
    axes[2].set_title("solid=true dR, faded=predicted dR")
    fig.suptitle(f"within-tile ranking ceiling, source={src}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--arms", default="a,b")
    # frozen_seed = the frozen SR2 re-sampled at different seeds: the within-tile
    # candidates an actor actually produces, and the real ranking test. frozen/hr
    # are single-per-unit anchors; intervention is the monotone alpha ladder.
    ap.add_argument("--sources", default="all,frozen_seed,intervention")
    ap.add_argument("--main-source", default="frozen_seed",
                    help="the source the figure and verdict summarise")
    args = ap.parse_args(argv)

    run = run_dir(args.run_name, create=True)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.main_source not in sources:
        sources.append(args.main_source)
    results, missing = [], []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        npz = run / f"ranking_arrays_{arm}.npz"
        if not npz.is_file():
            missing.append(str(npz))
            continue
        results.append(analyse_arm(arm, npz, sources))

    if missing:
        print(">>> missing (run gate_catalog_proxy.py for these arms first):")
        for m in missing:
            print(f"    {m}")
    if not results:
        print(">>> no ranking arrays found; nothing to diagnose.")
        return 0

    out_json = run / "rank_ceiling_diagnosis.json"
    write_json_atomic(out_json, {"run_name": args.run_name, "arms": results})
    draw(results, run / "rank_ceiling_diagnosis.png", args.main_source)

    # A compact verdict per arm, on the frozen-seed candidates (the real test).
    summary = []
    for res in results:
        g = res["by_source"].get(args.main_source, {})
        if "true_variance_decomp" not in g:
            continue
        tv = g["true_variance_decomp"]["within_fraction"]
        pv = g["pred_variance_decomp"]["within_fraction"]
        nf = res["noise_floor"]
        tie = g["ties"].get(f"{nf:.4g}", {}).get("tie_fraction", float("nan"))
        sw = g["ranking_margin_sweep"]
        rho0 = sw.get("0", {}).get("within_tile_spearman", float("nan"))
        rho_hi = sw[f"{4 * nf:.4g}"]["within_tile_spearman"] if f"{4 * nf:.4g}" in sw \
            else float("nan")
        cd = g.get("count_decomp", {})
        nsub = cd.get("n_sub", {}) if isinstance(cd, dict) else {}
        pred_nsub = nsub.get("pred_within_fraction", float("nan"))
        true_nsub = nsub.get("true_within_fraction", float("nan"))
        if tv < 0.15:
            verdict = "GENUINE: true dR has ~no within-tile signal -- nothing to rank"
        elif pv < 0.15:
            # Localise by whether the PREDICTED counts retain the (small but real)
            # within-tile variance the TRUE counts have. pred << true => the
            # collapse is already in the counts (proxy/features seed-insensitive);
            # pred ~ true but dR still flat => the count->dR reduction flattens it.
            if not (np.isfinite(pred_nsub) and np.isfinite(true_nsub)):
                where = "re-run the gate to save ranking_counts_*.npz to localise it"
            elif true_nsub <= 0.005:
                where = (f"little within-tile count signal to begin with "
                         f"(true n_sub within-frac {true_nsub:.3f})")
            elif pred_nsub < 0.3 * true_nsub:
                where = ("COUNTS: predicted counts are flat within a tile "
                         f"(n_sub within-frac pred {pred_nsub:.3f} vs true "
                         f"{true_nsub:.3f}) -- proxy/features seed-insensitive, "
                         "NOT the reduction")
            else:
                where = ("REDUCTION: predicted counts keep within-tile variance "
                         f"(pred {pred_nsub:.3f} vs true {true_nsub:.3f}) but dR "
                         "does not -- the count->dR step flattens them")
            verdict = f"COLLAPSE: proxy predicts ~one value per tile. {where}"
        elif np.isfinite(rho_hi) and np.isfinite(rho0) and rho_hi - rho0 > 0.15:
            verdict = "NOISE-LIMITED: rho climbs with margin -- signal real, buried in label ties"
        else:
            verdict = "GAP: rankable structure exists but proxy does not capture it"
        summary.append({"arm": res["arm"], "true_within_frac": round(tv, 3),
                        "pred_within_frac": round(pv, 3),
                        "pred_nsub_within_frac": round(pred_nsub, 3),
                        "true_nsub_within_frac": round(true_nsub, 3),
                        "tie_frac_at_noise": round(tie, 3),
                        "rho_margin0": round(rho0, 3), "rho_margin4x": round(rho_hi, 3),
                        "sel_lift": round(g["selection"]["lift"], 3), "verdict": verdict})

    banner(json.dumps(summary, indent=2))
    print(f"  diagnosis -> {out_json}")
    print(f"  figure    -> {run / 'rank_ceiling_diagnosis.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
