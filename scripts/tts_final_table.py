#!/usr/bin/env python
"""The final test-time-scaling comparison table.

Puts every method on the same held-out boxes with the same metrics:

    sr2_single       one random SR2 draw (K=1)
    random_k         a random pick among K draws (same compute as best-of-K)
    handcrafted_k    best-of-K under the strongest single hand-crafted feature
    verifier_k       best-of-K under the learned pairwise ranker
    oracle_k         best-of-K under the statistical oracle (the ceiling)
    refine           verifier selection followed by noise refinement
    global_joint     verifier selection with overlap-consistent joint tiling

Selection-only methods are read off the Stage-1 candidate pool (no extra
inference); ``refine`` and ``global_joint`` come from ``scripts/tts_stage45.py``
and carry their own rows.

Reported for each: every density / displacement / velocity / higher-order metric
with 95% box-bootstrap intervals, remaining oracle regret, candidate diversity
before and after selection, wall clock, and quality against inference compute.
The ensemble-level power spectrum and log-density PDF are compared *before and
after* selection, because a selector that improves a metric by quietly shifting
the output distribution has not improved anything -- it has introduced a bias,
and that only shows up at the ensemble level.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

TABLE_METRICS = (
    "density_power_error", "density_pdf_error", "density_sigma_ratio",
    "bispectrum_equilateral_error", "bispectrum_squeezed_error",
    "velocity_power_error", "velocity_divergence_pdf_error",
    "density_rk_transition", "density_rk_high",
    "disp_rk_transition", "disp_rk_high", "disp_Tk_error_high",
    "lr_recon_rel_disp", "boundary_ratio",
)


def load_rows(path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="runs/tts_oracle/rows.jsonl")
    ap.add_argument("--verifier", default="runs/tts_verifier")
    ap.add_argument("--refine-rows", default="runs/tts_stage4/rows.jsonl")
    ap.add_argument("--global-rows", default="runs/tts_stage5/rows.jsonl")
    ap.add_argument("--profiles", default="runs/tts_oracle/profiles.npz")
    ap.add_argument("--box-summary", default="runs/tts_oracle/box_summary.json")
    ap.add_argument("--val-boxes", nargs="*", default=["set8", "set9", "set10", "set11"])
    ap.add_argument("--test-boxes", nargs="*", default=["set12", "set13", "set14", "set15"])
    ap.add_argument("--k-values", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--k-main", type=int, default=16)
    ap.add_argument("--n-repeats", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/tts_final")
    args = ap.parse_args()

    import torch
    from cosmo_sr.tts.bootstrap import best_of_k, bootstrap_ci, paired_bootstrap, subset_draws
    from cosmo_sr.tts.features import feature_matrix
    from cosmo_sr.tts.scores import (
        METRIC_DIRECTION, STATISTICAL_ORACLE_COMPONENTS, ScoreNormalizer,
        composite_score, derive_metrics,
    )
    from cosmo_sr.tts.verifier import FeatureRanker, Standardizer

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [derive_metrics(r) for r in load_rows(args.rows)]
    if not rows:
        raise SystemExit(f"no candidate rows in {args.rows}")
    test = set(args.test_boxes)

    norm = ScoreNormalizer.fit([r for r in rows if r["box"] in set(args.val_boxes)] or rows)
    for r in rows:
        r["oracle_score"] = composite_score(r, STATISTICAL_ORACLE_COMPONENTS, norm)

    # --- verifier + best hand-crafted feature ---------------------------- #
    vpath = Path(args.verifier)
    scores: Dict[str, np.ndarray] = {"random": np.zeros(len(rows))}
    if (vpath / "verifier.pt").exists():
        ckpt = torch.load(vpath / "verifier.pt", map_location="cpu", weights_only=False)
        std = Standardizer.load(vpath / "standardizer.json")
        model = FeatureRanker(len(ckpt["keys"]), hidden=ckpt["hidden"])
        model.load_state_dict(ckpt["model"]); model.eval()
        x = std(feature_matrix(rows, ckpt["keys"]))
        with torch.no_grad():
            scores["verifier"] = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
        report = json.loads((vpath / "verifier_report.json").read_text())
        best = report["best_handcrafted"]
        sign, key = (-1.0, best[1:]) if best.startswith("-") else (1.0, best)
        scores["handcrafted"] = sign * np.asarray([r.get(key, np.nan) for r in rows], float)
    else:
        print(f"[warn] no verifier at {vpath}; reporting oracle and random only")
    scores["oracle"] = np.asarray([r["oracle_score"] for r in rows], float)

    boxes = sorted({r["box"] for r in rows if r["box"] in test})
    groups = {b: np.asarray([i for i, r in enumerate(rows) if r["box"] == b]) for b in boxes}
    n_cand = min(len(g) for g in groups.values())
    k_values = [k for k in args.k_values if k <= n_cand] or [n_cand]
    k_main = args.k_main if args.k_main in k_values else max(k_values)
    rng = np.random.default_rng(args.seed)

    methods = ["sr2_single", "random_k"]
    methods += [m for m in ("handcrafted", "verifier") if m in scores]
    methods += ["oracle_k"]

    table: Dict[str, Dict[str, Dict]] = {}
    for metric in TABLE_METRICS:
        vals = np.asarray([r.get(metric, np.nan) for r in rows], float)
        for k in k_values:
            draws = {b: subset_draws(k, len(groups[b]), args.n_repeats, rng) for b in boxes}
            for method in methods:
                key = "random" if method in ("sr2_single", "random_k") else \
                      ("oracle" if method == "oracle_k" else method)
                kk = 1 if method == "sr2_single" else k
                if method == "sr2_single" and k != k_values[0]:
                    continue
                per_box = []
                for b in boxes:
                    g = groups[b]
                    if key == "random":
                        d = draws[b] if kk == k else subset_draws(1, len(g), args.n_repeats, rng)
                        per_box.append(float(np.nanmean(vals[g][d])))
                    else:
                        mean, _ = best_of_k(vals[g], scores[key][g], k, draws=draws[b])
                        per_box.append(mean)
                table.setdefault(metric, {}).setdefault(method, {})[kk] = {
                    **bootstrap_ci(per_box, n_boot=args.n_boot, rng=rng), "per_box": per_box,
                }

    # --- extra-compute methods from their own rows ----------------------- #
    for method, path in (("refine", args.refine_rows), ("global_joint", args.global_rows)):
        extra = [derive_metrics(r) for r in load_rows(path) if r.get("box") in test]
        if not extra:
            continue
        methods.append(method)
        by_box: Dict[str, List[Dict]] = {}
        for r in extra:
            by_box.setdefault(r["box"], []).append(r)
        for metric in TABLE_METRICS:
            per_box = [
                float(np.nanmean([r.get(metric, np.nan) for r in by_box[b]]))
                for b in sorted(by_box)
            ]
            table.setdefault(metric, {}).setdefault(method, {})[k_main] = {
                **bootstrap_ci(per_box, n_boot=args.n_boot, rng=rng), "per_box": per_box,
            }

    # --- regret, diversity, compute -------------------------------------- #
    summary: Dict[str, Dict] = {}
    for method in methods:
        entry: Dict[str, float] = {}
        k = 1 if method == "sr2_single" else k_main
        key = "random" if method in ("sr2_single", "random_k") else \
              ("oracle" if method == "oracle_k" else method)
        if key in scores and method not in ("sr2_single", "random_k", "refine", "global_joint"):
            draws = {b: subset_draws(k, len(groups[b]), args.n_repeats, rng) for b in boxes}
            sel, orc = [], []
            for b in boxes:
                g = groups[b]
                m_sel, _ = best_of_k(scores["oracle"][g], scores[key][g], k, draws=draws[b])
                m_orc, _ = best_of_k(scores["oracle"][g], scores["oracle"][g], k, draws=draws[b])
                sel.append(m_sel); orc.append(m_orc)
            entry["oracle_regret"] = float(np.mean(np.asarray(sel) - np.asarray(orc)))
        summary[method] = entry

    box_summary = json.loads(Path(args.box_summary).read_text()) \
        if Path(args.box_summary).exists() else {}
    diversity_before = {b: box_summary.get(b, {}) for b in boxes}

    # Candidate diversity after selection: how much of the pool a selector
    # actually reaches, measured as the fraction of distinct candidates it picks
    # across many random size-K subsets. A selector locked onto one kind of
    # realisation collapses this even when the pool is diverse.
    #
    # It has to be measured at K strictly below the pool size: at K = pool there
    # is exactly one subset, so every selector trivially scores 1/pool and the
    # number says nothing. Fall back to half the pool and record which K was used.
    k_div = k_main if k_main < n_cand else max(1, n_cand // 2)
    diversity_after: Dict[str, float] = {}
    for method in methods:
        key = "random" if method in ("sr2_single", "random_k") else \
              ("oracle" if method == "oracle_k" else method)
        if key not in scores:
            continue
        spread = []
        for b in boxes:
            g = groups[b]
            draws = subset_draws(k_div, len(g), args.n_repeats, rng)
            if key == "random":
                picked = draws[:, 0]
            else:
                s = np.where(np.isfinite(scores[key][g]), scores[key][g], np.inf)
                picked = draws[np.arange(len(draws)), np.argmin(s[draws], axis=1)]
            spread.append(float(len(np.unique(picked)) / len(g)))
        diversity_after[method] = float(np.mean(spread))

    wall = {}
    for method in methods:
        if method in ("refine", "global_joint"):
            path = args.refine_rows if method == "refine" else args.global_rows
            ws = [r.get("wall_s", np.nan) for r in load_rows(path) if r.get("box") in test]
            wall[method] = float(np.nanmean(ws)) if ws else float("nan")
        else:
            k = 1 if method == "sr2_single" else k_main
            base = float(np.nanmean([r.get("wall_s", np.nan) for r in rows]))
            wall[method] = base * k

    peak = {}
    base_peak = float(np.nanmean([r.get("peak_mem_gb", np.nan) for r in rows]))
    for method in methods:
        if method in ("refine", "global_joint"):
            path = args.refine_rows if method == "refine" else args.global_rows
            ms = [r.get("peak_mem_gb", np.nan) for r in load_rows(path) if r.get("box") in test]
            peak[method] = float(np.nanmean(ms)) if ms else float("nan")
        else:
            # candidates are generated one at a time, so peak memory is per
            # candidate and does not grow with K
            peak[method] = base_peak

    result = {
        "boxes": boxes, "n_candidates": n_cand, "k_values": k_values, "k_main": k_main,
        "methods": methods, "table": table, "summary": summary,
        "diversity_before": diversity_before, "diversity_after_unique_frac": diversity_after,
        "diversity_measured_at_k": k_div,
        "wall_s_per_box": wall, "peak_mem_gb": peak,
    }
    result["distribution_bias"] = _distribution_bias(args, rows, groups, scores, k_main, rng)
    (out / "final_table.json").write_text(json.dumps(result, indent=2, default=float))
    _plot(out, result, rows, boxes, groups, scores, k_values)

    # --- print ------------------------------------------------------------ #
    print(f"\ntest boxes: {boxes}   candidates/box: {n_cand}   K_main: {k_main}\n")
    head = f"{'metric':34s}" + "".join(f"{m:>18s}" for m in methods)
    print(head); print("-" * len(head))
    for metric in TABLE_METRICS:
        arrow = "^" if METRIC_DIRECTION.get(metric, -1) > 0 else "v"
        line = f"{metric + ' (' + arrow + ')':34s}"
        for m in methods:
            k = 1 if m == "sr2_single" else k_main
            cell = table.get(metric, {}).get(m, {}).get(k)
            line += f"{cell['mean']:18.4g}" if cell else f"{'-':>18s}"
        print(line)
    print("\nremaining oracle regret (statistical composite; 0 = perfect selection):")
    for m, e in summary.items():
        if "oracle_regret" in e:
            print(f"  {m:16s} {e['oracle_regret']:+.4f}")
    print(f"\nselection diversity at K={k_div} "
          f"(fraction of the {n_cand}-candidate pool actually chosen):")
    for m, v in diversity_after.items():
        print(f"  {m:16s} {v:.3f}")
    print("\nwall clock per box (s) and peak memory (GB):")
    for m in methods:
        print(f"  {m:16s} {wall.get(m, float('nan')):10.1f}   "
              f"{peak.get(m, float('nan')):8.2f}")
    print(f"\nwrote {out}/final_table.json, quality_vs_compute.png, ensemble_bias.png")


def _distribution_bias(args, rows, groups, scores, k_main, rng) -> Dict:
    """Ensemble density power / PDF before vs after selection."""
    from cosmo_sr.tts.bootstrap import subset_draws

    path = Path(args.profiles)
    if not path.exists():
        return {}
    prof = np.load(path)
    out: Dict[str, Dict] = {}
    for key in scores:
        pk_sel, pdf_sel, pk_all, pdf_all = [], [], [], []
        for b, g in groups.items():
            names = [(rows[i]["box"], rows[i]["seed"]) for i in g]
            pks = [prof.get(f"{bb}|{ss}|pk") for bb, ss in names]
            pdfs = [prof.get(f"{bb}|{ss}|pdf") for bb, ss in names]
            if any(p is None for p in pks):
                return {}
            draws = subset_draws(k_main, len(g), args.n_repeats, rng)
            s = np.where(np.isfinite(scores[key][g]), scores[key][g], np.inf)
            picked = (draws[:, 0] if key == "random"
                      else draws[np.arange(len(draws)), np.argmin(s[draws], axis=1)])
            pk_sel.append(np.mean([pks[i] for i in picked], axis=0))
            pdf_sel.append(np.mean([pdfs[i] for i in picked], axis=0))
            pk_all.append(np.mean(pks, axis=0))
            pdf_all.append(np.mean(pdfs, axis=0))
        pk_s, pk_a = np.mean(pk_sel, axis=0), np.mean(pk_all, axis=0)
        pdf_s, pdf_a = np.mean(pdf_sel, axis=0), np.mean(pdf_all, axis=0)
        out[key] = {
            "pk_log_shift_mean": float(np.abs(np.log(np.maximum(pk_s, 1e-30) /
                                                     np.maximum(pk_a, 1e-30))).mean()),
            "pdf_l1_shift": float(np.abs(pdf_s - pdf_a).sum()),
            "pk_selected": pk_s.tolist(), "pk_ensemble": pk_a.tolist(),
        }
    return out


def _plot(out, result, rows, boxes, groups, scores, k_values) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["density_power_error", "bispectrum_equilateral_error",
               "velocity_power_error", "density_pdf_error"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 3.6))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for method, style in (("random_k", "k--"), ("handcrafted", "C2-^"),
                              ("verifier", "C0-o"), ("oracle_k", "C3-s")):
            cells = result["table"].get(metric, {}).get(method, {})
            if not cells:
                continue
            ks = sorted(int(k) for k in cells)
            base = float(np.nanmean([r.get("wall_s", 1.0) for r in rows]))
            ax.plot([k * base for k in ks], [cells[k]["mean"] for k in ks], style,
                    label=method, ms=4)
            ax.fill_between([k * base for k in ks], [cells[k]["lo"] for k in ks],
                            [cells[k]["hi"] for k in ks], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("inference wall clock per box [s]")
        ax.set_title(metric, fontsize=9)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("error (lower is better)")
    fig.suptitle("SR2 test-time scaling: quality vs inference compute "
                 "(95% box bootstrap)", y=1.02)
    fig.tight_layout()
    fig.savefig(Path(out) / "quality_vs_compute.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    bias = result.get("distribution_bias") or {}
    if bias:
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, d in bias.items():
            pk_s = np.asarray(d["pk_selected"]); pk_a = np.asarray(d["pk_ensemble"])
            ax.plot(np.arange(len(pk_s)), pk_s / np.maximum(pk_a, 1e-30), "-o",
                    ms=3, label=key)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xlabel("k bin"); ax.set_ylabel("P_selected / P_ensemble")
        ax.set_title("Selection-induced distribution bias (density power)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(out) / "ensemble_bias.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
