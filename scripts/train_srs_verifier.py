#!/usr/bin/env python
"""Stage 2: train and evaluate a test-time-available selector for SR2 candidates.

Consumes the ``rows.jsonl`` written by ``scripts/eval_srs_tts.py`` (which records
both the HR-referenced metrics and the HR-free features for every candidate) and
asks whether anything computable at test time can recover the statistical
oracle's best-of-K gain.

Order of business, cheapest first:

1. **Single hand-crafted features.** Each feature is used directly as a selector.
   If one of them already recovers most of the oracle gain, no model is needed.
2. **A pairwise ranker** over all summary features (linear, then a small MLP),
   trained on candidate *pairs* from the same LR input with a RankNet loss and a
   statistical-oracle target.

Everything is split **by simulation box**: train / val / test boxes never mix,
because candidates from one box share its large-scale modes.

Decision gate (reported at the end):

* at K=16 the selector must beat random selection with a box-bootstrap CI
  excluding zero, and
* recover at least ``--recover-frac`` (default 50%) of the statistical-oracle
  gain, and
* not damage power, bispectra, velocities or diversity (checked and printed).

Example::

    python scripts/train_srs_verifier.py \
        --rows runs/tts_oracle/rows.jsonl \
        --train-boxes set0 ... set7 --val-boxes set8 ... --test-boxes set12 ... \
        --out runs/tts_verifier
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

#: Metrics that must not get worse while the selector improves its target.
GUARD_METRICS = (
    "density_power_error", "velocity_power_error",
    "bispectrum_equilateral_error", "bispectrum_squeezed_error",
)


def load_rows(path: Path) -> List[Dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def group_indices(rows: Sequence[Dict], boxes: Sequence[str]) -> List[List[int]]:
    """Row indices grouped by box (one group = one LR input's candidate set)."""
    by: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        if r["box"] in set(boxes):
            by.setdefault(r["box"], []).append(i)
    return [by[b] for b in sorted(by)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="runs/tts_oracle/rows.jsonl")
    ap.add_argument("--train-boxes", nargs="*", default=[f"set{i}" for i in range(8)])
    ap.add_argument("--val-boxes", nargs="*", default=["set8", "set9", "set10", "set11"])
    ap.add_argument("--test-boxes", nargs="*", default=["set12", "set13", "set14", "set15"])
    ap.add_argument("--k-gate", type=int, default=16)
    ap.add_argument("--k-values", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--hidden", nargs="*", type=int, default=[32])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--recover-frac", type=float, default=0.5)
    ap.add_argument("--n-repeats", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--report-metric", default="statistical_oracle_score",
                    help="quality reported vs K (default: the statistical-oracle composite)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/tts_verifier")
    args = ap.parse_args()

    import torch
    from cosmo_sr.tts.bootstrap import best_of_k, bootstrap_ci, paired_bootstrap, subset_draws
    from cosmo_sr.tts.features import FEATURE_KEYS, feature_matrix
    from cosmo_sr.tts.scores import (
        STATISTICAL_ORACLE_COMPONENTS, ScoreNormalizer, composite_score, derive_metrics,
    )
    from cosmo_sr.tts.verifier import (
        Standardizer, rank_metrics, score_rows, train_feature_ranker,
    )

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [derive_metrics(r) for r in load_rows(Path(args.rows))]
    if not rows:
        raise SystemExit(f"no rows in {args.rows}")

    # The oracle target is normalised on validation boxes -- the same convention
    # as the Stage-1 audit, so "recovered fraction of the oracle gain" compares
    # like with like.
    val_rows = [r for r in rows if r["box"] in set(args.val_boxes)]
    norm = ScoreNormalizer.fit(val_rows or rows)
    for r in rows:
        r["statistical_oracle_score"] = composite_score(r, STATISTICAL_ORACLE_COMPONENTS, norm)

    keys = [k for k in FEATURE_KEYS if any(k in r for r in rows)]
    missing = [k for k in FEATURE_KEYS if k not in keys]
    if missing:
        print(f"[warn] features absent from rows (rerun eval with --equivariance?): {missing}")
    x_all = feature_matrix(rows, keys)
    target = np.asarray([r["statistical_oracle_score"] for r in rows], dtype=float)

    g_train = group_indices(rows, args.train_boxes)
    g_val = group_indices(rows, args.val_boxes)
    g_test = group_indices(rows, args.test_boxes)
    if not g_train or not g_test:
        raise SystemExit("need at least one train box and one test box")
    print(f"boxes: train {len(g_train)} / val {len(g_val)} / test {len(g_test)}; "
          f"{len(keys)} features, {len(rows)} candidate rows")

    std = Standardizer.fit(x_all[np.concatenate(g_train)], keys)
    x_std = std(x_all)

    # ------------------------------------------------------------------ #
    # 1. single hand-crafted features as selectors
    # ------------------------------------------------------------------ #
    def _rho(m: Dict[str, float]) -> float:
        """Spearman with NaN treated as "no signal", so sorting stays well defined."""
        v = m.get("spearman", np.nan)
        return float(v) if np.isfinite(v) else -np.inf

    # Both the feature and its sign are chosen on **validation** boxes and only
    # then scored on test. Picking the best-on-test feature and reporting that
    # number is selection on the test set: with ~16 features x 2 signs it will
    # reliably manufacture an oracle-looking selector out of noise.
    g_select = g_val or g_train
    single: Dict[str, Dict[str, float]] = {}
    for j, key in enumerate(keys):
        for sign, tag in ((+1.0, ""), (-1.0, "-")):
            m_sel = rank_metrics(sign * x_all[:, j], target, g_select)
            if key not in single or _rho(m_sel) > single[key]["select_spearman"]:
                single[key] = {
                    **rank_metrics(sign * x_all[:, j], target, g_test),
                    "select_spearman": _rho(m_sel), "sign": sign, "name": f"{tag}{key}",
                }
    ranked = sorted(single.items(), key=lambda kv: -max(kv[1]["select_spearman"], -1.0))
    print("\ntop hand-crafted single-feature selectors "
          "(chosen on val boxes, scored on test):")
    for key, m in ranked[:6]:
        print(f"  {m['name']:26s} val-spearman {m['select_spearman']:+.3f}  "
              f"test-spearman {m['spearman']:+.3f}  "
              f"test-pair-acc {m['pairwise_accuracy']:.3f}  regret {m['regret']:.4f}")

    # ------------------------------------------------------------------ #
    # 2. learned pairwise ranker
    # ------------------------------------------------------------------ #
    idx_train = np.concatenate(g_train)
    remap = {int(v): i for i, v in enumerate(idx_train)}
    groups_train = [[remap[int(i)] for i in g] for g in g_train]
    idx_val = np.concatenate(g_val) if g_val else idx_train
    remap_v = {int(v): i for i, v in enumerate(idx_val)}
    groups_val = [[remap_v[int(i)] for i in g] for g in g_val] if g_val else None

    model, history = train_feature_ranker(
        x_std[idx_train], groups_train, target[idx_train],
        x_val=x_std[idx_val] if g_val else None,
        groups_val=groups_val,
        target_val=target[idx_val] if g_val else None,
        hidden=args.hidden, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, min_margin=args.min_margin,
        seed=args.seed, verbose=True,
    )
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x_std, dtype=torch.float32)).cpu().numpy()
    for r, p in zip(rows, pred):
        r["verifier_score"] = float(p)

    verifier_test = rank_metrics(pred, target, g_test)
    print("\nlearned ranker (test boxes): "
          f"spearman {verifier_test['spearman']:+.3f}  "
          f"pair-acc {verifier_test['pairwise_accuracy']:.3f}  "
          f"top1 {verifier_test['top1_rate']:.3f}  regret {verifier_test['regret']:.4f}")

    # ------------------------------------------------------------------ #
    # 3. best-of-K curves and the gate
    # ------------------------------------------------------------------ #
    best_single = ranked[0][0]
    best_sign = single[best_single]["sign"]
    selectors = {
        "random": np.zeros(len(rows)),
        "handcrafted": best_sign * x_all[:, keys.index(best_single)],
        "verifier": pred,
        "oracle": target,
    }
    rng = np.random.default_rng(args.seed)
    curves: Dict[str, Dict[int, Dict]] = {}
    guard: Dict[str, Dict[str, Dict]] = {}
    for k in args.k_values:
        draws = {gi: subset_draws(k, len(g), args.n_repeats, rng) for gi, g in enumerate(g_test)}
        for sel, score in selectors.items():
            per_box = []
            for gi, g in enumerate(g_test):
                g = np.asarray(g)
                vals = target[g]
                if sel == "random":
                    per_box.append(float(np.nanmean(vals[draws[gi]])))
                else:
                    mean, _ = best_of_k(vals, score[g], k, draws=draws[gi])
                    per_box.append(mean)
            curves.setdefault(sel, {})[k] = {**bootstrap_ci(per_box, n_boot=args.n_boot, rng=rng),
                                             "per_box": per_box}
        if k == args.k_gate:
            for metric in GUARD_METRICS:
                vals_all = np.asarray([r.get(metric, np.nan) for r in rows], dtype=float)
                for sel, score in selectors.items():
                    per_box = []
                    for gi, g in enumerate(g_test):
                        g = np.asarray(g)
                        if sel == "random":
                            per_box.append(float(np.nanmean(vals_all[g][draws[gi]])))
                        else:
                            mean, _ = best_of_k(vals_all[g], score[g], k, draws=draws[gi])
                            per_box.append(mean)
                    guard.setdefault(metric, {})[sel] = {
                        **bootstrap_ci(per_box, n_boot=args.n_boot, rng=rng),
                        "per_box": per_box,
                    }

    k = args.k_gate if args.k_gate in curves["verifier"] else max(curves["verifier"])
    v_box = curves["verifier"][k]["per_box"]
    r_box = curves["random"][k]["per_box"]
    o_box = curves["oracle"][k]["per_box"]
    vs_random = paired_bootstrap(v_box, r_box, n_boot=args.n_boot, rng=rng)
    oracle_gain = float(np.mean(r_box) - np.mean(o_box))
    verifier_gain = float(np.mean(r_box) - np.mean(v_box))
    recovered = verifier_gain / oracle_gain if abs(oracle_gain) > 1e-12 else float("nan")

    damage = {}
    for metric, d in guard.items():
        worse = float(np.mean(d["verifier"]["per_box"]) - np.mean(d["random"]["per_box"]))
        cmp = paired_bootstrap(d["verifier"]["per_box"], d["random"]["per_box"],
                               n_boot=args.n_boot, rng=rng)
        damage[metric] = {"delta": worse, "significant_worse": bool(cmp["significant"] and worse > 0),
                          **cmp}

    gate = {
        "k": k,
        "beats_random": bool(vs_random["significant"] and vs_random["mean"] < 0),
        "oracle_gain": oracle_gain,
        "verifier_gain": verifier_gain,
        "recovered_fraction": recovered,
        "recovers_enough": bool(np.isfinite(recovered) and recovered >= args.recover_frac),
        "no_significant_damage": not any(d["significant_worse"] for d in damage.values()),
    }
    gate["pass"] = bool(gate["beats_random"] and gate["recovers_enough"]
                        and gate["no_significant_damage"])

    report = {
        "features": keys, "single_feature": {k2: v for k2, v in single.items()},
        "verifier_test": verifier_test, "history_best_acc": history["best_acc"][0],
        "curves": curves, "guard_metrics": guard, "damage": damage, "gate": gate,
        "best_handcrafted": single[best_single]["name"],
    }
    (out / "verifier_report.json").write_text(json.dumps(report, indent=2, default=float))
    std.save(out / "standardizer.json")
    torch.save({"model": model.state_dict(), "keys": keys, "hidden": args.hidden},
               out / "verifier.pt")

    print(f"\n=== best-of-K on '{args.report_metric}' (test boxes) ===")
    for sel in ("random", "handcrafted", "verifier", "oracle"):
        line = "  ".join(f"K={kk}: {curves[sel][kk]['mean']:+.4f}" for kk in sorted(curves[sel]))
        print(f"  {sel:12s} {line}")
    print(f"\nDECISION GATE at K={k}: {'PASS' if gate['pass'] else 'FAIL'}")
    print(f"  beats random           : {gate['beats_random']} "
          f"(delta {vs_random['mean']:+.4f}, CI [{vs_random['lo']:+.4f}, {vs_random['hi']:+.4f}])")
    print(f"  oracle gain recovered  : {100 * recovered:.1f}% "
          f"(need >= {100 * args.recover_frac:.0f}%)")
    print(f"  no significant damage  : {gate['no_significant_damage']}")
    for metric, d in damage.items():
        print(f"    {metric:32s} delta {d['delta']:+.4g} "
              f"{'WORSE' if d['significant_worse'] else 'ok'}")
    print(f"\nwrote {out}/verifier_report.json, verifier.pt, standardizer.json")


if __name__ == "__main__":
    main()
