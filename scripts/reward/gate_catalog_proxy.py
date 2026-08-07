#!/usr/bin/env python
"""Gate the proxy BEFORE any SR2 parameter is allowed to move.

The criteria are section 7 of the plan, evaluated on the held-out boxes
(set8-11) and on within-``(box, tile)`` ranking:

1. within-``(box, tile)`` Spearman >= 0.5;
2. pairwise ranking accuracy >= 0.65;
3. proxy-selected candidates enriched for positive real ``dR_occ``;
4. correct ordering along the oracle-intervention ``alpha`` sequence;
5. no reliance on a single feature (leave-one-feature-out ablation);
6. reasonable pooled occupation-curve prediction;
7. real tile swaps through ``splice_tiles``, verified against a fresh Rockstar
   run (queued separately; this script consumes their results if present and
   reports the criterion as unmet rather than passed if they are missing).

If the proxy fails, actor training does not start. The response is to improve
the proxy or its data, or to move to the true-catalog oracle-renderer
experiment -- **never** to unfreeze more of the generator, which would only give
a bad gradient more parameters to damage.

    python scripts/reward/gate_catalog_proxy.py --run-name direct_a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, direct_root,
    load_direct_config, load_reward_models, read_jsonl, run_dir, write_json,
)
from train_catalog_proxy import (  # noqa: E402
    _predicted_delta, as_arrays, evaluate, load_rows, true_delta_rewards,
)

from cosmo_sr.reward.catalog_proxy import ProxyEnsemble, spearman  # noqa: E402


def ensemble_prediction(ens: ProxyEnsemble, arrays, rows, reward_t, *,
                        w_joint: float, w_occ: float) -> np.ndarray:
    feats = torch.as_tensor(arrays["features"], dtype=torch.float64)
    vol = torch.as_tensor(arrays["volume"], dtype=torch.float64)
    with torch.no_grad():
        return torch.stack([
            _predicted_delta(m, feats, vol, rows, reward_t,
                             w_joint=w_joint, w_occ=w_occ)
            for m in ens.members]).mean(dim=0).numpy()


def selection_enrichment(pred: np.ndarray, true: np.ndarray, *,
                         top_fraction: float = 0.2) -> Dict[str, float]:
    """Of the candidates the proxy would pick, how many really improved?

    The decision the proxy is used for is an argmax, so this is closer to what
    it costs to be wrong than a correlation is.
    """
    ok = np.isfinite(pred) & np.isfinite(true)
    if ok.sum() < 5:
        return {"selected_positive_fraction": float("nan"),
                "baseline_positive_fraction": float("nan"), "n": int(ok.sum())}
    p, t = pred[ok], true[ok]
    k = max(1, int(round(float(top_fraction) * p.size)))
    top = np.argsort(-p)[:k]
    return {
        "selected_positive_fraction": float(np.mean(t[top] > 0)),
        "baseline_positive_fraction": float(np.mean(t > 0)),
        "n": int(p.size), "n_selected": int(k),
    }


def alpha_ordering(rows, pred: np.ndarray) -> Dict[str, float]:
    """Does the proxy order the oracle-intervention ``alpha`` ladder correctly?

    Grouped by ``(box, tile)`` so this is again a within-tile question. A proxy
    that gets this wrong has not learned the direction of improvement even
    though it may correlate well on average.
    """
    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, r in enumerate(rows):
        if r.get("source") != "intervention" or r.get("alpha") is None:
            continue
        groups.setdefault((r["box"], int(r["tile_id"])), []).append(i)
    rhos = []
    for idx in groups.values():
        if len(idx) < 3:
            continue
        a = np.asarray([float(rows[i]["alpha"]) for i in idx])
        p = pred[np.asarray(idx)]
        if np.all(np.isfinite(p)):
            rhos.append(spearman(a, p))
    return {"alpha_ordering_spearman": float(np.nanmean(rhos)) if rhos else float("nan"),
            "n_alpha_groups": len(rhos)}


def feature_ablation(ens: ProxyEnsemble, arrays, rows, reward_t, true, *,
                     w_joint: float, w_occ: float) -> Dict[str, float]:
    """Leave-one-feature-out: is any single coordinate doing all the work?

    Each feature in turn is replaced by its training mean and the within-tile
    Spearman is recomputed. If knocking one out leaves the proxy essentially as
    good (``retention`` near 1 for every feature) then no single feature is
    load-bearing, which is what we want. If one ablation *destroys* the score,
    the proxy is that feature wearing a vector's name -- and the criterion here
    is the reverse of the obvious one, so it is stated explicitly:
    ``max_retention`` close to 1 is healthy; the failure is a feature whose
    removal leaves retention near 1 for *all others* while its own removal
    collapses the score.
    """
    base = evaluate(ens, arrays, rows, reward_t, true,
                    w_joint=w_joint, w_occ=w_occ)["within_tile_spearman"]
    out: Dict[str, float] = {"full_spearman": float(base)}
    n_feat = arrays["features"].shape[1]
    retentions = []
    for j in range(n_feat):
        a = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in arrays.items()}
        a["features"][:, j] = arrays["features"][:, j].mean()
        s = evaluate(ens, a, rows, reward_t, true,
                     w_joint=w_joint, w_occ=w_occ)["within_tile_spearman"]
        r = float(s / base) if np.isfinite(base) and abs(base) > 1e-9 else float("nan")
        retentions.append(r)
        out[f"retention_without_{j}"] = r
    finite = [r for r in retentions if np.isfinite(r)]
    out["min_retention"] = float(np.min(finite)) if finite else float("nan")
    out["max_retention"] = float(np.max(finite)) if finite else float("nan")
    # "No reliance on one feature" = removing the single most important one
    # still leaves most of the signal.
    out["single_feature_dependence"] = float(1.0 - min(finite)) if finite else float("nan")
    return out


def occupation_curve_error(ens: ProxyEnsemble, arrays, rows, reward_t) -> Dict[str, float]:
    """Pooled occupation prediction: sum_j S_j / sum_j H_j, per host bin.

    Pooled, not per tile, because occupation is a ratio of sums and a per-tile
    ratio is meaningless where a tile holds a fraction of one host.
    """
    feats = torch.as_tensor(arrays["features"], dtype=torch.float64)
    vol = torch.as_tensor(arrays["volume"], dtype=torch.float64)
    with torch.no_grad():
        preds = [m.summary(feats, vol) for m in ens.members]
    pred_S = torch.stack([p.occ_numerator for p in preds]).mean(dim=0).sum(dim=0)
    pred_H = torch.stack([p.n_host for p in preds]).mean(dim=0).sum(dim=0)
    true_S = torch.as_tensor(arrays["occ_numerator"], dtype=torch.float64).sum(dim=0)
    true_H = torch.as_tensor(arrays["n_host"], dtype=torch.float64).sum(dim=0)
    po = (pred_S / pred_H.clamp_min(1e-12)).numpy()
    to = (true_S / true_H.clamp_min(1e-12)).numpy()
    ok = (to > 0) & (po > 0)
    err = np.abs(np.log10(po[ok]) - np.log10(to[ok])) if ok.any() else np.array([np.nan])
    return {
        "predicted_occupation": po.tolist(), "true_occupation": to.tolist(),
        "occupation_curve_log_error_mean": float(np.nanmean(err)),
        "occupation_curve_log_error_max": float(np.nanmax(err)),
    }


def splice_verification(run: Path) -> Dict[str, float]:
    """Consume the real ``splice_tiles`` + Rockstar checks, if they have landed.

    Deliberately not computed here: verifying a predicted swap means splicing a
    tile into a real field and running the halo finder on it, which is a CPU job
    of its own. Missing results make the criterion *unmet*, never *passed*.
    """
    p = run / "splice_verification.jsonl"
    if not p.is_file():
        return {"n_splices": 0, "sign_agreement": float("nan"),
                "note": f"no splice verification at {p}"}
    rows = read_jsonl(p)
    ok = [r for r in rows if np.isfinite(r.get("predicted_dR", np.nan))
          and np.isfinite(r.get("measured_dR", np.nan))]
    if not ok:
        return {"n_splices": 0, "sign_agreement": float("nan")}
    agree = [np.sign(r["predicted_dR"]) == np.sign(r["measured_dR"]) for r in ok]
    return {
        "n_splices": len(ok),
        "sign_agreement": float(np.mean(agree)),
        "predicted": [float(r["predicted_dR"]) for r in ok],
        "measured": [float(r["measured_dR"]) for r in ok],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--table", default="")
    ap.add_argument("--proxy-dir", default="")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    gate = dict(cfg.get("proxy_gate", {}))
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)

    table = Path(args.table) if args.table else direct_root("proxy_data") / "rows.jsonl"
    rows_all = load_rows(table)
    run = run_dir(args.run_name, create=True)
    proxy_dir = Path(args.proxy_dir) if args.proxy_dir else run / "proxy"
    if not proxy_dir.is_dir():
        print(f">>> MISSING INPUT: {proxy_dir}")
        print(">>> produced by: scripts/reward/train_catalog_proxy.py")
        return 0
    ens = ProxyEnsemble.load(proxy_dir).freeze()

    gate_boxes = set(boxes_of(cfg, "proxy_gate"))
    rows = [r for r in rows_all if r["box"] in gate_boxes]
    if not rows:
        print(f">>> GATE FAILED: no held-out rows from {sorted(gate_boxes)}; "
              ">>> the proxy has not been evaluated out of sample at all.")
        write_json(run / "proxy_gate.json",
                   {"passed": False, "reason": "no held-out rows"})
        return 0

    arrays = as_arrays(rows)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    true = true_delta_rewards(rows, reward_t, w_joint=w_joint, w_occ=w_occ)
    pred = ensemble_prediction(ens, arrays, rows, reward_t,
                               w_joint=w_joint, w_occ=w_occ)

    rank = evaluate(ens, arrays, rows, reward_t, true, w_joint=w_joint, w_occ=w_occ)
    sel = selection_enrichment(pred, true)
    alpha = alpha_ordering(rows, pred)
    abl = feature_ablation(ens, arrays, rows, reward_t, true,
                           w_joint=w_joint, w_occ=w_occ)
    occ = occupation_curve_error(ens, arrays, rows, reward_t)
    splice = splice_verification(run)

    checks: List[Dict] = [
        {"name": "within_tile_spearman", "value": rank["within_tile_spearman"],
         "bound": float(gate.get("min_within_tile_spearman", 0.5)), "kind": "min"},
        {"name": "pairwise_accuracy", "value": rank["pairwise_accuracy"],
         "bound": float(gate.get("min_pairwise_accuracy", 0.65)), "kind": "min"},
        {"name": "selected_positive_fraction", "value": sel["selected_positive_fraction"],
         "bound": float(gate.get("min_selected_positive_fraction", 0.6)), "kind": "min"},
        {"name": "alpha_ordering_spearman", "value": alpha["alpha_ordering_spearman"],
         "bound": float(gate.get("min_alpha_ordering_spearman", 0.8)), "kind": "min"},
        {"name": "single_feature_dependence", "value": abl["single_feature_dependence"],
         "bound": float(gate.get("max_single_feature_ablation_retention", 0.9)),
         "kind": "max"},
        {"name": "occupation_curve_log_error_mean",
         "value": occ["occupation_curve_log_error_mean"],
         "bound": float(gate.get("max_occupation_curve_log_error", 0.3)), "kind": "max"},
        {"name": "splice_sign_agreement", "value": splice["sign_agreement"],
         "bound": float(gate.get("min_splice_sign_agreement", 0.75)), "kind": "min"},
        {"name": "n_splice_verifications", "value": float(splice["n_splices"]),
         "bound": float(gate.get("n_splice_verifications", 4)), "kind": "min"},
    ]
    failures = []
    for c in checks:
        v, b = c["value"], c["bound"]
        c["passed"] = bool(
            np.isfinite(v) and ((v >= b) if c["kind"] == "min" else (v <= b)))
        if not c["passed"]:
            failures.append(
                f"{c['name']}={v:.4g} "
                f"{'<' if c['kind'] == 'min' else '>'} {b:.4g}")

    report = {
        "passed": not failures, "failures": failures, "checks": checks,
        "gate_boxes": sorted(gate_boxes), "n_rows": len(rows),
        "ranking": rank, "selection": sel, "alpha": alpha,
        "ablation": abl, "occupation": occ, "splice": splice,
        "proxy_dir": str(proxy_dir), "table": str(table),
    }
    write_json(run / "proxy_gate.json", report)
    banner(json.dumps({"passed": report["passed"], "failures": failures}, indent=2))
    if not report["passed"]:
        print(">>> GATE FAILED. Do NOT start actor training. Improve the proxy or")
        print(">>> its data, or move to the true-catalog oracle renderer.")
        print(">>> Unfreezing more of SR2 is not a response to this.")
    print(f"  report -> {run / 'proxy_gate.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
