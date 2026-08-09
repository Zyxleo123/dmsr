#!/usr/bin/env python
"""Score one proxy arm on set8-11, against criteria frozen before it was looked at.

The held-out boxes are spent once. Everything this script measures is defined in
``proxy_gate:`` and ``proxy_report_slices:`` of the config, which are committed
before any of it runs -- a threshold chosen after seeing the number it judges is
not a threshold.

The criteria, and what each is for
----------------------------------
1. within-``(box, tile)`` Spearman -- can it order versions of the same tile;
2. pairwise ranking accuracy -- the same question as a decision rate;
3. selected-positive fraction -- of the candidates it would *pick*, how many
   really improved. Closer to what being wrong costs than a correlation is,
   because the proxy is used as an argmax;
4. intervention-strength ordering -- does the ``alpha`` ladder come out in
   order, within a tile. A proxy can correlate well on average and still have
   the direction of improvement backwards;
5. single-feature ablation -- is one coordinate doing all the work;
6. occupancy log error in **every reliable host bin**, not on average. A mean
   hides one bin wrong by a factor of two behind four that are right, and the
   bins are not interchangeable: 2 and 3 are where the decision is;
7. pooled full-box ``(N, H, S, O)`` better than zero-change, training-mean and
   linear regression. A proxy that cannot beat "predict no change" has learned
   nothing an actor can use;
8. ensemble uncertainty rising with prediction error. The actor maximises
   ``mean - beta*std``; if ``std`` is uncorrelated with error then ``beta`` is
   penalising noise and the ensemble is decoration;
9. real ``splice_tiles`` swaps re-run through Rockstar (produced separately by
   ``splice_verify.py``; missing results make the criterion **unmet**, never
   passed).

If an arm fails, the response is to improve the proxy or its data, or to move to
the true-catalog oracle-renderer experiment -- **never** to unfreeze more of the
generator, which would only give a bad gradient more parameters to damage.

    python scripts/reward/gate_catalog_proxy.py --run-name direct_a --arm a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from _proxy_baselines import load_baselines  # noqa: E402
from _proxy_data import (  # noqa: E402
    ARMS, COUNT_KEYS, as_arrays, build_row_context, ensemble_delta, load_rows,
    pooled_count_error, rank_metrics, slice_rows, true_delta_rewards,
)
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, direct_root,
    load_direct_config, load_reward_models, read_jsonl, run_dir, write_json,
)

from cosmo_sr.reward.catalog_proxy import ProxyEnsemble, spearman  # noqa: E402


def selection_enrichment(pred: np.ndarray, true: np.ndarray, *,
                         top_fraction: float = 0.2) -> Dict[str, float]:
    """Of the candidates the proxy would pick, how many really improved?"""
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
        "n": int(p.size), "n_selected": int(k), "top_fraction": float(top_fraction),
    }


def alpha_ordering(arrays, pred: np.ndarray,
                   modes: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Does the proxy order the intervention ``alpha`` ladder within a tile?

    Restricted to one channel mode when asked, because that is the diagnostic
    the arm comparison turns on: a density-only proxy has no way to order a
    ``vel`` ladder, and reporting the three modes pooled would let the ``both``
    rows carry it.
    """
    sel = arrays["source"] == "intervention"
    if modes is not None:
        sel &= np.isin(arrays["mode"], [str(m) for m in modes])
    idx_all = np.nonzero(sel & np.isfinite(arrays["alpha"]))[0]
    groups: Dict = {}
    for i in idx_all:
        groups.setdefault((str(arrays["box"][i]), int(arrays["tile_id"][i])), []).append(i)
    rhos = []
    for idx in groups.values():
        if len(idx) < 3:
            continue
        a = arrays["alpha"][np.asarray(idx)]
        p = pred[np.asarray(idx)]
        if np.all(np.isfinite(p)):
            rhos.append(spearman(a, p))
    return {"alpha_ordering_spearman": float(np.nanmean(rhos)) if rhos else float("nan"),
            "n_alpha_groups": len(rhos),
            "modes": (list(modes) if modes else "all")}


def uncertainty_calibration(mean: np.ndarray, std: np.ndarray,
                            true: np.ndarray) -> Dict[str, float]:
    """Does the ensemble disagree where it is wrong?

    Spearman between the spread and the absolute error, plus the ratio of mean
    error in the top and bottom uncertainty terciles -- the same claim stated
    twice, because a rank correlation can be positive while the actual errors in
    the "uncertain" bucket are no larger, which is what would make the actor's
    penalty useless in practice.
    """
    ok = np.isfinite(mean) & np.isfinite(std) & np.isfinite(true)
    if ok.sum() < 10:
        return {"uncertainty_error_spearman": float("nan"),
                "error_ratio_top_bottom_tercile": float("nan"), "n": int(ok.sum())}
    err = np.abs(mean[ok] - true[ok])
    s = std[ok]
    order = np.argsort(s)
    k = max(1, len(order) // 3)
    lo = float(np.mean(err[order[:k]]))
    hi = float(np.mean(err[order[-k:]]))
    return {
        "uncertainty_error_spearman": spearman(s, err),
        "error_ratio_top_bottom_tercile": (hi / lo if lo > 0 else float("nan")),
        "mean_error_low_uncertainty": lo, "mean_error_high_uncertainty": hi,
        "n": int(ok.sum()),
    }


def feature_ablation(members, features: np.ndarray, ctx, reward_t, arrays,
                     true: np.ndarray, *, w_joint: float, w_occ: float,
                     names: Sequence[str]) -> Dict[str, object]:
    """Leave-one-feature-out: is any single coordinate doing all the work?

    Each feature in turn is replaced by its own mean and the within-tile
    Spearman recomputed. ``single_feature_dependence = 1 - min retention`` is
    how much the most important single coordinate is worth: near 0 means the
    vector is genuinely a vector, near 1 means the "proxy" is one number wearing
    a vector's name.
    """
    base = rank_metrics(
        ensemble_delta(members, torch.as_tensor(features, dtype=torch.float64),
                       ctx, reward_t, w_joint=w_joint, w_occ=w_occ)[0],
        true, arrays["box"], arrays["tile_id"])["within_tile_spearman"]
    retentions, per_feature = [], {}
    for j in range(features.shape[1]):
        x = features.copy()
        x[:, j] = features[:, j].mean()
        s = rank_metrics(
            ensemble_delta(members, torch.as_tensor(x, dtype=torch.float64), ctx,
                           reward_t, w_joint=w_joint, w_occ=w_occ)[0],
            true, arrays["box"], arrays["tile_id"])["within_tile_spearman"]
        r = float(s / base) if np.isfinite(base) and abs(base) > 1e-9 else float("nan")
        retentions.append(r)
        per_feature[str(names[j]) if j < len(names) else f"f{j}"] = r
    finite = [r for r in retentions if np.isfinite(r)]
    worst = int(np.nanargmin(retentions)) if finite else -1
    return {
        "full_spearman": float(base),
        "retention_by_feature": per_feature,
        "min_retention": float(np.min(finite)) if finite else float("nan"),
        "max_retention": float(np.max(finite)) if finite else float("nan"),
        "most_important_feature": (str(names[worst]) if 0 <= worst < len(names)
                                   else ""),
        "single_feature_dependence": (float(1.0 - min(finite)) if finite
                                      else float("nan")),
    }


def splice_verification(path: Path, gate: Dict) -> Dict[str, object]:
    """Consume the real ``splice_tiles`` + Rockstar checks, if they have landed.

    Deliberately not computed here: verifying a predicted swap means splicing a
    tile into a real field and running the halo finder on it, which is a CPU job
    of its own. Missing results make the criterion *unmet*, never *passed*.
    """
    if not path.is_file():
        return {"n_splices": 0, "n_boxes": 0, "sign_agreement": float("nan"),
                "note": f"no splice verification at {path}"}
    rows = read_jsonl(path)
    ok = [r for r in rows
          if np.isfinite(r.get("predicted_dR", np.nan))
          and np.isfinite(r.get("measured_dR", np.nan))]
    if not ok:
        return {"n_splices": 0, "n_boxes": 0, "sign_agreement": float("nan")}
    agree = [np.sign(r["predicted_dR"]) == np.sign(r["measured_dR"]) for r in ok]
    strata = {}
    for r, a in zip(ok, agree):
        s = str(r.get("stratum", "unknown"))
        strata.setdefault(s, []).append(bool(a))
    return {
        "n_splices": len(ok),
        "n_boxes": len({str(r.get("box", "")) for r in ok}),
        "sign_agreement": float(np.mean(agree)),
        "agreement_by_stratum": {k: float(np.mean(v)) for k, v in strata.items()},
        "n_by_stratum": {k: len(v) for k, v in strata.items()},
        "strata_present": sorted(strata),
        "missing_strata": sorted(set(str(s) for s in gate.get("splice_strata", []))
                                 - set(strata)),
        "rows": ok,
    }


def _checks(gate: Dict, rank, sel, alpha, abl, occ_bins, base_cmp, unc,
            splice) -> List[Dict]:
    beat = list(gate.get("must_beat_baselines", []))
    worst_margin = (min((base_cmp[b]["margin"] for b in beat if b in base_cmp),
                        default=float("nan")))
    return [
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
        {"name": "occupation_log_error_worst_reliable_bin",
         "value": occ_bins["worst_reliable"],
         "bound": float(gate.get("max_occupation_curve_log_error", 0.3)), "kind": "max"},
        {"name": "pooled_margin_over_worst_baseline", "value": worst_margin,
         "bound": 0.0, "kind": "min"},
        {"name": "uncertainty_error_spearman", "value": unc["uncertainty_error_spearman"],
         "bound": float(gate.get("min_uncertainty_error_spearman", 0.2)), "kind": "min"},
        {"name": "splice_sign_agreement", "value": splice["sign_agreement"],
         "bound": float(gate.get("min_splice_sign_agreement", 0.75)), "kind": "min"},
        {"name": "n_splice_verifications", "value": float(splice["n_splices"]),
         "bound": float(gate.get("n_splice_verifications", 12)), "kind": "min"},
        {"name": "n_splice_boxes", "value": float(splice["n_boxes"]),
         "bound": float(gate.get("min_splice_boxes", 3)), "kind": "min"},
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--table", default="")
    ap.add_argument("--arm", default="a", choices=list(ARMS))
    ap.add_argument("--proxy-dir", default="")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    gate = dict(cfg.get("proxy_gate", {}))
    slices = dict(cfg.get("proxy_report_slices", {}))
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    arm = args.arm
    occ_cfg = cfg["_reward"].get("reward", {}).get("occupation", {})
    reliable = [int(b) for b in occ_cfg.get("reliable_host_bins", [0, 1, 2, 3])]
    sparse = [int(b) for b in occ_cfg.get("sparse_host_bins", [4])]
    report_only = [int(b) for b in gate.get("report_only_host_bins", sparse)]

    table = Path(args.table) if args.table else direct_root("proxy_data") / "rows.jsonl"
    rows_all = load_rows(table, require_complete=not args.allow_incomplete)
    run = run_dir(args.run_name, create=True)
    proxy_dir = Path(args.proxy_dir) if args.proxy_dir else run / f"proxy_{arm}"
    verdict_path = run / f"proxy_gate_{arm}.json"
    if not proxy_dir.is_dir():
        print(f">>> MISSING INPUT: {proxy_dir}")
        print(">>> produced by: scripts/reward/train_catalog_proxy.py")
        return 0
    ens = ProxyEnsemble.load(proxy_dir).freeze()

    gate_boxes = set(boxes_of(cfg, "proxy_gate"))
    rows = [r for r in rows_all if r["box"] in gate_boxes]
    if not rows:
        print(f">>> GATE FAILED: no held-out rows from {sorted(gate_boxes)}; "
              "the proxy has not been evaluated out of sample at all.")
        write_json(verdict_path, {"arm": arm, "passed": False,
                                  "reason": "no held-out rows"})
        return 0

    arrays = as_arrays(rows, arm)
    ctx = build_row_context(rows)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    true = true_delta_rewards(ctx, reward_t, w_joint=w_joint, w_occ=w_occ)
    feats = torch.as_tensor(arrays["features"], dtype=torch.float64)
    mean, std = ensemble_delta(ens.members, feats, ctx, reward_t,
                               w_joint=w_joint, w_occ=w_occ)

    rank = rank_metrics(mean, true, arrays["box"], arrays["tile_id"])
    sel = selection_enrichment(mean, true)
    alpha = alpha_ordering(arrays, mean)
    unc = uncertainty_calibration(mean, std, true)

    names = []
    npz_names = proxy_dir / "feature_names.json"
    if npz_names.is_file():
        names = list(json.loads(npz_names.read_text()))
    if len(names) != arrays["features"].shape[1]:
        from cosmo_sr.reward.phase_space import arm_paired_feature_names
        from _sr2_direct import phase_space_config_of, soft_config_of
        names = arm_paired_feature_names(arm, soft_config_of(cfg),
                                         phase_space_config_of(cfg))
    abl = feature_ablation(ens.members, arrays["features"], ctx, reward_t, arrays,
                           true, w_joint=w_joint, w_occ=w_occ, names=names)

    with torch.no_grad():
        pred_counts = [m.summary(feats, ctx.tile_volume) for m in ens.members]
    pooled = pooled_count_error(
        {k: torch.stack([getattr(p, k) for p in pred_counts]).mean(0).numpy()
         for k in COUNT_KEYS},
        {k: arrays[k] for k in COUNT_KEYS})
    occ_err = np.asarray(pooled["occupation_log_error"], dtype=np.float64)
    occ_bins = {
        "per_bin": occ_err.tolist(),
        "reliable_host_bins": reliable,
        "worst_reliable": float(np.nanmax(occ_err[reliable])) if reliable else float("nan"),
        # Reported, never gated: ~1 host per box in the 1e14 bin makes a
        # "failure" there a Poisson draw rather than a property of the proxy.
        "report_only_bins": report_only,
        "report_only_values": [float(occ_err[b]) for b in report_only
                               if b < occ_err.size],
    }

    base_cmp: Dict[str, Dict] = {}
    bl_path = run / "proxy_baselines.json"
    if bl_path.is_file():
        for name, b in load_baselines(bl_path).get(arm, {}).items():
            with torch.no_grad():
                s = b.summary(feats, ctx.tile_volume, ctx.frozen)
            bp = pooled_count_error({k: getattr(s, k).numpy() for k in COUNT_KEYS},
                                    {k: arrays[k] for k in COUNT_KEYS})
            base_cmp[name] = {
                "mean_log_error": bp["mean_log_error"],
                "occupation_log_error_max": bp["occupation_log_error_max"],
                # Positive means the proxy is closer to the truth than this
                # baseline, in dex, pooled over N, H, S and O.
                "margin": float(bp["mean_log_error"] - pooled["mean_log_error"]),
            }
    else:
        print(f">>> no {bl_path}; the baseline comparison cannot be made and its "
              ">>> criterion will be reported as unmet.")

    splice = splice_verification(run / f"splice_verification_{arm}.jsonl", gate)

    # --- predeclared slices -----------------------------------------------
    sliced: Dict[str, Dict] = {}
    for name, spec in slices.items():
        idx = slice_rows(arrays, spec or {})
        if idx.size == 0:
            sliced[name] = {"n_rows": 0, "note": "no rows in this slice"}
            continue
        sub = ctx.index(idx)
        m, s2 = ensemble_delta(ens.members, feats[idx], sub, reward_t,
                               w_joint=w_joint, w_occ=w_occ)
        sliced[name] = {
            "n_rows": int(idx.size),
            "ranking": rank_metrics(m, true[idx], arrays["box"][idx],
                                    arrays["tile_id"][idx]),
            "selection": selection_enrichment(m, true[idx]),
            "uncertainty": uncertainty_calibration(m, s2, true[idx]),
        }
    for mode in ("both", "disp", "vel"):
        sliced[f"alpha_ordering_{mode}"] = alpha_ordering(arrays, mean, [mode])

    checks = _checks(gate, rank, sel, alpha, abl, occ_bins, base_cmp, unc, splice)
    failures = []
    for c in checks:
        v, b = c["value"], c["bound"]
        c["passed"] = bool(
            np.isfinite(v) and ((v >= b) if c["kind"] == "min" else (v <= b)))
        if not c["passed"]:
            failures.append(f"{c['name']}={v:.4g} "
                            f"{'<' if c['kind'] == 'min' else '>'} {b:.4g}")
    missing_baselines = [b for b in gate.get("must_beat_baselines", [])
                         if b not in base_cmp]
    if missing_baselines:
        failures.append(f"baselines not evaluated: {missing_baselines}")

    report = {
        "arm": arm, "passed": not failures, "failures": failures, "checks": checks,
        "gate_boxes": sorted(gate_boxes), "n_rows": len(rows),
        "n_features": int(arrays["features"].shape[1]),
        "ranking": rank, "selection": sel, "alpha": alpha, "ablation": abl,
        "pooled": pooled, "occupation_by_bin": occ_bins,
        "baseline_comparison": base_cmp, "uncertainty": unc,
        "splice": splice, "slices": sliced,
        "proxy_dir": str(proxy_dir), "table": str(table),
        "criteria": gate,
    }
    write_json(verdict_path, report)
    banner(json.dumps({"arm": arm, "passed": report["passed"],
                       "failures": failures}, indent=2))
    if not report["passed"]:
        print(f">>> ARM {arm.upper()} FAILED. Do NOT start actor training on it.")
        print(">>> Improve the proxy or its data, or move to the true-catalog")
        print(">>> oracle renderer. Unfreezing more of SR2 is not a response to this.")
    print(f"  report -> {verdict_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
