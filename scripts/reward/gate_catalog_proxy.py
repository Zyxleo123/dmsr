#!/usr/bin/env python
"""Score one proxy arm on set8-11, against criteria frozen before it was looked at.

The held-out boxes are spent once. Everything this script measures is defined in
``proxy_gate:`` and ``proxy_report_slices:`` of the config, which are committed
before any of it runs -- a threshold chosen after seeing the number it judges is
not a threshold.

The criteria, and what each is for
----------------------------------
Offline (this is ``offline_passed``, and what makes an arm eligible for the
actor-like probe):

1. within-``(box, unit)`` Spearman -- can it order versions of the same
   supervision unit (the tile itself at the deployed ``region_width: 1``);
2. pairwise ranking accuracy -- the same question as a decision rate;
3. selected-positive fraction -- of the candidates it would *pick*, how many
   really improved. Closer to what being wrong costs than a correlation is,
   because the proxy is used as an argmax;
4. single-feature ablation -- is one coordinate doing all the work. For the
   token arm the "feature" is a token channel, ablated across all tokens;
5. occupancy log error in **every reliable host bin**, not on average. A mean
   hides one bin wrong by a factor of two behind four that are right, and the
   bins are not interchangeable: 2 and 3 are where the decision is;
6. pooled full-box ``(N, H, S, O)`` better than zero-change, training-mean and
   linear regression. A proxy that cannot beat "predict no change" has learned
   nothing an actor can use;
7. ensemble uncertainty rising with prediction error. The actor maximises
   ``mean - beta*std``; if ``std`` is uncorrelated with error then ``beta`` is
   penalising noise and the ensemble is decoration.

The intervention alpha-ladder ordering is REPORTED, never gated: its
within-unit order is monotone by construction, so it measures the easy part of
the problem, and a winner must not be chosen from it alone.

Actor-like (this is the ``actor`` section, produced separately by
``actor_rockstar_verify.py`` from a probe_only fine-tune's own generated
tiles): 12 real tile replacements spliced into held-out frozen boxes and re-run
through Rockstar, across at least three boxes, with >= 75% sign agreement.
Missing results make the criterion **unmet**, never passed, and ``passed`` --
the flag the benchmark's advancement decision reads -- is offline AND actor.

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
    ARMS, COUNT_KEYS, as_arrays, build_row_context, channel_mean_transform,
    load_rows, make_arm_features, pooled_count_error,
    pooled_count_error_by_candidate, rank_metrics, robust_scale, slice_rows,
    stream_ensemble_delta, stream_pred_counts, true_delta_rewards, unit_ids_of,
)
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, direct_root,
    load_direct_config, load_reward_models, read_jsonl, region_width_of,
    run_dir, write_json,
)

from cosmo_sr.reward.arms import arm_storage  # noqa: E402
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


def alpha_ordering(arrays, pred: np.ndarray, unit: np.ndarray,
                   modes: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """Does the proxy order the intervention ``alpha`` ladder within a unit?

    REPORTED, never gated (``proxy_gate.report_only_alpha_ordering``): the
    ladder's within-unit order is monotone by construction, so it measures the
    easy part of the problem. Restricted to one channel mode when asked,
    because that is the diagnostic the arm comparison turns on: a density-only
    proxy has no way to order a ``vel`` ladder, and reporting the three modes
    pooled would let the ``both`` rows carry it.
    """
    sel = arrays["source"] == "intervention"
    if modes is not None:
        sel &= np.isin(arrays["mode"], [str(m) for m in modes])
    idx_all = np.nonzero(sel & np.isfinite(arrays["alpha"]))[0]
    groups: Dict = {}
    for i in idx_all:
        groups.setdefault((str(arrays["box"][i]), int(unit[i])), []).append(i)
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


def feature_ablation(provider, members, ctx, reward_t, arrays, true: np.ndarray,
                     unit: np.ndarray, *, w_joint: float, w_occ: float,
                     names: Sequence[str], storage: str, n_channels: int,
                     device, chunk_rows: int, abl_rows: Optional[np.ndarray] = None,
                     key: str = "dR_combined") -> Dict[str, object]:
    """Leave-one-feature-out: is any single coordinate doing all the work?

    Each input channel in turn is replaced by its own mean and the within-unit
    Spearman recomputed. ``single_feature_dependence = 1 - min retention`` is how
    much the most important single coordinate is worth: near 0 means the vector is
    genuinely a vector, near 1 means the "proxy" is one number wearing a vector's
    name. The unit of "one feature" per storage is defined in
    :func:`_proxy_data.channel_mean_transform`.

    Streamed, so it works for the dense arm E (10 channels) and the field arm F
    (20 channels) without materialising their input. Evaluated on ``abl_rows`` (a
    row subsample for the spatial arms) to bound cost: the dependence is a
    robustness check, not a precise metric, and a few thousand rows estimate it.
    """
    box = arrays["box"]
    idxs = (np.arange(len(provider)) if abl_rows is None
            else np.asarray(abl_rows, dtype=np.int64))

    def _within(transform=None) -> float:
        mean, _ = stream_ensemble_delta(
            provider, members, ctx, reward_t, w_joint=w_joint, w_occ=w_occ,
            key=key, chunk_rows=chunk_rows, device=device, rows=idxs,
            transform=transform)
        return rank_metrics(mean[idxs], true[idxs], box[idxs],
                            unit[idxs])["within_tile_spearman"]

    base = _within()
    retentions, per_feature = [], {}
    for j in range(int(n_channels)):
        tf = channel_mean_transform(provider, j, storage, rows=idxs, device=device)
        s = _within(tf)
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
        "n_ablation_rows": int(idxs.size),
    }


def actor_verification(path: Path, agate: Dict) -> Dict[str, object]:
    """Consume the actor-like Rockstar checks, if they have landed.

    Produced separately by ``actor_rockstar_verify.py``: tiles GENERATED by a
    probe_only fine-tuned checkpoint, spliced into held-out frozen boxes, and
    re-run through the real halo finder. Deliberately not computed here -- it
    needs a GPU probe and 12 Rockstar runs of its own. Missing results make
    the criterion *unmet*, never *passed*.
    """
    if not path.is_file():
        return {"n_checks": 0, "n_boxes": 0, "sign_agreement": float("nan"),
                "note": f"no actor verification at {path}"}
    rows = read_jsonl(path)
    ok = [r for r in rows
          if np.isfinite(r.get("predicted_dR", np.nan))
          and np.isfinite(r.get("measured_dR", np.nan))]
    if not ok:
        return {"n_checks": 0, "n_boxes": 0, "sign_agreement": float("nan")}
    agree = [np.sign(r["predicted_dR"]) == np.sign(r["measured_dR"]) for r in ok]
    strata = {}
    for r, a in zip(ok, agree):
        s = str(r.get("stratum", "unknown"))
        strata.setdefault(s, []).append(bool(a))
    return {
        "n_checks": len(ok),
        "n_boxes": len({str(r.get("box", "")) for r in ok}),
        "sign_agreement": float(np.mean(agree)),
        "agreement_by_stratum": {k: float(np.mean(v)) for k, v in strata.items()},
        "n_by_stratum": {k: len(v) for k, v in strata.items()},
        "strata_present": sorted(strata),
        "missing_strata": sorted(set(str(s) for s in agate.get("actor_strata", []))
                                 - set(strata)),
        "rows": ok,
    }


def _offline_checks(gate: Dict, rank, sel, abl, occ_bins, base_cmp,
                    unc) -> List[Dict]:
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
    ]


def _actor_checks(agate: Dict, actor: Dict) -> List[Dict]:
    return [
        {"name": "actor_sign_agreement", "value": actor["sign_agreement"],
         "bound": float(agate.get("min_actor_sign_agreement", 0.75)), "kind": "min"},
        {"name": "n_actor_checks", "value": float(actor["n_checks"]),
         "bound": float(agate.get("n_actor_checks", 12)), "kind": "min"},
        {"name": "n_actor_boxes", "value": float(actor["n_boxes"]),
         "bound": float(agate.get("min_actor_boxes", 3)), "kind": "min"},
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--table", default="")
    ap.add_argument("--arm", default="a", choices=list(ARMS))
    ap.add_argument("--proxy-dir", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ablation-max-rows", type=int, default=3000,
                    help="row subsample for the spatial arms' feature ablation")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    gate = dict(cfg.get("proxy_gate", {}))
    agate = dict(cfg.get("actor_gate", {}))
    slices = dict(cfg.get("proxy_report_slices", {}))
    width = region_width_of(cfg)
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

    # Metadata is arm-independent; the per-arm features come from the streaming
    # provider (arm E is ~80 GB float64 on the gate boxes, arm F has no cache).
    arrays = as_arrays(rows, "a", table_dir=table.parent)
    storage = arm_storage(arm)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    # Gate on the SAME reward the proxy was trained to predict, or every rank and
    # selection number below scores it against a target it never learned.
    reward_key = str(cfg.get("proxy", {}).get("reward_target", "dR_combined"))
    device = torch.device(args.device if (args.device != "cuda"
                                          or torch.cuda.is_available()) else "cpu")
    unit = unit_ids_of(arrays["tile_id"], width=width)

    # The true dR is a fixed numpy quantity -- compute it on CPU before the reward
    # model is moved to the device the streaming eval runs on.
    ctx_cpu = build_row_context(rows)
    true = true_delta_rewards(ctx_cpu, reward_t, w_joint=w_joint, w_occ=w_occ,
                              key=reward_key)
    reward_t = reward_t.to(device)
    ctx = ctx_cpu.to(device)
    members = [m.to(device) for m in ens.members]

    provider = make_arm_features(arm, rows, table_dir=table.parent, cfg=cfg)
    shape = provider.per_row_shape
    if storage == "inline" or storage == "field":
        n_channels = int(shape[0])
    elif len(shape) == 3:                       # tokens (2, T, F)
        n_channels = 2 * int(shape[-1])
    else:                                       # grid (2, C, D, D, D)
        n_channels = 2 * int(shape[1])
    chunk = {"a": len(rows), "b": len(rows), "c": 4096, "d": 4096,
             "e": 512, "f": 48}[arm]

    mean, std = stream_ensemble_delta(provider, members, ctx, reward_t,
                                      w_joint=w_joint, w_occ=w_occ, key=reward_key,
                                      chunk_rows=chunk, device=device)

    # Seed-noise floor of the per-tile true reward: the interventions carry ~no
    # signal under a numerator-count target (they edit ~2% of the box around 24
    # hosts), so the spread of their true dR is a data-driven tie threshold.
    interv = arrays["source"] == "intervention"
    noise_floor = (float(robust_scale(true[interv & np.isfinite(true)]))
                   if interv.any() else 0.0)
    rank_margin = gate.get("rank_min_margin", "auto")
    rank_margin = noise_floor if rank_margin == "auto" else float(rank_margin)

    rank = rank_metrics(mean, true, arrays["box"], unit, min_margin=rank_margin)
    # A margin sweep, always reported: it shows whether the within-tile ranking
    # is real once tied (noise-separated) candidates stop being graded, without
    # silently moving the gate -- the bound applies at rank_margin only.
    ranking_margin_sweep = {
        f"{mm:.4g}": rank_metrics(mean, true, arrays["box"], unit, min_margin=mm)
        for mm in sorted({0.0, noise_floor, 2 * noise_floor, 4 * noise_floor})
    }
    # Save the arrays so any future re-score (other margins, other cuts) is a
    # cheap CPU read rather than another GPU pass over the features.
    np.savez(run / f"ranking_arrays_{arm}.npz", mean=mean, std=std, true=true,
             box=arrays["box"], tile_id=arrays["tile_id"], unit=unit,
             source=arrays["source"], alpha=arrays["alpha"])
    sel = selection_enrichment(mean, true)
    alpha = alpha_ordering(arrays, mean, unit)
    unc = uncertainty_calibration(mean, std, true)

    names = []
    npz_names = proxy_dir / "feature_names.json"
    if npz_names.is_file():
        names = list(json.loads(npz_names.read_text()))
    abl_rows = None
    if storage != "inline" and len(rows) > int(args.ablation_max_rows):
        abl_rows = np.random.default_rng(0).choice(
            len(rows), size=int(args.ablation_max_rows), replace=False)
    abl = feature_ablation(provider, members, ctx, reward_t, arrays, true, unit,
                           w_joint=w_joint, w_occ=w_occ, key=reward_key, names=names,
                           storage=storage, n_channels=n_channels, device=device,
                           chunk_rows=chunk, abl_rows=abl_rows)

    pred_counts = stream_pred_counts(provider, members, ctx, chunk_rows=chunk,
                                     device=device)
    true_counts = {k: arrays[k] for k in COUNT_KEYS}
    # Save the per-candidate count vectors alongside the dR arrays, in identical
    # row order, so diagnose_rank_ceiling.py can ask whether a flat within-tile
    # PREDICTED dR originates in flat predicted COUNTS (proxy/features) or in the
    # count->dR reduction -- a cheap CPU read, no second GPU pass over features.
    np.savez(run / f"ranking_counts_{arm}.npz",
             **{f"pred_{k}": np.asarray(pred_counts[k], dtype=np.float64)
                for k in COUNT_KEYS},
             **{f"true_{k}": np.asarray(true_counts[k], dtype=np.float64)
                for k in COUNT_KEYS})
    # The gated metric is PER CANDIDATE (sum a candidate's 512 tiles on their own,
    # then average by box) so one candidate's error cannot cancel another's; the
    # all-row aggregate is kept only under an explicitly diagnostic name.
    pooled = pooled_count_error_by_candidate(pred_counts, true_counts,
                                             arrays["box"], arrays["tag"])
    pooled_all_rows = pooled_count_error(pred_counts, true_counts)
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
    # The linear baseline is a flat design matrix; for a spatial arm it was fitted
    # on arm B's flat features (see train_catalog_proxy.py) and is evaluated on
    # the same, per candidate, against the same per-candidate pooled metric.
    bl_feats = torch.as_tensor(
        as_arrays(rows, arm if storage == "inline" else "b",
                  table_dir=table.parent)["features"], dtype=torch.float64)
    if bl_path.is_file():
        for name, b in load_baselines(bl_path).get(arm, {}).items():
            with torch.no_grad():
                s = b.summary(bl_feats, ctx_cpu.tile_volume, ctx_cpu.frozen)
            bp = pooled_count_error_by_candidate(
                {k: getattr(s, k).numpy() for k in COUNT_KEYS}, true_counts,
                arrays["box"], arrays["tag"])
            base_cmp[name] = {
                "mean_log_error": bp["mean_log_error"],
                "occupation_log_error_max": bp["occupation_log_error_max"],
                # Positive means the proxy is closer to the truth than this
                # baseline, in dex, per candidate pooled over N, H, S and O.
                "margin": float(bp["mean_log_error"] - pooled["mean_log_error"]),
            }
    else:
        print(f">>> no {bl_path}; the baseline comparison cannot be made and its "
              ">>> criterion will be reported as unmet.")

    actor = actor_verification(run / f"actor_verification_{arm}.jsonl", agate)

    # --- predeclared slices -- index the already-streamed mean/std ---------
    sliced: Dict[str, Dict] = {}
    for name, spec in slices.items():
        idx = slice_rows(arrays, spec or {})
        if idx.size == 0:
            sliced[name] = {"n_rows": 0, "note": "no rows in this slice"}
            continue
        sliced[name] = {
            "n_rows": int(idx.size),
            "ranking": rank_metrics(mean[idx], true[idx], arrays["box"][idx], unit[idx]),
            "selection": selection_enrichment(mean[idx], true[idx]),
            "uncertainty": uncertainty_calibration(mean[idx], std[idx], true[idx]),
        }
    for mode in ("both", "disp", "vel"):
        sliced[f"alpha_ordering_{mode}"] = alpha_ordering(arrays, mean, unit,
                                                          [mode])

    def _judge(check_list: List[Dict]) -> List[str]:
        fails = []
        for c in check_list:
            v, b = c["value"], c["bound"]
            c["passed"] = bool(
                np.isfinite(v) and ((v >= b) if c["kind"] == "min" else (v <= b)))
            if not c["passed"]:
                fails.append(f"{c['name']}={v:.4g} "
                             f"{'<' if c['kind'] == 'min' else '>'} {b:.4g}")
        return fails

    offline_checks = _offline_checks(gate, rank, sel, abl, occ_bins, base_cmp, unc)
    failures = _judge(offline_checks)
    missing_baselines = [b for b in gate.get("must_beat_baselines", [])
                         if b not in base_cmp]
    if missing_baselines:
        failures.append(f"baselines not evaluated: {missing_baselines}")
    offline_passed = not failures

    actor_checks = _actor_checks(agate, actor)
    actor_failures = _judge(actor_checks)
    actor_passed = not actor_failures

    report = {
        "arm": arm,
        # Full authorization: the offline screen AND the actor-like Rockstar
        # verification. This is what the benchmark's advancement decision reads.
        "passed": offline_passed and actor_passed,
        # The offline screen alone: what makes the arm eligible for the probe.
        "offline_passed": offline_passed,
        "actor_passed": actor_passed,
        "failures": failures, "actor_failures": actor_failures,
        "checks": offline_checks, "actor_checks": actor_checks,
        "gate_boxes": sorted(gate_boxes), "n_rows": len(rows),
        "n_features": n_channels,
        "region_width": width,
        "reward_target": reward_key,
        "rank_min_margin": rank_margin,
        "rank_noise_floor": noise_floor,
        "ranking_margin_sweep": ranking_margin_sweep,
        "ranking": rank, "selection": sel, "alpha": alpha, "ablation": abl,
        "pooled": pooled,
        "pooled_all_rows_DIAGNOSTIC": {
            "mean_log_error": pooled_all_rows["mean_log_error"],
            "occupation_log_error_max": pooled_all_rows["occupation_log_error_max"],
            "note": "cross-candidate cancellation; not gated. Use 'pooled'."},
        "occupation_by_bin": occ_bins,
        "baseline_comparison": base_cmp, "uncertainty": unc,
        "actor": actor, "slices": sliced,
        "proxy_dir": str(proxy_dir), "table": str(table),
        "criteria": gate, "actor_criteria": agate,
    }
    write_json(verdict_path, report)
    banner(json.dumps({"arm": arm, "passed": report["passed"],
                       "offline_passed": offline_passed,
                       "actor_passed": actor_passed,
                       "failures": failures,
                       "actor_failures": actor_failures}, indent=2))
    if not offline_passed:
        print(f">>> ARM {arm.upper()} FAILED the offline gate. Do NOT probe it, "
              "do NOT start actor training on it.")
        print(">>> Improve the proxy or its data, or move to the true-catalog")
        print(">>> oracle renderer. Unfreezing more of SR2 is not a response to this.")
    elif not actor_passed:
        print(f">>> ARM {arm.upper()} passed offline but its actor-like Rockstar "
              "verification is missing or failing.")
        print(">>> Run the probe + actor_rockstar_verify.py pipeline, then "
              "re-run this gate. Only real Rockstar can authorize fine-tuning.")
    print(f"  report -> {verdict_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
