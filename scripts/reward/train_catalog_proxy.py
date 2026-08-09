#!/usr/bin/env python
"""Fit both proxy arms on set0-7. Nothing here ever reads set8-11.

Two arms, one job, on purpose
-----------------------------
Arm A sees the thirteen density features and their frozen-relative differences;
arm B sees those plus nine phase-space summaries and *their* differences. The
plan requires that everything else be identical -- same candidate fields, same
labels, same output heads, same architecture, optimiser, losses, splits and
training seeds -- so both arms are fitted in **one process, from one table, in
one loop**. Two jobs with two configs would be two chances for a difference
nobody intended, and the comparison's whole value is that only the feature
vector changed.

The same argument settles hyperparameters: the grouped cross-validation below
scores every grid point on the **mean over both arms** and picks one setting for
both. An arm allowed its own tuning could win the comparison on tuning budget.

What is fitted
--------------
Sixteen additive quantities per tile -- six subhalo-abundance bins, five host
bins, five occupation numerators -- never occupancy itself. Occupancy is
``sum S / sum H`` after pooling, a ratio of sums; a per-tile ratio is undefined
for the many tiles holding a fraction of one host, and averaging tile
occupancies is simply the wrong arithmetic.

Two losses, doing different jobs (see :mod:`cosmo_sr.reward.catalog_proxy`): a
Huber on ``log1p`` counts with capped per-bin weights, and a pairwise ranking
loss on the **true baseline-relative catalog reward** with pairs formed only
within the same ``(box, tile_id)``. The ranking pairs are the part that matters
and the part that is easy to get subtly wrong. A pair across two tiles asks
"which environment is richer", which is true, learnable, and something the actor
cannot change. A pair within one tile asks "which version of this tile is
better", which is exactly the decision the actor's gradient encodes.

Ensemble members are **box-bootstrap** samples, not re-initialisations. Five
seeds on identical data measure optimisation noise; the spread the actor's
lower-confidence bound needs is the spread over which boxes were seen, because
twelve boxes -- not fifty thousand tiles -- are the independent units here.

    python scripts/reward/train_catalog_proxy.py --run-name direct_a
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from _proxy_baselines import fit_baselines, save_baselines  # noqa: E402
from _proxy_data import (  # noqa: E402
    ARMS, COUNT_KEYS, as_arrays, build_row_context, ensemble_delta,
    group_kfold_by_box, load_rows, pooled_count_error, predicted_delta,
    rank_metrics, row_weights, true_delta_rewards,
)
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, direct_root,
    load_direct_config, load_reward_models, phase_space_config_of, run_dir,
    soft_config_of, write_json,
)

from cosmo_sr.reward.catalog_proxy import (  # noqa: E402
    CatalogProxy, ProxyConfig, ProxyEnsemble, bin_weights_from_counts, count_loss,
    make_within_tile_pairs, pairwise_ranking_loss,
)
from cosmo_sr.reward.phase_space import arm_paired_feature_names  # noqa: E402


# --------------------------------------------------------------------------- #
# One fit
# --------------------------------------------------------------------------- #
def _train_member(*, features: np.ndarray, arrays, ctx, reward_t, targets,
                  weight: np.ndarray, hp: Dict, pcfg: Dict, scale: np.ndarray,
                  bin_w: torch.Tensor, w_joint: float, w_occ: float,
                  seed: int, epochs: int, device, log_prefix: str = "") -> Tuple:
    """Fit one :class:`CatalogProxy` full-batch under the given row weights.

    Rows of weight zero are the box-bootstrap's way of saying "this box was not
    drawn": they are still in the batch (so every index stays aligned with
    ``rows``) but contribute nothing to either loss. The standardiser is fitted
    on the drawn rows only, for the same reason it is fitted on the training
    rows only -- it is part of the model, not part of the data.
    """
    drawn = weight > 0
    if not drawn.any():
        raise ValueError("every row has zero weight; nothing to fit")

    cfg_m = ProxyConfig(
        n_features=int(features.shape[1]),
        n_sub_bins=int(arrays["n_sub"].shape[1]),
        n_host_bins=int(arrays["n_host"].shape[1]),
        hidden=tuple(int(h) for h in hp["hidden"]),
        dropout=float(pcfg.get("dropout", 0.0)),
        output_scale=tuple(float(x) for x in scale))
    model = CatalogProxy(cfg_m, seed=int(seed)).fit_standardizer(
        features[drawn]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]),
                           weight_decay=float(hp["weight_decay"]))

    x = torch.as_tensor(features, dtype=torch.float64, device=device)
    y = {k: torch.as_tensor(arrays[k], dtype=torch.float64, device=device)
         for k in COUNT_KEYS}
    rw = torch.as_tensor(weight, dtype=torch.float64, device=device)

    pairs = make_within_tile_pairs(
        arrays["box"], arrays["tile_id"], targets,
        max_pairs_per_group=int(pcfg.get("max_pairs_per_group", 32)),
        min_margin=float(pcfg.get("min_pair_margin", 0.0)),
        rng=np.random.default_rng(int(seed)))
    if len(pairs):
        pw = np.minimum(weight[pairs[:, 0]], weight[pairs[:, 1]])
        keep = pw > 0
        pairs, pw = pairs[keep], pw[keep]
    if len(pairs):
        a = torch.as_tensor(pairs[:, 0], device=device)
        b = torch.as_tensor(pairs[:, 1], device=device)
        pwt = torch.as_tensor(pw, dtype=torch.float64, device=device)

    t0 = time.time()
    total = torch.tensor(float("nan"))
    for epoch in range(int(epochs)):
        model.train()
        pred = model(x)
        total = count_loss(pred, y, weights=bin_w, row_weights=rw,
                           huber_delta=float(pcfg.get("huber_delta", 1.0)))["loss"]
        if len(pairs):
            dr = predicted_delta(model, x, ctx, reward_t,
                                 w_joint=w_joint, w_occ=w_occ)
            total = total + float(pcfg.get("ranking_weight", 1.0)) * \
                pairwise_ranking_loss(dr[a], dr[b], pwt)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if log_prefix and epoch % max(1, int(epochs) // 5) == 0:
            print(f"  {log_prefix} epoch {epoch:4d} loss {float(total.detach()):.5f}",
                  flush=True)
    return model.cpu(), {
        "seed": int(seed), "final_loss": float(total.detach()),
        "n_pairs": int(len(pairs)), "n_rows_drawn": int(drawn.sum()),
        "seconds": round(time.time() - t0, 1),
    }


def _subset(arrays, ctx, idx: np.ndarray):
    return ({k: v[idx] for k, v in arrays.items()}, ctx.index(idx))


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(per_arm, arrays_common, ctx, reward_t, targets, weight, *,
                   cvcfg: Dict, pcfg: Dict, bin_w, scale, w_joint, w_occ,
                   device) -> Dict:
    """Grouped-by-box CV, scored on the mean over arms. One winner for both.

    The held-out fold is a set of *boxes* from set0-7. set8-11 do not appear in
    this function and must not: they are spent once, on the decision, and a
    hyperparameter chosen against them would spend them on tuning instead.
    """
    folds = group_kfold_by_box(arrays_common["box"], int(cvcfg.get("n_folds", 4)),
                               seed=0)
    grid = list(cvcfg.get("grid", []))
    objective = str(cvcfg.get("objective", "within_tile_spearman"))
    epochs = int(cvcfg.get("epochs", 200))
    table: List[Dict] = []

    for gi, hp in enumerate(grid):
        per_point: Dict[str, List[float]] = {arm: [] for arm in per_arm}
        for fi, held in enumerate(folds):
            va = np.nonzero(np.isin(arrays_common["box"], held))[0]
            tr = np.nonzero(~np.isin(arrays_common["box"], held))[0]
            if va.size == 0 or tr.size == 0:
                continue
            for arm, feats in per_arm.items():
                a_tr, c_tr = _subset(arrays_common, ctx, tr)
                a_va, c_va = _subset(arrays_common, ctx, va)
                model, _ = _train_member(
                    features=feats[tr], arrays=a_tr, ctx=c_tr, reward_t=reward_t,
                    targets=targets[tr], weight=weight[tr], hp=hp, pcfg=pcfg,
                    scale=scale, bin_w=bin_w, w_joint=w_joint, w_occ=w_occ,
                    seed=1000 * gi + fi, epochs=epochs, device=device)
                pred, _ = ensemble_delta([model],
                                         torch.as_tensor(feats[va],
                                                         dtype=torch.float64),
                                         c_va, reward_t,
                                         w_joint=w_joint, w_occ=w_occ)
                m = rank_metrics(pred, targets[va], a_va["box"], a_va["tile_id"])
                per_point[arm].append(float(m[objective]))
        row = {"grid_index": gi, "hyperparameters": hp}
        for arm, vals in per_point.items():
            row[f"{arm}_{objective}"] = (float(np.nanmean(vals)) if vals
                                         else float("nan"))
        finite = [row[f"{arm}_{objective}"] for arm in per_arm]
        row["mean_over_arms"] = (float(np.nanmean(finite))
                                 if np.any(np.isfinite(finite)) else float("nan"))
        table.append(row)
        print(f"  CV grid {gi} {hp} -> " + "  ".join(
            f"{arm}={row[f'{arm}_{objective}']:.4f}" for arm in per_arm), flush=True)

    scores = [r["mean_over_arms"] for r in table]
    best = int(np.nanargmax(scores)) if np.any(np.isfinite(scores)) else 0
    return {"folds": [list(f) for f in folds], "objective": objective,
            "table": table, "selected_index": best,
            "selected": table[best]["hyperparameters"] if table else None}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--table", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="fit on a partial table; nothing gated may use the result")
    ap.add_argument("--skip-cv", action="store_true",
                    help="use the configured proxy: block directly (debugging)")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    pcfg = dict(cfg.get("proxy", {}))
    cvcfg = dict(cfg.get("proxy_cv", {}))
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    device = torch.device(args.device)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    arms = [a for a in args.arms if a in ARMS]
    if not arms:
        raise SystemExit(f"--arms must name at least one of {list(ARMS)}")

    table = Path(args.table) if args.table else direct_root("proxy_data") / "rows.jsonl"
    rows_all = load_rows(table, require_complete=not args.allow_incomplete)
    if not rows_all:
        raise SystemExit(f"{table} is empty")

    fit_boxes, gate_boxes = boxes_of(cfg, "proxy_fit"), boxes_of(cfg, "proxy_gate")
    overlap = sorted(set(fit_boxes) & set(gate_boxes))
    if overlap:
        raise SystemExit(
            f"boxes {overlap} are in both the fit and the gate split; a box-level "
            "split is the only thing standing between this proxy and a leaked "
            "held-out score")
    rows = [r for r in rows_all if r["box"] in set(fit_boxes)]
    if not rows:
        raise SystemExit(f"no rows from the fit boxes {fit_boxes}")
    banner(f"{len(rows_all)} rows total; fitting on {len(rows)} from "
           f"{sorted({r['box'] for r in rows})}. set8-11 are not read here.")

    # Everything except the feature block is shared between the arms by
    # construction: one table, one context, one target vector.
    common = as_arrays(rows, arms[0])
    ctx = build_row_context(rows)
    per_arm = {arm: as_arrays(rows, arm)["features"] for arm in arms}
    targets = true_delta_rewards(ctx, reward_t, w_joint=w_joint, w_occ=w_occ)

    # Row weights come from arm A's difference block ALWAYS, even when only arm
    # B is being fitted. They are part of the shared training setup, not part of
    # a feature vector: derived per arm they would differ between the arms and
    # between a both-arms run and a single-arm one, and the comparison would no
    # longer be about features.
    rw = row_weights(
        common["source"], as_arrays(rows, "a")["features"],
        balance_sources=bool(pcfg.get("balance_sources", True)),
        changed_weight=float(pcfg.get("changed_tile_weight", 1.0)),
        changed_threshold=float(pcfg.get("changed_tile_threshold", 0.5)))
    weight = rw["weight"]
    banner(f"{int(rw['changed'].sum())} of {weight.size} rows differ from their "
           f"frozen reference; source balance "
           + ", ".join(f"{s}:{c}" for s, c in
                       zip(*np.unique(common["source"], return_counts=True))))

    counts = np.concatenate([common[k] for k in COUNT_KEYS], axis=1)
    bin_w = torch.as_tensor(
        bin_weights_from_counts(counts, cap=float(pcfg.get("bin_weight_cap", 8.0)),
                                floor=float(pcfg.get("bin_weight_floor", 0.25))),
        dtype=torch.float64, device=device)
    # Start the network at the right magnitude instead of learning the units.
    scale = np.maximum(counts.mean(axis=0), 1e-3)

    # --- model selection ---------------------------------------------------
    default_hp = {"hidden": list(pcfg.get("hidden", (128, 128))),
                  "lr": float(pcfg.get("lr", 1e-3)),
                  "weight_decay": float(pcfg.get("weight_decay", 1e-4))}
    if bool(cvcfg.get("enabled", True)) and not args.skip_cv and cvcfg.get("grid"):
        banner("grouped cross-validation on the fit boxes (shared across arms)")
        cv = cross_validate(per_arm, common, ctx, reward_t, targets, weight,
                            cvcfg=cvcfg, pcfg=pcfg, bin_w=bin_w, scale=scale,
                            w_joint=w_joint, w_occ=w_occ, device=device)
        hp = cv["selected"] or default_hp
    else:
        cv = {"skipped": True, "reason": ("--skip-cv" if args.skip_cv
                                          else "proxy_cv.enabled is false")}
        hp = default_hp
    hp = {"hidden": list(hp["hidden"]), "lr": float(hp["lr"]),
          "weight_decay": float(hp["weight_decay"])}
    banner(f"shared hyperparameters for BOTH arms: {hp}")

    # --- refit on all of set0-7, as a box-bootstrap ensemble ---------------
    n_members = int(pcfg.get("n_members", 5))
    epochs = int(pcfg.get("epochs", 400))
    boxes_present = sorted({str(b) for b in common["box"]})
    bootstrap = bool(cvcfg.get("bootstrap_boxes", True)) and len(boxes_present) > 1
    draws: List[Dict[str, int]] = []
    for m in range(n_members):
        if not bootstrap:
            draws.append({b: 1 for b in boxes_present})
            continue
        rng = np.random.default_rng(10_000 + m)
        picked = rng.choice(boxes_present, size=len(boxes_present), replace=True)
        uniq, cnt = np.unique(picked, return_counts=True)
        draws.append({str(b): int(c) for b, c in zip(uniq, cnt)})

    out_root = run_dir(args.run_name, create=True)
    report: Dict = {
        "table": str(table), "n_rows_total": len(rows_all), "n_rows_fit": len(rows),
        "fit_boxes": fit_boxes, "gate_boxes": gate_boxes,
        "arms": arms, "hyperparameters": hp, "cross_validation": cv,
        "bin_weights": bin_w.cpu().tolist(),
        "output_scale": [float(x) for x in scale],
        "bootstrap_draws": draws,
        "row_weighting": {
            "balance_sources": bool(pcfg.get("balance_sources", True)),
            "changed_tile_weight": float(pcfg.get("changed_tile_weight", 1.0)),
            "changed_tile_threshold": float(pcfg.get("changed_tile_threshold", 0.5)),
            "n_changed_rows": int(rw["changed"].sum()),
        },
        "allow_incomplete": bool(args.allow_incomplete),
        "arm_reports": {},
    }

    baselines: Dict[str, Dict] = {}
    for arm in arms:
        banner(f"arm {arm}: {per_arm[arm].shape[1]} features, {n_members} members")
        feats = per_arm[arm]
        members, history = [], []
        for m, draw in enumerate(draws):
            mult = np.asarray([draw.get(str(b), 0) for b in common["box"]],
                              dtype=np.float64)
            # A box-bootstrap in a full-batch trainer is a multiplicity weight:
            # drawing a box twice doubles its rows' contribution to both losses,
            # which is exactly what resampling it twice would do, without
            # duplicating rows and breaking the (box, tile) grouping.
            w_m = weight * mult
            model, hist = _train_member(
                features=feats, arrays=common, ctx=ctx, reward_t=reward_t,
                targets=targets, weight=w_m, hp=hp, pcfg=pcfg, scale=scale,
                bin_w=bin_w, w_joint=w_joint, w_occ=w_occ, seed=m, epochs=epochs,
                device=device, log_prefix=f"arm {arm} member {m}")
            hist["boxes_drawn"] = draw
            members.append(model)
            history.append(hist)

        ens = ProxyEnsemble(members)
        ens.save(out_root / f"proxy_{arm}")
        # Names travel with the ensemble so the leave-one-out ablation can say
        # *which* coordinate is load-bearing rather than "feature 17".
        (out_root / f"proxy_{arm}" / "feature_names.json").write_text(
            json.dumps(arm_paired_feature_names(arm, soft_config_of(cfg),
                                                phase_space_config_of(cfg)),
                       indent=2))

        ft = torch.as_tensor(feats, dtype=torch.float64)
        mean, std = ensemble_delta(members, ft, ctx, reward_t,
                                   w_joint=w_joint, w_occ=w_occ)
        with torch.no_grad():
            pred_counts = [m_.summary(ft, ctx.tile_volume) for m_ in members]
        pooled = pooled_count_error(
            {k: torch.stack([getattr(p, k) for p in pred_counts]).mean(0).numpy()
             for k in COUNT_KEYS},
            {k: common[k] for k in COUNT_KEYS})

        bl = fit_baselines(feats, {k: common[k] for k in COUNT_KEYS},
                           row_weight=weight)
        baselines[arm] = bl
        bl_report = {}
        for name, b in bl.items():
            with torch.no_grad():
                s = b.summary(ft, ctx.tile_volume, ctx.frozen)
            bl_report[name] = pooled_count_error(
                {k: getattr(s, k).numpy() for k in COUNT_KEYS},
                {k: common[k] for k in COUNT_KEYS})

        report["arm_reports"][arm] = {
            "n_features": int(feats.shape[1]),
            "members": history,
            "train_ranking": rank_metrics(mean, targets, common["box"],
                                          common["tile_id"]),
            "train_pooled": pooled,
            "train_baselines_pooled": {
                k: {"mean_log_error": v["mean_log_error"],
                    "occupation_log_error_max": v["occupation_log_error_max"]}
                for k, v in bl_report.items()},
            "mean_ensemble_std": float(np.nanmean(std)),
        }
        print(f"  arm {arm}: train within-tile spearman "
              f"{report['arm_reports'][arm]['train_ranking']['within_tile_spearman']:.4f}"
              f", pooled mean log error {pooled['mean_log_error']:.4f}", flush=True)

    save_baselines(out_root / "proxy_baselines.json", baselines)
    write_json(out_root / "train_report.json", report)
    banner(json.dumps({
        "hyperparameters": hp,
        "arms": {a: report["arm_reports"][a]["train_ranking"] for a in arms},
    }, indent=2))
    print(f"  ensembles -> {out_root}/proxy_<arm>/", flush=True)
    print("  !! These are TRAINING numbers. The decision is made once, on "
          "set8-11, by scripts/reward/gate_catalog_proxy.py.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
