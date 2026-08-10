#!/usr/bin/env python
"""Fit the proxy arms on set0-7. Nothing here ever reads set8-11.

Three arms, one job, on purpose
-------------------------------
Arm A sees the thirteen density features and their frozen-relative differences;
arm B sees those plus nine phase-space summaries and *their* differences; arm C
sees a per-tile token grid (:mod:`cosmo_sr.reward.soft_rockstar`) and learns its
own pooling. The plan requires that everything else be identical -- same
candidate fields, same labels, same output heads, same losses, splits, row
weights, epochs, bootstrap draws and training seeds -- so all arms are fitted in
**one process, from one table, in one loop**. Separate jobs with separate
configs would be separate chances for a difference nobody intended, and the
comparison's whole value is that only the tile representation changed.

There is **no hyperparameter sweep**. Each model class has one fixed,
predeclared configuration (``proxy:`` for the flat arms, ``proxy.arm_c:`` for
the token arm), declared before any held-out number was seen. The earlier
grouped-CV grid was removed deliberately: with three arms it multiplied the
budget and invited a winner chosen on tuning rather than on features.

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
within the same ``(box, unit)``, where the unit is the width-``region_width``
region of the tile grid -- the tile itself at the deployed width 1. A pair
across two units asks "which environment is richer", which is true, learnable,
and something the actor cannot change. A pair within one unit asks "which
version of this material is better", which is exactly the decision the actor's
gradient encodes.

Ensemble members are **box-bootstrap** samples, not re-initialisations. Five
seeds on identical data measure optimisation noise; the spread the actor's
lower-confidence bound needs is the spread over which boxes were seen, because
twelve boxes -- not fifty thousand tiles -- are the independent units here.

Arm C is optimised in group-aligned CHUNKS with gradient accumulation: both
losses are weighted means, so summing per-chunk weighted sums and dividing by
the global weight reproduces the full-batch loss exactly, while a 39k-row x
512-token batch never has to exist in memory at once. Chunks never split a
``(box, unit)`` group, so every ranking pair stays inside one chunk. The flat
arms run as a single chunk -- the identical numerics they always had.

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
    ARMS, COUNT_KEYS, as_arrays, build_row_context, ensemble_delta, load_rows,
    pooled_count_error, predicted_delta, rank_metrics, row_weights,
    true_delta_rewards, unit_ids_of,
)
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, direct_root,
    load_direct_config, load_reward_models, phase_space_config_of,
    region_width_of, run_dir, soft_config_of, soft_rockstar_config_of,
    write_json,
)

from cosmo_sr.reward.arms import arm_storage  # noqa: E402
from cosmo_sr.reward.catalog_proxy import (  # noqa: E402
    CatalogProxy, ProxyConfig, ProxyEnsemble, bin_weights_from_counts, count_loss,
    make_within_tile_pairs, pairwise_ranking_loss,
)
from cosmo_sr.reward.phase_space import arm_paired_feature_names  # noqa: E402
from cosmo_sr.reward.soft_rockstar import (  # noqa: E402
    SoftRockstarProxy, SoftRockstarProxyConfig, paired_token_feature_names,
)

#: Rows per optimisation chunk for the token arm. Group-aligned, so the real
#: chunk sizes wobble around this; 4096 rows x 512 tokens x 16 channels is a
#: ~0.5 GB forward, safe on CPU and GPU alike.
TOKEN_CHUNK_ROWS = 4096


# --------------------------------------------------------------------------- #
# Fixed model configurations -- one per model class, no sweep
# --------------------------------------------------------------------------- #
def make_model(arm: str, features: np.ndarray, arrays, pcfg: Dict,
               scale: np.ndarray, seed: int):
    """The one predeclared architecture for this arm's model class."""
    if arm_storage(arm) == "inline":
        cfg_m = ProxyConfig(
            n_features=int(features.shape[1]),
            n_sub_bins=int(arrays["n_sub"].shape[1]),
            n_host_bins=int(arrays["n_host"].shape[1]),
            hidden=tuple(int(h) for h in pcfg.get("hidden", (128, 128))),
            dropout=float(pcfg.get("dropout", 0.0)),
            output_scale=tuple(float(x) for x in scale))
        return CatalogProxy(cfg_m, seed=int(seed))
    ccfg = dict(pcfg.get("arm_c", {}))
    cfg_m = SoftRockstarProxyConfig(
        n_token_features=int(features.shape[-1]),
        n_tokens=int(features.shape[2]),
        n_sub_bins=int(arrays["n_sub"].shape[1]),
        n_host_bins=int(arrays["n_host"].shape[1]),
        token_hidden=tuple(int(h) for h in ccfg.get("token_hidden", (64, 64))),
        embed_dim=int(ccfg.get("embed_dim", 32)),
        pools=tuple(str(p) for p in ccfg.get("pools", ("mean", "max", "lse"))),
        dropout=float(pcfg.get("dropout", 0.0)),
        output_scale=tuple(float(x) for x in scale))
    return SoftRockstarProxy(cfg_m, seed=int(seed))


def feature_names_of(arm: str, cfg) -> List[str]:
    if arm_storage(arm) == "inline":
        return arm_paired_feature_names(arm, soft_config_of(cfg),
                                        phase_space_config_of(cfg))
    return paired_token_feature_names(soft_rockstar_config_of(cfg))


# --------------------------------------------------------------------------- #
# One fit
# --------------------------------------------------------------------------- #
def _pack_chunks(box: np.ndarray, unit: np.ndarray, max_rows: int) -> List[np.ndarray]:
    """Row-index chunks that never split a ``(box, unit)`` group.

    Ranking pairs exist only within a group, so group-aligned chunks let each
    pair be scored inside one forward pass; the count and ranking losses then
    decompose exactly across chunks (they are weighted means, and a weighted
    mean is a sum of per-chunk weighted sums over one global normaliser).
    """
    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, (b, u) in enumerate(zip(box, unit)):
        groups.setdefault((str(b), int(u)), []).append(i)
    chunks, cur = [], []
    for key in sorted(groups):
        if cur and len(cur) + len(groups[key]) > int(max_rows):
            chunks.append(np.asarray(cur, dtype=np.int64))
            cur = []
        cur.extend(groups[key])
    if cur:
        chunks.append(np.asarray(cur, dtype=np.int64))
    return chunks


def _train_member(*, arm: str, features: np.ndarray, arrays, ctx, reward_t,
                  targets, unit: np.ndarray, weight: np.ndarray, pcfg: Dict,
                  scale: np.ndarray, bin_w: torch.Tensor, w_joint: float,
                  w_occ: float, seed: int, epochs: int, device,
                  log_prefix: str = "") -> Tuple:
    """Fit one proxy member full-batch (in exact chunks) under the row weights.

    Rows of weight zero are the box-bootstrap's way of saying "this box was not
    drawn": they are still in the batch (so every index stays aligned with
    ``rows``) but contribute nothing to either loss. The standardiser is fitted
    on the drawn rows only, for the same reason it is fitted on the training
    rows only -- it is part of the model, not part of the data.
    """
    drawn = weight > 0
    if not drawn.any():
        raise ValueError("every row has zero weight; nothing to fit")

    model = make_model(arm, features, arrays, pcfg, scale, seed)
    model.fit_standardizer(features[drawn])
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(pcfg.get("lr", 1e-3)),
                           weight_decay=float(pcfg.get("weight_decay", 1e-4)))

    # The token arm's full feature tensor stays on the CPU; only the active
    # chunk visits the device. The flat arms fit comfortably and keep their
    # original single-resident-tensor behaviour.
    sidecar = arm_storage(arm) == "sidecar"
    x = torch.as_tensor(features, dtype=torch.float64,
                        device="cpu" if sidecar else device)
    y = {k: torch.as_tensor(arrays[k], dtype=torch.float64, device=device)
         for k in COUNT_KEYS}
    rw = torch.as_tensor(weight, dtype=torch.float64, device=device)

    pairs = make_within_tile_pairs(
        arrays["box"], unit, targets,
        max_pairs_per_group=int(pcfg.get("max_pairs_per_group", 32)),
        min_margin=float(pcfg.get("min_pair_margin", 0.0)),
        rng=np.random.default_rng(int(seed)))
    pw = np.zeros(0)
    if len(pairs):
        pw = np.minimum(weight[pairs[:, 0]], weight[pairs[:, 1]])
        keep = pw > 0
        pairs, pw = pairs[keep], pw[keep]

    max_rows = (TOKEN_CHUNK_ROWS if arm_storage(arm) == "sidecar"
                else features.shape[0])
    chunks = _pack_chunks(arrays["box"], unit, max_rows)
    # Assign every pair to its chunk once, via a global -> local index map.
    local = np.full(features.shape[0], -1, dtype=np.int64)
    chunk_of = np.full(features.shape[0], -1, dtype=np.int64)
    for ci, idx in enumerate(chunks):
        local[idx] = np.arange(idx.size)
        chunk_of[idx] = ci
    pair_sets = [[] for _ in chunks]
    for p in range(len(pairs)):
        a, b = int(pairs[p, 0]), int(pairs[p, 1])
        assert chunk_of[a] == chunk_of[b], "a pair crossed a chunk boundary"
        pair_sets[chunk_of[a]].append(p)
    pair_sets = [np.asarray(s, dtype=np.int64) for s in pair_sets]

    rank_w = float(pcfg.get("ranking_weight", 1.0))
    huber = float(pcfg.get("huber_delta", 1.0))
    total_rw = max(float(weight.sum()), 1e-12)
    total_pw = max(float(pw.sum()), 1e-12) if len(pairs) else 1.0

    t0 = time.time()
    total = float("nan")
    for epoch in range(int(epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for ci, idx in enumerate(chunks):
            it = torch.as_tensor(idx, device=device)
            xi = x[torch.as_tensor(idx)].to(device) if sidecar else x[it]
            pred = model(xi)
            # count_loss normalises by THIS chunk's row weight; rescaling by
            # the chunk's share of the global weight makes the sum over chunks
            # exactly the full-batch loss.
            frac = float(weight[idx].sum()) / total_rw
            loss_c = count_loss(pred, {k: y[k][it] for k in COUNT_KEYS},
                                weights=bin_w, row_weights=rw[it],
                                huber_delta=huber)["loss"] * frac
            ps = pair_sets[ci]
            if ps.size:
                dr = predicted_delta(model, xi, ctx.index(idx), reward_t,
                                     w_joint=w_joint, w_occ=w_occ)
                a = torch.as_tensor(local[pairs[ps, 0]], device=device)
                b = torch.as_tensor(local[pairs[ps, 1]], device=device)
                pwt = torch.as_tensor(pw[ps], dtype=torch.float64, device=device)
                pfrac = float(pw[ps].sum()) / total_pw
                loss_c = loss_c + rank_w * pfrac * \
                    pairwise_ranking_loss(dr[a], dr[b], pwt)
            loss_c.backward()
            total += float(loss_c.detach())
        opt.step()
        if log_prefix and epoch % max(1, int(epochs) // 5) == 0:
            print(f"  {log_prefix} epoch {epoch:4d} loss {total:.5f}", flush=True)
    return model.cpu(), {
        "seed": int(seed), "final_loss": float(total),
        "n_pairs": int(len(pairs)), "n_rows_drawn": int(drawn.sum()),
        "n_chunks": len(chunks), "seconds": round(time.time() - t0, 1),
    }


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
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    pcfg = dict(cfg.get("proxy", {}))
    cvcfg = dict(cfg.get("proxy_cv", {}))
    if bool(cvcfg.get("enabled", False)) or cvcfg.get("grid"):
        raise SystemExit(
            "proxy_cv is enabled or carries a grid, but the hyperparameter "
            "sweep was removed from this experiment: each model class trains "
            "once with its fixed predeclared configuration. Disable proxy_cv.")
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    device = torch.device(args.device)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    width = region_width_of(cfg)
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
    # construction: one table, one context, one target vector, one unit array.
    common = as_arrays(rows, arms[0] if arm_storage(arms[0]) == "inline" else "a",
                       table_dir=table.parent)
    ctx = build_row_context(rows)
    per_arm = {arm: as_arrays(rows, arm, table_dir=table.parent)["features"]
               for arm in arms}
    targets = true_delta_rewards(ctx, reward_t, w_joint=w_joint, w_occ=w_occ)
    unit = unit_ids_of(common["tile_id"], width=width)

    # Row weights come from arm A's difference block ALWAYS, even when only arm
    # B or C is being fitted. They are part of the shared training setup, not
    # part of a feature vector: derived per arm they would differ between the
    # arms and between a three-arm run and a single-arm one, and the comparison
    # would no longer be about features.
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

    # --- refit on all of set0-7, as a box-bootstrap ensemble ---------------
    n_members = int(pcfg.get("n_members", 5))
    epochs = int(pcfg.get("epochs", 400))
    boxes_present = sorted({str(b) for b in common["box"]})
    bootstrap = bool(pcfg.get("bootstrap_boxes", True)) and len(boxes_present) > 1
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
        "arms": arms,
        "model_selection": "fixed_predeclared",
        "proxy_unit": "region", "region_width": width,
        "hyperparameters": {
            "flat": {"hidden": list(pcfg.get("hidden", (128, 128))),
                     "lr": float(pcfg.get("lr", 1e-3)),
                     "weight_decay": float(pcfg.get("weight_decay", 1e-4))},
            "arm_c": dict(pcfg.get("arm_c", {})),
        },
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
        feats = per_arm[arm]
        shape_note = (f"{feats.shape[1]} features" if feats.ndim == 2
                      else f"{feats.shape[2]} tokens x {feats.shape[3]} channels")
        banner(f"arm {arm}: {shape_note}, {n_members} members")
        members, history = [], []
        for m, draw in enumerate(draws):
            mult = np.asarray([draw.get(str(b), 0) for b in common["box"]],
                              dtype=np.float64)
            # A box-bootstrap in a full-batch trainer is a multiplicity weight:
            # drawing a box twice doubles its rows' contribution to both losses,
            # which is exactly what resampling it twice would do, without
            # duplicating rows and breaking the (box, unit) grouping.
            w_m = weight * mult
            model, hist = _train_member(
                arm=arm, features=feats, arrays=common, ctx=ctx,
                reward_t=reward_t, targets=targets, unit=unit, weight=w_m,
                pcfg=pcfg, scale=scale, bin_w=bin_w, w_joint=w_joint,
                w_occ=w_occ, seed=m, epochs=epochs, device=device,
                log_prefix=f"arm {arm} member {m}")
            hist["boxes_drawn"] = draw
            members.append(model)
            history.append(hist)

        ens = ProxyEnsemble(members)
        ens.save(out_root / f"proxy_{arm}")
        # Names travel with the ensemble so the leave-one-out ablation can say
        # *which* coordinate is load-bearing rather than "feature 17".
        (out_root / f"proxy_{arm}" / "feature_names.json").write_text(
            json.dumps(feature_names_of(arm, cfg), indent=2))

        ft = torch.as_tensor(feats, dtype=torch.float64)
        mean, std = ensemble_delta(members, ft, ctx, reward_t,
                                   w_joint=w_joint, w_occ=w_occ)
        with torch.no_grad():
            pred_counts = [m_.summary(ft, ctx.tile_volume) for m_ in members]
        pooled = pooled_count_error(
            {k: torch.stack([getattr(p, k) for p in pred_counts]).mean(0).numpy()
             for k in COUNT_KEYS},
            {k: common[k] for k in COUNT_KEYS})

        # The linear baseline needs a flat design matrix. For the token arm the
        # bar is arm B's flat features -- "is the token grid better than a line
        # through the flat summaries of the same crop" is the honest question.
        bl_feats = feats if feats.ndim == 2 else as_arrays(rows, "b")["features"]
        bl = fit_baselines(bl_feats, {k: common[k] for k in COUNT_KEYS},
                           row_weight=weight)
        baselines[arm] = bl
        bl_report = {}
        for name, b in bl.items():
            with torch.no_grad():
                s = b.summary(torch.as_tensor(bl_feats, dtype=torch.float64),
                              ctx.tile_volume, ctx.frozen)
            bl_report[name] = pooled_count_error(
                {k: getattr(s, k).numpy() for k in COUNT_KEYS},
                {k: common[k] for k in COUNT_KEYS})

        report["arm_reports"][arm] = {
            "n_features": int(feats.shape[1] if feats.ndim == 2
                              else 2 * feats.shape[3]),
            "feature_shape": list(feats.shape[1:]),
            "members": history,
            "train_ranking": rank_metrics(mean, targets, common["box"], unit),
            "train_pooled": pooled,
            "train_baselines_pooled": {
                k: {"mean_log_error": v["mean_log_error"],
                    "occupation_log_error_max": v["occupation_log_error_max"]}
                for k, v in bl_report.items()},
            "mean_ensemble_std": float(np.nanmean(std)),
        }
        print(f"  arm {arm}: train within-unit spearman "
              f"{report['arm_reports'][arm]['train_ranking']['within_tile_spearman']:.4f}"
              f", pooled mean log error {pooled['mean_log_error']:.4f}", flush=True)

    save_baselines(out_root / "proxy_baselines.json", baselines)
    write_json(out_root / "train_report.json", report)
    banner(json.dumps({
        "model_selection": "fixed_predeclared",
        "arms": {a: report["arm_reports"][a]["train_ranking"] for a in arms},
    }, indent=2))
    print(f"  ensembles -> {out_root}/proxy_<arm>/", flush=True)
    print("  !! These are TRAINING numbers. The decision is made once, on "
          "set8-11, by scripts/reward/gate_catalog_proxy.py.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
