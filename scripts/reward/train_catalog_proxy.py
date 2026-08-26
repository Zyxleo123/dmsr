#!/usr/bin/env python
"""Fit the proxy arms on set0-7. Nothing here ever reads set8-11.

Six arms, identical everything else
-----------------------------------
Arm A sees the density features and their frozen-relative differences; arm B
adds phase space; arm C reads a per-tile token grid with permutation-invariant
pooling; arm D reads the SAME token grid through a 3-D CNN; arm E reads the full
32^3 phase-space grid through a strided CNN; arm F is the SR2 discriminator
architecture on its exact 20-channel input. Everything else is identical -- same
candidate fields, same labels, same output heads, same losses, splits, row
weights, bootstrap draws and training seeds -- so the comparison's whole value
is that only the tile representation changed.

``submit_proxy_benchmark.sh fit`` launches one GPU job per arm. Shared state is
kept identical across those siblings: the first writer stamps
``bootstrap_draws.json`` under a flock and every later arm reuses it; baselines
and ``train_report.json`` merge under the same lock. Packing several arms into
one ``--arms`` invocation still works (sequential within that process).

There is **no hyperparameter sweep**. Each model class has one fixed,
predeclared configuration (``proxy:`` for the flat arms, ``proxy.arm_c/arm_d/
arm_e/arm_f:`` for the spatial arms), declared before any held-out number was
seen. The spatial arms train for a fixed, smaller epoch budget (``spatial_epochs``,
~50 with AMP) than the MLP (400): a convolution over 512 tokens or 32^3 cells is a
far larger step than a two-layer MLP on 44 numbers.

The objective (see :mod:`cosmo_sr.reward.catalog_proxy`)
--------------------------------------------------------
Every arm predicts a **signed frozen-relative log-count residual** and
reconstructs counts against the measured frozen tile. Three losses:

* a normalised **reward-change regression** ``Huber((dR_hat - dR) / sigma_R)`` --
  direct supervision on the magnitude the actor is scored on, with ``sigma_R`` a
  robust scale of the true ``dR`` on the fit boxes only;
* the **pairwise ranking** loss, on the true baseline-relative reward, within
  each ``(box, unit)``, with the priority pairs (frozen vs each intervention,
  then adjacent alpha) built before any random fill;
* a **small absolute-count calibration** term (Huber on ``log1p`` counts).

Row and pair weighting fold in source balance, a 3x emphasis on tiles the raw
field actually changed (the arm-neutral ``field_changed`` flag), and the
box-bootstrap multiplicity. Ensemble members are box-bootstrap samples, not
re-initialisations, because twelve boxes -- not fifty thousand tiles -- are the
independent units.

Streaming
---------
Features are read through :func:`_proxy_data.make_arm_features`, which memory-maps
the sidecars (arm E is ~160 GB as float64) and builds arm F's 20-channel input on
the GPU from the raw field. The step runs in group-aligned chunks so a
``(box, unit)`` ranking pair never crosses a chunk boundary and both losses
decompose exactly across chunks. The chunk size is per arm: one chunk for the flat
arms (their original numerics), a few thousand rows for the tokens, a few hundred
for the dense grid, and a few dozen for the streamed field arm.

    python scripts/reward/train_catalog_proxy.py --run-name direct_a --device cuda
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from _proxy_baselines import fit_baselines, load_baselines, save_baselines  # noqa: E402
from _proxy_data import (  # noqa: E402
    ARMS, COUNT_KEYS, as_arrays, build_row_context, delta_of_summary, load_rows,
    make_arm_features, pair_weights, pooled_count_error,
    pooled_count_error_by_candidate, rank_metrics, robust_scale, row_weights,
    stream_ensemble_delta, stream_pred_counts, true_delta_rewards, unit_ids_of,
)
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, direct_root,
    load_direct_config, load_reward_models, phase_space_config_of,
    region_width_of, run_dir, soft_config_of, soft_rockstar_config_of,
    write_json_atomic,
)

from cosmo_sr.reward.arms import arm_storage  # noqa: E402
from cosmo_sr.train.common import finish_wandb, maybe_init_wandb  # noqa: E402
from cosmo_sr.reward.catalog_proxy import (  # noqa: E402
    CatalogProxy, ProxyConfig, ProxyEnsemble, bin_weights_from_counts, count_loss,
    make_within_tile_pairs, pairwise_ranking_loss, reward_change_loss,
)
from cosmo_sr.reward.phase_space import (  # noqa: E402
    arm_paired_feature_names, phase_space_grid_channel_names,
)
from cosmo_sr.reward.soft_rockstar import paired_token_feature_names  # noqa: E402
from cosmo_sr.reward.spatial_proxy import (  # noqa: E402
    FullGridProxy, FullGridProxyConfig, SR2DiscriminatorProxy,
    SR2DiscriminatorProxyConfig, SpatialTokenProxy, SpatialTokenProxyConfig,
)


def chunk_rows_for(arm: str, n_rows: int) -> int:
    """Rows per optimisation chunk, by arm -- group-aligned, so a wobble is fine.

    One chunk for the flat arms keeps their exact original numerics; the spatial
    arms are capped by how much of their input fits on the GPU at once, which is
    the whole reason the step is chunked. Arm F's 20x64^3 input is ~20 MB a tile,
    so its chunk is a few dozen rows.
    """
    st = arm_storage(arm)
    if st == "inline":
        return int(n_rows)
    return {"c": 4096, "d": 4096, "e": 512, "f": 48}[arm]


# --------------------------------------------------------------------------- #
# Fixed model configurations -- one per model class, no sweep
# --------------------------------------------------------------------------- #
def make_model(arm: str, provider, arrays, pcfg: Dict, scale: np.ndarray, seed: int):
    """The one predeclared architecture for this arm's model class."""
    shape = provider.per_row_shape
    j = int(arrays["n_sub"].shape[1])
    i = int(arrays["n_host"].shape[1])
    scale_t = tuple(float(x) for x in scale)
    if arm_storage(arm) == "inline":
        cfg_m = ProxyConfig(
            n_features=int(shape[0]), n_sub_bins=j, n_host_bins=i,
            hidden=tuple(int(h) for h in pcfg.get("hidden", (128, 128))),
            dropout=float(pcfg.get("dropout", 0.0)), output_scale=scale_t)
        return CatalogProxy(cfg_m, seed=int(seed))
    if arm == "c":
        from cosmo_sr.reward.soft_rockstar import (SoftRockstarProxy,
                                                   SoftRockstarProxyConfig)
        ccfg = dict(pcfg.get("arm_c", {}))
        cfg_m = SoftRockstarProxyConfig(
            n_token_features=int(shape[-1]), n_tokens=int(shape[1]),
            n_sub_bins=j, n_host_bins=i,
            token_hidden=tuple(int(h) for h in ccfg.get("token_hidden", (64, 64))),
            embed_dim=int(ccfg.get("embed_dim", 32)),
            pools=tuple(str(p) for p in ccfg.get("pools", ("mean", "max", "lse"))),
            dropout=float(pcfg.get("dropout", 0.0)), output_scale=scale_t)
        return SoftRockstarProxy(cfg_m, seed=int(seed))
    if arm == "d":
        dcfg = dict(pcfg.get("arm_d", {}))
        cfg_m = SpatialTokenProxyConfig(
            n_token_features=int(shape[-1]), n_tokens=int(shape[1]),
            n_sub_bins=j, n_host_bins=i,
            channels=tuple(int(c) for c in dcfg.get("channels", (32, 64, 64))),
            strides=tuple(int(s) for s in dcfg.get("strides", (1, 2, 1))),
            dropout=float(pcfg.get("dropout", 0.0)), output_scale=scale_t)
        return SpatialTokenProxy(cfg_m, seed=int(seed))
    if arm == "e":
        ecfg = dict(pcfg.get("arm_e", {}))
        cfg_m = FullGridProxyConfig(
            n_grid_channels=int(shape[1]), grid_size=int(shape[2]),
            n_sub_bins=j, n_host_bins=i,
            channels=tuple(int(c) for c in ecfg.get("channels", (16, 32, 64))),
            strides=tuple(int(s) for s in ecfg.get("strides", (2, 2, 2))),
            dropout=float(pcfg.get("dropout", 0.0)), output_scale=scale_t)
        return FullGridProxy(cfg_m, seed=int(seed))
    if arm == "f":
        fcfg = dict(pcfg.get("arm_f", {}))
        cfg_m = SR2DiscriminatorProxyConfig(
            in_channels=int(shape[0]), width=int(fcfg.get("width", 64)),
            depth=int(fcfg.get("depth", 4)), n_sub_bins=j, n_host_bins=i,
            output_scale=scale_t)
        return SR2DiscriminatorProxy(cfg_m, seed=int(seed))
    raise ValueError(f"no model factory for arm {arm!r}")


def feature_names_of(arm: str, cfg) -> List[str]:
    if arm_storage(arm) == "inline":
        return arm_paired_feature_names(arm, soft_config_of(cfg),
                                        phase_space_config_of(cfg))
    if arm in ("c", "d"):
        return paired_token_feature_names(soft_rockstar_config_of(cfg))
    if arm == "e":
        base = phase_space_grid_channel_names()
        return list(base) + [f"d_{n}" for n in base]
    # arm f: the 20 critic-input channels, named by their block.
    return ([f"up_lr_{k}" for k in range(6)] + [f"field_{k}" for k in range(6)]
            + [f"density_{k}" for k in range(8)])


# --------------------------------------------------------------------------- #
# Chunking (shared logic; the chunk SIZE is per arm)
# --------------------------------------------------------------------------- #
def _pack_chunks(box: np.ndarray, unit: np.ndarray, max_rows: int) -> List[np.ndarray]:
    """Row-index chunks that never split a ``(box, unit)`` group."""
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


def _assign_pairs(chunks: List[np.ndarray], pairs: np.ndarray, n_rows: int):
    """Global->local index maps and each chunk's pair set (pairs never cross)."""
    local = np.full(n_rows, -1, dtype=np.int64)
    chunk_of = np.full(n_rows, -1, dtype=np.int64)
    for ci, idx in enumerate(chunks):
        local[idx] = np.arange(idx.size)
        chunk_of[idx] = ci
    pair_sets = [[] for _ in chunks]
    for p in range(len(pairs)):
        a, b = int(pairs[p, 0]), int(pairs[p, 1])
        assert chunk_of[a] == chunk_of[b], "a pair crossed a chunk boundary"
        pair_sets[chunk_of[a]].append(p)
    return local, [np.asarray(s, dtype=np.int64) for s in pair_sets]


# --------------------------------------------------------------------------- #
# Shared state across parallel per-arm jobs
# --------------------------------------------------------------------------- #
def _fit_lock(out_root: Path):
    """Exclusive flock on ``.proxy_fit.lock`` under ``out_root``.

    Used for the bootstrap-draws stamp and for merging baselines / train_report
    so sibling GPU jobs for different arms do not race.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    lock_path = out_root / ".proxy_fit.lock"
    fh = open(lock_path, "a+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def _shared_bootstrap_draws(out_root: Path, boxes_present: List[str],
                            n_members: int, *, bootstrap: bool
                            ) -> List[Dict[str, int]]:
    """Stamp or reuse ``bootstrap_draws.json`` so every arm sees the same draws.

    The first parallel fit job to take the lock writes the file; every later job
    loads it. Deterministic given ``n_members`` and the sorted box list.
    """
    path = out_root / "bootstrap_draws.json"
    with _fit_lock(out_root) as _lock:
        if path.is_file():
            draws = json.loads(path.read_text())
            if len(draws) != int(n_members):
                raise SystemExit(
                    f"{path} has {len(draws)} members but this run asked for "
                    f"{n_members}; delete it (or match n_members) before refitting")
            return [{str(k): int(v) for k, v in d.items()} for d in draws]
        draws: List[Dict[str, int]] = []
        for m in range(int(n_members)):
            if not bootstrap:
                draws.append({b: 1 for b in boxes_present})
                continue
            rng = np.random.default_rng(10_000 + m)
            picked = rng.choice(boxes_present, size=len(boxes_present), replace=True)
            uniq, cnt = np.unique(picked, return_counts=True)
            draws.append({str(b): int(c) for b, c in zip(uniq, cnt)})
        write_json_atomic(path, draws)
        return draws


def _merge_arm_artifacts(out_root: Path, baselines: Dict, report: Dict,
                         arms: List[str]) -> None:
    """Flock-merge this job's baselines / arm_reports into the shared files."""
    bl_path = out_root / "proxy_baselines.json"
    rep_path = out_root / "train_report.json"
    with _fit_lock(out_root) as _lock:
        if bl_path.is_file():
            merged = load_baselines(bl_path)
            merged.update(baselines)
            baselines = merged
        save_baselines(bl_path, baselines)
        if rep_path.is_file():
            prev = json.loads(rep_path.read_text())
            prev_reports = dict(prev.get("arm_reports", {}))
            prev_reports.update(report["arm_reports"])
            report["arm_reports"] = prev_reports
            report["arms"] = sorted(set(prev.get("arms", [])) | set(arms))
        write_json_atomic(rep_path, report)


# --------------------------------------------------------------------------- #
# One member
# --------------------------------------------------------------------------- #
def _train_member(*, arm, provider, arrays, ctx, reward_t, targets, unit, weight,
                  mult, changed, pairs, kinds, chunks, pcfg, scale, bin_w, sigma_r,
                  w_joint, w_occ, seed, epochs, device, lambdas,
                  reward_key: str = "dR_combined",
                  log_prefix: str = "",
                  wlog: Optional[Callable[[Dict], None]] = None,
                  wtag: str = "") -> Tuple:
    """Fit one proxy member full-batch (in exact chunks) under the row weights.

    Rows of weight zero are the box-bootstrap's way of saying "this box was not
    drawn": they stay in the batch (so every index stays aligned) but contribute
    nothing. The standardiser is fitted on the drawn rows only.
    """
    drawn = weight > 0
    if not drawn.any():
        raise ValueError("every row has zero weight; nothing to fit")

    model = make_model(arm, provider, arrays, pcfg, scale, seed).to(device)
    if hasattr(model, "fit_standardizer"):
        model.fit_standardizer(
            provider.standardizer_sample(np.nonzero(drawn)[0], seed=int(seed)))
    opt = torch.optim.Adam(model.parameters(), lr=float(pcfg.get("lr", 1e-3)),
                           weight_decay=float(pcfg.get("weight_decay", 1e-4)))

    y = {k: torch.as_tensor(arrays[k], dtype=torch.float64, device=device)
         for k in COUNT_KEYS}
    rw = torch.as_tensor(weight, dtype=torch.float64, device=device)
    tgt = torch.as_tensor(targets, dtype=torch.float64, device=device)

    # Per-member pair weights: box-bootstrap multiplicity x pair-type x 3x-changed.
    type_w = {0: float(pcfg.get("pair_weight_frozen_vs_intervention", 1.0)),
              1: float(pcfg.get("pair_weight_adjacent_alpha", 1.0)),
              2: float(pcfg.get("pair_weight_random", 1.0))}
    pw = pair_weights(pairs, kinds, mult, changed, type_weights=type_w,
                      changed_weight=float(pcfg.get("changed_pair_weight", 3.0)))

    _, pair_sets = _assign_pairs(chunks, pairs, weight.size)
    # Global->local for indexing dR within a chunk.
    local = np.full(weight.size, -1, dtype=np.int64)
    for idx in chunks:
        local[idx] = np.arange(idx.size)

    count_w = float(lambdas["count_calib"])
    rank_w = float(lambdas["ranking"])
    dr_w = float(lambdas["reward_change"])
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
            xi = provider.get(idx, device=device)
            sub = ctx.index(idx)
            s = model.summary(xi, sub.tile_volume, sub.frozen)
            pred = {"n_sub": s.n_sub, "n_host": s.n_host,
                    "occ_numerator": s.occ_numerator}
            frac = float(weight[idx].sum()) / total_rw
            loss = count_w * frac * count_loss(
                pred, {k: y[k][it] for k in COUNT_KEYS}, weights=bin_w,
                row_weights=rw[it], huber_delta=huber)["loss"]
            dr = delta_of_summary(s, sub, reward_t, w_joint=w_joint, w_occ=w_occ,
                                  key=reward_key)
            loss = loss + dr_w * frac * reward_change_loss(
                dr, tgt[it], sigma_r, weights=rw[it], huber_delta=huber)
            ps = pair_sets[ci]
            if ps.size:
                a = torch.as_tensor(local[pairs[ps, 0]], device=device)
                b = torch.as_tensor(local[pairs[ps, 1]], device=device)
                pwt = torch.as_tensor(pw[ps], dtype=torch.float64, device=device)
                pfrac = float(pw[ps].sum()) / total_pw
                loss = loss + rank_w * pfrac * pairwise_ranking_loss(dr[a], dr[b], pwt)
            loss.backward()
            total += float(loss.detach())
        opt.step()
        # A member is many full-batch epochs; report on a cadence that stays
        # readable in the .out log (~20 lines/member) and carries an ETA, and
        # mirror every epoch to wandb where it is watchable live. metrics stay in
        # train_report.json as the source of truth, so a wandb outage loses
        # nothing.
        if wlog is not None:
            wlog({f"{wtag}/loss": total, f"{wtag}/epoch": epoch})
        if log_prefix and (epoch % max(1, int(epochs) // 20) == 0
                           or epoch == int(epochs) - 1):
            done = epoch + 1
            rate = (time.time() - t0) / done
            eta = rate * (int(epochs) - done)
            print(f"  {log_prefix} epoch {epoch:4d}/{int(epochs)} "
                  f"loss {total:.5f}  {rate:.2f}s/ep  eta {eta:5.0f}s", flush=True)
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="fit on a partial table; nothing gated may use the result")
    ap.add_argument("--no-wandb", action="store_true",
                    help="disable Weights & Biases logging (overrides wandb.mode)")
    ap.add_argument("--wandb-mode", default="",
                    help="override wandb.mode (online/offline/disabled)")
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
    device = torch.device(args.device if (args.device != "cuda"
                                          or torch.cuda.is_available()) else "cpu")
    if str(device) != args.device:
        print(f">>> requested device {args.device} unavailable; using {device}",
              flush=True)
    reward_t = reward_t.to(device)
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

    # Which reward the proxy predicts, and any source excluded from the fit.
    reward_key = str(pcfg.get("reward_target", "dR_combined"))
    drop_sources = {str(s) for s in (pcfg.get("drop_sources") or [])}
    if drop_sources:
        before = len(rows)
        rows = [r for r in rows if str(r["source"]) not in drop_sources]
        if not rows:
            raise SystemExit(
                f"drop_sources={sorted(drop_sources)} removed every fit row")
        banner(f"drop_sources={sorted(drop_sources)}: {before} -> {len(rows)} rows")
    banner(f"{len(rows_all)} rows total; fitting on {len(rows)} from "
           f"{sorted({r['box'] for r in rows})} with target {reward_key}. "
           "set8-11 are not read here.")

    # Everything except the feature block is shared between the arms.
    common = as_arrays(rows, "a", table_dir=table.parent)
    ctx = build_row_context(rows).to(device)
    targets = true_delta_rewards(ctx, reward_t, w_joint=w_joint, w_occ=w_occ,
                                 key=reward_key)
    unit = unit_ids_of(common["tile_id"], width=width)
    sigma_r = robust_scale(targets)

    # Row weights come from the arm-neutral field_changed flag ALWAYS -- it is
    # part of the shared training setup, not a feature vector, so it is identical
    # across the arms and between a six-arm run and a single-arm one.
    rw = row_weights(
        common["source"], common["field_changed"],
        balance_sources=bool(pcfg.get("balance_sources", True)),
        changed_weight=float(pcfg.get("changed_tile_weight", 3.0)))
    weight = rw["weight"]
    changed = rw["changed"]
    banner(f"{int(changed.sum())} of {weight.size} rows changed from frozen; "
           f"sigma_R={sigma_r:.4g}; source balance "
           + ", ".join(f"{s}:{c}" for s, c in
                       zip(*np.unique(common["source"], return_counts=True))))

    # Priority pairs, built ONCE and shared across arms and members.
    pairs, kinds = make_within_tile_pairs(
        common["box"], unit, targets, source=common["source"],
        alpha=common["alpha"], mode=common["mode"],
        max_pairs_per_group=int(pcfg.get("max_pairs_per_group", 32)),
        min_margin=float(pcfg.get("min_pair_margin", 0.0)),
        rng=np.random.default_rng(0), return_kinds=True)

    counts = np.concatenate([common[k] for k in COUNT_KEYS], axis=1)
    bin_w = torch.as_tensor(
        bin_weights_from_counts(counts, cap=float(pcfg.get("bin_weight_cap", 8.0)),
                                floor=float(pcfg.get("bin_weight_floor", 0.25))),
        dtype=torch.float64, device=device)
    scale = np.maximum(counts.mean(axis=0), 1e-3)

    lambdas = {
        "count_calib": float(pcfg.get("count_calibration_weight", 0.1)),
        "ranking": float(pcfg.get("ranking_weight", 1.0)),
        "reward_change": float(pcfg.get("reward_change_weight", 1.0)),
    }
    flat_epochs = int(pcfg.get("epochs", 400))
    # Default to the flat budget, not a smaller one: the larger spatial nets need
    # at least as many full-batch updates to converge (see the config note).
    spatial_epochs = int(pcfg.get("spatial_epochs", flat_epochs))

    # --- box-bootstrap ensemble --------------------------------------------- #
    n_members = int(pcfg.get("n_members", 5))
    boxes_present = sorted({str(b) for b in common["box"]})
    bootstrap = bool(pcfg.get("bootstrap_boxes", True)) and len(boxes_present) > 1
    out_root = run_dir(args.run_name, create=True)
    draws = _shared_bootstrap_draws(
        out_root, boxes_present, n_members, bootstrap=bootstrap)

    report: Dict = {
        "table": str(table), "n_rows_total": len(rows_all), "n_rows_fit": len(rows),
        "fit_boxes": fit_boxes, "gate_boxes": gate_boxes, "arms": arms,
        "device": str(device), "model_selection": "fixed_predeclared",
        "proxy_unit": "region", "region_width": width,
        "objective": {"losses": ["reward_change_huber", "pairwise_ranking",
                                  "count_calibration"],
                      "lambdas": lambdas, "sigma_R": float(sigma_r),
                      "head": "frozen_relative_residual"},
        "hyperparameters": {
            "flat": {"hidden": list(pcfg.get("hidden", (128, 128))),
                     "epochs": flat_epochs, "lr": float(pcfg.get("lr", 1e-3))},
            "spatial_epochs": spatial_epochs,
            "arm_c": dict(pcfg.get("arm_c", {})), "arm_d": dict(pcfg.get("arm_d", {})),
            "arm_e": dict(pcfg.get("arm_e", {})), "arm_f": dict(pcfg.get("arm_f", {}))},
        "bin_weights": bin_w.cpu().tolist(), "output_scale": [float(x) for x in scale],
        "bootstrap_draws": draws,
        "row_weighting": {
            "balance_sources": bool(pcfg.get("balance_sources", True)),
            "changed_tile_weight": float(pcfg.get("changed_tile_weight", 3.0)),
            "n_changed_rows": int(changed.sum()),
            "changed_source": "field_changed_flag_raw_6ch"},
        "n_pairs": int(len(pairs)),
        "n_priority_pairs": int(np.sum(kinds < 2)) if len(kinds) else 0,
        "allow_incomplete": bool(args.allow_incomplete), "arm_reports": {},
    }

    # ---- wandb ------------------------------------------------------------
    # One run per fit job (the benchmark launches one job per arm), grouped by
    # run_name so a sweep's arms land together. Every logged metric is tagged
    # arm_<arm>/member_<m>/... so a multi-arm process stays legible in one run,
    # and a shared monotonic step keeps wandb happy across members and arms.
    wcfg = dict(cfg.get("wandb", {}) or {})
    if args.no_wandb:
        wcfg["mode"] = "disabled"
    elif args.wandb_mode:
        wcfg["mode"] = args.wandb_mode
    wcfg.setdefault("group", args.run_name)
    wcfg.setdefault("name", f"{args.run_name}-proxyfit-{''.join(arms)}")
    cfg["wandb"] = wcfg
    use_wandb = maybe_init_wandb(cfg, out_root, job_type="proxy_fit")
    if use_wandb:
        banner(f"wandb: logging to project "
               f"{wcfg.get('project', os.environ.get('WANDB_PROJECT', 'cosmo_sr'))}"
               f" as {wcfg['name']}")
    _gstep = {"n": 0}

    def wlog(d: Dict) -> None:
        if not use_wandb:
            return
        try:
            import wandb
            wandb.log(d, step=_gstep["n"])
            _gstep["n"] += 1
        except Exception:
            pass

    baselines: Dict[str, Dict] = {}
    for arm in arms:
        provider = make_arm_features(arm, rows, table_dir=table.parent, cfg=cfg)
        shape = provider.per_row_shape
        epochs = flat_epochs if arm_storage(arm) == "inline" else spatial_epochs
        max_rows = chunk_rows_for(arm, len(rows))
        chunks = _pack_chunks(common["box"], unit, max_rows)
        banner(f"arm {arm}: per-row shape {tuple(shape)}, {n_members} members, "
               f"{epochs} epochs, {len(chunks)} chunks (<= {max_rows} rows)")
        members, history = [], []
        for m, draw in enumerate(draws):
            mult = np.asarray([draw.get(str(b), 0) for b in common["box"]],
                              dtype=np.float64)
            w_m = weight * mult
            model, hist = _train_member(
                arm=arm, provider=provider, arrays=common, ctx=ctx,
                reward_t=reward_t, targets=targets, unit=unit, weight=w_m,
                mult=mult, changed=changed, pairs=pairs, kinds=kinds, chunks=chunks,
                pcfg=pcfg, scale=scale, bin_w=bin_w, sigma_r=sigma_r,
                w_joint=w_joint, w_occ=w_occ, seed=m, epochs=epochs, device=device,
                lambdas=lambdas, reward_key=reward_key,
                log_prefix=f"arm {arm} member {m}",
                wlog=wlog, wtag=f"arm_{arm}/member_{m}")
            hist["boxes_drawn"] = draw
            members.append(model)
            history.append(hist)

        ens = ProxyEnsemble(members)
        ens.save(out_root / f"proxy_{arm}")
        (out_root / f"proxy_{arm}" / "feature_names.json").write_text(
            json.dumps(feature_names_of(arm, cfg), indent=2))

        # Streamed train metrics -- never materialise the full feature tensor.
        dev_members = [m_.to(device) for m_ in members]
        mean, _std = stream_ensemble_delta(
            provider, dev_members, ctx, reward_t, w_joint=w_joint, w_occ=w_occ,
            key=reward_key, chunk_rows=max_rows, device=device)
        pred_counts = stream_pred_counts(provider, dev_members, ctx,
                                         chunk_rows=max_rows, device=device)
        for m_ in members:
            m_.cpu()
        pooled = pooled_count_error(pred_counts, {k: common[k] for k in COUNT_KEYS})
        pooled_cand = pooled_count_error_by_candidate(
            pred_counts, {k: common[k] for k in COUNT_KEYS},
            common["box"], common["tag"])

        # The linear baseline needs a flat design matrix; for a spatial arm the
        # bar is arm B's flat features -- "is the model better than a line through
        # the flat summaries of the same crop" is the honest question.
        bl_feats = (as_arrays(rows, arm, table_dir=table.parent)["features"]
                    if arm_storage(arm) == "inline"
                    else as_arrays(rows, "b")["features"])
        bl = fit_baselines(bl_feats, {k: common[k] for k in COUNT_KEYS},
                           row_weight=weight)
        baselines[arm] = bl

        report["arm_reports"][arm] = {
            "feature_shape": [int(x) for x in shape],
            "members": history,
            "train_ranking": rank_metrics(mean, targets, common["box"], unit),
            "train_pooled_all_rows_DIAGNOSTIC": {
                "mean_log_error": pooled["mean_log_error"],
                "occupation_log_error_max": pooled["occupation_log_error_max"]},
            "train_pooled_per_candidate": {
                "mean_log_error": pooled_cand["mean_log_error"],
                "occupation_log_error": pooled_cand["occupation_log_error"],
                "occupation_log_error_max": pooled_cand["occupation_log_error_max"]},
        }
        sp = report['arm_reports'][arm]['train_ranking']['within_tile_spearman']
        print(f"  arm {arm}: train within-unit spearman "
              f"{sp:.4f}"
              f", per-candidate pooled mean log error "
              f"{pooled_cand['mean_log_error']:.4f}", flush=True)
        # End-of-arm summary metrics, watchable alongside the per-epoch loss.
        wlog({f"arm_{arm}/train_within_tile_spearman": float(sp),
              f"arm_{arm}/train_pooled_mean_log_error":
                  float(pooled_cand["mean_log_error"]),
              f"arm_{arm}/train_final_loss":
                  float(np.mean([h["final_loss"] for h in history]))})

    # Merge under a flock so parallel per-arm GPU jobs compose cleanly.
    _merge_arm_artifacts(out_root, baselines, report, arms)
    banner(json.dumps({
        "model_selection": "fixed_predeclared",
        "arms": {a: report["arm_reports"][a]["train_ranking"] for a in arms},
    }, indent=2))
    print(f"  ensembles -> {out_root}/proxy_<arm>/", flush=True)
    print("  !! These are TRAINING numbers. The decision is made once, on "
          "set8-11, by scripts/reward/gate_catalog_proxy.py.", flush=True)
    finish_wandb()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
