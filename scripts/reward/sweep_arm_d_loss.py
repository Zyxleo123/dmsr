#!/usr/bin/env python
"""Arm-D loss/hyperparameter sweep under LEAVE-ONE-BOX-OUT validation on set0-7.

Why this is a SEPARATE script from ``train_catalog_proxy.py``
-------------------------------------------------------------
The production trainer is deliberately sweep-proof: ``proxy_cv.enabled`` must be
false and every arm trains once with one predeclared configuration, so the A-F
comparison is an answer about *features* and not about tuning. This script does
not change that and never writes into the production run directory. It is a
DIAGNOSTIC: it answers "why does arm D fit the training boxes far better than
A-C while generalising no better", and its winner is not an arm-comparison
result. If a configuration found here materially helps D, the honest follow-up
is to re-fit ALL SIX arms under it through the production path.

Never reads set8-11
-------------------
Validation is leave-one-box-out *within the fit split* (set0-7). The gate boxes
are not opened here, so repeated selection against this script cannot turn them
into training data. The held-out box's rows carry training weight exactly zero
(the same mechanism the box bootstrap uses), so it is excluded from every loss
term and from the pairs, not merely unscored.

What is measured
----------------
Every ``--val-every`` epochs the partially-trained member is scored on BOTH the
training boxes and the held-out box with :func:`_proxy_data.rank_metrics` -- the
same function the offline gate uses -- so the train/validation gap is a number
at every checkpoint rather than an inference from the final loss. One JSONL row
per (variant, fold, epoch). Selection is on validation within-tile Spearman,
never on training loss.

The variants
------------
The DEPLOYED objective is ``0.1 * count_calibration + 1.0 * reward_change +
1.0 * pairwise_ranking`` on an absolute ``log1p`` count target, and it is
included here as ``D0_deployed`` -- without that control none of the other rows
mean anything. The rest vary the loss along two axes: what the count term is
computed on (absolute ``log1p`` counts vs the normalised frozen-relative count
residual) and how the reward/ranking terms are weighted.

    python scripts/reward/sweep_arm_d_loss.py --stage loss --variant D1_resid_huber
"""
from __future__ import annotations

import argparse
import fcntl
import json
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from _proxy_data import (  # noqa: E402
    COUNT_KEYS, as_arrays, build_row_context, delta_of_summary, load_rows,
    make_arm_features, pair_weights, rank_metrics, robust_scale, row_weights,
    true_delta_rewards, unit_ids_of,
)
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, direct_root,
    load_direct_config, load_reward_models, region_width_of, write_json_atomic,
)
from train_catalog_proxy import (  # noqa: E402
    _assign_pairs, _pack_chunks, chunk_rows_for, make_model,
)

from cosmo_sr.reward.catalog_proxy import (  # noqa: E402
    bin_weights_from_counts, count_loss, make_within_tile_pairs,
    pairwise_ranking_loss, reward_change_loss,
)


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """One objective. ``count_target`` picks WHAT the count term regresses.

    ``absolute``
        Huber on ``log1p(counts)`` -- the deployed calibration term.
    ``residual``
        Huber/MSE on ``(log1p(c_hat) - log1p(c_frozen)) / s_j``, the normalised
        frozen-relative count change, with ``s_j`` a per-bin robust scale of that
        change measured on the TRAINING boxes of the current fold only. This is
        the target that emphasises what the candidate changed and puts the 16
        outputs on one scale; the absolute target does neither.
    """

    name: str
    w_count: float = 0.0
    w_reward: float = 0.0
    w_rank: float = 0.0
    w_margin_rank: float = 0.0
    count_target: str = "absolute"        # absolute | residual
    count_kind: str = "huber"             # huber | mse
    note: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "w_count": self.w_count, "w_reward": self.w_reward,
            "w_rank": self.w_rank, "w_margin_rank": self.w_margin_rank,
            "count_target": self.count_target, "count_kind": self.count_kind,
            "note": self.note,
        }


#: The loss-selection stage. D0 is the deployed objective and is the control.
LOSS_VARIANTS: Dict[str, Variant] = {v.name: v for v in [
    Variant("D0_deployed", w_count=0.1, w_reward=1.0, w_rank=1.0,
            count_target="absolute", count_kind="huber",
            note="the objective currently in configs/reward/sr2_direct_finetune.yaml"),
    Variant("D1_resid_huber", w_count=1.0, count_target="residual", count_kind="huber",
            note="normalised residual-count Huber alone"),
    Variant("D2_resid_mse", w_count=1.0, count_target="residual", count_kind="mse",
            note="same target, squared -- does punishing large errors help"),
    Variant("D3_resid_huber_reward03", w_count=1.0, w_reward=0.3,
            count_target="residual", count_kind="huber",
            note="count Huber primary + modest reward alignment"),
    Variant("D4_reward_only", w_reward=1.0,
            note="are the 16 count targets obstructing learning"),
    Variant("D5_reward_rank01", w_reward=1.0, w_rank=0.1,
            note="light ranking supervision instead of the deployed weight 1.0"),
    Variant("D6_margin_rank", w_count=0.1, w_margin_rank=1.0,
            count_target="residual", count_kind="huber",
            note="gap-proportional margin ranking; only mis-ordered pairs pay"),
]}


# --------------------------------------------------------------------------- #
# The residual-count loss (not in the library: the library is the frozen,
# pre-registered path and this is a diagnostic)
# --------------------------------------------------------------------------- #
def residual_count_loss(
    pred: Mapping[str, torch.Tensor],
    true: Mapping[str, torch.Tensor],
    frozen: Mapping[str, torch.Tensor],
    bin_scale: torch.Tensor,
    *,
    row_weights_t: Optional[torch.Tensor] = None,
    bin_weights_t: Optional[torch.Tensor] = None,
    kind: str = "huber",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Loss on the normalised frozen-relative count residual.

    ``y_j = (log1p(c_j) - log1p(c_frozen_j)) / s_j`` for both the prediction and
    the label. ``s_j`` is a per-bin robust scale of the TRUE residual on this
    fold's training rows, so every one of the 16 outputs contributes on the same
    scale -- which the absolute ``log1p`` target does not, the 1e14 host bin
    being three orders of magnitude smaller than the lowest sub bin.

    The model already carries a frozen-relative residual head, so
    ``log1p(c_hat) - log1p(c_frozen)`` recovers its raw output exactly; this is
    computed through the public ``summary`` API rather than reaching into the
    head, so it works unchanged for every arm.
    """
    p_parts, t_parts = [], []
    for k in COUNT_KEYS:
        f = torch.log1p(frozen[k].clamp_min(0.0).to(torch.float64))
        p_parts.append(torch.log1p(pred[k].clamp_min(0.0).to(torch.float64)) - f)
        t_parts.append(torch.log1p(true[k].clamp_min(0.0).to(torch.float64)) - f)
    p = torch.cat(p_parts, dim=1) / bin_scale.reshape(1, -1)
    t = torch.cat(t_parts, dim=1) / bin_scale.reshape(1, -1)

    if str(kind) == "mse":
        per_elem = (p - t) ** 2
    elif str(kind) == "huber":
        per_elem = F.huber_loss(p, t, delta=float(huber_delta), reduction="none")
    else:
        raise ValueError(f"count_kind must be 'huber' or 'mse', got {kind!r}")

    w = torch.ones(1, per_elem.shape[1], dtype=per_elem.dtype, device=per_elem.device)
    if bin_weights_t is not None:
        w = bin_weights_t.to(per_elem.dtype).to(per_elem.device).reshape(1, -1)
    r = torch.ones(per_elem.shape[0], 1, dtype=per_elem.dtype, device=per_elem.device)
    if row_weights_t is not None:
        r = row_weights_t.to(per_elem.dtype).to(per_elem.device).reshape(-1, 1)
    return (per_elem * w * r).sum() / (w.sum() * r.sum()).clamp_min(1e-12)


def margin_ranking_loss(better: torch.Tensor, worse: torch.Tensor,
                        margin: torch.Tensor,
                        weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``max(0, m - (R_a - R_b))`` with a GAP-PROPORTIONAL margin.

    Unlike RankNet's softplus, a pair already separated by more than its own true
    reward gap contributes exactly zero, so the gradient budget goes to the
    mis-ordered and the insufficiently-separated pairs instead of continuing to
    push apart pairs that are already right.
    """
    per_pair = F.relu(margin - (better - worse))
    if weights is None:
        return per_pair.mean()
    w = weights.to(per_pair.dtype).to(per_pair.device).reshape(-1)
    return (per_pair * w).sum() / w.sum().clamp_min(1e-12)


# --------------------------------------------------------------------------- #
# One LOBO fold
# --------------------------------------------------------------------------- #
def _bin_residual_scale(counts: Dict[str, np.ndarray], frozen: Dict[str, np.ndarray],
                        rows_idx: np.ndarray) -> np.ndarray:
    """Per-bin robust scale ``s_j`` of the true frozen-relative log-count change.

    Measured on this fold's TRAINING rows only -- a normaliser fitted with the
    held-out box in it is a leak, small but free to avoid.
    """
    cols = []
    for k in COUNT_KEYS:
        d = (np.log1p(np.maximum(counts[k][rows_idx], 0.0))
             - np.log1p(np.maximum(frozen[k][rows_idx], 0.0)))
        cols.append(d)
    d = np.concatenate(cols, axis=1)
    out = np.empty(d.shape[1], dtype=np.float64)
    for j in range(d.shape[1]):
        out[j] = robust_scale(d[:, j], floor=1e-3)
    return out


@torch.no_grad()
def _evaluate(model, provider, ctx, reward_t, *, rows_idx: np.ndarray,
              targets: np.ndarray, box: np.ndarray, unit: np.ndarray,
              w_joint: float, w_occ: float, chunk_rows: int, device
              ) -> Dict[str, float]:
    """Within-unit ranking metrics on a row subset, streamed.

    Uses :func:`rank_metrics` -- the SAME function the offline gate scores with --
    so a validation number here and a gate number later mean the same thing.
    """
    model.eval()
    preds = np.full(rows_idx.size, np.nan, dtype=np.float64)
    for s in range(0, rows_idx.size, int(chunk_rows)):
        idx = rows_idx[s:s + int(chunk_rows)]
        xi = provider.get(idx, device=device)
        sub = ctx.index(idx)
        summ = model.summary(xi, sub.tile_volume, sub.frozen)
        dr = delta_of_summary(summ, sub, reward_t, w_joint=w_joint, w_occ=w_occ)
        preds[s:s + idx.size] = dr.detach().cpu().numpy()
    model.train()
    return rank_metrics(preds, targets[rows_idx], box[rows_idx], unit[rows_idx])


def run_fold(
    *, variant: Variant, held_out: str, arm: str, provider, common, ctx, reward_t,
    targets, unit, sigma_r, base_weight, changed, pairs, kinds, pcfg, scale, bin_w,
    w_joint, w_occ, epochs: int, val_every: int, lr: float, seed: int, device,
    writer, run_tag: str, hparams: Optional[Dict] = None,
) -> Dict:
    """Train one member with the held-out box zero-weighted; score both splits."""
    box = common["box"]
    is_val = np.asarray([str(b) == str(held_out) for b in box])
    train_rows = np.nonzero(~is_val)[0].astype(np.int64)
    val_rows = np.nonzero(is_val)[0].astype(np.int64)
    if val_rows.size == 0:
        raise SystemExit(f"held-out box {held_out!r} has no rows")

    # Zero weight on the held-out box: it drops out of every loss term AND out of
    # every pair (pair_weights multiplies by the row multiplicity), which is what
    # makes this a real leave-one-box-out and not just an unscored box.
    mult = (~is_val).astype(np.float64)
    weight = base_weight * mult

    hp = dict(hparams or {})
    pcfg_fold = dict(pcfg)
    for k in ("dropout", "weight_decay", "hidden", "arm_d"):
        if k in hp:
            pcfg_fold[k] = hp[k]

    max_rows = chunk_rows_for(arm, len(box))
    chunks = _pack_chunks(box, unit, max_rows)
    _, pair_sets = _assign_pairs(chunks, pairs, weight.size)
    local = np.full(weight.size, -1, dtype=np.int64)
    for idx in chunks:
        local[idx] = np.arange(idx.size)

    model = make_model(arm, provider, common, pcfg_fold, scale, seed).to(device)
    if hasattr(model, "fit_standardizer"):
        model.fit_standardizer(provider.standardizer_sample(train_rows, seed=int(seed)))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr),
                           weight_decay=float(pcfg_fold.get("weight_decay", 1e-4)))

    y = {k: torch.as_tensor(common[k], dtype=torch.float64, device=device)
         for k in COUNT_KEYS}
    frozen_np = {
        "n_sub": ctx.frozen.n_sub.detach().cpu().numpy(),
        "n_host": ctx.frozen.n_host.detach().cpu().numpy(),
        "occ_numerator": ctx.frozen.occ_numerator.detach().cpu().numpy(),
    }
    bin_scale = torch.as_tensor(
        _bin_residual_scale(common, frozen_np, train_rows),
        dtype=torch.float64, device=device)

    rw = torch.as_tensor(weight, dtype=torch.float64, device=device)
    tgt = torch.as_tensor(targets, dtype=torch.float64, device=device)
    type_w = {0: float(pcfg.get("pair_weight_frozen_vs_intervention", 1.0)),
              1: float(pcfg.get("pair_weight_adjacent_alpha", 1.0)),
              2: float(pcfg.get("pair_weight_random", 1.0))}
    pw = pair_weights(pairs, kinds, mult, changed, type_weights=type_w,
                      changed_weight=float(pcfg.get("changed_pair_weight", 3.0)))
    huber = float(pcfg.get("huber_delta", 1.0))
    total_rw = max(float(weight.sum()), 1e-12)
    total_pw = max(float(pw.sum()), 1e-12) if len(pairs) else 1.0

    best = {"val_within_tile_spearman": float("-inf"), "epoch": -1}
    t0 = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for ci, idx in enumerate(chunks):
            it = torch.as_tensor(idx, device=device)
            xi = provider.get(idx, device=device)
            sub = ctx.index(idx)
            s = model.summary(xi, sub.tile_volume, sub.frozen)
            pred = {"n_sub": s.n_sub, "n_host": s.n_host,
                    "occ_numerator": s.occ_numerator}
            frac = float(weight[idx].sum()) / total_rw
            loss = torch.zeros((), dtype=torch.float64, device=device)

            if variant.w_count:
                tru = {k: y[k][it] for k in COUNT_KEYS}
                if variant.count_target == "residual":
                    frz = {"n_sub": sub.frozen.n_sub, "n_host": sub.frozen.n_host,
                           "occ_numerator": sub.frozen.occ_numerator}
                    c = residual_count_loss(
                        pred, tru, frz, bin_scale, row_weights_t=rw[it],
                        bin_weights_t=bin_w, kind=variant.count_kind,
                        huber_delta=huber)
                else:
                    c = count_loss(pred, tru, weights=bin_w, row_weights=rw[it],
                                   huber_delta=huber)["loss"]
                loss = loss + variant.w_count * frac * c

            need_dr = bool(variant.w_reward or variant.w_rank or variant.w_margin_rank)
            if need_dr:
                dr = delta_of_summary(s, sub, reward_t, w_joint=w_joint, w_occ=w_occ)
                if variant.w_reward:
                    loss = loss + variant.w_reward * frac * reward_change_loss(
                        dr, tgt[it], sigma_r, weights=rw[it], huber_delta=huber)
                ps = pair_sets[ci]
                if ps.size and (variant.w_rank or variant.w_margin_rank):
                    a = torch.as_tensor(local[pairs[ps, 0]], device=device)
                    b = torch.as_tensor(local[pairs[ps, 1]], device=device)
                    pwt = torch.as_tensor(pw[ps], dtype=torch.float64, device=device)
                    pfrac = float(pw[ps].sum()) / total_pw
                    if variant.w_rank:
                        loss = loss + variant.w_rank * pfrac * pairwise_ranking_loss(
                            dr[a], dr[b], pwt)
                    if variant.w_margin_rank:
                        # Margin proportional to the true gap, in sigma_R units, so
                        # it is scale-free and a near-tie asks for almost nothing.
                        gap = (tgt[torch.as_tensor(pairs[ps, 0], device=device)]
                               - tgt[torch.as_tensor(pairs[ps, 1], device=device)])
                        m = (gap.abs() / float(max(sigma_r, 1e-12))).clamp(0.0, 10.0)
                        loss = loss + variant.w_margin_rank * pfrac * margin_ranking_loss(
                            dr[a], dr[b], m, pwt)

            if loss.requires_grad:
                loss.backward()
            running += float(loss.detach())
        opt.step()

        if epoch % int(val_every) == 0 or epoch == int(epochs):
            tr = _evaluate(model, provider, ctx, reward_t, rows_idx=train_rows,
                           targets=targets, box=box, unit=unit, w_joint=w_joint,
                           w_occ=w_occ, chunk_rows=max_rows, device=device)
            va = _evaluate(model, provider, ctx, reward_t, rows_idx=val_rows,
                           targets=targets, box=box, unit=unit, w_joint=w_joint,
                           w_occ=w_occ, chunk_rows=max_rows, device=device)
            row = {
                "run_tag": run_tag, "variant": variant.name, "held_out": held_out,
                "arm": arm, "epoch": int(epoch), "lr": float(lr), "seed": int(seed),
                "train_loss": float(running),
                "train_within_tile_spearman": tr["within_tile_spearman"],
                "train_pairwise_accuracy": tr["pairwise_accuracy"],
                "val_within_tile_spearman": va["within_tile_spearman"],
                "val_pairwise_accuracy": va["pairwise_accuracy"],
                "val_n_groups": va["n_groups"], "train_n_groups": tr["n_groups"],
                "generalization_gap": (tr["within_tile_spearman"]
                                       - va["within_tile_spearman"]),
                "seconds": round(time.time() - t0, 1),
                "hparams": hp, "variant_spec": variant.to_dict(),
            }
            writer(row)
            if np.isfinite(va["within_tile_spearman"]) and (
                    va["within_tile_spearman"] > best["val_within_tile_spearman"]):
                best = {"val_within_tile_spearman": va["within_tile_spearman"],
                        "epoch": int(epoch),
                        "train_within_tile_spearman": tr["within_tile_spearman"]}
            print(f"  {variant.name} hold={held_out} ep {epoch:4d}/{epochs} "
                  f"loss {running:.5f}  train_rho {tr['within_tile_spearman']:.4f}  "
                  f"val_rho {va['within_tile_spearman']:.4f}  "
                  f"gap {row['generalization_gap']:+.4f}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


# --------------------------------------------------------------------------- #
# JSONL writer (append-only, flocked, resumable)
# --------------------------------------------------------------------------- #
def _append_jsonl(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def _completed_folds(path: Path, epochs: int) -> set:
    """``(variant, held_out, lr, seed)`` keys already carried to the final epoch.

    Resumability: hitting the time limit means resubmitting, not restarting.
    """
    done = set()
    if not path.is_file():
        return done
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(d.get("epoch", -1)) == int(epochs):
            done.add((d.get("variant"), d.get("held_out"),
                      float(d.get("lr", 0.0)), int(d.get("seed", 0))))
    return done


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--table", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", default="d")
    ap.add_argument("--stage", choices=["loss", "hparam"], default="loss")
    ap.add_argument("--variant", default="",
                    help="one variant name; empty runs all of the stage's variants")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--lrs", nargs="*", type=float, default=[1e-3])
    ap.add_argument("--folds", nargs="*", default=[],
                    help="held-out boxes; empty = every fit box (full LOBO)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="",
                    help="JSONL path; default <root>/sweeps/arm_<arm>_<stage>.jsonl")
    ap.add_argument("--plan-index", type=int, default=-1,
                    help="run only this entry of the stage's plan (the SLURM array "
                         "index). -1 runs the whole plan in one process. An index "
                         "past the end prints so and exits 0, so an over-sized "
                         "array never fails a task.")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    pcfg = dict(cfg.get("proxy", {}))
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    device = torch.device(args.device if (args.device != "cuda"
                                          or torch.cuda.is_available()) else "cpu")
    reward_t = reward_t.to(device)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    width = region_width_of(cfg)
    arm = str(args.arm)

    table = Path(args.table) if args.table else direct_root("proxy_data") / "rows.jsonl"
    rows_all = load_rows(table, require_complete=not args.allow_incomplete)
    fit_boxes, gate_boxes = boxes_of(cfg, "proxy_fit"), boxes_of(cfg, "proxy_gate")
    rows = [r for r in rows_all if r["box"] in set(fit_boxes)]
    if not rows:
        raise SystemExit(f"no rows from the fit boxes {fit_boxes}")
    banner(f"arm {arm} {args.stage} sweep: {len(rows)} rows from "
           f"{sorted({r['box'] for r in rows})}. "
           f"GATE BOXES {gate_boxes} ARE NOT READ.")

    common = as_arrays(rows, "a", table_dir=table.parent)
    ctx = build_row_context(rows).to(device)
    targets = true_delta_rewards(ctx, reward_t, w_joint=w_joint, w_occ=w_occ)
    unit = unit_ids_of(common["tile_id"], width=width)
    sigma_r = robust_scale(targets)

    rw = row_weights(common["source"], common["field_changed"],
                     balance_sources=bool(pcfg.get("balance_sources", True)),
                     changed_weight=float(pcfg.get("changed_tile_weight", 3.0)))
    base_weight, changed = rw["weight"], rw["changed"]

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
    provider = make_arm_features(arm, rows, table_dir=table.parent, cfg=cfg)

    root = direct_root("sweeps")
    out_path = Path(args.out) if args.out else root / f"arm_{arm}_{args.stage}.jsonl"
    done = _completed_folds(out_path, int(args.epochs))
    if done:
        banner(f"resuming: {len(done)} (variant, fold, lr, seed) combinations already "
               f"complete in {out_path}")

    folds = list(args.folds) if args.folds else list(fit_boxes)
    if args.stage == "loss":
        names = [args.variant] if args.variant else list(LOSS_VARIANTS)
        plan = [(LOSS_VARIANTS[n], lr, {}) for n in names for lr in args.lrs]
    else:
        # Hyperparameter stage: the winning loss is read from the loss stage's
        # decision file, so this cannot silently sweep against a stale winner.
        dec = root / f"arm_{arm}_loss_decision.json"
        if not dec.is_file():
            print(f">>> MISSING INPUT: {dec}")
            print(">>> produced by: scripts/reward/aggregate_arm_d_sweep.py --stage loss")
            print(">>> exiting 0 so dependents report the same rather than stranding.")
            return 0
        winner = json.loads(dec.read_text())["winner"]
        v = LOSS_VARIANTS[winner]
        banner(f"hyperparameter stage on the winning loss {winner!r}")
        grid = []
        for lr in (args.lrs if args.lrs != [1e-3] else [1e-3, 3e-4]):
            for wd in (1e-4, 1e-3):
                for do in (0.0, 0.1):
                    grid.append((v, lr, {"weight_decay": wd, "dropout": do}))
        plan = grid

    # One array task takes one plan entry. Printed either way so a log says which
    # slice it ran, and an index past the end exits 0 rather than failing.
    if int(args.plan_index) >= 0:
        if int(args.plan_index) >= len(plan):
            print(f">>> plan index {args.plan_index} is past the end of a "
                  f"{len(plan)}-entry plan; nothing to do. Exiting 0.")
            return 0
        plan = [plan[int(args.plan_index)]]
    banner(f"plan: {len(plan)} configuration(s) x {len(folds)} LOBO folds "
           f"({', '.join(v.name for v, _, _ in plan)})")

    run_tag = f"{arm}_{args.stage}"
    summary: List[Dict] = []
    for variant, lr, hp in plan:
        for held in folds:
            key = (variant.name, str(held), float(lr), int(args.seed))
            if key in done:
                print(f"  skip {key} (already complete)", flush=True)
                continue
            banner(f"{variant.name} | hold-out {held} | lr {lr} | {hp or 'base hparams'}")
            best = run_fold(
                variant=variant, held_out=str(held), arm=arm, provider=provider,
                common=common, ctx=ctx, reward_t=reward_t, targets=targets, unit=unit,
                sigma_r=sigma_r, base_weight=base_weight, changed=changed, pairs=pairs,
                kinds=kinds, pcfg=pcfg, scale=scale, bin_w=bin_w, w_joint=w_joint,
                w_occ=w_occ, epochs=int(args.epochs), val_every=int(args.val_every),
                lr=float(lr), seed=int(args.seed), device=device,
                writer=lambda r: _append_jsonl(out_path, r), run_tag=run_tag,
                hparams=hp)
            summary.append({"variant": variant.name, "held_out": held, "lr": lr,
                            "hparams": hp, **best})

    # Per-task filename: parallel array tasks must not clobber each other's record.
    # The JSONL is the source of truth; these are per-task receipts.
    tag = ("all" if int(args.plan_index) < 0 else f"task{int(args.plan_index):03d}")
    write_json_atomic(root / f"arm_{arm}_{args.stage}_run_{tag}.json", {
        "stage": args.stage, "arm": arm, "epochs": int(args.epochs),
        "val_every": int(args.val_every), "folds": folds,
        "fit_boxes": fit_boxes, "gate_boxes_untouched": gate_boxes,
        "sigma_R": float(sigma_r), "jsonl": str(out_path), "summary": summary,
    })
    print(f"\n  rows -> {out_path}")
    print("  !! LOBO validation numbers. set8-11 have not been read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
