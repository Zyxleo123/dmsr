#!/usr/bin/env python
"""Fit the catalog-proxy ensemble on set0-7, validated on set8-11.

Two losses, doing different jobs (see
:mod:`cosmo_sr.reward.catalog_proxy`): a Huber on ``log1p`` counts with capped
per-bin weights, and a pairwise ranking loss on the **true baseline-relative
catalog reward** with pairs formed only within the same ``(box, tile_id)``.

The ranking pairs are the part that matters and the part that is easy to get
subtly wrong. A pair across two different tiles asks "which environment is
richer", which is true, learnable, and something the actor cannot change. A pair
within one tile asks "which version of this tile is better", which is exactly
the decision the actor's gradient encodes.

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

from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, boxes_of, direct_root,
    load_direct_config, load_reward_models, read_jsonl, run_dir, write_json,
)

from cosmo_sr.reward.catalog_proxy import (  # noqa: E402
    CatalogProxy, ProxyConfig, ProxyEnsemble, bin_weights_from_counts, count_loss,
    make_within_tile_pairs, pairwise_ranking_loss, spearman, split_indices_by_box,
)
from cosmo_sr.reward.torch_reward import TorchSummary  # noqa: E402


def load_rows(path: Path) -> List[Dict]:
    if not path.is_file():
        raise SystemExit(
            f"no proxy table at {path}; run "
            "scripts/reward/collect_catalog_proxy_data.py --stage index first")
    return read_jsonl(path)


def as_arrays(rows: List[Dict]) -> Dict[str, np.ndarray]:
    return {
        "features": np.asarray([r["features"] for r in rows], dtype=np.float64),
        "n_sub": np.asarray([r["n_sub"] for r in rows], dtype=np.float64),
        "n_host": np.asarray([r["n_host"] for r in rows], dtype=np.float64),
        "occ_numerator": np.asarray([r["occ_numerator"] for r in rows], dtype=np.float64),
        "volume": np.asarray([r["volume_mpc3"] for r in rows], dtype=np.float64),
        "box": np.asarray([r["box"] for r in rows]),
        "tile_id": np.asarray([int(r["tile_id"]) for r in rows], dtype=np.int64),
        "source": np.asarray([r["source"] for r in rows]),
    }


def frozen_reference(rows: List[Dict]) -> Dict[Tuple[str, int], Dict]:
    """``(box, tile) -> the frozen candidate's own row``: the ``s_{0,j}`` labels."""
    out: Dict[Tuple[str, int], Dict] = {}
    for r in rows:
        if r["source"] == "frozen":
            out[(r["box"], int(r["tile_id"]))] = r
    return out


def box_summaries(rows: List[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
    """``S_0``: the pooled frozen box summary, summed over that box's tiles."""
    acc: Dict[str, Dict[str, np.ndarray]] = {}
    for r in rows:
        if r["source"] != "frozen":
            continue
        a = acc.setdefault(r["box"], {"n_sub": 0.0, "n_host": 0.0,
                                      "occ_numerator": 0.0, "volume_mpc3": 0.0})
        for k in ("n_sub", "n_host", "occ_numerator"):
            a[k] = a[k] + np.asarray(r[k], dtype=np.float64)
        a["volume_mpc3"] = a["volume_mpc3"] + float(r["volume_mpc3"])
    return acc


def true_delta_rewards(rows: List[Dict], reward_t, *, w_joint: float,
                       w_occ: float, key: str = "dR_combined") -> np.ndarray:
    """The ranking target: the **real** ``dR`` this candidate's tile would give.

    Computed from the measured tile summary, not from a prediction, so a proxy
    that ranks these correctly has learned the thing the actor is scored on.
    Rows without a frozen counterpart get NaN and are dropped from pairing.
    """
    frozen = frozen_reference(rows)
    boxes = box_summaries(rows)
    out = np.full(len(rows), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        f = frozen.get((r["box"], int(r["tile_id"])))
        b = boxes.get(r["box"])
        if f is None or b is None:
            continue

        def ts(d, vol):
            def col(k):
                return torch.as_tensor(
                    np.asarray(d[k], dtype=np.float64).reshape(1, -1))
            return TorchSummary(
                n_sub=col("n_sub"), n_host=col("n_host"),
                occ_numerator=col("occ_numerator"),
                volume_mpc3=torch.tensor([float(vol)], dtype=torch.float64))

        box_s = ts(b, b["volume_mpc3"])
        d = reward_t.delta_reward_swap(
            box_s, ts(f, f["volume_mpc3"]), ts(r, r["volume_mpc3"]),
            w_joint=w_joint, w_occ=w_occ)
        out[i] = float(d[key][0])
    return out


def evaluate(model: ProxyEnsemble, arrays, rows, reward_t, targets, *,
             w_joint: float, w_occ: float) -> Dict[str, float]:
    """Within-``(box, tile)`` ranking quality of the ensemble mean."""
    feats = torch.as_tensor(arrays["features"], dtype=torch.float64)
    vol = torch.as_tensor(arrays["volume"], dtype=torch.float64)
    with torch.no_grad():
        preds = torch.stack([
            _predicted_delta(m, feats, vol, rows, reward_t,
                             w_joint=w_joint, w_occ=w_occ)
            for m in model.members
        ]).mean(dim=0).numpy()

    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, (b, t) in enumerate(zip(arrays["box"], arrays["tile_id"])):
        groups.setdefault((str(b), int(t)), []).append(i)

    rhos, accs = [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        p, t = preds[idx], targets[idx]
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() < 2:
            continue
        rhos.append(spearman(p[ok], t[ok]))
        pi, ti = p[ok], t[ok]
        agree = [(pi[a] > pi[b]) == (ti[a] > ti[b])
                 for a in range(len(pi)) for b in range(a + 1, len(pi))
                 if ti[a] != ti[b]]
        if agree:
            accs.append(float(np.mean(agree)))
    return {
        "within_tile_spearman": float(np.nanmean(rhos)) if rhos else float("nan"),
        "pairwise_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "n_groups": len(rhos),
        "n_rows": int(len(targets)),
    }


def _predicted_delta(member: CatalogProxy, feats, vol, rows, reward_t, *,
                     w_joint: float, w_occ: float) -> torch.Tensor:
    frozen = frozen_reference(rows)
    boxes = box_summaries(rows)
    pred = member.summary(feats, vol)
    n = feats.shape[0]
    fz = {"n_sub": [], "n_host": [], "occ_numerator": [], "vol": []}
    bx = {"n_sub": [], "n_host": [], "occ_numerator": [], "vol": []}
    keep = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows):
        f = frozen.get((r["box"], int(r["tile_id"])))
        b = boxes.get(r["box"])
        if f is None or b is None:
            f = {"n_sub": np.zeros_like(r["n_sub"]),
                 "n_host": np.zeros_like(r["n_host"]),
                 "occ_numerator": np.zeros_like(r["occ_numerator"]),
                 "volume_mpc3": r["volume_mpc3"]}
            b = {"n_sub": np.ones_like(r["n_sub"]),
                 "n_host": np.ones_like(r["n_host"]),
                 "occ_numerator": np.ones_like(r["occ_numerator"]),
                 "volume_mpc3": r["volume_mpc3"]}
        else:
            keep[i] = True
        for k in ("n_sub", "n_host", "occ_numerator"):
            fz[k].append(np.asarray(f[k], dtype=np.float64))
            bx[k].append(np.asarray(b[k], dtype=np.float64))
        fz["vol"].append(float(f["volume_mpc3"]))
        bx["vol"].append(float(b["volume_mpc3"]))

    def stack(d):
        return TorchSummary(
            n_sub=torch.tensor(np.asarray(d["n_sub"]), dtype=torch.float64),
            n_host=torch.tensor(np.asarray(d["n_host"]), dtype=torch.float64),
            occ_numerator=torch.tensor(np.asarray(d["occ_numerator"]),
                                       dtype=torch.float64),
            volume_mpc3=torch.tensor(np.asarray(d["vol"]), dtype=torch.float64))

    out = reward_t.delta_reward_swap(stack(bx), stack(fz), pred,
                                     w_joint=w_joint, w_occ=w_occ)["dR_combined"]
    return torch.where(torch.as_tensor(keep), out, torch.full_like(out, float("nan")))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--table", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    pcfg = dict(cfg.get("proxy", {}))
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    device = torch.device(args.device)

    table = Path(args.table) if args.table else direct_root("proxy_data") / "rows.jsonl"
    rows = load_rows(table)
    if not rows:
        raise SystemExit(f"{table} is empty")

    fit_boxes, gate_boxes = boxes_of(cfg, "proxy_fit"), boxes_of(cfg, "proxy_gate")
    arrays = as_arrays(rows)
    split = split_indices_by_box(arrays["box"], fit_boxes, gate_boxes)
    if split["train"].size == 0:
        raise SystemExit(f"no rows from the fit boxes {fit_boxes}")
    banner(f"{len(rows)} rows: {split['train'].size} fit, {split['val'].size} gate; "
           f"dropped boxes {list(split['dropped_boxes'])}")

    targets = true_delta_rewards(rows, reward_t,
                                 w_joint=float(acfg.w_joint_reward),
                                 w_occ=float(acfg.w_occ_reward))

    tr = split["train"]
    x_tr = torch.as_tensor(arrays["features"][tr], dtype=torch.float64, device=device)
    vol_tr = torch.as_tensor(arrays["volume"][tr], dtype=torch.float64, device=device)
    y_tr = {k: torch.as_tensor(arrays[k][tr], dtype=torch.float64, device=device)
            for k in ("n_sub", "n_host", "occ_numerator")}
    counts = np.concatenate(
        [arrays[k][tr] for k in ("n_sub", "n_host", "occ_numerator")], axis=1)
    weights = torch.as_tensor(
        bin_weights_from_counts(counts, cap=float(pcfg.get("bin_weight_cap", 8.0)),
                                floor=float(pcfg.get("bin_weight_floor", 0.25))),
        dtype=torch.float64, device=device)
    # Start the network at the right magnitude instead of learning the units.
    scale = np.maximum(counts.mean(axis=0), 1e-3)

    pairs = make_within_tile_pairs(
        arrays["box"][tr], arrays["tile_id"][tr], targets[tr],
        max_pairs_per_group=int(pcfg.get("max_pairs_per_group", 32)),
        min_margin=float(pcfg.get("min_pair_margin", 0.0)))
    banner(f"{len(pairs)} within-(box,tile) ranking pairs")
    if len(pairs) == 0:
        print(">>> WARNING: no ranking pairs. Every (box, tile) has one candidate, "
              ">>> so the proxy can only learn the count regression and nothing "
              ">>> about local improvement. Generate frozen_seed / intervention "
              ">>> candidates before trusting this fit.", flush=True)
    rows_tr = [rows[i] for i in tr]
    tgt_tr = torch.as_tensor(targets[tr], dtype=torch.float64, device=device)

    n_members = int(pcfg.get("n_members", 5))
    members: List[CatalogProxy] = []
    history: List[Dict] = []
    for m in range(n_members):
        cfg_m = ProxyConfig(
            n_features=int(arrays["features"].shape[1]),
            n_sub_bins=int(arrays["n_sub"].shape[1]),
            n_host_bins=int(arrays["n_host"].shape[1]),
            hidden=tuple(int(h) for h in pcfg.get("hidden", (128, 128))),
            dropout=float(pcfg.get("dropout", 0.0)),
            output_scale=tuple(float(x) for x in scale))
        model = CatalogProxy(cfg_m, seed=m).fit_standardizer(
            arrays["features"][tr]).to(device)
        opt = torch.optim.Adam(model.parameters(),
                               lr=float(pcfg.get("lr", 1e-3)),
                               weight_decay=float(pcfg.get("weight_decay", 1e-4)))
        t0 = time.time()
        for epoch in range(int(pcfg.get("epochs", 400))):
            model.train()
            pred = model(x_tr)
            losses = count_loss(pred, y_tr, weights=weights,
                                huber_delta=float(pcfg.get("huber_delta", 1.0)))
            total = losses["loss"]
            if len(pairs):
                dr = _predicted_delta(
                    model, x_tr, vol_tr, rows_tr, reward_t,
                    w_joint=float(acfg.w_joint_reward),
                    w_occ=float(acfg.w_occ_reward))
                a = torch.as_tensor(pairs[:, 0], device=device)
                b = torch.as_tensor(pairs[:, 1], device=device)
                rank = pairwise_ranking_loss(dr[a], dr[b])
                total = total + float(pcfg.get("ranking_weight", 1.0)) * rank
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            if epoch % max(1, int(pcfg.get("epochs", 400)) // 10) == 0:
                print(f"  member {m} epoch {epoch:4d} loss {float(total.detach()):.5f}",
                      flush=True)
        members.append(model.cpu())
        history.append({"member": m, "final_loss": float(total.detach()),
                        "seconds": round(time.time() - t0, 1)})

    ens = ProxyEnsemble(members)
    out = run_dir(args.run_name, "proxy", create=True)
    ens.save(out)

    report = {
        "table": str(table), "n_rows": len(rows),
        "fit_boxes": fit_boxes, "gate_boxes": gate_boxes,
        "n_train_rows": int(tr.size), "n_val_rows": int(split["val"].size),
        "n_ranking_pairs": int(len(pairs)),
        "bin_weights": weights.cpu().tolist(),
        "output_scale": [float(x) for x in scale],
        "members": history,
        "train": evaluate(ens, {k: v[tr] for k, v in arrays.items()},
                          rows_tr, reward_t, targets[tr],
                          w_joint=float(acfg.w_joint_reward),
                          w_occ=float(acfg.w_occ_reward)),
    }
    va = split["val"]
    if va.size:
        report["val"] = evaluate(
            ens, {k: v[va] for k, v in arrays.items()}, [rows[i] for i in va],
            reward_t, targets[va], w_joint=float(acfg.w_joint_reward),
            w_occ=float(acfg.w_occ_reward))
    write_json(out / "train_report.json", report)
    banner(json.dumps({k: v for k, v in report.items()
                       if k in ("train", "val", "n_ranking_pairs")}, indent=2))
    print(f"  ensemble -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
