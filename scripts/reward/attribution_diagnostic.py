#!/usr/bin/env python
"""Are the per-tile labels repeatable enough to rank on? Fractional vs majority.

The question this answers
-------------------------
A proxy is asked to rank versions of the *same tile*. That is only learnable if
the label of a tile is a repeatable function of that tile -- and on set0 it was
not: between two frozen SR2 seeds of the same box, ``sum_t |dS_t|`` was 4033
against ``|sum_t dS_t|`` of 317, so 92% of the per-tile churn cancelled on
pooling. The whole-box catalog is stable; its decomposition into tiles is not.

So the decomposition is the suspect, and this script puts a number on it. It
re-derives the per-tile summaries from the **already-saved** ``tile_weights.npz``
and catalogs -- no Rockstar run -- under two attribution schemes:

``fractional``
    Each object's weight split over the tiles its member particles came from.
    Exact, and what the pipeline currently uses.
``majority``
    Each object assigned wholly to its single busiest tile
    (:func:`cosmo_sr.reward.tiles.majority_weights`). Also exact in the sense
    that matters -- the tile sums still reproduce the whole-box catalog -- but
    it only moves when an object's majority tile changes.

The decisive number: a repeatability ceiling
--------------------------------------------
Split the frozen seeds into two disjoint halves. Build the ranking target
``dR`` twice, once against each half's averaged baseline. If the labels were
noiseless the two would agree perfectly. Their Spearman correlation is therefore
an **upper bound on the within-tile Spearman any model can reach**, whatever its
features. Comparing that ceiling to the gate's 0.5 is the whole decision:

* ceiling comfortably above the threshold -> the labels can be ranked, proceed;
* ceiling at or below it -> no proxy can pass, and the A-vs-B comparison would
  be two models fitting the same noise. Predict pooled or whole-box reward
  changes instead, which the same split measures for comparison.

Everything is reported on **touched tiles** -- those whose field actually
changed -- as well as on all tiles. An intervention moves ~30 of 512 tiles, so
an all-tile average is dominated by rows where the input is bit-identical and
only the label moved, which is exactly the noise being diagnosed.

    python scripts/reward/attribution_diagnostic.py --boxes set0 set1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from _proxy_matrix import candidate_tag
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, bins_of, dataset_of, direct_root,
    load_direct_config, load_reward_models, run_dir, tile_grid_of,
    write_json_atomic,
)

from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward.catalog_proxy import spearman  # noqa: E402
from cosmo_sr.reward.tiles import (  # noqa: E402
    MemberWeights, majority_weights, tile_summaries,
)
from cosmo_sr.reward.torch_reward import TorchSummary  # noqa: E402

SCHEMES = ("fractional", "majority")
COUNTS = ("n_sub", "n_host", "occ_numerator")


def candidate_dir(box: str, tag: str) -> Path:
    return direct_root("candidates", f"{box}__{tag}")


def load_summaries(cfg, box: str, tag: str,
                   schemes: Sequence[str]) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """``{scheme: {count_key: (n_tiles, n_bins)}}`` re-derived from saved weights."""
    d = candidate_dir(box, tag)
    rep, wnpz = d / "label_report.json", d / "tile_weights.npz"
    if not (rep.is_file() and wnpz.is_file()):
        return None
    doc = json.loads(rep.read_text())
    if not doc.get("label_ok", False):
        return None
    cat_path = Path(doc.get("catalog_path", ""))
    if not cat_path.is_file():
        print(f"    !! {box}/{tag}: catalog gone ({cat_path}); skipped", flush=True)
        return None

    cat = load_rockstar_ascii(cat_path)
    base = MemberWeights.from_npz(wnpz)
    grid, bins = tile_grid_of(cfg), bins_of(cfg["_reward"])
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for scheme in schemes:
        w = base if scheme == "fractional" else majority_weights(base)
        s = tile_summaries(cat, w, bins, grid, box=box, source=tag)
        out[scheme] = {k: np.asarray([getattr(s[t], k) for t in sorted(s)],
                                     dtype=np.float64) for k in COUNTS}
        out[scheme]["volume_mpc3"] = np.asarray(
            [s[t].volume_mpc3 for t in sorted(s)], dtype=np.float64)
    return out


def touched_mask(cfg, box: str, tag: str, threshold: float) -> np.ndarray:
    """Tiles whose FIELD changed, from the generation-time feature difference.

    Read from features.npz rather than recomputed: it is the same quantity the
    trainer would use to decide a row is worth a ranking pair, so the diagnostic
    and the training filter cannot disagree.
    """
    f = candidate_dir(box, tag) / "features.npz"
    if not f.is_file():
        return np.zeros(tile_grid_of(cfg).n_tiles, dtype=bool)
    z = np.load(f)
    a = z["features_a"]
    diff = a[:, a.shape[1] // 2:]
    return np.linalg.norm(diff, axis=1) > float(threshold)


def _summary(counts: Dict[str, np.ndarray], idx: Optional[np.ndarray] = None,
             pooled: bool = False) -> TorchSummary:
    def col(k):
        v = counts[k] if idx is None else counts[k][idx]
        return torch.as_tensor(v.sum(axis=0, keepdims=True) if pooled else v)
    vol = counts["volume_mpc3"] if idx is None else counts["volume_mpc3"][idx]
    return TorchSummary(
        n_sub=col("n_sub"), n_host=col("n_host"), occ_numerator=col("occ_numerator"),
        volume_mpc3=torch.as_tensor([vol.sum()] if pooled else vol))


def tile_dr(reward_t, box_counts: Dict[str, np.ndarray],
            baseline: Dict[str, np.ndarray], cand: Dict[str, np.ndarray], *,
            w_joint: float, w_occ: float) -> np.ndarray:
    """Per-tile ``dR`` of swapping this candidate's tile into the baseline box."""
    n = cand["n_sub"].shape[0]
    box = _summary(box_counts, pooled=True)
    rep = TorchSummary(
        n_sub=box.n_sub.repeat(n, 1), n_host=box.n_host.repeat(n, 1),
        occ_numerator=box.occ_numerator.repeat(n, 1),
        volume_mpc3=box.volume_mpc3.repeat(n))
    with torch.no_grad():
        d = reward_t.delta_reward_swap(rep, _summary(baseline), _summary(cand),
                                       w_joint=w_joint, w_occ=w_occ)
    return d["dR_combined"].numpy()


def whole_box_dr(reward_t, baseline: Dict[str, np.ndarray],
                 cand: Dict[str, np.ndarray], *, w_joint: float,
                 w_occ: float) -> float:
    """The pooled alternative: one number per candidate, no decomposition."""
    with torch.no_grad():
        b = _summary(baseline, pooled=True)
        c = _summary(cand, pooled=True)
        return float(reward_t.combined(c, w_joint=w_joint, w_occ=w_occ)[0]
                     - reward_t.combined(b, w_joint=w_joint, w_occ=w_occ)[0])


def mean_counts(items: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.mean([it[k] for it in items], axis=0)
            for k in list(COUNTS) + ["volume_mpc3"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--boxes", nargs="*", default=["set0", "set1"])
    ap.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    ap.add_argument("--touched-threshold", type=float, default=0.01)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    seeds = [int(s) for s in dataset_of(cfg).get("frozen_seeds", [0, 1, 2, 3])]
    alphas = [float(a) for a in dataset_of(cfg).get("alphas", [0.25, 0.5, 1.0])]
    modes = ["both"] + [m for m in dataset_of(cfg).get("diagnostic_modes", [])]
    if len(seeds) < 4:
        raise SystemExit(
            f"the repeatability ceiling needs two DISJOINT baseline halves, so at "
            f"least four frozen seeds; the config has {seeds}")
    half_a, half_b = seeds[: len(seeds) // 2], seeds[len(seeds) // 2:]

    report: Dict = {"boxes": [], "schemes": list(args.schemes),
                    "frozen_seeds": seeds, "baseline_halves": [half_a, half_b],
                    "touched_threshold": float(args.touched_threshold),
                    "per_box": {}, "verdict": {}}
    ceilings: Dict[str, List[float]] = {s: [] for s in args.schemes}
    ceilings_all: Dict[str, List[float]] = {s: [] for s in args.schemes}
    snr: Dict[str, List[float]] = {s: [] for s in args.schemes}
    box_ceiling: Dict[str, List[float]] = {s: [] for s in args.schemes}

    for box in args.boxes:
        banner(f"{box}")
        frozen = {}
        for sd in seeds:
            got = load_summaries(cfg, box, candidate_tag("frozen", seed=sd),
                                 args.schemes)
            if got is not None:
                frozen[sd] = got
        missing = [s for s in seeds if s not in frozen]
        if missing:
            print(f"  !! frozen seeds {missing} not labelled ok; {box} skipped",
                  flush=True)
            continue

        cands: Dict[Tuple[str, float, str], Dict] = {}
        for m in modes:
            for a in alphas:
                tag = candidate_tag("intervention", alpha=a, mode=m)
                got = load_summaries(cfg, box, tag, args.schemes)
                if got is not None:
                    cands[(tag, a, m)] = got
        if not cands:
            print(f"  !! no labelled interventions in {box}; skipped", flush=True)
            continue
        print(f"  {len(frozen)} frozen seeds, {len(cands)} interventions", flush=True)

        per_box: Dict = {}
        for scheme in args.schemes:
            base_a = mean_counts([frozen[s][scheme] for s in half_a])
            base_b = mean_counts([frozen[s][scheme] for s in half_b])
            base_all = mean_counts([frozen[s][scheme] for s in seeds])

            # --- the noise floor, in label units --------------------------
            pairs = [(i, j) for k, i in enumerate(seeds) for j in seeds[k + 1:]]
            null = float(np.mean([
                np.abs(frozen[i][scheme]["occ_numerator"]
                       - frozen[j][scheme]["occ_numerator"]).sum(axis=1).mean()
                for i, j in pairs]))
            churn = float(np.mean([
                np.abs(frozen[i][scheme]["occ_numerator"]
                       - frozen[j][scheme]["occ_numerator"]).sum()
                for i, j in pairs]))
            pooled_churn = float(np.mean([
                np.abs((frozen[i][scheme]["occ_numerator"]
                        - frozen[j][scheme]["occ_numerator"]).sum(axis=0)).sum()
                for i, j in pairs]))

            rows_a, rows_b, touched, sig = [], [], [], []
            wb_a, wb_b = [], []
            for (tag, a, m), c in cands.items():
                rows_a.append(tile_dr(reward_t, base_all, base_a, c[scheme],
                                      w_joint=w_joint, w_occ=w_occ))
                rows_b.append(tile_dr(reward_t, base_all, base_b, c[scheme],
                                      w_joint=w_joint, w_occ=w_occ))
                touched.append(touched_mask(cfg, box, tag, args.touched_threshold))
                sig.append(np.abs(c[scheme]["occ_numerator"]
                                  - base_all["occ_numerator"]).sum(axis=1))
                wb_a.append(whole_box_dr(reward_t, base_a, c[scheme],
                                         w_joint=w_joint, w_occ=w_occ))
                wb_b.append(whole_box_dr(reward_t, base_b, c[scheme],
                                         w_joint=w_joint, w_occ=w_occ))

            A, B = np.concatenate(rows_a), np.concatenate(rows_b)
            T = np.concatenate(touched)
            S = np.concatenate(sig)
            ceil_all = spearman(A, B)
            ceil_touched = spearman(A[T], B[T]) if T.sum() > 2 else float("nan")
            ratio = float(S[T].mean() / null) if (T.any() and null > 0) else float("nan")
            wb_ceiling = spearman(np.asarray(wb_a), np.asarray(wb_b))

            per_box[scheme] = {
                "null_mean_abs_dS_per_tile": null,
                "seed_churn_sum_abs": churn,
                "seed_churn_pooled": pooled_churn,
                "fraction_cancelling_on_pooling": (
                    float(1.0 - pooled_churn / churn) if churn > 0 else float("nan")),
                "n_touched_tiles_mean": float(T.reshape(len(cands), -1).sum(1).mean()),
                "signal_mean_abs_dS_touched": float(S[T].mean()) if T.any() else float("nan"),
                "signal_to_noise_touched": ratio,
                "repeatability_ceiling_all_tiles": ceil_all,
                "repeatability_ceiling_touched": ceil_touched,
                "whole_box_repeatability_ceiling": wb_ceiling,
                "n_rows": int(A.size), "n_touched": int(T.sum()),
            }
            ceilings[scheme].append(ceil_touched)
            ceilings_all[scheme].append(ceil_all)
            snr[scheme].append(ratio)
            box_ceiling[scheme].append(wb_ceiling)
            print(f"    {scheme:<11s} ceiling(touched)={ceil_touched:6.3f}  "
                  f"ceiling(all)={ceil_all:6.3f}  SNR={ratio:5.2f}  "
                  f"pooled-cancel={100 * per_box[scheme]['fraction_cancelling_on_pooling']:.0f}%"
                  f"  whole-box ceiling={wb_ceiling:6.3f}", flush=True)

        report["per_box"][box] = per_box
        report["boxes"].append(box)

    if not report["boxes"]:
        print(">>> no box had a complete set of frozen seeds plus interventions.")
        print(">>> produced by: scripts/slurm/submit_proxy_labels.sh all BOXES=...")
        return 0

    threshold = float(cfg.get("proxy_gate", {}).get("min_within_tile_spearman", 0.5))
    margin = float(cfg.get("proxy_dataset", {}).get("min_ceiling_margin", 1.3))
    for scheme in args.schemes:
        c = float(np.nanmean(ceilings[scheme]))
        report["verdict"][scheme] = {
            "mean_repeatability_ceiling_touched": c,
            "mean_repeatability_ceiling_all": float(np.nanmean(ceilings_all[scheme])),
            "mean_signal_to_noise_touched": float(np.nanmean(snr[scheme])),
            "mean_whole_box_ceiling": float(np.nanmean(box_ceiling[scheme])),
            "gate_threshold": threshold,
            # A ceiling merely AT the threshold means a perfect model just barely
            # passes and any real model fails, so the bar is the threshold with
            # headroom -- and the arm comparison needs room to separate the arms,
            # not just room to clear a line.
            "rankable": bool(np.isfinite(c) and c >= threshold * margin),
        }
    best = max(args.schemes,
               key=lambda s: (report["verdict"][s]["mean_repeatability_ceiling_touched"]
                              if np.isfinite(report["verdict"][s]
                                             ["mean_repeatability_ceiling_touched"])
                              else -np.inf))
    any_ok = any(report["verdict"][s]["rankable"] for s in args.schemes)
    report["recommendation"] = {
        "best_scheme": best,
        "per_tile_ranking_viable": bool(any_ok),
        "action": ("proceed_per_tile" if any_ok else "switch_to_pooled_target"),
        "note": (
            f"{best} attribution clears {threshold} x {margin} on touched tiles; "
            "fit the proxy per tile with this scheme."
            if any_ok else
            "No attribution scheme leaves headroom above the gate threshold, so "
            "no proxy can pass and the A-vs-B comparison would be two models "
            "fitting the same label noise. Predict pooled / whole-box reward "
            "changes instead; the whole_box_repeatability_ceiling above says "
            "whether that target is stable enough to be worth fitting."),
    }

    out = Path(args.out) if args.out else run_dir(
        args.run_name, create=True) / "attribution_diagnostic.json"
    write_json_atomic(out, report)
    banner(json.dumps({"verdict": report["verdict"],
                       "recommendation": report["recommendation"]}, indent=2))
    print(f"  report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
