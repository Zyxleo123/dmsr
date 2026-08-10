#!/usr/bin/env python
"""At which spatial scale is the catalog-reward label rankable? Tiles -> box.

What replaced the per-tile "repeatability ceiling"
--------------------------------------------------
The first diagnostic (:mod:`attribution_diagnostic`) asked one question -- are
per-*tile* labels repeatable? -- got "barely, and 92% cancels on pooling", and
concluded the local-credit route was unlearnable. That conclusion overreached in
two ways this script corrects:

* it measured a single spatial scale (one tile) when the real question is *which*
  scale is the smallest reliable one -- tiles, ``2^3`` tiles, ``4^3`` tiles, or
  the whole box;
* it called the frozen-seed comparison a "repeatability ceiling", as if the four
  frozen seeds were re-measurements of one field. They are not: different SR2
  seeds produce different fields and different proxy inputs, so the right name is
  **baseline-context stability** -- how much a candidate's *ranking within a
  region* depends on which frozen box it is scored against.

The counterfactual, computed consistently
------------------------------------------
For a frozen seed ``r`` with complete box summary ``C_r`` and region-``g``
contribution ``c_{r,g}``, and a candidate whose region-``g`` contribution is
``c_{c,g}``::

    dR_g^{(r)} = R(C_r - c_{r,g} + c_{c,g}) - R(C_r).

``C_r`` and ``c_{r,g}`` always come from the **same** baseline ``r``; a pooled
box from one seed is never combined with a removed region from another. Only the
inserted region ``c_{c,g}`` comes from the candidate. This is exactly the swap
form the actor and the proxy gate use, lifted from a tile to a region.

What is reported, and what is deliberately not
----------------------------------------------
Per ``(box, attribution scheme, region width, partition offset)``: the number of
informative regions and non-tied candidate pairs; the within-region tie-aware
Spearman and pairwise agreement of the candidate ordering across frozen baseline
contexts; the regional signal magnitude and how much per-tile churn cancels once
pooled to the region; and the sensitivity of the ordering to the baseline
context. Rankings are formed **within** ``(box, region, offset)`` and averaged
per box -- never concatenated into one global correlation, which would reward
knowing which regions are host-rich. Bootstrap is by box only.

With only ``set0`` and ``set1`` this is **exploratory**: the report stamps
``evidence_status = exploratory_insufficient_boxes`` and proposes which width(s)
to carry into the held-out pilot screen. It does *not* emit an "unlearnable"
verdict -- two development boxes cannot support one, and the trained held-out
proxy gate remains the decision.

    python scripts/reward/region_attribution_diagnostic.py --boxes set0 set1
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Reuse the loaders from the tile diagnostic verbatim, so the two scripts read
# the saved labels the same way and cannot drift on what a tile summary is.
from attribution_diagnostic import (  # noqa: E402
    COUNTS, SCHEMES, candidate_dir, load_summaries, touched_mask,
)
from _proxy_matrix import candidate_tag  # noqa: E402
from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, dataset_of, load_direct_config,
    load_reward_models, run_dir, tile_grid_of, write_json_atomic,
)

from cosmo_sr.reward.catalog_proxy import spearman, tie_aware_agreement  # noqa: E402
from cosmo_sr.reward.regions import REGION_WIDTHS, RegionGrid  # noqa: E402
from cosmo_sr.reward.torch_reward import TorchSummary  # noqa: E402


def _region_summary(region_counts: Dict[str, np.ndarray],
                    region_vol: np.ndarray) -> TorchSummary:
    return TorchSummary(
        n_sub=torch.as_tensor(region_counts["n_sub"]),
        n_host=torch.as_tensor(region_counts["n_host"]),
        occ_numerator=torch.as_tensor(region_counts["occ_numerator"]),
        volume_mpc3=torch.as_tensor(region_vol))


def region_delta_r(reward_t, base: Dict[str, np.ndarray],
                   cand: Dict[str, np.ndarray], partition, *,
                   w_joint: float, w_occ: float) -> np.ndarray:
    """``dR_g`` over every region of a partition, swapping the candidate region in.

    ``base`` and ``cand`` are per-tile arrays ``{count_key: (n_tiles, D),
    volume_mpc3: (n_tiles,)}`` under one attribution scheme. The box ``C_r`` is
    the sum of the baseline's own regions, so it is the same baseline throughout.
    """
    from cosmo_sr.reward.regions import aggregate_tile_counts, aggregate_tile_volume

    base_r = aggregate_tile_counts(partition, {k: base[k] for k in COUNTS})
    cand_r = aggregate_tile_counts(partition, {k: cand[k] for k in COUNTS})
    base_vol = aggregate_tile_volume(partition, base["volume_mpc3"])
    n_r = partition.n_regions

    box_counts = {k: base_r[k].sum(axis=0, keepdims=True).repeat(n_r, axis=0)
                  for k in COUNTS}
    box = TorchSummary(
        n_sub=torch.as_tensor(box_counts["n_sub"]),
        n_host=torch.as_tensor(box_counts["n_host"]),
        occ_numerator=torch.as_tensor(box_counts["occ_numerator"]),
        volume_mpc3=torch.as_tensor(np.full(n_r, float(base_vol.sum()))))
    frozen = _region_summary(base_r, base_vol)
    cand_s = _region_summary(cand_r, base_vol)
    with torch.no_grad():
        d = reward_t.delta_reward_swap(box, frozen, cand_s,
                                       w_joint=w_joint, w_occ=w_occ)
    return d["dR_combined"].numpy()


def _region_occ_change(region_counts: Dict[str, np.ndarray],
                       base_region: np.ndarray) -> np.ndarray:
    """``|sum_bins (S_cand,g - S_base,g)|`` per region: the regional signal in label units."""
    return np.abs(region_counts["occ_numerator"].sum(axis=1)
                  - base_region.sum(axis=1))


def offset_metrics(reward_t, frozen: Dict[int, Dict], cand_list: List,
                   touched_tiles: np.ndarray, partition, scheme: str, *,
                   w_joint: float, w_occ: float) -> Dict:
    """All region-scale numbers for one ``(scheme, width, offset)`` of one box."""
    from cosmo_sr.reward.regions import aggregate_tile_counts, aggregate_tile_volume

    seeds = sorted(frozen)
    n_r = partition.n_regions

    # Which regions the intervention actually reached (same tiles for every
    # candidate of a box, so computed once from the mask).
    touched_region = aggregate_tile_volume(
        partition, touched_tiles.astype(np.float64)) > 0.0

    # dR[r] : (n_cands, n_regions), the candidate ordering under baseline seed r.
    dR: Dict[int, np.ndarray] = {}
    for r in seeds:
        base = frozen[r][scheme]
        dR[r] = np.stack([
            region_delta_r(reward_t, base, c[scheme], partition,
                           w_joint=w_joint, w_occ=w_occ)
            for c in cand_list], axis=0)
    dR_ref = np.mean([dR[r] for r in seeds], axis=0)  # (n_cands, n_regions)

    # Informative region: touched AND the candidates are not all tied under the
    # reference baseline (an untouched region ties every candidate by
    # construction, carrying no ordering to be stable or unstable).
    informative = np.zeros(n_r, dtype=bool)
    nontied_pairs = 0
    for g in np.nonzero(touched_region)[0]:
        vals = dR_ref[:, g]
        # Non-tied *candidate* pairs are pairs of candidates with distinct dR.
        pairs = int(vals.size * (vals.size - 1) / 2) - _tied_pairs(vals)
        if pairs > 0:
            informative[g] = True
            nontied_pairs += pairs

    # Baseline-context stability: for every region, compare the candidate
    # ordering between each pair of frozen baselines. Averaged over baseline
    # pairs, then over informative regions.
    rhos: List[float] = []
    accs: List[float] = []
    for g in np.nonzero(informative)[0]:
        pair_rho, pair_acc = [], []
        for r1, r2 in itertools.combinations(seeds, 2):
            rho = spearman(dR[r1][:, g], dR[r2][:, g])
            acc, npair = tie_aware_agreement(dR[r1][:, g], dR[r2][:, g])
            if np.isfinite(rho):
                pair_rho.append(rho)
            if npair:
                pair_acc.append(acc)
        if pair_rho:
            rhos.append(float(np.mean(pair_rho)))
        if pair_acc:
            accs.append(float(np.mean(pair_acc)))

    # Regional signal and churn cancellation, in occupation-numerator units.
    base_region_occ = np.mean(
        [aggregate_tile_counts(partition,
                               {"occ_numerator": frozen[r][scheme]["occ_numerator"]})
         ["occ_numerator"] for r in seeds], axis=0)  # (n_regions, I)
    sig = []
    for c in cand_list:
        cr = aggregate_tile_counts(
            partition, {"occ_numerator": c[scheme]["occ_numerator"]})["occ_numerator"]
        s = _region_occ_change({"occ_numerator": cr}, base_region_occ)
        if touched_region.any():
            sig.append(float(s[touched_region].mean()))
    signal = float(np.mean(sig)) if sig else float("nan")

    tile_abs_all, region_abs_all, noise_touched = [], [], []
    for i, j in itertools.combinations(seeds, 2):
        tile_dS = (frozen[i][scheme]["occ_numerator"]
                   - frozen[j][scheme]["occ_numerator"])              # (T, I)
        region_dS = aggregate_tile_counts(
            partition, {"occ_numerator": tile_dS})["occ_numerator"]   # (R, I)
        tile_abs_all.append(float(np.abs(tile_dS).sum()))
        region_abs_all.append(float(np.abs(region_dS).sum()))
        if touched_region.any():
            noise_touched.append(
                float(np.abs(region_dS.sum(axis=1))[touched_region].mean()))
    tile_abs = float(np.mean(tile_abs_all))
    region_abs = float(np.mean(region_abs_all))
    noise = float(np.mean(noise_touched)) if noise_touched else float("nan")

    return {
        "n_regions": int(n_r),
        "n_touched_regions": int(touched_region.sum()),
        "n_informative_regions": int(informative.sum()),
        "n_nontied_candidate_pairs": int(nontied_pairs),
        "baseline_context_stability_spearman": (
            float(np.mean(rhos)) if rhos else float("nan")),
        "baseline_context_stability_pairwise": (
            float(np.mean(accs)) if accs else float("nan")),
        "regional_signal_mean_abs_dS_touched": signal,
        "regional_noise_mean_abs_dS_touched": noise,
        "regional_signal_to_noise": (
            float(signal / noise) if (np.isfinite(signal) and noise and noise > 0)
            else float("nan")),
        "pooled_cancellation_fraction": (
            float(1.0 - region_abs / tile_abs) if tile_abs > 0 else float("nan")),
    }


def _tied_pairs(vals: np.ndarray) -> int:
    """Number of pairs of equal values -- the pairs an ordering cannot rank."""
    v = np.round(np.asarray(vals, dtype=np.float64), 12)
    _, counts = np.unique(v, return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def _mean_finite(xs: Sequence[float]) -> float:
    a = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    return float(a.mean()) if a.size else float("nan")


def _box_bootstrap(per_box: Dict[str, float], *, n_boot: int = 2000,
                   seed: int = 0) -> Dict[str, float]:
    """Box-level bootstrap CI of a per-box scalar. Honest about small n."""
    vals = np.asarray([v for v in per_box.values() if np.isfinite(v)],
                      dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_boxes": 0}
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(vals, size=vals.size, replace=True).mean()
                       for _ in range(n_boot)])
    return {"mean": float(vals.mean()),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
            "n_boxes": int(vals.size)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--boxes", nargs="*", default=["set0", "set1"])
    ap.add_argument("--schemes", nargs="*", default=list(SCHEMES))
    ap.add_argument("--widths", nargs="*", type=int, default=list(REGION_WIDTHS))
    ap.add_argument("--touched-threshold", type=float, default=0.01)
    ap.add_argument("--eligible-threshold", type=float, default=0.65,
                    help="ordering-stability bar for a width to be pilot-eligible")
    ap.add_argument("--min-screen-boxes", type=int, default=4,
                    help="boxes needed before an eligibility verdict is emitted")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    acfg = actor_config_of(cfg)
    _, reward_t = load_reward_models(cfg)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    seeds = [int(s) for s in dataset_of(cfg).get("frozen_seeds", [0, 1, 2, 3])]
    alphas = [float(a) for a in dataset_of(cfg).get("alphas", [0.25, 0.5, 1.0])]
    modes = ["both"] + [m for m in dataset_of(cfg).get("diagnostic_modes", [])]
    if len(seeds) < 2:
        raise SystemExit(
            "baseline-context stability needs at least two frozen baselines; "
            f"the config has {seeds}")
    tg = tile_grid_of(cfg)

    report: Dict = {
        "kind": "region_scale_attribution",
        "boxes": [], "schemes": list(args.schemes),
        "widths": [int(w) for w in args.widths],
        "frozen_seeds": seeds, "touched_threshold": float(args.touched_threshold),
        "eligible_threshold": float(args.eligible_threshold),
        "per_box": {},
    }
    # per_box_scalar[(scheme,width)][box] = averaged-over-offset stability
    agg: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = {}

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

        cand_list_by_scheme: List[Dict] = []
        for m in modes:
            for a in alphas:
                tag = candidate_tag("intervention", alpha=a, mode=m)
                got = load_summaries(cfg, box, tag, args.schemes)
                if got is not None:
                    cand_list_by_scheme.append(got)
        if not cand_list_by_scheme:
            print(f"  !! no labelled interventions in {box}; skipped", flush=True)
            continue
        touched_tiles = touched_mask(cfg, box, candidate_tag("intervention",
                                     alpha=alphas[-1], mode="both"),
                                     args.touched_threshold)
        print(f"  {len(frozen)} frozen seeds, {len(cand_list_by_scheme)} "
              f"interventions, {int(touched_tiles.sum())} touched tiles", flush=True)

        per_box: Dict = {}
        for scheme in args.schemes:
            for width in args.widths:
                if tg.n_per_axis % width:
                    continue
                rg = RegionGrid(tg, int(width))
                offs = rg.valid_offsets()
                rows = [offset_metrics(
                    reward_t, frozen, cand_list_by_scheme, touched_tiles,
                    rg.partition(off), scheme, w_joint=w_joint, w_occ=w_occ)
                    for off in offs]
                # Average each metric over the partition offsets.
                keys = rows[0].keys()
                avg = {k: _mean_finite([r[k] for r in rows]) for k in keys}
                avg["n_offsets"] = len(offs)
                per_box[f"{scheme}/w{width}"] = avg
                agg.setdefault((scheme, int(width)), {})[box] = \
                    avg["baseline_context_stability_spearman"]
                print(f"    {scheme:<11s} w{width}: "
                      f"stab_rho={avg['baseline_context_stability_spearman']:6.3f} "
                      f"pairwise={avg['baseline_context_stability_pairwise']:6.3f} "
                      f"SNR={avg['regional_signal_to_noise']:5.2f} "
                      f"pooled-cancel="
                      f"{100 * avg['pooled_cancellation_fraction']:4.0f}%  "
                      f"informative={avg['n_informative_regions']:.1f}/"
                      f"{avg['n_regions']:.0f}", flush=True)
        report["per_box"][box] = per_box
        report["boxes"].append(box)

    if not report["boxes"]:
        print(">>> no box had a complete set of frozen seeds plus interventions.")
        print(">>> produced by: scripts/slurm/submit_proxy_labels.sh all BOXES=...")
        report["evidence_status"] = "no_labelled_boxes"
        out = Path(args.out) if args.out else run_dir(
            args.run_name, create=True) / "region_attribution_diagnostic.json"
        write_json_atomic(out, report)
        return 0

    # --- Cross-box summary. Average within box already done (over offsets);
    #     here average over boxes and bootstrap by box. ------------------------
    n_boxes = len(report["boxes"])
    enough = n_boxes >= int(args.min_screen_boxes)
    report["evidence_status"] = (
        "screen" if enough else "exploratory_insufficient_boxes")

    summary: Dict[str, Dict] = {}
    for (scheme, width), per_box_scalar in sorted(agg.items()):
        # Pull the paired pairwise-agreement per box for the same (scheme,width).
        pair_by_box = {b: report["per_box"][b][f"{scheme}/w{width}"]
                       ["baseline_context_stability_pairwise"]
                       for b in report["boxes"]}
        cancel_by_box = {b: report["per_box"][b][f"{scheme}/w{width}"]
                         ["pooled_cancellation_fraction"]
                         for b in report["boxes"]}
        snr_by_box = {b: report["per_box"][b][f"{scheme}/w{width}"]
                      ["regional_signal_to_noise"] for b in report["boxes"]}
        rho_boot = _box_bootstrap(per_box_scalar)
        acc_boot = _box_bootstrap(pair_by_box, seed=1)
        min_rho = min((v for v in per_box_scalar.values() if np.isfinite(v)),
                      default=float("nan"))
        min_acc = min((v for v in pair_by_box.values() if np.isfinite(v)),
                      default=float("nan"))
        thr = float(args.eligible_threshold)
        eligible = bool(np.isfinite(min_rho) and np.isfinite(min_acc)
                        and min_rho >= thr and min_acc >= thr)
        summary[f"{scheme}/w{width}"] = {
            "scheme": scheme, "width": int(width),
            "baseline_context_stability_spearman": rho_boot,
            "baseline_context_stability_pairwise": acc_boot,
            "min_box_spearman": float(min_rho),
            "min_box_pairwise": float(min_acc),
            "mean_signal_to_noise": _mean_finite(list(snr_by_box.values())),
            "mean_pooled_cancellation": _mean_finite(list(cancel_by_box.values())),
            # "Eligible" only means the ordering is stable with headroom and
            # neither box collapses. It is NOT a learnability claim: a trained
            # held-out proxy gate is what decides that (phase 5).
            "pilot_eligible_if_screen": eligible,
        }
    report["summary"] = summary

    # --- Recommendation. Exploratory when only the dev boxes are present. -----
    eligible_widths = sorted({
        s["width"] for s in summary.values() if s["pilot_eligible_if_screen"]})
    smallest = eligible_widths[0] if eligible_widths else None
    if enough:
        if smallest == 1:
            action = "retain_tile_path"
            note = ("Corrected width-1 ordering is stable with headroom on every "
                    "screen box; the simpler per-tile path is retained.")
        elif smallest is not None:
            carry = [w for w in eligible_widths if w in (2, 4)] or [smallest]
            action = f"carry_widths_{'_'.join(map(str, carry))}_into_proxy_benchmark"
            note = (f"Smallest stable region width is {smallest}; carry widths "
                    f"{carry} into the proxy benchmark (width 8 is diagnostic "
                    "evidence only).")
        else:
            action = "no_region_width_stable"
            note = ("No region width up to 8 gives a stable within-region "
                    "ordering on the screen boxes. The local-credit route has "
                    "failed the ordering screen; the next principled step is the "
                    "true-catalog oracle renderer, NOT unfreezing more of SR2.")
    else:
        # Exploratory: describe the trend, propose widths, refuse a verdict.
        ranked = sorted(
            summary.values(),
            key=lambda s: (-(s["baseline_context_stability_spearman"]["mean"]
                             if np.isfinite(s["baseline_context_stability_spearman"]
                                            ["mean"]) else -np.inf), s["width"]))
        best = ranked[0] if ranked else None
        action = "collect_pilot_boxes_then_screen"
        note = (
            "EXPLORATORY on the two development boxes only -- no eligibility "
            "verdict is emitted. "
            + (f"Most stable so far: {best['scheme']}/w{best['width']} "
               f"(stability rho {best['baseline_context_stability_spearman']['mean']:.3f}). "
               if best else "")
            + "Run the predeclared held-out pilot (set8, set9) and rerun this "
              "diagnostic on set0,set1,set8,set9 before choosing a width. The "
              "trained held-out proxy gate remains decisive.")

    report["recommendation"] = {
        "evidence_status": report["evidence_status"],
        "n_boxes": n_boxes,
        "eligible_widths_if_screen": eligible_widths,
        "smallest_eligible_width": smallest,
        "action": action,
        "note": note,
    }

    out = Path(args.out) if args.out else run_dir(
        args.run_name, create=True) / "region_attribution_diagnostic.json"
    write_json_atomic(out, report)
    banner(json.dumps({"evidence_status": report["evidence_status"],
                       "summary": summary,
                       "recommendation": report["recommendation"]}, indent=2))
    print(f"  report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
