#!/usr/bin/env python
"""Stage 6c (CPU): assemble ensembles, score them, and decide Gate B.

**One group is one box, and a box is scored whole.** The reward statistic is the
direct full-periodic-box catalog (``*__fullbox.jsonl``), not the pooled chunks:
chunk attribution rejects every halo whose Rvir sphere straddles a chunk
boundary, that rejection was measured to rise with host mass, and summing chunks
cannot put those objects back. The chunk summaries are still read and still
matter -- they are what ``marginal_contributions`` needs to attribute an
improvement to a Lagrangian region for the replay buffer -- but nothing scores
them.

An earlier version drew ``B`` chunks stratified *across* boxes: the baseline
pooled all ``B``, but a candidate exists for only one box, so it was scored on
the one or two chunks of the group that happened to be its own -- a 2-chunk
ensemble compared against an 8-chunk baseline, through a covariance fitted for
8. Reproducibility across groups now means "across boxes", which is the
statement Gate B is supposed to make anyway.

Writes a machine-readable manifest with, for every candidate: seed, chunk ids,
base checkpoint, residual checkpoint, residual scale, catalog summary, catalog
reward, every constraint value, the feasible flag, the optimized metrics and the
held-out metrics available at this stage.

    python scripts/reward/oracle_report.py --run-name prior_k8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, bins_of, constraints_of,
                     load_reward_config, require_calibrated_constraints,
                     write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import pool, read_summaries
from cosmo_sr.reward.constraints import check_constraints, diversity_value
from cosmo_sr.reward.replay import marginal_contributions
from cosmo_sr.reward.reward import RewardModel


def _groups(base_by_box, ensemble_size, candidate_boxes, seed):
    """One group per box: ``(box, [chunk ids])``, all from that box alone.

    A box must supply at least ``ensemble_size`` usable chunks. If it has more,
    a deterministic subset is taken -- the same subset for the baseline, the HR
    reference and every candidate of that box, which is what makes the
    comparison paired.
    """
    rng = np.random.default_rng(int(seed))
    groups = []
    for box in sorted(base_by_box):
        if candidate_boxes and box not in candidate_boxes:
            continue
        usable = sorted(cid for cid, cs in base_by_box[box].items()
                        if cs.volume_mpc3 > 0)
        if len(usable) < ensemble_size:
            print(f"  ! {box}: {len(usable)} usable chunks < B={ensemble_size}; "
                  f"skipping this box rather than scoring a short ensemble "
                  f"through a covariance fitted for B={ensemble_size}", flush=True)
            continue
        if len(usable) > ensemble_size:
            take = sorted(rng.choice(usable, size=ensemble_size, replace=False).tolist())
        else:
            take = usable
        groups.append((box, [(box, int(c)) for c in take]))
    if not groups:
        raise SystemExit(
            f"no box supplies a complete B={ensemble_size} ensemble of usable "
            f"chunks; Gate B cannot be decided"
        )
    return groups


def _catalog_diversity(records, feasible_only: bool = True):
    """Spread of the CATALOG across candidates of one group, per group.

    Field diversity is not the quantity Gate B needs. SR2's documented failure
    is exactly that the continuous field varies while the subhalos barely move,
    so ``diag_sample_diversity`` can sit comfortably above its floor while every
    candidate produces the same occupation curve -- and then best-of-K has
    nothing to select from no matter how many samples are drawn. These are the
    numbers that say whether the catalog itself moved.

    Only FEASIBLE candidates count. Spread contributed by a candidate that is
    already rejected -- one that wrecked the low-k field, or emptied a reliable
    host bin -- is not spread best-of-K can select from, so counting it would let
    the diversity floor be cleared by exactly the samples the gate throws away.
    """
    out = {}
    by_group = {}
    for r in records:
        if feasible_only and not r.get("feasible", False):
            continue
        by_group.setdefault(r["group"], []).append(r)
    for g in {r["group"] for r in records}:
        by_group.setdefault(g, [])
    for g, rs in by_group.items():
        if len(rs) < 2:
            out[g] = {"n": len(rs), "occupation_rel_spread": float("nan"),
                      "R_occ_spread": float("nan"),
                      "note": "fewer than 2 feasible candidates"}
            continue
        occ = np.asarray([r["occupation"] for r in rs], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            sd = np.nanstd(occ, axis=0, ddof=1)
            mean = np.abs(np.nanmean(occ, axis=0))
            rel = sd / np.where(mean > 0, mean, np.nan)
        r_occ = np.asarray([r["sub_rewards"].get("R_occ", np.nan) for r in rs],
                           dtype=np.float64)
        out[g] = {
            "n": len(rs),
            "feasible_only": bool(feasible_only),
            # Per host bin, then averaged over the bins that had hosts.
            "occupation_rel_spread": float(np.nanmean(rel)) if np.isfinite(rel).any()
            else float("nan"),
            "occupation_rel_spread_per_bin": [
                float(v) if np.isfinite(v) else None for v in rel
            ],
            "R_occ_spread": float(np.nanstd(r_occ, ddof=1))
            if np.isfinite(r_occ).sum() > 1 else float("nan"),
        }
    return out


def _diversity(cand_rows, sub: int = 128):
    """FIELD-level residual spread across candidates, on one sub-cube per box.

    This is the quantity the ``diversity`` feasibility floor bounds. It says
    nothing about whether the *catalog* varies -- see :func:`_catalog_diversity`.
    """
    out = {}
    by_box = {}
    for r in cand_rows:
        if r.get("residual_path"):
            by_box.setdefault(r["box"], []).append(r["residual_path"])
    for box, ps in by_box.items():
        if len(ps) < 2:
            out[box] = float("nan")
            continue
        cubes = []
        for p in ps:
            a = np.load(p, mmap_mode="r")
            n = a.shape[-1]
            o = (n - sub) // 2
            cubes.append(np.asarray(a[0:3, o:o + sub, o:o + sub, o:o + sub],
                                    dtype=np.float32))
        out[box] = diversity_value(cubes)
    return out


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--reward-model", default=None)
    ap.add_argument("--groups", type=int, default=0,
                    help="cap on the number of groups (= boxes); 0 = every box")
    ap.add_argument("--ensemble-size", type=int, default=None)
    ap.add_argument("--group-seed", type=int, default=0)
    ap.add_argument("--gate-target", type=float, default=0.20,
                    help="fractional reduction in catalog discrepancy for a pass")
    ap.add_argument("--random-seed", type=int, default=17,
                    help="seed for the 'randomly selected candidate' reference")
    ap.add_argument("--credit-score", default="R_occ",
                    choices=["R_occ", "R_abund", "R_cat"],
                    help="which reward per-chunk credit is measured in; must "
                         "match what selection is decided on (default R_occ)")
    ap.add_argument("--allow-chunk-pooled", action="store_true",
                    help="score pooled chunks when a full-box summary is missing "
                         "(diagnostic only: pooled chunks drop every "
                         "boundary-crossing halo, mass-dependently)")
    ap.add_argument("--allow-empty-reliable-bins", action="store_true",
                    help="do NOT mark a candidate infeasible for having zero "
                         "hosts in a reliable bin (diagnostic only)")
    ap.add_argument("--allow-uncalibrated", action="store_true",
                    help="report a verdict even though constraints.calibrated is "
                         "false (diagnostic only: the feasibility filter, and so "
                         "the elite set, is then made of placeholders)")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    cons = constraints_of(cfg)
    rcfg = cfg.get("reward", {})
    # A pass decided with placeholder feasibility thresholds is not a pass: the
    # elite set is defined by them. Report everything, then refuse the verdict.
    uncalibrated = None if args.allow_uncalibrated else require_calibrated_constraints(cfg)
    out = paths.ORACLE(args.run_name)
    manifest = json.loads((out / "candidates.json").read_text())
    B = int(args.ensemble_size or rcfg.get("ensemble_size_B", 16))

    rm_path = Path(args.reward_model) if args.reward_model else \
        paths.subdir("reward_model") / "reward_model.json"
    if not Path(rm_path).is_file():
        raise SystemExit(
            f"no reward model at {rm_path}; run scripts/reward/fit_reward_model.py"
        )
    model = RewardModel.from_dict(json.loads(Path(rm_path).read_text()))

    rows = [json.loads(p.read_text()) for p in sorted((out / "scored").glob("*.json"))]
    if not rows:
        raise SystemExit(f"no scored candidates in {out / 'scored'}")
    by_tag = {r["tag"]: r for r in rows}
    cand_rows = [r for r in rows if r.get("seed") is not None]
    base_rows = [r for r in rows if r.get("seed") is None]
    boxes = sorted({r["box"] for r in rows})

    def summaries_of(row):
        p = row.get("summaries")
        if not p:
            return {}
        return {s.chunk_id: s for s in read_summaries(p)}

    def full_box_of(row):
        """The whole-box summary a row is SCORED on, or None."""
        p = row.get("full_box_summary")
        if p and Path(p).is_file():
            return pool(read_summaries(p))
        return None

    base_by_box = {r["box"]: summaries_of(r) for r in base_rows}
    base_full = {r["box"]: full_box_of(r) for r in base_rows}
    hr_by_box, hr_full = {}, {}
    for b in boxes:
        p = paths.CATALOG_CACHE() / f"{b}__hr__hr.jsonl"
        if p.is_file():
            hr_by_box[b] = {s.chunk_id: s for s in read_summaries(p)}
        fp = paths.CATALOG_CACHE() / f"{b}__hr__hr__fullbox.jsonl"
        if fp.is_file():
            hr_full[b] = pool(read_summaries(fp))

    # Scoring on pooled chunks instead of the full box would compare a candidate
    # against a reward model fitted on a different statistic, so it is refused
    # rather than silently downgraded.
    missing_full = sorted(b for b, v in base_full.items() if v is None)
    if missing_full and not args.allow_chunk_pooled:
        raise SystemExit(
            f"no full-box summary for the baseline of {missing_full}. Gate B "
            f"scores the direct full-periodic-box catalog; chunk-pooled vectors "
            f"are missing every boundary-crossing object and are not the "
            f"statistic the reward model was fitted on. Re-run the scoring "
            f"array (scripts/reward/score_oracle.py) with --overwrite, or pass "
            f"--allow-chunk-pooled for a diagnostic-only report."
        )

    occ_cfg = dict(rcfg.get("occupation", {}))
    reliable = [int(i) for i in occ_cfg.get("reliable_host_bins", [0, 1, 2, 3])]
    upper = [int(i) for i in occ_cfg.get("upper_reliable_host_bins", [2, 3])]
    sparse = [int(i) for i in occ_cfg.get("sparse_host_bins", [])]
    need_reliable = int(occ_cfg.get("min_improved_reliable_bins", 2))
    need_upper = int(occ_cfg.get("min_improved_upper_bins", 1))
    min_cat_div = float(occ_cfg.get("min_catalog_occupation_spread", 0.02))

    groups = _groups(base_by_box, B, {r["box"] for r in cand_rows}, args.group_seed)
    if args.groups and len(groups) > int(args.groups):
        groups = groups[:int(args.groups)]
    div = _diversity(cand_rows)

    # Feasibility, now including the ensemble-level diversity floor.
    for r in cand_rows:
        vals = dict(r["constraints"])
        vals["diversity"] = float(div.get(r["box"], float("nan")))
        chk = check_constraints(vals, cons)
        r["constraints"] = vals
        r["feasible"] = bool(chk["feasible"])
        r["violations"] = chk["violations"]
        # Non-blocking breaches are re-derived here rather than read off the row,
        # because `diversity` only becomes measurable at this stage and the
        # severity map may have changed since the row was scored.
        r["constraint_warnings"] = chk["warnings"]
        r["constraint_critical"] = chk["critical"]

    records, group_reports = [], []
    rng_pick = np.random.default_rng(int(args.random_seed))
    for gi, (gbox, keys) in enumerate(groups):
        gname = f"group{gi}_{gbox}"
        # SCORED on the whole box; the chunk pool is only the fallback the
        # --allow-chunk-pooled escape hatch permits.
        base_ens = base_full.get(gbox) or pool([base_by_box[b][c] for b, c in keys])
        r_base = model.reward(base_ens)
        base_scores = model.scores(base_ens, reliable)
        base_gap = model.occupation_gap(base_ens)
        base_empty = model.empty_host_bins(base_ens, reliable)
        if base_empty:
            print(f"  ! {gname}: the FROZEN BASELINE has no hosts in reliable "
                  f"bin(s) {base_empty}; every comparison in those bins is "
                  f"against a filled-in value", flush=True)
        hr_ens = hr_full.get(gbox)
        if hr_ens is None and all(b in hr_by_box for b, _ in keys):
            hr_ens = pool([hr_by_box[b][c] for b, c in keys])
        if hr_ens is not None:
            r_hr = model.reward(hr_ens)
            hr_occ = model.occupation_curve(hr_ens).tolist()
        else:
            r_hr, hr_occ = float("nan"), None
        per_cand = []
        for r in cand_rows:
            if r["box"] != gbox:
                continue
            s = summaries_of(r)
            if not s:
                continue
            ens = {c: s[c] for _, c in keys if c in s}
            basel = {c: base_by_box[gbox][c] for _, c in keys
                     if c in base_by_box[gbox]}
            if len(ens) != len(keys) or len(basel) != len(keys):
                # A candidate missing any chunk of the group is not comparable
                # to a baseline that has all of them; drop it rather than pool
                # a shorter ensemble.
                print(f"  ! {r['tag']}: has {len(ens)}/{len(keys)} chunks of "
                      f"{gname}; excluded", flush=True)
                continue
            cand_full = full_box_of(r)
            cand_ens = cand_full or pool(list(ens.values()))
            reward = model.reward(cand_ens)
            # An empty reliable host bin is filled with mu, so it contributes
            # EXACTLY ZERO to the distance: a candidate can raise R_occ by
            # deleting the hosts whose occupation it cannot reproduce, and the
            # quadratic form cannot tell. The "bins improved" test does not catch
            # it either -- an empty bin is NaN there, and only two reliable bins
            # have to improve. So an empty reliable bin makes the candidate
            # infeasible instead of scoring it.
            empty_bins = model.empty_host_bins(cand_ens, reliable)
            if empty_bins and not args.allow_empty_reliable_bins:
                r["feasible"] = False
                r["violations"] = list(r.get("violations", [])) + [
                    f"empty_reliable_host_bin_{i}" for i in empty_bins
                ]
            # Credit in the score selection is decided on. Gate B is decided on
            # occupation, so the replay buffer is weighted by occupation credit;
            # the joint figure is kept alongside it for diagnosis, never as the
            # training signal.
            marg = marginal_contributions(model, ens, basel, score=args.credit_score)
            marg_cat = marginal_contributions(model, ens, basel, score="R_cat")
            cand_scores = model.scores(cand_ens, reliable)
            cand_gap = model.occupation_gap(cand_ens)
            # A host bin "improves" when the candidate's whitened distance from
            # the HR mean shrinks. NaN (no hosts in the bin) is not an
            # improvement -- an empty bin carries no evidence either way.
            improved = np.asarray(
                np.isfinite(cand_gap) & np.isfinite(base_gap) & (cand_gap < base_gap)
            )
            rec = {
                "ensemble_id": f"{args.run_name}:{gname}:{r['tag']}",
                "group": gname, "box": r["box"], "seed": r["seed"],
                "chunk_ids": sorted(ens),
                "base_checkpoint": manifest.get("checkpoint"),
                "base_field": r.get("base_path"),
                "base_seed": int(manifest.get("base_seed", 0)),
                "residual_checkpoint": manifest.get("checkpoint"),
                "residual_path": r.get("residual_path"),
                "residual_scale": r.get("residual_scale"),
                "catalog_summary": cand_ens.to_dict(),
                "catalog_scope": "full_box" if cand_full is not None else "pooled_chunks",
                "empty_reliable_host_bins": empty_bins,
                "catalog_reward": reward,
                # The score the elite cut is taken on: the same one the credit
                # is measured in, so selection and weighting agree.
                "selection_reward": float(cand_scores.get(args.credit_score, reward)),
                "selection_score": args.credit_score,
                "reward_base": r_base,
                "reward_hr": r_hr,
                "sub_rewards": cand_scores,
                "sub_rewards_base": base_scores,
                "occupation": model.occupation_curve(cand_ens).tolist(),
                "occupation_base": model.occupation_curve(base_ens).tolist(),
                "occupation_hr": hr_occ,
                "occupation_gap": cand_gap.tolist(),
                "occupation_gap_base": base_gap.tolist(),
                "occupation_bins_improved": [bool(v) for v in improved],
                "n_improved_reliable": int(sum(improved[i] for i in reliable)),
                "n_improved_upper": int(sum(improved[i] for i in upper)),
                "n_improved_sparse": int(sum(improved[i] for i in sparse)),
                "constraint_values": r["constraints"],
                "feasible": bool(r["feasible"]),
                "violations": r["violations"],
                "constraint_warnings": r.get("constraint_warnings", []),
                "constraint_critical": r.get("constraint_critical", []),
                "marginal_contributions": {str(k): float(v) for k, v in marg.items()},
                "marginal_contributions_score": args.credit_score,
                "marginal_contributions_R_cat": {
                    str(k): float(v) for k, v in marg_cat.items()},
                "optimized_metrics": model.components(cand_ens),
                "held_out_metrics": {
                    "n_subs_full_box": r.get("n_subs"),
                    "n_hosts_full_box": r.get("n_hosts"),
                    "assigned_fraction": r.get("assigned_fraction"),
                    "density_sigma_ratio": r["constraints"].get("density_sigma_ratio"),
                    "density_pdf_l1": r["constraints"].get("density_pdf_l1"),
                    "displacement_rk_low_k": r["constraints"].get("displacement_rk_low_k"),
                },
            }
            records.append(rec)
            per_cand.append(rec)

        feas = [x for x in per_cand if x["feasible"]]
        d_base = -r_base
        report = {
            "group": gname,
            "box": gbox,
            "chunks": [[b, int(c)] for b, c in keys],
            "n_candidates": len(per_cand),
            "n_feasible": len(feas),
            "feasible_fraction": len(feas) / max(len(per_cand), 1),
            "D2_base": d_base,
            "D2_hr": -r_hr,
            "D2_mean_sample": float(np.mean([-x["catalog_reward"] for x in per_cand]))
            if per_cand else float("nan"),
            "D2_best_feasible": float(np.min([-x["catalog_reward"] for x in feas]))
            if feas else float("nan"),
            "occupation_hr": hr_occ,
            "occupation_base": model.occupation_curve(base_ens).tolist(),
        }
        for key, ref in (("vs_base", d_base), ("vs_mean", report["D2_mean_sample"])):
            best = report["D2_best_feasible"]
            report[f"best_reduction_{key}"] = (
                float((ref - best) / ref) if np.isfinite(best) and ref > 0 else float("nan")
            )

        # --- the four required references, in each of the three scores -------
        # average / random / best-joint / best-occupation. "Random" is a real
        # single draw, not the mean: best-of-K has to beat the sample you would
        # have taken anyway, and the mean of K is a strictly easier target.
        rand_ref = (feas[int(rng_pick.integers(len(feas)))] if feas
                    else (per_cand[int(rng_pick.integers(len(per_cand)))]
                          if per_cand else None))
        best_joint = min(feas, key=lambda x: -x["catalog_reward"]) if feas else None
        best_occ = min(feas, key=lambda x: -x["sub_rewards"]["R_occ"]) if feas else None

        for name, key in (("R_cat", "R_cat"), ("R_occ", "R_occ"),
                          ("R_abund", "R_abund"), ("R_occ_reliable", "R_occ_reliable")):
            vals = [-x["sub_rewards"][key] for x in per_cand
                    if key in x["sub_rewards"]]
            fvals = [-x["sub_rewards"][key] for x in feas if key in x["sub_rewards"]]
            report[f"D2_{name}_base"] = -base_scores.get(key, float("nan"))
            report[f"D2_{name}_mean"] = float(np.mean(vals)) if vals else float("nan")
            report[f"D2_{name}_random"] = (
                -rand_ref["sub_rewards"][key] if rand_ref else float("nan"))
            report[f"D2_{name}_best"] = float(np.min(fvals)) if fvals else float("nan")
            report[f"D2_{name}_best_joint"] = (
                -best_joint["sub_rewards"][key] if best_joint else float("nan"))
            report[f"D2_{name}_best_occ"] = (
                -best_occ["sub_rewards"][key] if best_occ else float("nan"))

        report["candidate_tags"] = {
            "random": rand_ref["ensemble_id"] if rand_ref else None,
            "best_joint": best_joint["ensemble_id"] if best_joint else None,
            "best_occupation": best_occ["ensemble_id"] if best_occ else None,
        }
        # Occupation bin accounting for the best-occupation candidate, which is
        # the one Gate B is decided on.
        report["occ_bins_improved"] = (
            best_occ["occupation_bins_improved"] if best_occ else None)
        report["n_improved_reliable"] = (
            int(best_occ["n_improved_reliable"]) if best_occ else 0)
        report["n_improved_upper"] = (
            int(best_occ["n_improved_upper"]) if best_occ else 0)
        report["occupation_best_occ"] = best_occ["occupation"] if best_occ else None
        # Best-of-K must also beat a single random draw, not only the mean.
        for ref_key in ("mean", "random"):
            ref = report[f"D2_R_occ_{ref_key}"]
            best = report["D2_R_occ_best"]
            report[f"occ_reduction_vs_{ref_key}"] = (
                float((ref - best) / ref)
                if np.isfinite(best) and np.isfinite(ref) and ref > 0 else float("nan")
            )
        report["abundance_only"] = bool(
            report["n_improved_reliable"] < need_reliable
            and np.isfinite(report["D2_R_abund_best"])
            and report["D2_R_abund_best"] < report["D2_R_abund_base"]
        )
        group_reports.append(report)

    with open(out / "oracle_manifest.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, default=float) + "\n")

    cat_div = _catalog_diversity(records)
    reductions = [g["best_reduction_vs_mean"] for g in group_reports]
    finite = [r for r in reductions if np.isfinite(r)]
    n_ok = sum(1 for r in finite if r >= args.gate_target)
    feas_frac = float(np.mean([g["feasible_fraction"] for g in group_reports])) \
        if group_reports else 0.0

    # --- Gate B is decided on OCCUPATION -------------------------------------
    # A joint Mahalanobis improvement alone does not pass: R_cat can fall
    # because the abundance block moved while <N_sub|M_host> stayed flat, and
    # flat occupation is the failure this project exists to fix.
    # The target is a SIZE, not a sign. "Better than a random draw by any
    # margin" was the old test, and with K candidates the best of K beats the
    # median of K essentially by construction: a >0 criterion measures that
    # sampling noise exists, not that occupation can be controlled. The same
    # gate_target the joint score is held to now applies to occupation.
    #
    # CATALOG diversity is part of the criterion, not a footnote next to it. If
    # every feasible candidate of a group produces the same occupation curve,
    # "the best of K beat a random draw by 20%" is measuring which draw the RNG
    # called random, not a controllable degree of freedom -- and that is SR2's
    # original failure (field moves, subhalos do not) reproduced in the residual.
    def _cat_div_ok(g):
        v = cat_div.get(g["group"], {}).get("occupation_rel_spread", float("nan"))
        return bool(np.isfinite(v) and v >= min_cat_div)

    occ_groups = [
        g for g in group_reports
        if g["n_improved_reliable"] >= need_reliable
        and g["n_improved_upper"] >= need_upper
        and np.isfinite(g["occ_reduction_vs_random"])
        and g["occ_reduction_vs_random"] >= args.gate_target
        and _cat_div_ok(g)
    ]
    n_occ_ok = len(occ_groups)
    joint_ok = n_ok >= 2 and feas_frac > 0
    occ_ok = n_occ_ok >= 2 and feas_frac > 0
    abundance_only = (not occ_ok) and any(g["abundance_only"] for g in group_reports)

    if uncalibrated:
        decision = "blocked_uncalibrated_constraints"
    elif occ_ok:
        decision = "support_present_occupation"
    elif abundance_only:
        decision = "abundance_only_improvement"
    else:
        decision = "support_absent"

    gate = {
        "target_reduction": args.gate_target,
        "per_group_reduction_vs_mean": reductions,
        "per_group_reduction_vs_base": [g["best_reduction_vs_base"] for g in group_reports],
        "n_groups_meeting_target": n_ok,
        "n_groups": len(group_reports),
        "mean_feasible_fraction": feas_frac,
        "field_diversity_per_box": div,
        "catalog_diversity_per_group": cat_div,
        # occupation criterion
        "reliable_host_bins": reliable,
        "upper_reliable_host_bins": upper,
        "sparse_host_bins_excluded": sparse,
        "min_improved_reliable_bins": need_reliable,
        "min_improved_upper_bins": need_upper,
        "min_catalog_occupation_spread": min_cat_div,
        "per_group_catalog_occupation_spread": [
            cat_div.get(g["group"], {}).get("occupation_rel_spread", float("nan"))
            for g in group_reports
        ],
        "per_group_catalog_diversity_ok": [_cat_div_ok(g) for g in group_reports],
        "per_group_n_empty_reliable_bin_candidates": [
            sum(1 for r in records
                if r["group"] == g["group"] and r.get("empty_reliable_host_bins"))
            for g in group_reports
        ],
        "per_group_n_improved_reliable": [g["n_improved_reliable"] for g in group_reports],
        "per_group_n_improved_upper": [g["n_improved_upper"] for g in group_reports],
        "per_group_occ_reduction_vs_mean": [g["occ_reduction_vs_mean"] for g in group_reports],
        "per_group_occ_reduction_vs_random": [g["occ_reduction_vs_random"] for g in group_reports],
        "n_groups_occupation_ok": n_occ_ok,
        "joint_criterion_met": bool(joint_ok),
        "occupation_criterion_met": bool(occ_ok),
        "constraints_calibrated": uncalibrated is None,
        "constraints_uncalibrated_reason": uncalibrated,
        "group_definition": "one complete box per group (B chunks from that box)",
        "decision": decision,
        "note": (
            "Gate B passes only on OCCUPATION: >= "
            f"{need_reliable} reliable host bins improved, including >= "
            f"{need_upper} of the upper reliable bins ({upper}), beating a "
            f"single random draw by >= {100 * args.gate_target:.0f}%, with a "
            f"catalog occupation spread >= {min_cat_div} among the FEASIBLE "
            "candidates, reproduced "
            "on >= 2 boxes, with the "
            f"sparse bins {sparse} excluded from the criterion. Candidates with "
            "an empty reliable host bin are infeasible: an empty bin is filled "
            "with mu and would otherwise score a perfect zero for having deleted "
            "its hosts. Every score is taken on the DIRECT FULL-BOX catalog. "
            "A joint R_cat "
            "improvement alone is reported as 'abundance_only_improvement', "
            "not a pass. A negative result shows only that ORDINARY SAMPLES "
            "from the CURRENT prior do not contain accessible good candidates; "
            "it does not show that search or RL cannot find them -- follow the "
            "escalation ladder in docs/reward_residual_diffusion.md sec. 5."
        ),
    }
    if decision != "support_present_occupation":
        gate["diagnosis"] = _diagnose(group_reports, div, cand_rows, cat_div)
        if uncalibrated:
            gate["diagnosis"].insert(0, uncalibrated)
        gate["diagnosis"].append(
            "Gate B is not a final no-go test. Escalate: (1) sweep K, "
            "temperature and residual scale; (2) CEM/evolutionary search over "
            "the diffusion noise; (3) fixed-host reward overfitting; "
            "(4) distil or DDPO only if search shows reward variation; "
            "(5) larger-context or host-conditioned model; (6) the true-HR "
            "catalog oracle renderer."
        )

    write_json(out / "gate_b.json",
               {"run_name": args.run_name, "groups": group_reports, "gate": gate,
                "reward_model": str(rm_path),
                "reward_model_condition_number": model.condition_number,
                "reward_model_lambda": model.lam})

    banner(f"Gate B: {gate['decision']}")
    if uncalibrated:
        print(f"  !! {uncalibrated}", flush=True)
        print("  !! No verdict is reported until then. Everything below is still "
              "computed, for diagnosis only.", flush=True)
    for g in group_reports:
        print(f"  {g['group']}: D2 base={g['D2_base']:.4g} mean={g['D2_mean_sample']:.4g} "
              f"best={g['D2_best_feasible']:.4g} "
              f"reduction vs mean={100 * g['best_reduction_vs_mean']:.1f}% "
              f"feasible={g['n_feasible']}/{g['n_candidates']}", flush=True)
        print(f"      R_occ   base={g['D2_R_occ_base']:.4g} "
              f"mean={g['D2_R_occ_mean']:.4g} rand={g['D2_R_occ_random']:.4g} "
              f"best={g['D2_R_occ_best']:.4g}   "
              f"reliable bins improved={g['n_improved_reliable']}/{len(reliable)} "
              f"(upper {g['n_improved_upper']}/{len(upper)})", flush=True)
        print(f"      R_abund base={g['D2_R_abund_base']:.4g} "
              f"mean={g['D2_R_abund_mean']:.4g} best={g['D2_R_abund_best']:.4g}",
              flush=True)
    if "diagnosis" in gate:
        for d in gate["diagnosis"]:
            print(f"  ! {d}", flush=True)
    print(f"  -> {out / 'gate_b.json'}", flush=True)


def _diagnose(group_reports, div, cand_rows, cat_div=None):
    out = []
    dv = [v for v in div.values() if np.isfinite(v)]
    if dv and float(np.mean(dv)) < 0.02:
        out.append(
            f"residual FIELD diversity is {np.mean(dv):.3g}: the sampler is nearly "
            "deterministic, so best-of-K has nothing to select from. Check "
            "residual_scale and the prior's sample diversity before blaming support."
        )
    cv = [g["occupation_rel_spread"] for g in (cat_div or {}).values()
          if np.isfinite(g.get("occupation_rel_spread", np.nan))]
    if cv and float(np.mean(cv)) < 0.02:
        msg = (
            f"CATALOG diversity is {np.mean(cv):.3g}: the occupation curve is "
            "essentially the same for every candidate"
        )
        if dv and float(np.mean(dv)) >= 0.02:
            msg += (
                f", even though field diversity is {np.mean(dv):.3g}. That is "
                "SR2's original failure reproduced in the residual: the field "
                "moves, the subhalos do not. More samples will not help; the "
                "residual has to reach the collapsed scales"
            )
        out.append(msg + ".")
    feas = [r for r in cand_rows if r.get("feasible")]
    if not feas:
        out.append(
            "no candidate is feasible: either the residual is genuinely damaging "
            "the field, or the constraint thresholds are too tight -- recalibrate "
            "with scripts/reward/calibrate_constraints.py before loosening them."
        )
    spread = [g["D2_mean_sample"] - g["D2_best_feasible"] for g in group_reports
              if np.isfinite(g["D2_best_feasible"])]
    if spread and max(spread) <= 0:
        out.append(
            "the best sample is no better than the average: the reward does not "
            "vary across residual realisations. Increase K, widen the residual "
            "scale, or enlarge the conditioning context."
        )
    out.append("Do not proceed to distillation; policy support is insufficient.")
    return out


if __name__ == "__main__":
    main()
