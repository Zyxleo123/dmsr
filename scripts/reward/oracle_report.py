#!/usr/bin/env python
"""Stage 6c (CPU): assemble ensembles, score them, and decide Gate B.

Every candidate of a box is scored on the **same** stratified chunk subsets, so
the comparison between candidates is paired and the spread is sampling noise in
the residual rather than in the choice of chunks. Several independent groups are
built so "the best sample was better" can be checked for reproducibility instead
of being read off one lucky draw.

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

from _common import (add_common_args, banner, bins_of, chunk_grid, constraints_of,
                     load_reward_config, write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import pool, read_summaries
from cosmo_sr.reward.constraints import check_feasible, diversity_value
from cosmo_sr.reward.replay import marginal_contributions
from cosmo_sr.reward.reward import RewardModel


def _strata_label(cs, mode):
    parts = []
    if "box" in mode:
        parts.append(cs.box)
    if "host_mass_class" in mode:
        nz = np.nonzero(np.asarray(cs.n_host) > 0)[0]
        parts.append(f"hm{int(nz.max()) if nz.size else -1}")
    if "environment_class" in mode:
        parts.append("env_hi" if float(np.sum(cs.n_host)) > 0 else "env_lo")
    return "|".join(parts) or cs.box


def _groups(base_by_box, ensemble_size, n_groups, strata_mode, seed):
    """Stratified chunk subsets, defined from the *baseline* summaries."""
    pool_keys, labels = [], []
    for box, d in sorted(base_by_box.items()):
        for cid, cs in sorted(d.items()):
            if cs.volume_mpc3 <= 0:
                continue
            pool_keys.append((box, cid))
            labels.append(_strata_label(cs, strata_mode))
    if not pool_keys:
        raise SystemExit("no usable chunks: every chunk has zero core volume")
    rng = np.random.default_rng(int(seed))
    by_label = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)
    keys = sorted(by_label)
    groups = []
    for g in range(int(n_groups)):
        take, k = [], 0
        order = list(keys)
        rng.shuffle(order)
        seen = set()
        while len(take) < ensemble_size and k < 100 * ensemble_size:
            cand = int(rng.choice(by_label[order[k % len(order)]]))
            if cand not in seen:
                seen.add(cand)
                take.append(cand)
            k += 1
        groups.append([pool_keys[i] for i in take])
    return groups


def _diversity(cand_rows, sub: int = 128):
    """Residual spread across candidates, measured on one fixed sub-cube per box."""
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
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--ensemble-size", type=int, default=None)
    ap.add_argument("--group-seed", type=int, default=0)
    ap.add_argument("--gate-target", type=float, default=0.20,
                    help="fractional reduction in catalog discrepancy for a pass")
    ap.add_argument("--random-seed", type=int, default=17,
                    help="seed for the 'randomly selected candidate' reference")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    cons = constraints_of(cfg)
    rcfg = cfg.get("reward", {})
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

    base_by_box = {r["box"]: summaries_of(r) for r in base_rows}
    hr_by_box = {}
    for b in boxes:
        p = paths.CATALOG_CACHE() / f"{b}__hr__hr.jsonl"
        if p.is_file():
            hr_by_box[b] = {s.chunk_id: s for s in read_summaries(p)}

    occ_cfg = dict(rcfg.get("occupation", {}))
    reliable = [int(i) for i in occ_cfg.get("reliable_host_bins", [0, 1, 2, 3])]
    upper = [int(i) for i in occ_cfg.get("upper_reliable_host_bins", [2, 3])]
    sparse = [int(i) for i in occ_cfg.get("sparse_host_bins", [])]
    need_reliable = int(occ_cfg.get("min_improved_reliable_bins", 2))
    need_upper = int(occ_cfg.get("min_improved_upper_bins", 1))

    groups = _groups(base_by_box, B, args.groups,
                     rcfg.get("strata", ["box"]), args.group_seed)
    div = _diversity(cand_rows)

    # Feasibility, now including the ensemble-level diversity floor.
    for r in cand_rows:
        vals = dict(r["constraints"])
        vals["diversity"] = float(div.get(r["box"], float("nan")))
        feas, viol = check_feasible(vals, cons)
        r["constraints"] = vals
        r["feasible"] = bool(feas)
        r["violations"] = viol

    records, group_reports = [], []
    rng_pick = np.random.default_rng(int(args.random_seed))
    for gi, keys in enumerate(groups):
        gname = f"group{gi}"
        base_ens = pool([base_by_box[b][c] for b, c in keys])
        r_base = model.reward(base_ens)
        base_scores = model.scores(base_ens, reliable)
        base_gap = model.occupation_gap(base_ens)
        if all(b in hr_by_box for b, _ in keys):
            hr_ens = pool([hr_by_box[b][c] for b, c in keys])
            r_hr = model.reward(hr_ens)
            hr_occ = model.occupation_curve(hr_ens).tolist()
        else:
            r_hr, hr_occ = float("nan"), None
        per_cand = []
        for r in cand_rows:
            s = summaries_of(r)
            if not s:
                continue
            sub = [(b, c) for b, c in keys if b == r["box"]]
            if not sub:
                continue
            ens = {c: s[c] for _, c in sub if c in s}
            basel = {c: base_by_box[r["box"]][c] for _, c in sub
                     if c in base_by_box[r["box"]]}
            if len(ens) != len(sub):
                continue
            cand_ens = pool(list(ens.values()))
            reward = model.reward(cand_ens)
            marg = marginal_contributions(model, ens, basel)
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
                "residual_checkpoint": manifest.get("checkpoint"),
                "residual_path": r.get("residual_path"),
                "residual_scale": r.get("residual_scale"),
                "catalog_summary": cand_ens.to_dict(),
                "catalog_reward": reward,
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
                "marginal_contributions": {str(k): float(v) for k, v in marg.items()},
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

    reductions = [g["best_reduction_vs_mean"] for g in group_reports]
    finite = [r for r in reductions if np.isfinite(r)]
    n_ok = sum(1 for r in finite if r >= args.gate_target)
    feas_frac = float(np.mean([g["feasible_fraction"] for g in group_reports])) \
        if group_reports else 0.0

    # --- Gate B is decided on OCCUPATION -------------------------------------
    # A joint Mahalanobis improvement alone does not pass: R_cat can fall
    # because the abundance block moved while <N_sub|M_host> stayed flat, and
    # flat occupation is the failure this project exists to fix.
    occ_groups = [
        g for g in group_reports
        if g["n_improved_reliable"] >= need_reliable
        and g["n_improved_upper"] >= need_upper
        and np.isfinite(g["occ_reduction_vs_random"])
        and g["occ_reduction_vs_random"] > 0
    ]
    n_occ_ok = len(occ_groups)
    joint_ok = n_ok >= 2 and feas_frac > 0
    occ_ok = n_occ_ok >= 2 and feas_frac > 0
    abundance_only = (not occ_ok) and any(g["abundance_only"] for g in group_reports)

    if occ_ok:
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
        "diversity_per_box": div,
        # occupation criterion
        "reliable_host_bins": reliable,
        "upper_reliable_host_bins": upper,
        "sparse_host_bins_excluded": sparse,
        "min_improved_reliable_bins": need_reliable,
        "min_improved_upper_bins": need_upper,
        "per_group_n_improved_reliable": [g["n_improved_reliable"] for g in group_reports],
        "per_group_n_improved_upper": [g["n_improved_upper"] for g in group_reports],
        "per_group_occ_reduction_vs_mean": [g["occ_reduction_vs_mean"] for g in group_reports],
        "per_group_occ_reduction_vs_random": [g["occ_reduction_vs_random"] for g in group_reports],
        "n_groups_occupation_ok": n_occ_ok,
        "joint_criterion_met": bool(joint_ok),
        "occupation_criterion_met": bool(occ_ok),
        "decision": decision,
        "note": (
            "Gate B passes only on OCCUPATION: >= "
            f"{need_reliable} reliable host bins improved, including >= "
            f"{need_upper} of the upper reliable bins ({upper}), beating a "
            "single random draw, reproduced on >= 2 ensemble groups, with the "
            f"sparse bins {sparse} excluded from the criterion. A joint R_cat "
            "improvement alone is reported as 'abundance_only_improvement', "
            "not a pass. A negative result shows only that ORDINARY SAMPLES "
            "from the CURRENT prior do not contain accessible good candidates; "
            "it does not show that search or RL cannot find them -- follow the "
            "escalation ladder in docs/reward_residual_diffusion.md sec. 5."
        ),
    }
    if decision != "support_present_occupation":
        gate["diagnosis"] = _diagnose(group_reports, div, cand_rows)
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


def _diagnose(group_reports, div, cand_rows):
    out = []
    dv = [v for v in div.values() if np.isfinite(v)]
    if dv and float(np.mean(dv)) < 0.02:
        out.append(
            f"residual diversity is {np.mean(dv):.3g}: the sampler is nearly "
            "deterministic, so best-of-K has nothing to select from. Check "
            "residual_scale and the prior's sample diversity before blaming support."
        )
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
