#!/usr/bin/env python
"""Which scalar catalog target is stable across SR2 seeds, and still says
"more subhalos, preferably in hosts"?

The question
------------
The line's objective is ``R_cat = -D^2``, an 11-d Mahalanobis distance to the HR
mean over 6 subhalo-abundance bins and 5 occupation bins. A reward is only a
usable training signal if the improvement it is meant to detect is large
compared with the churn it shows when the *same* generator is re-run with a
different noise seed. That ratio, not a correlation coefficient, is the thing to
optimise when choosing the functional.

What this measures
------------------
Every candidate directory under ``candidates/`` already carries the exact
per-tile catalog sufficient statistics (``tile_summaries.jsonl``), which sum to
the whole-box catalog by construction. For each box we therefore have

* ``frozen_seed0..3``  -- the SAME frozen generator, four noise seeds. Spread
  here is pure seed churn: noise, by definition.
* ``hr``               -- the target. Distance from frozen is the headroom.
* ``intervention_*``   -- the monotone alpha-ladder, seed 0 only. Distance from
  ``frozen_seed0`` is a realistic *achievable* step.

For each candidate functional ``T`` the report gives

``sigma_seed``   within-box std over the four frozen seeds, pooled over boxes.
``sigma_box``    box-to-box std of the seed-mean: cosmic variance, for scale.
``d_hr``         mean (T_hr - T_frozen) / sigma_seed -- total headroom in noise
                 units.
``d_alpha1``     mean (T_int_alpha1 - T_frozen_seed0) / sigma_seed -- one
                 realistic step in noise units.
``p_rank``       P(a single-seed evaluation ranks the alpha=1 step above the
                 frozen model) = Phi(d_alpha1 / sqrt(2)). This is the
                 "rank consistency across seeds" number, expressed for the
                 effect size we actually need to resolve.
``rho_boxes``    directly measured: Spearman of the box ordering under seed r vs
                 seed r', averaged over seed pairs. Boxes differ genuinely
                 (cosmic variance), so this asks whether a real difference of
                 that size survives seed churn. No modelling.
``alpha_monotone`` fraction of boxes whose alpha-ladder comes out ordered.

Read ``p_rank`` and ``rho_boxes`` together: the first is the signal we need, the
second is a model-free check that the statistic orders anything at all.

CPU-only, reads ~17 MB of JSON. Submit it; do not run it on the login node.

    python scripts/reward/reward_stability_scan.py [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from _sr2_direct import (  # noqa: E402
    add_direct_args, banner, direct_root, load_direct_config, load_reward_models,
    run_dir, write_json_atomic,
)

from cosmo_sr.reward.catalog import ChunkSummary, EnsembleSummary, pool  # noqa: E402
from cosmo_sr.reward.catalog_proxy import spearman  # noqa: E402
from cosmo_sr.reward.reward import RewardModel  # noqa: E402

FROZEN = re.compile(r"^frozen_seed(\d+)$")
INTERVENTION = re.compile(r"^intervention_(\w+?)_a(\d\.\d+)_seed(\d+)$")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_box(path: Path) -> Optional[EnsembleSummary]:
    """Whole-box pooled summary from one candidate's tile table, or None."""
    f = path / "tile_summaries.jsonl"
    if not f.is_file():
        return None
    rows = []
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("chunk_id", d.get("tile_id", 0))
            rows.append(ChunkSummary.from_dict(d))
    return pool(rows) if rows else None


def collect(root: Path) -> Dict[str, Dict[str, EnsembleSummary]]:
    """``{box: {candidate_tag: whole-box summary}}``."""
    out: Dict[str, Dict[str, EnsembleSummary]] = defaultdict(dict)
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        box, tag = d.name.split("__", 1)
        ens = load_box(d)
        if ens is not None:
            out[box][tag] = ens
    return dict(out)


# --------------------------------------------------------------------------- #
# The menu of candidate targets
#
# Every entry is a scalar of one whole-box catalog. The axis being scanned is
# *quadratic distance vs monotone functional*, and *which counts feed it*,
# because that is what sets the noise behaviour:
#
#   - a distance rectifies noise (E[-D^2] is depressed by seed churn even at
#     zero bias, and its gradient sign flips at the target), so its seed
#     variance carries a term a linear functional does not have;
#   - a flat bin-mean in log space weights a bin with 30 hosts the same as a bin
#     with 1300, so the sparse bin dominates the churn;
#   - a pooled count weights by evidence, which is the Poisson-optimal thing to
#     do when every bin is deficient in the same direction.
# --------------------------------------------------------------------------- #
def make_statistics(model: RewardModel, cfg: Dict):
    bins = model.bins
    # The occupation policy lives in reward.yaml, which load_direct_config
    # parks under "_reward" -- reading cfg["reward"] here would silently fall
    # back to the defaults and score different bins than the gates do.
    occ_cfg = ((cfg.get("_reward", {}) or {}).get("reward", {}) or {}).get(
        "occupation", {}) or {}
    reliable = [int(i) for i in occ_cfg.get("reliable_host_bins", [0, 1, 2, 3])]
    upper = [int(i) for i in occ_cfg.get("upper_reliable_host_bins", [2, 3])]
    floor = float(bins.occupation_floor)

    def occ(ens: EnsembleSummary) -> np.ndarray:
        o = ens.occupation()
        return np.where(np.isfinite(o), np.nan_to_num(o, nan=0.0), 0.0)

    def logocc(ens: EnsembleSummary, idx: Sequence[int]) -> np.ndarray:
        return np.log10(occ(ens)[list(idx)] + floor)

    stats = {}

    # --- what the line uses today --------------------------------------- #
    stats["R_cat"] = lambda e: model.reward(e)
    stats["R_occ"] = model.reward_occupation
    stats["R_abund"] = model.reward_abundance

    # --- monotone functionals of the same sufficient statistics --------- #
    # "subhalos that live in a resolved host", pooled over host mass: the
    # literal reading of the goal, and an evidence-weighted count.
    stats["log_hosted_subs"] = lambda e: math.log10(
        max(float(np.sum(e.occ_numerator)), 0.5))
    # every resolved subhalo, hosted or not: the weaker goal, kept as a control.
    stats["log_all_subs"] = lambda e: math.log10(max(float(np.sum(e.n_sub)), 0.5))
    # hosts only: should carry NO gradient for this goal. A control that must
    # come out uninformative; if it does not, the target is partly measuring the
    # host finder rather than substructure.
    stats["log_hosts"] = lambda e: math.log10(max(float(np.sum(e.n_host)), 0.5))

    # mean occupation in log space: the shape of the occupation curve, equal
    # weight per host-mass decade.
    stats["mean_log_occ"] = lambda e: float(np.mean(logocc(e, reliable)))
    stats["mean_log_occ_upper"] = lambda e: float(np.mean(logocc(e, upper)))

    # host-count-weighted log occupation: same curve, but a bin contributes in
    # proportion to how well it is measured. Sits between the pooled count and
    # the flat bin mean.
    def w_log_occ(e: EnsembleSummary) -> float:
        h = np.asarray(e.n_host, dtype=np.float64)[reliable]
        v = logocc(e, reliable)
        w = h / max(h.sum(), 1e-30)
        return float(np.sum(w * v))
    stats["nhost_weighted_log_occ"] = w_log_occ

    # one-sided: no credit past HR. mu is in the model's transformed space, so
    # its occupation block is already log10(occ + floor).
    j = bins.n_sub_bins
    mu_occ = np.asarray(model.mu, dtype=np.float64)[j:j + bins.n_host_bins]

    def capped(e: EnsembleSummary) -> float:
        v = logocc(e, reliable)
        return float(np.mean(np.minimum(v, mu_occ[reliable])))
    stats["mean_log_occ_capped_at_hr"] = capped

    return stats, reliable, upper


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def evaluate(values: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """``values[box][tag] -> scalar`` for ONE statistic -> its stability report."""
    seed_sd: List[float] = []
    seed_mean: Dict[str, float] = {}
    hr_gap: List[float] = []
    a1_gap: List[float] = []
    ladder_ok = ladder_n = 0
    per_seed: Dict[int, Dict[str, float]] = defaultdict(dict)

    for b in sorted(values):
        v = values[b]
        seeds = {}
        for t, x in v.items():
            m = FROZEN.match(t)
            if m:
                seeds[int(m.group(1))] = float(x)
        arr = np.array([seeds[k] for k in sorted(seeds)], dtype=np.float64)
        if arr.size < 2 or not np.all(np.isfinite(arr)):
            continue
        seed_sd.append(float(np.std(arr, ddof=1)))
        seed_mean[b] = float(np.mean(arr))
        for k, x in seeds.items():
            per_seed[k][b] = x

        if "hr" in v and np.isfinite(v["hr"]):
            hr_gap.append(float(v["hr"]) - seed_mean[b])

        base = seeds.get(0)
        ladder: Dict[float, float] = {}
        for t, x in v.items():
            m = INTERVENTION.match(t)
            if m and m.group(1) == "both" and int(m.group(3)) == 0:
                ladder[float(m.group(2))] = float(x)
        if base is not None and ladder:
            if 1.0 in ladder and np.isfinite(ladder[1.0]):
                a1_gap.append(ladder[1.0] - base)
            xs = [base] + [ladder[a] for a in sorted(ladder)]
            if all(np.isfinite(xs)):
                ladder_n += 1
                d = np.diff(xs)
                ladder_ok += int(bool(np.all(d >= 0) or np.all(d <= 0)))

    # Seed noise: pool the within-box variances. Each box contributes its own
    # 4-seed sample; they are not draws around one common mean.
    sigma_seed = float(np.sqrt(np.mean(np.square(seed_sd)))) if seed_sd else float("nan")
    sigma_box = (float(np.std(list(seed_mean.values()), ddof=1))
                 if len(seed_mean) > 1 else float("nan"))

    # Directly measured cross-seed rank consistency of the box ordering.
    rhos: List[float] = []
    ks = sorted(per_seed)
    for i in range(len(ks)):
        for jj in range(i + 1, len(ks)):
            common = sorted(set(per_seed[ks[i]]) & set(per_seed[ks[jj]]))
            if len(common) >= 4:
                r = spearman(np.array([per_seed[ks[i]][b] for b in common]),
                             np.array([per_seed[ks[jj]][b] for b in common]))
                if np.isfinite(r):
                    rhos.append(float(r))

    def dprime(gaps: List[float]) -> float:
        if not gaps or not np.isfinite(sigma_seed) or sigma_seed <= 0:
            return float("nan")
        return float(np.mean(gaps) / sigma_seed)

    d_hr, d_a1 = dprime(hr_gap), dprime(a1_gap)
    return {
        "n_boxes": len(seed_mean),
        "sigma_seed": sigma_seed,
        "sigma_box": sigma_box,
        "seed_noise_over_cosmic": (float(sigma_seed / sigma_box)
                                   if np.isfinite(sigma_box) and sigma_box > 0
                                   else float("nan")),
        "gap_hr": float(np.mean(hr_gap)) if hr_gap else float("nan"),
        "d_hr": d_hr,
        "gap_alpha1": float(np.mean(a1_gap)) if a1_gap else float("nan"),
        "d_alpha1": d_a1,
        "p_rank_alpha1": (_phi(abs(d_a1) / math.sqrt(2.0))
                          if np.isfinite(d_a1) else float("nan")),
        "p_rank_hr": (_phi(abs(d_hr) / math.sqrt(2.0))
                      if np.isfinite(d_hr) else float("nan")),
        "rho_boxes": float(np.mean(rhos)) if rhos else float("nan"),
        "rho_boxes_min": float(np.min(rhos)) if rhos else float("nan"),
        "alpha_monotone": (ladder_ok / ladder_n) if ladder_n else float("nan"),
        "n_alpha_ladders": ladder_n,
    }


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--candidates", default=None,
                    help="candidates/ root (default: <reward root>/sr2_direct/candidates)")
    ap.add_argument("--out", default=None, help="report path")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    model, _ = load_reward_models(cfg)

    root = Path(args.candidates) if args.candidates else direct_root("candidates")
    banner(f"reward stability scan over {root}")
    data = collect(root)
    if not data:
        raise SystemExit(f"no labelled candidates under {root}")

    stats, reliable, upper = make_statistics(model, cfg)
    report = {
        "candidates_root": str(root),
        "boxes": sorted(data),
        "reliable_host_bins": reliable,
        "upper_reliable_host_bins": upper,
        "statistics": {},
    }
    for name, fn in stats.items():
        vals: Dict[str, Dict[str, float]] = {}
        for box, cands in data.items():
            row = {}
            for tag, ens in cands.items():
                try:
                    row[tag] = float(fn(ens))
                except Exception:
                    row[tag] = float("nan")
            vals[box] = row
        report["statistics"][name] = {"metrics": evaluate(vals), "values": vals}

    out = (Path(args.out) if args.out
           else run_dir(args.run_name, "reward_stability_scan.json", create=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, report)

    hdr = (f"{'statistic':<28}{'sigma_seed':>11}{'d_alpha1':>10}{'p_rank':>8}"
           f"{'d_hr':>8}{'rho_box':>9}{'mono':>7}")
    print(hdr)
    print("-" * len(hdr))

    def key(n: str) -> float:
        d = report["statistics"][n]["metrics"]["d_alpha1"]
        return -abs(d) if np.isfinite(d) else 1.0
    for name in sorted(report["statistics"], key=key):
        m = report["statistics"][name]["metrics"]
        print(f"{name:<28}{m['sigma_seed']:>11.4g}{m['d_alpha1']:>10.2f}"
              f"{m['p_rank_alpha1']:>8.3f}{m['d_hr']:>8.2f}"
              f"{m['rho_boxes']:>9.3f}{m['alpha_monotone']:>7.2f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
