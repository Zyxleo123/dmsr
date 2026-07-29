#!/usr/bin/env python
"""Stage 2 — nested best-of-N oracle TTS ceiling for subhalo recovery.

Reads Stage-1 ``match_rows.jsonl`` / ``halo_rows.jsonl`` (and optional field rows)
produced with nested seeds ``0 .. N_max-1``. For each box and each
``N ∈ {1,2,4,...,N_max}``:

* diversity: across-seed spread of n_subs, SHMF, one-halo, field metrics
* global oracle: pick the seed with best *oracle objective* (not the reported metric)
* host oracle: fraction of matched hosts for which some seed improves N_sub
* coverage: fraction of HR subhalos recovered by ≥1 of the N candidates

Gate A is written to ``gate_a.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _nested_ns(n_max: int) -> List[int]:
    n, out = 1, []
    while n <= n_max:
        out.append(n)
        n *= 2
    return out


def _oracle_objective(match_row: dict) -> float:
    """Lower is better. Separate from reported metrics (coverage / SHMF).

    Uses: fraction missing + 0.5*fraction merged — not the coverage number we report.
    """
    cc = match_row.get("class_counts", {})
    n = max(int(match_row.get("n_hr_subs_classified", 0)), 1)
    missing = (
        cc.get("missing", 0) + cc.get("merged_into_host", 0)
        + cc.get("absent_peak", 0) + cc.get("diffuse_peak", 0)
    )
    biased = (
        cc.get("recovered_biased", 0) + cc.get("spatially_shifted", 0)
        + cc.get("velocity_incoherent", 0)
    )
    return float(missing + 0.5 * biased) / n


def _coverage(match_rows: List[dict]) -> float:
    """Fraction of HR subhalos recovered by ≥1 candidate.

    Prefer per-object ``records`` when present; else fall back to class_counts
    (union bound unavailable → report mean recovered fraction across seeds).
    """
    recovered = set()
    all_ids = set()
    has_records = False
    for row in match_rows:
        recs = row.get("records") or []
        if recs:
            has_records = True
        for rec in recs:
            hid = rec["hr_id"]
            all_ids.add(hid)
            if rec["class"] in (
                "recovered", "recovered_biased", "spatially_shifted",
                "velocity_incoherent",
            ):
                recovered.add(hid)
    if has_records:
        return float(len(recovered) / max(len(all_ids), 1))
    # No records: approximate with mean (recovered+biased+shifted)/n per seed.
    fracs = []
    for row in match_rows:
        n = max(int(row.get("n_hr_subs_classified", 0)), 1)
        cc = row.get("class_counts", {})
        hit = (
            cc.get("recovered", 0) + cc.get("recovered_biased", 0)
            + cc.get("spatially_shifted", 0) + cc.get("velocity_incoherent", 0)
        )
        fracs.append(hit / n)
    return float(np.mean(fracs)) if fracs else 0.0


def _class_fraction(match_rows: List[dict], label: str) -> float:
    tot = sum(int(r.get("n_hr_subs_classified", 0)) for r in match_rows)
    hit = sum(int(r.get("class_counts", {}).get(label, 0)) for r in match_rows)
    return float(hit / max(tot, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1", default=str(ROOT / "runs/sr2_baseline/stage1"))
    ap.add_argument("--out", default=str(ROOT / "runs/sr2_baseline/stage2"))
    ap.add_argument("--n-max", type=int, default=32)
    args = ap.parse_args()

    stage1 = Path(args.stage1)
    match_rows = _load_jsonl(stage1 / "match_rows.jsonl")
    halo_rows = _load_jsonl(stage1 / "halo_rows.jsonl")
    field_rows = _load_jsonl(stage1 / "field_rows.jsonl")
    if not match_rows:
        raise SystemExit(f"No match_rows.jsonl in {stage1}; run Stage 1 first.")

    by_box: Dict[str, List[dict]] = defaultdict(list)
    for r in match_rows:
        by_box[r["box"]].append(r)
    for box in by_box:
        by_box[box].sort(key=lambda r: int(r["seed"]))

    # Halo / field indexed by (box, seed)
    halo_sr = {(r["box"], int(r["seed"])): r for r in halo_rows if r.get("tag") == "sr"}
    field_ix = {(r["box"], int(r["seed"])): r for r in field_rows}

    ns = _nested_ns(args.n_max)
    curves = []
    for n in ns:
        per_box = []
        for box, rows in by_box.items():
            rows_n = [r for r in rows if int(r["seed"]) < n]
            if not rows_n:
                continue
            # Ensure nested: seeds 0..n-1 present
            seeds = sorted(int(r["seed"]) for r in rows_n)
            if seeds != list(range(min(n, max(seeds) + 1))):
                # still proceed but flag
                nested = seeds == list(range(len(seeds))) and seeds[0] == 0
            else:
                nested = True

            objs = [_oracle_objective(r) for r in rows_n]
            best_i = int(np.argmin(objs))
            best = rows_n[best_i]
            cov = _coverage(rows_n)
            # Host oracle: hosts where min missing-across-seeds beats seed-0
            # Approximate via class_counts on recovered fraction per seed.
            rec_fracs = [
                1.0 - (r.get("class_counts", {}).get("missing", 0)
                       + r.get("class_counts", {}).get("merged_into_host", 0))
                / max(r.get("n_hr_subs_classified", 1), 1)
                for r in rows_n
            ]
            n_subs = [halo_sr.get((box, int(r["seed"])), {}).get("n_subs", float("nan"))
                      for r in rows_n]
            dens_sig = [field_ix.get((box, int(r["seed"])), {}).get("density_sigma_ratio",
                                                                     float("nan"))
                        for r in rows_n]
            per_box.append({
                "box": box,
                "n": n,
                "nested_ok": nested,
                "oracle_objective": float(objs[best_i]),
                "oracle_seed": int(best["seed"]),
                "coverage": cov,
                "recovered_frac_best": float(max(rec_fracs)),
                "recovered_frac_seed0": float(rec_fracs[0]),
                "n_subs_mean": float(np.nanmean(n_subs)),
                "n_subs_std": float(np.nanstd(n_subs)),
                "density_sigma_std": float(np.nanstd(dens_sig)),
                "missing_frac_mean": _class_fraction(rows_n, "missing"),
            })

        if not per_box:
            continue
        # Box-level means
        def mean_key(k):
            return float(np.mean([p[k] for p in per_box]))

        curves.append({
            "n": n,
            "n_boxes": len(per_box),
            "oracle_objective_mean": mean_key("oracle_objective"),
            "coverage_mean": mean_key("coverage"),
            "recovered_frac_best_mean": mean_key("recovered_frac_best"),
            "recovered_frac_seed0_mean": mean_key("recovered_frac_seed0"),
            "n_subs_std_mean": mean_key("n_subs_std"),
            "density_sigma_std_mean": mean_key("density_sigma_std"),
            "per_box": per_box,
        })

    # Gate A
    if len(curves) >= 2:
        obj = [c["oracle_objective_mean"] for c in curves]
        cov = [c["coverage_mean"] for c in curves]
        # Monotonic improvement: objective decreases, coverage increases
        mono_obj = all(obj[i] >= obj[i + 1] - 1e-6 for i in range(len(obj) - 1))
        mono_cov = all(cov[i] <= cov[i + 1] + 1e-6 for i in range(len(cov) - 1))
        gain = float(cov[-1] - cov[0])
        diversity = float(np.mean([c["n_subs_std_mean"] for c in curves]))
        if gain >= 0.05 and (mono_cov or mono_obj):
            decision = "pass_tts"
            reason = (
                f"coverage gain {gain:.3f} from N={curves[0]['n']} to N={curves[-1]['n']}; "
                "proceed to practical TTS scoring"
            )
        elif diversity < 1e-3:
            decision = "fail_noise_ignore"
            reason = "samples barely differ (n_subs std ~ 0); prioritise stochastic refinement"
        else:
            decision = "fail_unhelpful_diversity"
            reason = (
                "samples differ but oracle subhalo quality does not improve enough; "
                "improve training before building a selector"
            )
    else:
        decision, reason, gain, diversity = "incomplete", "need >=2 N values", 0.0, 0.0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "oracle_curves.json", "w") as fh:
        json.dump({"curves": curves}, fh, indent=2)
    gate = {
        "decision": decision,
        "reason": reason,
        "coverage_gain_Nmax_minus_N1": gain,
        "mean_n_subs_std": diversity,
        "n_max": args.n_max,
        "note": (
            "Not calibrated conditional uncertainty — one HR realisation per LR. "
            "Report as diversity / conditional coverage only."
        ),
    }
    with open(out / "gate_a.json", "w") as fh:
        json.dump(gate, fh, indent=2)

    print(json.dumps(gate, indent=2))
    print(f"Wrote {out}/oracle_curves.json and gate_a.json")


if __name__ == "__main__":
    main()
