#!/usr/bin/env python
"""Audit: is the fitted HR covariance a real 11-direction reward, or one bin?

The reward is ``-(s - mu)^T C_reg^{-1} (s - mu)``. If ``C_reg`` is badly
conditioned, its inverse concentrates almost all weight on the least-constrained
direction, and an optimiser will find that direction and ignore the physics.
The sanity check on ``set12`` already showed what this looks like in practice:
the SR2 *seed-scatter* covariance gives a Mahalanobis distance of 98007 with 88%
of it in three bins. **Seed scatter is the generator's own noise, not cosmic
variance, and is never a valid reward covariance** -- this script refuses to
accept one.

Reported, for the covariance fitted from HR chunk summaries:

* ``cond(C_reg)`` after shrinkage, and ``cond(C)`` before it;
* the eigenvalue spectrum and each eigenvector's dominant bins;
* leave-one-box-out stability of ``mu``, of the eigenvalues, and of the reward
  ordering (Spearman) -- the test that matters, because a covariance that flips
  its principal direction when one box is dropped is fitting the boxes, not the
  cosmology;
* per-bin contributions to ``D^2`` for the frozen SR2 baseline, i.e. which bins
  the reward will actually push on;
* the same with and without the sparse ``1e14 Msun/h`` host bin.

    python scripts/reward/audit_reward_covariance.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from _common import (add_common_args, banner, bins_of, load_reward_config,
                     write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import ChunkSummary, pool, read_summaries, summary_vector
from cosmo_sr.reward.reward import RewardModel, fit_reward_model


def _drop_bins(chunks: Sequence[ChunkSummary], bins, drop_host: Sequence[int]):
    """Rebuild summaries and bins with some host-mass bins removed.

    Dropping a host bin is not the same as zeroing it: the bin has to leave the
    summary vector entirely, or its (large) variance still shapes the whitening
    of every other bin through the off-diagonals.
    """
    from dataclasses import replace

    keep = [i for i in range(bins.n_host_bins) if i not in set(drop_host)]
    if len(keep) == bins.n_host_bins:
        return list(chunks), bins
    # Host edges must stay contiguous for CatalogBins; the audit only needs the
    # count vectors to line up, so edges are rebuilt from the kept bins' lower
    # edges plus the last kept upper edge.
    edges = tuple([bins.host_mass_edges[i] for i in keep]
                  + [bins.host_mass_edges[keep[-1] + 1]])
    nb = replace(bins, host_mass_edges=edges)
    out = []
    for c in chunks:
        out.append(ChunkSummary(
            box=c.box, chunk_id=c.chunk_id, source=c.source,
            n_sub=np.asarray(c.n_sub).copy(),
            n_host=np.asarray(c.n_host)[keep].copy(),
            occ_numerator=np.asarray(c.occ_numerator)[keep].copy(),
            volume_mpc3=c.volume_mpc3, n_sub_total=c.n_sub_total,
            n_host_total=c.n_host_total, meta=dict(c.meta),
        ))
    return out, nb


def _spectrum(model: RewardModel) -> Dict:
    w, v = np.linalg.eigh(model.cov_reg)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    labels = list(model.labels or model.bins.labels())
    modes = []
    for k in range(len(w)):
        load = np.abs(v[:, k])
        top = np.argsort(load)[::-1][:3]
        modes.append({
            "index": int(k),
            "eigenvalue": float(w[k]),
            "variance_fraction": float(w[k] / np.sum(w)),
            "dominant_bins": [labels[int(t)] for t in top],
            "loadings": [float(v[int(t), k]) for t in top],
        })
    return {
        "eigenvalues": [float(x) for x in w],
        "cond_cov_reg": float(model.condition_number),
        "cond_cov_raw": float(np.linalg.cond(model.cov)),
        "lambda": float(model.lam),
        "lambda_over_mean_diag": float(model.lam / max(np.mean(np.diag(model.cov)), 1e-300)),
        "modes": modes,
    }


def _fit(chunks, bins, rcfg, seed_offset: int = 0) -> RewardModel:
    return fit_reward_model(
        chunks, bins,
        ensemble_size=int(rcfg.get("ensemble_size_B", 16)),
        n_draws=int(rcfg.get("bootstrap_draws", 400)),
        shrinkage=float(rcfg.get("shrinkage", 0.1)),
        seed=int(rcfg.get("bootstrap_seed", 0)) + seed_offset,
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else float("nan")


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--summaries-dir", default=None,
                    help="directory of *__hr__hr.jsonl (default: the catalog cache)")
    ap.add_argument("--base-summaries", default=None,
                    help="frozen-SR2 chunk summaries, for the per-bin "
                         "contribution table; optional")
    ap.add_argument("--drop-host-bins", default="4",
                    help="comma-separated host bins for the 'without' variant "
                         "(default 4 = the sparse 1e14 bin)")
    ap.add_argument("--max-cond", type=float, default=1e4,
                    help="cond(C_reg) above this fails the audit")
    ap.add_argument("--max-bin-share", type=float, default=0.5,
                    help="fraction of D^2 a single bin may contribute")
    ap.add_argument("--out", default="runs/reward/covariance_audit.json")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    rcfg = cfg.get("reward", {})

    src = Path(args.summaries_dir) if args.summaries_dir else paths.CATALOG_CACHE()
    files = sorted(Path(src).glob("*__hr__hr.jsonl"))
    if not files:
        raise SystemExit(
            f"no HR chunk summaries in {src}. Submit "
            "`sbatch --array=0-15 scripts/slurm/hr_catalog_summaries_cpu.sbatch` "
            "first. This audit never falls back to SR2 seed scatter: seed "
            "scatter is generator noise, not cosmic variance, and would give a "
            "degenerate one-direction reward."
        )
    chunks: List[ChunkSummary] = []
    for f in files:
        chunks.extend(read_summaries(f))
    bad = sorted({c.source for c in chunks} - {"hr"})
    if bad:
        raise SystemExit(f"non-HR sources in the covariance fit: {bad}")
    boxes = sorted({c.box for c in chunks})

    drop = [int(x) for x in args.drop_host_bins.split(",") if x.strip() != ""]
    variants = {"with_sparse_bin": ([], bins), "without_sparse_bin": (drop, None)}

    result = {"boxes": boxes, "n_chunks": len(chunks), "variants": {}}
    for name, (dropped, _) in variants.items():
        ch, bb = _drop_bins(chunks, bins, dropped)
        model = _fit(ch, bb, rcfg)
        entry = {"dropped_host_bins": dropped, "spectrum": _spectrum(model),
                 "mu": model.mu.tolist(), "labels": list(model.labels)}

        # --- per-bin contributions for the frozen baseline -------------------
        if args.base_summaries:
            base = []
            for f in sorted(Path(args.base_summaries).glob("*.jsonl")):
                base.extend(read_summaries(f))
            base, _ = _drop_bins(base, bins, dropped)
            comp = model.components(pool(base))
            d2 = comp["mahalanobis2"]
            shares = {k: (v / d2 if d2 else float("nan"))
                      for k, v in comp.items() if k.startswith("contrib_")}
            entry["baseline_components"] = comp
            entry["baseline_bin_shares"] = shares
            entry["baseline_max_bin_share"] = float(max(shares.values())) if shares else float("nan")
            entry["baseline_negative_bins"] = [k for k, v in shares.items() if v < 0]
            entry["sub_rewards_baseline"] = model.scores(
                pool(base), rcfg.get("occupation", {}).get("reliable_host_bins"))

        # --- leave-one-box-out ------------------------------------------------
        lobo = []
        if len(boxes) >= 3:
            ref_vecs = []
            for idx in range(200):
                rng = np.random.default_rng(1000 + idx)
                sel = [ch[i] for i in rng.choice(len(ch), size=model.ensemble_size)]
                s, _ = summary_vector(pool(sel), bb)
                ref_vecs.append(s)
            ref_vecs = np.asarray(ref_vecs)
            ref_r = np.asarray([-(v - model.mu) @ model.precision @ (v - model.mu)
                                for v in ref_vecs])
            for b in boxes:
                sub = [c for c in ch if c.box != b]
                m = _fit(sub, bb, rcfg)
                rr = np.asarray([-(v - m.mu) @ m.precision @ (v - m.mu)
                                 for v in ref_vecs])
                lobo.append({
                    "held_out_box": b,
                    "cond_cov_reg": float(m.condition_number),
                    "mu_max_abs_shift": float(np.max(np.abs(m.mu - model.mu))),
                    "mu_max_shift_in_sigma": float(np.max(
                        np.abs(m.mu - model.mu) / np.sqrt(np.diag(model.cov_reg)))),
                    "top_eigenvalue_ratio": float(
                        np.linalg.eigvalsh(m.cov_reg)[-1]
                        / np.linalg.eigvalsh(model.cov_reg)[-1]),
                    "reward_rank_spearman": _spearman(ref_r, rr),
                })
            entry["leave_one_box_out"] = lobo
            entry["lobo_min_rank_spearman"] = float(
                min(x["reward_rank_spearman"] for x in lobo))
            entry["lobo_max_mu_shift_sigma"] = float(
                max(x["mu_max_shift_in_sigma"] for x in lobo))
        else:
            entry["leave_one_box_out"] = None
            entry["lobo_skipped_reason"] = (
                f"{len(boxes)} box(es) with HR summaries; leave-one-box-out needs "
                ">= 3 to say anything. Run the HR catalog array job."
            )
        result["variants"][name] = entry

    # --- verdict -------------------------------------------------------------
    ver = {}
    for name, e in result["variants"].items():
        cond = e["spectrum"]["cond_cov_reg"]
        share = e.get("baseline_max_bin_share", float("nan"))
        checks = {
            "cond_ok": bool(cond <= args.max_cond),
            "bin_share_ok": bool(not np.isfinite(share) or share <= args.max_bin_share),
            "lobo_ok": bool(e.get("lobo_min_rank_spearman", np.nan) >= 0.9)
            if e.get("leave_one_box_out") else None,
        }
        ver[name] = {"cond_cov_reg": cond, "baseline_max_bin_share": share,
                     "checks": checks,
                     "pass": bool(checks["cond_ok"] and checks["bin_share_ok"]
                                  and checks["lobo_ok"] is not False)}
    with_, without = ver["with_sparse_bin"], ver["without_sparse_bin"]
    ver["recommendation"] = (
        "keep_sparse_bin_in_reward" if with_["pass"]
        else ("drop_sparse_bin_to_evaluation_only" if without["pass"]
              else "covariance_unusable_as_fitted")
    )
    result["verdict"] = ver

    write_json(Path(args.out), result)
    banner("Reward covariance audit")
    for name, e in result["variants"].items():
        s = e["spectrum"]
        print(f"  {name}: cond(C_reg)={s['cond_cov_reg']:.4g} "
              f"(raw {s['cond_cov_raw']:.4g}), lambda={s['lambda']:.4g} "
              f"= {s['lambda_over_mean_diag']:.3f} * mean(diag C)")
        print(f"      top mode {s['modes'][0]['variance_fraction']:.1%} of variance, "
              f"dominated by {', '.join(s['modes'][0]['dominant_bins'])}")
        if e.get("lobo_min_rank_spearman") is not None:
            print(f"      LOBO: min reward-rank Spearman "
                  f"{e['lobo_min_rank_spearman']:.3f}, max mu shift "
                  f"{e['lobo_max_mu_shift_sigma']:.2f} sigma")
        elif e.get("lobo_skipped_reason"):
            print(f"      LOBO: skipped -- {e['lobo_skipped_reason']}")
    print(f"  recommendation: {ver['recommendation']}")
    print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
