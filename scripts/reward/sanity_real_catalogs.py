#!/usr/bin/env python
"""Reward sanity check on the real catalogs that already exist on scratch.

The full reward model needs HR catalogs for every training box, which needs the
CPU halo-finder jobs. Before spending that, this measures the two things the
reward has to get right, using only the set12 catalogs the SR2 subhalo study
already produced (one HR run and nine frozen-SR2 seeds):

1. the summary vector really does register the low-mass subhalo deficit;
2. the SR2-vs-HR Mahalanobis distance is large compared with the SR2 seed-to-seed
   scatter -- if it were not, the reward could not tell a good sample from a bad
   one and the whole plan is unfounded.

Whole-box catalogs, no chunk attribution, so this is independent of the purity
grid. Writes a small JSON summary; nothing large.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import add_common_args, load_reward_config

from cosmo_sr.eval.rockstar import load_rockstar_ascii
from cosmo_sr.reward.catalog import CatalogBins, EnsembleSummary, load_bins, summary_vector

STAGE1 = Path("/zfsauton/scratch/yixiz/DMSR/sr2_baseline/stage1/halos")


def whole_box_summary(path: Path, bins: CatalogBins, volume: float) -> EnsembleSummary:
    """Pool a whole box as a single 'chunk': every resolved halo counts."""
    cat = load_rockstar_ascii(path)
    parent = np.asarray(cat.parent_ids, dtype=np.int64)
    mvir = np.asarray(cat.mvir, dtype=np.float64)
    num_p = np.asarray(cat.num_p, dtype=np.int64)

    is_host = parent < 0
    host_ok = is_host & (num_p >= bins.min_host_particles)
    sub_ok = (~is_host) & (num_p >= bins.min_sub_particles)

    ids = np.asarray(cat.ids, dtype=np.int64)
    host_row = {int(i): r for r, i in enumerate(ids) if host_ok[r]}
    rows = np.nonzero(sub_ok)[0]
    keep = np.asarray([r for r in rows if int(parent[r]) in host_row], dtype=np.int64)
    hosts_of = np.asarray([host_row[int(parent[r])] for r in keep], dtype=np.int64)

    n_sub, _ = np.histogram(mvir[keep] if keep.size else np.zeros(0),
                            bins=np.asarray(bins.sub_mass_edges))
    n_host, _ = np.histogram(mvir[np.nonzero(host_ok)[0]],
                             bins=np.asarray(bins.host_mass_edges))
    occ_num, _ = np.histogram(mvir[hosts_of] if hosts_of.size else np.zeros(0),
                              bins=np.asarray(bins.host_mass_edges))
    return EnsembleSummary(n_sub.astype(float), n_host.astype(float),
                           occ_num.astype(float), volume, members=(path.parent.name,))


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--box", default="set12")
    ap.add_argument("--out", default="runs/reward/real_catalog_sanity.json")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    bins = load_bins(cfg.get("catalog", {}))
    volume = float(cfg["data"]["boxsize_mpc_h"]) ** 3

    root = STAGE1 / args.box
    hr_path = root / "hr" / "hr_rockstar" / "halos_0.0.ascii"
    if not hr_path.is_file():
        raise FileNotFoundError(f"no HR catalog at {hr_path}")
    sr_paths = sorted(root.glob("sr_seed*/sr*_rockstar/halos_0.0.ascii"))
    if len(sr_paths) < 2:
        raise FileNotFoundError(f"need >=2 SR2 seeds under {root}, found {len(sr_paths)}")

    hr = whole_box_summary(hr_path, bins, volume)
    srs = [whole_box_summary(p, bins, volume) for p in sr_paths]

    s_hr, _ = summary_vector(hr, bins)
    s_sr = np.stack([summary_vector(s, bins, empty_fill=s_hr)[0] for s in srs])

    # Covariance from the SR2 seed scatter: the only multi-realisation scatter
    # available before the HR catalog jobs run. It is the *generator's* noise, not
    # cosmic variance, so treat the resulting distance as indicative only.
    mu_sr = s_sr.mean(axis=0)
    cov = np.cov(s_sr, rowvar=False)
    lam = float(cfg["reward"].get("shrinkage", 0.1)) * float(np.mean(np.diag(cov)))
    creg = cov + lam * np.eye(cov.shape[0])
    d = s_hr - mu_sr
    maha = float(d @ np.linalg.solve(creg, d))
    per_bin = d * np.linalg.solve(creg, d)

    labels = bins.labels()
    sub_ratio = (np.mean([s.n_sub for s in srs], axis=0)
                 / np.maximum(hr.n_sub, 1e-30))
    occ_ratio = (np.mean([s.occupation() for s in srs], axis=0)
                 / np.maximum(hr.occupation(), 1e-30))

    report = {
        "box": args.box,
        "n_sr_seeds": len(srs),
        "labels": labels,
        "hr_n_sub": hr.n_sub.astype(int).tolist(),
        "sr_n_sub_mean": np.mean([s.n_sub for s in srs], axis=0).tolist(),
        "sub_count_ratio_sr_over_hr": sub_ratio.tolist(),
        "hr_occupation": hr.occupation().tolist(),
        "sr_occupation_mean": np.mean([s.occupation() for s in srs], axis=0).tolist(),
        "occupation_ratio_sr_over_hr": occ_ratio.tolist(),
        "hr_n_host": hr.n_host.astype(int).tolist(),
        "mahalanobis_hr_vs_sr_mean": maha,
        "per_bin_contribution": dict(zip(labels, per_bin.tolist())),
        "cov_condition_number": float(np.linalg.cond(creg)),
        "shrinkage_lambda": lam,
        "note": "covariance is SR2 seed scatter, not HR cosmic variance; "
                "indicative only until the HR catalog jobs have run",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"box {args.box}: HR vs mean of {len(srs)} SR2 seeds")
    print(f"{'bin':32s} {'HR':>10s} {'SR2':>10s} {'SR2/HR':>8s}")
    for k in range(bins.n_sub_bins):
        print(f"{labels[k]:32s} {hr.n_sub[k]:10.0f} "
              f"{np.mean([s.n_sub[k] for s in srs]):10.1f} {sub_ratio[k]:8.3f}")
    for i in range(bins.n_host_bins):
        k = bins.n_sub_bins + i
        print(f"{labels[k]:32s} {hr.occupation()[i]:10.3f} "
              f"{np.mean([s.occupation()[i] for s in srs]):10.3f} {occ_ratio[i]:8.3f}")
    print(f"\nMahalanobis(HR, SR2 mean) = {maha:.1f} over {len(labels)} bins")
    print(f"cond(C_reg) = {report['cov_condition_number']:.3g}, lambda = {lam:.4g}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
