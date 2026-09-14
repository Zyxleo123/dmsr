#!/usr/bin/env python
"""How much of a host's subhalo radius DISTRIBUTION is host-specific and real?

The measurement behind docs/sr2_member_gather_training.md section 16b.1. Reads
`subhalo_radial/<box>/hr.npz` (built by scripts/features/subhalo_radial.py) and
writes nothing -- it prints. Seconds of numpy on ~100k rows, but submit it like
everything else rather than running it on the login node.

Variance decomposition of the per-host log-radius quantile function:
  total spread across hosts = universal(mass) + real host-specific + sampling noise
The split-half correlation is the decisive number: it is the fraction of the
per-host deviation that a second draw of the SAME host would reproduce, i.e. the
ceiling on what any predictor conditioned on that host can ever explain.
"""
import argparse
import os

import numpy as np

ROOT = os.environ.get("DMSR_REWARD_ROOT",
                      "/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--box", default="set9")
ap.add_argument("--npz", default="", help="override the input path")
args = ap.parse_args()
path = args.npz or os.path.join(ROOT, "subhalo_radial", args.box, "hr.npz")
print(f"reading {path}")

z = np.load(path)
hid, r, hnp = z['host_id'], z['r_over_rvir'], z['host_num_p']
ok = np.isfinite(r) & (r > 0)
hid, r, hnp = hid[ok], r[ok], hnp[ok]

U = np.array([0.1, 0.25, 0.5, 0.75, 0.9])           # rank fractions
rng = np.random.default_rng(0)

def qcurve(x):
    return np.log(np.quantile(x, U))

hosts, counts = np.unique(hid, return_counts=True)
for NMIN in (20, 50, 100):
    sel = hosts[counts >= NMIN]
    Q, QA, QB, M = [], [], [], []
    for h in sel:
        x = r[hid == h]
        p = rng.permutation(len(x)); a, b = x[p[:len(x)//2]], x[p[len(x)//2:]]
        Q.append(qcurve(x)); QA.append(qcurve(a)); QB.append(qcurve(b))
        M.append(np.log10(hnp[hid == h][0]))
    Q, QA, QB, M = map(np.asarray, (Q, QA, QB, M))
    print(f"\n=== hosts with >= {NMIN} subhalos: n={len(sel)}, median N={np.median(counts[counts>=NMIN]):.0f}")
    # universal part: regress each quantile on log host mass
    X = np.stack([np.ones_like(M), M], 1)
    for j, u in enumerate(U):
        beta, *_ = np.linalg.lstsq(X, Q[:, j], rcond=None)
        pred = X @ beta
        dev = Q[:, j] - pred                       # host-specific deviation
        # does a second draw of the same host reproduce that deviation?
        da = QA[:, j] - (X @ np.linalg.lstsq(X, QA[:, j], rcond=None)[0])
        db = QB[:, j] - (X @ np.linalg.lstsq(X, QB[:, j], rcond=None)[0])
        rho = np.corrcoef(da, db)[0, 1]
        # reliability-corrected: rho is the split-half corr; Spearman-Brown ->
        # reliability of the FULL sample estimate
        rel = 2 * rho / (1 + rho) if rho > -1 else np.nan
        print(f"  u={u:.2f}  sd(logr) across hosts {Q[:,j].std():.3f} | "
              f"R^2 from host mass {1-dev.var()/Q[:,j].var():5.3f} | "
              f"split-half rho {rho:+.3f} -> reliability {rel:+.3f} | "
              f"real host-specific sd {np.sqrt(max(rel,0))*dev.std():.3f} "
              f"noise sd {np.sqrt(max(1-max(rel,0),0))*dev.std():.3f}")
