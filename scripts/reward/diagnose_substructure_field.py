#!/usr/bin/env python
"""Isolate WHY the substructure candidate field destroys halos.

The full-box gate found the candidate field (Psi_SR2 + Pi(d)) makes almost no
halos (904 vs 213080 for base SR2). Field statistics show d is correctly scaled
and coherent, so this is not a units bug -- the suspicion is that adding a broad
conditional *sample* of the realization-dominated residual decorrelates SR2, and
that the velocity channel does most of the damage.

This is a channel-attribution control on one host, needing no model: it derives
d = final - sr2 from the two saved whole-box fields, crops a cube around the
cluster host, and runs the frozen Rockstar on four variants:

    sr2            frozen SR2 only (d=0)      -- must be healthy, else regen bug
    sr2+d          the full candidate         -- must reproduce the collapse
    sr2+d_disp     add only displacement d    -- keep SR2's velocity
    sr2+d_vel      add only velocity d        -- keep SR2's displacement

If sr2+d_disp is healthy and sr2+d_vel collapses, the velocity residual is the
culprit and the fix is to not add a sampled velocity. Region crop under a
mini-box (like the in-loop eval), so counts are relative, not absolute.

    python scripts/reward/diagnose_substructure_field.py --box set8 \
        --tag substructure_set8 --region-sites 192
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.eval.rockstar import run_rockstar_on_field  # noqa: E402
from cosmo_sr.features.lagrangian_host import LagrangianHostFeatures  # noqa: E402


def reward_root() -> Path:
    import os
    return Path(os.environ.get(
        "DMSR_REWARD_ROOT", "/zfsauton/scratch/yixiz/DMSR/dmsr_reward"))


def crop_around(field, centre_site, m):
    sh = tuple(m // 2 - int(c) for c in centre_site)
    rolled = np.roll(field, shift=sh, axis=(1, 2, 3))
    return np.ascontiguousarray(rolled[:, :m, :m, :m])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", default="set8")
    ap.add_argument("--tag", default="substructure_set8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host-id", type=int, default=717)
    ap.add_argument("--region-sites", type=int, default=192)
    args = ap.parse_args()

    rr = reward_root()
    final = np.load(rr / "flow_rockstar" / "fields"
                    / f"{args.box}__{args.tag}__seed{args.seed}.npy", mmap_mode="r")
    sr2 = np.load(rr / "moment_target" / args.box / f"{args.box}_sr2_box.npy",
                  mmap_mode="r")
    feat = LagrangianHostFeatures.from_npz(
        str(rr / "lagrangian_host" / args.box / f"{args.box}_lagrangian_host.npz"))

    row = feat.table.row_of(args.host_id)
    if row < 0:
        row = int(np.argmax(feat.table.mvir))
    cell = feat.grid.boxsize_mpc_h / feat.grid.ng_hr
    centre = tuple(int(np.floor(c / cell)) % feat.grid.ng_hr
                   for c in feat.table.center_lag[row])
    m = int(args.region_sites)
    box_kpc = m * cell * 1000.0
    print(f"host row {row} (id {int(feat.table.host_id[row])}, "
          f"logM {np.log10(feat.table.mvir[row]):.2f}); cube {m}^3 "
          f"= {m * cell:.1f} Mpc/h centred at {centre}", flush=True)

    sr2c = crop_around(np.asarray(sr2, np.float32), centre, m)
    finc = crop_around(np.asarray(final, np.float32), centre, m)
    d = finc - sr2c

    variants = {
        "sr2 (d=0)": sr2c,
        "sr2+d (full)": finc,
        "sr2+d_disp": np.concatenate([sr2c[0:3] + d[0:3], sr2c[3:6]], axis=0),
        "sr2+d_vel": np.concatenate([sr2c[0:3], sr2c[3:6] + d[3:6]], axis=0),
    }
    work = rr / "flow_rockstar" / "diag" / f"{args.box}__{args.tag}"
    print(f"\n{'variant':<16}{'halos':>8}{'hosts':>8}{'subs':>8}", flush=True)
    print("-" * 40, flush=True)
    for name, fld in variants.items():
        tag = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        cat = run_rockstar_on_field(np.ascontiguousarray(fld, np.float32),
                                    work / tag, tag=tag, boxsize_kpc_h=box_kpc,
                                    overwrite=True)
        print(f"{name:<16}{cat.n:>8}{cat.hosts().n:>8}{cat.subhalos().n:>8}",
              flush=True)
    print(f"\nwork -> {work}", flush=True)


if __name__ == "__main__":
    main()
