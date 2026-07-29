"""Does adding LR-only boxes at z != 0 actually move the environment distribution?

docs/dmsr_stage.md 9.0b established that the 349 same-cosmology, same-epoch
LR-only boxes are indistinguishable from the paired pool (AUC 0.506/0.500/0.490,
box-level control 0.454). This re-measures the same quantity for LR fields
harvested at higher redshift by ~/slurm/dmsr/run_lr_sim_multiz.sh.

TWO THINGS THAT WOULD OTHERWISE MAKE THE NUMBER MEANINGLESS:

1. `redshift` is DESCRIPTOR_NAMES[0] and is constant within each pool, so it is a
   perfect source label and the classifier scores AUC = 1.0 by construction. It is
   zeroed here so the standardizer drops it, and the AUC is over FIELD-DERIVED
   descriptors only -- that is the question we actually care about.

2. `preproc.py` applies cosmology.disnorm/velnorm at the snapshot's own redshift,
   dividing out linear growth D(z). So a z=2 field is not "the z=0 field, fainter";
   the linear amplitude is gone and what remains is the nonlinear difference. Any
   AUC above the control is therefore evidence of genuinely different NONLINEAR
   structure, which is the axis the critic can use.

Read the result against the 0.454 box-level control, not against 0.50: the paired
side is only 3 boxes, so a classifier can key on box idiosyncrasy in both halves.

Usage:
    cd cosmo_sr_project && python scripts/_multiz_auc.py
    python scripts/_multiz_auc.py --z 0 1 2 --n-crops 2048 --seeds 3
"""
import argparse
import glob
import sys

sys.path.insert(0, "src")

import numpy as np

from cosmo_sr.utils.config import load_config
from cosmo_sr.dmsr.data import LRCropPool, resolve_split, build_balanced_sampler
from cosmo_sr.dmsr.density import cellsizes
from cosmo_sr.dmsr.env import (
    DESCRIPTOR_NAMES,
    DescriptorStandardizer,
    source_classifier_auc,
)

REDSHIFT_IDX = DESCRIPTOR_NAMES.index("redshift")


def build_pool(paths, n_crops, seed, data_cfg, descriptor_cfg, redshift):
    """LRCropPool carries ONE redshift for the whole pool, which is why each z
    gets its own pool rather than one mixed pool."""
    dk = dict(descriptor_cfg)
    dk["redshift"] = float(redshift)
    return LRCropPool(
        paths,
        n_crops=int(n_crops),
        seed=int(seed),
        crop_lr=int(data_cfg["crop_lr"]),
        channels=int(data_cfg["channels"]),
        use_channels=data_cfg.get("use_channels"),
        mmap=bool(data_cfg["mmap"]),
        descriptor_kwargs=dk,
    )


def drop_redshift(d):
    """Zero the redshift column so DescriptorStandardizer drops it as constant."""
    out = np.array(d, copy=True)
    out[:, REDSHIFT_IDX] = 0.0
    return out


def measure(paired_pool, unpaired_pool, n_dims, n_bins, seed):
    pd_ = drop_redshift(paired_pool.descriptors())
    ud_ = drop_redshift(unpaired_pool.descriptors())
    std = DescriptorStandardizer.fit(pd_)
    auc_before = source_classifier_auc(std.transform(pd_), std.transform(ud_), seed=seed)
    sampler, _ = build_balanced_sampler(
        paired_pool, unpaired_pool, n_dims=int(n_dims), n_bins=int(n_bins), seed=seed
    )
    rep = sampler.report()
    return auc_before, rep, std, pd_, ud_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dmsr/stage_d_critic_alllr.yaml")
    ap.add_argument("--z", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])
    ap.add_argument("--n-crops", type=int, default=None, help="paired crops (default: env.pool_size)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--lr-only-tmpl",
        default="/zfsauton/scratch/yixiz/DMSR/lr_sims/set*/catnorm_z{z:g}.npy",
        help="glob template for the multi-z fields; {z:g} is the redshift",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, e = cfg["data"], cfg["env"]
    split = resolve_split(d)
    n = int(args.n_crops or e["pool_size"])

    # The config leaves descriptor.cellsize null and train_dmsr.py fills it from the
    # box geometry (docs 9.0i: a hand-written 15625.0 was 10x too big at this site).
    # Passing the config block through unmodified would silently reinstate that bug,
    # because environment_descriptors defaults to exactly 15625.0.
    _, lr_cellsize = cellsizes(d, int(cfg.get("factor", 8)))
    desc = {**e["descriptor"], "cellsize": lr_cellsize}

    print(f"config          : {args.config}")
    print(f"paired boxes    : {len(split.train_lr)} (all z=0)")
    print(f"crops           : {n} paired / {2 * n} unpaired, {args.seeds} seed(s)")
    print(f"descriptor      : use_tidal={desc.get('use_tidal')} cellsize={desc.get('cellsize')}")
    print("AUC excludes the `redshift` descriptor (constant per pool = perfect label).")
    print()

    # --- box-level control: paired box 0 vs paired boxes 1,2 -----------------
    # This is what "chance" looks like when both halves are the same physics but
    # different boxes. docs 9.0b measured 0.454.
    if len(split.train_lr) >= 3:
        ctrl_a = build_pool(split.train_lr[:1], n, 0, d, desc, 0.0)
        ctrl_b = build_pool(split.train_lr[1:3], 2 * n, 1, d, desc, 0.0)
        ctrl_auc, _, _, _, _ = measure(ctrl_a, ctrl_b, e["n_dims"], e["n_bins"], 0)
        print(f"CONTROL  paired-box-0 vs paired-boxes-1,2 : AUC = {ctrl_auc:.4f}")
        print("  (reference scale for 'indistinguishable'; docs 9.0b got 0.454)")
        print()

    rows = []
    for z in args.z:
        pattern = args.lr_only_tmpl.format(z=z)
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"z={z:g}: NO FILES matching {pattern} -- run run_lr_sim_multiz.sh first")
            print()
            continue

        aucs_before, aucs_after, supports = [], [], []
        kept, pd_last, ud_last = None, None, None
        for s in range(args.seeds):
            paired = build_pool(split.train_lr, n, 100 + s, d, desc, 0.0)
            unpaired = build_pool(paths, 2 * n, 200 + s, d, desc, z)
            ab, rep, std, pd_, ud_ = measure(paired, unpaired, e["n_dims"], e["n_bins"], s)
            aucs_before.append(ab)
            aucs_after.append(rep.auc_after)
            supports.append(rep.n_in_support / max(rep.n_unpaired, 1))
            kept, pd_last, ud_last = std.kept_names, pd_, ud_

        print(f"=== z = {z:g}  ({len(paths)} boxes) ===")
        print(f"  kept descriptors : {kept}")
        print(f"  AUC before       : {np.mean(aucs_before):.4f} "
              f"({', '.join(f'{v:.4f}' for v in aucs_before)})")
        print(f"  AUC after        : {np.mean(aucs_after):.4f} "
              f"({', '.join(f'{v:.4f}' for v in aucs_after)})")
        print(f"  in paired support: {np.mean(supports) * 100:.1f}%")
        for nm in ("var_density", "disp_rms", "tidal_I2", "tidal_I3"):
            i = DESCRIPTOR_NAMES.index(nm)
            pm, um = pd_last[:, i].mean(), ud_last[:, i].mean()
            ps = pd_last[:, i].std() + 1e-30
            print(f"    {nm:12s} paired={pm:+.4g}  z={z:g}={um:+.4g}  "
                  f"shift={abs(um - pm) / ps:.2f} sd")
        print()
        rows.append((z, np.mean(aucs_before), np.mean(aucs_after), np.mean(supports)))

    if rows:
        print("SUMMARY")
        print(f"{'z':>5} {'AUC before':>11} {'AUC after':>10} {'in support':>11}")
        for z, ab, aa, sup in rows:
            print(f"{z:>5g} {ab:>11.4f} {aa:>10.4f} {sup * 100:>10.1f}%")
        print()
        print("Reading the result:")
        print("  AUC ~ control (~0.45-0.55) -> this z adds SAMPLES, not environments;")
        print("                                same conclusion as docs 9.0b, drop the axis.")
        print("  AUC well above control     -> genuinely new environments. Stage D's")
        print("                                max_auc=0.60 guard will then ABORT: it")
        print("                                cannot tell 'new physics' from 'source")
        print("                                shortcut'. Condition the critic on the")
        print("                                descriptor before scaling this up.")
        print("  in support << 100%         -> the balanced sampler will discard most of")
        print("                                the new crops; extrapolation, not coverage.")


if __name__ == "__main__":
    main()
