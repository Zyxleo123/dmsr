#!/usr/bin/env python
"""Preprocess paired LR/HR sims into canonical catnorm fields for training.

Source (raw disp/vel, MP-Gadget kpc/h units):
  $ZFS/DMSR/int_redshift_same_cosmology/dmo-64/<set>/PART_009/{disp,vel}.npy   (LR, Ng=64)
  $ZFS/DMSR/int_redshift_same_cosmology/dmo-512/<set>/PART_009/{disp,vel}.npy  (HR, Ng=512)

Output (canonical (6,Ng,Ng,Ng) float32):
  <out>/lr/<set>.npy   and   <out>/hr/<set>.npy

Example:
  python scripts/preprocess_paired.py --out /zfsauton/scratch/yixiz/DMSR/paired_catnorm
  python scripts/preprocess_paired.py --sets set0 set1 --out .../paired_catnorm
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from cosmo_sr.data.preprocess_srs import disp_vel_to_field

DEFAULT_ROOT = os.path.expanduser(
    os.environ.get("ZFS", "/zfsauton/scratch/yixiz")
) + "/DMSR/int_redshift_same_cosmology"


def _part(root, res, set_name):
    return Path(root) / f"dmo-{res}" / set_name / "PART_009"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="int_redshift_same_cosmology dir")
    ap.add_argument("--out", required=True, help="output dir (creates lr/ and hr/)")
    ap.add_argument("--sets", nargs="*", default=None, help="set names (default: all in dmo-64)")
    ap.add_argument("--lr-res", type=int, default=64)
    ap.add_argument("--hr-res", type=int, default=512)
    ap.add_argument("--redshift", type=float, default=0.0)
    args = ap.parse_args()

    root = Path(args.root)
    sets = args.sets or sorted(p.name for p in (root / f"dmo-{args.lr_res}").iterdir() if p.is_dir())

    out = Path(args.out)
    (out / "lr").mkdir(parents=True, exist_ok=True)
    (out / "hr").mkdir(parents=True, exist_ok=True)

    for s in sets:
        lr_dir = _part(root, args.lr_res, s)
        hr_dir = _part(root, args.hr_res, s)
        lr_out = out / "lr" / f"{s}.npy"
        hr_out = out / "hr" / f"{s}.npy"
        print(f"[{s}] LR {lr_dir}")
        disp_vel_to_field(lr_dir / "disp.npy", lr_dir / "vel.npy", lr_out, args.redshift)
        print(f"[{s}] HR {hr_dir} (this is ~3GB, may take a while)")
        disp_vel_to_field(hr_dir / "disp.npy", hr_dir / "vel.npy", hr_out, args.redshift)
        print(f"[{s}] wrote {lr_out} and {hr_out}")

    print(f"Done. LR files: {out/'lr'}/*.npy   HR files: {out/'hr'}/*.npy")


if __name__ == "__main__":
    main()
