#!/usr/bin/env python
"""Preprocess an MP-Gadget bigfile snapshot into a canonical (6,Ng,Ng,Ng) field.

Thin CLI wrapper around ``cosmo_sr.data.preprocess_srs.snapshot_to_field_bigfile``.

Example::

    python scripts/preprocess_snapshot.py \
        --inpath  /path/to/snapshot \
        --outpath /path/to/catnorm.npy
"""
import argparse

from cosmo_sr.data.preprocess_srs import snapshot_to_field_bigfile


def main():
    parser = argparse.ArgumentParser(description="snapshot -> canonical field")
    parser.add_argument("--inpath", required=True, help="LR snapshot (bigfile)")
    parser.add_argument("--outpath", required=True, help="output .npy path")
    args = parser.parse_args()
    field = snapshot_to_field_bigfile(args.inpath, args.outpath)
    print(f"Wrote {args.outpath} with shape {field.shape} dtype {field.dtype}")


if __name__ == "__main__":
    main()
