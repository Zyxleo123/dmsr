#!/usr/bin/env python
"""Lay our paired boxes out the way ``map2map``'s loader expects.

Why this exists
---------------
We are switching from our own trainer to the ``styled_srsgan`` branch of
``map2map`` itself, so that nothing in the adversary is a reimplementation of
ours. That branch's loader wants one ``.npy`` per *field*, not one per box, and
the reason is not cosmetic -- it is the augmentation:

``map2map/data/fields.py::flip``::

    if x.shape[0] == ndim:      # flip vector components
        x[axes] = - x[axes]

A field is treated as a **vector** field only when its channel count equals the
spatial dimensionality. Our combined ``(6, N, N, N)`` boxes have 6 != 3, so
``--augment`` would flip the voxel axes and leave the displacement and velocity
*components* pointing the old way -- a physically inconsistent field, produced
silently, with nothing in any log to say so. Splitting into ``disp.npy`` and
``vel.npy`` (3 channels each) is what makes the negation fire.

The style file
--------------
The ``styled_srsgan`` branch calls ``model(input, style)`` unconditionally and
``lag2eul(lag_out, a=np.float(style))``, so a style is mandatory, not optional.
``style`` is the scale factor ``a``; our boxes are z = 0, so ``a = 1.0``. One
one-element array per box, which is what ``fields.py`` reads
(``style_size = np.load(style_files[0]).shape[0]``).

What is NOT done here
---------------------
No renormalisation. Our data is already in ``cosmology.dis`` space: that norm is
``x / (dis_std * D(z))`` with ``dis_std = 6000`` kpc/h, and D(0) = 1, which is
exactly the ``dis_norm_kpc_h = 6000`` our boxes were written with. Applying
``--in-norms cosmology.dis`` on top would normalise twice. The run passes an
identity norm through ``--callback-at`` instead -- map2map's own documented
escape hatch, not an edit to their code.

Layout written::

    <out>/<split>/lr/<box>/{disp,vel}.npy      (3, 64, 64, 64)
    <out>/<split>/hr/<box>/{disp,vel}.npy      (3, 512, 512, 512)
    <out>/<split>/style/<box>.npy              (1,)  -- the scale factor

``sorted(glob(...))`` over ``<split>/lr/set*/disp.npy`` and over
``<split>/hr/set*/disp.npy`` visits the boxes in the same order, because the
paths differ only in one component at the same depth. That is what pairs input
with target, so the two globs must never be given different box sets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, write_json  # noqa: E402

#: (name, channel slice) of the two vector fields inside our 6-channel boxes.
FIELDS = (("disp", slice(0, 3)), ("vel", slice(3, 6)))


def split_box(src: Path, dst_dir: Path, *, overwrite: bool = False,
              chunk: int = 64) -> List[Path]:
    """Write ``disp.npy`` and ``vel.npy`` from one combined ``(6, N, N, N)`` box.

    Written a slab at a time through ``np.lib.format.open_memmap``: an HR box is
    3 GB per field, and materialising both plus the source would be 9 GB of
    resident array for what is a pure copy.
    """
    a = np.load(src, mmap_mode="r")
    if a.ndim != 4 or a.shape[0] != 6:
        raise ValueError(f"{src}: expected (6, N, N, N), got {a.shape}")
    n = int(a.shape[-1])
    dst_dir.mkdir(parents=True, exist_ok=True)

    out: List[Path] = []
    for name, sl in FIELDS:
        p = dst_dir / f"{name}.npy"
        out.append(p)
        if p.is_file() and not overwrite:
            got = np.load(p, mmap_mode="r")
            if got.shape == (3, n, n, n):
                print(f"    {p} exists, skipping", flush=True)
                continue
            print(f"    {p} exists with wrong shape {got.shape}; rewriting",
                  flush=True)
        tmp = p.with_suffix(".npy.tmp")
        m = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                      shape=(3, n, n, n))
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            m[:, i:j] = np.asarray(a[sl, i:j], dtype=np.float32)
        m.flush()
        del m
        tmp.replace(p)          # atomic: a half-written box must never be globbed
        print(f"    wrote {p}  (3, {n}, {n}, {n})", flush=True)
    return out


def run(args) -> dict:
    src_root = Path(args.data_root)
    out_root = Path(args.out)
    train = [b for b in str(args.train_boxes).replace(",", " ").split() if b]
    val = [b for b in str(args.val_boxes).replace(",", " ").split() if b]
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise SystemExit(
            f"boxes {overlap} are in BOTH --train-boxes and --val-boxes. The "
            "whole verdict of this line is a box never trained on.")

    banner(f"map2map layout -> {out_root}")
    print(f"  train: {train}\n  val:   {val}")
    print(f"  style: a = {args.scale_factor_a} (z = "
          f"{1.0 / float(args.scale_factor_a) - 1.0:.3g})", flush=True)

    manifest = {"source": str(src_root), "out": str(out_root),
                "train_boxes": train, "val_boxes": val,
                "style_a": float(args.scale_factor_a), "boxes": {}}

    for split, boxes in (("train", train), ("val", val)):
        for box in boxes:
            print(f"  [{split}] {box}", flush=True)
            rec = {}
            for side, ng in (("lr", None), ("hr", None)):
                src = src_root / side / f"{box}.npy"
                if not src.is_file():
                    raise SystemExit(f"missing {src}")
                paths = split_box(src, out_root / split / side / box,
                                  overwrite=bool(args.overwrite))
                rec[side] = [str(p) for p in paths]
            # the style: one scale factor per box, shape (1,) so style_size == 1
            sdir = out_root / split / "style"
            sdir.mkdir(parents=True, exist_ok=True)
            sp = sdir / f"{box}.npy"
            np.save(sp, np.asarray([float(args.scale_factor_a)],
                                   dtype=np.float32))
            rec["style"] = str(sp)
            manifest["boxes"][box] = rec

    # The globs the training job will use, recorded so the job and this script
    # cannot drift apart.
    manifest["patterns"] = {
        f"{split}_{kind}": ",".join(
            str(out_root / split / side / "set*" / f"{f}.npy")
            for f, _ in FIELDS)
        for split in ("train", "val")
        for kind, side in (("in", "lr"), ("tgt", "hr"))
    }
    for split in ("train", "val"):
        manifest["patterns"][f"{split}_style"] = str(
            out_root / split / "style" / "set*.npy")

    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "manifest.json", manifest)
    print(f"\nwrote {out_root / 'manifest.json'}")
    for k, v in manifest["patterns"].items():
        print(f"  {k:12s} {v}")
    return {"ok": True, **manifest}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root",
                    default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm")
    ap.add_argument("--out", required=True,
                    help="layout root; ~3 GB per HR box, so scratch only")
    ap.add_argument("--train-boxes",
                    default="set0 set1 set2 set3 set4 set5 set6 set7 set8 "
                            "set11 set12")
    ap.add_argument("--val-boxes", default="set9 set10")
    ap.add_argument("--scale-factor-a", type=float, default=1.0,
                    help="the style: the scale factor a. Our boxes are z = 0, "
                         "so 1.0. NOTE map2map's cosmology norms default to "
                         "a = 0.3333 (z = 2); nothing here uses that default, "
                         "but a run that switched to a real norm would.")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main(argv=None) -> int:
    return 0 if run(build_parser().parse_args(argv)).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
