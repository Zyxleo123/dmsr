#!/usr/bin/env python
"""Write a map2map checkpoint's full 512^3 candidate box to .npy, for a halo finder.

WHY THIS EXISTS. P(k) cannot settle the question this line is actually asking.
Subhalos are sub-grid -- frozen SR2 already matches HR's density spectrum to
1-5% at every resolvable k, and the structures at issue live above k = 32 where
the LR input carries nothing. So the spectrum ranks arms on power it can see and
stays silent about the objects. Rockstar and SUBFIND read the particles.

The inference path here is `super_resolve_srs` with the same geometry, seed and
generator loader as `m2m_field_report.py`, deliberately and by import rather
than by restatement: a box scored with one path and halo-found with another
would not be the same candidate, and the discrepancy would be invisible.

Output is `(6, 512, 512, 512)` float32 -- the layout `catalog_summaries.py
--source candidate --field` and `subfind_snapshot.py --field` both read, and the
same layout as the cached frozen boxes in `cache/sr2_base`.

A box is 3.2 GB. Write it, find halos, then DELETE it: the catalogs are the
measurement and the field is regenerable from the checkpoint in ~12 min. The
2026-09-14 quota death (`rockstar-label-fails-are-quota`) was 46 of these left
lying around.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features", PROJECT_ROOT / "scripts" / "sr2"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner  # noqa: E402
from _sr2_direct import geometry_of, load_direct_config, load_lr  # noqa: E402
from m2m_field_report import load_m2m_generator  # noqa: E402

from cosmo_sr.eval.baseline_srs import super_resolve_srs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a map2map state_N.pt")
    ap.add_argument("--repo", default="/zfsauton2/home/yixiz/DMSR/map2map_colleague/map2map")
    ap.add_argument("--model", default="srsgan.G")
    ap.add_argument("--style-a", type=float, default=1.0)
    ap.add_argument("--style-size", type=int, default=1)
    ap.add_argument("--config", default="configs/reward/sr2_direct_finetune.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="destination .npy")
    ap.add_argument("--device", default="")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.is_file() and not a.overwrite:
        # Not an error: the chain re-runs and a 3.2 GB regeneration is 12 min of
        # GPU for a byte-identical result (the seed is fixed).
        print(f"  {out} exists; keeping it (pass --overwrite to redo)")
        return 0

    cfg = load_direct_config(a)
    geom = geometry_of(cfg)
    device = torch.device(a.device
                          or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = Path(a.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint at {ckpt}")

    g, prov = load_m2m_generator(
        ckpt, Path(a.repo), scale_factor=geom.scale_factor, in_chan=6,
        out_chan=6, style_size=a.style_size, style_a=a.style_a,
        model=a.model, device=device)
    banner(f"m2m box dump: {ckpt.name} (epoch {prov['epoch']}) on {a.box}")
    print(f"  provenance: {json.dumps(prov)}", flush=True)

    t0 = time.time()
    lr = np.asarray(load_lr(cfg, a.box), dtype=np.float32)
    sr = super_resolve_srs(g, lr, scale_factor=geom.scale_factor,
                           nsplit=geom.nsplit, pad=geom.pad,
                           seed=int(a.seed), device=device)
    sr = np.ascontiguousarray(sr, dtype=np.float32)
    print(f"  generated {sr.shape} in {time.time() - t0:.0f}s", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npy.tmp")
    # np.save APPENDS ".npy" when the name does not already end in it, so
    # np.save("x.npy.tmp", a) silently writes "x.npy.tmp.npy" and the rename
    # below then fails on a file that was never created -- after the 3.2 GB
    # had already been written. Handing it an open file object is the only
    # spelling that writes exactly where it is told.
    with open(tmp, "wb") as fh:
        np.save(fh, sr)
    tmp.replace(out)
    # Written via a temp name and renamed: a halo finder that starts on a
    # half-written 3.2 GB array fails deep inside GADGET with a shape error,
    # hours later, and reads as a physics problem.
    print(f"  wrote {out}  ({out.stat().st_size / 2**30:.2f} GiB)")
    side = {"ckpt": str(ckpt), "epoch": prov["epoch"], "box": a.box,
            "seed": int(a.seed), "model": a.model, "shape": list(sr.shape)}
    out.with_suffix(".json").write_text(json.dumps(side, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
