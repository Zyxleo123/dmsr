#!/usr/bin/env python
"""Save 2D slices of candidate / frozen SR2 / HR so the fields can be LOOKED at.

The P(k) curves say the candidate tracks HR to k=32 and then dies. This is the
same statement in real space, which is worth having because a spectrum can hide
what a picture makes obvious -- and because "generated nothing above the input
Nyquist" predicts something very specific about how the slice should look: the
candidate should be a blurred version of HR with no fine texture, not a noisy or
displaced version of it.

GPU, because it needs one full-box realisation from the generator. It writes the
SLICES ONLY (a few MB) and draws nothing: the figure is a CPU job reading this
npz, so it can be redrawn or re-styled without regenerating a 512^3 box.

Channels follow the spectra convention -- ONE component per field, channel 0 for
displacement and channel 3 for velocity -- so the picture and the curve are of
the same quantity and can be read against each other.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features", PROJECT_ROOT / "scripts" / "sr2"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner  # noqa: E402
from _sr2_direct import geometry_of, load_direct_config, load_hr, load_lr  # noqa: E402
from m2m_field_report import load_m2m_generator  # noqa: E402

from cosmo_sr.eval.baseline_srs import super_resolve_srs  # noqa: E402
from cosmo_sr.reward.base import find_base_field  # noqa: E402

FIELDS = (("displacement", 0), ("velocity", 3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", default="srsgan.G")
    ap.add_argument("--style-a", type=float, default=1.0)
    ap.add_argument("--style-size", type=int, default=1)
    ap.add_argument("--config", default="configs/reward/sr2_direct_finetune.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", type=int, default=-1,
                    help="slice index along axis 0; default the box mid-plane")
    ap.add_argument("--zoom", type=int, default=128,
                    help="side of the zoom crop, in HR voxels. The full slice is "
                         "512 voxels across and at that scale every field looks "
                         "alike -- they agree below k=32 by construction. The "
                         "zoom is where a missing octave is visible.")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="")
    a = ap.parse_args(argv)

    cfg = load_direct_config(a)
    geom = geometry_of(cfg)
    device = torch.device(a.device
                          or ("cuda" if torch.cuda.is_available() else "cpu"))

    frozen_path = find_base_field(a.box, seed=int(a.seed))
    if frozen_path is None:
        print(f"no cached frozen SR2 box for {a.box} seed {a.seed}; the whole "
              "point is the three-way comparison, so nothing is written.")
        return 0

    g, prov = load_m2m_generator(
        Path(a.ckpt), Path(a.repo), scale_factor=geom.scale_factor,
        in_chan=6, out_chan=6, style_size=a.style_size, style_a=a.style_a,
        model=a.model, device=device)
    banner(f"slices: {Path(a.ckpt).name} (epoch {prov['epoch']}) on {a.box}")

    hr = np.ascontiguousarray(load_hr(cfg, a.box), dtype=np.float32)
    lr = np.asarray(load_lr(cfg, a.box), dtype=np.float32)
    sr = super_resolve_srs(g, lr, scale_factor=geom.scale_factor,
                           nsplit=geom.nsplit, pad=geom.pad,
                           seed=int(a.seed), device=device)
    frozen = np.asarray(np.load(str(frozen_path), mmap_mode="r"),
                        dtype=np.float32)

    ng = hr.shape[1]
    idx = (ng // 2) if a.index < 0 else int(a.index)
    z = int(a.zoom)
    lo = (ng - z) // 2

    out_arrays = {"index": idx, "ng": ng, "zoom": z, "zoom_lo": lo,
                  "box": a.box, "ckpt": str(a.ckpt), "epoch": prov["epoch"]}
    for name, ch in FIELDS:
        for label, fld in (("cand", sr), ("frozen", frozen), ("hr", hr)):
            full = fld[ch, idx]
            out_arrays[f"{name}_{label}_full"] = full
            out_arrays[f"{name}_{label}_zoom"] = full[lo:lo + z, lo:lo + z]
        print(f"  {name}: slice {idx} of {ng}, zoom {z}^2 at ({lo},{lo})  "
              f"std cand {sr[ch, idx].std():.4f} / frozen "
              f"{frozen[ch, idx].std():.4f} / hr {hr[ch, idx].std():.4f}")

    out = Path(a.out) if a.out else Path(a.ckpt).parent / f"slices_{Path(a.ckpt).stem}_{a.box}"
    np.savez_compressed(str(out.with_suffix(".npz")), **out_arrays)
    print(f"wrote {out.with_suffix('.npz')}  (slices; plot on CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
