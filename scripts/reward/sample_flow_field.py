#!/usr/bin/env python
"""Sample a full-box 512^3 catnorm field from a trained flow cascade.

The residual null-space flow was only ever *evaluated* on power spectra
(``scripts/eval_flow.py`` on center crops). To put its output through the same
Rockstar halo finder every other field in this project is judged by, we need a
complete periodic 512^3 box, not a crop. This script produces exactly that.

Input is the pooled-HR 64^3 box (``A`` applied three times to the held-out HR
field). set15 has no real LR simulation, and the pooled-HR box is precisely what
the frozen ``set15__base__base`` catalog derives from, so the flow catalog this
enables is directly comparable to both the frozen HR-truth and base catalogs.

The multiscale operators (A/U/P_null) are all local (avg-pool / nearest
upsample), and the velocity net is a small fully-convolutional stack, so the
whole cascade is well defined on the full box; we run it in one shot on a large
GPU under ``torch.no_grad``.

    python scripts/reward/sample_flow_field.py \
        --config configs/flow_cascade.yaml \
        --checkpoint runs/flow_cascade/ckpt_last.pt \
        --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set15.npy \
        --n-steps 20 --seed 0 \
        --out /path/to/flow_cascade_set15_seed0.npy
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hr", required=True, help="held-out HR catnorm .npy (Ng=512)")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output field .npy (6, 512, 512, 512)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.is_file() and not args.overwrite:
        print(f"cached -> {out_path}", flush=True)
        return

    from cosmo_sr.utils.config import load_config
    from cosmo_sr.data.field_io import load_field
    from cosmo_sr.operators.multiscale import MultiScaleOperators
    from cosmo_sr.train.train_flow import _build_base_upscaler, _build_flow_model
    from cosmo_sr.inference.flow_sample import super_resolve_cascade

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = int(cfg.get("model", {}).get("channels", 6))
    factor = int(cfg.get("factor", 2))
    resolutions = list(cfg.get("resolutions", [64, 128, 256]))
    full_res = int(cfg.get("data", {}).get("full_res", 512))

    ops = MultiScaleOperators(factor=factor).to(device)
    base_upscaler = _build_base_upscaler(cfg, channels, factor, device)
    model = _build_flow_model(cfg.get("model", {}), channels, factor).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    if "extra" in state and "base_upscaler" in state["extra"]:
        base_upscaler.load_state_dict(state["extra"]["base_upscaler"])
    model.eval()

    # Build the coarsest input y0 = A^k(HR) on grid min(resolutions).
    hr = load_field(args.hr, mmap=True)
    if hr.shape[-1] != full_res:
        raise SystemExit(
            f"HR grid {hr.shape[-1]} != full_res {full_res} from config")
    x = torch.from_numpy(np.ascontiguousarray(hr)).float().unsqueeze(0).to(device)
    coarsest = min(resolutions)
    with torch.no_grad():
        while x.shape[-1] > coarsest:
            x = ops.A(x)
    if x.shape[-1] != coarsest:
        raise SystemExit(
            f"pooled HR to grid {x.shape[-1]}, expected {coarsest}")

    t0 = time.time()
    with torch.no_grad():
        levels = super_resolve_cascade(
            model, ops, base_upscaler, x,
            resolutions=tuple(resolutions), n_steps=args.n_steps, seed=args.seed,
        )
    top = 2 * max(resolutions)
    field = levels[top][0].float().cpu().numpy()
    dt = time.time() - t0

    if field.shape != (channels, full_res, full_res, full_res):
        raise SystemExit(f"unexpected output shape {field.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npy")
    np.save(tmp, field.astype(np.float32))
    tmp.replace(out_path)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak = 0.0
    print(f"[sample_flow] wrote {out_path}  shape={field.shape}  "
          f"grid{coarsest}->{top}  n_steps={args.n_steps}  seed={args.seed}  "
          f"{dt:.0f}s  peak_gpu={peak:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
