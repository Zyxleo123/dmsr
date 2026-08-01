#!/usr/bin/env python
"""Gate A criterion 6 (GPU): does the written core depend on the tile size?

Valid-core tiling is only exact if the margin covers the receptive field. The
training run cannot answer that -- it never tiles -- so criterion 6 of sec. 3a
needs its own job, and without the JSON this writes ``gate_a_check.py`` reports
``not_evaluated`` and Gate A can never pass.

Two full-box residual realisations are drawn from the SAME seed at two different
tile *cores* and compared voxel by voxel. Same seed means the initial noise is
identical (it is drawn on the CPU from the seed alone), so any difference is the
tiling and nothing else. The margin is held fixed: it is the margin that has to
be large enough, and changing both at once would not say which one mattered.

Agreement is expected at float32 rounding, not bit-exactly -- torch picks
different convolution algorithms for different input sizes -- so the tolerance is
~1e-4 relative, far below the ~1 relative difference a too-small margin produces
(measured with nn.GroupNorm: the tiled and untiled results differed by the full
signal amplitude).

    python scripts/reward/tile_scan.py --checkpoint .../ckpt_best.pt --box set8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, load_reward_config, lr_path,
                     parse_boxes, select_device, write_json)
from sample_oracle_candidates import load_model

from cosmo_sr.reward.base import find_base_field
from cosmo_sr.reward.diffusion import DiffusionConfig
from cosmo_sr.reward.sampling import (TileSpec, measure_receptive_field,
                                      sample_residual_box, tile_margin_for)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--model-config", default="configs/reward/residual_prior.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--box", default=None, help="default: the first val box")
    ap.add_argument("--split", default="val",
                    choices=["train", "val", "test", "dev", "final"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--tile-cores", default="128,64",
                    help="two core sizes, both dividing ng_hr")
    ap.add_argument("--tile-margin", type=int, default=48)
    ap.add_argument("--tile-batch", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="fewer DDIM steps make the scan cheaper; the tiling "
                         "identity holds at every step count")
    ap.add_argument("--tolerance", type=float, default=1e-4,
                    help="max |a-b| / rms(a) accepted as float32 rounding")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    from cosmo_sr.utils.config import load_config
    mcfg = load_config(args.model_config)
    cfg_model = {**cfg, "model": mcfg.get("model", {}),
                 "diffusion": mcfg.get("diffusion", {})}

    box = args.box or parse_boxes(None, cfg, args.split)[0]
    device = select_device(args.device)
    model = load_model(args.checkpoint, cfg_model, device, use_ema=not args.no_ema)

    diff = DiffusionConfig(**{
        k: v for k, v in dict(cfg_model.get("diffusion", {})).items()
        if k in DiffusionConfig.__dataclass_fields__
    })
    if args.n_steps:
        diff.n_steps = int(args.n_steps)

    ng_hr = int(cfg["data"]["ng_hr"])
    sf = int(cfg["data"]["scale_factor"])
    cores = [int(c) for c in str(args.tile_cores).split(",") if c.strip()]
    if len(cores) != 2:
        raise SystemExit(f"--tile-cores needs exactly two sizes, got {cores}")

    rf = measure_receptive_field(
        model, channels=int(cfg_model["model"].get("channels", 6)),
        scale_factor=sf, device=device)
    need = tile_margin_for(rf, sf)
    banner(f"{box}: receptive field {rf} -> minimum margin {need}; "
           f"scanning cores {cores} at margin {args.tile_margin}")

    lr = np.load(lr_path(cfg, box))
    base_path = find_base_field(box, args.base_seed)
    if base_path is None:
        raise SystemExit(f"no cached SR2 base for {box}")
    base = np.asarray(np.load(base_path, mmap_mode="r"))

    fields, timings = [], []
    for core in cores:
        spec = TileSpec(ng_hr, core=core, margin=int(args.tile_margin),
                        scale_factor=sf)
        t0 = time.time()
        fields.append(sample_residual_box(
            model, base, lr, seed=int(args.seed), cfg=diff, spec=spec,
            device=device, redshift=float(cfg["data"].get("redshift", 0.0)),
            tile_batch=int(args.tile_batch), verify_margin=False,
        ))
        timings.append(time.time() - t0)
        print(f"  core={core} tile={core + 2 * args.tile_margin} "
              f"({timings[-1]:.0f}s)", flush=True)

    a, b = (np.asarray(f, dtype=np.float64) for f in fields)
    scale = float(np.sqrt(np.mean(a ** 2)))
    diff_abs = np.abs(a - b)
    rel = float(diff_abs.max() / scale) if scale > 0 else float("nan")
    passed = bool(np.isfinite(rel) and rel <= float(args.tolerance))

    out = write_json(args.out, {
        "box": box,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "use_ema": not args.no_ema,
        "seed": int(args.seed),
        "tile_cores": cores,
        "tile_margin": int(args.tile_margin),
        "receptive_field_halfwidth": int(rf),
        "minimum_margin": int(need),
        "margin_sufficient": bool(int(args.tile_margin) >= need),
        "n_steps": int(diff.n_steps),
        # The name gate_a_check.py reads.
        "max_rel_difference": rel,
        "mean_rel_difference": float(diff_abs.mean() / scale) if scale > 0 else float("nan"),
        "residual_rms": scale,
        "tolerance": float(args.tolerance),
        "passed": passed,
        "seconds": timings,
        "note": (
            "Same seed, same margin, two tile cores. A relative difference of "
            "order 1e-7 is float32/algorithm selection; order 1e-2 or more means "
            "the margin does not cover the receptive field and the written core "
            "carries tile-boundary padding -- the seams that show up later as "
            "spurious small-scale power."
        ),
    })
    banner(f"tile scan: {'PASS' if passed else 'FAIL'} "
           f"(max relative core difference {rel:.3g}, tolerance {args.tolerance})")
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
