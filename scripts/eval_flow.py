#!/usr/bin/env python
"""Evaluate a trained residual null-space flow cascade on a held-out HR box.

Builds the ground-truth resolution pyramid from an HR field (by repeated
block-average ``A``), runs the cascade from the coarsest level up to 512, and
scores each octave plus the final level. Metrics:

  * per-octave: A-consistency, high-k power recovery, residual power/octave, z-diversity
  * final 512 : power-spectrum recovery + cross-correlation r(k) (SR2-style),
                equilateral bispectrum (channel 0), velocity statistics.

Because the model is stochastic and cascade-based, distributional metrics (power,
bispectrum, diversity) are the meaningful ones -- not pixel MSE.

Example (GPU node):
  python scripts/eval_flow.py \
    --config configs/flow_cascade.yaml \
    --checkpoint runs/flow_cascade/ckpt_last.pt \
    --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set15.npy \
    --crop 256 --n-steps 20 --diversity 3 --out runs/flow_cascade/eval_set15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--hr", required=True, help="HR catnorm .npy (Ng=512)")
    ap.add_argument("--crop", type=int, default=256,
                    help="center-crop size at HR res for eval (keeps FFTs cheap)")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--diversity", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from cosmo_sr.utils.config import load_config
    from cosmo_sr.data.field_io import load_field
    from cosmo_sr.operators.multiscale import MultiScaleOperators
    from cosmo_sr.train.train_flow import _build_base_upscaler, _build_flow_model
    from cosmo_sr.eval.flow_eval import evaluate_cascade, sr2_power_summary
    from cosmo_sr.eval.sr2_stats import equilateral_bispectrum, velocity_statistics
    from cosmo_sr.inference.flow_sample import sample_step

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = int(cfg.get("model", {}).get("channels", 6))
    factor = int(cfg.get("factor", 2))
    resolutions = list(cfg.get("resolutions", [64, 128, 256]))
    full_res = int(cfg.get("data", {}).get("full_res", 512))

    ops = MultiScaleOperators(factor=factor).to(device)
    base_upscaler = _build_base_upscaler(cfg, channels, factor, device)
    mcfg = cfg.get("model", {})
    model = _build_flow_model(mcfg, channels, factor).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    if "extra" in state and "base_upscaler" in state["extra"]:
        base_upscaler.load_state_dict(state["extra"]["base_upscaler"])
    model.eval()

    # load + center-crop HR, build the ground-truth pyramid via A
    hr = load_field(args.hr, mmap=True)
    N = hr.shape[1]
    c = args.crop
    s = (N - c) // 2
    crop = np.ascontiguousarray(hr[:, s:s + c, s:s + c, s:s + c])
    x_fine = torch.from_numpy(crop).float().unsqueeze(0).to(device)

    pyramid = {full_res: x_fine}
    cur = x_fine
    label = full_res
    n_levels = len(resolutions) + 1
    for _ in range(n_levels - 1):
        cur = ops.A(cur)
        label //= 2
        pyramid[label] = cur

    out = evaluate_cascade(
        model, ops, base_upscaler, pyramid,
        resolutions=tuple(resolutions), n_steps=args.n_steps,
        diversity_samples=args.diversity,
    )

    # final-level SR2 summary vs ground truth
    top_R = max(resolutions)
    y0 = pyramid[min(resolutions)]
    cur = y0
    with torch.no_grad():
        for R in resolutions:
            cur = sample_step(model, ops, base_upscaler, cur, float(R), n_steps=args.n_steps)
    gen_top = cur[0].cpu()
    true_top = pyramid[2 * top_R][0].cpu()
    out["final"] = {
        "level": int(2 * top_R),
        "sr2_power_summary": sr2_power_summary(gen_top, true_top),
        "velocity_gen": velocity_statistics(gen_top),
        "velocity_true": velocity_statistics(true_top),
    }
    kg, bg = equilateral_bispectrum(gen_top[0].numpy())
    kt, bt = equilateral_bispectrum(true_top[0].numpy())
    out["final"]["bispectrum_k"] = kt.tolist()
    out["final"]["bispectrum_gen"] = bg.tolist()
    out["final"]["bispectrum_true"] = bt.tolist()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "flow_eval.json", "w") as f:
        json.dump(out, f, indent=2)

    # power-spectrum overlay at the final level
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from cosmo_sr.eval.flow_eval import mean_power_spectrum
        kk, pg = mean_power_spectrum(gen_top)
        _, pt = mean_power_spectrum(true_top)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(kk, pt, label="true", lw=2)
        ax.loglog(kk, pg, label="flow (gen)", lw=2, ls="--")
        ax.set_xlabel("k [modes]"); ax.set_ylabel("P(k)"); ax.legend()
        ax.set_title(f"Power spectrum @ {2 * top_R}")
        fig.tight_layout(); fig.savefig(outdir / "power.png", dpi=110); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")

    for R2, rec in out["octaves"].items():
        line = f"[octave->{R2}] consistency_rel={rec.get('consistency_rel'):.3g}"
        if "highk_power_ratio" in rec:
            line += f"  highk_Pratio={rec['highk_power_ratio']:.3f}"
        line += f"  z_diversity={rec.get('z_rel_diversity'):.3f}"
        print(line)
    print(f"[flow-eval] wrote {outdir/'flow_eval.json'}")


if __name__ == "__main__":
    main()
