#!/usr/bin/env python
"""Standardized density eval: ALL models on IDENTICAL crops, BOTH r(k) conventions.

Resolves the training-val (0.816) vs eval_test (0.324) density_rk gap by reporting
per-crop-then-mean r(k) (the eval_test convention) AND stacked-spectra-per-box r(k)
(the training-val convention) side by side, on the same held-out crops, for every
model including the unconstrained ablation and mean-innovation runs that the
existing eval_test table omits.

Aggregation is per-box so both conventions stay box-bootstrappable. Density is the
same `HighPassDensity.density` (CIC overdensity) used everywhere else.

Run on a GPU (rhea srun) -- see the companion launcher.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cosmo_sr.data.datasets import finite_loader
from cosmo_sr.dmsr.data import build_val_dataset, resolve_split
from cosmo_sr.dmsr.density import HighPassDensity, cellsizes
from cosmo_sr.dmsr.evaluate import auto_cross_power, BandEdges, power_error, pdf_error, sample_diversity
from cosmo_sr.dmsr.operator import NullSpaceOperator
from cosmo_sr.train import common
from cosmo_sr.utils.config import load_config
from dmsr_eval import load_flow, _UpsampleBaseline

# name : (config, ckpt)   -- ckpt None => A_plus baseline
MODELS = {
    "baseline_upsample": (None, None),
    "paired_det":        ("configs/dmsr/paired_deterministic.yaml", "runs/dmsr/paired_deterministic/ckpt_best.pt"),
    "stage_a":           ("configs/dmsr/stage_a_paired_flow.yaml",  "runs/dmsr/stage_a/ckpt_best.pt"),
    "stage_c":           ("configs/dmsr/stage_c_critic_pairedlr.yaml", "runs/dmsr/stage_c_s0/ckpt_best.pt"),
    "mean_innovation":   ("configs/dmsr/mean_innovation_e.yaml",    "runs/dmsr/mean_innovation_e_s0/ckpt_best.pt"),
    "unconstrained":     ("configs/dmsr/unconstrained_ablation.yaml", "runs/dmsr/unconstrained_ablation_s0/ckpt_best.pt"),
}
BANDS = ("low", "transition", "high")


def band_avg(shell_vals, mask):
    v = shell_vals[mask]
    return float(v.mean()) if v.numel() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-config", default="configs/dmsr/stage_c_critic_pairedlr.yaml",
                    help="config used ONLY to build the shared crop set")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-crops", type=int, default=384)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--diversity-samples", type=int, default=3)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = common.select_device(None)
    ref = load_config(args.ref_config)
    dcfg = ref.get("data", {})
    factor = int(ref.get("factor", 8))
    crop_lr = int(dcfg.get("crop_lr", 8))
    channels = int(dcfg.get("channels", 6))
    split = resolve_split(dcfg)

    # ONE dataset -> identical crops for every model (deterministic tiling).
    ds = build_val_dataset(split, crop_lr=crop_lr, scale_factor=factor, channels=channels,
                           use_channels=dcfg.get("use_channels"), mmap=bool(dcfg.get("mmap", True)),
                           max_crops=args.max_crops, which=args.split)
    hr_cell, _ = cellsizes(dcfg, factor)
    highpass = HighPassDensity(factor=factor, cellsize=hr_cell,
                               dis_norm=float(dcfg.get("dis_norm", 6000.0))).to(device)
    bands = BandEdges(float(ref.get("eval", {}).get("low_frac", 0.5)),
                      float(ref.get("eval", {}).get("high_frac", 1.5)))
    boxes = [Path(b).stem for b in getattr(split, f"{args.split}_hr")]
    print(f"[eval] split={args.split} boxes={boxes} crops={len(ds)} device={device}", flush=True)

    # cache the crops once (list of (box, y, x)) so each model sees identical inputs
    crops = []
    for batch in finite_loader(ds, 1):
        crops.append((int(batch["box"][0]), batch["lr"].to(device), batch["hr"].to(device)))
    print(f"[eval] cached {len(crops)} crops", flush=True)

    results = {}
    for name in args.models:
        cfgp, ckpt = MODELS[name]
        if ckpt is None:
            model = _UpsampleBaseline(NullSpaceOperator(factor=factor).to(device))
            mcfg = ref
        else:
            mcfg = load_config(cfgp)
            uc = len(mcfg.get("data", {}).get("use_channels") or []) or channels
            model = load_flow(mcfg, uc, ckpt, device, use_ema=True)
        # per-box accumulators
        acc = {}  # box -> dict
        for bi, y, x in crops:
            a = acc.setdefault(bi, {"phh": 0.0, "ptt": 0.0, "pht": 0.0, "cen": None,
                                    "rk": {b: [] for b in BANDS}, "tk": {b: [] for b in BANDS},
                                    "pe": [], "pdf": [], "sig": [], "div": [], "cons": []})
            with torch.no_grad():
                x_hat = model.generate(y, n_steps=args.n_steps) if ckpt else model.generate(y)
                rho_h = highpass.density(x_hat)
                rho_t = highpass.density(x)
                phh, ptt, pht, cen = auto_cross_power(rho_h, rho_t, 24)
                a["phh"] += phh; a["ptt"] += ptt; a["pht"] += pht; a["cen"] = cen
                k_lr = rho_h.shape[-1] / (2.0 * factor)
                masks = bands.masks(cen, k_lr)
                rk = pht / (phh * ptt).clamp_min(1e-30).sqrt()
                tk = (phh / ptt.clamp_min(1e-30)).clamp_min(0).sqrt()
                for b in BANDS:
                    a["rk"][b].append(band_avg(rk, masks[b]))
                    a["tk"][b].append(band_avg((tk - 1).abs(), masks[b]))
                a["pe"].append(power_error(rho_h, rho_t))
                a["pdf"].append(pdf_error(rho_h, rho_t))
                a["sig"].append(float(rho_h.std() / rho_t.std().clamp_min(1e-12)))
                if ckpt:
                    a["div"].append(sample_diversity(model, y, n_samples=args.diversity_samples,
                                                      n_steps=args.n_steps)["sample_diversity"])
                    ab, rel = model.operator.consistency_error(x_hat, y)
                    a["cons"].append(rel)
                else:
                    a["div"].append(0.0); a["cons"].append(0.0)
        # per-box -> both conventions.  HR crop N = LR crop * factor; k_lr = N/(2*factor).
        hr_n = crops[0][1].shape[-1] * factor
        k_lr = hr_n / (2.0 * factor)
        perbox = {}
        for bi, a in acc.items():
            masks = bands.masks(a["cen"], k_lr)
            rk_s = a["pht"] / (a["phh"] * a["ptt"]).clamp_min(1e-30).sqrt()
            tk_s = (a["phh"] / a["ptt"].clamp_min(1e-30)).clamp_min(0).sqrt()
            row = {}
            for b in BANDS:
                row[f"stacked_rk_{b}"] = band_avg(rk_s, masks[b])
                row[f"stacked_Tk_{b}"] = band_avg((tk_s - 1).abs(), masks[b])
                row[f"percrop_rk_{b}"] = float(np.nanmean(a["rk"][b]))
                row[f"percrop_Tk_{b}"] = float(np.nanmean(a["tk"][b]))
            row["density_power_error"] = float(np.mean(a["pe"]))
            row["density_pdf_error"] = float(np.mean(a["pdf"]))
            row["density_sigma_ratio"] = float(np.mean(a["sig"]))
            row["sample_diversity"] = float(np.mean(a["div"]))
            row["consistency_rel"] = float(np.mean(a["cons"]))
            perbox[bi] = row
        # aggregate across boxes: mean + box sd
        keys = next(iter(perbox.values())).keys()
        agg = {}
        for k in keys:
            vals = np.array([perbox[b][k] for b in perbox], float)
            vals = vals[np.isfinite(vals)]
            agg[k] = {"mean": float(vals.mean()), "box_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                      "n_boxes": int(len(vals))}
        results[name] = {"aggregate": agg, "per_box": perbox}
        print(f"[{name}] stacked_rk_high={agg['stacked_rk_high']['mean']:.3f} "
              f"percrop_rk_high={agg['percrop_rk_high']['mean']:.3f} "
              f"pow_err={agg['density_power_error']['mean']:.3f} "
              f"Tk_high(st)={agg['stacked_Tk_high']['mean']:.3f} "
              f"pdf={agg['density_pdf_error']['mean']:.3f} div={agg['sample_diversity']['mean']:.3f} "
              f"cons={agg['consistency_rel']['mean']:.1e}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({"split": args.split, "boxes": boxes, "n_crops": len(crops),
               "max_crops": args.max_crops, "n_steps": args.n_steps, "results": results},
              open(out / "density_std.json", "w"), indent=2)

    # console table
    cols = ["stacked_rk_transition", "stacked_rk_high", "percrop_rk_transition", "percrop_rk_high",
            "stacked_Tk_high", "density_power_error", "density_pdf_error", "density_sigma_ratio",
            "sample_diversity", "consistency_rel"]
    print("\n" + f"{'model':<18}" + "".join(f"{c[:15]:>17}" for c in cols))
    for name in args.models:
        a = results[name]["aggregate"]
        print(f"{name:<18}" + "".join(f"{a[c]['mean']:>17.4g}" for c in cols))
    print(f"\n[eval] wrote {out/'density_std.json'}")


if __name__ == "__main__":
    main()
