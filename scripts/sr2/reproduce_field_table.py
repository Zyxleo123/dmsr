#!/usr/bin/env python
"""One-command field-level table for the frozen SR2 baseline (Stage 0).

Uses the freeze.yaml inference settings (nsplit/pad/seed) and the seeded
ControlledG path so later seed comparisons share the same noise bookkeeping.
Default boxes = freeze split.field_table_boxes (test boxes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _highk(arr: np.ndarray, frac: float = 1.0 / 3.0) -> float:
    lo = int(len(arr) * (1.0 - frac))
    return float(np.mean(arr[lo:]))


def _spectra(sr_c, hr_c):
    from cosmo_sr.eval.spectra import power_spectrum, cross_correlation_coefficient
    k, pk_sr = power_spectrum(sr_c)
    _, pk_hr = power_spectrum(hr_c)
    _, rk = cross_correlation_coefficient(sr_c, hr_c)
    denom = np.clip(pk_hr, 1e-30, None)
    return {
        "k": k, "pk_sr": pk_sr, "pk_hr": pk_hr,
        "ratio": pk_sr / denom, "transfer": np.sqrt(pk_sr / denom), "rk": rk,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", default=str(ROOT / "configs/sr2_baseline/freeze.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs/sr2_baseline/field_table"))
    ap.add_argument("--sets", default=None, help="comma-separated override of boxes")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.freeze).read_text())
    from cosmo_sr.data.field_io import load_field
    from cosmo_sr.eval.density import cic_density
    from cosmo_sr.tts.srs_noise import load_controlled_generator
    from cosmo_sr.tts.sampling import super_resolve_srs_seeded

    data_root = Path(cfg["data"]["root"])
    sets = ([s.strip() for s in args.sets.split(",")] if args.sets
            else list(cfg["split"]["field_table_boxes"]))
    inf = cfg["inference"]
    seed = int(inf["field_table_seed"])
    nsplit, pad = int(inf["nsplit"]), int(inf["pad"])
    scale = int(cfg["model"]["scale_factor"])
    boxsize = float(cfg["cosmology_sim"]["boxsize_kpc_h"])
    dis_norm = float(cfg["data"]["dis_norm_kpc_h"])
    model_path = str(ROOT / cfg["model"]["path"])

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    G = load_controlled_generator(model_path, scale_factor=scale, device=device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_box = []
    agg_spec = {}

    for name in sets:
        lr = load_field(str(data_root / "lr" / f"{name}.npy")).astype(np.float32)
        hr = np.ascontiguousarray(
            load_field(str(data_root / "hr" / f"{name}.npy"), mmap=True), dtype=np.float32
        )
        sr = super_resolve_srs_seeded(
            G, lr, seed, scale_factor=scale, nsplit=nsplit, pad=pad, device=device,
            noise_mode=inf.get("noise_mode", "per_tile"),
        )
        cell = boxsize / hr.shape[1]
        metrics = {"box": name, "seed": seed}
        for field_name, ch in (("displacement", 0), ("velocity", 3)):
            sp = _spectra(sr[ch], hr[ch])
            metrics[f"{field_name}_transfer_highk"] = _highk(sp["transfer"])
            metrics[f"{field_name}_ratio_highk"] = _highk(sp["ratio"])
            metrics[f"{field_name}_rk_highk"] = _highk(sp["rk"])
            agg_spec.setdefault(field_name, []).append(sp)

        d_sr = cic_density(torch.as_tensor(sr[0:3])[None].to(device), cell, dis_norm)[0, 0]
        d_hr = cic_density(torch.as_tensor(hr[0:3])[None].to(device), cell, dis_norm)[0, 0]
        d_sr_np, d_hr_np = d_sr.cpu().numpy(), d_hr.cpu().numpy()
        sp = _spectra(d_sr_np, d_hr_np)
        metrics["density_transfer_highk"] = _highk(sp["transfer"])
        metrics["density_ratio_highk"] = _highk(sp["ratio"])
        metrics["density_rk_highk"] = _highk(sp["rk"])
        metrics["density_sigma_ratio"] = float(d_sr_np.std() / max(d_hr_np.std(), 1e-12))
        agg_spec.setdefault("density", []).append(sp)
        per_box.append(metrics)
        print(
            f"[{name}] dens P/P={metrics['density_ratio_highk']:.3f} "
            f"T={metrics['density_transfer_highk']:.3f} "
            f"sig={metrics['density_sigma_ratio']:.3f}",
            flush=True,
        )

    # Mean spectra across boxes
    npz = {}
    for fname, sps in agg_spec.items():
        for key in sps[0]:
            npz[f"{fname}__{key}"] = np.mean([s[key] for s in sps], axis=0)

    summary = {}
    for key in per_box[0]:
        if key in ("box", "seed"):
            continue
        vals = np.array([m[key] for m in per_box], dtype=np.float64)
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}

    np.savez(out / "field_table_spectra.npz", **npz)
    with open(out / "field_table.json", "w") as fh:
        json.dump({
            "freeze": str(Path(args.freeze).resolve()),
            "model": model_path,
            "sets": sets,
            "seed": seed,
            "nsplit": nsplit,
            "pad": pad,
            "device": str(device),
            "metrics_mean_std": summary,
            "per_box": per_box,
        }, fh, indent=2)

    # Markdown table
    lines = [
        "# SR2 baseline field-level table",
        "",
        f"boxes: {', '.join(sets)}  |  seed={seed}  |  nsplit={nsplit}  |  pad={pad}",
        "",
        "| field | P/P_HR (hi-k) | T(k) (hi-k) | r(k) (hi-k) |",
        "|---|---:|---:|---:|",
    ]
    for f in ("displacement", "velocity", "density"):
        lines.append(
            f"| {f} | {summary[f'{f}_ratio_highk']['mean']:.3f} "
            f"| {summary[f'{f}_transfer_highk']['mean']:.3f} "
            f"| {summary[f'{f}_rk_highk']['mean']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"density sigma-ratio: {summary['density_sigma_ratio']['mean']:.3f} "
        f"± {summary['density_sigma_ratio']['std']:.3f}"
    )
    (out / "field_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}/field_table.{{json,md,npz}}")


if __name__ == "__main__":
    main()
