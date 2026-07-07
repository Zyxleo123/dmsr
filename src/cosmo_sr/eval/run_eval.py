"""Evaluation runner: writes metrics.json, spectra.npz and slice PNG(s).

Usage::

    python -m cosmo_sr.eval.run_eval --config configs/ambient_smoke.yaml \
        --checkpoint runs/ambient/ckpt_last.pt --lr LR.npy [--hr HR.npy] --out eval_out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..data.field_io import load_field
from ..operators.degrader import FixedDegrader
from ..models.wrappers import build_generator, model_config_kwargs
from ..inference.tile_sr import super_resolve_full_box
from .metrics import compute_metrics
from .spectra import power_spectrum
from .slices import save_eval_slices


def evaluate(
    model: torch.nn.Module,
    degrader: FixedDegrader,
    lr_field,
    out_dir: str,
    scale_factor: int,
    hr_field=None,
    boxsize: Optional[float] = None,
    nsplit: int = 1,
    pad_lr: int = 0,
) -> Dict[str, Any]:
    """Run SR + metrics + spectra + slices and write outputs to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lr_np = np.asarray(lr_field, dtype=np.float32)
    x_hat = super_resolve_full_box(model, lr_np, scale_factor, nsplit, pad_lr)
    x_hat = np.asarray(x_hat, dtype=np.float32)

    with torch.no_grad():
        y_recon = degrader(torch.from_numpy(x_hat).unsqueeze(0)).squeeze(0).cpu().numpy()

    metrics = compute_metrics(degrader, x_hat, lr_np, x_hr=hr_field)

    # spectra per channel of generated HR (and HR target if present)
    spectra: Dict[str, np.ndarray] = {}
    for c in range(x_hat.shape[0]):
        k, pk = power_spectrum(x_hat[c], boxsize=boxsize)
        spectra[f"k_gen_c{c}"] = k
        spectra[f"Pk_gen_c{c}"] = pk
    if hr_field is not None:
        hr_np = np.asarray(hr_field, dtype=np.float32)
        for c in range(hr_np.shape[0]):
            k, pk = power_spectrum(hr_np[c], boxsize=boxsize)
            spectra[f"k_hr_c{c}"] = k
            spectra[f"Pk_hr_c{c}"] = pk

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    spectra_path = out_dir / "spectra.npz"
    np.savez(spectra_path, **spectra)

    slice_path = out_dir / "slices.png"
    save_eval_slices(slice_path, lr_np, x_hat, y_recon=y_recon, x_hr=hr_field, channel=0)

    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "spectra_path": str(spectra_path),
        "slice_path": str(slice_path),
        "out_dir": str(out_dir),
    }


def _build_model_from_cfg(cfg: Dict[str, Any], scale: int) -> torch.nn.Module:
    model_cfg = dict(cfg.get("model", {"name": "SimpleSRGenerator"}))
    kwargs = model_config_kwargs(model_cfg)
    kwargs.setdefault("scale_factor", scale)
    return build_generator(model_cfg.get("name", "SimpleSRGenerator"), **kwargs)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an SR checkpoint")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=False, type=str, default=None)
    parser.add_argument("--lr", type=str, default=None, help="LR .npy input")
    parser.add_argument("--hr", type=str, default=None, help="optional HR target .npy")
    parser.add_argument("--out", type=str, default="eval_out")
    parser.add_argument("--nsplit", type=int, default=1)
    parser.add_argument("--pad-lr", type=int, default=0)
    parser.add_argument("--boxsize", type=float, default=None)
    args = parser.parse_args(argv)

    from ..utils.config import load_config

    cfg = load_config(args.config)
    scale = int(cfg.get("data", {}).get("scale_factor", 8))
    model = _build_model_from_cfg(cfg, scale)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
    model.eval()

    degrader = FixedDegrader(scale, mode=cfg.get("degrader", {}).get("mode", "average"))

    if args.lr:
        lr_field = load_field(args.lr)
    else:
        # synthesize a small LR box so eval runs without real data
        from ..data.datasets import SyntheticSRDataset

        sample = SyntheticSRDataset(1, crop_lr=8, scale_factor=scale)[0]
        lr_field = sample["lr"].numpy()

    hr_field = load_field(args.hr) if args.hr else None

    result = evaluate(
        model, degrader, lr_field, args.out, scale, hr_field=hr_field,
        boxsize=args.boxsize, nsplit=args.nsplit, pad_lr=args.pad_lr,
    )
    print(f"[eval] wrote {result['metrics_path']}, {result['spectra_path']}, {result['slice_path']}")


if __name__ == "__main__":
    main()
