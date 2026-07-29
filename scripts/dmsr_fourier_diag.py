"""Offline Fourier-band loss diagnostic + gate for a DMSR checkpoint (Parts 3/4).

Runs the *non-mutating* diagnostic on a trained Stage C or Experiment E checkpoint
using held-out validation (or training) crops -- never the 12-box benchmark. For a
set of representative flow times (early / middle / late ``t``) and several crops it
accumulates the per-channel, per-shell statistics, writes a machine-readable JSON
and a human-readable report, optional per-channel / per-k plots, and applies the
Part-4 decision gate (``NO_IMBALANCE`` / ``IMBALANCE_SUPPORTED`` / ``INCONCLUSIVE``).

Example
-------
    PYTHONPATH=src python scripts/dmsr_fourier_diag.py \
        --config configs/dmsr/stage_c_critic_pairedlr.yaml \
        --ckpt runs/dmsr/stage_c_s0/ckpt_best.pt \
        --split val --max-batches 8 \
        --out runs/dmsr/fourier_diag/stage_c_s0

The verdict is a control gate: **do not launch the spectral-balancing runs until
this report is reviewed** and returns ``IMBALANCE_SUPPORTED`` stably.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from cosmo_sr.dmsr.data import build_paired_dataset, build_val_dataset, resolve_split
from cosmo_sr.data.datasets import finite_loader
from cosmo_sr.dmsr.flow import build_flow, null_space_flow_loss
from cosmo_sr.dmsr.fourier_diag import (
    BandDiagnosticAccumulator,
    ShellDiagnostic,
    gate_decision,
)
from cosmo_sr.dmsr.mean_innovation import build_mean_innovation, innovation_flow_loss
from cosmo_sr.utils.config import apply_overrides, load_config


def _parse_t_bins(specs: List[str]) -> List[Tuple[str, float]]:
    out = []
    for s in specs:
        name, val = s.split(":")
        out.append((name, float(val)))
    return out


def load_model(cfg, ckpt_path, channels, device, prefer_ema=True):
    use_mi = bool(cfg.get("mean_innovation", {}).get("enabled", False))
    if use_mi:
        model = build_mean_innovation(cfg, channels, device, load_ckpts=True)
    else:
        model = build_flow(cfg, channels).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = None
    if prefer_ema and isinstance(state, dict) and "extra" in state \
            and isinstance(state["extra"], dict) and "ema" in state["extra"]:
        sd = state["extra"]["ema"]
        which = "ema"
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
        which = "model"
    else:
        sd, which = state, "raw"
    model.load_state_dict(sd)
    model.eval()
    return model, use_mi, which


def compute_fields(model, use_mi, y, x, t_val, z):
    """(v_pred, v_target) at a fixed flow time ``t_val`` with a fixed ``z``."""
    b = y.shape[0]
    t = torch.full((b,), float(t_val), device=y.device, dtype=y.dtype)
    loss_fn = innovation_flow_loss if use_mi else null_space_flow_loss
    _, _, fields = loss_fn(model, y, x, t=t, z=z, return_fields=True)
    return fields["v_pred"], fields["v_target"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--set", nargs="*", default=None)
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--t-bins", nargs="*", default=["early:0.1", "mid:0.5", "late:0.9"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true", help="use the online model, not EMA")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor = int(cfg.get("factor", 8))
    dcfg = cfg.get("data", {})
    use_channels = dcfg.get("use_channels")
    channels = len(use_channels) if use_channels else int(dcfg.get("channels", 6))
    ch_names = [f"disp{c}" if c < 3 else f"vel{c-3}" for c in (use_channels or range(channels))]

    model, use_mi, which = load_model(cfg, args.ckpt, channels, device,
                                      prefer_ema=not args.no_ema)
    print(f"[diag] loaded {'mean+innovation' if use_mi else 'null-space flow'} "
          f"from {args.ckpt} (weights={which})")

    split = resolve_split(dcfg)
    crop_lr = int(dcfg.get("crop_lr", 8))
    if args.split == "val":
        ds = build_val_dataset(split, crop_lr=crop_lr, scale_factor=factor,
                               channels=int(dcfg.get("channels", 6)), use_channels=use_channels,
                               mmap=bool(dcfg.get("mmap", True)), max_crops=args.max_batches * args.batch_size)
    else:
        ds = build_paired_dataset(split, crop_lr=crop_lr, scale_factor=factor, seed=args.seed,
                                  augment=False, channels=int(dcfg.get("channels", 6)),
                                  use_channels=use_channels, mmap=bool(dcfg.get("mmap", True)))
    loader = finite_loader(ds, args.batch_size)
    t_bins = _parse_t_bins(args.t_bins)

    low_frac = float(cfg.get("eval", {}).get("low_frac", 0.5))
    high_frac = float(cfg.get("eval", {}).get("high_frac", 1.5))

    # one pooled accumulator per t-bin (for the report/plots) + one diagnostic per
    # (t-bin, batch) for the gate's stability criterion.
    pooled = {name: BandDiagnosticAccumulator(factor=factor, low_frac=low_frac,
                                              high_frac=high_frac, channel_names=ch_names,
                                              flow_time_bin=name) for name, _ in t_bins}
    per_batch_diags: List[ShellDiagnostic] = []

    torch.manual_seed(args.seed)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.max_batches:
                break
            y = batch["lr"].to(device)
            x = batch["hr"].to(device)
            # fixed z per batch so flow-time bins are compared on the same noise
            z = torch.randn(x.shape, device=device, dtype=x.dtype)
            for name, tv in t_bins:
                vp, vt = compute_fields(model, use_mi, y, x, tv, z)
                pooled[name].add(vp, vt, model.operator)
                acc_b = BandDiagnosticAccumulator(factor=factor, low_frac=low_frac,
                                                  high_frac=high_frac, channel_names=ch_names,
                                                  flow_time_bin=name)
                acc_b.add(vp, vt, model.operator)
                per_batch_diags.append(acc_b.result())

    diag_by_bin = {name: acc.result() for name, acc in pooled.items()}
    report = gate_decision(per_batch_diags)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    machine = {
        "config": args.config, "ckpt": args.ckpt, "weights": which,
        "model": "mean_innovation" if use_mi else "null_space_flow",
        "split": args.split, "n_batches": args.max_batches, "batch_size": args.batch_size,
        "channels": ch_names, "t_bins": dict(t_bins),
        "gate": report.to_dict(),
        "per_flow_time": {name: d.to_dict() for name, d in diag_by_bin.items()},
    }
    with open(out / "fourier_diag.json", "w") as f:
        json.dump(machine, f, indent=2)

    human = [report.human_summary(), "", "Per-flow-time band totals (pooled channels):"]
    for name, d in diag_by_bin.items():
        human.append(f"\n[t={dict(t_bins)[name]:.2f} :: {name}]")
        for band, v in d.band_totals().items():
            human.append(f"  {band:11s} loss_frac={v['loss_fraction']:.4f} "
                         f"rel_err={v['relative_shell_error']:.4g} "
                         f"tgt_energy={v['target_shell_energy']:.4g}")
    (out / "fourier_diag_report.txt").write_text("\n".join(human))
    print("\n".join(human))
    print(f"\n[diag] wrote {out/'fourier_diag.json'} and {out/'fourier_diag_report.txt'}")

    if args.plots:
        _plots(diag_by_bin, out)


def _plots(diag_by_bin: Dict[str, ShellDiagnostic], out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[diag] plotting skipped ({e})")
        return
    quantities = [
        ("target_power_per_mode", "target power / mode", True),
        ("target_shell_energy", "target shell energy", True),
        ("error_power_per_mode", "error power / mode", True),
        ("error_shell_energy", "error shell energy", True),
        ("loss_fraction", "loss fraction", False),
        ("relative_shell_error", "relative shell error", True),
    ]
    for qkey, qlabel, logy in quantities:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, d in diag_by_bin.items():
            arr = getattr(d, qkey)                    # (C, S)
            pooled = arr.sum(axis=0) if "energy" in qkey or qkey == "loss_fraction" else arr.mean(axis=0)
            ax.plot(d.shell_k, np.maximum(pooled, 1e-30), label=name, marker=".")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("radial shell k (integer modes)")
        ax.set_ylabel(qlabel)
        ax.legend()
        ax.set_title(qlabel)
        fig.tight_layout()
        fig.savefig(out / f"plot_{qkey}.png", dpi=100)
        plt.close(fig)
    # per-channel relative error at the middle flow-time
    mid = list(diag_by_bin.values())[len(diag_by_bin) // 2]
    fig, ax = plt.subplots(figsize=(6, 4))
    for ci, cname in enumerate(mid.channel_names):
        ax.plot(mid.shell_k, np.maximum(mid.relative_shell_error[ci], 1e-30), label=cname, marker=".")
    ax.set_yscale("log")
    ax.set_xlabel("radial shell k")
    ax.set_ylabel("relative shell error")
    ax.set_title(f"per-channel relative error (t bin: {mid.flow_time_bin})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plot_per_channel_relative_error.png", dpi=100)
    plt.close(fig)
    print(f"[diag] wrote plots to {out}")


if __name__ == "__main__":
    main()
