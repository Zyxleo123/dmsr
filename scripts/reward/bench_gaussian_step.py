#!/usr/bin/env python
"""Time one reward-only training step of the Gaussian residual policy.

The step in ``scripts/reward/train_gaussian_reward.py`` is three forward passes
plus one backward: the *behaviour* policy (no_grad, to reconstruct ``a_i``), the
*reference* policy (no_grad, for the closed-form KL), and the trained policy
(forward + backward). This script reproduces exactly that sequence on synthetic
tensors of the configured crop shape, so the per-step cost can be measured
before a replay buffer exists.

Synthetic inputs are legitimate here: the cost of a convolution does not depend
on what the array contains, only on its shape. What is *not* measured is the
replay crop I/O (``periodic_crop`` off an mmap'd 512^3 base), which is reported
separately as a note, not folded into the number.

    sbatch scripts/slurm/bench_gaussian_step_gpu.sbatch
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from _common import (add_common_args, banner, load_reward_config, select_device,
                     write_json)
from sample_gaussian_candidates import load_policy

from cosmo_sr.utils.config import load_config


def bench_one(policy, behavior, reference, *, core: int, margin: int, batch: int,
              sf: int, steps: int, warmup: int, device, beta: float,
              lam_edit: float, lr_: float, normalize: bool) -> Dict:
    n_hr = core + 2 * margin
    n_lr = n_hr // sf
    c = int(policy.cfg.channels)

    base = torch.randn(batch, c, n_hr, n_hr, n_hr, device=device)
    y_lr = torch.randn(batch, c, n_lr, n_lr, n_lr, device=device)
    w = torch.ones(batch, device=device)
    opt_ = torch.optim.AdamW(policy.parameters(), lr=lr_)

    def one_step() -> None:
        with torch.no_grad():
            b_dist = behavior.distribution(base, y_lr)
            acts = {}
            for s in behavior.specs:
                # Same shape and same per-step CPU cost as the training loop:
                # one global_noise draw per batch element, stacked.
                eps = torch.stack([
                    behavior.window_noise(i, origin=(0, 0, 0), n_hr=n_hr,
                                          grid_hr=512, batch=1)[s.name][0]
                    for i in range(batch)]).to(device)
                mu, sig = b_dist[s.name]
                acts[s.name] = (mu + sig * eps).detach()
            behavior.log_prob(b_dist, acts).detach()

        dist = policy.distribution(base, y_lr)
        logp = policy.log_prob(dist, acts)
        denom = float(policy.n_action_elements(n_hr)) if normalize else 1.0
        nll = -(w * logp / denom).sum() / w.sum().clamp_min(1e-12)
        with torch.no_grad():
            ref_dist = reference.distribution(base, y_lr)
        kl = policy.kl_to(dist, ref_dist).mean() / denom
        mu_h = policy.combine({s.name: dist[s.name][0] for s in policy.specs}, n_hr)
        c_edit = (torch.tanh(mu_h) ** 2).mean()
        (nll + beta * kl + lam_edit * c_edit).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_.step()
        opt_.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times: List[float] = []
    for _ in range(steps):
        t0 = time.time()
        one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

    times.sort()
    med = times[len(times) // 2]
    peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
    return {
        "core_hr": core, "context_margin": margin, "tile_hr": n_hr,
        "batch": batch, "volume_overhead": (n_hr / core) ** 3,
        "seconds_per_step_median": med,
        "seconds_per_step_min": times[0], "seconds_per_step_max": times[-1],
        "peak_gib": peak,
        "hours_per_10k_steps": med * 10000 / 3600.0,
        "hours_per_20k_steps": med * 20000 / 3600.0,
    }


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--model-config", default="configs/reward/gaussian_policy.yaml")
    ap.add_argument("--steps", type=int, default=10, help="timed steps per shape")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--shapes", default="128:56:2",
                    help="comma list of core:margin:batch to time. Prefer ONE "
                         "shape per process: a shape that OOMs leaves its graph "
                         "reachable from the exception, so the next shape in the "
                         "same process starts with a poisoned allocator (job "
                         "25850: 44.8 -> 46.0 -> 46.7 GiB in use across three "
                         "'independent' shapes)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--append", action="store_true",
                    help="merge these rows into --out instead of replacing it, "
                         "so one process per shape still yields one table")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_reward_config(args)
    mc = load_config(args.model_config)
    full = {**cfg, "model": mc.get("model", {}), "correction": mc.get("correction", {})}
    tcfg = dict(mc.get("train", {}))
    device = select_device(args.device)
    sf = int(cfg["data"]["scale_factor"])

    policy = load_policy(None, full, device).train()
    behavior = load_policy(None, full, device).eval()
    reference = load_policy(None, full, device).eval()
    for p in list(behavior.parameters()) + list(reference.parameters()):
        p.requires_grad_(False)

    n_par = sum(p.numel() for p in policy.parameters())
    banner(f"per-step benchmark: {n_par/1e6:.2f}M parameters, device={device}, "
           f"{args.steps} timed steps after {args.warmup} warmup")

    rows = []
    for shape in args.shapes.split(","):
        core, margin, batch = (int(v) for v in shape.split(":"))
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            r = bench_one(policy, behavior, reference, core=core, margin=margin,
                          batch=batch, sf=sf, steps=args.steps, warmup=args.warmup,
                          device=device, beta=float(tcfg.get("beta_kl", 0.01)),
                          lam_edit=float(tcfg.get("lambda_edit", 1e-3)),
                          lr_=float(tcfg.get("lr", 1e-4)),
                          normalize=bool(tcfg.get("normalize_log_prob", True)))
        except torch.cuda.OutOfMemoryError as exc:
            # The peak BEFORE the refused allocation is the useful number: it
            # says how far this shape got, which a bare "OOM" does not.
            peak = torch.cuda.max_memory_allocated() / 2**30
            first = str(exc).split(". ")[0]
            print(f"[core={core} margin={margin} batch={batch}] "
                  f"tile {core + 2 * margin}^3 OOM at {peak:.1f} GiB allocated: "
                  f"{first}", flush=True)
            rows.append({"core_hr": core, "context_margin": margin, "batch": batch,
                         "tile_hr": core + 2 * margin, "oom": True,
                         "peak_gib_before_oom": peak})
            gc.collect()
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        print(f"[core={core} margin={margin} batch={batch}] tile {r['tile_hr']}^3 "
              f"({r['volume_overhead']:.1f}x volume)  "
              f"{r['seconds_per_step_median']:.2f} s/step  "
              f"peak {r['peak_gib']:.1f} GiB  "
              f"-> {r['hours_per_20k_steps']:.0f} h for 20k steps", flush=True)

    out = args.out or "bench_gaussian_step.json"
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else ""
    if args.append and Path(out).is_file():
        prev = json.loads(Path(out).read_text()).get("rows", [])
        key = lambda r: (r["core_hr"], r["context_margin"], r["batch"])  # noqa: E731
        new = {key(r) for r in rows}
        rows = [r for r in prev if key(r) not in new] + rows
        rows.sort(key=key, reverse=True)
    write_json(out, {"parameters": n_par, "device": str(device), "gpu": gpu,
                     "total_gib": (torch.cuda.get_device_properties(0).total_memory
                                   / 2**30) if device.type == "cuda" else 0.0,
                     "rows": rows})
    print(json.dumps(rows, indent=2))
    banner(f"-> {out}")


if __name__ == "__main__":
    main()
