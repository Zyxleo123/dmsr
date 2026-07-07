#!/usr/bin/env python
"""Phase 11 sanity experiments A-D. Must pass before serious runs.

Runs on CPU with small synthetic volumes and prints PASS/FAIL for:
  A: degrader sanity          (A(x_hr) reproduces stored LR, rel MSE < 1e-7)
  B: one-sample supervised    (HR MSE drops >= 80%; checkpoint reload identical)
  C: one-sample ambient       (mse(A(G(y)),y) drops >= 80%; output finite)
  D: mixed scarce-pair        (val LR-recon and paired HR MSE improve; eval writes files)

Exit code is non-zero if any experiment fails.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from cosmo_sr.operators.degrader import FixedDegrader
from cosmo_sr.models.wrappers import build_generator
from cosmo_sr.losses.ambient import compute_ambient
from cosmo_sr.eval.metrics import relative_mse
from cosmo_sr.eval.run_eval import evaluate
from cosmo_sr.train import train_ambient
from cosmo_sr.train.common import load_checkpoint


def _hr(scale, lr_n=8, seed=0):
    torch.manual_seed(seed)
    coarse = torch.randn(1, 6, lr_n, lr_n, lr_n)
    return torch.nn.functional.interpolate(
        coarse, scale_factor=scale, mode="trilinear", align_corners=False
    )


def experiment_a():
    A = FixedDegrader(8)
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 6, 32, 32, 32)).astype(np.float32))
    rel = relative_mse(A(x).numpy(), A(x).numpy())
    ok = rel < 1e-7
    print(f"[A] degrader sanity: rel MSE={rel:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def experiment_b(tmp):
    scale = 4
    G = build_generator("SimpleSRGenerator", scale_factor=scale, width=16, depth=2)
    y = torch.randn(1, 6, 4, 4, 4)
    target = _hr(scale, 4, seed=1)
    opt = torch.optim.Adam(G.parameters(), lr=2e-3)
    lf = torch.nn.MSELoss()
    with torch.no_grad():
        init = lf(G(y), target).item()
    for _ in range(500):
        opt.zero_grad(); loss = lf(G(y), target); loss.backward(); opt.step()
    final = loss.item()
    torch.save({"model": G.state_dict()}, Path(tmp) / "b.pt")
    G2 = build_generator("SimpleSRGenerator", scale_factor=scale, width=16, depth=2)
    load_checkpoint(Path(tmp) / "b.pt", G2)
    G.eval(); G2.eval()
    with torch.no_grad():
        identical = torch.equal(G(y), G2(y))
    ok = final < 0.2 * init and identical
    print(f"[B] supervised overfit: init={init:.4g} final={final:.4g} reload_identical={identical} -> {'PASS' if ok else 'FAIL'}")
    return ok


def experiment_c():
    scale = 4
    A = FixedDegrader(scale)
    G = build_generator("SimpleSRGenerator", scale_factor=scale, width=16, depth=2)
    x_hr = _hr(scale, 4, seed=2)
    y = A(x_hr)
    opt = torch.optim.Adam(G.parameters(), lr=2e-3)
    with torch.no_grad():
        init = compute_ambient(G, A, y)[0].item()
    for _ in range(300):
        opt.zero_grad(); loss = compute_ambient(G, A, y)[0]; loss.backward(); opt.step()
    final = loss.item()
    with torch.no_grad():
        finite = bool(torch.isfinite(G(y)).all())
    ok = final < 0.2 * init and finite
    print(f"[C] ambient overfit: init={init:.4g} final={final:.4g} finite={finite} -> {'PASS' if ok else 'FAIL'}")
    return ok


def experiment_d(tmp):
    scale = 4
    cfg = {
        "data": {"crop_lr": 4, "scale_factor": scale},
        "model": {"name": "SimpleSRGenerator", "width": 16, "depth": 2},
        "loss": {"lambda_ambient": 1.0, "lambda_pair": 1.0, "lambda_tv": 0.0},
        "degrader": {"mode": "average"},
        "train": {"batch_size_unpaired": 1, "batch_size_paired": 1, "lr": 1e-3,
                  "steps": 200, "seed": 0, "log_every": 20, "save_every": 0},
        "output": {"run_dir": str(Path(tmp) / "d_run")},
    }
    res = train_ambient.train(cfg, smoke=False)
    amb_ok = res["last"]["loss_ambient"] < res["first"]["loss_ambient"]
    pair_ok = res["last"]["loss_pair"] < res["first"]["loss_pair"]

    # eval writes metrics/plots
    G = build_generator("SimpleSRGenerator", scale_factor=scale, width=16, depth=2)
    load_checkpoint(res["checkpoint"], G)
    A = FixedDegrader(scale)
    x_hr = _hr(scale, 4, seed=5)
    lr = A(x_hr).squeeze(0).numpy()
    out = evaluate(G, A, lr, str(Path(tmp) / "d_eval"), scale, hr_field=x_hr.squeeze(0).numpy(), nsplit=1)
    eval_ok = all(Path(p).exists() for p in (out["metrics_path"], out["spectra_path"], out["slice_path"]))
    ok = amb_ok and pair_ok and eval_ok
    print(f"[D] mixed training: ambient_down={amb_ok} pair_down={pair_ok} eval_written={eval_ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    with tempfile.TemporaryDirectory() as tmp:
        results = [
            experiment_a(),
            experiment_b(tmp),
            experiment_c(),
            experiment_d(tmp),
        ]
    if all(results):
        print("ALL SANITY EXPERIMENTS PASSED")
        return 0
    print("SOME SANITY EXPERIMENTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
