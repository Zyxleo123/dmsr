#!/usr/bin/env python
"""Stage 3: train the supervised residual diffusion prior (GPU).

Ordinary denoising objective on real paired residuals; no catalog reward.

    python scripts/reward/train_residual_prior.py \
        --config configs/reward/residual_prior.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import PROJECT_ROOT, banner, load_reward_config, write_json

from cosmo_sr.reward import paths
from cosmo_sr.reward.train import run_training
from cosmo_sr.utils.config import apply_overrides, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/reward/residual_prior.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    ap.add_argument("--out", default=None, help="run dir; must be under $ZFS")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--sigma-res", default=None,
                    help="sigma_res.json from the residual-target audit")
    ap.add_argument("--fixed-hosts", default=None,
                    help="fixed_hosts.json from select_fixed_hosts.py; restricts "
                         "training crops to those chunks (rung 3 overfitting)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.fixed_hosts:
        h = json.loads(Path(args.fixed_hosts).read_text())
        cfg.setdefault("data", {})["fixed_host_chunks"] = [
            (str(b), int(c)) for b, c in h["chunks"]
        ]
        print(f"fixed hosts <- {args.fixed_hosts}: {len(h['chunks'])} chunks",
              flush=True)
    if args.sigma_res:
        s = json.loads(Path(args.sigma_res).read_text())
        cfg.setdefault("model", {})["sigma_res"] = s["sigma_res"]
        print(f"sigma_res <- {args.sigma_res}: {s['sigma_res']}", flush=True)

    out = args.out or cfg.get("output", {}).get("run_dir")
    if not out:
        out = str(paths.CHECKPOINTS("residual_prior", create=True))
    if not args.smoke and not str(out).startswith(("/zfsauton/scratch", "/tmp")):
        raise SystemExit(
            f"refusing to write checkpoints to {out}: home has a tight quota, "
            f"use a path under $ZFS"
        )
    banner(f"residual prior -> {out}")
    run_dir = run_training(cfg, mode="prior", smoke=args.smoke, resume=args.resume,
                           run_dir=out)
    write_json(Path(run_dir) / "stage.json",
               {"stage": "residual_prior", "smoke": bool(args.smoke)})


if __name__ == "__main__":
    main()
