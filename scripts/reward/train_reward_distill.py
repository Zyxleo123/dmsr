#!/usr/bin/env python
"""Stage 8: one offline reward-weighted distillation round (GPU).

Initialises from the frozen residual prior, keeps paired supervision throughout,
ramps ``lambda_elite`` from zero, and never regenerates candidates.

    python scripts/reward/train_reward_distill.py \
        --config configs/reward/distill_round0.yaml \
        --init-from $ZFS/dmsr_reward/checkpoints/residual_prior/ckpt_best.pt \
        --replay   $ZFS/dmsr_reward/replay/round_000/replay.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import banner, write_json

from cosmo_sr.reward import paths
from cosmo_sr.reward.train import run_training
from cosmo_sr.utils.config import apply_overrides, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/reward/distill_round0.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    ap.add_argument("--init-from", default=None, help="residual-prior checkpoint")
    ap.add_argument("--teacher", default=None, help="frozen teacher (default: init-from)")
    ap.add_argument("--replay", default=None, help="replay manifest jsonl")
    ap.add_argument("--sigma-res", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    d = cfg.setdefault("distill", {})
    if args.init_from:
        d["init_from"] = args.init_from
    if args.teacher:
        d["teacher_checkpoint"] = args.teacher
    if args.replay:
        d["replay_manifest"] = args.replay
    if args.sigma_res:
        cfg.setdefault("model", {})["sigma_res"] = json.loads(
            Path(args.sigma_res).read_text()
        )["sigma_res"]

    if not args.smoke:
        for k in ("init_from", "replay_manifest"):
            if not d.get(k):
                raise SystemExit(f"distill.{k} is required (pass --{k.replace('_', '-')})")

    out = args.out or cfg.get("output", {}).get("run_dir") or str(
        paths.CHECKPOINTS("reward_distill_round0", create=True)
    )
    if not args.smoke and not str(out).startswith(("/zfsauton/scratch", "/tmp")):
        raise SystemExit(f"refusing to write checkpoints to {out}; use a path under $ZFS")

    banner(f"reward distillation round 0 -> {out}")
    run_dir = run_training(cfg, mode="distill", smoke=args.smoke, resume=args.resume,
                           run_dir=out)
    write_json(Path(run_dir) / "stage.json", {
        "stage": "reward_distill_round0",
        "smoke": bool(args.smoke),
        "init_from": d.get("init_from"),
        "replay_manifest": d.get("replay_manifest"),
        "lambda_elite": d.get("lambda_elite"),
        "lambda_ref": d.get("lambda_ref"),
    })


if __name__ == "__main__":
    main()
