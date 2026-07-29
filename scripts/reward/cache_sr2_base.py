#!/usr/bin/env python
"""Cache the frozen SR2 baseline field for a list of boxes (GPU, ~20 s/box).

Every later stage -- residual targets, oracle candidates, counterfactual
summaries, evaluation -- conditions on the *same* frozen output, so it is
generated once and memory-mapped afterwards. 3.2 GB per box, on scratch only.

    python scripts/reward/cache_sr2_base.py --boxes set0,set1 --seed 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, chunk_grid, load_freeze,
                     load_reward_config, lr_path, parse_boxes, select_device,
                     write_json, PROJECT_ROOT)

from cosmo_sr.reward import paths
from cosmo_sr.reward.base import FrozenSR2Base


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--boxes", default=None, help="comma list; default = all splits")
    ap.add_argument("--seed", type=int, default=0, help="frozen SR2 realisation seed")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="cache dir (default $ZFS/dmsr_reward/cache/sr2_base)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    freeze = load_freeze(cfg)
    if args.boxes:
        boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    else:
        s = cfg["split"]
        boxes = list(s["train_boxes"]) + list(s["val_boxes"]) + list(s["test_boxes"])

    cache = Path(args.out) if args.out else paths.SR2_BASE_CACHE(create=True)
    cache.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    model_path = PROJECT_ROOT / freeze["model"]["path"]

    banner(f"caching SR2 base for {len(boxes)} boxes on {device} -> {cache}")
    base = FrozenSR2Base(
        model_path,
        scale_factor=int(freeze["model"]["scale_factor"]),
        nsplit=int(freeze["inference"]["nsplit"]),
        pad=int(freeze["inference"]["pad"]),
        noise_mode=str(freeze["inference"]["noise_mode"]),
        redshift=float(freeze.get("redshift", 0.0)),
        device=device,
        cache_dir=cache,
    )
    base.assert_frozen()

    rows = []
    for b in boxes:
        path = base.cache_path(b, args.seed)
        if path.is_file() and not args.overwrite:
            print(f"[{b}] cached already -> {path.name}", flush=True)
            rows.append({"box": b, "path": str(path), "regenerated": False})
            continue
        lr = np.load(lr_path(cfg, b))
        t0 = time.time()
        field = base.base_field(lr, args.seed, box=b, use_cache=True, mmap=False)
        dt = time.time() - t0
        print(f"[{b}] {tuple(field.shape)} in {dt:.1f}s -> {path.name}", flush=True)
        rows.append({"box": b, "path": str(path), "regenerated": True, "seconds": dt})

    write_json(cache / "cache_manifest.json", {
        "seed": int(args.seed),
        "model_sha": base.model_sha(),
        "model_path": str(model_path),
        "spec": base.spec(boxes[0], args.seed).__dict__ if boxes else {},
        "boxes": rows,
    })
    banner("done")


if __name__ == "__main__":
    main()
