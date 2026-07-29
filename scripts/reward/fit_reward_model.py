#!/usr/bin/env python
"""CPU stage: estimate ``mu_HR`` and ``C_reg`` from HR chunk summaries.

The covariance is the covariance of an *ensemble* summary, so it is built by
drawing ensembles of exactly the size and stratification that will be scored,
resampling boxes with replacement (boxes are the independent unit; chunks inside
a box are not).

    python scripts/reward/fit_reward_model.py --boxes set0,set1,...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (add_common_args, banner, bins_of, load_reward_config,
                     parse_boxes, split_boxes, write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import read_summaries
from cosmo_sr.reward.reward import fit_reward_model


def _strata(chunks, mode):
    """Stratification labels: box, and coarse host-mass / environment classes."""
    labels = []
    for c in chunks:
        parts = []
        if "box" in mode:
            parts.append(c.box)
        if "host_mass_class" in mode:
            # heaviest populated host bin in this chunk
            nz = np.nonzero(np.asarray(c.n_host) > 0)[0]
            parts.append(f"hm{int(nz.max()) if nz.size else -1}")
        if "environment_class" in mode:
            # total host count is a robust proxy for over/under-density
            n = float(np.sum(c.n_host))
            parts.append("env_hi" if n > 0 else "env_lo")
        labels.append("|".join(parts) or c.box)
    return labels


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--boxes", default=None, help="comma list; default = train split")
    ap.add_argument("--cache", default=None, help="catalog summary cache dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ensemble-size", type=int, default=None)
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--shrinkage", type=float, default=None)
    args = ap.parse_args()

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    rcfg = cfg.get("reward", {})
    boxes = parse_boxes(args.boxes, cfg, "train")
    cache = Path(args.cache) if args.cache else paths.CATALOG_CACHE()

    chunks = []
    for b in boxes:
        p = cache / f"{b}__hr__hr.jsonl"
        if not p.is_file():
            raise SystemExit(
                f"no HR summary for {b} at {p}; run\n"
                f"  python scripts/reward/catalog_summaries.py --box {b} --source hr"
            )
        chunks.extend(read_summaries(p))
    # A chunk with zero effective volume is entirely boundary-masked: it carries
    # no information and would divide by ~0 in the number density.
    usable = [c for c in chunks if c.volume_mpc3 > 0]
    dropped = len(chunks) - len(usable)

    banner(f"fitting reward model on {len(usable)} HR chunks from {len(boxes)} boxes")
    if dropped:
        print(f"  dropped {dropped} chunks with zero core volume", flush=True)

    ens = int(args.ensemble_size or rcfg.get("ensemble_size_B", 16))
    model = fit_reward_model(
        usable, bins,
        ensemble_size=ens,
        n_draws=int(args.draws or rcfg.get("bootstrap_draws", 400)),
        shrinkage=float(args.shrinkage if args.shrinkage is not None
                        else rcfg.get("shrinkage", 0.1)),
        strata=_strata(usable, rcfg.get("strata", ["box"])),
        seed=int(rcfg.get("bootstrap_seed", 0)),
    )
    out = Path(args.out) if args.out else paths.subdir("reward_model", create=True) / \
        "reward_model.json"
    write_json(out, model.to_dict())

    cond = model.condition_number
    diag = np.sqrt(np.diag(model.cov))
    print(f"  dim={model.dim}  lambda={model.lam:.4g}  cond(C_reg)={cond:.3g}", flush=True)
    print(f"  per-bin bootstrap sigma: min={diag.min():.4g} max={diag.max():.4g}",
          flush=True)
    if cond > 1e4:
        print("  ! cond(C_reg) > 1e4: the reward is dominated by one nearly "
              "unconstrained direction; raise reward.shrinkage before using it",
              flush=True)
    elif cond > 1e3:
        print("  cond(C_reg) is high but workable; keep an eye on which bin "
              "dominates the Mahalanobis contributions", flush=True)
    else:
        print("  conditioning looks healthy", flush=True)
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
