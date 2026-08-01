#!/usr/bin/env python
"""CPU stage: estimate ``mu_HR`` and ``C_reg`` from HR box summaries.

The reward is scored on a whole box, so the moments are estimated on whole
boxes: each HR box contributes exactly one summary vector -- taken from the
direct full-periodic-box catalog when one is cached, not from pooled chunks --
and ``mu``/``C`` are the mean and covariance over boxes. With ~10 boxes and ~10
active dimensions the off-diagonal covariance is not identifiable, so it is
shrunk toward the diagonal (``reward.covariance_estimator``) rather than
reported as if it had been measured.

    python scripts/reward/fit_reward_model.py --boxes set0,set1,...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (active_dims_of, add_common_args, banner, bins_of,
                     load_reward_config, parse_boxes, split_boxes, write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import pool, read_summaries
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
    ap.add_argument("--method", default="whole_box",
                    choices=["whole_box", "bootstrap"],
                    help="whole_box: one vector per box, the statistic Gate B "
                         "scores. bootstrap: the old mixed-box chunk draw, kept "
                         "for the covariance audit only")
    ap.add_argument("--covariance", default=None,
                    choices=["auto", "diagonal", "shrunk", "full"])
    ap.add_argument("--allow-chunk-pooled", action="store_true",
                    help="fit a box whose direct full-box summary is missing "
                         "from its pooled chunks instead of refusing. The pooled "
                         "vector is missing every boundary-crossing object, so "
                         "mu is then not the mean of the scored statistic")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    rcfg = cfg.get("reward", {})
    boxes = parse_boxes(args.boxes, cfg, "train")
    cache = Path(args.cache) if args.cache else paths.CATALOG_CACHE()

    chunks = []
    full_box = {}
    missing_full = []
    for b in boxes:
        p = cache / f"{b}__hr__hr.jsonl"
        if not p.is_file():
            raise SystemExit(
                f"no HR summary for {b} at {p}; run\n"
                f"  python scripts/reward/catalog_summaries.py --box {b} --source hr"
            )
        chunks.extend(read_summaries(p))
        fp = cache / f"{b}__hr__hr__fullbox.jsonl"
        if fp.is_file():
            full_box[b] = pool(read_summaries(fp))
        else:
            missing_full.append(b)
    if missing_full and not args.allow_chunk_pooled:
        raise SystemExit(
            f"no full-box HR summary for {missing_full}. The reward is scored on "
            f"the direct full-periodic-box catalog, so mu must be the mean of "
            f"that same statistic -- pooled chunks are missing every "
            f"boundary-crossing object. Re-run\n"
            f"  python scripts/reward/catalog_summaries.py --box <box> --source hr "
            f"--overwrite\n"
            f"(the Rockstar catalog is reused, so this only re-parses it), or pass "
            f"--allow-chunk-pooled and treat mu as biased."
        )
    # A chunk with zero effective volume is entirely boundary-masked: it carries
    # no information and would divide by ~0 in the number density.
    usable = [c for c in chunks if c.volume_mpc3 > 0]
    dropped = len(chunks) - len(usable)

    banner(f"fitting reward model on {len(usable)} HR chunks from {len(boxes)} boxes")
    if dropped:
        print(f"  dropped {dropped} chunks with zero core volume", flush=True)

    ens = int(args.ensemble_size or rcfg.get("ensemble_size_B", 16))
    active = active_dims_of(cfg, bins)
    n_boxes = len({c.box for c in usable})
    if len(active) < bins.dim:
        excluded = [bins.labels()[i] for i in range(bins.dim) if i not in set(active)]
        print(f"  reward dimensions: {len(active)} of {bins.dim} "
              f"(excluded: {', '.join(excluded)})", flush=True)
    if n_boxes < 2 * len(active):
        # A full covariance on D dimensions from n independent boxes has at most
        # n-1 nonzero eigenvalues; below 2D boxes the off-diagonal structure is a
        # property of the sample. shrink_covariance() drops it rather than
        # letting cond(C_reg) look like a measurement.
        print(f"  ! {n_boxes} independent boxes for {len(active)} reward "
              f"dimensions: the off-diagonal covariance is not identifiable and "
              f"is shrunk toward the diagonal. Treat any correlation in the "
              f"manifest as regularised, not measured.", flush=True)
    model = fit_reward_model(
        usable, bins,
        method=args.method,
        ensemble_size=ens,
        active_dims=active,
        full_box=full_box or None,
        covariance=str(args.covariance or rcfg.get("covariance_estimator", "auto")),
        off_diagonal_shrinkage=float(rcfg.get("off_diagonal_shrinkage", 0.5)),
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
    print(f"  method={model.meta.get('method')} "
          f"estimator={model.meta.get('estimator')} "
          f"n_boxes={model.meta.get('n_boxes')} "
          f"(source: {'full-box catalog' if full_box else 'pooled chunks'})",
          flush=True)
    print(f"  dim={model.active_dim}/{model.dim}  lambda={model.lam:.4g}  "
          f"cond(C_reg active)={cond:.3g}  "
          f"cond(C_reg full)={model.condition_number_full:.3g}", flush=True)
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
