#!/usr/bin/env python
"""Measure the Gaussian policy's conditioning receptive field, and the margin it needs.

The parallel of ``scripts/reward/measure_receptive_field.py`` for the one-step
policy, which has a different call signature (no ``u_t``, no ``t``) and a
different answer: the policy's ``y_lr`` input is block-upsampled by
``scale_factor`` before the trunk sees it, so perturbing ONE LR cell perturbs an
8-cell block and the reach is materially wider than the diffusion trunk's 41.

Why this is a submitted job and not a one-liner: a converged answer needs a
probe box comfortably larger than twice the reach, and at width 48 a 128^3 probe
already allocates several GB per stage. Run too small and the perturbation wraps
around the periodic padding and the reported number *saturates at the probe's own
half-width* -- a silent underestimate that would then be used to justify too
small a tile margin, which is exactly the seam the margin exists to prevent.
This script refuses a saturated measurement rather than reporting it.

    python scripts/reward/measure_policy_receptive_field.py --sizes 128,192,256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import add_common_args, banner, load_reward_config, write_json
from sample_gaussian_candidates import build_policy

from cosmo_sr.reward import paths
from cosmo_sr.reward.gaussian_policy import (analytic_receptive_field,
                                             policy_receptive_field)
from cosmo_sr.reward.sampling import tile_margin_for
from cosmo_sr.utils.config import load_config


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--model-config", default="configs/reward/gaussian_policy.yaml")
    ap.add_argument("--sizes", default="128,192,256",
                    help="probe box sizes, increasing; the answer has converged "
                         "when two consecutive sizes agree and neither saturates")
    ap.add_argument("--tile-core", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_reward_config(args)
    mc = load_config(args.model_config)
    if args.threads:
        torch.set_num_threads(int(args.threads))

    policy = build_policy({"model": mc.get("model", {}),
                           "correction": mc.get("correction", {})})
    sf = int(cfg["data"]["scale_factor"])
    core = int(args.tile_core or mc.get("sampling", {}).get("tile_core", 128))
    configured = int(mc.get("sampling", {}).get("tile_margin", 0))

    rows = []
    for size in [int(s) for s in args.sizes.split(",") if s.strip()]:
        rf = policy_receptive_field(policy, size=size)
        # The probe perturbs at an aligned interior point; if the reported reach
        # is within one alignment step of the largest distance the box can
        # express, the perturbation wrapped and the number is a floor, not a
        # measurement.
        align = sf * policy.divisor
        j = max(size // 2 // align, 1) * align
        max_expressible = max(j, size - 1 - j)
        saturated = rf >= max_expressible - align
        rows.append({
            "probe_size": size, "receptive_field_halfwidth": int(rf),
            "min_tile_margin": int(tile_margin_for(rf, sf)),
            "max_expressible": int(max_expressible), "saturated": bool(saturated),
        })
        print(f"probe {size:4d}^3 -> rf={rf:4d}  min margin={tile_margin_for(rf, sf):4d}"
              + ("   SATURATED (probe too small; this is a floor)" if saturated
                 else "   converged"), flush=True)

    # The closed form is exact for a stack of local operators, so it says what
    # the probe SHOULD converge to. Disagreement is informative in both
    # directions: a probe below it has not converged, and a probe above it means
    # some layer is not local (a spatially-reducing norm is the way that
    # happens), in which case no finite margin makes tiling exact.
    analytic = analytic_receptive_field(policy.cfg)
    print(f"\nanalytic: trunk half-width {analytic['trunk_halfwidth']}, "
          f"+{sf} for the block-upsampled y_lr path "
          f"-> {analytic['policy_halfwidth']}", flush=True)

    good = [r for r in rows if not r["saturated"]]
    if not good:
        raise SystemExit(
            "every probe saturated: the receptive field is at least "
            f"{max(r['receptive_field_halfwidth'] for r in rows)} cells but was "
            f"never resolved. Re-run with larger --sizes on a node with more "
            f"memory. Do NOT set a tile margin from a saturated probe -- the "
            f"analytic expectation is {analytic['policy_halfwidth']}, so a probe "
            f"box of at least {4 * analytic['policy_halfwidth']} should resolve it."
        )
    rf = max(r["receptive_field_halfwidth"] for r in good)
    # The margin comes from the ANALYTIC bound, not the probe. The probe's
    # all-positive 1/fan_in weights let the outermost shells decay below the
    # float32 noise floor, so it under-reports by more the deeper the net --
    # exactly 0 at levels=1 and 34 cells at levels=3. Those truncated shells are
    # real support, merely small, and a margin set from the probe would leave
    # them outside the valid core.
    rf_for_margin = max(rf, analytic["policy_halfwidth"])
    need = tile_margin_for(rf_for_margin, sf)

    if rf > analytic["policy_halfwidth"]:
        print(f"\n  ! measured {rf} EXCEEDS the analytic {analytic['policy_halfwidth']}. "
              f"The formula assumes every layer is spatially local; a norm that "
              f"reduces over space (nn.GroupNorm rather than ChannelGroupNorm3d) "
              f"breaks that, and then NO margin makes tiling exact. Check "
              f"model.norm before trusting any full-box number.", flush=True)
    elif rf < analytic["policy_halfwidth"]:
        print(f"\n  note: measured {rf} < analytic {analytic['policy_halfwidth']}; "
              f"the outer shells are below the float32 noise floor of the probe. "
              f"The margin below uses the analytic bound.", flush=True)

    verdict = {
        "receptive_field_halfwidth_measured": int(rf),
        "receptive_field_halfwidth_used": int(rf_for_margin),
        "min_tile_margin": int(need),
        "margin_source": ("analytic upper bound" if rf_for_margin > rf
                          else "probe (agrees with the analytic bound)"),
        "configured_tile_margin": configured,
        "configured_is_sufficient": bool(configured >= need),
        "tile_core": core,
        "tile_size": int(core + 2 * need),
        "tiling_overhead": float(((core + 2 * need) / core) ** 3),
        "analytic": analytic,
        "analytic_agrees": bool(rf <= analytic["policy_halfwidth"]),
        "probes": rows,
        "note": (
            "The y_lr input is block-upsampled by scale_factor before the trunk, "
            "so one LR cell drives a whole block and the policy's reach is wider "
            "than the diffusion trunk's. Set sampling.tile_margin and "
            "train.context_margin to at least min_tile_margin: the sampler "
            "refuses a smaller one, and training on a smaller context teaches a "
            "mapping full-box tiling never asks for."
        ),
    }
    out = Path(args.out) if args.out else \
        paths.AUDITS("policy_receptive_field", create=True) / "policy_receptive_field.json"
    write_json(out, verdict)

    banner("policy receptive field")
    print(f"  measured half-width   : {rf} HR cells ({0.195 * rf:.1f} Mpc/h)")
    print(f"  analytic half-width   : {analytic['policy_halfwidth']} "
          f"(trunk {analytic['trunk_halfwidth']} + {sf} for the y_lr block)")
    print(f"  minimum tile margin   : {need}")
    print(f"  configured            : {configured} -> "
          + ("OK" if configured >= need else "TOO SMALL, the sampler will refuse"))
    print(f"  tile at core={core}     : {core + 2 * need}^3 "
          f"({verdict['tiling_overhead']:.2f}x overhead)")
    print(f"  -> {out}")
    if configured < need:
        print(f"\n  Set BOTH in configs/reward/gaussian_policy.yaml:")
        print(f"    sampling.tile_margin: {need}")
        print(f"    train.context_margin: {need}")


if __name__ == "__main__":
    main()
