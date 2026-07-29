#!/usr/bin/env python
"""CPU stage: measure the residual denoiser's receptive-field half-width.

The tile margin used for full-box sampling has to be at least this, and picking it
by counting convolutions by hand is exactly the kind of arithmetic that produces
seams nobody notices until the small-scale power looks wrong. This measures it,
prints the tiling cost that follows, and writes the table the model config cites.

The half-width does not depend on channel width, so an expensive configuration is
measured on a narrow one with the same depth, kernel size and scale factor -- the
production width-48 model would take hours of CPU, a width-2 stand-in takes
seconds and gives the identical answer.

    python scripts/reward/measure_receptive_field.py
    python scripts/reward/measure_receptive_field.py --grid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import add_common_args, banner, load_reward_config, write_json

from cosmo_sr.reward.model import build_residual_denoiser
from cosmo_sr.reward.sampling import measure_receptive_field, tile_margin_for

# (num_levels, blocks_per_level) pairs worth comparing when choosing depth.
GRID = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]


def probe_model(levels: int, blocks: int, scale_factor: int, width: int = 2):
    """Structurally identical to the production model, cheap to evaluate.

    ``num_groups=2`` with ``width=2`` would give the channel norm a group size of
    1, which collapses each channel to its own sign; the probe strips norms, so
    this only matters if someone reuses the model for anything else.
    """
    return build_residual_denoiser({
        "channels": 6, "scale_factor": scale_factor, "width": width,
        "num_levels": levels, "blocks_per_level": blocks,
        "embed_dim": 32, "num_groups": 2, "norm": "channel",
        "use_checkpoint": False, "sigma_res": [0.02] * 6,
    })


def tiling_cost(rf: int, scale_factor: int, ng_hr: int, core: int) -> dict:
    margin = tile_margin_for(rf, scale_factor)
    tile = core + 2 * margin
    return {
        "margin": margin,
        "tile": tile,
        "n_tiles": (ng_hr // core) ** 3,
        "overhead": (tile / core) ** 3,
        "fits_in_box": tile <= ng_hr,
    }


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model-config", default="configs/reward/residual_prior.yaml")
    ap.add_argument("--grid", action="store_true",
                    help="sweep depth options instead of only the configured one")
    ap.add_argument("--tile-core", type=int, default=128)
    ap.add_argument("--out", default=None,
                    help="default: receptive_field.json, or _grid.json with --grid, "
                         "so a sweep does not overwrite the configured model's record")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("runs/reward/receptive_field_grid.json" if args.grid
                    else "runs/reward/receptive_field.json")

    cfg = load_reward_config(args)
    from cosmo_sr.utils.config import load_config

    mcfg = dict(load_config(args.model_config).get("model", {}))
    sf = int(mcfg.get("scale_factor", cfg["data"].get("scale_factor", 8)))
    ng_hr = int(cfg["data"].get("ng_hr", 512))
    cell_mpc = float(cfg["data"]["boxsize_mpc_h"]) / ng_hr

    configured = (int(mcfg.get("num_levels", 3)), int(mcfg.get("blocks_per_level", 2)))
    pairs = GRID if args.grid else [configured]

    banner(f"receptive field, scale_factor={sf}, cell={cell_mpc:.3f} Mpc/h, "
           f"core={args.tile_core}")
    print(f"{'levels':>6} {'blocks':>6} {'rf':>5} {'rf[Mpc/h]':>10} "
          f"{'margin':>7} {'tile':>6} {'tiles':>6} {'overhead':>9}", flush=True)
    rows = []
    for levels, blocks in pairs:
        rf = measure_receptive_field(probe_model(levels, blocks, sf),
                                     channels=6, scale_factor=sf)
        cost = tiling_cost(rf, sf, ng_hr, args.tile_core)
        rows.append({"num_levels": levels, "blocks_per_level": blocks,
                     "rf_cells": rf, "rf_mpc_h": rf * cell_mpc, **cost})
        mark = "  <- configured" if (levels, blocks) == configured else ""
        print(f"{levels:6d} {blocks:6d} {rf:5d} {rf * cell_mpc:10.2f} "
              f"{cost['margin']:7d} {cost['tile']:6d} {cost['n_tiles']:6d} "
              f"{cost['overhead']:8.1f}x{mark}", flush=True)
        if not cost["fits_in_box"]:
            print(f"       tile {cost['tile']} exceeds the box {ng_hr}: "
                  f"a periodic crop would self-overlap, so this depth cannot be "
                  f"tiled at core={args.tile_core}", flush=True)

    chosen = next(r for r in rows if (r["num_levels"], r["blocks_per_level"]) == configured)
    write_json(args.out, {
        "scale_factor": sf, "ng_hr": ng_hr, "cell_mpc_h": cell_mpc,
        "tile_core": args.tile_core, "configured": list(configured),
        "rows": rows,
        "recommended_tile_margin": chosen["margin"],
    })
    print(f"\nconfigured model needs --tile-margin {chosen['margin']} "
          f"(measured half-width {chosen['rf_cells']} cells)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
