#!/usr/bin/env python
"""Per-tile credit A_j from cached statistics, and its verification against Rockstar.

Two forms, and the difference between them is the point:

``A_j = R(S) - R(S - s_j)``     (``--mode credit``)
    The removal form the experiment brief specifies. Pure arithmetic on cached
    sufficient statistics -- the halo finder is never re-run. It answers "what
    does tile j contribute to the ensemble reward?"

``dR_j = R(S - s_j^base + s_j^HR) - R(S)``   (``--mode credit``, reported alongside)
    The swap form. This one is **checkable against a real run**, because
    "replace tile j's field with HR" is something you can actually do, whereas
    "delete tile j" is not. ``--mode plan`` picks the tiles to check and
    ``--mode verify`` compares the prediction with the measured change.

The gap between predicted and measured ``dR_j`` is the cross-tile interaction
that per-tile credit assumes away -- risk 3 in ``docs/reward_residual_diffusion.md``.
Measuring it on 8-16 tiles is the cheapest honest bound on that assumption.

    python scripts/reward/tile_loo_credit.py --mode credit --box set8
    python scripts/reward/tile_loo_credit.py --mode plan   --box set8 --n-verify 12
    python scripts/reward/tile_loo_credit.py --mode verify --box set8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from _common import (  # noqa: E402
    add_common_args, banner, bins_of, load_reward_config, paths, write_json,
)

from cosmo_sr.reward.reward import RewardModel, fit_reward_model  # noqa: E402
from cosmo_sr.reward.tiles import (  # noqa: E402
    leave_one_out_credit, pool_tiles, read_tile_summaries, swap_prediction,
)


def tile_jsonl(box: str, source: str) -> Path:
    return paths.subdir("tile_cache") / f"{box}__{source}__{source}.jsonl"


def reward_model_path() -> Path:
    return paths.AUDITS("tile_decomposition") / "tile_reward_model.json"


def fit_tile_reward_model(cfg, bins, boxes: List[str]) -> RewardModel:
    """``mu_HR`` and ``C`` at the *tile* geometry, one ensemble = one whole box.

    The reward model shipped in ``configs/reward`` was fitted on purity-masked
    chunk summaries, whose counts and effective volumes are not the same
    quantities as the unmasked tile decomposition. Scoring tile-summed
    statistics against that ``mu`` would compare a masked target with an
    unmasked measurement, so the model is refitted here on HR *tile* summaries.

    One box is one ensemble because a box is the independent cosmological unit;
    the covariance is therefore box-to-box scatter, which is what the reward
    needs, and not the seed scatter that was already measured to be degenerate.
    """
    from cosmo_sr.reward.catalog import ChunkSummary

    chunks: List[ChunkSummary] = []
    for b in boxes:
        p = tile_jsonl(b, "hr")
        if not p.is_file():
            continue
        pooled = pool_tiles(read_tile_summaries(p))
        chunks.append(ChunkSummary(
            box=b, chunk_id=0, source="hr",
            n_sub=pooled.n_sub, n_host=pooled.n_host,
            occ_numerator=pooled.occ_numerator, volume_mpc3=pooled.volume_mpc3,
        ))
    if len(chunks) < 3:
        raise SystemExit(
            f"only {len(chunks)} HR boxes have tile summaries; a box-level "
            "covariance needs several. Run rockstar_particles.py on more boxes."
        )
    r = cfg.get("reward", {})
    return fit_reward_model(
        chunks, bins, ensemble_size=1,
        n_draws=int(r.get("bootstrap_draws", 400)),
        shrinkage=float(r.get("shrinkage", 0.1)),
        seed=int(r.get("bootstrap_seed", 0)),
    )


def load_or_fit_model(cfg, bins, boxes: List[str], refit: bool) -> RewardModel:
    p = reward_model_path()
    if p.is_file() and not refit:
        return RewardModel.from_dict(json.loads(p.read_text()))
    rm = fit_tile_reward_model(cfg, bins, boxes)
    write_json(p, rm.to_dict())
    print(f"fitted tile-geometry reward model -> {p} "
          f"(cond(C_reg) = {rm.condition_number:.1f})", flush=True)
    return rm


def credit_table(box: str, bins, cfg, rm: RewardModel) -> Dict:
    base = read_tile_summaries(tile_jsonl(box, "base"))
    hr = {int(s.tile_id): s for s in read_tile_summaries(tile_jsonl(box, "hr"))}

    a_cat = leave_one_out_credit(base, rm.reward)
    a_occ = leave_one_out_credit(base, rm.reward_occupation)
    swap_cat = {t: swap_prediction(base, hr, t, rm.reward) for t in a_cat}
    swap_occ = {t: swap_prediction(base, hr, t, rm.reward_occupation) for t in a_cat}

    # Upper reliable host bins are what Gate B is decided on; a tile's share of
    # them is the natural "is this tile worth regenerating?" ranking.
    upper = cfg.get("reward", {}).get("occupation", {}).get(
        "upper_reliable_host_bins", [2, 3])
    share = {int(s.tile_id): float(sum(s.n_host[b] for b in upper)) for s in base}
    return {
        "box": box,
        "A_cat": a_cat, "A_occ": a_occ,
        "swap_dR_cat": swap_cat, "swap_dR_occ": swap_occ,
        "upper_host_share": share,
        "R_base": float(rm.reward(pool_tiles(base))),
        "R_occ_base": float(rm.reward_occupation(pool_tiles(base))),
        "R_hr": float(rm.reward(pool_tiles(list(hr.values())))),
        "R_occ_hr": float(rm.reward_occupation(pool_tiles(list(hr.values())))),
    }


def select_verify_tiles(tab: Dict, n: int) -> List[int]:
    """Tiles whose swap is predicted to matter most, plus a null control.

    Taking only the largest predictions would measure the interaction term where
    it is easiest to see and call that a bound. Two tiles with near-zero
    predicted effect are added so a systematic offset -- prediction wrong even
    where nothing should happen -- is visible.
    """
    pred = tab["swap_dR_occ"]
    order = sorted(pred, key=lambda t: -abs(pred[t]))
    top = order[:max(int(n) - 2, 1)]
    null = [t for t in reversed(order) if abs(pred[t]) < 1e-9][:2]
    return sorted({int(t) for t in top + null})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--mode", choices=("credit", "plan", "verify"), default="credit")
    ap.add_argument("--box", required=True)
    ap.add_argument("--model-boxes", default="",
                    help="boxes used to fit the tile-geometry reward model "
                         "(default: every box with HR tile summaries)")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--n-verify", type=int, default=12)
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    box = str(args.box)
    out_dir = paths.AUDITS("tile_decomposition", create=True)

    if args.model_boxes:
        model_boxes = [b.strip() for b in args.model_boxes.split(",") if b.strip()]
    else:
        model_boxes = sorted(
            p.name.split("__")[0]
            for p in paths.subdir("tile_cache").glob("*__hr__hr.jsonl")
        )
    rm = load_or_fit_model(cfg, bins, model_boxes, args.refit)

    if args.mode in ("credit", "plan"):
        for src in ("base", "hr"):
            if not tile_jsonl(box, src).is_file():
                raise SystemExit(
                    f"missing {src} tile summaries for {box}: {tile_jsonl(box, src)}"
                )
        tab = credit_table(box, bins, cfg, rm)
        write_json(out_dir / f"credit_{box}.json", tab)

        banner(f"{box}: per-tile credit (reward model cond {rm.condition_number:.1f})")
        print(f"R_cat  base = {tab['R_base']:.4f}   HR = {tab['R_hr']:.4f}")
        print(f"R_occ  base = {tab['R_occ_base']:.4f}   HR = {tab['R_occ_hr']:.4f}")
        a = np.array(list(tab["A_occ"].values()))
        print(f"A_occ over {a.size} tiles: mean {a.mean():+.3e}  "
              f"sd {a.std():.3e}  min {a.min():+.3e}  max {a.max():+.3e}")
        print(f"nonzero A_occ tiles: {int(np.count_nonzero(np.abs(a) > 1e-12))}")

        top = sorted(tab["swap_dR_occ"], key=lambda t: -tab["swap_dR_occ"][t])[:10]
        print(f"\n{'tile':>6s} {'A_occ':>12s} {'A_cat':>12s} "
              f"{'swap dR_occ':>13s} {'swap dR_cat':>13s} {'upper hosts':>12s}")
        for t in top:
            print(f"{t:>6} {tab['A_occ'][t]:12.4e} {tab['A_cat'][t]:12.4e} "
                  f"{tab['swap_dR_occ'][t]:13.4e} {tab['swap_dR_cat'][t]:13.4e} "
                  f"{tab['upper_host_share'][t]:12.3f}")

        if args.mode == "plan":
            sel = select_verify_tiles(tab, args.n_verify)
            manifest = {
                "box": box, "tiles": sel,
                "predicted_dR_occ": {str(t): tab["swap_dR_occ"][t] for t in sel},
                "predicted_dR_cat": {str(t): tab["swap_dR_cat"][t] for t in sel},
                "R_occ_base": tab["R_occ_base"], "R_base": tab["R_base"],
            }
            p = write_json(out_dir / f"verify_plan_{box}.json", manifest)
            banner(f"{len(sel)} tiles selected for Rockstar verification -> {p}")
            print("tile ids:", ",".join(str(t) for t in sel))
        return 0

    # ------------------------------------------------------------- verify
    plan_path = out_dir / f"verify_plan_{box}.json"
    if not plan_path.is_file():
        raise SystemExit(f"no verification plan at {plan_path}; run --mode plan first")
    plan = json.loads(plan_path.read_text())
    base_tiles = read_tile_summaries(tile_jsonl(box, "base"))
    r_occ_base = rm.reward_occupation(pool_tiles(base_tiles))
    r_cat_base = rm.reward(pool_tiles(base_tiles))

    rows = []
    for t in plan["tiles"]:
        tag = f"splice{t}"
        jl = paths.subdir("tile_cache") / f"{box}__base__{tag}.jsonl"
        if not jl.is_file():
            print(f"!! tile {t}: no scored splice yet ({jl})", flush=True)
            continue
        ts = read_tile_summaries(jl)
        rows.append({
            "tile": int(t),
            "predicted_dR_occ": float(plan["predicted_dR_occ"][str(t)]),
            "measured_dR_occ": float(rm.reward_occupation(pool_tiles(ts)) - r_occ_base),
            "predicted_dR_cat": float(plan["predicted_dR_cat"][str(t)]),
            "measured_dR_cat": float(rm.reward(pool_tiles(ts)) - r_cat_base),
        })

    if not rows:
        raise SystemExit("no spliced boxes scored yet; run the verify array first")

    banner(f"{box}: predicted vs measured reward change for a real tile swap")
    head = (f"{'tile':>6s} {'pred dR_occ':>13s} {'meas dR_occ':>13s} {'ratio':>8s} "
            f"{'pred dR_cat':>13s} {'meas dR_cat':>13s}")
    print(head)
    print("-" * len(head))
    for r in rows:
        ratio = (r["measured_dR_occ"] / r["predicted_dR_occ"]
                 if abs(r["predicted_dR_occ"]) > 1e-12 else np.nan)
        print(f"{r['tile']:>6} {r['predicted_dR_occ']:13.4e} "
              f"{r['measured_dR_occ']:13.4e} {ratio:8.3f} "
              f"{r['predicted_dR_cat']:13.4e} {r['measured_dR_cat']:13.4e}")

    p = np.array([r["predicted_dR_occ"] for r in rows])
    m = np.array([r["measured_dR_occ"] for r in rows])
    live = np.abs(p) > 1e-12
    summary = {
        "box": box, "rows": rows, "n_tiles": len(rows),
        "pearson_r": float(np.corrcoef(p, m)[0, 1]) if len(rows) > 2 else float("nan"),
        "sign_agreement": float(np.mean(np.sign(p[live]) == np.sign(m[live])))
        if live.any() else float("nan"),
        "median_abs_ratio": float(np.median(np.abs(m[live] / p[live])))
        if live.any() else float("nan"),
        "max_abs_residual": float(np.max(np.abs(m - p))),
    }
    print(f"\npearson r        = {summary['pearson_r']:.3f}")
    print(f"sign agreement   = {summary['sign_agreement']:.2f}")
    print(f"median |meas/pred| = {summary['median_abs_ratio']:.3f}")
    print("\nA ratio far from 1 does not invalidate the decomposition -- the "
          "decomposition is exact by construction (Experiment 0's table). It "
          "bounds how much of a tile's effect is carried by its neighbours, "
          "which is what per-tile credit assumes is small.")
    write_json(out_dir / f"verify_{box}.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
