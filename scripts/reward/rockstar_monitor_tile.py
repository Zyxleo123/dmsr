#!/usr/bin/env python
"""Run the REAL halo finder on tile-overfit snapshots -- proxy vs ground truth.

The tile-overfit optimiser reports `dR_occ` from the PROXY. This job measures the
same quantity for real: it takes a saved snapshot of the optimised tile, splices
it into the frozen box (an isolated 64^3 tile has no host context, so we re-run
Rockstar on the complete periodic box -- exactly what actor_rockstar_verify does),
and computes the measured occupation reward change against the frozen box's own
catalog. Splicing the iter-0 snapshot reproduces the frozen box, so its measured
dR should be ~0 -- a built-in sanity check.

One array task per snapshot; each writes its own one-row file so concurrent tasks
never race on a shared append. The plot job aggregates them and overlays the real
trajectory on the proxy's.

    python scripts/reward/rockstar_monitor_tile.py --tag set0_t486_c --index 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, banner, bins_of,
    direct_root, load_direct_config, load_reward_models, run_dir,
    tile_grid_of, write_json_atomic,
)
from splice_verify import _box_summary  # noqa: E402

from cosmo_sr.eval.rockstar import run_rockstar_on_field  # noqa: E402
from cosmo_sr.reward.tiles import direct_full_box_stats  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--tag", required=True, help="e.g. set0_t486_c")
    ap.add_argument("--index", type=int, required=True,
                    help="which snapshot (position in snap_iters) to score")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing Rockstar catalog in the work dir")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    run = run_dir(args.run_name)
    of = run / "tile_overfit"
    summ_p = of / f"tile_overfit_{args.tag}.json"
    if not summ_p.is_file():
        print(f">>> MISSING INPUT: {summ_p}")
        print(">>> produced by: scripts/reward/overfit_tile_to_proxy.py --ckpt-every>0")
        return 0
    summ = json.loads(summ_p.read_text())
    snap_iters = [int(x) for x in summ.get("snap_iters", [])]
    if not snap_iters:
        print(">>> no snapshots recorded; rerun the optimiser with --ckpt-every>0")
        return 0
    if args.index < 0 or args.index >= len(snap_iters):
        print(f">>> index {args.index} out of range 0..{len(snap_iters) - 1}")
        return 0
    it = snap_iters[args.index]
    box, tile, seed = summ["box"], int(summ["tile"]), int(summ["seed"])
    snap = Path(summ["snap_dir"]) / f"field_it{it:04d}.npy"

    out_dir = of / f"rockstar_monitor_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_row = out_dir / f"iter_{it:04d}.json"
    if out_row.is_file() and not args.reuse:
        print(f"=== iter {it} already scored -> {out_row}; skipping")
        return 0

    # ---- frozen box: the field we splice into + its measured baseline -----
    cand = direct_root("candidates", f"{box}__frozen_seed{seed}")
    base_npy = cand / "field.npy"
    report = cand / "label_report.json"
    for p, prod in ((base_npy, "collect_catalog_proxy_data.py --stage generate"),
                    (report, "collect_catalog_proxy_data.py --stage label")):
        if not p.is_file():
            print(f">>> MISSING INPUT: {p}\n>>> produced by: {prod}")
            return 0
    before_counts = json.loads(report.read_text())["full_box"]

    grid = tile_grid_of(cfg)
    bins = bins_of(cfg["_reward"])
    d = cfg["_reward"]["data"]          # same source as actor_rockstar_verify
    box_l = float(d.get("boxsize_mpc_h", 100.0))
    acfg = actor_config_of(cfg)
    w_joint, w_occ = float(acfg.w_joint_reward), float(acfg.w_occ_reward)
    _, reward_t = load_reward_models(cfg)

    # ---- splice the snapshot tile into the frozen box, run Rockstar -------
    field = np.array(np.load(base_npy), dtype=np.float32, copy=True)
    tile_field = np.load(snap)
    sx, sy, sz = grid.slices(int(tile))
    field[:, sx, sy, sz] = tile_field
    banner(f"Rockstar on {box} with tile {tile} <- overfit iter {it} "
           f"(snapshot {args.index + 1}/{len(snap_iters)})")

    work = out_dir / f"work_it{it:04d}"
    cat = run_rockstar_on_field(
        field, work, tag=f"mon_it{it:04d}", boxsize_kpc_h=box_l * 1000.0,
        redshift=float(d.get("redshift", 0.0)), overwrite=not args.reuse)
    del field
    for pat in ("*.gadget2", "*.particles", "*.bin"):
        for g in Path(work).glob(pat):
            g.unlink(missing_ok=True)

    after = direct_full_box_stats(cat, bins)
    vol = box_l ** 3
    with torch.no_grad():
        s_before = _box_summary(before_counts, vol)
        s_after = _box_summary({k: after[k].tolist()
                                for k in ("n_sub", "n_host", "occ_numerator")}, vol)
        sb, sa = reward_t.scores(s_before), reward_t.scores(s_after)
        measured = {f"dR_{k[2:]}": float(sa[k][0] - sb[k][0]) for k in sa}
        measured["dR_combined"] = float(
            reward_t.combined(s_after, w_joint=w_joint, w_occ=w_occ)[0]
            - reward_t.combined(s_before, w_joint=w_joint, w_occ=w_occ)[0])

    nh_b = np.asarray(before_counts["n_host"], dtype=float)
    nh_a = after["n_host"].astype(float)
    occ_b = np.asarray(before_counts["occ_numerator"], dtype=float)
    occ_a = after["occ_numerator"].astype(float)
    row = {
        "tag": args.tag, "index": int(args.index), "iter": it,
        "box": box, "tile": tile, "seed": seed,
        "measured_dR_occ": measured.get("dR_occ"),
        "measured_dR_combined": measured["dR_combined"],
        "measured_dR_hosted_subs": measured.get("dR_hosted_subs"),
        "n_objects": int(cat.n),
        "n_host_before": nh_b.tolist(), "n_host_after": nh_a.tolist(),
        "occ_numerator_before": occ_b.tolist(),
        "occ_numerator_after": occ_a.tolist(),
        "n_sub_before": list(map(float, before_counts["n_sub"])),
        "n_sub_after": after["n_sub"].astype(float).tolist(),
        "occupation_after": after["occupation"].astype(float).tolist(),
    }
    write_json_atomic(out_row, row)
    banner(f"iter {it}: measured dR_occ {row['measured_dR_occ']:+.4g}  "
           f"dR_comb {row['measured_dR_combined']:+.4g}  n_obj {cat.n}  -> {out_row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
