#!/usr/bin/env python
"""Rung 3 (CPU): pick the handful of chunks the overfitting run trains on.

Rung 3 asks whether reward can be raised on *fixed* massive hosts at all. If it
cannot even here, the failure is representational -- the residual action space
or the conditioning cannot express the change -- which is a different conclusion
from "the search did not find it" (rung 2).

Chunks are ranked by their host count in the upper reliable bins, read from the
cached HR summaries. The sparse ``1e14`` bin is excluded: §2's audit shows it is
evaluation-only and does not clear the effective-host bar, so ranking on it
would pick chunks whose reward signal is a handful of objects.

    python scripts/reward/select_fixed_hosts.py --boxes set8 --n-chunks 4
"""
from __future__ import annotations

import argparse

import numpy as np

from _common import (add_common_args, assert_no_leak, banner, load_reward_config,
                     split_boxes, write_json)

from cosmo_sr.reward import paths
from cosmo_sr.reward.catalog import read_summaries


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--boxes", default=None, help="comma list; default = train split")
    ap.add_argument("--n-chunks", type=int, default=4)
    ap.add_argument("--out", default=None, help="default: $DMSR_REWARD_ROOT/overfit/fixed_hosts.json")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    rcfg = cfg.get("reward", {})
    occ = dict(rcfg.get("occupation", {}))
    upper = [int(i) for i in occ.get("upper_reliable_host_bins", [2, 3])]
    sparse = {int(i) for i in occ.get("sparse_host_bins", [])}
    rank_bins = [i for i in upper if i not in sparse]
    if not rank_bins:
        raise SystemExit(
            f"every upper reliable bin {upper} is marked sparse; nothing safe to "
            f"rank on"
        )

    boxes = [b.strip() for b in args.boxes.split(",")] if args.boxes \
        else split_boxes(cfg, "train")
    # These chunks become *training* crops, so they must be training boxes --
    # overfitting a val box would put validation data in the training set and
    # make the resulting numbers unusable for anything downstream.
    assert_no_leak(cfg, boxes, ["train"])

    rows = []
    for box in boxes:
        p = paths.CATALOG_CACHE() / f"{box}__hr__hr.jsonl"
        if not p.is_file():
            banner(f"no cached HR summaries for {box} at {p} -- skipping")
            continue
        for s in read_summaries(p):
            if s.volume_mpc3 <= 0:
                continue
            n_upper = float(np.sum(np.asarray(s.n_host)[rank_bins]))
            rows.append({
                "box": box,
                "chunk_id": int(s.chunk_id),
                "n_host_upper": n_upper,
                "n_host_total": int(s.n_host_total),
                "n_sub_total": int(s.n_sub_total),
            })

    if not rows:
        raise SystemExit(
            "no HR chunk summaries found; run the catalog cache job before this"
        )

    rows.sort(key=lambda r: (r["n_host_upper"], r["n_sub_total"]), reverse=True)
    picked = rows[:max(1, int(args.n_chunks))]
    if picked[0]["n_host_upper"] <= 0:
        raise SystemExit(
            f"the best chunk has no hosts in bins {rank_bins}; overfitting it "
            f"would have no occupation signal to raise"
        )

    for r in picked:
        print(f"[{r['box']} c{r['chunk_id']}] upper hosts={r['n_host_upper']:.0f} "
              f"subs={r['n_sub_total']}", flush=True)

    out = args.out or (paths.subdir("overfit", create=True) / "fixed_hosts.json")
    write_json(out, {
        "boxes": boxes,
        "rank_bins": rank_bins,
        "n_chunks": len(picked),
        "chunks": [[r["box"], r["chunk_id"]] for r in picked],
        "detail": picked,
    })
    banner(f"{len(picked)} fixed host chunks -> {out}")


if __name__ == "__main__":
    main()
