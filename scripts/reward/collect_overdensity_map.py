#!/usr/bin/env python
"""Project the frozen-SR2 overdensity in a thin slab and mark every centre on it.

This is the visual companion to the candidate/partition diagnostics. It answers,
by eye rather than by number, the question those tests raised: do the candidate
centres we can build *without ground truth* actually land on the real subhalos --
and in particular on the *missing* ones no frozen-SR2 subhalo marks?

For one box it computes, from artifacts already on disk (no Rockstar, no GPU, no
training):

* a **projected overdensity map** -- particles of frozen SR2 that fall inside a
  slab ``|z - z0| <= dz`` histogrammed onto an ``ng_img x ng_img`` (x, y) grid and
  divided by the mean cell count, i.e. ``1 + delta`` projected through the slab.
  This is the "cosmic web" background image.
* **candidate centres (no ground truth)**: the frozen-SR2 catalog's own subhalos,
  and difference-of-Gaussians density peaks (subhalo-scale minus host-scale, the
  :func:`diagnose_peak_targeting.dog_residual` finder). Both are what a deployed
  reward could place without HR.
* **real subhalo centres (ground truth)**: every HR subhalo, with the *missing
  targets* (the ones frozen SR2 fails to form) flagged separately.

Only centres whose ``z`` lies in the slab are kept, so markers and image share the
same slice. Everything is written to ``map.npz`` (+ ``summary.json``) so
:mod:`scripts/reward/render_overdensity_html.py` can redraw the HTML without
recomputing anything.

    python scripts/reward/collect_overdensity_map.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from _common import banner, paths, write_json  # noqa: E402

from diagnose_candidate_partition import (  # noqa: E402
    _FieldCfg, load_catalog, load_targets,
)
from diagnose_peak_targeting import dog_residual, peaks_from_residual  # noqa: E402
from diagnose_progress_signal import particles_of  # noqa: E402
from rockstar_particles import load_field  # noqa: E402


def _slab_mask(z: np.ndarray, z0: float, dz: float, box_l: float) -> np.ndarray:
    """Periodic ``|z - z0| <= dz`` in a box of side ``box_l``."""
    d = (np.asarray(z) - z0 + 0.5 * box_l) % box_l - 0.5 * box_l
    return np.abs(d) <= dz


def _projected_overdensity(pos_xy: np.ndarray, box_l: float, ng: int) -> np.ndarray:
    """``1 + delta`` of the slab particles projected onto an ``ng x ng`` (x,y) grid."""
    counts, _, _ = np.histogram2d(
        pos_xy[:, 0], pos_xy[:, 1], bins=ng,
        range=[[0.0, box_l], [0.0, box_l]])
    mean = counts.mean()
    return (counts / mean).astype(np.float32) if mean > 0 else counts.astype(np.float32)


def run_box(box: str, args, out_dir: Path) -> Dict:
    box_l = float(args.boxsize_mpc_h)
    banner(f"{box}: loading frozen-SR2 field")
    sr2 = load_field(_FieldCfg(args), box, "base", args.base_seed)
    pos, _vel, _m_p, n_part = particles_of(sr2, box_l)
    del sr2, _vel

    base_cat = load_catalog(box, "base")
    hr_cat = load_catalog(box, "hr")
    targets = load_targets(box)
    target_ids = {int(t["hr_sub_id"]) for t in targets}

    # Slab centre: default to the median z of the missing targets so the slice is
    # guaranteed to cut through the structure the reward cares about.
    if args.slab_z_mpc >= 0.0:
        z0 = float(args.slab_z_mpc)
    else:
        z0 = float(np.median([t["sub_pos_mpc_h"][2] for t in targets])) \
            if targets else 0.5 * box_l
    dz = float(args.slab_dz_mpc)
    banner(f"{box}: slab z0={z0:.2f} +/- {dz:.2f} Mpc/h")

    in_slab = _slab_mask(pos[:, 2], z0, dz, box_l)
    over = _projected_overdensity(pos[in_slab, :2], box_l, int(args.ng_img))
    n_slab = int(in_slab.sum())

    # --- candidate centres (no ground truth) -------------------------------- #
    bsub = base_cat.subhalos()
    b_in = _slab_mask(bsub.pos[:, 2], z0, dz, box_l)
    base_sub_xy = bsub.pos[b_in, :2].astype(np.float32)
    base_sub_logm = np.log10(np.clip(bsub.mvir[b_in], 1.0, None)).astype(np.float32)

    banner(f"{box}: difference-of-Gaussians peaks (ng={args.dog_ng})")
    resid, local_max, cell = dog_residual(
        pos, box_l, ng=int(args.dog_ng),
        fine_cells=args.dog_fine, coarse_cells=args.dog_coarse)
    peaks, pstr = peaks_from_residual(
        resid, local_max, cell, threshold=args.dog_threshold,
        max_peaks=args.max_peaks)
    if peaks.shape[0]:
        p_in = _slab_mask(peaks[:, 2], z0, dz, box_l)
        dog_xy = peaks[p_in, :2].astype(np.float32)
        dog_str = pstr[p_in].astype(np.float32)
    else:
        dog_xy = np.zeros((0, 2), np.float32)
        dog_str = np.zeros(0, np.float32)

    # --- real subhalo centres (ground truth) -------------------------------- #
    hsub = hr_cat.subhalos()
    h_in = _slab_mask(hsub.pos[:, 2], z0, dz, box_l)
    hr_sub_xy = hsub.pos[h_in, :2].astype(np.float32)
    hr_sub_logm = np.log10(np.clip(hsub.mvir[h_in], 1.0, None)).astype(np.float32)
    hr_missing = np.array([int(i) in target_ids for i in hsub.ids[h_in]], dtype=bool)

    # Targets straight from the oracle file (guaranteed markers even if an id is
    # absent from the HR subhalo slice for any reason).
    tgt = [t for t in targets if _slab_mask(np.array([t["sub_pos_mpc_h"][2]]),
                                            z0, dz, box_l)[0]]
    target_xy = np.array([[t["sub_pos_mpc_h"][0], t["sub_pos_mpc_h"][1]]
                          for t in tgt], dtype=np.float32).reshape(-1, 2)
    target_logm = np.array([np.log10(max(float(t["sub_mvir"]), 1.0)) for t in tgt],
                           dtype=np.float32)

    box_dir = out_dir / box
    box_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        box_dir / "map.npz",
        overdensity=over, box_l=np.float32(box_l), z0=np.float32(z0),
        dz=np.float32(dz), ng_img=np.int64(args.ng_img),
        cell_mpc=np.float32(box_l / int(args.ng_img)),
        base_sub_xy=base_sub_xy, base_sub_logm=base_sub_logm,
        dog_xy=dog_xy, dog_str=dog_str,
        hr_sub_xy=hr_sub_xy, hr_sub_logm=hr_sub_logm, hr_missing=hr_missing,
        target_xy=target_xy, target_logm=target_logm)

    summary = {
        "box": box, "box_l_mpc_h": box_l, "slab_z0_mpc_h": z0, "slab_dz_mpc_h": dz,
        "ng_img": int(args.ng_img), "cell_mpc_h": box_l / int(args.ng_img),
        "n_particles_total": int(n_part), "n_particles_slab": n_slab,
        "n_base_subhalos_slab": int(base_sub_xy.shape[0]),
        "n_dog_peaks_slab": int(dog_xy.shape[0]),
        "n_hr_subhalos_slab": int(hr_sub_xy.shape[0]),
        "n_missing_subhalos_slab": int(hr_missing.sum()),
        "n_targets_slab": int(target_xy.shape[0]),
        "dog": {"ng": int(args.dog_ng), "fine": args.dog_fine,
                "coarse": args.dog_coarse, "threshold": args.dog_threshold},
    }
    write_json(box_dir / "summary.json", summary)
    _print_summary(summary)
    return summary


def _print_summary(s: Dict) -> None:
    print(f"\n### {s['box']} overdensity slab "
          f"(z={s['slab_z0_mpc_h']:.1f}+/-{s['slab_dz_mpc_h']:.1f} Mpc/h, "
          f"{s['n_particles_slab']} particles)", flush=True)
    print(f"   candidates : {s['n_base_subhalos_slab']} frozen-SR2 subhalos, "
          f"{s['n_dog_peaks_slab']} DoG peaks", flush=True)
    print(f"   real (HR)  : {s['n_hr_subhalos_slab']} subhalos, of which "
          f"{s['n_missing_subhalos_slab']} missing", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--data-root", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm")
    ap.add_argument("--boxsize-mpc-h", type=float, default=100.0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--ng-img", type=int, default=512,
                    help="(x,y) grid for the projected overdensity image")
    ap.add_argument("--slab-z-mpc", type=float, default=-1.0,
                    help="slab centre in Mpc/h; <0 -> median z of missing targets")
    ap.add_argument("--slab-dz-mpc", type=float, default=2.5,
                    help="slab half-thickness in Mpc/h")
    # DoG peak finder (same defaults as diagnose_peak_targeting).
    ap.add_argument("--dog-ng", type=int, default=512)
    ap.add_argument("--dog-fine", type=float, default=1.5)
    ap.add_argument("--dog-coarse", type=float, default=8.0)
    ap.add_argument("--dog-threshold", type=float, default=1.0)
    ap.add_argument("--max-peaks", type=int, default=200000)
    ap.add_argument("--out-name", default="overdensity_map")
    args = ap.parse_args(argv)

    out_dir = paths.subdir("audits", args.out_name, create=True)
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    summaries = [run_box(b, args, out_dir) for b in boxes]
    write_json(out_dir / "summary.json", {"boxes": boxes, "per_box": summaries})
    banner(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
