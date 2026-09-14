#!/usr/bin/env python
"""Build the LR-host conditioning field for a set of boxes. CPU, minutes per box.

What it is for
--------------
``radial-profile-is-flat-in-host-mass`` measured the defect this channel exists
to fix: HR's median subhalo radius scales 6.5x with host mass, every candidate
catalog is flat at ~0.1 Mpc/h, and a local translation-equivariant operator fed
only ``(lr, noise)`` cannot do otherwise. :mod:`cosmo_sr.features.lr_host_channel`
says what the channels are; this is the job that produces them.

Why it is a separate job
------------------------
It runs Rockstar on the 64^3 LR particles. That is small (262k particles) but it
is a halo finder, and it has no business inside a 10-hour GPU training run: the
trainer would pay it per box, on a GPU node, before it took a single step. So
the field is built once, cached under ``$DMSR_REWARD_ROOT/lr_host_channel``, and
the trainer loads it -- and refuses to start if the cached settings do not match
the ones it was asked for.

Not leakage: the LR catalog is a function of the LR box alone, and exists at
inference for every box the generator can run on.

    python scripts/features/build_lr_host_channel.py --boxes set3,set9 \
        --channels r_over_rvir,log_mhost,valid

Writes, per box, ``<out-dir>/<box>.npz`` (the ``(C, 64, 64, 64)`` stack plus its
config and provenance) and one ``summary.json`` over all boxes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import add_common_args, banner, load_reward_config, lr_path, write_json  # noqa: E402
from gather_holdout_figures_data import BOX  # noqa: E402
from lr_host_fidelity import lr_catalog  # noqa: E402

from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.features.lr_host_channel import (  # noqa: E402
    DEFAULT_CHANNELS, LRHostChannelConfig, build_channel_field,
    channel_field_path, gate_hosts, save_channel_field,
)
from cosmo_sr.reward import paths  # noqa: E402


def membership_index(work_root: Path, box: str, cat, gated_rows: np.ndarray,
                     ng_lr: int) -> np.ndarray:
    """``(ng_lr^3,)`` row index into ``gated_rows``, or -1, from Rockstar membership.

    MEASURED (``lr_host_fidelity/membership_summary.json``, set8 and set9):
    membership is COVERED out to 2 R_vir -- median correct-assignment fraction
    0.79-0.81 in the 1.0-1.5 R_vir shell HR's subhalos peak in -- and only falls
    to PARTIAL (0.45-0.49) past 2 R_vir. So this mode is a real alternative to
    ``derived`` rather than a known-worse one; see
    :mod:`cosmo_sr.features.lr_host_channel` for why ``derived`` is nonetheless
    the default.
    """
    from lr_membership_coverage import lr_site_host

    parts = sorted((work_root / box / f"{box}_lr_rockstar").glob("halos_*.particles"))
    if not parts:
        raise SystemExit(
            f"--mode membership needs the LR run's halos_*.particles table, and "
            f"none is under {work_root / box / f'{box}_lr_rockstar'}. The "
            "derived mode needs no such file.")
    site_host_id, stats = lr_site_host(parts[0], cat)
    print(f"    membership table: {stats}", flush=True)
    # `lr_site_host` returns a top-level halo ID; the channel builder wants a
    # row index into the GATED host list, and a site owned by a host below the
    # gate is as good as unowned.
    row_of_id = {int(hid): r for r, hid in enumerate(np.asarray(cat.ids))}
    gated_pos = {int(r): k for k, r in enumerate(gated_rows)}
    out = np.full(int(ng_lr) ** 3, -1, dtype=np.int64)
    for site, hid in enumerate(site_host_id):
        if hid < 0:
            continue
        r = row_of_id.get(int(hid))
        if r is None:
            continue
        k = gated_pos.get(int(r))
        if k is not None:
            out[site] = k
    return out


def run_box(cfg, box: str, ccfg: LRHostChannelConfig, args, out_dir: Path) -> Dict:
    t0 = time.time()
    field_lr = np.load(lr_path(cfg, box), mmap_mode="r")
    ng_lr = int(field_lr.shape[1])
    cat, particles = lr_catalog(cfg, box, Path(args.work_root), rerun=args.rerun_rockstar)

    gated = gate_hosts(cat, min_hr_num_p=ccfg.min_hr_num_p,
                       ng_lr=ng_lr, ng_hr=int(args.ng_hr))
    ratio = (float(args.ng_hr) / float(ng_lr)) ** 3
    print(f"  {cat.n} LR halos -> {gated.size} top-level hosts above "
          f"{ccfg.min_hr_num_p} HR particles "
          f"(an LR floor of {ccfg.min_hr_num_p / ratio:.1f} particles)",
          flush=True)
    if gated.size == 0:
        raise SystemExit(
            f"{box}: no LR host survives the {ccfg.min_hr_num_p} HR-particle "
            "gate, so every channel would be its fill value and the arm would "
            "train on a constant. Lower --min-hr-num-p, knowing "
            "lr-host-channel-fidelity measured the channel unusable below "
            "~20000.")

    site_index = None
    if ccfg.mode == "membership":
        site_index = membership_index(Path(args.work_root), box, cat, gated, ng_lr)

    field, stats = build_channel_field(
        np.asarray(particles.pos_mpc_h, dtype=np.float64),
        np.asarray(cat.pos)[gated],
        np.asarray(cat.rvir, dtype=np.float64)[gated] / 1000.0,   # kpc/h -> Mpc/h
        np.asarray(cat.mvir, dtype=np.float64)[gated],
        ccfg, ng_lr=ng_lr, boxsize_mpc_h=float(args.boxsize),
        site_host_index=site_index)

    meta = {"box": box, "ng_lr": ng_lr, "boxsize_mpc_h": float(args.boxsize),
            "lr_catalog": str(cat.path), "n_lr_halos": int(cat.n),
            "lr_particle_mass_msun_h": float(particles.particle_mass_msun_h),
            "seconds": round(time.time() - t0, 1), **stats}
    path = channel_field_path(box, out_dir)
    save_channel_field(path, field, ccfg, meta)
    print(f"  wrote {path}  {field.shape} "
          f"valid={stats['valid_fraction']:.3f} "
          f"median r/Rvir={stats['median_r_over_rvir_valid']:.2f} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    for c in ccfg.channels:
        print(f"    {c:<12s} mean {stats['per_channel_mean'][c]:+.4f}  "
              f"std {stats['per_channel_std'][c]:.4f}")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--boxes", required=True,
                    help="comma or space separated, e.g. 'set3 set4 set9'")
    ap.add_argument("--reward-root", default="",
                    help="dmsr_reward root. Exported as DMSR_REWARD_ROOT so "
                         "that the default --out-dir and --work-root resolve "
                         "under the SAME root the trainer will read from; a "
                         "field written under one root and looked for under "
                         "another reads as 'not built'.")
    ap.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    ap.add_argument("--mode", default="derived", choices=("derived", "membership"))
    ap.add_argument("--min-hr-num-p", type=int, default=20000)
    ap.add_argument("--r-max-rvir", type=float, default=3.0)
    ap.add_argument("--log-mhost-ref", type=float, default=13.0)
    ap.add_argument("--log-mhost-scale", type=float, default=1.0)
    ap.add_argument("--out-dir", default="",
                    help="default $DMSR_REWARD_ROOT/lr_host_channel")
    ap.add_argument("--work-root", default="",
                    help="scratch for the LR Rockstar runs; default "
                         "$DMSR_REWARD_ROOT/lr_host_fidelity, which is where "
                         "an existing LR catalog already lives and will be "
                         "REUSED rather than re-found")
    ap.add_argument("--rerun-rockstar", action="store_true")
    ap.add_argument("--boxsize", type=float, default=BOX)
    ap.add_argument("--ng-hr", type=int, default=512,
                    help="HR grid the --min-hr-num-p gate is stated in")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild boxes whose field already exists. Without "
                         "it an existing file is left alone, so a resubmitted "
                         "job resumes instead of restarting.")
    args = ap.parse_args(argv)

    if args.reward_root:
        os.environ["DMSR_REWARD_ROOT"] = str(args.reward_root)
    cfg = load_reward_config(args)
    boxes = [b for b in str(args.boxes).replace(",", " ").split() if b]
    ccfg = LRHostChannelConfig(
        channels=args.channels, mode=args.mode,
        min_hr_num_p=int(args.min_hr_num_p), r_max_rvir=float(args.r_max_rvir),
        log_mhost_ref=float(args.log_mhost_ref),
        log_mhost_scale=float(args.log_mhost_scale))
    out_dir = Path(args.out_dir) if args.out_dir else paths.subdir(
        "lr_host_channel", create=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.work_root:
        args.work_root = str(paths.subdir("lr_host_fidelity", create=True))

    banner(f"LR-host conditioning field: {len(boxes)} box(es), "
           f"{ccfg.n_chan} channel(s) {list(ccfg.channels)}")
    print(f"  mode {ccfg.mode}, gate >={ccfg.min_hr_num_p} HR particles, "
          f"r_max {ccfg.r_max_rvir} R_vir -> {out_dir}", flush=True)

    rows: List[Dict] = []
    for box in boxes:
        path = channel_field_path(box, out_dir)
        if path.is_file() and not args.overwrite:
            print(f"\n{box}: {path.name} exists, skipping "
                  "(--overwrite to rebuild)", flush=True)
            continue
        print(f"\n{box}:", flush=True)
        rows.append(run_box(cfg, box, ccfg, args, out_dir))

    write_json(out_dir / "summary.json",
               {"ok": True, "config": ccfg.to_dict(), "boxes": boxes,
                "built": rows})
    print(f"\n  summary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
