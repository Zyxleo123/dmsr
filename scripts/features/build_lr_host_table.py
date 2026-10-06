#!/usr/bin/env python
"""The gated LR host table: centre, ``R_vir`` and ``M_vir`` per box.

Why a separate job
------------------
``--soft-host-profile lr`` needs, per supervised host, the LR catalog's
``(x, R_vir, M_vir)`` for the host that owns it. Reading that inside the
trainer would put an ASCII catalog parse -- and, for a box that has never been
run, a Rockstar invocation -- inside a GPU job, where it would be re-done every
resubmission and would hold a GPU while it ran. So it is precomputed here, on
the CPU partition, into one small ``.npz`` per box that the trainer memory-maps.

Every box this line trains on already HAS an LR catalog on disk (built for the
Lagrangian-host channel), so this job parses rather than recomputes and takes
seconds. ``--rerun-rockstar`` exists for a box that does not, and is the only
path here that costs real time.

Not leakage
-----------
The table is a function of the LR box alone -- Rockstar run on the 64^3 LR
particles -- which is exactly the input the generator gets at inference. This
is the same provenance argument as ``lr_host_channel``, and the same
incompleteness applies: ``lr-host-channel-fidelity`` measured the LR catalog
usable only above ~20k HR-equivalent particles, so the gate here drops the rest
rather than inventing a host for them. A supervised host with no gated LR owner
gets zeros, and the solve falls back to the flat background for it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reward"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from _common import add_common_args, banner, load_reward_config, write_json  # noqa: E402

from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.features.lr_host_channel import gate_hosts  # noqa: E402
from cosmo_sr.reward import paths  # noqa: E402

#: Where ``build_lagrangian_host.py`` and ``lr_host_fidelity.py`` leave the LR
#: catalogs. One place, so this job never runs Rockstar for a box that already
#: has one under a different root.
LR_ROCKSTAR = "lagrangian_host/{box}/{box}_lr_rockstar"


def host_table(cat, *, min_hr_num_p: int, ng_lr: int, ng_hr: int) -> Dict:
    """Gated top-level LR hosts as plain arrays, in Mpc/h and Msun/h."""
    rows = gate_hosts(cat, min_hr_num_p=int(min_hr_num_p), ng_lr=int(ng_lr),
                      ng_hr=int(ng_hr))
    pos = np.asarray(cat.pos, dtype=np.float64)[rows]
    # Rockstar reports R_vir in kpc/h and every consumer here works in Mpc/h.
    rvir = np.asarray(cat.rvir, dtype=np.float64)[rows] / 1000.0
    mvir = np.asarray(cat.mvir, dtype=np.float64)[rows]
    return {"host_pos_mpc_h": pos, "host_rvir_mpc_h": rvir,
            "host_mvir_msun_h": mvir,
            "host_id": np.asarray(cat.ids, dtype=np.int64)[rows],
            "host_num_p": np.asarray(cat.num_p, dtype=np.int64)[rows]}


def table_path(root: Path, box: str) -> Path:
    return Path(root) / "lr_host_table" / f"{box}.npz"


def run_box(cfg, box: str, args, out_dir: Path) -> Dict:
    root = Path(args.work_root)
    d = root / LR_ROCKSTAR.format(box=box)
    ascii_files = sorted(d.glob("halos_*.ascii"))
    if not ascii_files:
        if not bool(args.rerun_rockstar):
            raise SystemExit(
                f"no LR catalog for {box} under {d}. Every box this line has "
                "trained on already has one (built for the Lagrangian-host "
                "channel); pass --rerun-rockstar to build it here, which costs "
                "a full Rockstar run and is why it is not the default.")
        from lr_host_fidelity import lr_catalog  # noqa: E402  (heavy import)
        cat, _ = lr_catalog(cfg, box, root, rerun=True)
    else:
        cat = load_rockstar_ascii(ascii_files[0])

    t = host_table(cat, min_hr_num_p=int(args.min_hr_num_p),
                   ng_lr=int(args.ng_lr), ng_hr=int(args.ng_hr))
    out = table_path(out_dir, box)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, min_hr_num_p=np.int64(args.min_hr_num_p),
             ng_lr=np.int64(args.ng_lr), ng_hr=np.int64(args.ng_hr), **t)
    rep = {
        "box": box, "path": str(out), "n_lr_halos": int(cat.n),
        "n_gated_hosts": int(t["host_rvir_mpc_h"].size),
        "min_hr_num_p": int(args.min_hr_num_p),
    }
    if t["host_rvir_mpc_h"].size:
        rep["median_rvir_mpc_h"] = float(np.median(t["host_rvir_mpc_h"]))
        rep["log10_mvir_range"] = [float(np.log10(t["host_mvir_msun_h"].min())),
                                   float(np.log10(t["host_mvir_msun_h"].max()))]
    print(f"  {box}: {rep['n_gated_hosts']} gated hosts of {cat.n} LR halos",
          flush=True)
    return rep


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--boxes", nargs="+", required=True)
    ap.add_argument("--reward-root", default="")
    ap.add_argument("--out-dir", default="",
                    help="default $DMSR_REWARD_ROOT/lr_host_table, which is "
                         "where the trainer looks when given no --lr-host-table")
    ap.add_argument("--work-root", default="",
                    help="where the LR Rockstar catalogs already live; default "
                         "$DMSR_REWARD_ROOT, under which build_lagrangian_host "
                         "left one per box and which is REUSED, never re-found")
    ap.add_argument("--min-hr-num-p", type=int, default=20000,
                    help="the LR completeness gate, in HR particles. 20000 is "
                         "where lr-host-channel-fidelity measured the LR "
                         "catalog stop being usable.")
    ap.add_argument("--ng-lr", type=int, default=64)
    ap.add_argument("--ng-hr", type=int, default=512)
    ap.add_argument("--rerun-rockstar", action="store_true")
    args = ap.parse_args(argv)

    banner("LR host table")
    if args.reward_root:
        os.environ["DMSR_REWARD_ROOT"] = str(args.reward_root)
    cfg = load_reward_config(args)
    if not args.work_root:
        args.work_root = str(paths.reward_root())
    out_dir = (Path(args.out_dir) if args.out_dir
               else paths.subdir("lr_host_table", create=True).parent)
    boxes = [b for b in " ".join(args.boxes).replace(",", " ").split() if b]
    rows = [run_box(cfg, b, args, out_dir) for b in boxes]
    summary = out_dir / "lr_host_table" / "summary.json"
    write_json(summary, {"boxes": rows})
    print(f"\nwrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
