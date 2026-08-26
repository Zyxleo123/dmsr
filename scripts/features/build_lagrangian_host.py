#!/usr/bin/env python
"""Run Rockstar on the LR box and cache the Lagrangian host features.

One command per box. It

1. loads the LR field ``(6, 64, 64, 64)`` and turns it into particles whose ids
   are the flat Lagrangian index (``cosmo_sr.eval.particles.field_to_particles``);
2. runs Rockstar on that periodic box with
   ``configs/sr2_baseline/rockstar_lr_particles.cfg`` (member-id output on, LR
   force resolution) -- or reuses an earlier run's catalog and member table;
3. streams the member table into ``owner[lr_particle_id]``;
4. builds the 64^3 feature volumes with
   :func:`cosmo_sr.features.build_host_features` and writes them, plus a
   normalisation report, next to the catalog.

The member table is ~20 MB at 64^3 (not the ~7 GB the HR box produces), so it is
kept: rebuilding the features after a definition change then costs seconds and
does not re-run the halo finder.

    python scripts/features/build_lagrangian_host.py --boxes set8,set9
    python scripts/features/build_lagrangian_host.py --boxes set8 --rerun-rockstar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (  # noqa: E402
    add_common_args, banner, load_reward_config, lr_path, paths, write_json,
)

from cosmo_sr.eval.particle_identity import (  # noqa: E402
    build_owner_index, stream_owner_assignment,
)
from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.eval.rockstar import (  # noqa: E402
    load_rockstar_ascii, run_rockstar_on_particles,
)
from cosmo_sr.features import (  # noqa: E402
    LagrangianGrid, build_host_features, normalization_report,
)

LR_CFG = PROJECT_ROOT / "configs" / "sr2_baseline" / "rockstar_lr_particles.cfg"


def feature_root(create: bool = False) -> Path:
    return paths.subdir("lagrangian_host", create=create)


def box_dir(box: str, create: bool = False) -> Path:
    return paths.subdir("lagrangian_host", box, create=create)


def _grid(cfg, ng_lr: int) -> LagrangianGrid:
    d = cfg.get("data", {})
    t = cfg.get("tiles", {})
    return LagrangianGrid(
        ng_lr=int(ng_lr),
        ng_hr=int(d.get("ng_hr", 512)),
        tile_hr=int(t.get("tile_hr", 64)),
        boxsize_mpc_h=float(d.get("boxsize_mpc_h", 100.0)),
    )


def _catalog_and_members(work: Path, tag: str, particles, reuse: bool):
    """The LR catalog plus the path of its member-particle table.

    A catalog directory without a ``.particles`` table is not a usable cache
    here -- the features are built *from* that table -- so its absence forces a
    rerun rather than a confusing failure three steps later.
    """
    rk_dir = work / f"{tag}_rockstar"
    tables = sorted(rk_dir.glob("*.particles")) if rk_dir.is_dir() else []
    if reuse and tables:
        lists = sorted(set(list(rk_dir.glob("halos*.ascii"))
                           + list(rk_dir.glob("halos*.list"))
                           + list(rk_dir.glob("out_*.list"))))
        if lists:
            print(f"    reusing {lists[0].name} + {tables[0].name}")
            return load_rockstar_ascii(lists[0]), tables[0]

    print(f"    running Rockstar on {particles.ids.size} LR particles ...")
    t0 = time.time()
    cat = run_rockstar_on_particles(
        particles, work, cfg=LR_CFG, tag=tag, overwrite=True,
    )
    print(f"    Rockstar done in {time.time() - t0:.1f}s -> {cat.n} objects")
    tables = sorted((work / f"{tag}_rockstar").glob("*.particles"))
    if not tables:
        raise SystemExit(
            f"Rockstar wrote no .particles table under {work}; "
            f"{LR_CFG} must set FULL_PARTICLE_CHUNKS = 1")
    return cat, tables[0]


def build_one(cfg, box: str, args) -> dict:
    banner(f"box {box}")
    field_path = Path(args.field_npy) if args.field_npy else lr_path(cfg, box)
    if not field_path.is_file():
        raise SystemExit(f"no LR field at {field_path}")
    field = np.load(field_path, mmap_mode="r")
    ng_lr = int(field.shape[1])
    grid = _grid(cfg, ng_lr)
    print(f"    field {field_path}  shape={tuple(field.shape)}")
    print(f"    grid  ng_lr={grid.ng_lr} ng_hr={grid.ng_hr} "
          f"upsample={grid.upsample} tile_lr={grid.tile_lr} "
          f"n_tiles={grid.n_tiles} cell={grid.cell_mpc_h:.4f} Mpc/h")

    work = box_dir(box, create=True)
    out_npz = work / f"{box}_lagrangian_host.npz"
    out_json = work / f"{box}_lagrangian_host.json"

    particles = field_to_particles(
        np.asarray(field, dtype=np.float32),
        boxsize_kpc_h=float(grid.boxsize_mpc_h) * 1000.0,
        redshift=float(args.redshift),
    )
    print(f"    particle mass {particles.particle_mass_msun_h:.3e} Msun/h")

    cat, table_path = _catalog_and_members(
        work, f"{box}_lr", particles, reuse=not args.rerun_rockstar)
    hosts = cat.hosts()
    print(f"    catalog: {cat.n} objects, {hosts.n} hosts, "
          f"{cat.n - hosts.n} subhalos")
    if cat.n == 0:
        raise SystemExit(
            f"the LR catalog for {box} is empty -- at the LR particle mass the "
            "20-particle floor may be above every halo in this box")

    owner = stream_owner_assignment(table_path, grid.n_lr)
    idx = build_owner_index(owner)
    print(f"    ownership: {grid.n_lr - idx.n_unowned}/{grid.n_lr} sites bound "
          f"({100.0 * (1 - idx.n_unowned / grid.n_lr):.1f}%) over "
          f"{idx.halo_id.size} leaf objects")

    feat = build_host_features(
        cat, owner, grid, box=box, source="lr_rockstar",
        with_subhalo_budget=not args.no_budget,
    )
    rep = normalization_report(feat)
    rep.update(
        box=box, field=str(field_path), catalog=str(cat.path),
        member_table=str(table_path),
        particle_mass_msun_h=float(particles.particle_mass_msun_h),
        n_catalog_objects=int(cat.n), n_catalog_hosts=int(hosts.n),
        rockstar_cfg=str(LR_CFG), features=str(out_npz),
    )
    feat.to_npz(out_npz)
    write_json(out_json, rep)

    print(f"    hosts on the lattice : {rep['n_hosts']}")
    print(f"    sites with a host    : {rep['n_sites_with_host']} "
          f"({100 * rep['frac_sites_with_host']:.1f}%)")
    print(f"    tiles per host       : median {rep.get('median_tiles_per_host')}, "
          f"max {rep.get('max_tiles_per_host')}, "
          f"{rep.get('n_hosts_spanning_tiles')} span >1")
    print(f"    sum_t f[h,t] - 1     : max |err| {rep['max_abs_tile_frac_error']:.2e}")
    print(f"    sum_i lambda_i - N_h : max |err| {rep['max_abs_budget_error']:.2e}")
    print(f"    normalisation ok     : {rep['ok']}")
    print(f"    wrote {out_npz}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--boxes", default="set8",
                    help="comma-separated box names (default set8)")
    ap.add_argument("--field-npy", default="",
                    help="use this LR field instead of the dataset one "
                         "(single box only)")
    ap.add_argument("--redshift", type=float, default=0.0)
    ap.add_argument("--rerun-rockstar", action="store_true",
                    help="re-run the halo finder even if a catalog + member "
                         "table already exist")
    ap.add_argument("--no-budget", action="store_true",
                    help="skip the optional subhalo_budget channel")
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    if args.field_npy and len(boxes) != 1:
        raise SystemExit("--field-npy applies to a single --boxes entry")

    reports = [build_one(cfg, b, args) for b in boxes]
    banner("summary")
    for r in reports:
        print(f"  {r['box']}: {r['n_hosts']} hosts, "
              f"{100 * r['frac_sites_with_host']:.1f}% of sites bound, "
              f"ok={r['ok']}  -> {r['features']}")
    write_json(feature_root(create=True) / "build_summary.json", reports)
    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
