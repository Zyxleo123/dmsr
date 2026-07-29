"""Short local CPU check of the riskiest assumption in Experiment 0.

Runs the REAL Rockstar binary twice on the same small clustered box -- once with
the frozen config and once with configs/sr2_baseline/rockstar_particles.cfg --
and checks, against actual Rockstar output rather than a synthetic fixture:

1. the member-id config really writes a `.particles` table;
2. the halo catalog is UNCHANGED by it (the flag is supposed to be output-only);
3. the grouped member sets are consistent with the catalog in the form that is
   actually true -- rows(o) >= num_p(o), equality exactly for leaves -- which is
   the invariant the whole tile decomposition rests on;
4. the printed `particle_id` values are our GADGET ids, so the flat-Lagrangian
   mapping is real and not an assumption.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/src")
from cosmo_sr.eval.particles import ParticleBox
from cosmo_sr.eval.rockstar import run_rockstar_on_particles
from cosmo_sr.reward.tiles import (
    TileGrid, member_weights_from_particles, stream_particle_tile_counts,
    tile_of_particle_id,
)

ROOT = Path("/zfsauton2/home/yixiz/DMSR/cosmo_sr_project")
_DEFAULT_WORK = os.environ.get("ZFS", "/zfsauton/scratch/yixiz") + "/tmp_rs_verify"
NG = 128                      # 2.1e6 particles
BOX = 100.0
N = NG ** 3


def make_box(seed=0):
    """Uniform background plus dense clumps, so Rockstar has halos to find.

    Particle ids stay the flat lattice index arange(N): that is the convention
    field_to_particles uses and the one tile_of_particle_id inverts.
    """
    rng = np.random.default_rng(seed)
    pos = rng.random((N, 3)) * BOX

    # 40 clumps, each massive enough to be found (>= 20 particles by a wide
    # margin) and compact enough to be bound.
    n_clump, per = 40, 8000
    centres = rng.random((n_clump, 3)) * BOX
    idx = rng.choice(N, size=n_clump * per, replace=False)
    for c in range(n_clump):
        sl = idx[c * per:(c + 1) * per]
        pos[sl] = (centres[c] + rng.normal(0.0, 0.35, size=(per, 3))) % BOX

    vel = rng.normal(0.0, 200.0, size=(N, 3)).astype(np.float32)
    omega_m, rho_crit = 0.2814, 2.77536627e11
    return ParticleBox(
        pos_mpc_h=np.ascontiguousarray(pos, dtype=np.float32),
        vel_kms=vel,
        ids=np.arange(N, dtype=np.int64),
        boxsize_mpc_h=BOX,
        redshift=0.0,
        particle_mass_msun_h=omega_m * rho_crit * BOX ** 3 / N,
    )


def main(argv=None):
    # argparse rather than sys.argv[1]: a bare `--help` would otherwise be taken
    # as the work directory and silently create a directory named "--help" in
    # whatever the cwd happens to be.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work_dir", nargs="?", default=_DEFAULT_WORK,
                    help="scratch directory for the two test runs "
                         f"(default: {_DEFAULT_WORK})")
    ap.add_argument("--keep", action="store_true",
                    help="keep the test catalogs instead of deleting them")
    args = ap.parse_args(argv)
    WORK = Path(args.work_dir)
    WORK.mkdir(parents=True, exist_ok=True)
    pb = make_box()
    print(f"box: {N} particles, m_p = {pb.particle_mass_msun_h:.3e} Msun/h", flush=True)

    frozen_cfg = ROOT / "configs" / "sr2_baseline" / "rockstar.cfg"
    parts_cfg = ROOT / "configs" / "sr2_baseline" / "rockstar_particles.cfg"

    print("\n--- run 1: frozen config ---", flush=True)
    cat_a = run_rockstar_on_particles(pb, WORK / "frozen", cfg=frozen_cfg,
                                      tag="box", overwrite=True)
    print(f"    {cat_a.n} objects", flush=True)

    print("\n--- run 2: member-id config ---", flush=True)
    cat_b = run_rockstar_on_particles(pb, WORK / "parts", cfg=parts_cfg,
                                      tag="box", overwrite=True)
    print(f"    {cat_b.n} objects", flush=True)

    ok = True

    # (2) the catalog must be unchanged
    print("\n[2] catalog identical under the member-id config?")
    if cat_a.n != cat_b.n:
        print(f"    FAIL: {cat_a.n} vs {cat_b.n} objects")
        ok = False
    else:
        oa, ob = np.argsort(cat_a.ids), np.argsort(cat_b.ids)
        dm = np.max(np.abs(cat_a.mvir[oa] - cat_b.mvir[ob]))
        dx = np.max(np.abs(cat_a.pos[oa] - cat_b.pos[ob]))
        dp = np.max(np.abs(cat_a.num_p[oa] - cat_b.num_p[ob]))
        same = bool(np.array_equal(cat_a.ids[oa], cat_b.ids[ob]))
        print(f"    same ids={same}  max|dMvir|={dm:.3e}  max|dpos|={dx:.3e}  "
              f"max|dnum_p|={dp}")
        if not (same and dm == 0 and dx == 0 and dp == 0):
            print("    FAIL: the output-only flag changed the catalog")
            ok = False
        else:
            print("    PASS")

    # (1) the table exists
    pf = sorted((WORK / "parts" / "box_rockstar").glob("*.particles"))
    print(f"\n[1] .particles written? {[p.name for p in pf]}")
    if not pf:
        print("    FAIL: no member table")
        return 1
    print(f"    PASS ({pf[0].stat().st_size / 2**20:.1f} MB for {N} particles "
          f"-> ~{pf[0].stat().st_size / 2**30 * (512**3 / N):.1f} GB at 512^3)")

    # (4) ids are ours
    grid = TileGrid(ng_hr=NG, tile_hr=NG // 8, boxsize_mpc_h=BOX)
    halo, tile, count = stream_particle_tile_counts(pf[0], grid, chunk_rows=500_000)
    print(f"\n[4] particle ids are GADGET/Lagrangian ids?")
    print(f"    {len(np.unique(halo))} objects, {count.sum()} member rows kept")
    print("    PASS (tile_of_particle_id accepted every id; it raises otherwise)")

    # (3) THE invariant, in its measured form: rows >= num_p, equality for leaves.
    print("\n[3] member sets consistent with the catalog?")
    from cosmo_sr.reward.tiles import check_member_consistency
    w = member_weights_from_particles(pf[0], grid, chunk_rows=500_000)
    c = check_member_consistency(cat_b, w)
    for k in ("n_catalog_objects", "n_with_member_rows", "n_missing",
              "n_rows_below_num_p", "n_leaf_exact", "n_with_substructure"):
        print(f"    {k:26s} {c[k]}")
    if not c["ok"]:
        print("    FAIL", c["missing"][:5], c["rows_below_num_p"][:5])
        ok = False
    else:
        print("    PASS")

    # partition of unity
    print(f"\n[5] partition of unity: max |sum_j w - 1| = {w.max_weight_error():.3e}")
    if w.max_weight_error() > 1e-12:
        ok = False

    print("\n=== OVERALL:", "PASS" if ok else "FAIL")
    if not args.keep:
        import shutil
        shutil.rmtree(WORK, ignore_errors=True)
        print(f"(removed {WORK}; pass --keep to inspect it)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
