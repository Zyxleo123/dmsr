#!/usr/bin/env python
"""Write a GADGET-2 snapshot and a GADGET-4 parameter file for a SUBFIND run.

Why this stage exists at all
----------------------------
Every subhalo number in this project comes from Rockstar, which links particles
in **six dimensions**. That is its strength and also why it cannot referee
itself: a phase-space finder can report an object that has no *spatial*
overdensity behind it, and "Rockstar says +347 subhalos, the picture shows
nothing" is exactly the shape a fine-tune sharpening velocities without moving
mass would produce. SUBFIND disagrees with Rockstar in the right way -- it
defines substructure as a locally overdense region bounded by a **saddle point
of the 3-D density field** and keeps it only if it is self-bound, with
velocities entering once, at the end, as an unbinding test and never as a
linking dimension. An object both finders report is real by two independent
definitions; one only Rockstar reports is the false positive the picture would
be right about.

GADGET-4 is run in post-processing mode for this (``Gadget4 param.txt 3 <num>``),
which reads a snapshot, runs FOF and SUBFIND, writes the group catalogue and
exits. Nothing is re-simulated.

What this script produces
-------------------------
``<out-dir>/snapshot_<NNN>``  the legacy format-1 binary, byte-identical in
construction to the dumps Rockstar reads (:func:`write_gadget2_snapshot`), so
both finders see *the same particles* -- the comparison is about the algorithms,
not about two slightly different fields.

``<out-dir>/param.txt``  the GADGET-4 parameter file, with the box, the
cosmology and the particle count filled in from the field itself rather than
assumed.

``<out-dir>/manifest.json``  what was written, from which field, with which
particle mass -- so a catalogue can always be traced back to its source.

The smoke mode
--------------
``--smoke N`` writes a small synthetic snapshot instead of reading a field: an
``N^3`` lattice with a handful of planted, self-bound clumps whose centres and
particle counts are recorded in the manifest. It costs a second and it is the
only cheap way to find out whether the whole chain -- the header GADGET-4 reads,
the parameter file it accepts, the catalogue we parse -- works, **before** 88
cores spend an hour on a 512^3 box. Because the answer is known, it also checks
that the pipeline finds what is there rather than merely running.

    python scripts/features/subfind_snapshot.py --smoke 64 --out-dir /tmp/g4smoke
    python scripts/features/subfind_snapshot.py --box set9 --source hr \\
        --out-dir $SUBFIND_ROOT/set9__hr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.eval.particles import (  # noqa: E402
    ParticleBox, field_to_particles, particle_mass_msun_h,
    write_gadget2_snapshot,
)

#: The frozen simulation cosmology, the same numbers configs/sr2_baseline/
#: rockstar.cfg states. Duplicated here rather than imported because GADGET-4
#: needs them in its own parameter file, and a silent divergence between the two
#: finders' cosmologies would be invisible in the results.
OMEGA_M = 0.2814
OMEGA_L = 0.7186
HUBBLE_PARAM = 0.697
BOXSIZE_MPC_H = 100.0

#: Force resolution, Mpc/h. Matches Rockstar's FORCE_RES so neither finder has a
#: softer potential than the other when it decides what is bound.
SOFTENING_MPC_H = 0.005


def _param_txt(out_dir: Path, snap_base: str, n_part: int, boxsize: float,
               *, max_mem_mb: int, softening: float, time_limit_s: int,
               des_link_ngb: int = 20) -> str:
    """The GADGET-4 parameter file for a post-processing FOF/SUBFIND run.

    Only the tags a DM-only, non-integrating run actually reads. The time
    parameters still have to be present and consistent (``TimeBegin ==
    TimeMax == 1`` is z = 0) even though nothing is integrated.

    Units are the project's: Mpc/h, 1e10 Msun/h, km/s -- matching both the
    snapshot this script writes and ``GADGET_LENGTH_CONVERSION = 1`` in the
    Rockstar config.
    """
    return f"""%---- GADGET-4 post-processing FOF/SUBFIND.
%     Written by scripts/features/subfind_snapshot.py -- do not hand-edit; the
%     values below are derived from the snapshot next to it.
%     Run with:  mpirun -np <N> ./Gadget4 param.txt 3 0

%---- Files. OutputDir MUST keep its trailing slash: GADGET-4 builds the
%     snapshot path as OutputDir + SnapshotFileBase + "_%03d" with no separator
%     of its own (src/main/main.cc), so dropping it looks for the file one
%     directory up under a mangled name.
InitCondFile              {out_dir}/{snap_base}_{int(0):03d}
OutputDir                 {out_dir}/
SnapshotFileBase          {snap_base}
OutputListFilename        {out_dir}/outputs.txt

%---- Formats. In post-processing mode 3 the snapshot is read with ICFormat
%     (only mode 2 copies SnapFormat over it), so 1 = our legacy GADGET-2
%     binary in. SnapFormat governs what the GROUP CATALOGUE is written as, so
%     3 = HDF5 out, which is what subfind_catalog.py parses.
ICFormat                  1
SnapFormat                3

%---- CPU-time limits
TimeLimitCPU              {int(time_limit_s)}
CpuTimeBetRestartFile     7200

%---- Memory, MByte PER MPI RANK. {n_part} particles.
MaxMemSize                {int(max_mem_mb)}

%---- Nothing is integrated: begin and end at a = 1 (z = 0).
TimeBegin                 1.0
TimeMax                   1.0
ComovingIntegrationOn     1

%---- Cosmology (frozen; identical to configs/sr2_baseline/rockstar.cfg)
Omega0                    {OMEGA_M}
OmegaLambda               {OMEGA_L}
OmegaBaryon               0.0
HubbleParam               {HUBBLE_PARAM}
Hubble                    100.0
BoxSize                   {boxsize}

%---- Output. OutputListOn=1 reads outputs.txt (one line, "1.0" = z=0) rather
%     than deriving output times from a cadence. That is the honest setting --
%     post-processing writes one group catalogue and no snapshots at all -- and
%     it also skips a startup check that only runs when OutputListOn=0:
%     begrun.cc:184 terminates on log(TimeBetSnapshot) < MaxSizeTimestep, which
%     a cadence of 1.0 fails outright since log(1)=0 (job 62537).
%     TimeBetSnapshot is unused here but kept above that bound so flipping the
%     flag does not resurrect the trap.
OutputListOn              1
TimeBetSnapshot           2.0
TimeOfFirstSnapshot       0.0
TimeBetStatistics         0.01
NumFilesPerSnapshot       1
MaxFilesWithConcurrentIO  1

%---- Time integration accuracy (unused here, but parsed)
ErrTolIntAccuracy         0.01
CourantFac                0.3
MaxSizeTimestep           0.005
MinSizeTimestep           0.0

%---- Tree / force accuracy
TypeOfOpeningCriterion            1
ErrTolTheta                       0.75
ErrTolThetaMax                    1.0
ErrTolForceAcc                    0.002
TopNodeFactor                     3.0
ActivePartFracForNewDomainDecomp  0.01
ActivePartFracForPMinsteadOfEwald 0.05

%---- SUBFIND's two neighbour numbers, and they are NOT the same knob:
%       DesNumNgb  (64) smooths the DENSITY field the excursion set descends.
%       DesLinkNgb (20) is how many nearest neighbours each particle consults
%                       for denser companions when the set is walked down --
%                       Springel et al. 2001's DesLinkNgb. It sets how early two
%                       peaks are judged to touch, i.e. where a subhalo ends.
%     Both are the standard values; changing either changes the catalogue.
DesNumNgb                 64
MaxNumNgbDeviation        1
DesLinkNgb                {des_link_ngb}

%---- SPH parameters. This run has no gas -- NTYPES=2 carries an empty type 0 --
%     but the parameter file is parsed against the compiled-in tag list, not
%     against the particles present, so GADGET-4 refuses to start without them
%     ("Error. I miss a value for tag 'ArtBulkViscConst'", job 62460). They are
%     never read: nothing here integrates hydrodynamics.
ArtBulkViscConst          1.0
MinEgySpec                0
InitGasTemp               0

%---- Units: Mpc/h, 1e10 Msun/h, km/s
UnitLength_in_cm          3.085678e24
UnitMass_in_g             1.989e43
UnitVelocity_in_cm_per_s  1e5
GravityConstantInternal   0

%---- Gravitational softening
SofteningComovingClass0   {softening}
SofteningMaxPhysClass0    {softening}
SofteningClassOfPartType0 0
SofteningClassOfPartType1 0
"""


def _smoke_particles(ng: int, boxsize: float, n_clumps: int,
                     clump_n: int, seed: int) -> tuple:
    """A uniform lattice plus planted, self-bound clumps; returns the answer too.

    The clumps are drawn from a Gaussian of width ``sigma`` with velocities
    small compared with their own escape speed, so SUBFIND must both *find*
    them (they are density peaks bounded by saddles) and *keep* them (they
    unbind to almost nothing lost). A pipeline that returns zero subhalos here
    is broken in a way the 512^3 run would take an hour to reveal.
    """
    rng = np.random.default_rng(int(seed))
    q = (np.arange(ng, dtype=np.float64) + 0.5) * (boxsize / ng)
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    # A little jitter, or the lattice is a perfect crystal and every particle
    # has the same density -- which is a degenerate input for a density-ordered
    # algorithm, not a representative one.
    grid += rng.normal(0.0, 0.15 * boxsize / ng, size=grid.shape)
    n_bg = grid.shape[0]
    m_p = particle_mass_msun_h(OMEGA_M, boxsize, n_bg)

    centres, parts, vels = [], [grid], [rng.normal(0.0, 50.0, size=grid.shape)]
    for k in range(int(n_clumps)):
        c = rng.uniform(0.15 * boxsize, 0.85 * boxsize, size=3)
        sigma = 0.004 * boxsize
        p = c[None, :] + rng.normal(0.0, sigma, size=(int(clump_n), 3))
        # Virial-ish dispersion for this mass and size, then a bulk motion, so
        # the clump is bound in its own frame but moving in the box frame --
        # the situation a subhalo is actually in.
        v_c = np.sqrt(4.30091e-9 * m_p * clump_n / sigma)
        v = (rng.normal(0.0, 0.35 * v_c, size=(int(clump_n), 3))
             + rng.normal(0.0, 200.0, size=3)[None, :])
        centres.append({"index": k, "centre_mpc_h": c.tolist(),
                        "n_particles": int(clump_n),
                        "sigma_mpc_h": float(sigma),
                        "v_circ_kms": float(v_c)})
        parts.append(p)
        vels.append(v)

    pos = np.concatenate(parts, axis=0) % boxsize
    vel = np.concatenate(vels, axis=0)
    ids = np.arange(pos.shape[0], dtype=np.int64)
    box = ParticleBox(
        pos_mpc_h=np.ascontiguousarray(pos, dtype=np.float32),
        vel_kms=np.ascontiguousarray(vel, dtype=np.float32),
        ids=ids, boxsize_mpc_h=float(boxsize), redshift=0.0,
        particle_mass_msun_h=float(m_p))
    return box, centres


def _resolve_field(args) -> Path:
    """The field to convert: an explicit path, or box+source through paths."""
    if args.field:
        return Path(args.field)
    from cosmo_sr.reward import paths  # noqa: E402  (needs the reward config)
    if args.source == "base":
        hits = sorted(Path(paths.SR2_BASE_CACHE()).glob(
            f"{args.box}_seed{int(args.seed)}_*.npy"))
        if not hits:
            raise SystemExit(
                f"no frozen SR2 cache for {args.box} seed {args.seed}; run "
                "scripts/slurm/cache_sr2_base.sbatch first")
        return hits[0]
    if args.source == "candidate":
        p = (paths.reward_root() / "flow_rockstar" / "fields"
             / f"{args.box}__{args.tag}__seed{int(args.seed)}.npy")
        if not p.is_file():
            raise SystemExit(
                f"no spliced field {p}; run scripts/slurm/gather_splice_cpu.sbatch "
                f"for tag {args.tag} first")
        return p
    # hr: the paired dataset's own field, the same one _common.hr_path builds.
    p = Path(args.data_root) / "hr" / f"{args.box}.npy"
    if not p.is_file():
        raise SystemExit(f"no HR field at {p}; check --data-root")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True,
                    help="GADGET-4 OutputDir; the snapshot is written into it")
    ap.add_argument("--field", default="", help="a (6, Ng, Ng, Ng) .npy field")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--source", default="hr",
                    choices=("hr", "base", "candidate"),
                    help="where to find the field when --field is not given")
    ap.add_argument("--tag", default="", help="candidate tag, e.g. mgho_<arm>_set9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root",
                    default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm",
                    help="paired dataset root; the HR field is <root>/hr/<box>.npy")
    ap.add_argument("--snap-base", default="snapshot")
    ap.add_argument("--snapnum", type=int, default=0)
    ap.add_argument("--boxsize", type=float, default=BOXSIZE_MPC_H)
    ap.add_argument("--softening", type=float, default=SOFTENING_MPC_H)
    ap.add_argument("--max-mem-mb", type=int, default=7000,
                    help="GADGET-4 MaxMemSize, MByte PER RANK")
    ap.add_argument("--time-limit-s", type=int, default=86400)
    ap.add_argument("--des-link-ngb", type=int, default=20,
                    help="SUBFIND's DesLinkNgb: neighbours consulted for denser "
                         "companions in the excursion set, i.e. where a subhalo "
                         "is judged to end. 20 is the published default")
    ap.add_argument("--smoke", type=int, default=0,
                    help="write an N^3 synthetic snapshot with planted clumps "
                         "instead of reading a field")
    ap.add_argument("--smoke-clumps", type=int, default=6)
    ap.add_argument("--smoke-clump-n", type=int, default=400)
    ap.add_argument("--force", action="store_true",
                    help="rewrite the snapshot even if it is already there")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = out_dir / f"{args.snap_base}_{int(args.snapnum):03d}"

    planted: Optional[List[Dict]] = None
    source_desc: str
    if args.smoke > 0:
        pb, planted = _smoke_particles(int(args.smoke), float(args.boxsize),
                                       args.smoke_clumps, args.smoke_clump_n,
                                       seed=args.seed)
        source_desc = (f"smoke {args.smoke}^3 + {args.smoke_clumps} clumps "
                       f"of {args.smoke_clump_n}")
    else:
        field_path = _resolve_field(args)
        print(f"=== field: {field_path}", flush=True)
        field = np.load(field_path, mmap_mode="r")
        pb = field_to_particles(np.asarray(field),
                                boxsize_kpc_h=float(args.boxsize) * 1000.0,
                                redshift=0.0)
        del field
        source_desc = str(field_path)

    n = int(pb.ids.shape[0])
    print(f"=== {n} particles, m_p = {pb.particle_mass_msun_h:.4e} Msun/h, "
          f"box {pb.boxsize_mpc_h} Mpc/h", flush=True)

    if snap.is_file() and not args.force:
        print(f"=== snapshot already present, keeping it: {snap}", flush=True)
    else:
        write_gadget2_snapshot(str(snap), pb)
        print(f"=== wrote {snap} ({snap.stat().st_size / 1e9:.2f} GB)", flush=True)

    # GADGET-4 insists on an output list file even with OutputListOn = 0.
    (out_dir / "outputs.txt").write_text("1.0\n")
    param = out_dir / "param.txt"
    param.write_text(_param_txt(
        out_dir, args.snap_base, n, float(pb.boxsize_mpc_h),
        max_mem_mb=args.max_mem_mb, softening=float(args.softening),
        time_limit_s=args.time_limit_s, des_link_ngb=int(args.des_link_ngb)))
    print(f"=== wrote {param}", flush=True)

    manifest = {
        "source": source_desc, "box": args.box, "tag": args.tag,
        "snapshot": str(snap), "snapnum": int(args.snapnum),
        "n_particles": n,
        "particle_mass_msun_h": float(pb.particle_mass_msun_h),
        "boxsize_mpc_h": float(pb.boxsize_mpc_h),
        "softening_mpc_h": float(args.softening),
        "des_num_ngb": 64, "des_link_ngb": int(args.des_link_ngb),
        "omega_m": OMEGA_M, "omega_l": OMEGA_L, "hubble_param": HUBBLE_PARAM,
    }
    if planted is not None:
        manifest["planted_clumps"] = planted
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"=== wrote {out_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
