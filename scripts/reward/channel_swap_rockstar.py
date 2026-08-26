#!/usr/bin/env python
"""Channel-swap intervention: which half of the SR2 field costs it the subhalos?

`docs/sr2_subhalo_deficit.md` establishes *what* is missing -- SR2 keeps 97% of
the bound particles and 46% of the subhalos -- and a follow-up measurement on
the two catalogs adds that SR2's halos are internally **too cold**: at fixed
Mvir their `vrms` is 0.82-0.94 of HR's and their `T/|U|` sits at 0.38-0.51
against HR's 0.54-0.65, i.e. *below* virial equilibrium, while their NFW
concentration is if anything *higher*. Rockstar is a 6D finder, so a subhalo has
to be compact in position **and** velocity to be found at all.

That leaves one question no further statistic can settle, because it is a
question about cause: the cold interiors could be why the substructure is
missing, or merely what missing substructure looks like afterwards. Global field
statistics cannot separate them either -- over 24 random slabs the SR2/HR
cell-scale power ratio is 0.95-1.09 for the velocity channels and 0.74-0.90 for
the displacement ones, because ~85% of the volume is smooth field and swamps the
halo interiors.

So intervene instead. The canonical field is `(6, 512, 512, 512)` with
`disp[0:3]` and `vel[3:6]`, and HR and SR2 share a Lagrangian lattice -- particle
`i` is the same particle in both boxes. Assembling one field's displacement with
the other's velocity is therefore a per-particle-consistent swap, and running the
frozen halo finder on it asks the finder directly which half it needed:

| arm | displacement | velocity | reads as |
| --- | --- | --- | --- |
| `srpos_hrvel` | SR2 | HR | subhalos recovered => the velocity head is the fault |
| `hrpos_srvel` | HR | SR2 | subhalos destroyed => same conclusion, other side |

Both controls already exist and are **not** re-run: the frozen `hr` and `base`
catalogs under `$REWARD_ROOT/halos/<box>__<src>__<src>/` came from exactly these
two fields.

**What this is not.** A swapped box is not a self-consistent N-body state, and
nothing here claims it is. It is a probe of what the halo finder needs. One
consequence is worth watching rather than hiding: Rockstar unbinds particles
whose kinetic energy exceeds their potential, so handing SR2's cold velocities to
HR's positions under-unbinds and handing HR's hot ones to SR2's positions
over-unbinds. The report therefore carries bound-particle occupancy next to every
count, so a count that moved only because the unbinding step moved is visible as
such instead of being read as recovered substructure.

Config note: this runs the **frozen** `rockstar.cfg`, not the member-id
`rockstar_particles.cfg`, because only the catalog is needed. The two are already
known to agree object-by-object -- `particles_report.json` records
`verify_frozen: {agree: true, max_dmvir: 0.0, max_dnum_p: 0}` for both sources --
so the swap arms are comparable to the frozen controls.

    python scripts/reward/channel_swap_rockstar.py --box set8 --arm srpos_hrvel
    python scripts/reward/channel_swap_rockstar.py --box set8 --arm hrpos_srvel
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
    add_common_args, banner, hr_path, load_reward_config, paths, write_json,
)

from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.eval.rockstar import run_rockstar_on_particles  # noqa: E402

FROZEN_CFG = PROJECT_ROOT / "configs" / "sr2_baseline" / "rockstar.cfg"

#: ``arm -> {"disp": source, "vel": source}``. ``hr``/``base`` are the two pure
#: controls; they are listed so the same code path can reproduce a control if the
#: frozen catalog is ever in doubt, not because the pipeline re-runs them.
ARMS: dict[str, dict[str, str]] = {
    "srpos_hrvel": {"disp": "base", "vel": "hr"},
    "hrpos_srvel": {"disp": "hr", "vel": "base"},
    "hr": {"disp": "hr", "vel": "hr"},
    "base": {"disp": "base", "vel": "base"},
}

#: Channel groups of the canonical ``(6, Ng, Ng, Ng)`` catnorm field.
GROUPS: dict[str, slice] = {"disp": slice(0, 3), "vel": slice(3, 6)}


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/reward/test_channel_swap.py)
# --------------------------------------------------------------------------

def assemble_field(sources: dict[str, np.ndarray], spec: dict[str, str],
                   *, dtype=np.float32) -> np.ndarray:
    """One ``(6, N, N, N)`` field, each channel group taken from its named source.

    ``sources`` maps a source name (``"hr"``, ``"base"``) to a field -- a memmap
    is fine, only the requested slice is read. ``spec`` maps a group name in
    :data:`GROUPS` to a source name.

    The slice is copied group by group *from the same channel positions* it will
    occupy in the output: displacement stays channels 0:3 and velocity stays
    3:6, so a swap exchanges the provenance of a group and never its meaning.
    Getting that backwards would produce a field that still looks plausible --
    hence the test.
    """
    shapes = {tuple(np.shape(f)) for f in sources.values()}
    if len(shapes) != 1:
        raise ValueError(f"sources disagree on shape: {sorted(shapes)}")
    shape = shapes.pop()
    if len(shape) != 4 or shape[0] != 6:
        raise ValueError(f"expected a (6, N, N, N) catnorm field, got {shape}")
    missing = set(GROUPS) - set(spec)
    if missing:
        raise ValueError(f"spec does not name a source for {sorted(missing)}")

    out = np.empty(shape, dtype=dtype)
    for group, sl in GROUPS.items():
        name = spec[group]
        if name not in sources:
            raise ValueError(f"spec wants group {group!r} from unknown source "
                             f"{name!r}; have {sorted(sources)}")
        out[sl] = np.asarray(sources[name][sl], dtype=dtype)
    return out


def arm_spec(arm: str) -> dict[str, str]:
    """The channel-source mapping of a named arm, or a clear error."""
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    return dict(ARMS[arm])


def catalog_counts(cat) -> dict:
    """Host / subhalo counts and bound-particle occupancy of one catalog.

    ``num_p`` is per-object and does **not** include a halo's substructure --
    `build_lagrangian_host` verifies exactly that identity against the member
    table for 100.0% of hosts -- so summing it over every object counts each
    bound particle once. Checked against the independent owner arrays on set8:
    this gives 0.5397 (HR) and 0.5149 (SR2), matching
    ``1 - frac_particles_unowned`` in each ``particles_report.json`` to four
    decimals. (`docs/sr2_subhalo_deficit.md` quotes 0.5683/0.5537 for
    "occupancy" -- a different denominator, not a disagreement with this.)

    It is carried next to every count so a change driven by Rockstar's unbinding
    step is visible as one rather than being read as recovered substructure.
    """
    is_sub = np.asarray(cat.parent_ids) >= 0
    num_p = np.asarray(cat.num_p, dtype=np.int64)
    return {
        "n_objects": int(cat.n),
        "n_hosts": int((~is_sub).sum()),
        "n_subhalos": int(is_sub.sum()),
        "n_bound_particles": int(num_p.sum()),
        "sub_particles": int(num_p[is_sub].sum()),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def source_field(cfg, box: str, source: str, base_seed: int) -> np.ndarray:
    """HR from the paired dataset, ``base`` from the frozen SR2 cache (memmap)."""
    if source == "hr":
        p = hr_path(cfg, box)
        if not Path(p).is_file():
            raise SystemExit(f"no HR field at {p}")
        return np.load(p, mmap_mode="r")
    hits = sorted(Path(paths.SR2_BASE_CACHE()).glob(f"{box}_seed{int(base_seed)}_*.npy"))
    if not hits:
        raise SystemExit(
            f"no frozen SR2 cache for {box} seed {base_seed} under "
            f"{paths.SR2_BASE_CACHE()}; run scripts/slurm/cache_sr2_base.sbatch")
    if len(hits) > 1:
        raise SystemExit(
            f"{len(hits)} SR2 caches match {box} seed {base_seed}: "
            + ", ".join(h.name for h in hits)
            + " -- refusing to guess which model produced the frozen catalog")
    return np.load(hits[0], mmap_mode="r")


def run_arm(cfg, box: str, arm: str, args) -> dict:
    banner(f"{box} / arm {arm}")
    spec = arm_spec(arm)
    work = paths.subdir("channel_swap", f"{box}__{arm}", create=True)
    out_json = work / f"{box}_{arm}_summary.json"

    # Resumability: assembling the field and writing the 3 GB snapshot costs
    # minutes before the halo finder would decide it has nothing to do, so the
    # already-done check happens here rather than inside run_rockstar_on_particles.
    if args.reuse and out_json.is_file() and any(
            (work / f"{arm}_rockstar").glob("halos*.ascii")):
        banner(f"{box}/{arm}: catalog already built -> {out_json}")
        return json.loads(out_json.read_text())

    sources = {}
    provenance = {}
    for source in sorted(set(spec.values())):
        f = source_field(cfg, box, source, args.base_seed)
        sources[source] = f
        provenance[source] = str(getattr(f, "filename", "") or source)
        print(f"    {source:5s} field {tuple(f.shape)} {f.dtype} "
              f"<- {provenance[source]}")
    print(f"    spec: disp <- {spec['disp']},  vel <- {spec['vel']}")

    t0 = time.time()
    field = assemble_field(sources, spec)
    del sources
    d = cfg.get("data", {})
    particles = field_to_particles(
        field,
        boxsize_kpc_h=float(d.get("boxsize_mpc_h", 100.0)) * 1000.0,
        redshift=float(d.get("redshift", 0.0)),
    )
    del field
    print(f"    assembled + converted in {time.time() - t0:.1f}s: "
          f"{particles.ids.size} particles, "
          f"m_p {particles.particle_mass_msun_h:.3e} Msun/h", flush=True)

    t0 = time.time()
    cat = run_rockstar_on_particles(
        particles, work, cfg=FROZEN_CFG, tag=arm, overwrite=not args.reuse,
    )
    del particles
    halo_min = (time.time() - t0) / 60.0
    counts = catalog_counts(cat)
    n_lattice = int(args.occupancy_norm)
    report = {
        "box": box, "arm": arm, "spec": spec, "sources": provenance,
        "rockstar_cfg": str(FROZEN_CFG), "catalog": str(cat.path),
        "halo_finder_min": halo_min,
        "occupancy": counts["n_bound_particles"] / float(n_lattice),
        **counts,
    }
    write_json(out_json, report)
    print(f"    Rockstar {halo_min:.1f} min -> {counts['n_objects']} objects "
          f"({counts['n_hosts']} hosts, {counts['n_subhalos']} subhalos), "
          f"occupancy {report['occupancy']:.4f}")
    print(f"    wrote {out_json}")
    # The GADGET2 snapshot is ~3 GB and is only an input to the halo finder that
    # just consumed it. Keeping one per arm would cost more than the catalogs.
    snap = work / f"{arm}.gadget2"
    if snap.is_file() and not args.keep_snapshot:
        size_gb = snap.stat().st_size / 2 ** 30
        snap.unlink()
        print(f"    deleted the {size_gb:.1f} GB snapshot ({snap.name}); "
              f"pass --keep-snapshot to retain it")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--box", default="set8")
    ap.add_argument("--arm", default="srpos_hrvel", choices=sorted(ARMS),
                    help="which channel-source combination to build")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", default=True,
                    help="keep an existing catalog for this arm (default)")
    ap.add_argument("--overwrite", dest="reuse", action="store_false")
    ap.add_argument("--keep-snapshot", action="store_true",
                    help="do not delete the ~3 GB GADGET2 snapshot")
    ap.add_argument("--occupancy-norm", type=int, default=512 ** 3,
                    help="particle count the occupancy is divided by "
                         "(default 512^3, the HR lattice)")
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    run_arm(cfg, str(args.box), str(args.arm), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
