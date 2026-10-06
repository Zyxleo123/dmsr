#!/usr/bin/env python
"""Read a GADGET-4 ``fof_subhalo_tab`` and write it as a plain npz + json.

This is the only step that needs HDF5, so it is the only step that runs in the
GADGET-4 toolchain environment: it depends on ``h5py`` and ``numpy`` and nothing
else -- no ``cosmo_sr`` import, no reward config. Everything downstream reads
the npz from the project's own environment.

What comes out, and in whose convention
---------------------------------------
GADGET-4 stores subhalos grouped by their FOF host and ordered by decreasing
size inside it, with ``SubhaloRankInGr == 0`` marking the **central** -- the
object that carries the bulk of the FOF group -- and every higher rank a
satellite. Rockstar instead marks hosts with ``parent_id < 0``. The npz carries
both the raw SUBFIND columns and a ``is_satellite`` flag built from the rank, so
a comparison never has to re-derive that mapping and get it subtly wrong.

Masses are converted out of GADGET's ``1e10 Msun/h`` into ``Msun/h``, which is
what the Rockstar catalogs in this project are already in. Lengths stay Mpc/h in
both. Nothing else is transformed: ``SubhaloLen`` is the bound particle count
SUBFIND decided on, and it is the number the comparison actually leans on,
because a particle count is free of every mass-definition difference between the
two finders.

    python scripts/features/subfind_catalog.py \\
        --group-file $SUBFIND_ROOT/set9__hr/fof_subhalo_tab_000.hdf5 \\
        --out $SUBFIND_ROOT/set9__hr/subfind_catalog
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

#: GADGET's internal mass unit, in Msun/h.
MASS_UNIT_MSUN_H = 1e10

#: Datasets we keep, and whether they are a mass (so need the unit conversion).
SUBHALO_FIELDS = {
    "SubhaloLen": False, "SubhaloMass": True, "SubhaloPos": False,
    "SubhaloVel": False, "SubhaloVelDisp": False, "SubhaloVmax": False,
    "SubhaloVmaxRad": False, "SubhaloHalfmassRad": False,
    "SubhaloGroupNr": False, "SubhaloRankInGr": False,
    "SubhaloIDMostbound": False,
}
GROUP_FIELDS = {
    "GroupLen": False, "GroupMass": True, "GroupPos": False,
    "GroupNsubs": False, "GroupFirstSub": False,
}


def _jsonable(v):
    """HDF5 attribute -> something json.dumps will accept.

    GADGET-4 writes provenance strings (Git_commit, Git_date, the compile-time
    flags) as HDF5 fixed-length strings, which h5py hands back as **bytes**, and
    json refuses them. They are worth keeping rather than dropping: the header is
    the only record of which build produced a catalogue.
    """
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _files(path: str) -> List[Path]:
    """One catalogue file, or every sub-file of a split catalogue, in order."""
    p = Path(path)
    if p.is_file():
        return [p]
    # A split catalogue lives in groups_XXX/fof_subhalo_tab_XXX.N.hdf5
    hits = sorted(glob.glob(str(p) + ".*.hdf5")) or sorted(glob.glob(str(p) + ".*"))
    if not hits:
        raise SystemExit(f"no group catalogue at {p} (and no .N sub-files)")
    return [Path(h) for h in sorted(hits, key=lambda s: int(
        Path(s).name.split(".")[-2] if s.endswith(".hdf5") else
        Path(s).name.split(".")[-1]))]


def read_group_catalogue(path: str) -> Dict[str, np.ndarray]:
    """Concatenate the Subhalo and Group tables of a (possibly split) catalogue."""
    import h5py

    files = _files(path)
    subs: Dict[str, List[np.ndarray]] = {}
    grps: Dict[str, List[np.ndarray]] = {}
    header: Dict[str, float] = {}
    for f in files:
        with h5py.File(f, "r") as h:
            if not header:
                header = {k: _jsonable(v) for k, v in h["Header"].attrs.items()}
            for name in SUBHALO_FIELDS:
                if "Subhalo" in h and name in h["Subhalo"]:
                    subs.setdefault(name, []).append(np.asarray(h["Subhalo"][name]))
            for name in GROUP_FIELDS:
                if "Group" in h and name in h["Group"]:
                    grps.setdefault(name, []).append(np.asarray(h["Group"][name]))

    out: Dict[str, np.ndarray] = {}
    for name, parts in list(subs.items()) + list(grps.items()):
        arr = np.concatenate(parts, axis=0) if parts else np.zeros(0)
        is_mass = SUBHALO_FIELDS.get(name, GROUP_FIELDS.get(name, False))
        out[name] = (arr.astype(np.float64) * MASS_UNIT_MSUN_H) if is_mass else arr
    out["_header"] = np.asarray(json.dumps(header))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group-file", required=True,
                    help="fof_subhalo_tab_000.hdf5, or its stem for a split catalogue")
    ap.add_argument("--out", required=True, help="output stem (.npz + .json)")
    ap.add_argument("--min-particles", type=int, default=50,
                    help="only for the printed summary; nothing is dropped")
    args = ap.parse_args(argv)

    cat = read_group_catalogue(args.group_file)
    header = json.loads(str(cat.pop("_header")))

    n_sub = int(cat["SubhaloLen"].shape[0]) if "SubhaloLen" in cat else 0
    if n_sub == 0:
        print("!!! the catalogue has no subhalos at all -- check the run log",
              flush=True)
    rank = cat.get("SubhaloRankInGr", np.zeros(n_sub, dtype=np.int64))
    cat["is_satellite"] = (np.asarray(rank) > 0)

    lens = np.asarray(cat.get("SubhaloLen", np.zeros(0)))
    sat = cat["is_satellite"]
    keep = lens >= int(args.min_particles)
    summary = {
        "group_file": str(args.group_file),
        "header": header,
        "n_groups": int(cat["GroupLen"].shape[0]) if "GroupLen" in cat else 0,
        "n_subhalo_rows": n_sub,
        "n_centrals": int((~sat).sum()),
        "n_satellites": int(sat.sum()),
        "n_satellites_ge_min_p": int((sat & keep).sum()),
        "min_particles": int(args.min_particles),
        "largest_len": int(lens.max()) if lens.size else 0,
    }

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(stem) + ".npz", **{k: v for k, v in cat.items()})
    (Path(str(stem) + ".json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"=== wrote {stem}.npz and {stem}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
