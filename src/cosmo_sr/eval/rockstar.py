"""Run Rockstar on GADGET2 dumps and parse ASCII halo catalogs.

The binary path and config are frozen in ``configs/sr2_baseline/``. Halo finding
is always on a complete periodic box — never on an isolated crop.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .particles import ParticleBox, field_to_particles, write_gadget2_snapshot

__all__ = [
    "HaloCatalog",
    "default_rockstar_binary",
    "default_rockstar_cfg",
    "load_rockstar_ascii",
    "run_rockstar_on_field",
    "run_rockstar_on_particles",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_rockstar_binary() -> Path:
    return _project_root() / "external" / "rockstar" / "rockstar"


def default_rockstar_cfg() -> Path:
    return _project_root() / "configs" / "sr2_baseline" / "rockstar.cfg"


@dataclass
class HaloCatalog:
    """Rockstar ASCII catalog (one row per halo/subhalo)."""

    ids: np.ndarray
    parent_ids: np.ndarray
    mvir: np.ndarray
    rvir: np.ndarray          # kpc/h (Rockstar convention in .list files)
    vmax: np.ndarray
    pos: np.ndarray          # (N,3) Mpc/h
    vel: np.ndarray          # (N,3) km/s
    num_p: np.ndarray
    path: str = ""

    @property
    def n(self) -> int:
        return int(self.ids.shape[0])

    def hosts(self) -> "HaloCatalog":
        m = self.parent_ids < 0
        return self._mask(m)

    def subhalos(self) -> "HaloCatalog":
        m = self.parent_ids >= 0
        return self._mask(m)

    def _mask(self, m: np.ndarray) -> "HaloCatalog":
        return HaloCatalog(
            ids=self.ids[m], parent_ids=self.parent_ids[m], mvir=self.mvir[m],
            rvir=self.rvir[m], vmax=self.vmax[m], pos=self.pos[m],
            vel=self.vel[m], num_p=self.num_p[m], path=self.path,
        )


def load_rockstar_ascii(path: str | os.PathLike) -> HaloCatalog:
    """Parse a Rockstar ``halos_*.list`` / ``out_*.list`` ASCII catalog.

    Column layout (Rockstar 0.99 ASCII header)::

        id num_p mvir mbound_vir rvir vmax rvmax vrms x y z vx vy vz ...
        ... idx i_so i_ph num_cp mmetric

    ``i_so`` is the *internal* parent index (``sub_of``); ``-1`` marks hosts.
    We remap it to the printed halo ``id`` via the ``idx`` column.
    """
    path = Path(path)
    colmap: Dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            toks = re.split(r"\s+", line.lstrip("#").strip().lower())
            if toks and toks[0] == "id" and "id" not in colmap:
                for i, t in enumerate(toks):
                    colmap[t.strip("()")] = i

    # Defaults match the fprintf in io/meta_io.c::output_halos (ASCII).
    defaults = {
        "id": 0, "num_p": 1, "mvir": 2, "rvir": 4, "vmax": 5,
        "x": 8, "y": 9, "z": 10, "vx": 11, "vy": 12, "vz": 13,
        "idx": -5, "i_so": -4,
    }

    # Fast path: numpy load of all numeric columns.
    try:
        data = np.loadtxt(path, comments="#")
    except Exception:
        data = np.zeros((0, 20))
    if data.size == 0:
        z = np.zeros(0)
        return HaloCatalog(z.astype(np.int64), z.astype(np.int64), z, z, z,
                           np.zeros((0, 3)), np.zeros((0, 3)), z.astype(np.int64),
                           str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ncols = data.shape[1]

    def _idx(name: str) -> int:
        if name in colmap:
            return colmap[name]
        d = defaults[name]
        return d if d >= 0 else ncols + d

    def icol(name: str) -> np.ndarray:
        return np.asarray(data[:, _idx(name)], dtype=np.int64)

    def fcol(name: str) -> np.ndarray:
        return np.asarray(data[:, _idx(name)], dtype=np.float64)

    ids = icol("id")
    idx = icol("idx")
    i_so = icol("i_so")
    idx_to_id = {int(i): int(h) for i, h in zip(idx, ids)}
    parent = np.asarray(
        [idx_to_id.get(int(s), -1) if int(s) >= 0 else -1 for s in i_so],
        dtype=np.int64,
    )
    pos = np.stack([fcol("x"), fcol("y"), fcol("z")], axis=1)
    vel = np.stack([fcol("vx"), fcol("vy"), fcol("vz")], axis=1)
    return HaloCatalog(
        ids=ids, parent_ids=parent, mvir=fcol("mvir"), rvir=fcol("rvir"),
        vmax=fcol("vmax"), pos=pos, vel=vel, num_p=icol("num_p"), path=str(path),
    )


def run_rockstar_on_particles(
    particles: ParticleBox,
    out_dir: str | os.PathLike,
    *,
    binary: Optional[str | os.PathLike] = None,
    cfg: Optional[str | os.PathLike] = None,
    tag: str = "box",
    overwrite: bool = False,
) -> HaloCatalog:
    """Write GADGET2, run Rockstar, return the ASCII catalog."""
    # Absolute paths are required: Rockstar is launched with cwd=catalog dir, so
    # a relative snap/OUTBASE would be resolved from the wrong place.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = (out_dir / f"{tag}.gadget2").resolve()
    catalog_glob_dir = (out_dir / f"{tag}_rockstar").resolve()
    def _find_catalogs(d: Path):
        # Rockstar 0.99 writes halos_<scale>.ascii (and optionally .list).
        pats = ("halos*.ascii", "halos*.list", "out_*.list", "out_*.ascii")
        found = []
        for pat in pats:
            found.extend(d.glob(pat))
        return sorted(set(found))

    if catalog_glob_dir.exists() and not overwrite:
        existing = _find_catalogs(catalog_glob_dir)
        if existing:
            return load_rockstar_ascii(existing[0])

    if catalog_glob_dir.exists() and overwrite:
        shutil.rmtree(catalog_glob_dir)
    catalog_glob_dir.mkdir(parents=True, exist_ok=True)

    write_gadget2_snapshot(str(snap), particles)
    binary = Path(binary or default_rockstar_binary()).resolve()
    cfg = Path(cfg or default_rockstar_cfg()).resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"Rockstar binary missing: {binary}")
    if not cfg.is_file():
        raise FileNotFoundError(f"Rockstar config missing: {cfg}")

    # Rockstar writes into the cwd; pin OUTBASE via a per-run cfg copy.
    run_cfg = catalog_glob_dir / "rockstar.cfg"
    text = cfg.read_text()
    outbase = str(catalog_glob_dir)
    if re.search(r"^\s*OUTBASE\s*=", text, flags=re.M):
        text = re.sub(r"^\s*OUTBASE\s*=.*$", f'OUTBASE = "{outbase}"',
                      text, flags=re.M)
    else:
        text += f'\nOUTBASE = "{outbase}"\n'
    run_cfg.write_text(text)

    log = catalog_glob_dir / "rockstar.log"
    with open(log, "w") as fh:
        proc = subprocess.run(
            [str(binary), "-c", str(run_cfg), str(snap)],
            cwd=str(catalog_glob_dir),
            stdout=fh, stderr=subprocess.STDOUT, check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Rockstar failed (rc={proc.returncode}); see {log}"
        )
    lists = _find_catalogs(catalog_glob_dir)
    if not lists:
        raise FileNotFoundError(f"no Rockstar ASCII catalog in {catalog_glob_dir}")
    return load_rockstar_ascii(lists[0])


def run_rockstar_on_field(
    field,
    out_dir: str | os.PathLike,
    *,
    tag: str = "box",
    boxsize_kpc_h: float = 100000.0,
    redshift: float = 0.0,
    **kwargs,
) -> HaloCatalog:
    particles = field_to_particles(
        field, boxsize_kpc_h=boxsize_kpc_h, redshift=redshift,
    )
    return run_rockstar_on_particles(particles, out_dir, tag=tag, **kwargs)
