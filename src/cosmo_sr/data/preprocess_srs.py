"""SRS-compatible preprocessing wrapped in clean, testable functions.

This reuses the logic of ``SRS-map2map/preproc.py`` (particle snapshot ->
normalized displacement+velocity field on the Lagrangian grid) but exposes it as
pure functions instead of a script.

Convention (from SRS): particles are ordered by their particle ID, which defines
the original Lagrangian grid ordering. Sort by ID before reshaping to
``(Ng, Ng, Ng, 3)``.

The cosmological normalisation (:func:`disnorm`, :func:`velnorm`) uses the exact
same formulae as ``SRS-map2map/map2map/norms/cosmology.py`` so outputs match the
external pipeline numerically.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .field_io import DEFAULT_DTYPE

# --------------------------------------------------------------------------- #
# Cosmology normalisation (mirrors SRS-map2map/map2map/norms/cosmology.py)
# --------------------------------------------------------------------------- #

def growth_D(z: float, Om: float = 0.31) -> float:
    """Linear growth function for flat LambdaCDM, normalized to 1 at z=0."""
    from scipy.special import hyp2f1

    OL = 1 - Om
    a = 1 / (1 + z)
    return (
        a
        * hyp2f1(1, 1 / 3, 11 / 6, -OL * a ** 3 / Om)
        / hyp2f1(1, 1 / 3, 11 / 6, -OL / Om)
    )


def growth_f(z: float, Om: float = 0.31) -> float:
    """Linear growth rate for flat LambdaCDM."""
    from scipy.special import hyp2f1

    OL = 1 - Om
    a = 1 / (1 + z)
    aa3 = OL * a ** 3 / Om
    return 1 - 6 / 11 * aa3 * hyp2f1(2, 4 / 3, 17 / 6, -aa3) / hyp2f1(
        1, 1 / 3, 11 / 6, -aa3
    )


def hubble_H(z: float, Om: float = 0.31) -> float:
    """Hubble parameter [h km/s/Mpc] for flat LambdaCDM."""
    OL = 1 - Om
    a = 1 / (1 + z)
    return 100 * np.sqrt(Om / a ** 3 + OL)


def disnorm(x: np.ndarray, z: float = 0.0, undo: bool = False) -> np.ndarray:
    """Normalize (or undo) displacement field by ``6000 * D(z)``."""
    dis_norm = 6000 * growth_D(z)
    if not undo:
        dis_norm = 1 / dis_norm
    return x * dis_norm


def velnorm(x: np.ndarray, z: float = 0.0, undo: bool = False) -> np.ndarray:
    """Normalize (or undo) velocity field by ``6 D(z) H(z) f(z) / (1+z)``."""
    vel_norm = 6 * growth_D(z) * hubble_H(z) * growth_f(z) / (1 + z)
    if not undo:
        vel_norm = 1 / vel_norm
    return x * vel_norm


# --------------------------------------------------------------------------- #
# Lagrangian lattice <-> displacement
# --------------------------------------------------------------------------- #

def _lattice(boxsize: float, Ng: int) -> np.ndarray:
    cellsize = boxsize / Ng
    return np.arange(Ng) * cellsize + 0.5 * cellsize


def pos_to_displacement(pos: np.ndarray, boxsize: float, Ng: int) -> np.ndarray:
    """Convert positions on the Lagrangian grid to periodic displacements.

    ``pos`` must be shaped ``(Ng, Ng, Ng, 3)`` and ordered by particle ID so it
    aligns with the Lagrangian lattice. Displacements are wrapped to the minimal
    periodic image (assumes displacement magnitude < half box).
    """
    pos = np.asarray(pos)
    if pos.shape != (Ng, Ng, Ng, 3):
        raise ValueError(
            f"pos must have shape (Ng, Ng, Ng, 3)=({Ng},{Ng},{Ng},3), got {pos.shape}"
        )
    lattice = _lattice(boxsize, Ng)
    disp = pos.astype(np.float64, copy=True)
    disp[..., 0] -= lattice.reshape(-1, 1, 1)
    disp[..., 1] -= lattice.reshape(-1, 1)
    disp[..., 2] -= lattice
    disp -= np.rint(disp / boxsize) * boxsize
    return disp


def displacement_to_pos(displacement: np.ndarray, boxsize: float, Ng: int) -> np.ndarray:
    """Inverse of :func:`pos_to_displacement`; returns positions in ``[0, boxsize)``."""
    displacement = np.asarray(displacement)
    if displacement.shape != (Ng, Ng, Ng, 3):
        raise ValueError(
            f"displacement must have shape (Ng, Ng, Ng, 3)=({Ng},{Ng},{Ng},3), "
            f"got {displacement.shape}"
        )
    lattice = _lattice(boxsize, Ng)
    pos = displacement.astype(np.float64, copy=True)
    pos[..., 0] += lattice.reshape(-1, 1, 1)
    pos[..., 1] += lattice.reshape(-1, 1)
    pos[..., 2] += lattice
    pos = np.mod(pos, boxsize)
    return pos


def _infer_Ng(npart: int) -> int:
    Ng = int(np.rint(npart ** (1 / 3)))
    if Ng ** 3 != npart:
        raise ValueError(
            f"Number of particles {npart} is not a perfect cube; cannot infer Ng."
        )
    return Ng


def particles_to_field(
    position: np.ndarray,
    velocity: np.ndarray,
    particle_id: np.ndarray,
    boxsize: float,
    redshift: float,
    normalize: bool = True,
) -> np.ndarray:
    """Build a canonical ``(6, Ng, Ng, Ng)`` displacement+velocity field.

    Particles are sorted by ``particle_id`` (the Lagrangian ordering) before
    reshaping, so shuffled inputs produce identical output.
    """
    position = np.asarray(position)
    velocity = np.asarray(velocity)
    particle_id = np.asarray(particle_id).ravel()

    npart = position.shape[0]
    if position.shape != (npart, 3) or velocity.shape != (npart, 3):
        raise ValueError(
            f"position/velocity must be (N, 3); got {position.shape}, {velocity.shape}"
        )
    if particle_id.shape[0] != npart:
        raise ValueError("particle_id length must match number of particles.")

    Ng = _infer_Ng(npart)

    order = np.argsort(particle_id, kind="stable")
    if not np.array_equal(np.unique(particle_id), particle_id[order]):
        # IDs are not a contiguous permutation; still sort but warn via error if
        # there are duplicates (which would corrupt the grid mapping).
        if len(np.unique(particle_id)) != npart:
            raise ValueError("particle_id contains duplicates; cannot map to grid.")

    pos = position[order].reshape(Ng, Ng, Ng, 3)
    vel = velocity[order].reshape(Ng, Ng, Ng, 3)

    disp = pos_to_displacement(pos, boxsize, Ng)

    disp = np.moveaxis(disp, -1, 0).astype(np.float64)
    vel = np.moveaxis(vel, -1, 0).astype(np.float64)

    if normalize:
        disp = disnorm(disp, z=redshift)
        vel = velnorm(vel, z=redshift)

    field = np.concatenate([disp, vel], axis=0).astype(DEFAULT_DTYPE)
    return field


def disp_vel_to_field(
    disp_path: str | os.PathLike,
    vel_path: str | os.PathLike,
    out_path: Optional[str | os.PathLike] = None,
    redshift: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Build a canonical ``(6,Ng,Ng,Ng)`` field from raw ``disp.npy``+``vel.npy``.

    This matches ``SRS-map2map/make_catnorm.py`` (cosmological normalisation +
    channel concat) but uses our own re-implemented norms so it does not depend
    on which ``map2map`` fork is importable. Inputs are ``(3,Ng,Ng,Ng)`` arrays
    in MP-Gadget kpc/h units.
    """
    # Memory-lean path: HR (Ng=512) disp/vel are ~3 GB each. Avoid float64
    # upcasts and concatenation (which would peak at tens of GB). Instead stream
    # each block directly into a single float32 output and scale in place.
    disp = np.load(os.fspath(disp_path), mmap_mode="r")
    vel = np.load(os.fspath(vel_path), mmap_mode="r")
    if disp.shape != vel.shape or disp.ndim != 4 or disp.shape[0] != 3:
        raise ValueError(
            f"disp/vel must be (3,Ng,Ng,Ng) and match; got {disp.shape}, {vel.shape}"
        )
    Ng = disp.shape[1]
    field = np.empty((6, Ng, Ng, Ng), dtype=DEFAULT_DTYPE)
    field[0:3] = disp  # copy from (mmapped) disk into float32
    field[3:6] = vel
    del disp, vel
    if normalize:
        # scalar normalisations (same as disnorm/velnorm with undo=False)
        dis_scale = np.float32(1.0 / (6000 * growth_D(redshift)))
        vel_scale = np.float32(
            1.0 / (6 * growth_D(redshift) * hubble_H(redshift) * growth_f(redshift) / (1 + redshift))
        )
        field[0:3] *= dis_scale
        field[3:6] *= vel_scale
    if out_path is not None:
        out_path = os.fspath(out_path)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.save(out_path, field)
    return field


def snapshot_to_field_bigfile(snapshot_path: str | os.PathLike, out_path: str | os.PathLike) -> np.ndarray:
    """Read an MP-Gadget bigfile snapshot and write a canonical ``(6,Ng,Ng,Ng)`` field.

    Mirrors ``SRS-map2map/preproc.py::get_nonlin_fields``. Requires ``bigfile``.
    Returns the field that was written.
    """
    from bigfile import File

    bigf = File(os.fspath(snapshot_path))
    header = bigf.open("Header")
    boxsize = float(header.attrs["BoxSize"][0])
    redshift = 1.0 / header.attrs["Time"][0] - 1.0

    pid = bigf.open("1/ID")[:]
    pos = bigf.open("1/Position")[:]
    vel = bigf.open("1/Velocity")[:]

    field = particles_to_field(
        pos, vel, pid, boxsize=boxsize, redshift=float(redshift), normalize=True
    )

    out_path = os.fspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.save(out_path, field)
    return field
