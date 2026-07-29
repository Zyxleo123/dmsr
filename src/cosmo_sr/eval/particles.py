"""Convert catnorm displacement+velocity fields to Eulerian particles.

Canonical field layout is ``(6, Ng, Ng, Ng)`` with channels ``disp[0:3]`` and
``vel[3:6]`` in the SRS *normalized* units. Positions are reconstructed as

    x = (q + Psi) mod L

on the Lagrangian lattice ``q``, with ``Psi`` and ``v`` undoing
:func:`cosmo_sr.data.preprocess_srs.disnorm` /
:func:`cosmo_sr.data.preprocess_srs.velnorm`.

Particle IDs are the flat Lagrangian grid indices ``0 .. Ng^3-1``. HR and SR
fields that share the same lattice therefore share a particle-ID space, which
is what Stage-1 subhalo matching uses for overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..data.preprocess_srs import disnorm, velnorm

__all__ = [
    "ParticleBox",
    "field_to_particles",
    "particle_mass_msun_h",
    "write_gadget2_snapshot",
]


@dataclass
class ParticleBox:
    """Eulerian particles for one periodic box."""

    pos_mpc_h: np.ndarray   # (N, 3) float32, comoving Mpc/h
    vel_kms: np.ndarray     # (N, 3) float32, peculiar km/s
    ids: np.ndarray         # (N,) int64, Lagrangian flat index
    boxsize_mpc_h: float
    redshift: float
    particle_mass_msun_h: float


def particle_mass_msun_h(
    omega_m: float,
    boxsize_mpc_h: float,
    n_part: int,
    rho_crit: float = 2.77536627e11,
) -> float:
    """Equal-mass tracer mass in ``Msun/h`` for a cubic box."""
    return float(omega_m * rho_crit * (boxsize_mpc_h ** 3) / int(n_part))


def field_to_particles(
    field,
    *,
    boxsize_kpc_h: float = 100000.0,
    redshift: float = 0.0,
    omega_m_norm: float = 0.31,
    omega_m_sim: float = 0.2814,
) -> ParticleBox:
    """Undo catnorm and build positions/velocities/IDs.

    ``omega_m_norm`` must match the Om used when the field was normalized
    (SRS default 0.31). ``omega_m_sim`` is the simulation cosmology used only
    for the reported particle mass.
    """
    x = np.asarray(field, dtype=np.float32)
    if x.ndim != 4 or x.shape[0] < 6 or not (x.shape[1] == x.shape[2] == x.shape[3]):
        raise ValueError(f"expected (6, Ng, Ng, Ng), got {x.shape}")
    ng = int(x.shape[1])
    n_part = ng ** 3
    box_kpc = float(boxsize_kpc_h)
    cell = box_kpc / ng

    disp = disnorm(x[0:3].astype(np.float64, copy=True), z=redshift, undo=True)
    vel = velnorm(x[3:6].astype(np.float64, copy=True), z=redshift, undo=True)
    # growth helpers take Om via default; disnorm/velnorm close over Om=0.31.
    # At z=0 D=1 so disp undo is exact (×6000) regardless of Om; vel uses Om.
    _ = omega_m_norm  # documented; helpers already default to 0.31

    q = (np.arange(ng, dtype=np.float64) + 0.5) * cell
    # Broadcast lattice: (Ng,Ng,Ng,3)
    qx = q.reshape(ng, 1, 1)
    qy = q.reshape(1, ng, 1)
    qz = q.reshape(1, 1, ng)
    pos = np.empty((3, ng, ng, ng), dtype=np.float64)
    pos[0] = (qx + disp[0]) % box_kpc
    pos[1] = (qy + disp[1]) % box_kpc
    pos[2] = (qz + disp[2]) % box_kpc

    pos_mpc = np.ascontiguousarray(
        pos.reshape(3, -1).T * 1e-3, dtype=np.float32
    )  # kpc/h -> Mpc/h
    vel_kms = np.ascontiguousarray(vel.reshape(3, -1).T, dtype=np.float32)
    ids = np.arange(n_part, dtype=np.int64)
    box_mpc = box_kpc * 1e-3
    return ParticleBox(
        pos_mpc_h=pos_mpc,
        vel_kms=vel_kms,
        ids=ids,
        boxsize_mpc_h=box_mpc,
        redshift=float(redshift),
        particle_mass_msun_h=particle_mass_msun_h(omega_m_sim, box_mpc, n_part),
    )


def write_gadget2_snapshot(
    path: str,
    particles: ParticleBox,
    *,
    mass_unit_1e10_msun_h: float = 1e10,
) -> str:
    """Write a single-file GADGET-2 binary snapshot (DM only, type 1).

    Lengths in the file are **Mpc/h** (so Rockstar ``GADGET_LENGTH_CONVERSION=1``).
    Masses are stored in units of ``1e10 Msun/h`` (Rockstar
    ``GADGET_MASS_CONVERSION=1e10``). Velocities are peculiar km/s.
    """
    import struct

    n = int(particles.ids.shape[0])
    mass = particles.particle_mass_msun_h / float(mass_unit_1e10_msun_h)
    a = 1.0 / (1.0 + particles.redshift)

    # GADGET header: 256 bytes after the block size int
    npart = [0, n, 0, 0, 0, 0]
    massarr = [0.0, mass, 0.0, 0.0, 0.0, 0.0]
    npart_total = list(npart)
    header = bytearray(256)
    struct.pack_into("6i", header, 0, *npart)
    struct.pack_into("6d", header, 24, *massarr)
    struct.pack_into("d", header, 72, a)                          # time
    struct.pack_into("d", header, 80, particles.redshift)         # redshift
    struct.pack_into("i", header, 88, 0)                          # flag_sfr
    struct.pack_into("i", header, 92, 0)                          # flag_feedback
    struct.pack_into("6i", header, 96, *npart_total)
    struct.pack_into("i", header, 120, 0)                         # flag_cooling
    struct.pack_into("i", header, 124, 1)                         # num_files
    struct.pack_into("d", header, 128, particles.boxsize_mpc_h)  # BoxSize
    struct.pack_into("d", header, 136, 0.2814)                    # Omega0
    struct.pack_into("d", header, 144, 0.7186)                    # OmegaLambda
    struct.pack_into("d", header, 152, 0.697)                     # HubbleParam

    def _block(payload: bytes) -> bytes:
        nbyte = len(payload)
        return struct.pack("i", nbyte) + payload + struct.pack("i", nbyte)

    pos = np.ascontiguousarray(particles.pos_mpc_h, dtype=np.float32)
    vel = np.ascontiguousarray(particles.vel_kms, dtype=np.float32)
    ids = np.ascontiguousarray(particles.ids, dtype=np.int32)  # GADGET classic IDs

    with open(path, "wb") as fh:
        fh.write(_block(bytes(header)))
        fh.write(_block(pos.tobytes(order="C")))
        fh.write(_block(vel.tobytes(order="C")))
        fh.write(_block(ids.tobytes(order="C")))
    return path
