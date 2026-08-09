"""Eulerian density diagnostics: does the degraded field make the right universe?

Field-space MSE and displacement power spectra both completely hide the failure
that actually matters. Measured on set0 (see also the module docstring of
``models/stochastic_degrader.py``):

    density from true LR displacement :  sigma_delta = 4.67
    density from A(disp_HR)           :  sigma_delta = 5.26   (+13%)
    P_A / P_true                      :  1.03 at large scales -> 1.61 at Nyquist

i.e. plain block-averaging makes the universe **too clumpy**. Block-averaging HR
displacements returns the exact centre of mass of each Lagrangian patch, which
lands tightly in the halo centre; the real LR run -- heavier particles, coarser
softening -- carries extra small-scale scatter that puffs haloes out and lowers
the contrast. That scatter is exactly the "negligible" 0.8% of displacement
variance that ``A`` fails to reproduce, and it is 98% high-k and concentrated in
collapsed regions.

An MSE-trained deterministic degrader cannot fix this: regression to the
conditional mean *smooths further*, making the over-clumping worse. Only a
sampler that puts the scatter back can get sigma_delta and P(k) right -- which is
why this module exists and why it is the headline eval metric.

Positions are reconstructed as ``q + Psi`` on the Lagrangian lattice and CIC-
deposited. Within a crop the deposit wraps periodically, which is an
approximation (particles that leave the crop reappear on the far side), but it is
applied identically to the prediction, the truth and the ``A`` baseline, so the
*ratios* below are fair.
"""
from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch

from ..losses.flow import _kbin_index


def cic_density(disp: torch.Tensor, cellsize: float, dis_norm: float,
                grid_mult: int = 1) -> torch.Tensor:
    """CIC-deposit particles at ``q + Psi`` onto the Lagrangian grid.

    Parameters
    ----------
    disp:
        ``(B, 3, N, N, N)`` *normalized* displacement (the on-disk convention:
        ``disnorm`` divides by ``6000 * D(z)`` kpc/h).
    cellsize:
        Lagrangian cell size in kpc/h (``boxsize / lr_grid_res``).
    dis_norm:
        kpc/h per normalized displacement unit (``6000 * D(z)``).
    grid_mult:
        Deposit the same ``N**3`` particles onto a ``(grid_mult*N)**3`` grid, i.e.
        a ``grid_mult``x-**finer** density mesh (default 1 = the particle grid).
        The finer mesh resolves the sub-cell density (caustics/haloes) that CIC at
        the particle resolution smooths over -- this is how SR2 feeds its
        discriminator an inverse-pixel-shuffle density channel. Positions simply
        scale by ``grid_mult`` onto the finer lattice; the mean-normalisation makes
        the returned overdensity grid-independent.

    Returns
    -------
    ``(B, 1, grid_mult*N, grid_mult*N, grid_mult*N)`` overdensity ``delta = rho/rho_bar - 1``.
    """
    if disp.dim() != 5 or disp.shape[1] != 3:
        raise ValueError(f"cic_density expects (B, 3, N, N, N), got {tuple(disp.shape)}")
    b, _, n, _, _ = disp.shape
    m = int(grid_mult)
    ng = n * m
    dev, dt = disp.device, torch.float32

    lat = torch.arange(n, device=dev, dtype=dt) + 0.5
    q = torch.stack(torch.meshgrid(lat, lat, lat, indexing="ij"))       # (3, N, N, N)
    # coarse positions (particle grid units), then scale onto the finer mesh
    g = disp.float() * (dis_norm / cellsize) + q.unsqueeze(0)           # (B, 3, N, N, N)
    g = (g * m) % ng

    i0 = torch.floor(g).long()
    frac = g - i0.float()

    dens = torch.zeros(b, ng ** 3, device=dev, dtype=dt)
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                wx = frac[:, 0] if ox else (1.0 - frac[:, 0])
                wy = frac[:, 1] if oy else (1.0 - frac[:, 1])
                wz = frac[:, 2] if oz else (1.0 - frac[:, 2])
                w = (wx * wy * wz).reshape(b, -1)
                ix = (i0[:, 0] + ox) % ng
                iy = (i0[:, 1] + oy) % ng
                iz = (i0[:, 2] + oz) % ng
                idx = ((ix * ng + iy) * ng + iz).reshape(b, -1)
                dens.scatter_add_(1, idx, w)

    dens = dens.view(b, 1, ng, ng, ng)
    mean = dens.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-12)
    return dens / mean - 1.0


def _valid_center_corners(
    disp: torch.Tensor,
    cellsize: float,
    dis_norm: float,
    region: int,
    grid_mult: int = 1,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """The eight CIC corners of the valid-centre deposit, one at a time.

    Yields ``(idx, weight)``, both ``(B, N^3)``, where ``idx`` is the flat
    destination cell and ``weight`` the trilinear share (zero for particles that
    left the scored cube). A generator rather than a list because the caller
    accumulates them one corner at a time: materialising all eight would hold
    eight copies of the crop's particle arrays at once for no benefit.

    Shared by :func:`cic_density_valid_center` and
    :func:`cic_deposit_valid_center` so a mass grid and a momentum grid built
    from the same displacement are guaranteed to be the *same* geometry --
    identical origin, identical bulk offset, identical discards. Anything less
    and a per-cell ratio like ``momentum / mass`` would be dividing two grids
    that do not describe the same cells.
    """
    if disp.dim() != 5 or disp.shape[1] != 3:
        raise ValueError(f"expects (B, 3, N, N, N), got {tuple(disp.shape)}")
    b, _, n, _, _ = disp.shape
    R = int(region)
    if R > n:
        raise ValueError(f"region {R} exceeds crop {n}")
    m = int(grid_mult)
    ng = R * m
    dev, dt = disp.device, torch.float32

    lat = torch.arange(n, device=dev, dtype=dt) + 0.5
    q = torch.stack(torch.meshgrid(lat, lat, lat, indexing="ij"))
    g = disp.float() * (dis_norm / cellsize) + q.unsqueeze(0)          # (B,3,N,N,N)

    # The crop's bulk displacement is a rigid translation: it moves where the
    # matter is without changing its clustering, so the scored cube follows it.
    # Rounded to whole cells (and detached) so it introduces no sub-cell offset
    # and no gradient path of its own.
    bulk = g.detach().mean(dim=(2, 3, 4)) - q.mean(dim=(1, 2, 3)).unsqueeze(0)
    bulk = torch.round(bulk)                                            # (B,3)
    origin = (n - R) / 2.0 + bulk                                       # (B,3)
    u = (g - origin.view(b, 3, 1, 1, 1)) * m                            # region coords

    i0 = torch.floor(u).long()
    frac = u - i0.float()

    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                wx = frac[:, 0] if ox else (1.0 - frac[:, 0])
                wy = frac[:, 1] if oy else (1.0 - frac[:, 1])
                wz = frac[:, 2] if oz else (1.0 - frac[:, 2])
                ix, iy, iz = i0[:, 0] + ox, i0[:, 1] + oy, i0[:, 2] + oz
                inside = ((ix >= 0) & (ix < ng) & (iy >= 0) & (iy < ng)
                          & (iz >= 0) & (iz < ng))
                w = (wx * wy * wz * inside.to(dt)).reshape(b, -1)
                idx = ((ix.clamp(0, ng - 1) * ng + iy.clamp(0, ng - 1)) * ng
                       + iz.clamp(0, ng - 1)).reshape(b, -1)
                yield idx, w


def cic_deposit_valid_center(
    disp: torch.Tensor,
    values: torch.Tensor,
    cellsize: float,
    dis_norm: float,
    region: int,
    grid_mult: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(mass, weighted)``: the valid-centre CIC of ``1`` and of ``values``.

    ``values`` is ``(B, K, N, N, N)``, one scalar per particle per channel --
    velocity components and squared speed, in practice. Returns the raw mass
    deposit ``(B, 1, ng, ng, ng)`` (**not** an overdensity: the caller usually
    wants it as the denominator of a mass-weighted mean, where the uniform
    normalisation would cancel anyway) and the value deposit
    ``(B, K, ng, ng, ng)``.

    The two share one pass over the corners, so ``weighted / mass`` is the
    mass-weighted mean of ``values`` in each cell exactly, with no possibility
    of the numerator and denominator disagreeing about which particles landed
    where.
    """
    if values.dim() != 5 or values.shape[0] != disp.shape[0] \
            or values.shape[2:] != disp.shape[2:]:
        raise ValueError(
            f"values {tuple(values.shape)} must be (B, K, N, N, N) matching "
            f"disp {tuple(disp.shape)}")
    b, k = int(disp.shape[0]), int(values.shape[1])
    ng = int(region) * int(grid_mult)
    dt = torch.float32
    mass = torch.zeros(b, ng ** 3, device=disp.device, dtype=dt)
    acc = torch.zeros(b, k, ng ** 3, device=disp.device, dtype=dt)
    flat = values.to(dt).reshape(b, k, -1)
    for idx, w in _valid_center_corners(disp, cellsize, dis_norm, region, grid_mult):
        mass.scatter_add_(1, idx, w)
        acc.scatter_add_(2, idx.unsqueeze(1).expand(-1, k, -1), flat * w.unsqueeze(1))
    return mass.view(b, 1, ng, ng, ng), acc.view(b, k, ng, ng, ng)


def cic_density_valid_center(
    disp: torch.Tensor,
    cellsize: float,
    dis_norm: float,
    region: int,
    grid_mult: int = 1,
) -> torch.Tensor:
    """CIC of a crop's own particles into the Eulerian sub-cube it actually fills.

    :func:`cic_density` deposits with ``% ng``, i.e. it wraps particles back inside
    the crop. On a whole periodic box that is exact; on a crop it is not an
    approximation but a scrambling. Measured on set14 with a 64^3 crop: the
    wrapped field is correlated only **r = 0.08** with the true density of that
    region, with sigma inflated 2.2x, because only 9.75% of a region's particles
    stay inside it (median |Psi| = 36 HR cells against a 64-cell crop).

    Particles from a Lagrangian crop ``B`` land in the Eulerian cube centred at
    ``centre(B) + <Psi>_B`` and shrunk by the internal spread of ``Psi - <Psi>_B``.
    This function therefore

    1. converts to absolute positions in crop coordinates,
    2. offsets by the crop's own (rounded, detached) mean displacement, and
    3. deposits into the central ``region`` cube, **discarding** rather than
       wrapping anything that leaves.

    ``region`` must be small enough that the crop can fill it. Measured internal
    spread on set14 (mean over crops): 64^3 crop -> 50 cells max / 13.6 rms;
    192^3 -> 81 / 18.8. See ``scripts/dmsr_crop_spread.py`` for the accuracy of a
    given ``(crop, region)`` pair; a region much larger than ``crop - 2*spread``
    loses mass at the faces.

    Measured accuracy of this construction against an exactly-buffered reference
    (set14, ``scripts/dmsr_crop_spread.py``), for a 128^3 crop::

        region/crop   rel RMS   corr    sigma ratio   mass kept
            0.25       0.000    1.0000     1.000        1.000
            0.50       0.030    0.9995     0.997        0.972
            0.75       0.213    0.9783     0.929        0.853
            1.00       0.396    0.9210     0.854        0.676

    against ``rel RMS = 2.36, corr = 0.080`` for the wrapped ``cic_density`` it
    replaces. Note that even ``region == crop`` is a large improvement: most of the
    damage was the ``% ng`` wrap, not the missing buffer.

    Returns ``(B, 1, grid_mult*region, ...)`` overdensity. Differentiable in
    ``disp`` through the CIC weights, as ``cic_density`` is.

    With ``grid_mult > 1`` an *unperturbed* lattice produces a checkerboard of
    ``delta = m^3 - 1`` and ``-1`` rather than a uniform field, because the
    particles sit exactly on fine-cell boundaries. That is the true CIC response
    to a lattice and ``cic_density`` shows it too; real displacements (median 36
    HR cells here) wash it out completely.
    """
    b, ng = disp.shape[0], int(region) * int(grid_mult)
    dev, dt = disp.device, torch.float32
    dens = torch.zeros(b, ng ** 3, device=dev, dtype=dt)
    for idx, w in _valid_center_corners(disp, cellsize, dis_norm, region, grid_mult):
        dens.scatter_add_(1, idx, w)

    dens = dens.view(b, 1, ng, ng, ng)
    # Normalised by the UNIFORM expectation, NOT by its own mean. The input grid
    # carries one particle per HR cell, and each fine cell has volume (1/m)^3 HR
    # cells, so a uniform universe deposits 1/m^3 of mass per fine cell.
    # Self-normalising would rescale away a mass deficit caused by an over-large
    # `region`, turning a detectable error into a silent bias.
    return dens * float(int(grid_mult) ** 3) - 1.0


def _band_auto_cross(a: torch.Tensor, b: torch.Tensor, n_bands: int):
    """Radially binned auto- and cross-power of two ``(B, 1, N, N, N)`` fields."""
    n = a.shape[-1]
    fa = torch.fft.rfftn(a.float(), dim=(-3, -2, -1))
    fb = torch.fft.rfftn(b.float(), dim=(-3, -2, -1))
    paa = (fa.real ** 2 + fa.imag ** 2) / (n ** 3)
    pbb = (fb.real ** 2 + fb.imag ** 2) / (n ** 3)
    pab = (fa.real * fb.real + fa.imag * fb.imag) / (n ** 3)

    idx = _kbin_index(n, n_bands, a.device).reshape(-1)
    out = []
    for p in (paa, pbb, pab):
        flat = p.reshape(p.shape[0], p.shape[1], -1).mean(dim=(0, 1))
        sums = torch.zeros(n_bands, device=a.device, dtype=flat.dtype)
        counts = torch.zeros(n_bands, device=a.device, dtype=flat.dtype)
        sums.index_add_(0, idx, flat)
        counts.index_add_(0, idx, torch.ones_like(flat))
        out.append(sums / counts.clamp_min(1.0))
    return out  # [P_aa, P_bb, P_ab]


@torch.no_grad()
def density_metrics(
    pred_disp: torch.Tensor,
    true_disp: torch.Tensor,
    base_disp: torch.Tensor,
    cellsize: float,
    dis_norm: float,
    n_bands: int = 6,
    prefix: str = "",
) -> Dict[str, float]:
    """Density-field agreement of a prediction and of the ``A`` baseline.

    ``sigma_ratio`` near 1 and ``pk_ratio_b*`` near 1 mean the degraded field
    clusters like the real LR field. The ``_A_`` keys are the block-average
    baseline measured the same way, so every number has its control alongside it.
    """
    d_pred = cic_density(pred_disp, cellsize, dis_norm)
    d_true = cic_density(true_disp, cellsize, dis_norm)
    d_base = cic_density(base_disp, cellsize, dis_norm)

    out: Dict[str, float] = {
        f"{prefix}sigma_delta_true": float(d_true.std()),
        f"{prefix}sigma_ratio": float(d_pred.std() / d_true.std().clamp_min(1e-12)),
        f"{prefix}sigma_ratio_A": float(d_base.std() / d_true.std().clamp_min(1e-12)),
    }
    for tag, field in (("", d_pred), ("A_", d_base)):
        p_xx, p_tt, p_xt = _band_auto_cross(field, d_true, n_bands)
        ratio = p_xx / p_tt.clamp_min(1e-20)
        rk = p_xt / (p_xx * p_tt).clamp_min(1e-20).sqrt()
        out[f"{prefix}pk_logbias_{tag}density"] = float(
            torch.log(ratio.clamp_min(1e-12)).abs().mean()
        )
        for i in range(n_bands):
            out[f"{prefix}pk_ratio_{tag}density_b{i}"] = float(ratio[i])
            out[f"{prefix}rk_{tag}density_b{i}"] = float(rk[i])
    return out
