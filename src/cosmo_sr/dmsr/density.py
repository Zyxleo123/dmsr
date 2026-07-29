"""Differentiable Eulerian density and the high-pass density critic channel.

The critic must judge *unresolved* structure. Field-space residuals alone are a
weak signal for that: the failure mode that actually matters here is Eulerian
(haloes too puffy or too concentrated), and it is invisible to Lagrangian
displacement statistics -- see the module docstring of
:mod:`cosmo_sr.eval.density`, which measured block-averaging making the universe
13% too clumpy while displacement MSE looked fine.

So each critic example carries a density channel built by moving particles to
``q + Psi`` and CIC-depositing them:

    rho      = differentiable_CIC_density(x)
    rho_high = rho - lowpass_to_LR_and_upsample(rho)

Everything here stays differentiable so that ``loss_G_adv`` reaches the flow
parameters *through the Eulerian path* (asserted in
``tests/dmsr/test_adversarial_grad.py``). CIC is differentiable in the standard
way: the deposit *weights* are smooth functions of the sub-cell offsets, while
the integer target cells are not differentiated through.

Low-pass definitions
--------------------
``lowpass`` is configurable because "the modes the LR simulation represents" has
more than one reasonable formalisation:

``"blockavg"`` (default)
    ``lowpass(rho) = A_plus(A(rho))`` with the *same* block factor ``s`` as the
    degradation operator. Then ``rho_high = P_A(rho)`` exactly, i.e. the density
    high-pass is the same projector used for the field residual. This is the
    most defensible choice: it removes precisely the density information the LR
    grid can carry under ``A``, no more and no less, and it introduces no new
    free parameter.

``"fourier"``
    Sharp isotropic ``k``-space cut: modes with ``|k| <= kcut_frac * k_Nyq_LR``
    are removed, where ``k_Nyq_LR = N_lr / 2`` in integer mode units. Spectrally
    cleaner (no block-shaped window leakage) but not an exact complement of
    ``A``, and it rings in configuration space. Provided for ablation.

Note the two differ: block-averaging is a top-hat window in real space, so
``"blockavg"`` leaves some sub-Nyquist power in ``rho_high`` and removes some
super-Nyquist power. That is a feature for our purpose (it matches ``A``), but it
means the two settings are not interchangeable -- report which was used.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..eval.density import cic_density, cic_density_valid_center
from ..operators.multiscale import block_average, block_upsample

_KCUT_CACHE: Dict[Tuple[int, float, str], torch.Tensor] = {}


def _pixel_unshuffle_3d(x: torch.Tensor, r: int) -> torch.Tensor:
    """3D space-to-depth: ``(B, C, rN, rN, rN) -> (B, C*r**3, N, N, N)``.

    ``torch.nn.functional.pixel_unshuffle`` is 2D only, so do it by hand. The
    ``r**3`` sub-cells of each output cell become channels, in a fixed order.
    """
    b, c, d, h, w = x.shape
    if d % r or h % r or w % r:
        raise ValueError(f"pixel_unshuffle_3d: {(d, h, w)} not divisible by r={r}")
    x = x.view(b, c, d // r, r, h // r, r, w // r, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return x.view(b, c * r ** 3, d // r, h // r, w // r)


def density_channels(density_mode: str) -> int:
    """Number of critic density channels for a given ``density_mode``."""
    return {"off": 0, "highpass": 1, "full": 1, "pshuffle8": 8}[density_mode]


def _lowpass_fourier(field: torch.Tensor, kcut: float) -> torch.Tensor:
    """Keep only modes with ``|k| <= kcut`` (integer mode units)."""
    n = field.shape[-1]
    key = (n, float(kcut), str(field.device))
    mask = _KCUT_CACHE.get(key)
    if mask is None:
        kx = torch.fft.fftfreq(n, device=field.device) * n
        kz = torch.fft.rfftfreq(n, device=field.device) * n
        KX, KY, KZ = torch.meshgrid(kx, kx, kz, indexing="ij")
        mask = (torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2) <= kcut).to(field.dtype)
        _KCUT_CACHE[key] = mask
    # fp32 FFT regardless of autocast: cuFFT half support is limited, and this
    # is cheap next to the conv stack (same rationale as losses/flow.band_power).
    fft = torch.fft.rfftn(field.float(), dim=(-3, -2, -1))
    return torch.fft.irfftn(fft * mask, s=field.shape[-3:], dim=(-3, -2, -1)).to(field.dtype)


class HighPassDensity(nn.Module):
    """Build the ``rho_high`` critic channel from an HR field. Differentiable.

    Parameters
    ----------
    factor:
        Block factor ``s`` of the degradation (8 here). Sets both the
        ``"blockavg"`` window and the ``"fourier"`` cut (``k_Nyq_LR = N_hr/(2s)``).
    lowpass:
        ``"blockavg"`` or ``"fourier"`` -- see the module docstring.
    kcut_frac:
        ``"fourier"`` only: cut in units of the LR Nyquist frequency.
    disp_channels:
        Which channels of ``x`` hold the displacement 3-vector. The on-disk
        canonical layout is ``disp[0:3] + vel[3:6]``.
    cellsize, dis_norm:
        Physical scales passed through to :func:`cosmo_sr.eval.density.cic_density`
        (``boxsize / lr_grid_res`` in kpc/h, and kpc/h per normalized displacement
        unit ``6000 * D(z)``).
    """

    def __init__(
        self,
        factor: int = 8,
        lowpass: str = "blockavg",
        kcut_frac: float = 1.0,
        disp_channels: Sequence[int] = (0, 1, 2),
        cellsize: float = 1000.0 * 1000.0 / 512.0,
        dis_norm: float = 6000.0,
        valid_center: int = 0,
    ):
        super().__init__()
        lowpass = str(lowpass).lower()
        if lowpass not in ("blockavg", "fourier"):
            raise ValueError(f"lowpass must be 'blockavg' or 'fourier', got {lowpass!r}")
        if len(disp_channels) != 3:
            raise ValueError(f"disp_channels must name 3 channels, got {disp_channels}")
        self.factor = int(factor)
        self.lowpass = lowpass
        self.kcut_frac = float(kcut_frac)
        self.disp_channels = tuple(int(c) for c in disp_channels)
        self.cellsize = float(cellsize)
        self.dis_norm = float(dis_norm)
        # 0 keeps the legacy wrapped deposit (and byte-identical behaviour); a
        # positive value scores only that central Eulerian cube, offset by the
        # crop's own bulk displacement. See `cic_density_valid_center` -- the
        # wrapped version is correlated only r=0.08 with the true density of a
        # 64^3 crop, so a critic trained on it is matching a scrambled field.
        self.valid_center = int(valid_center)

    def density(self, x_hr: torch.Tensor) -> torch.Tensor:
        """Eulerian overdensity ``delta`` from the displacement channels of ``x``."""
        idx = torch.tensor(self.disp_channels, device=x_hr.device)
        disp = x_hr.index_select(1, idx)
        if self.valid_center:
            return cic_density_valid_center(
                disp, self.cellsize, self.dis_norm, self.valid_center)
        return cic_density(disp, self.cellsize, self.dis_norm)

    def density_pshuffle(self, x_hr: torch.Tensor, r: int = 2) -> torch.Tensor:
        """SR2-style density channel: full ``delta`` on an ``r``x-finer CIC mesh,
        folded back to the field resolution by an inverse pixel shuffle.

        Returns ``(B, r**3, N, N, N)`` -- e.g. ``r=2`` -> 8 channels. The finer mesh
        exposes the sub-cell density structure (caustics/haloes) that a base-res
        CIC smooths away, and the space-to-depth fold keeps the spatial size equal
        to the residual channels so the critic input concatenates cleanly.
        """
        idx = torch.tensor(self.disp_channels, device=x_hr.device)
        disp = x_hr.index_select(1, idx)
        if self.valid_center:
            delta_fine = cic_density_valid_center(
                disp, self.cellsize, self.dis_norm, self.valid_center, grid_mult=r)
        else:
            delta_fine = cic_density(disp, self.cellsize, self.dis_norm, grid_mult=r)
        return _pixel_unshuffle_3d(delta_fine, r)

    def forward(self, x_hr: torch.Tensor) -> torch.Tensor:
        """``rho_high`` as a ``(B, 1, N, N, N)`` field. Gradients reach ``x_hr``."""
        rho = self.density(x_hr)
        if self.lowpass == "blockavg":
            low = block_upsample(block_average(rho, self.factor), self.factor)
        else:
            n_hr = rho.shape[-1]
            kcut = self.kcut_frac * (n_hr / self.factor) / 2.0
            low = _lowpass_fourier(rho, kcut)
        return rho - low

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"factor={self.factor}, lowpass={self.lowpass!r}, "
            f"kcut_frac={self.kcut_frac}, disp_channels={self.disp_channels}"
        )


def cellsizes(data_cfg, factor: int):
    """``(hr_cellsize, lr_cellsize)`` in kpc/h, derived from the box geometry.

    ``cic_density`` builds its Lagrangian lattice in units of **one cell of the grid
    the field lives on**, so the physical ``cellsize`` differs between call sites:

    * :class:`HighPassDensity` sees **HR** fields (crops of the ``lr_grid*factor``
      grid)  -> ``boxsize / (lr_grid * factor)``
    * :func:`cosmo_sr.dmsr.env.environment_descriptors` sees **LR** crops
      -> ``boxsize / lr_grid``

    Deriving both from `data.boxsize` / `data.lr_grid_res` instead of hand-writing
    them: a single hand-written constant (15625.0) was used for both and was wrong
    at *both* sites -- 80x at the HR one and 10x at the LR one.
    """
    boxsize = float(data_cfg.get("boxsize", 100000.0))
    lr_grid = float(data_cfg.get("lr_grid_res", 64))
    return boxsize / (lr_grid * float(factor)), boxsize / lr_grid


class CriticInputNormalizer(nn.Module):
    """Fixed per-channel scales so ``residual`` and ``rho_high`` enter comparably.

    The critic input is ``concat(P_A(x)[3ch], rho_high(x)[1ch])`` -- two physically
    different quantities in different units. Without normalisation their relative
    scale is an accident of the unit constants, and it was badly wrong here:

        cellsize 15625 (as run, 80x too large):  std(rho_high)/std(residual) = 0.53
        cellsize 195.3 (correct):                std(rho_high)/std(residual) = 138

    i.e. correcting the units alone would swing the critic from a mildly
    under-weighted density channel to one that dominates by two orders of magnitude,
    starving the residual channels of gradient (spectral norm caps the first conv's
    ability to rescale). Normalising makes the balance explicit and makes the critic
    invariant to this whole class of unit error.

    **The scales are constants estimated from REAL crops only, and the identical
    constants are applied to real and fake.** Per-batch statistics must not be used:
    real and fake would then be normalised differently, handing the critic a
    discriminative signal that has nothing to do with sample quality -- the same
    class of shortcut that withholding the raw LR tensor is meant to prevent.
    """

    def __init__(self, residual_scale: float, density_scale: float):
        super().__init__()
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        self.register_buffer("density_scale", torch.tensor(float(density_scale)))

    @classmethod
    @torch.no_grad()
    def fit(cls, batches, operator, highpass, eps: float = 1e-12,
            residual_mode: str = "nullspace", density_mode: str = "highpass"
            ) -> "CriticInputNormalizer":
        """Estimate scales from an iterable of **real** HR tensors.

        The scales must match the fields that :func:`critic_input` actually emits,
        so the modes are threaded through here: ``residual_mode`` selects
        ``P_A(x)`` vs the full field ``x``; ``density_mode`` selects the high-pass
        vs the full overdensity (or ``"off"`` for no density channel).
        """
        r_sq = d_sq = n = 0.0
        for x in batches:
            r_sq += float(_residual_view(x, operator, residual_mode).pow(2).mean())
            if density_mode != "off":
                d_sq += float(_density_view(x, highpass, density_mode).pow(2).mean())
            n += 1.0
        if n == 0:
            raise ValueError("CriticInputNormalizer.fit got no batches")
        d_scale = eps if density_mode == "off" else max((d_sq / n) ** 0.5, eps)
        return cls(max((r_sq / n) ** 0.5, eps), d_scale)

    def forward(self, residual: torch.Tensor,
                rho_high: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = residual / self.residual_scale
        if rho_high is None:
            return res
        return torch.cat([res, rho_high / self.density_scale], dim=1)

    def to_dict(self) -> Dict[str, float]:
        return {"residual_scale": float(self.residual_scale),
                "density_scale": float(self.density_scale)}

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (f"residual_scale={float(self.residual_scale):.6g}, "
                f"density_scale={float(self.density_scale):.6g}")


def _residual_view(x_hr: torch.Tensor, operator, residual_mode: str) -> torch.Tensor:
    """The field-residual channels the critic sees.

    ``"nullspace"`` (default) -> ``P_A(x)``: strips the coarse/range-space
    displacement, so the critic judges only the unresolved (null-space) detail.
    ``"full"`` -> ``x`` itself: the critic sees the FULL displacement field,
    including the coarse component ``P_A`` removes. This mirrors SR2's
    discriminator (which takes the whole displacement) and lets the critic police
    the range-space correction that an unconstrained generator uses to fix the
    misspecified operator -- content ``P_A`` would otherwise hide from it.
    """
    if residual_mode == "nullspace":
        return operator.P_A(x_hr)
    if residual_mode == "full":
        return x_hr
    raise ValueError(f"residual_mode must be 'nullspace' or 'full', got {residual_mode!r}")


def _density_view(x_hr: torch.Tensor, highpass: HighPassDensity, density_mode: str) -> torch.Tensor:
    """The density channel the critic sees.

    ``"highpass"`` (default) -> ``rho - lowpass(rho)``; ``"full"`` -> the full
    Eulerian overdensity ``delta`` (no low-pass removed), giving the critic the
    density at every scale rather than only the unresolved band; ``"pshuffle8"`` ->
    SR2's channel: full ``delta`` on a 2x-finer CIC mesh, inverse-pixel-shuffled to
    8 channels at field resolution. ``"off"`` is handled by the caller (no channel).
    """
    if density_mode == "highpass":
        return highpass(x_hr)
    if density_mode == "full":
        return highpass.density(x_hr)
    if density_mode == "pshuffle8":
        return highpass.density_pshuffle(x_hr, r=2)
    raise ValueError(
        f"density_mode must be 'highpass', 'full', 'pshuffle8' or 'off', got {density_mode!r}")


def critic_input(
    x_hr: torch.Tensor,
    operator,
    highpass: HighPassDensity,
    residual: Optional[torch.Tensor] = None,
    normalizer: Optional["CriticInputNormalizer"] = None,
    residual_mode: str = "nullspace",
    density_mode: str = "highpass",
) -> torch.Tensor:
    """``concat(residual_view(x), density_view(x))`` -- the critic's view of an HR field.

    ``residual`` may be supplied when the caller already holds the residual view
    (the generator does), to avoid recomputing it. ``residual_mode`` /
    ``density_mode`` select what those views are (see :func:`_residual_view` /
    :func:`_density_view`); the defaults reproduce the original
    ``concat(P_A(x), rho_high(x))`` byte-for-byte.

    Deliberately contains **no raw LR tensor**: the critic is meant to judge
    unresolved structure, and handing it ``y`` invites it to key on the LR
    distribution instead -- which is exactly the paired-vs-unpaired shortcut that
    Stage D's environment balancing exists to close.
    """
    res = _residual_view(x_hr, operator, residual_mode) if residual is None else residual
    rho = None if density_mode == "off" else _density_view(x_hr, highpass, density_mode)
    if rho is not None and rho.shape[-1] != res.shape[-1]:
        # `highpass.valid_center` scores only the Eulerian cube the crop can
        # actually fill, so the density view is smaller than the residual view.
        # Centre-crop the residual to match: the critic must see the two channel
        # groups on the same voxels, and the residual is the one that is valid
        # everywhere. Anything other than a shrink is a bug, so it is not silent.
        n_res, n_rho = res.shape[-1], rho.shape[-1]
        if n_rho > n_res or (n_res - n_rho) % 2:
            raise ValueError(
                f"density view {n_rho} cannot be centred in residual view {n_res}; "
                f"valid_center must be smaller than the crop and of equal parity"
            )
        o = (n_res - n_rho) // 2
        res = res[..., o:o + n_rho, o:o + n_rho, o:o + n_rho]
    if normalizer is not None:
        return normalizer(res, rho)
    return res if rho is None else torch.cat([res, rho], dim=1)
