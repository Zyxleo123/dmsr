"""Differentiable field-level guards for the member-gather objective.

``docs/sr2_member_gather.md`` section 7 item 2 is the reason this module exists.
The id-gathered loss is a valid specification of *a bound halo at a location*
and says nothing whatever about the field between the halos, so the free-field
runs drove displacement power above ``k_split`` to **5.50x HR** and it was still
climbing at step 2000. On a free field editing 0.8% of one box that is a
recorded caveat. On a **generator**, which applies one learned operator at every
site of every box, it is a corruption of everything the model does -- so it has
to be an active term before any real training run, not a reported number.

Why a hinge and not an anchor
-----------------------------
``docs/sr2_gather_finetune.md`` section 3.3 ran the alternative and measured it:
an L2 anchor ``||Psi - Psi_0||^2`` raised from 0.1 to 10 moved local peak
contrast outside the supervised windows **0.570 -> 0.517**, the wrong direction.
The reason is the same one that makes an L2 *objective* blur -- a squared
penalty is cheapest to satisfy with broad, low-amplitude change spread over the
whole field, which is exactly what erases local peaks. An anchor is therefore
the wrong instrument for "do not invent small-scale power": it charges for the
structure we want as readily as for the excess we do not.

The hinge charges **only for exceeding HR**:

    L_highk = [P_cand(k > k_split) / P_HR(k > k_split) - 1]_+^2

Below HR it is exactly zero with exactly zero gradient, which is the same
property every hinged term in :mod:`cosmo_sr.features.member_gather` has, and
for the same reason: the objective must never be able to ask for more structure
than HR has, and must never penalise a run for not yet having enough. Building
substructure *raises* high-k power, so an anchor and this hinge disagree about
the sign of the very thing the line is trying to produce; only the hinge is
compatible with the objective.

The Hann window is not optional
-------------------------------
A tile is a sub-cube of a periodic box and is **not** itself periodic, so its
FFT sees a step at every face and that step leaks power across every ``k``.
:func:`cosmo_sr.features.cond_spread.hann_window` records the measurement: an
unwindowed tile spectrum reported SR2's displacement tracking HR to 0.83 at
subhalo scale where the windowed value is ~0.00. The guard windows both fields
identically. Every quantity here is a **ratio** of powers computed the same way,
so the window's normalisation and the ``dx^3 / n^3`` power normalisation both
cancel and no correction factor is applied -- which is also why the guard may
run on the on-disk normalised channels rather than on Mpc/h, as long as both
sides use the same units.

Conventions match :func:`cosmo_sr.features.cond_spread.radial_cross_spectra`
exactly -- ``2 pi`` wavenumbers, power summed over the three vector components,
``k = 0`` dropped -- so ``tests/features/test_field_guards.py`` can pin this
module against that one's numpy on identical inputs. A disagreement between the
guard and the reported ``highk_power_ratio`` would be indistinguishable from a
run that misbehaved, which is precisely the confusion this line has paid for
before.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

__all__ = [
    "band_edges_torch",
    "banded_highk_hinge",
    "banded_power_ratio_torch",
    "hann_window_torch",
    "highk_hinge",
    "highk_power_ratio_torch",
    "wavenumbers_torch",
]

_CACHE: Dict[Tuple, torch.Tensor] = {}


def wavenumbers_torch(n: int, dx: float, device, dtype=torch.float32
                      ) -> torch.Tensor:
    """``(n, n, n)`` ``|k|`` in ``h/Mpc``, the ``2 pi`` convention.

    Mirrors :func:`cosmo_sr.features.cond_spread.wavenumbers`. Cached per
    ``(n, dx, device)``: it is rebuilt every eval step otherwise, and it never
    changes within a run.
    """
    key = ("k", int(n), float(dx), str(device), str(dtype))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    f = 2.0 * torch.pi * torch.fft.fftfreq(int(n), d=float(dx),
                                           device=device, dtype=dtype)
    kmag = torch.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2
                      + f[None, None, :] ** 2)
    _CACHE[key] = kmag
    return kmag


def hann_window_torch(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """``(n, n, n)`` separable Hann window.

    ``numpy.hanning(n + 2)[1:-1]`` is reproduced exactly rather than using
    ``torch.hann_window``, whose ``periodic=True`` default is a *different*
    window. The two differ by one sample of phase, which is small and which
    would silently decouple this guard from the reported spectra.
    """
    key = ("w", int(n), str(device), str(dtype))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    m = int(n) + 2
    i = torch.arange(m, device=device, dtype=torch.float64)
    w = (0.5 - 0.5 * torch.cos(2.0 * torch.pi * i / (m - 1)))[1:-1]
    win = (w[:, None, None] * w[None, :, None] * w[None, None, :]).to(dtype)
    _CACHE[key] = win
    return win


def _highk_power(field: torch.Tensor, mask: torch.Tensor,
                 window: torch.Tensor) -> torch.Tensor:
    """Mean power over the modes selected by ``mask``, summed over components.

    ``field`` is ``(B, C, n, n, n)``. The ``dx^3 / n^3`` normalisation of
    ``radial_cross_spectra`` is a constant factor and is omitted: every use of
    this function is a ratio of two such powers on the same grid.
    """
    fft = torch.fft.fftn((field * window).float(), dim=(-3, -2, -1))
    power = (fft.real ** 2 + fft.imag ** 2).sum(dim=1)      # (B, n, n, n)
    sel = power[:, mask]
    return sel.mean()


def highk_power_ratio_torch(
    candidate: torch.Tensor, reference: torch.Tensor, *,
    dx: float, k_split: float = 4.0, channels: slice = slice(0, 3),
    kmag: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``P_cand(k > k_split) / P_ref(k > k_split)``, differentiable in ``candidate``.

    ``channels`` defaults to the three displacement components, matching what
    ``field_report`` measures and what section 7 item 2 recorded at 5.50. Pass
    ``slice(3, 6)`` to watch velocity, which is a separate defect and a separate
    guard.

    ``k = 0`` is excluded by ``k_split > 0``: a tile's mean displacement is bulk
    motion and says nothing about whether anything inside it collapsed.
    """
    if candidate.shape != reference.shape or candidate.dim() != 5:
        raise ValueError("expected matching (B, C, n, n, n); got "
                         f"{tuple(candidate.shape)} {tuple(reference.shape)}")
    a = candidate[:, channels]
    b = reference[:, channels]
    n = int(a.shape[-1])
    if kmag is None:
        kmag = wavenumbers_torch(n, dx, a.device)
    mask = kmag >= float(k_split)
    if not bool(mask.any()):
        raise ValueError(
            f"k_split {k_split} h/Mpc selects no modes on a {n}^3 grid of "
            f"dx {dx} Mpc/h (Nyquist is {torch.pi / dx:.2f}). The guard would "
            "be silently inert, which is the failure mode it exists to prevent.")
    win = hann_window_torch(n, a.device, a.dtype)
    p_ref = _highk_power(b, mask, win).detach()
    p_cand = _highk_power(a, mask, win)
    return p_cand / p_ref.clamp_min(1e-30)


def highk_hinge(
    candidate: torch.Tensor, reference: torch.Tensor, *,
    dx: float, k_split: float = 4.0, channels: slice = slice(0, 3),
    kmag: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``([ratio - 1]_+^2, ratio)`` -- zero value and zero gradient below HR.

    Returned as a pair so a caller can log the raw ratio without recomputing an
    FFT, and so the reported number and the penalised number can never be two
    different quantities.
    """
    ratio = highk_power_ratio_torch(candidate, reference, dx=dx,
                                    k_split=k_split, channels=channels,
                                    kmag=kmag)
    return torch.clamp(ratio - 1.0, min=0.0) ** 2, ratio


# --------------------------------------------------------------------------- #
# Band-resolved high-k guard
# --------------------------------------------------------------------------- #
# WHY A SECOND FORM OF THE SAME GUARD EXISTS
#
# `highk_power_ratio_torch` reduces the whole band above `k_split` to one
# `sel.mean()`. The first guess about why that hides things -- that the mean is
# dominated by the outermost shells, since `k >= 4` admits 99.2% of a 64^3
# tile's modes and 48% of them sit above Nyquist -- is WRONG, and the measurement
# says so. On the held-out set9 export, HR's share of that mean per log-k bin is
#
#     k     4.4   5.2   6.2   7.4   8.8  10.4  12.4  14.8   h/Mpc
#     share 21.3  15.0  12.9  10.4   9.7   9.1   9.7  11.9  %
#
# a spread of only 2.3x, against a 35x spread in mode count: P(k) ~ k^-2 very
# nearly cancels N(k) ~ k^2. The scalar is a FAIR power-weighted average.
#
# That is exactly why it hides the defect. `all_blocks_self`, the arm that
# reproduced HR's subhalo mass function out of sample (20 -> 366 vs HR's 369),
# measures per bin:
#
#     k     4.4   5.2   6.2   7.4   8.8  10.4  12.4  14.8
#     ratio 3.34  3.39  2.94  2.09  1.39  0.64  0.27  0.11   x HR
#
# +3.4x at the subhalo scale and a 9x DEFICIT at the grid scale, which a fair
# average reports as 1.59. One number cannot say "far too much here, far too
# little there", and the one-sided hinge cannot charge the second half at all.
# The banded form resolves both, and `two_sided` is what makes the deficit
# chargeable -- though see `banded_highk_hinge` on why that must be aimed.

def band_edges_torch(n: int, dx: float, *, k_split: float, n_bins: int,
                     k_max: Optional[float], device, dtype=torch.float32
                     ) -> torch.Tensor:
    """``(n_bins + 1,)`` log-spaced edges from ``k_split`` to ``k_max``.

    ``k_max = None`` runs to the cube corner, which keeps the banded guard's
    support identical to :func:`highk_power_ratio_torch`'s mask so the two are
    comparable on the same field. Passing ``k_max = pi / dx`` restricts it to
    the isotropically sampled region instead -- above Nyquist a "shell" is only
    the corners of the cube, and its power is a direction-dependent quantity
    that no isotropic reference can be compared against cleanly.
    """
    hi = float(k_max) if k_max else float(wavenumbers_torch(n, dx, device).max())
    if not hi > float(k_split):
        raise ValueError(f"k_max {hi} is not above k_split {k_split}")
    return torch.logspace(float(torch.log10(torch.tensor(float(k_split)))),
                          float(torch.log10(torch.tensor(hi * 1.001))),
                          int(n_bins) + 1, device=device, dtype=dtype)


def _band_index(n: int, dx: float, *, k_split: float, n_bins: int,
                k_max: Optional[float], device
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(bin_of_mode, k_centres, counts)``; ``-1`` marks a mode outside the band.

    Tuple order matches :func:`banded_power_ratio_torch`'s deliberately: the two
    disagreeing is a silent centres-for-counts swap at the call site, which cost
    a test its meaning once already.

    Cached like the wavenumbers: it depends only on the grid and the binning,
    never on the field, and rebuilding it every step would dominate the guard.
    """
    key = ("b", int(n), float(dx), float(k_split), int(n_bins),
           float(k_max or 0.0), str(device))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    kmag = wavenumbers_torch(n, dx, device)
    edges = band_edges_torch(n, dx, k_split=k_split, n_bins=n_bins,
                             k_max=k_max, device=device)
    idx = torch.bucketize(kmag.reshape(-1), edges, right=False) - 1
    idx = torch.where((kmag.reshape(-1) >= float(k_split))
                      & (idx >= 0) & (idx < int(n_bins)),
                      idx, torch.full_like(idx, -1))
    counts = torch.zeros(int(n_bins), device=device, dtype=torch.float32)
    keep = idx >= 0
    counts.index_add_(0, idx[keep], torch.ones(int(keep.sum()), device=device))
    centres = torch.sqrt(edges[:-1] * edges[1:])
    out = (idx, centres, counts)
    _CACHE[key] = out
    return out


def banded_power_ratio_torch(
    candidate: torch.Tensor, reference: torch.Tensor, *,
    dx: float, k_split: float = 4.0, n_bins: int = 8,
    k_max: Optional[float] = None, channels: slice = slice(0, 3),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(ratio_per_bin, k_centres, mode_counts)``, differentiable in ``candidate``.

    Each bin's ratio is a mean power over the modes it holds, pooled over the
    batch exactly as :func:`_highk_power` pools -- so a one-bin call reproduces
    :func:`highk_power_ratio_torch` on the same mask, which
    ``tests/features/test_field_guards.py`` pins.

    Bins that hold no modes come back as NaN and MUST be dropped by the caller;
    they exist because log spacing at a fixed ``n_bins`` can under-populate the
    first octave on a small grid, and silently treating an empty bin as a ratio
    of zeros would put a fixed fictitious deficit into the penalty.
    """
    if candidate.shape != reference.shape or candidate.dim() != 5:
        raise ValueError("expected matching (B, C, n, n, n); got "
                         f"{tuple(candidate.shape)} {tuple(reference.shape)}")
    a, b = candidate[:, channels], reference[:, channels]
    n = int(a.shape[-1])
    idx, centres, counts = _band_index(n, dx, k_split=k_split, n_bins=n_bins,
                                       k_max=k_max, device=a.device)
    win = hann_window_torch(n, a.device, a.dtype)

    def _binned(field: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fftn((field * win).float(), dim=(-3, -2, -1))
        power = (fft.real ** 2 + fft.imag ** 2).sum(dim=1).reshape(-1)
        flat = idx.repeat(field.shape[0])
        keep = flat >= 0
        out = torch.zeros(int(n_bins), device=field.device, dtype=power.dtype)
        out = out.index_add(0, flat[keep], power[keep])
        return out / (counts * float(field.shape[0])).clamp_min(1.0)

    p_ref = _binned(b).detach()
    p_cand = _binned(a)
    ratio = torch.where(counts > 0, p_cand / p_ref.clamp_min(1e-30),
                        torch.full_like(p_ref, float("nan")))
    return ratio, centres, counts


def banded_highk_hinge(
    candidate: torch.Tensor, reference: torch.Tensor, *,
    dx: float, k_split: float = 4.0, n_bins: int = 8,
    k_max: Optional[float] = None, channels: slice = slice(0, 3),
    tol: float = 0.0, two_sided: bool = False, reduce: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(penalty, ratio_per_bin, k_centres)`` -- per-octave, optionally two-sided.

    ``tol`` is a DEAD ZONE in log-ratio: nothing is charged while the bin sits
    within a factor ``1 + tol`` of HR. The one-sided default with ``tol = 0``
    reproduces the original hinge's boundary exactly, one bin at a time.

    Why a dead zone at all. The original hinge is exactly zero with exactly zero
    gradient below HR, so the optimiser has no reason to keep any margin -- and
    `all_blocks_self` ended its run with its worst TRAIN host at 0.887, i.e. the
    term was contributing precisely nothing to the gradient, while the held-out
    worst host stood at 3.87. A term that switches itself off on the pool it is
    measured on cannot hold anything out of sample. A margin keeps it alive.

    ``two_sided`` additionally charges a bin for falling BELOW HR, symmetrically
    in log. This is deliberately not the default and is a different instrument
    from the L2 anchor that ``sr2_gather_finetune.md`` section 3.3 measured
    moving peak contrast the wrong way (0.570 -> 0.517): that anchor was to the
    FROZEN field and so charged for changing anything, whereas a lower bound
    referenced to **HR** asks for *more* small-scale power, which is the
    direction the objective already wants. It is what would have charged
    `all_blocks_nocentre` for taking high-k to 0.026 -- a 38x deficit the
    one-sided hinge rated as perfect.

    ``reduce`` is ``"mean"`` (every octave equally) or ``"max"`` (the worst
    octave alone). ``"mean"`` is the default because a max hands the whole
    gradient to one bin and the failure being fixed is a shape, not a spike.
    """
    ratio, centres, counts = banded_power_ratio_torch(
        candidate, reference, dx=dx, k_split=k_split, n_bins=n_bins,
        k_max=k_max, channels=channels)
    live = counts > 0
    if not bool(live.any()):
        raise ValueError(
            f"k_split {k_split} / n_bins {n_bins} left every band empty on a "
            f"{candidate.shape[-1]}^3 grid of dx {dx}. The guard would be "
            "silently inert, which is the failure mode it exists to prevent.")
    r = ratio[live].clamp_min(1e-12)
    over = torch.clamp(torch.log(r) - float(torch.log(torch.tensor(1.0 + tol))),
                       min=0.0) ** 2
    pen = over
    if two_sided:
        under = torch.clamp(-torch.log(r)
                            - float(torch.log(torch.tensor(1.0 + tol))),
                            min=0.0) ** 2
        pen = pen + under
    return (pen.max() if reduce == "max" else pen.mean()), ratio, centres
