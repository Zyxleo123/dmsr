"""The 24 orientation-preserving rotations of the cube, acting on 3-vector fields.

Used by Stage E (cubic-equivariance regularization). Only the ``det = +1``
subgroup is included: arbitrary-angle rotations would need interpolation, and
improper elements (reflections) are not symmetries of a chiral velocity field's
dynamics in the sense we want to impose here.

**A rotation acts on two things at once.** For a displacement or velocity field
``f: R^3 -> R^3`` sampled on the grid, the rotated field is

    (g . f)(x) = M f(M^{-1} x)

so the transform must permute/flip the **spatial voxel axes** *and* apply the same
signed permutation to the **vector components**. Rotating only the voxels (the
common bug) produces a field that is no longer a physically self-consistent
configuration -- the arrows point the wrong way.

The concrete convention here deliberately mirrors
:func:`cosmo_sr.data.datasets._augment_field`, which the existing training
augmentation already uses and trusts, restricted to the ``det = +1`` half:

    forward:  negate components on flipped axes -> flip those spatial axes
              -> permute spatial axes -> permute components
    inverse:  un-permute components -> un-permute spatial axes
              -> flip axes -> negate components

Because ``A`` block-averages and ``A_plus`` block-broadcasts over *axis-aligned*
blocks, and a flip of a length-``N`` axis maps block ``j`` exactly onto block
``N/s - 1 - j``, every element of this group commutes with the operator::

    A(g(x))    = g_LR(A(x))
    P_A(g(x))  = g(P_A(x))

Both are asserted for all 24 elements in ``tests/dmsr/test_cubic.py``; Stage E
must not be enabled unless those pass.
"""
from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import torch

Perm = Tuple[int, int, int]
Flip = Tuple[int, ...]


def _perm_sign(perm: Sequence[int]) -> int:
    """Parity of a permutation as +1 / -1 (bubble count on 3 elements)."""
    p = list(perm)
    sign = 1
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                sign = -sign
    return sign


class CubicRotation:
    """One orientation-preserving cube rotation.

    Parameters
    ----------
    perm:
        Spatial axis permutation; component ``i`` of the output is taken from
        component ``perm[i]`` of the input.
    flip_axes:
        Axes (in the *pre-permutation* frame) that are reversed.

    The pair must satisfy ``(-1)**len(flip_axes) * sign(perm) == +1``.
    """

    def __init__(self, perm: Perm, flip_axes: Flip):
        perm = tuple(int(p) for p in perm)
        flip_axes = tuple(sorted(int(a) for a in flip_axes))
        if sorted(perm) != [0, 1, 2]:
            raise ValueError(f"perm must be a permutation of (0,1,2), got {perm}")
        if any(not 0 <= a < 3 for a in flip_axes):
            raise ValueError(f"flip_axes out of range: {flip_axes}")
        det = ((-1) ** len(flip_axes)) * _perm_sign(perm)
        if det != 1:
            raise ValueError(
                f"perm={perm}, flip_axes={flip_axes} has det={det}; only "
                "orientation-preserving (det=+1) rotations are allowed"
            )
        self.perm = perm
        self.flip_axes = flip_axes
        # inverse permutation: iperm[perm[i]] = i
        iperm = [0, 0, 0]
        for i, p in enumerate(perm):
            iperm[p] = i
        self.iperm: Perm = tuple(iperm)  # type: ignore[assignment]

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _check(x: torch.Tensor, scalar: bool) -> None:
        if x.dim() != 5:
            raise ValueError(f"expected (B, C, N, N, N), got {tuple(x.shape)}")
        c = x.shape[1]
        if not scalar and c % 3 != 0:
            raise ValueError(
                f"vector field must have a channel count divisible by 3, got {c}; "
                "pass scalar=True for scalar fields such as density"
            )

    def _permute_components(self, x: torch.Tensor, perm: Sequence[int]) -> torch.Tensor:
        """Reorder each stacked 3-vector triple: ``out[i] = x[perm[i]]``."""
        c = x.shape[1]
        idx = torch.cat(
            [torch.tensor([t + p for p in perm], device=x.device) for t in range(0, c, 3)]
        )
        return x.index_select(1, idx)

    def _negate_components(self, x: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
        if not axes:
            return x
        c = x.shape[1]
        sign = torch.ones(c, device=x.device, dtype=x.dtype)
        for t in range(0, c, 3):
            for a in axes:
                sign[t + a] = -1.0
        return x * sign.view(1, c, 1, 1, 1)

    # -- the action --------------------------------------------------------- #
    def apply(self, x: torch.Tensor, scalar: bool = False) -> torch.Tensor:
        """Rotate a field. ``scalar=True`` skips all vector-component handling."""
        self._check(x, scalar)
        if not scalar:
            x = self._negate_components(x, self.flip_axes)
        if self.flip_axes:
            x = torch.flip(x, dims=[2 + a for a in self.flip_axes])
        x = x.permute(0, 1, *(2 + p for p in self.perm)).contiguous()
        if not scalar:
            x = self._permute_components(x, self.perm)
        return x

    def invert(self, x: torch.Tensor, scalar: bool = False) -> torch.Tensor:
        """Apply the inverse rotation (``invert(apply(x)) == x``)."""
        self._check(x, scalar)
        if not scalar:
            x = self._permute_components(x, self.iperm)
        x = x.permute(0, 1, *(2 + p for p in self.iperm)).contiguous()
        if self.flip_axes:
            x = torch.flip(x, dims=[2 + a for a in self.flip_axes])
        if not scalar:
            x = self._negate_components(x, self.flip_axes)
        return x

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CubicRotation(perm={self.perm}, flip_axes={self.flip_axes})"


def all_cubic_rotations() -> List[CubicRotation]:
    """All 24 orientation-preserving cube rotations.

    Enumerates the 48 signed axis permutations and keeps the ``det = +1`` half.
    """
    out: List[CubicRotation] = []
    for perm in itertools.permutations((0, 1, 2)):
        for n_flip in range(4):
            for flip_axes in itertools.combinations((0, 1, 2), n_flip):
                if ((-1) ** len(flip_axes)) * _perm_sign(perm) == 1:
                    out.append(CubicRotation(perm, flip_axes))  # type: ignore[arg-type]
    if len(out) != 24:  # pragma: no cover - structural invariant
        raise AssertionError(f"expected 24 rotations, built {len(out)}")
    return out


def sample_cubic_rotation(generator: torch.Generator | None = None) -> CubicRotation:
    """Draw one of the 24 rotations uniformly (seedable via ``generator``)."""
    rots = all_cubic_rotations()
    i = int(torch.randint(len(rots), (1,), generator=generator).item())
    return rots[i]
