"""Correctness diagnostics for the projected moment target Pi(Psi_HR - Psi_SR2).

The build job (``scripts/features/build_moment_target.py``) generates SR2, forms
the residual, projects it, and writes the target. These are the numbers and
slices that answer *is the target correct?* -- separated out so they are unit-
testable without a GPU, and so the CPU renderer redraws from what they write
rather than recomputing anything (project convention: figures are redraws).

Three questions, three checks:

1. **Did the projection remove the affine part inside footprints?** Per host, the
   affine-moment norm of the residual before vs after -- ``after`` must be ~0 --
   and the fraction of footprint variance that was affine (what got removed).
2. **Did it leave everything else alone?** Off every footprint, target == residual
   exactly.
3. **Does what remains look like substructure?** Slices of ``|disp|`` for HR, SR2,
   the raw residual and the projected target through a host, so the eye can see
   the bulk removed and the small-scale structure kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from cosmo_sr.features.moment_constraint import MomentProjector

__all__ = [
    "HostMomentRow",
    "per_host_moment_rows",
    "offfootprint_max_abs_diff",
    "host_slice_panels",
]


@dataclass
class HostMomentRow:
    """Per-host projection audit. ``moment_norm_after`` ~ 0 is the pass."""

    row: int
    n_sites: int
    rms_before: float          # rms |d| over the footprint, raw residual
    rms_after: float           # rms |d| over the footprint, projected target
    moment_norm_before: float  # ||Phi^T d|| on the residual
    moment_norm_after: float   # ||Phi^T d|| on the target -- must be ~0
    affine_var_frac: float     # ||P d||^2 / ||d||^2, the fraction removed


def _disp(field: np.ndarray) -> np.ndarray:
    """The three displacement channels, flattened to ``(3, n_hr)``."""
    flat = np.asarray(field).reshape(np.asarray(field).shape[0], -1)
    return flat[0:3]


def per_host_moment_rows(
    proj: MomentProjector,
    residual: np.ndarray,
    target: np.ndarray,
    rows: Sequence[int] | None = None,
) -> List[HostMomentRow]:
    """Audit each host's affine removal on the displacement channels.

    ``residual`` and ``target`` are whole-box fields, ``(C, ng, ng, ng)`` or
    ``(C, n_hr)`` with ``C >= 3``; only the displacement triplet is audited.
    """
    d_res, d_tgt = _disp(residual), _disp(target)
    blocks = {b.row: b for b in proj.blocks}
    want = list(blocks) if rows is None else [int(r) for r in rows]
    out: List[HostMomentRow] = []
    for r in want:
        blk = blocks[r]
        dr = d_res[:, blk.sites].T          # (n, 3)
        dt = d_tgt[:, blk.sites].T
        mom_before = blk.phi.T @ dr         # (k, 3)
        mom_after = blk.phi.T @ dt
        affine = dr - blk.remove_affine(dr)  # P d = d - (I-P) d
        e_tot = float(np.sum(dr ** 2))
        out.append(HostMomentRow(
            row=r,
            n_sites=blk.n_sites,
            rms_before=float(np.sqrt(np.mean(dr ** 2))),
            rms_after=float(np.sqrt(np.mean(dt ** 2))),
            moment_norm_before=float(np.linalg.norm(mom_before)),
            moment_norm_after=float(np.linalg.norm(mom_after)),
            affine_var_frac=float(np.sum(affine ** 2) / e_tot) if e_tot > 0 else 0.0,
        ))
    return out


def offfootprint_max_abs_diff(
    proj: MomentProjector, residual: np.ndarray, target: np.ndarray
) -> float:
    """Largest ``|target - residual|`` off every footprint -- must be 0.

    The projector touches only bound sites; anywhere it should not have acted the
    two fields are byte-identical. A nonzero here means a footprint leaked.
    """
    res = np.asarray(residual).reshape(np.asarray(residual).shape[0], -1)
    tgt = np.asarray(target).reshape(np.asarray(target).shape[0], -1)
    free = ~proj.footprint_mask()
    if not free.any():
        return 0.0
    return float(np.max(np.abs(tgt[:, free] - res[:, free])))


def host_slice_panels(
    fields: Dict[str, np.ndarray],
    centre_site: Tuple[int, int, int],
    ng_hr: int,
    half: int = 48,
    axis: int = 2,
) -> Dict[str, np.ndarray]:
    """``|disp|`` on a 2D slab through ``centre_site``, one array per field.

    ``fields`` maps a name (``hr``, ``sr2``, ``residual``, ``target``) to a
    whole-box ``(C, ng, ng, ng)`` field. The slab is a ``(2*half)`` square in the
    plane perpendicular to ``axis`` at the centre's coordinate on that axis, with
    periodic wrap, so a host near the box edge still lands in frame. Returns the
    displacement magnitude, the field-agnostic quantity the eye reads for "where
    is the structure."
    """
    cx, cy, cz = centre_site
    ax_c = (cx, cy, cz)[axis]
    keep = [a for a in range(3) if a != axis]
    c0, c1 = (cx, cy, cz)[keep[0]], (cx, cy, cz)[keep[1]]
    idx0 = (np.arange(c0 - half, c0 + half) % ng_hr)
    idx1 = (np.arange(c1 - half, c1 + half) % ng_hr)

    panels: Dict[str, np.ndarray] = {}
    for name, fld in fields.items():
        arr = np.asarray(fld)
        disp = arr[0:3]                                   # (3, ng, ng, ng)
        # take the axis plane, then the two kept-axis windows
        plane = np.take(disp, ax_c, axis=1 + axis)        # (3, ng, ng)
        sub = plane[:, idx0][:, :, idx1]                  # (3, 2half, 2half)
        panels[name] = np.sqrt(np.sum(sub.astype(np.float64) ** 2, axis=0))
    return panels
