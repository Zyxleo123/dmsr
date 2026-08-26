"""In-loop Rockstar eval on the cluster host, for the substructure module.

The trainer's field metrics (loss, conditional spread) cannot say whether the
generated ``d`` makes *bound* subhalos -- that is a nonlinear functional of the
output and only Rockstar decides it (``docs/sr2_substructure_module.md`` open
risk 1). The full-box gate is a 12 h CPU job, far too heavy to run in the loop.

This module gives a cheap physical signal every few thousand steps: sample ``d``
only on the tiles of the most massive host (set8's id 717, log Mvir 14.84 -- the
host shown in the moment-target figure), apply that host's ``Pi`` exactly (its
full footprint fits inside the region, so this is the real projector, not a
partial one), add to the frozen SR2, and run the frozen Rockstar on a Lagrangian
cube around the host as a mini-box. The frozen SR2 (the deficit baseline) and the
HR truth are run through the identical crop once and cached, so each eval reports

    rs_cand_sub / rs_base_sub / rs_hr_sub     (subhalo counts in the cube)
    rs_sub_ratio = cand / hr                  (the gate quantity, 0.07 -> 0.4+)

This is a *region diagnostic on one host under a mini-box*, not the full-box gate:
the crop uses periodic wrap on a non-periodic cube and Rockstar's softening was
tuned for 100 Mpc/h, so absolute counts carry an edge/scale bias. Candidate, base
and HR get the identical treatment, so the *trend* and the cand/base gap are the
signal; the reassembled-box gate (submit_substructure.sh) stays the verdict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..features.moment_constraint import build_projector
from . import substructure_data as sd


def host_row_of(feat, host_id: Optional[int] = None) -> int:
    """Table row of ``host_id`` (e.g. 717), or the most massive host."""
    if host_id is not None:
        r = feat.table.row_of(int(host_id))
        if r >= 0:
            return r
    return int(np.argmax(feat.table.mvir))


def _host_tiles(feat, row: int):
    return [int(t) for t in np.flatnonzero(feat.table.tile_frac[row] > 0)]


def _crop_around(field: np.ndarray, centre_site, m: int) -> np.ndarray:
    """``(C, m, m, m)`` cube centred on ``centre_site`` with periodic wrap."""
    sh = tuple(m // 2 - int(c) for c in centre_site)
    rolled = np.roll(field, shift=sh, axis=(1, 2, 3))
    return np.ascontiguousarray(rolled[:, :m, :m, :m])


@torch.no_grad()
def _sample_host_d(model, boxes: sd.SubstructureBoxes, feat, row: int,
                   n_steps: int, device, seed: int = 12345) -> np.ndarray:
    """``(6,512^3)`` d, zero except the host's tiles, projected by the host's Pi."""
    d_box = np.zeros((6, sd.NG_HR, sd.NG_HR, sd.NG_HR), dtype=np.float32)
    for t in _host_tiles(feat, row):
        tt = boxes.tile_tensors(t, device)
        g = torch.Generator(device=device.type).manual_seed(seed + t)
        d_norm = sd.integrate_tile(model, tt["x_in"][None], tt["context"][None],
                                   n_steps, generator=g)[0]
        d = sd.apply_scale(d_norm, tt["s_disp"], tt["s_vel"], undo=True)
        ix, iy, iz = sd.tile_coord(t)
        hx, hy, hz = sd.hr_block(ix, iy, iz)
        d_box[:, hx, hy, hz] = d.float().cpu().numpy()
    proj = build_projector(feat.grid, feat.host_index, feat.table.center_lag,
                           feat.table.r_lag_mpc_h, rows=[row])
    return proj.apply(d_box)


def _rockstar_counts(field_crop: np.ndarray, out_dir: Path, tag: str,
                     box_kpc: float) -> Dict[str, int]:
    from ..eval.rockstar import run_rockstar_on_field

    cat = run_rockstar_on_field(field_crop, out_dir, tag=tag,
                                boxsize_kpc_h=box_kpc, overwrite=True)
    return {"n_sub": int(cat.subhalos().n), "n_host": int(cat.hosts().n),
            "n_halo": int(cat.n)}


@torch.no_grad()
def region_rockstar_eval(model, boxes: sd.SubstructureBoxes, feat, hr_field,
                         row: int, *, work_dir: Path, cache: Dict[str, Any],
                         region_sites: int = 192, n_steps: int = 20,
                         device) -> Dict[str, float]:
    """Candidate vs frozen-SR2-base vs HR subhalo counts on the host's cube.

    ``cache`` persists the base/HR counts (they never change) across evals; pass
    the same dict every call. Returns ``{}`` and logs nothing on any failure
    (missing Rockstar binary, a crashed run) -- an eval must never kill training.
    """
    try:
        work_dir = Path(work_dir)
        cell_mpc = feat.grid.boxsize_mpc_h / feat.grid.ng_hr
        centre_mpc = feat.table.center_lag[row]
        centre_site = tuple(int(np.floor(c / cell_mpc)) % feat.grid.ng_hr
                            for c in centre_mpc)
        m = int(region_sites)
        box_kpc = m * cell_mpc * 1000.0

        # Candidate: SR2 + Pi(d) on the host's tiles, cropped to the cube.
        d_box = _sample_host_d(model, boxes, feat, row, n_steps, device)
        final = np.asarray(boxes.sr2, dtype=np.float32) + d_box
        cand = _rockstar_counts(_crop_around(final, centre_site, m),
                                work_dir / "cand", "cand", box_kpc)

        # Base (frozen SR2, no module) and HR truth: identical crop, cached once.
        if "base" not in cache:
            cache["base"] = _rockstar_counts(
                _crop_around(np.asarray(boxes.sr2, dtype=np.float32),
                             centre_site, m), work_dir / "base", "base", box_kpc)
        if "hr" not in cache:
            cache["hr"] = _rockstar_counts(
                _crop_around(np.asarray(hr_field, dtype=np.float32),
                             centre_site, m), work_dir / "hr", "hr", box_kpc)

        hr_sub = max(cache["hr"]["n_sub"], 1)
        base = cache["base"]
        return {
            "rs_cand_sub": float(cand["n_sub"]),
            "rs_base_sub": float(base["n_sub"]),
            "rs_hr_sub": float(cache["hr"]["n_sub"]),
            "rs_cand_host": float(cand["n_host"]),
            "rs_sub_ratio": float(cand["n_sub"]) / hr_sub,          # gate quantity
            "rs_base_ratio": float(base["n_sub"]) / hr_sub,         # the deficit
        }
    except Exception as e:  # pragma: no cover - Rockstar/env issues
        print(f"[rockstar-eval] skipped ({type(e).__name__}: {e})", flush=True)
        return {}
