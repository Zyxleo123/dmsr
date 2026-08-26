"""Pins the per-host affine-moment projector of docs/sr2_moment_constraint.md.

The checks are section 7 of that note: idempotence, that the forbidden affine
modes are killed exactly, that anharmonic substructure survives, that the
post-projection moments vanish, that footprints partition, and that the field
off every footprint is untouched. All field-only; no halo finder.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.features.lagrangian_host import LagrangianGrid
from cosmo_sr.features.moment_constraint import build_projector

# Small grid: 8^3 HR from 4^3 LR (upsample 2), one tile. Enough sites per host
# for a well-posed 4-parameter affine fit.
GRID = LagrangianGrid(ng_lr=4, ng_hr=8, tile_hr=8, boxsize_mpc_h=100.0)
BOX = GRID.boxsize_mpc_h


def _two_host_setup():
    """Two disjoint hosts on the 4^3 LR lattice, plus their HR geometry.

    Host 0 is a 2x2x2 LR block in one corner; host 1 a 2x2x2 block in the
    opposite corner. Centres and R_L are chosen so xi spans O(1).
    """
    host_index = np.full((4, 4, 4), -1, dtype=np.int32)
    host_index[0:2, 0:2, 0:2] = 0
    host_index[2:4, 2:4, 2:4] = 1

    # Periodic Lagrangian centre of each block, from LR cell centres.
    cell = BOX / GRID.ng_lr
    c0 = (np.array([0.5, 0.5, 0.5]) + 0.5) * cell   # centre of the 0:2 block
    c1 = (np.array([2.5, 2.5, 2.5]) + 0.5) * cell
    centres = np.stack([c0, c1])
    radii = np.array([1.5 * cell, 1.5 * cell])       # ~block half-size
    return host_index, centres, radii


def _projector(mode="affine"):
    host_index, centres, radii = _two_host_setup()
    return build_projector(GRID, host_index, centres, radii, mode=mode)


def _rng():
    return np.random.default_rng(0)


def _xi(blk):
    """The centred coordinate columns of a block's design matrix."""
    return blk.phi[:, 1:4]


# --------------------------------------------------------------------------
# 1. idempotence
# --------------------------------------------------------------------------

def test_projection_is_idempotent():
    proj = _projector()
    field = _rng().standard_normal((3, GRID.ng_hr ** 3))
    once = proj.apply(field)
    twice = proj.apply(once)
    assert np.allclose(once, twice, atol=1e-9)


# --------------------------------------------------------------------------
# 2. kills exactly the forbidden modes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode_name", ["translation", "dilation", "rotation", "shear"])
def test_affine_modes_are_removed_inside_footprints(mode_name):
    proj = _projector()
    field = np.zeros((3, GRID.ng_hr ** 3))
    for blk in proj.blocks:
        xi = _xi(blk)                                # (n, 3)
        if mode_name == "translation":
            d = np.tile(np.array([0.3, -0.2, 0.1]), (xi.shape[0], 1))
        elif mode_name == "dilation":
            d = 0.4 * xi                             # d = lambda * xi
        elif mode_name == "rotation":
            omega = np.array([0.0, 0.0, 0.5])
            d = np.cross(np.tile(omega, (xi.shape[0], 1)), xi)
        else:  # shear: symmetric traceless M
            m = np.array([[0.2, 0.1, 0.0],
                          [0.1, -0.2, 0.0],
                          [0.0, 0.0, 0.0]])
            d = xi @ m.T
        field[:, blk.sites] = d.T
    out = proj.apply(field)
    # every injected mode is affine, so it is removed to ~0 on the footprints.
    mask = proj.footprint_mask()
    assert np.max(np.abs(out[:, mask])) < 1e-9


def test_translation_only_mode_keeps_dilation():
    proj = _projector(mode="translation")
    geom = {b.row: b for b in _projector(mode="affine").blocks}  # xi source
    field = np.zeros((3, GRID.ng_hr ** 3))
    for blk in proj.blocks:
        field[:, blk.sites] = (0.4 * _xi(geom[blk.row])).T   # a pure dilation
    out = proj.apply(field)
    mask = proj.footprint_mask()
    # translation-only must NOT remove the dilation: energy survives.
    assert np.max(np.abs(out[:, mask])) > 1e-3
    # but it must still remove the mean (translation) per host.
    for blk in proj.blocks:
        assert np.allclose(out[:, blk.sites].mean(axis=1), 0.0, atol=1e-9)


# --------------------------------------------------------------------------
# 3. anharmonic substructure survives
# --------------------------------------------------------------------------

def test_substructure_survives_with_its_affine_part_removed():
    proj = _projector()
    field = np.zeros((3, GRID.ng_hr ** 3))
    for blk in proj.blocks:
        xi = _xi(blk)
        # a quadratic bump -- genuinely anharmonic in xi, cannot be affine.
        bump = (np.sum(xi ** 2, axis=1, keepdims=True)) * np.array([1.0, 0.0, 0.0])
        field[:, blk.sites] = bump.T
    out = proj.apply(field)
    mask = proj.footprint_mask()
    # the bump is not annihilated ...
    assert np.max(np.abs(out[:, mask])) > 1e-2
    # ... but its affine moments are gone (check 4, per host).
    for blk in proj.blocks:
        moments = proj.affine_moments(out, blk.row)
        assert np.max(np.abs(moments)) < 1e-8


# --------------------------------------------------------------------------
# 4. post-projection moments vanish
# --------------------------------------------------------------------------

def test_all_affine_moments_vanish_after_projection():
    proj = _projector()
    field = _rng().standard_normal((3, GRID.ng_hr ** 3))
    out = proj.apply(field)
    for blk in proj.blocks:
        assert np.max(np.abs(proj.affine_moments(out, blk.row))) < 1e-8


# --------------------------------------------------------------------------
# 5. footprints partition
# --------------------------------------------------------------------------

def test_footprints_are_disjoint_and_cover_the_bound_sites():
    host_index, centres, radii = _two_host_setup()
    proj = build_projector(GRID, host_index, centres, radii)
    # disjointness is enforced in the constructor; here confirm the HR mask
    # equals the HR broadcast of the bound LR sites.
    mask = proj.footprint_mask()
    bound_lr = np.flatnonzero(host_index.reshape(-1) >= 0)
    expected = np.concatenate([GRID.hr_children(int(s)) for s in bound_lr])
    got = np.flatnonzero(mask)
    assert np.array_equal(np.sort(got), np.sort(expected))


def test_overlapping_footprints_are_rejected():
    from cosmo_sr.features.moment_constraint import HostAffineBlock, MomentProjector
    sites = np.array([0, 1, 2], dtype=np.int64)
    phi = np.ones((3, 1))
    blk = HostAffineBlock(row=0, sites=sites, phi=phi, ginv=np.linalg.pinv(phi.T @ phi))
    dup = HostAffineBlock(row=1, sites=sites, phi=phi, ginv=blk.ginv)
    with pytest.raises(ValueError, match="overlap"):
        MomentProjector(GRID, [blk, dup])


# --------------------------------------------------------------------------
# 6. the field off every footprint is untouched
# --------------------------------------------------------------------------

def test_unbound_sites_pass_through_unchanged():
    proj = _projector()
    field = _rng().standard_normal((3, GRID.ng_hr ** 3))
    out = proj.apply(field)
    free = ~proj.footprint_mask()
    assert np.array_equal(out[:, free], field[:, free])


# --------------------------------------------------------------------------
# multi-block channels: displacement + velocity projected independently
# --------------------------------------------------------------------------

def test_six_channel_field_projects_each_triplet():
    proj = _projector()
    field = _rng().standard_normal((6, GRID.ng_hr ** 3))
    out = proj.apply(field)
    # each 3-block's per-host moments vanish; the two blocks are independent.
    for blk in proj.blocks:
        moments = proj.affine_moments(out, blk.row)   # (k, 6)
        assert np.max(np.abs(moments)) < 1e-8
    # velocity projection did not leak into displacement: a disp-only input
    # leaves velocity channels exactly zero.
    disp_only = np.zeros((6, GRID.ng_hr ** 3))
    disp_only[0:3] = field[0:3]
    out2 = proj.apply(disp_only)
    assert np.array_equal(out2[3:6], np.zeros((3, GRID.ng_hr ** 3)))


def test_accepts_spatial_shape():
    proj = _projector()
    field = _rng().standard_normal((3, GRID.ng_hr, GRID.ng_hr, GRID.ng_hr))
    out = proj.apply(field)
    assert out.shape == field.shape


# --------------------------------------------------------------------------
# diagnostics (src/cosmo_sr/features/moment_target_diag.py)
# --------------------------------------------------------------------------

def test_moment_rows_report_zero_residual_moment_after():
    from cosmo_sr.features.moment_target_diag import per_host_moment_rows
    proj = _projector()
    residual = _rng().standard_normal((6, GRID.ng_hr ** 3))
    target = proj.apply(residual)
    rows = per_host_moment_rows(proj, residual, target)
    assert rows, "expected one row per host"
    for r in rows:
        assert r.moment_norm_after < 1e-8 < r.moment_norm_before
        assert 0.0 <= r.affine_var_frac <= 1.0
        # a random field has substructure, so removal is partial, not total.
        assert r.rms_after < r.rms_before


def test_offfootprint_diff_is_zero_and_catches_a_leak():
    from cosmo_sr.features.moment_target_diag import offfootprint_max_abs_diff
    proj = _projector()
    residual = _rng().standard_normal((6, GRID.ng_hr ** 3))
    target = proj.apply(residual)
    assert offfootprint_max_abs_diff(proj, residual, target) == 0.0
    # perturb one free site: the check must see it.
    free = np.flatnonzero(~proj.footprint_mask())
    leaked = target.copy()
    leaked[0, free[0]] += 0.5
    assert offfootprint_max_abs_diff(proj, residual, leaked) == pytest.approx(0.5)


def test_slice_panels_shape_and_wrap():
    from cosmo_sr.features.moment_target_diag import host_slice_panels
    fields = {"residual": _rng().standard_normal((6, GRID.ng_hr, GRID.ng_hr, GRID.ng_hr))}
    half = 3
    panels = host_slice_panels(fields, (0, 0, 0), GRID.ng_hr, half=half, axis=2)
    assert panels["residual"].shape == (2 * half, 2 * half)
    assert np.all(panels["residual"] >= 0.0)   # it is a magnitude
