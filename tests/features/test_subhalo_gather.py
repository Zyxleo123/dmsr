"""Pins the auxiliary gather loss of cosmo_sr.features.subhalo_gather.

Four things can be silently wrong in a loss like this and none of them raise:

1. the statistic could be the wrong function (it must be soft_structure's
   compact-mass coordinate, not a re-derivation);
2. the window could sit in the wrong place -- the mapping from a Rockstar
   position in Mpc/h to a cell of the valid-centre deposit is three coordinate
   conventions deep, and an off-by-one there supervises the neighbouring cell;
3. the loss could be two-sided, which would let it ask for more contrast than
   HR has (the step-4 over-sharpening failure);
4. it could have no gradient into the displacement at all, or one that does not
   gather.

Field-only, tiny grids, no halo finder, no generator.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.eval.density import cic_density_valid_center, valid_center_bulk
from cosmo_sr.eval.particle_identity import OwnerIndex, build_owner_index
from cosmo_sr.eval.rockstar import HaloCatalog
from cosmo_sr.features.subhalo_gather import (
    GatherConfig, GatherTargets, attach_hr_reference, gather_loss,
    gather_statistics, region_coordinates, stack_tile_subhalos,
    subhalo_home_tiles, tile_subhalos,
)
from cosmo_sr.reward.soft_structure import SoftStructureConfig

SOFT = SoftStructureConfig(region_fraction=0.5, grid_mult=1)
CFG = GatherConfig.from_soft(SOFT)
CELLS_PER_UNIT = float(SOFT.dis_norm_kpc_h) / float(SOFT.cellsize_kpc_h)


def _delta_with_clump(g: int, cell, mass: float) -> torch.Tensor:
    """A uniform ``(1, 1, g, g, g)`` field with ``mass`` particles in one cell."""
    d = torch.zeros(1, 1, g, g, g)
    d[0, 0, cell[0], cell[1], cell[2]] = float(mass) - 1.0
    return d


def _targets(centre, sigma=1.0, hr_compact=1.0, hr_contrast=1.0) -> GatherTargets:
    c = torch.as_tensor(centre, dtype=torch.float32).reshape(1, -1, 3)
    s = c.shape[1]
    ones = torch.ones(1, s)
    return GatherTargets(
        centre=c, sigma=ones * float(sigma), mask=ones,
        hr_compact=ones * float(hr_compact), hr_contrast=ones * float(hr_contrast),
        hr_vbulk=torch.zeros(1, s, 3), hr_vdisp=ones * 300.0,
        half_width=3, num_p=torch.full((1, s), 100, dtype=torch.long),
        halo_id=torch.zeros(1, s, dtype=torch.long), tiles=[0])


# --------------------------------------------------------------------------
# 1. the statistic is soft_structure's coordinate, evaluated in a window
# --------------------------------------------------------------------------

def test_compact_statistic_matches_the_closed_form():
    """One dense cell exactly under the centre: the sum has a single term."""
    mass = 400.0
    d = _delta_with_clump(16, (8, 8, 8), mass)
    c, _ = gather_statistics(d, torch.tensor([[[8.0, 8.0, 8.0]]]),
                             torch.tensor([[1.0]]), 3, CFG)
    w = 1.0 / (1.0 + np.exp(-(np.log(mass) - np.log1p(CFG.compact_delta))
                            / CFG.tau_log))
    # kernel is 1 at zero offset; the rest of the window is uniform (delta = 0)
    # and contributes sigmoid(-13) ~ 2e-6 per cell.
    assert c.item() == pytest.approx(w * mass, rel=2e-3)


def test_blurring_the_same_mass_lowers_the_statistic():
    """The property MSE lacks: spreading mass out is the WORST move, not the best."""
    mass = 400.0
    sharp = _delta_with_clump(16, (8, 8, 8), mass)
    blur = torch.zeros(1, 1, 16, 16, 16)
    blur[0, 0, 7:10, 7:10, 7:10] = mass / 27.0 - 1.0     # same total mass
    args = (torch.tensor([[[8.0, 8.0, 8.0]]]), torch.tensor([[1.0]]), 3, CFG)
    c_sharp, _ = gather_statistics(sharp, *args)
    c_blur, _ = gather_statistics(blur, *args)
    assert float(c_sharp) > 20.0 * float(c_blur)


def test_kernel_falls_off_with_distance():
    d = _delta_with_clump(16, (8, 8, 8), 400.0)
    at, _ = gather_statistics(d, torch.tensor([[[8.0, 8.0, 8.0]]]),
                              torch.tensor([[1.0]]), 3, CFG)
    off, _ = gather_statistics(d, torch.tensor([[[10.0, 8.0, 8.0]]]),
                               torch.tensor([[1.0]]), 3, CFG)
    assert float(off) == pytest.approx(float(at) * np.exp(-2.0), rel=1e-3)


def test_sub_cell_motion_changes_the_statistic_smoothly():
    """The kernel reads the fractional offset, so the gradient is not stair-stepped."""
    d = _delta_with_clump(16, (8, 8, 8), 400.0)
    vals = [float(gather_statistics(d, torch.tensor([[[8.0 + f, 8.0, 8.0]]]),
                                    torch.tensor([[1.0]]), 3, CFG)[0])
            for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


# --------------------------------------------------------------------------
# 2. the window sits where Rockstar says the subhalo is
# --------------------------------------------------------------------------

def test_region_coordinates_land_on_the_cell_the_deposit_uses():
    """The decisive geometry check, end to end through the real CIC.

    Move ONE particle by a known displacement, ask ``region_coordinates`` where
    its new absolute position lands in the deposit grid, and confirm the deposit
    put its mass in exactly that cell. If the two conventions disagree by a cell
    the loss supervises the wrong window and nothing downstream would notice.
    """
    n, tile_id = 16, 5
    ng_hr, tile_hr = 64, 16          # a 4^3 tile grid, tile 5 = (0, 1, 1)
    soft = SoftStructureConfig(region_fraction=0.5, grid_mult=1, ng_hr=ng_hr,
                               boxsize_mpc_h=100.0)
    region = soft.region_of(n)
    cell_mpc = soft.boxsize_mpc_h / ng_hr

    disp = torch.zeros(1, 3, n, n, n)
    site = (6, 7, 9)
    shift_cells = np.array([2.0, -1.0, 0.5])
    disp[0, :, site[0], site[1], site[2]] = torch.tensor(
        (shift_cells / (float(soft.dis_norm_kpc_h) / float(soft.cellsize_kpc_h)))
        .astype(np.float32))

    bulk = valid_center_bulk(disp, soft.cellsize_kpc_h, soft.dis_norm_kpc_h)
    assert torch.allclose(bulk, torch.zeros_like(bulk))    # one particle in 16^3

    corner = np.array([tile_id // 16, (tile_id // 4) % 4, tile_id % 4]) * tile_hr
    q = np.array(site, dtype=float) + 0.5
    pos_mpc = (corner + q + shift_cells) * cell_mpc

    u = region_coordinates(pos_mpc, tile_id, bulk[0].numpy(), ng_hr=ng_hr,
                           tile_hr=tile_hr, boxsize_mpc_h=soft.boxsize_mpc_h,
                           region=region, grid_mult=1)[0]

    d = cic_density_valid_center(disp, soft.cellsize_kpc_h, soft.dis_norm_kpc_h,
                                 region=region)
    d0 = cic_density_valid_center(torch.zeros_like(disp), soft.cellsize_kpc_h,
                                  soft.dis_norm_kpc_h, region=region)
    gained = (d - d0)[0, 0]
    # The particle's mass is split between floor(u) and floor(u)+1 per axis; the
    # corner holding the largest share is the one region_coordinates points at.
    best = np.unravel_index(int(torch.argmax(gained)), gained.shape)
    assert np.all(np.abs(np.array(best) - u) <= 1.0 + 1e-6)
    assert float(gained[best]) > 0.0


def test_periodic_image_nearest_the_tile_is_chosen():
    """A subhalo just across the box face belongs to the tile, not 500 cells away."""
    u = region_coordinates(np.array([[99.9, 0.1, 0.1]]), 0, np.zeros(3),
                           ng_hr=512, tile_hr=64, boxsize_mpc_h=100.0, region=32)
    assert u[0, 0] < 0.0 and u[0, 0] > -20.0


# --------------------------------------------------------------------------
# 3. the loss is one-sided and per-subhalo normalised
# --------------------------------------------------------------------------

def test_loss_is_zero_once_hr_is_matched_and_never_asks_for_more():
    d = _delta_with_clump(16, (8, 8, 8), 400.0)
    c, p = gather_statistics(d, torch.tensor([[[8.0, 8.0, 8.0]]]),
                             torch.tensor([[1.0]]), 3, CFG)
    exact = _targets([[8.0, 8.0, 8.0]], hr_compact=float(c), hr_contrast=float(p))
    over = _targets([[8.0, 8.0, 8.0]], hr_compact=0.5 * float(c),
                    hr_contrast=0.5 * float(p))
    under = _targets([[8.0, 8.0, 8.0]], hr_compact=2.0 * float(c),
                     hr_contrast=2.0 * float(p))
    assert float(gather_loss(d, exact, CFG)[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(gather_loss(d, over, CFG)[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(gather_loss(d, under, CFG)[0]) > 0.1


def test_two_subhalos_of_different_mass_carry_the_same_weight():
    """The ratio normalisation is section 4.2's per-subhalo gradient equalisation."""
    d = torch.zeros(1, 1, 16, 16, 16)
    big, small = (5, 8, 8), (11, 8, 8)
    d[0, 0, big[0], big[1], big[2]] = 2000.0
    d[0, 0, small[0], small[1], small[2]] = 200.0
    c, _ = gather_statistics(d, torch.tensor([[[5.0, 8.0, 8.0], [11.0, 8.0, 8.0]]]),
                             torch.tensor([[1.0, 1.0]]), 3, CFG)
    # Both are at half of their own HR reference -> identical per-target terms.
    t = GatherTargets(
        centre=torch.tensor([[[5.0, 8.0, 8.0], [11.0, 8.0, 8.0]]]),
        sigma=torch.ones(1, 2), mask=torch.ones(1, 2),
        hr_compact=c * 2.0, hr_contrast=torch.ones(1, 2) * 1e9,
        hr_vbulk=torch.zeros(1, 2, 3), hr_vdisp=torch.ones(1, 2) * 300.0,
        half_width=3, num_p=torch.tensor([[2000, 200]]),
        halo_id=torch.zeros(1, 2, dtype=torch.long), tiles=[0])
    cfg = GatherConfig.from_soft(SOFT, w_contrast=0.0)
    loss, diag = gather_loss(d, t, cfg)
    assert float(loss) == pytest.approx(0.25, rel=1e-3)
    assert diag["compact_ratio"] == pytest.approx(0.5, rel=1e-3)


def test_targets_hr_cannot_show_are_masked_out():
    d = torch.zeros(1, 1, 16, 16, 16)          # uniform: no compact mass anywhere
    t = _targets([[8.0, 8.0, 8.0]])
    t = attach_hr_reference(t, d, CFG)
    assert float(t.mask.sum()) == 0.0
    assert gather_loss(d, t, CFG)[1]["n_targets"] == 0


# --------------------------------------------------------------------------
# 4. the gradient reaches the particles, and it gathers them
# --------------------------------------------------------------------------

def test_gradient_step_gathers_particles_toward_the_target():
    """One descent step must move nearby particles INWARD and raise the statistic.

    The whole claim of the module in one assertion: the loss is not just
    differentiable, its descent direction concentrates local material at the true
    subhalo's position.
    """
    n = 16
    soft = SoftStructureConfig(region_fraction=0.5, grid_mult=1)
    region = soft.region_of(n)
    disp = torch.zeros(1, 3, n, n, n, requires_grad=True)
    centre = torch.tensor([[[4.0, 4.0, 4.0]]])
    t = GatherTargets(centre=centre, sigma=torch.tensor([[2.0]]),
                      mask=torch.ones(1, 1), hr_compact=torch.tensor([[300.0]]),
                      hr_contrast=torch.tensor([[300.0]]),
                      hr_vbulk=torch.zeros(1, 1, 3),
                      hr_vdisp=torch.tensor([[300.0]]), half_width=5,
                      num_p=torch.tensor([[300]]),
                      halo_id=torch.zeros(1, 1, dtype=torch.long), tiles=[0])

    def stats(d):
        dens = cic_density_valid_center(d, soft.cellsize_kpc_h,
                                        soft.dis_norm_kpc_h, region=region)
        return gather_loss(dens, t, CFG)

    loss, diag0 = stats(disp)
    loss.backward()
    g = disp.grad
    assert torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0

    lat = torch.arange(n, dtype=torch.float32) + 0.5
    q = torch.stack(torch.meshgrid(lat, lat, lat, indexing="ij"))
    origin = (n - region) / 2.0

    def radius(d):
        u = (q + d[0].detach() * CELLS_PER_UNIT) - origin
        r = (u - centre.view(3, 1, 1, 1)).pow(2).sum(0).sqrt()
        near = ((q - origin) - centre.view(3, 1, 1, 1)).pow(2).sum(0).sqrt() < 3.0
        return float(r[near].mean()), int(near.sum())

    r0, n_near = radius(disp)
    # Displacements are in the on-disk normalised units (one unit is ~31
    # HR cells here), so a step of half a cell is `0.5 / CELLS_PER_UNIT`.
    step = 0.5 / CELLS_PER_UNIT
    stepped = (disp - step * g / g.abs().max()).detach()
    r1, _ = radius(stepped)
    _, diag1 = stats(stepped)

    assert n_near > 20
    assert r1 < r0                                   # they moved inward
    assert diag1["compact_ratio"] > diag0["compact_ratio"]   # into a clump


# --------------------------------------------------------------------------
# 5. selection: which subhalo belongs to which tile
# --------------------------------------------------------------------------

def _toy_catalog_and_owner(ng=16, tile=8):
    """Two subhalos: one clean inside tile 0, one straddling tiles 0 and 1."""
    ids = np.array([10, 11, 12], dtype=np.int64)
    parent = np.array([-1, 10, 10], dtype=np.int64)
    num_p = np.array([4000, 200, 200], dtype=np.int64)
    pos = np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0], [12.0, 12.0, 12.0]])
    cat = HaloCatalog(ids=ids, parent_ids=parent,
                      mvir=np.array([1e14, 1e12, 1e12]),
                      rvir=np.array([1000.0, 200.0, 200.0]),
                      vmax=np.zeros(3), pos=pos, vel=np.zeros((3, 3)),
                      num_p=num_p)
    owner = np.full(ng ** 3, -1, dtype=np.int64)

    def flat(x, y, z):
        return (x * ng + y) * ng + z

    for k in range(40):                       # halo 11: all inside tile 0
        owner[flat(1, 1, k % 8)] = 11
        owner[flat(2, 2, k % 8)] = 11
    for k in range(8):                        # halo 12: half in tile 0, half in 1
        owner[flat(3, 3, k)] = 12
        owner[flat(3, 3, 8 + k)] = 12
    return cat, build_owner_index(owner)


def test_top_level_selects_hosts_and_default_selects_subhalos():
    """The one line the host preservation guard turns on.

    ``top_level=False`` (default, every prior caller) selects subhalos
    (``parent_ids >= 0``); ``top_level=True`` selects the host that owns them
    (``parent_ids < 0``). The home-tile plurality is computed identically, so the
    switch is population-only and nothing downstream needs to know which it got.
    """
    ng, tile = 16, 8
    ids = np.array([10, 11], dtype=np.int64)          # 10 host, 11 its subhalo
    cat = HaloCatalog(
        ids=ids, parent_ids=np.array([-1, 10], dtype=np.int64),
        mvir=np.array([1e14, 1e12]), rvir=np.array([1000.0, 200.0]),
        vmax=np.zeros(2), pos=np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]]),
        vel=np.zeros((2, 3)), num_p=np.array([4000, 200], dtype=np.int64))
    owner = np.full(ng ** 3, -1, dtype=np.int64)

    def flat(x, y, z):
        return (x * ng + y) * ng + z

    for k in range(8):
        owner[flat(0, 0, k)] = 10          # host 10 owns its own smooth component
        owner[flat(1, 1, k)] = 11          # subhalo 11's sites, same tile 0
    oidx = build_owner_index(owner)

    sub = subhalo_home_tiles(cat, oidx, ng_hr=ng, tile_hr=tile, min_num_p=50)
    assert set(sub["halo_id"].tolist()) == {11}        # subhalos only, default

    host = subhalo_home_tiles(cat, oidx, ng_hr=ng, tile_hr=tile, min_num_p=50,
                              top_level=True)
    assert set(host["halo_id"].tolist()) == {10}       # hosts only, top_level
    assert int(host["tile"][0]) == 0


def test_home_tile_is_the_plurality_tile_and_purity_is_reported():
    cat, oidx = _toy_catalog_and_owner()
    home = subhalo_home_tiles(cat, oidx, ng_hr=16, tile_hr=8, min_num_p=50)
    assert set(home["halo_id"].tolist()) == {11, 12}
    by_id = {int(h): i for i, h in enumerate(home["halo_id"])}
    assert int(home["tile"][by_id[11]]) == 0
    assert home["purity"][by_id[11]] == pytest.approx(1.0)
    assert home["purity"][by_id[12]] == pytest.approx(0.5)   # 8 of 16 sites


def test_a_straddling_subhalo_is_dropped_by_min_purity():
    cat, oidx = _toy_catalog_and_owner()
    home = subhalo_home_tiles(cat, oidx, ng_hr=16, tile_hr=8, min_num_p=50)
    soft = SoftStructureConfig(region_fraction=0.5, grid_mult=1, ng_hr=16,
                               boxsize_mpc_h=100.0)
    cfg = GatherConfig.from_soft(soft, min_purity=0.9, sigma_floor_cells=0.5,
                                 radius_factor=1.0)
    kept = tile_subhalos(cat, home, 0, np.zeros(3), cfg, soft, tile_hr=8)
    assert kept.n <= 1
    assert 12 not in kept.halo_id.tolist()


def test_windows_leaving_the_scored_cube_are_dropped():
    cat, oidx = _toy_catalog_and_owner()
    home = subhalo_home_tiles(cat, oidx, ng_hr=16, tile_hr=8, min_num_p=50)
    soft = SoftStructureConfig(region_fraction=0.5, grid_mult=1, ng_hr=16,
                               boxsize_mpc_h=100.0)
    # Halo 11 sits at 10 Mpc/h of a 100 Mpc/h box on a 16^3 lattice = cell 1.6,
    # far outside tile 0's scored cube, so no window survives.
    cfg = GatherConfig.from_soft(soft, min_purity=0.0)
    assert tile_subhalos(cat, home, 0, np.zeros(3), cfg, soft, tile_hr=8).n == 0


def test_stacking_pads_tiles_of_different_length():
    cat, oidx = _toy_catalog_and_owner()
    home = subhalo_home_tiles(cat, oidx, ng_hr=16, tile_hr=8, min_num_p=50)
    soft = SoftStructureConfig(region_fraction=0.5, grid_mult=1, ng_hr=16,
                               boxsize_mpc_h=100.0)
    cfg = GatherConfig.from_soft(soft, min_purity=0.0)
    a = tile_subhalos(cat, home, 0, np.zeros(3), cfg, soft, tile_hr=8)
    b = tile_subhalos(cat, home, 1, np.zeros(3), cfg, soft, tile_hr=8)
    t = stack_tile_subhalos([a, b])
    assert t.centre.shape[0] == 2 and t.mask.shape[0] == 2
    assert t.sigma.min() > 0.0        # padded slots must not divide by zero
    assert t.select([1]).centre.shape[0] == 1


# --------------------------------------------------------------------------
# 6. velocity: the half Rockstar actually links on
# --------------------------------------------------------------------------
VEL_NORM = 313.42210244571896      # PhaseSpaceConfig.vel_norm_km_s at z = 0


def _rest_field(n=16, vel_kms=None):
    """Particles on the lattice (disp = 0) with a prescribed velocity field."""
    f = torch.zeros(1, 6, n, n, n)
    if vel_kms is not None:
        f[0, 3:6] = torch.as_tensor(vel_kms, dtype=torch.float32) / VEL_NORM
    return f


def _dep(field):
    from cosmo_sr.features.subhalo_gather import deposit_for_gather
    return deposit_for_gather(field, SOFT)


def test_velocity_statistics_recover_a_uniform_bulk_flow():
    """A clump all moving together: bulk = the flow, dispersion ~ 0."""
    from cosmo_sr.features.subhalo_gather import velocity_statistics
    v = torch.zeros(3, 16, 16, 16)
    v[0] = 250.0                                   # 250 km/s along x, everywhere
    dep = _dep(_rest_field(vel_kms=v))
    centre = torch.tensor([[[4.0, 4.0, 4.0]]])
    vb, vd = velocity_statistics(dep, centre, torch.tensor([[2.0]]), 3)
    assert float(vb[0, 0, 0]) == pytest.approx(250.0, rel=1e-3)
    assert float(vb[0, 0, 1]) == pytest.approx(0.0, abs=1e-2)
    assert float(vd[0, 0]) < 1.0                   # coherent flow is not dispersion


def test_dispersion_includes_shear_across_cells_not_just_within_them():
    """The law of total variance, and why it matters.

    Averaging each cell's own dispersion would report ~0 for a clump whose two
    halves stream past each other at 400 km/s -- which is exactly the
    configuration a phase-space finder rejects. The between-cell scatter of the
    cell means has to be in the number.
    """
    from cosmo_sr.features.subhalo_gather import velocity_statistics
    v = torch.zeros(3, 16, 16, 16)
    v[0, :8] = 200.0
    v[0, 8:] = -200.0                              # two halves, opposite streams
    dep = _dep(_rest_field(vel_kms=v))
    centre = torch.tensor([[[4.0, 4.0, 4.0]]])     # window straddles the split
    vb, vd = velocity_statistics(dep, centre, torch.tensor([[3.0]]), 4)
    assert abs(float(vb[0, 0, 0])) < 60.0          # the streams nearly cancel
    assert float(vd[0, 0]) > 150.0                 # but the object is not cold


def test_velocity_terms_are_two_sided():
    """Too hot and too cold must both be penalised: there is no safe direction."""
    from cosmo_sr.features.subhalo_gather import gather_loss
    cfg = GatherConfig.from_soft(SOFT, w_contrast=0.0, w_vbulk=0.0)
    centre = torch.tensor([[[4.0, 4.0, 4.0]]])

    def loss_at(scale, hr_vdisp):
        v = torch.zeros(3, 16, 16, 16)
        v[0, :8], v[0, 8:] = scale, -scale
        dep = _dep(_rest_field(vel_kms=v))
        t = GatherTargets(centre=centre, sigma=torch.tensor([[3.0]]),
                          mask=torch.ones(1, 1), hr_compact=torch.tensor([[1e-9]]),
                          hr_contrast=torch.tensor([[1e-9]]),
                          hr_vbulk=torch.zeros(1, 1, 3),
                          hr_vdisp=torch.tensor([[float(hr_vdisp)]]), half_width=4,
                          num_p=torch.tensor([[300]]),
                          halo_id=torch.zeros(1, 1, dtype=torch.long), tiles=[0])
        return float(gather_loss(dep, t, cfg)[0])

    # HR wants ~200 km/s of dispersion; 50 and 800 are both wrong.
    assert loss_at(50.0, 200.0) > 0.1
    assert loss_at(800.0, 200.0) > 0.1
    assert loss_at(200.0, 200.0) < min(loss_at(50.0, 200.0), loss_at(800.0, 200.0))


def test_gradient_reaches_the_velocity_channels():
    """The point of the whole addition: before it, dL/d(vel) was identically 0."""
    from cosmo_sr.features.subhalo_gather import deposit_for_gather, gather_loss
    v = torch.zeros(3, 16, 16, 16)
    v[0] = 100.0
    field = _rest_field(vel_kms=v).requires_grad_(True)
    dep = deposit_for_gather(field, SOFT)
    t = GatherTargets(centre=torch.tensor([[[4.0, 4.0, 4.0]]]),
                      sigma=torch.tensor([[2.0]]), mask=torch.ones(1, 1),
                      hr_compact=torch.tensor([[1e-9]]),
                      hr_contrast=torch.tensor([[1e-9]]),
                      hr_vbulk=torch.zeros(1, 1, 3),
                      hr_vdisp=torch.tensor([[400.0]]), half_width=3,
                      num_p=torch.tensor([[300]]),
                      halo_id=torch.zeros(1, 1, dtype=torch.long), tiles=[0])
    gather_loss(dep, t, GatherConfig.from_soft(SOFT))[0].backward()
    assert torch.isfinite(field.grad).all()
    assert float(field.grad[0, 3:6].abs().sum()) > 0.0


def test_empty_window_does_not_nan_the_backward():
    """sqrt at exactly zero has an infinite derivative -- the +1e-8 guard.

    Silent in the forward pass and fatal in the backward one; this is the same
    failure the arm-B/C dispersions hit in reward/phase_space.py.
    """
    from cosmo_sr.features.subhalo_gather import deposit_for_gather, gather_loss
    field = _rest_field().requires_grad_(True)      # every velocity exactly zero
    dep = deposit_for_gather(field, SOFT)
    t = GatherTargets(centre=torch.tensor([[[4.0, 4.0, 4.0]]]),
                      sigma=torch.tensor([[2.0]]), mask=torch.ones(1, 1),
                      hr_compact=torch.tensor([[1e-9]]),
                      hr_contrast=torch.tensor([[1e-9]]),
                      hr_vbulk=torch.zeros(1, 1, 3),
                      hr_vdisp=torch.tensor([[300.0]]), half_width=3,
                      num_p=torch.tensor([[300]]),
                      halo_id=torch.zeros(1, 1, dtype=torch.long), tiles=[0])
    loss, diag = gather_loss(dep, t, GatherConfig.from_soft(SOFT))
    assert np.isfinite(float(loss.detach()))
    loss.backward()
    assert torch.isfinite(field.grad).all()


def test_deposit_delta_matches_the_density_only_route():
    """One CIC pass for both halves: `delta` here must equal density_from_disp's."""
    from cosmo_sr.eval.density import valid_center_bulk
    from cosmo_sr.features.subhalo_gather import deposit_for_gather
    from cosmo_sr.reward.soft_structure import density_from_disp
    g = torch.Generator().manual_seed(5)
    field = torch.randn(1, 6, 16, 16, 16, generator=g) * 0.02
    bulk = valid_center_bulk(field[:, 0:3], SOFT.cellsize_kpc_h, SOFT.dis_norm_kpc_h)
    dep = deposit_for_gather(field, SOFT, bulk=bulk)
    ref = density_from_disp(field[:, 0:3], SOFT, bulk=bulk)
    assert torch.allclose(dep.delta, ref, atol=1e-5)


# --------------------------------------------------------------------------
# 7. defending the field the objective is blind to
# --------------------------------------------------------------------------
def test_outside_weight_map_is_the_complement_of_the_windows():
    from cosmo_sr.features.subhalo_gather import outside_weight_map
    centre = torch.tensor([[[8.0, 8.0, 8.0]]])
    w = outside_weight_map(centre, torch.tensor([[1.5]]), 4, 16,
                           mask=torch.ones(1, 1))
    assert w.shape == (1, 1, 16, 16, 16)
    assert float(w[0, 0, 8, 8, 8]) == pytest.approx(0.0, abs=1e-5)   # under it
    assert float(w[0, 0, 0, 0, 0]) == pytest.approx(1.0, abs=1e-5)   # far away
    assert 0.0 <= float(w.min()) and float(w.max()) <= 1.0


def test_preserve_term_is_blind_inside_and_penalises_blurring_outside():
    """The failure it exists to catch: peaks sharpened at the targets while the
    rest of the field goes smooth."""
    from cosmo_sr.features.subhalo_gather import (
        outside_weight_map, preserve_loss, preserve_statistic,
    )
    g = 16
    rng = np.random.default_rng(0)
    peaky = torch.from_numpy(rng.random((1, 1, g, g, g)).astype(np.float32)) * 200.0
    centre = torch.tensor([[[8.0, 8.0, 8.0]]])
    ow = outside_weight_map(centre, torch.tensor([[1.5]]), 4, g,
                            mask=torch.ones(1, 1))
    smooth = _blur(peaky)

    t = _targets([[8.0, 8.0, 8.0]])
    t.outside_w = ow
    t.frozen_preserve = preserve_statistic(peaky, ow, CFG)

    same, r_same = preserve_loss(peaky, t, CFG)
    worse, r_worse = preserve_loss(smooth, t, CFG)
    assert float(same) == pytest.approx(0.0, abs=1e-6) and r_same == pytest.approx(1.0, rel=1e-4)
    assert r_worse < 0.9 and float(worse) > 0.0


def _blur(x):
    import torch.nn.functional as F
    return F.avg_pool3d(F.pad(x, (1,) * 6, mode="replicate"), 3, stride=1)


def test_preserve_term_never_penalises_improvement():
    """Hinged: gaining structure away from the targets must be free."""
    from cosmo_sr.features.subhalo_gather import (
        outside_weight_map, preserve_loss, preserve_statistic,
    )
    g = 16
    base = torch.full((1, 1, g, g, g), 5.0)
    ow = outside_weight_map(torch.tensor([[[8.0, 8.0, 8.0]]]),
                            torch.tensor([[1.5]]), 4, g, mask=torch.ones(1, 1))
    t = _targets([[8.0, 8.0, 8.0]])
    t.outside_w = ow
    t.frozen_preserve = preserve_statistic(base, ow, CFG)
    sharper = base.clone()
    sharper[0, 0, ::4, ::4, ::4] = 500.0            # new peaks, away from centre
    loss, ratio = preserve_loss(sharper, t, CFG)
    assert ratio > 1.0
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# 6. occupancy: the live fraction against an arbitrary trained-tile set
#
# `gather_coverage_curve.py` reads the ceiling off this, so the two things it
# must not do are (a) disagree with the home tile/purity the loss selects with
# and (b) mis-align its rows with the per-subhalo arrays.
# --------------------------------------------------------------------------

def test_occupancy_rows_align_and_reproduce_purity_and_totals():
    cat, oidx = _toy_catalog_and_owner()
    home = subhalo_home_tiles(cat, oidx, ng_hr=16, tile_hr=8, min_num_p=50,
                              return_occupancy=True)
    n = home["halo_id"].size
    # every occupancy entry points at a real row, and the counts add back up
    assert home["occ_row"].min() >= 0 and home["occ_row"].max() < n
    tot = np.bincount(home["occ_row"], weights=home["occ_count"], minlength=n)
    assert np.array_equal(tot.astype(np.int64), home["n_sites"])
    # the home tile's own count is the purity numerator
    for i in range(n):
        m = (home["occ_row"] == i) & (home["occ_tile"] == home["tile"][i])
        assert home["occ_count"][m].sum() == pytest.approx(
            home["purity"][i] * home["n_sites"][i])


def test_live_fraction_matches_the_selection_it_stands_in_for():
    cat, oidx = _toy_catalog_and_owner()
    home = subhalo_home_tiles(cat, oidx, ng_hr=16, tile_hr=8, min_num_p=50,
                              return_occupancy=True)
    by_id = {int(h): i for i, h in enumerate(home["halo_id"])}

    def live(tiles):
        m = np.isin(home["occ_tile"], np.asarray(tiles))
        inside = np.bincount(home["occ_row"][m],
                             weights=home["occ_count"][m].astype(np.float64),
                             minlength=home["n_sites"].size)
        return inside / home["n_sites"]

    # halo 11 is wholly inside tile 0; halo 12 is split 0/1 by construction, so
    # the live fraction is exactly what widening the tile set buys.
    f0 = live([0])
    assert f0[by_id[11]] == pytest.approx(1.0)
    assert f0[by_id[12]] == pytest.approx(0.5)
    f01 = live([0, 1])
    assert f01[by_id[12]] == pytest.approx(1.0)
    assert live([7])[by_id[11]] == pytest.approx(0.0)


def test_occupancy_is_optional_and_off_by_default():
    cat, oidx = _toy_catalog_and_owner()
    home = subhalo_home_tiles(cat, oidx, ng_hr=16, tile_hr=8, min_num_p=50)
    assert "occ_row" not in home
    empty = subhalo_home_tiles(cat, oidx, ng_hr=16, tile_hr=8,
                               min_num_p=10 ** 9, return_occupancy=True)
    assert empty["occ_row"].size == 0 and empty["n_sites"].size == 0
