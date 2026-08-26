"""Per-host, per-tile HR/SR2 subhalo counts (viewer panel 7).

The whole point of this join is that it must not count a *neighbouring* host's
substructure, and that both sides must be counted over the same Lagrangian
footprint. Both are index bookkeeping over the owner array, which is invisible
in the rendered picture and easy to get wrong, so it is checked here against
hand-built catalogs where the right answer is known by construction.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from cosmo_sr.eval.rockstar import HaloCatalog
from cosmo_sr.features.lagrangian_host import LagrangianGrid

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRID = LagrangianGrid(ng_lr=16, ng_hr=32, tile_hr=8, boxsize_mpc_h=100.0)


def _load(name: str):
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "features" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("collect_host_subhalo_tiles")


def _cat(ids, parents, num_p):
    n = len(ids)
    return HaloCatalog(
        ids=np.asarray(ids, np.int64), parent_ids=np.asarray(parents, np.int64),
        mvir=np.asarray(num_p, float) * 1e10, rvir=np.full(n, 100.0),
        vmax=np.zeros(n), pos=np.zeros((n, 3)), vel=np.zeros((n, 3)),
        num_p=np.asarray(num_p, np.int64), path="toy")


# --------------------------------------------------------------------------
# hr_children_of_sites
# --------------------------------------------------------------------------

def test_children_agree_with_the_single_site_helper(mod):
    sites = np.array([0, 1, 17, 255, GRID.n_lr - 1], np.int64)
    got = mod.hr_children_of_sites(sites, GRID).reshape(sites.size, -1)
    for k, s in enumerate(sites):
        assert np.array_equal(got[k], GRID.hr_children(int(s)))


def test_children_stay_in_their_parents_tile(mod):
    sites = np.arange(0, GRID.n_lr, 37, dtype=np.int64)
    kids = mod.hr_children_of_sites(sites, GRID)
    from cosmo_sr.reward.tiles import tile_of_particle_id
    hr_tiles = tile_of_particle_id(kids, GRID.hr_tile_grid())
    want = np.repeat(GRID.tile_of_lr_site(sites), GRID.upsample ** 3)
    assert np.array_equal(hr_tiles, want)


# --------------------------------------------------------------------------
# host_tile_subhalos
# --------------------------------------------------------------------------

def _one_host_setup(n_sub_particles=(8, 8)):
    """Host 0 with two subhalos (ids 1, 2); host 10 with a subhalo (id 11).

    LR sites 0 and 1 are host 0's footprint; they sit in the same tile, so the
    per-tile split is checked by moving material rather than by tile geometry.
    """
    f3 = GRID.upsample ** 3
    owner = np.full(GRID.ng_hr ** 3, -1, np.int32)
    site0 = GRID.hr_children(0)
    site1 = GRID.hr_children(1)
    a, b = n_sub_particles
    owner[site0] = 0                      # smooth host material
    owner[site0[:a]] = 1                  # subhalo 1 lives in site 0
    owner[site1] = 0
    owner[site1[:b]] = 2                  # subhalo 2 lives in site 1
    # A neighbouring host and its subhalo, in the same LR sites.
    owner[site1[b:b + 4]] = 11
    cat = _cat([0, 1, 2, 10, 11], [-1, 0, 0, -1, 10],
               [2 * f3, a, b, f3, 4])
    return owner, cat, f3


def test_counts_only_the_selected_hosts_substructure(mod):
    owner, cat, f3 = _one_host_setup()
    root, num_p, mvir, is_sub = mod.catalog_tables(cat)
    got = mod.host_tile_subhalos(np.array([0, 1]), GRID, owner, root, num_p,
                                 is_sub, n_tiles=GRID.n_tiles)
    assert got["match"]["halo_id"] == 0
    # Two whole subhalos of host 0; halo 11's satellite is excluded even though
    # its particles are inside the footprint.
    assert got["sub_total_footprint"] == pytest.approx(2.0)


def test_a_subhalo_split_across_tiles_is_split_fractionally(mod):
    """Half of a subhalo's particles in each of two tiles -> 0.5 and 0.5."""
    f3 = GRID.upsample ** 3
    # Two LR sites in *different* tiles: tile_lr = tile_hr // upsample.
    s_a, s_b = 0, GRID.tile_lr
    ta, tb = (int(GRID.tile_of_lr_site(np.array([s_a]))[0]),
              int(GRID.tile_of_lr_site(np.array([s_b]))[0]))
    assert ta != tb
    owner = np.full(GRID.ng_hr ** 3, -1, np.int32)
    ka, kb = GRID.hr_children(s_a), GRID.hr_children(s_b)
    owner[ka] = 0
    owner[kb] = 0
    owner[ka[:6]] = 1
    owner[kb[:6]] = 1                     # same subhalo, 12 particles, 6 per tile
    cat = _cat([0, 1], [-1, 0], [2 * f3, 12])
    root, num_p, mvir, is_sub = mod.catalog_tables(cat)
    got = mod.host_tile_subhalos(np.array([s_a, s_b]), GRID, owner, root,
                                 num_p, is_sub, n_tiles=GRID.n_tiles)
    fp = got["footprint_count"]
    assert fp[ta] == pytest.approx(0.5)
    assert fp[tb] == pytest.approx(0.5)
    assert got["sub_total_footprint"] == pytest.approx(1.0)


def test_partial_overlap_counts_only_the_footprints_share(mod):
    """A subhalo half outside the LR footprint contributes 0.5, not 1."""
    f3 = GRID.upsample ** 3
    owner = np.full(GRID.ng_hr ** 3, -1, np.int32)
    inside, outside = GRID.hr_children(0), GRID.hr_children(1)
    owner[inside] = 0
    owner[inside[:5]] = 1
    owner[outside[:5]] = 1                # same subhalo, outside the footprint
    cat = _cat([0, 1], [-1, 0], [f3, 10])
    root, num_p, mvir, is_sub = mod.catalog_tables(cat)
    got = mod.host_tile_subhalos(np.array([0]), GRID, owner, root, num_p,
                                 is_sub, n_tiles=GRID.n_tiles)
    assert got["sub_total_footprint"] == pytest.approx(0.5)


def test_sub_subhalos_count_toward_their_top_level_host(mod):
    f3 = GRID.upsample ** 3
    owner = np.full(GRID.ng_hr ** 3, -1, np.int32)
    kids = GRID.hr_children(0)
    owner[kids] = 0
    owner[kids[:4]] = 2                   # sub-sub of host 0, via subhalo 1
    cat = _cat([0, 1, 2], [-1, 0, 1], [f3, 8, 4])
    root, num_p, mvir, is_sub = mod.catalog_tables(cat)
    got = mod.host_tile_subhalos(np.array([0]), GRID, owner, root, num_p,
                                 is_sub, n_tiles=GRID.n_tiles)
    assert got["match"]["halo_id"] == 0
    assert got["sub_total_footprint"] == pytest.approx(1.0)


def test_unbound_footprint_reports_no_match(mod):
    owner = np.full(GRID.ng_hr ** 3, -1, np.int32)
    cat = _cat([0], [-1], [10])
    root, num_p, mvir, is_sub = mod.catalog_tables(cat)
    got = mod.host_tile_subhalos(np.array([0]), GRID, owner, root, num_p,
                                 is_sub, n_tiles=GRID.n_tiles)
    assert got["match"] is None and got["sub_total_footprint"] == 0.0
    assert got["n_bound"] == 0 and "footprint_count" not in got


def test_chunking_does_not_change_the_answer(mod):
    owner, cat, _ = _one_host_setup()
    root, num_p, mvir, is_sub = mod.catalog_tables(cat)
    a = mod.host_tile_subhalos(np.array([0, 1]), GRID, owner, root, num_p,
                               is_sub, n_tiles=GRID.n_tiles, chunk_sites=1)
    b = mod.host_tile_subhalos(np.array([0, 1]), GRID, owner, root, num_p,
                               is_sub, n_tiles=GRID.n_tiles, chunk_sites=64)
    assert a["footprint_count"] == pytest.approx(b["footprint_count"])


def test_catalog_tables_pad_past_the_largest_catalog_id(mod):
    cat = _cat([0, 3], [-1, 0], [10, 4])
    root, num_p, mvir, is_sub = mod.catalog_tables(cat, n_ids=12)
    assert root.size == num_p.size == is_sub.size == 12
    assert root[3] == 0 and is_sub[3] and not is_sub[0]
    assert not is_sub[11] and num_p[11] == 0      # padding is inert


# --------------------------------------------------------------------------
# subhalo_lagrangian_centres -- panel 1's overlay
# --------------------------------------------------------------------------

def test_centre_is_the_mean_lattice_site_of_the_members(mod):
    """A subhalo built from one LR site's children centres on that site."""
    owner = np.full(GRID.ng_hr ** 3, -1, np.int32)
    kids = GRID.hr_children(17)
    owner[kids] = 1
    cat = _cat([0, 1], [-1, 0], [100, kids.size])
    _, _, _, is_sub = mod.catalog_tables(cat)
    pos, cnt = mod.subhalo_lagrangian_centres(owner, GRID, is_sub, is_sub.size)
    assert cnt[1] == kids.size and cnt[0] == 0
    # The site's own centre, in Mpc/h: LR site (a,b,c) spans one LR cell.
    n = GRID.ng_lr
    a, b, c = 17 // (n * n), (17 // n) % n, 17 % n
    want = (np.array([a, b, c]) + 0.5) * GRID.cell_mpc_h
    assert pos[1] == pytest.approx(want, abs=1e-6)
    assert np.all(pos[0] == 0.0)                 # hosts are left at zero


def test_centre_uses_a_circular_mean_across_the_box_seam(mod):
    """Members at both ends of an axis centre on the seam, not mid-box."""
    ngh = GRID.ng_hr
    owner = np.full(ngh ** 3, -1, np.int32)
    # Two HR particles straddling x = 0: the last plane and the first.
    for a in (ngh - 1, 0):
        owner[(a * ngh + 3) * ngh + 4] = 1
    cat = _cat([0, 1], [-1, 0], [100, 2])
    _, _, _, is_sub = mod.catalog_tables(cat)
    pos, cnt = mod.subhalo_lagrangian_centres(owner, GRID, is_sub, is_sub.size)
    assert cnt[1] == 2
    box = GRID.boxsize_mpc_h
    assert min(pos[1, 0], box - pos[1, 0]) < 0.51 * box / ngh
    # A plain mean would land at the middle of the box; this must not.
    assert abs(pos[1, 0] - box / 2) > 0.4 * box


def test_only_subhalos_are_centred(mod):
    owner = np.full(GRID.ng_hr ** 3, -1, np.int32)
    owner[GRID.hr_children(5)] = 0               # host material
    owner[GRID.hr_children(6)[:4]] = 1           # a subhalo
    cat = _cat([0, 1], [-1, 0], [500, 4])
    _, _, _, is_sub = mod.catalog_tables(cat)
    pos, cnt = mod.subhalo_lagrangian_centres(owner, GRID, is_sub, is_sub.size)
    assert cnt[0] == 0 and cnt[1] == 4


def test_centres_are_chunk_invariant(mod):
    owner, cat, _ = _one_host_setup()
    _, _, _, is_sub = mod.catalog_tables(cat)
    a = mod.subhalo_lagrangian_centres(owner, GRID, is_sub, is_sub.size,
                                       chunk=10 ** 9)
    b = mod.subhalo_lagrangian_centres(owner, GRID, is_sub, is_sub.size,
                                       chunk=1000)
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    assert np.array_equal(a[1], b[1])


def test_member_count_matches_the_catalogs_num_p(mod):
    """The consistency check the collector prints: leaf attribution is exact."""
    owner, cat, _ = _one_host_setup()
    _, num_p, _, is_sub = mod.catalog_tables(cat)
    _, cnt = mod.subhalo_lagrangian_centres(owner, GRID, is_sub, is_sub.size)
    have = cnt > 0
    assert np.array_equal(cnt[have], num_p[have])


# --------------------------------------------------------------------------
# RootTileSubhalos -- the whole-host counts the page plots
# --------------------------------------------------------------------------

def _tilew(halo_id, tile_id, weight):
    return {"halo_id": np.asarray(halo_id, np.int64),
            "tile_id": np.asarray(tile_id, np.int64),
            "weight": np.asarray(weight, float)}


def test_root_tile_counts_sum_to_the_hosts_subhalo_count(mod):
    # host 0: subhalos 1 (split 0.25/0.75 over tiles 3,4) and 2 (all in tile 4);
    # host 10: subhalo 11 in tile 4, which must not leak into host 0's row.
    cat = _cat([0, 1, 2, 10, 11], [-1, 0, 0, -1, 10], [100, 10, 10, 50, 5])
    root, num_p, mvir, is_sub = mod.catalog_tables(cat)
    z = _tilew([0, 1, 1, 2, 10, 11], [3, 3, 4, 4, 4, 4],
               [1.0, 0.25, 0.75, 1.0, 1.0, 1.0])
    rts = mod.RootTileSubhalos(z, root, is_sub, n_tiles=8)
    t, c = rts.of(0)
    assert dict(zip(t.tolist(), c.tolist())) == pytest.approx({3: 0.25, 4: 1.75})
    assert c.sum() == pytest.approx(2.0)          # host 0 has exactly 2 subhalos
    t10, c10 = rts.of(10)
    assert c10.sum() == pytest.approx(1.0)
    assert rts.of(999)[1].size == 0               # a host with no substructure


def test_root_tile_counts_are_ordered_most_populated_first(mod):
    cat = _cat([0, 1], [-1, 0], [100, 10])
    root, num_p, mvir, is_sub = mod.catalog_tables(cat)
    rts = mod.RootTileSubhalos(_tilew([1, 1], [7, 2], [0.3, 0.7]), root,
                               is_sub, n_tiles=8)
    t, c = rts.of(0)
    assert t.tolist() == [2, 7] and c.tolist() == pytest.approx([0.7, 0.3])


def test_sub_subhalos_are_credited_to_the_top_level_host(mod):
    cat = _cat([0, 1, 2], [-1, 0, 1], [100, 20, 5])
    root, num_p, mvir, is_sub = mod.catalog_tables(cat)
    rts = mod.RootTileSubhalos(_tilew([1, 2], [1, 1], [1.0, 1.0]), root,
                               is_sub, n_tiles=8)
    assert rts.of(0)[1].sum() == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Panel 1's overlay geometry
#
# There is no JS engine here, and a transposed axis would still paint a
# plausible-looking picture, so the page's own expressions are transliterated
# and checked against the cell each dot must land in.
# --------------------------------------------------------------------------

def _dot_on_slice(pos_mpc_h, grid, w=512):
    """The page's `scDot(g, cz * px, cy * px, ...)` in drawSlice."""
    px = w / grid.ng_lr
    cy = pos_mpc_h[1] / grid.cell_mpc_h
    cz = pos_mpc_h[2] / grid.cell_mpc_h
    return cz * px, cy * px


def _cell_rect_on_slice(b, c, grid, w=512):
    """The page's `g.fillRect(c * px, b * px, px, px)` in drawSlice."""
    px = w / grid.ng_lr
    return c * px, b * px, px


def test_slice_overlay_lands_in_the_cell_that_paints_the_site(mod):
    """A subhalo made of one LR site's children draws on that site's cell."""
    site = ((5 * GRID.ng_lr) + 9) * GRID.ng_lr + 3      # (a, b, c) = (5, 9, 3)
    owner = np.full(GRID.ng_hr ** 3, -1, np.int32)
    kids = GRID.hr_children(site)
    owner[kids] = 1
    cat = _cat([0, 1], [-1, 0], [100, kids.size])
    _, _, _, is_sub = mod.catalog_tables(cat)
    pos, _ = mod.subhalo_lagrangian_centres(owner, GRID, is_sub, is_sub.size)

    n = GRID.ng_lr
    a, b, c = site // (n * n), (site // n) % n, site % n
    # The page's plane test: floor(cx) === state.slice
    assert int(np.floor(pos[1][0] / GRID.cell_mpc_h)) == a
    dx, dy = _dot_on_slice(pos[1], GRID)
    rx, ry, px = _cell_rect_on_slice(b, c, GRID)
    assert dx == pytest.approx(rx + px / 2)     # dead centre of the right cell
    assert dy == pytest.approx(ry + px / 2)
    # And explicitly not the transposed cell, unless b == c.
    tx, ty, _ = _cell_rect_on_slice(c, b, GRID)
    assert (dx, dy) != pytest.approx((tx + px / 2, ty + px / 2))


def test_hr_overlay_lands_in_the_tile_crop_cell(mod):
    """Same check for the broadcast-HR panel, which is cropped to one tile."""
    n, S = GRID.ng_lr, GRID.tile_lr
    site = ((5 * n) + 9) * n + 3
    owner = np.full(GRID.ng_hr ** 3, -1, np.int32)
    kids = GRID.hr_children(site)
    owner[kids] = 1
    cat = _cat([0, 1], [-1, 0], [100, kids.size])
    _, _, _, is_sub = mod.catalog_tables(cat)
    pos, _ = mod.subhalo_lagrangian_centres(owner, GRID, is_sub, is_sub.size)

    a, b, c = site // (n * n), (site // n) % n, site % n
    tx, ty, tz = a // S, b // S, c // S
    W, f = 512, GRID.upsample
    px = W / (S * f)
    # The page's rect for LR site (b, c) inside the crop.
    rx, ry = (c - tz * S) * f * px, (b - ty * S) * f * px
    # The page's dot: ((cz - tz*S) * (W/S), (cy - ty*S) * (W/S)).
    cy, cz = pos[1][1] / GRID.cell_mpc_h, pos[1][2] / GRID.cell_mpc_h
    dx, dy = (cz - tz * S) * (W / S), (cy - ty * S) * (W / S)
    assert dx == pytest.approx(rx + f * px / 2)
    assert dy == pytest.approx(ry + f * px / 2)
    assert 0 <= dx < W and 0 <= dy < W
    # The tile the page crops to is the one the site belongs to.
    assert int(GRID.tile_of_lr_site(np.array([site]))[0]) == (tx * (n // S) + ty) * (n // S) + tz


def test_a_centre_outside_the_tile_is_cropped_out(mod):
    """The HR panel must reject dots from neighbouring tiles in the same plane."""
    n, S = GRID.ng_lr, GRID.tile_lr
    site = ((5 * n) + 9) * n + 3
    other = ((5 * n) + 9) * n + (3 + S)          # same plane, next tile over in z
    c = other % n
    tz_sel = (site % n) // S
    assert c // S != tz_sel
    dz = c - tz_sel * S
    assert not (0 <= dz < S)                     # the page's crop test rejects it


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

def _subtiles_json(tmp_path, payload_hosts, n_tiles):
    """A collected file covering the first host of the toy payload."""
    h = payload_hosts[0]
    t0 = h["tiles"][0]
    obj = {
        "box": "toy", "n_tiles": n_tiles, "upsample": 2,
        "summary": {"n_hosts": 1, "hr_sub_total": 10.0, "sr2_sub_total": 4.0,
                    "ratio": 0.4, "sources_present": ["base", "hr"]},
        "provenance": {},
        "hosts": [{
            "row": h["row"], "lr_host_id": h["host_id"],
            "lr_log_mvir": h["log_mvir"], "n_lr_sites": h["n_particles"],
            "lr_n_sub": h["n_sub"],
            "hr": {"n_hr_particles": 100, "n_bound": 90,
                   "tile_sub_footprint": [7.0], "sub_total_footprint": 7.0,
                   "match": {"halo_id": 7, "log_mvir": 13.5, "num_p": 900,
                             "recursive_num_p": 1000,
                             "is_sub_of_another_host": False,
                             "n_sub_catalog": 20, "n_shared_particles": 80,
                             "match_frac": 0.8, "match_frac_bound": 0.89,
                             "share_of_match": 0.8, "n_lr_hosts_sharing": 1},
                   "tile_id": [t0], "tile_sub": [10.0], "sub_total": 10.0},
            "sr2": {"n_hr_particles": 100, "n_bound": 88,
                    "tile_sub_footprint": [1.0], "sub_total_footprint": 1.0,
                    "match": {"halo_id": 9, "log_mvir": 13.4, "num_p": 880,
                              "recursive_num_p": 950,
                              "is_sub_of_another_host": False,
                              "n_sub_catalog": 6, "n_shared_particles": 78,
                              "match_frac": 0.78, "match_frac_bound": 0.886,
                              "share_of_match": 0.82, "n_lr_hosts_sharing": 1},
                    "tile_id": [t0], "tile_sub": [4.0], "sub_total": 4.0},
        }],
    }
    p = tmp_path / "subtiles.json"
    p.write_text(json.dumps(obj))
    return p, obj


def test_page_with_subtiles_renders(tmp_path):
    """The renderer embeds the collected file verbatim and keys it by row."""
    app = _load("render_lagrangian_host_app")
    from cosmo_sr.features.lagrangian_host import build_host_features

    g = LagrangianGrid(ng_lr=16, ng_hr=128, tile_hr=32, boxsize_mpc_h=100.0)
    owner = np.full(g.n_lr, -1, np.int32)
    owner[np.arange(0, 400)] = 0
    cat = _cat([0], [-1], [400])
    feat = build_host_features(cat, owner, g, box="toy", n_sub_per_host={0: 3})
    pl = app.build_payload(feat, n_hosts=0, n_sample=100, seed=0)
    p, obj = _subtiles_json(tmp_path, pl["hosts"], g.n_tiles)

    out = app.render(feat, tmp_path / "st.html", n_hosts=0, n_sample=100,
                     seed=0, subtiles_path=p)
    html = out.read_text()
    js = html.split("<script>", 1)[1]
    got = json.loads(re.search(r"^const D = (\{.*\});$", js, re.M).group(1))
    st = got["subtiles"]
    assert st is not None
    assert st["hosts"][0]["row"] == pl["hosts"][0]["row"]
    assert st["hosts"][0]["hr"]["sub_total"] == 10.0
    assert st["summary"]["ratio"] == pytest.approx(0.4)
    assert html.count("<canvas") == 8


def _centres_npz(tmp_path, grid, n=7):
    rng = np.random.default_rng(3)
    pos = rng.random((n, 3))
    p = tmp_path / "centres.npz"
    np.savez_compressed(
        p,
        boxsize_mpc_h=np.array(float(grid.boxsize_mpc_h)),
        ng_lr=np.array(int(grid.ng_lr)),
        hr_pos=np.round(pos * 65535).astype(np.uint16),
        hr_num_p=np.arange(20, 20 + n, dtype=np.uint16),
        hr_host_row=np.zeros(n, dtype=np.uint16),
        sr2_pos=np.round(pos[:3] * 65535).astype(np.uint16),
        sr2_num_p=np.arange(30, 33, dtype=np.uint16),
        sr2_host_row=np.full(3, 65535, dtype=np.uint16),
    )
    return p, pos


def test_subhalo_centres_round_trip_into_the_page(tmp_path):
    app = _load("render_lagrangian_host_app")
    from cosmo_sr.features.lagrangian_host import build_host_features
    g = LagrangianGrid(ng_lr=16, ng_hr=128, tile_hr=32, boxsize_mpc_h=100.0)
    owner = np.full(g.n_lr, -1, np.int32)
    owner[np.arange(0, 300)] = 0
    cat = _cat([0], [-1], [300])
    feat = build_host_features(cat, owner, g, box="toy", n_sub_per_host={0: 2})
    p, pos = _centres_npz(tmp_path, g)

    out = app.render(feat, tmp_path / "sc.html", n_hosts=0, n_sample=50, seed=0,
                     subcentres_path=p)
    html = out.read_text()
    js = html.split("<script>", 1)[1]
    got = json.loads(re.search(r"^const D = (\{.*\});$", js, re.M).group(1))
    sc = got["subcentres"]
    assert sc["boxsize"] == 100.0 and sc["hr"]["n"] == 7 and sc["sr2"]["n"] == 3
    # The page reads pos as uint16 little-endian, 3 per subhalo.
    import base64
    xyz = np.frombuffer(base64.b64decode(sc["hr"]["pos"]), dtype="<u2")
    assert xyz.size == 21
    back = xyz.reshape(-1, 3) / 65535 * sc["boxsize"]
    assert back == pytest.approx(pos * 100.0, abs=0.01)
    assert '"subs"' in html and 'id="subscope"' in html
    assert html.count("<canvas") == 8


def test_page_without_subhalo_centres_still_renders(tmp_path):
    app = _load("render_lagrangian_host_app")
    from cosmo_sr.features.lagrangian_host import build_host_features
    g = LagrangianGrid(ng_lr=16, ng_hr=128, tile_hr=32, boxsize_mpc_h=100.0)
    owner = np.full(g.n_lr, -1, np.int32)
    owner[np.arange(0, 120)] = 0
    cat = _cat([0], [-1], [120])
    feat = build_host_features(cat, owner, g, box="toy", n_sub_per_host={0: 1})
    out = app.render(feat, tmp_path / "nosc.html", n_hosts=0, n_sample=40, seed=0)
    html = out.read_text()
    assert '"subcentres":null' in html.replace(" ", "")
    assert html.count("<canvas") == 8


def test_page_without_subtiles_still_renders(tmp_path):
    app = _load("render_lagrangian_host_app")
    from cosmo_sr.features.lagrangian_host import build_host_features
    g = LagrangianGrid(ng_lr=16, ng_hr=128, tile_hr=32, boxsize_mpc_h=100.0)
    owner = np.full(g.n_lr, -1, np.int32)
    owner[np.arange(0, 200)] = 0
    cat = _cat([0], [-1], [200])
    feat = build_host_features(cat, owner, g, box="toy", n_sub_per_host={0: 1})
    out = app.render(feat, tmp_path / "nost.html", n_hosts=0, n_sample=50,
                     seed=0)
    html = out.read_text()
    assert '"subtiles":null' in html.replace(" ", "")
    assert html.count("<canvas") == 8
