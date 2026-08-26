"""The data the interactive page draws from.

The page's own drawing code cannot be unit-tested from Python, but everything it
draws is computed here first: the sampled coordinates, the palette that tells a
slice which sites belong to the selected host, the quantised volumes, and the
per-tile allocation the bar chart reports. A wrong index in any of those is
invisible in a screenshot (the picture still looks like a halo) and wrong, so
they are checked against the feature arrays directly.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from cosmo_sr.eval.rockstar import HaloCatalog
from cosmo_sr.features.lagrangian_host import LagrangianGrid, build_host_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_renderer():
    """Import the render script by path -- scripts/ is not an importable package."""
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "features" / "render_lagrangian_host_app.py"
    spec = importlib.util.spec_from_file_location("render_lagrangian_host_app", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GRID = LagrangianGrid(ng_lr=16, ng_hr=128, tile_hr=32, boxsize_mpc_h=100.0)


@pytest.fixture(scope="module")
def payload():
    mod = _load_renderer()
    rng = np.random.default_rng(1)
    ng = GRID.ng_lr
    owner = np.full(GRID.n_lr, -1, dtype=np.int32)
    ids, mvir, nump = [], [], []
    for k in range(6):
        c = rng.integers(0, ng, 3)
        r = 2 + k % 3
        a, b, cc = np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij")
        m = a * a + b * b + cc * cc <= r * r
        flat = np.unique(((c[0] + a[m]) % ng * ng + (c[1] + b[m]) % ng) * ng
                         + (c[2] + cc[m]) % ng)
        owner[flat] = k
        ids.append(k); mvir.append(1e12 * (k + 1) * flat.size); nump.append(flat.size)
    cat = HaloCatalog(
        ids=np.array(ids, np.int64), parent_ids=np.full(len(ids), -1, np.int64),
        mvir=np.array(mvir, float), rvir=np.full(len(ids), 200.0),
        vmax=np.zeros(len(ids)), pos=np.zeros((len(ids), 3)),
        vel=np.zeros((len(ids), 3)), num_p=np.array(nump, np.int64), path="toy")
    feat = build_host_features(
        cat, owner, GRID, box="toy",
        n_sub_per_host={i: 2 * i + 1 for i in ids})
    return mod, feat, mod.build_payload(feat, n_hosts=6, n_sample=1000, seed=0)


def _u8(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8)


def test_sampled_coords_belong_to_their_host(payload):
    _, feat, pl = payload
    ng = GRID.ng_lr
    site_row = feat.host_index.reshape(-1)
    for h in pl["hosts"]:
        xyz = _u8(h["coords"]).reshape(-1, 3).astype(np.int64)
        assert xyz.shape[0] == h["n_sampled"] <= h["n_particles"]
        flat = (xyz[:, 0] * ng + xyz[:, 1]) * ng + xyz[:, 2]
        assert np.all(site_row[flat] == h["row"])


def test_row_index_marks_exactly_each_hosts_sites(payload):
    """uint16 row ids, so >254 hosts cannot collide the way a uint8 palette did."""
    _, feat, pl = payload
    rows = np.frombuffer(base64.b64decode(pl["rowidx"]), dtype="<u2")
    site_row = feat.host_index.reshape(-1)
    assert rows.size == feat.grid.n_lr
    for h in pl["hosts"]:
        assert np.array_equal(rows == h["row"], site_row == h["row"])
    assert np.all(rows[site_row < 0] == 65535)


def test_every_host_is_selectable_by_default(payload):
    """n_hosts=0 must ship the whole table, not a mass-truncated head."""
    mod, feat, _ = payload
    pl = mod.build_payload(feat, n_hosts=0, n_sample=1000, seed=0)
    assert len(pl["hosts"]) == feat.table.n_hosts == pl["n_hosts_total"]
    got = sorted(h["row"] for h in pl["hosts"])
    assert got == list(range(feat.table.n_hosts))
    # ... and still mass-ordered, so the dropdown reads big-to-small.
    m = [h["mvir"] for h in pl["hosts"]]
    assert m == sorted(m, reverse=True)


def test_quantized_volumes_reserve_zero_for_no_host(payload):
    _, feat, pl = payload
    mask = (feat.host_member > 0).reshape(-1)
    for name, v in pl["vol"].items():
        q = _u8(v["data"])
        assert q.size == GRID.n_lr
        assert np.all(q[~mask] == 0)
        assert np.all(q[mask] >= 1)


def test_quantization_recovers_the_channel_within_a_step(payload):
    _, feat, pl = payload
    mask = (feat.host_member > 0).reshape(-1)
    v = pl["vol"]["log_host_mass"]
    q = _u8(v["data"]).astype(np.float64)
    back = v["lo"] + (q - 1) * (v["hi"] - v["lo"]) / 254.0
    truth = feat.log_host_mass.reshape(-1)
    step = (v["hi"] - v["lo"]) / 254.0
    assert np.max(np.abs(back[mask] - truth[mask])) <= step


def test_bar_chart_numbers_are_the_normalisations(payload):
    _, feat, pl = payload
    for h in pl["hosts"]:
        assert h["frac_sum"] == pytest.approx(1.0, abs=1e-6)
        assert h["alloc_sum"] == pytest.approx(h["n_sub"], abs=1e-4)
        # Allocation is measured from the per-site lambda, so it must also equal
        # the analytic N_h * f[h,t] tile by tile.
        want = np.array(h["tile_frac"]) * h["n_sub"]
        assert np.array(h["tile_alloc"]) == pytest.approx(want, abs=1e-4)


def test_tile_counts_are_exact_and_match_the_fractions(payload):
    """The dropdown shows a raw count beside the share; it must be the real one."""
    _, feat, pl = payload
    g = feat.grid
    site_row = feat.host_index.reshape(-1)
    site_tile = g.tile_of_lr_site(np.arange(g.n_lr))
    for h in pl["hosts"]:
        ids = np.flatnonzero(site_row == h["row"])
        assert sum(h["tile_count"]) == ids.size == h["n_particles"]
        for t, c, f in zip(h["tiles"], h["tile_count"], h["tile_frac"]):
            assert c == int(np.count_nonzero(site_tile[ids] == t))
            # tile_frac is stored float32; compare at that precision.
            assert c / h["n_particles"] == pytest.approx(f, rel=1e-6)


def test_tiles_are_listed_most_occupied_first(payload):
    _, _, pl = payload
    for h in pl["hosts"]:
        f = np.array(h["tile_frac"])
        assert np.all(np.diff(f) <= 1e-12)
        assert np.all(f > 0)


def test_page_renders_and_embeds_everything(tmp_path, payload):
    mod, feat, _ = payload
    out = mod.render(feat, tmp_path / "page.html", n_hosts=6, n_sample=500, seed=0)
    html = out.read_text()
    assert "__PAYLOAD__" not in html and "__BOX__" not in html
    # Self-contained: no network fetch of any kind at view time.
    for bad in ("http://", "https://", "src=", "fetch("):
        assert bad not in html
    assert html.count("<canvas") == 8


def test_hr_panel_index_expression_matches_tile_hr(payload):
    """Transliteration of the page's HR-panel lookup, checked against numpy.

    The viewer reads the flat volume with
    ``u8[(((tx*S+x)*NG) + ty*S+b)*NG + tz*S+c]`` and paints that value over an
    ``upsample x upsample`` block. There is no JS engine here to run that, and a
    transposed index would still draw a plausible-looking halo, so the
    expression is reproduced here and required to equal ``tile_lr`` /
    ``tile_hr``.
    """
    _, feat, pl = payload
    ng, s, f = GRID.ng_lr, GRID.tile_lr, GRID.upsample
    nt = GRID.n_per_axis
    flat = feat.log_host_mass.reshape(-1)
    chan = feat.channel_names().index("log_host_mass")

    for tile in (0, 3, GRID.n_tiles - 1):
        tx, ty, tz = tile // (nt * nt), (tile // nt) % nt, tile % nt
        crop = feat.tile_lr(tile)[chan]
        hr = feat.tile_hr(tile)[chan]
        for x in range(s):
            js = np.array([[flat[(((tx * s + x) * ng) + ty * s + b) * ng
                                 + tz * s + c]
                            for c in range(s)] for b in range(s)])
            assert np.array_equal(js, crop[x])
            # ... and the block it paints is what tile_hr holds there.
            for u in range(f):
                assert np.array_equal(hr[x * f + u], np.repeat(
                    np.repeat(js, f, axis=0), f, axis=1))


# --------------------------------------------------------------------------
# Panel 6: the displacement / Eulerian payload
# --------------------------------------------------------------------------

def _fake_lr_field(ng: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((6, ng, ng, ng)).astype(np.float32) * 0.2


@pytest.fixture(scope="module")
def eul(tmp_path_factory, payload):
    mod, feat, _ = payload
    p = tmp_path_factory.mktemp("lr") / "lr.npy"
    np.save(p, _fake_lr_field(GRID.ng_lr))
    return mod, feat, p, mod.eulerian_payload(feat, p)


def test_eulerian_positions_are_indexed_by_lr_particle_id(eul):
    """Positions must join to the features by id, not by some other order."""
    from cosmo_sr.eval.particles import field_to_particles
    _, feat, path, pl = eul
    box = GRID.boxsize_mpc_h
    truth = field_to_particles(np.load(path), boxsize_kpc_h=box * 1000.0,
                              redshift=0.0).pos_mpc_h.astype(np.float64)
    got = _u8(pl["pos"]).reshape(-1, 3).astype(np.float64) / 255.0 * box
    assert got.shape == truth.shape == (GRID.n_lr, 3)
    # One quantisation step is box/255; allow one step plus the wrap at box.
    d = np.abs(got - (truth % box))
    d = np.minimum(d, box - d)
    assert d.max() <= box / 255.0 + 1e-6


def test_displacement_magnitude_round_trips(eul):
    from cosmo_sr.eval.particle_identity import periodic_delta
    from cosmo_sr.eval.particles import field_to_particles
    from cosmo_sr.features import lagrangian_lattice_positions
    _, feat, path, pl = eul
    box = GRID.boxsize_mpc_h
    x = field_to_particles(np.load(path), boxsize_kpc_h=box * 1000.0,
                           redshift=0.0).pos_mpc_h.astype(np.float64)
    psi = periodic_delta(x, lagrangian_lattice_positions(GRID), box)
    truth = np.linalg.norm(psi, axis=1)
    hi = pl["disp"]["hi"]
    assert hi == pytest.approx(truth.max(), rel=1e-9)
    back = _u8(pl["disp"]["data"]).astype(np.float64) / 255.0 * hi
    assert np.max(np.abs(back - truth)) <= hi / 255.0 + 1e-6
    for k in ("median", "p90", "max"):
        assert pl["stats"][k] == pytest.approx(
            {"median": np.median(truth), "p90": np.quantile(truth, .9),
             "max": truth.max()}[k], rel=1e-9)


def test_panel6_tile_site_ids_match_the_tile_grid(eul):
    """Transliteration of the panel's tileSiteIds(), which has no JS engine here.

    The panel walks ``((tx*S+a)*NG + ty*S+b)*NG + tz*S+c``; if that disagreed
    with the tile grid the panel would draw a different tile's particles while
    every label still said the selected one.
    """
    _, feat, _, _ = eul
    ng, s, nt = GRID.ng_lr, GRID.tile_lr, GRID.n_per_axis
    for tile in (0, 7, GRID.n_tiles - 1):
        tx, ty, tz = tile // (nt * nt), (tile // nt) % nt, tile % nt
        js = np.array([((tx * s + a) * ng + ty * s + b) * ng + tz * s + c
                       for a in range(s) for b in range(s) for c in range(s)])
        assert js.size == s ** 3
        assert np.all(GRID.tile_of_lr_site(js) == tile)
        want = np.arange(GRID.n_lr).reshape((ng,) * 3)[GRID.lr_slices(tile)]
        assert np.array_equal(np.sort(js), np.sort(want.reshape(-1)))


def test_page_without_an_lr_field_still_renders(tmp_path, payload):
    mod, feat, _ = payload
    out = mod.render(feat, tmp_path / "nofield.html", n_hosts=3, n_sample=200,
                     seed=0, field_path=None)
    html = out.read_text()
    assert '"eulerian": null' in html.replace(" ", "") or '"eulerian":null' in html.replace(" ", "")
    assert html.count("<canvas") == 8


# --------------------------------------------------------------------------
# Panel 5: the SR2-vs-HR tile abundance
# --------------------------------------------------------------------------

def _abundance_json(tmp_path, n_tiles):
    import json
    hr = np.linspace(1.0, 40.0, n_tiles)
    sr = hr * 0.5
    sr[0] = hr[0] * 1.5                       # one tile over HR
    hr[1] = 0.0                               # one tile with no HR subhalo
    rel = [None if h < 1.0 else float((s - h) / h) for h, s in zip(hr, sr)]
    scored = [r for r in rel if r is not None]
    obj = {
        "box": "toy", "n_tiles": n_tiles, "particles_per_tile": 64 ** 3,
        "min_sub_particles": 0,
        "n_sub_hr": hr.tolist(), "n_sub_sr2": sr.tolist(), "rel_deficit": rel,
        "occupancy_hr": (hr / hr.max()).tolist(),
        "occupancy_sr2": (sr / hr.max()).tolist(),
        "totals": {
            "hr_subhalos": float(hr.sum()), "sr2_subhalos": float(sr.sum()),
            "ratio": float(sr.sum() / hr.sum()),
            "hr_occupancy": 0.57, "sr2_occupancy": 0.55,
            "tiles_short": int(sum(1 for r in scored if r < 0)),
            "tiles_scored": len(scored),
            "median_rel_deficit": float(np.median(scored)),
            "p10_rel_deficit": float(np.quantile(scored, .1)),
            "p90_rel_deficit": float(np.quantile(scored, .9)),
            "worst_rel_deficit": float(min(scored)),
        },
    }
    p = tmp_path / "ab.json"
    p.write_text(json.dumps(obj))
    return p, obj


def test_abundance_is_embedded_and_indexed_by_tile(tmp_path, payload):
    mod, feat, _ = payload
    p, obj = _abundance_json(tmp_path, GRID.n_tiles)
    out = mod.render(feat, tmp_path / "ab.html", n_hosts=0, n_sample=200,
                     seed=0, abundance_path=p)
    html = out.read_text()
    assert html.count("<canvas") == 8
    js = html.split("<script>", 1)[1]
    got = json.loads(re.search(r"^const D = (\{.*\});$", js, re.M).group(1))
    ab = got["abundance"]
    assert ab is not None
    assert len(ab["rel_deficit"]) == len(ab["n_sub_hr"]) == GRID.n_tiles
    assert ab["totals"]["ratio"] == pytest.approx(obj["totals"]["ratio"])
    # A tile HR never populated must stay null, not silently become -1 or 0.
    assert ab["rel_deficit"][1] is None
    assert ab["rel_deficit"][0] == pytest.approx(0.5)


def test_page_without_abundance_still_renders(tmp_path, payload):
    mod, feat, _ = payload
    out = mod.render(feat, tmp_path / "noab.html", n_hosts=0, n_sample=200,
                     seed=0, abundance_path=None)
    html = out.read_text()
    assert html.count("<canvas") == 8
    assert '"abundance":null' in html.replace(" ", "")
