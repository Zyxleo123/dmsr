"""The crop page's index math, transliterated.

There is no browser here, so the page's own drawing code cannot be executed.
What *can* be pinned is the arithmetic it does, because a wrong axis or a
transposed row/column still produces a picture that looks like a halo -- the
circles simply land in the wrong place, and the conclusion drawn from the page
is then wrong in a way no screenshot reveals.

So the three expressions the picture depends on are re-implemented here in
Python from the JS source, and the JS is asserted to still contain them:

* ``planeAxes`` -- which two axes a view along ``axis`` shows, and in which order;
* the flat index inside ``slabMax``;
* the overlay's ``(row, col)`` placement of a subhalo centre.

The property that ties them together: a bright voxel and a subhalo centre at
the *same* crop coordinate must come out at the same place on the canvas.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load():
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "features" / "render_host_crops_app.py"
    spec = importlib.util.spec_from_file_location("render_host_crops_app", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


# --- the transliteration ---------------------------------------------------

PLANE_AXES = [(1, 2), (0, 2), (0, 1)]


def slab_max(q, g, axis, k0, k1):
    """``slabMax`` in Python: row/col are the two axes ``axis`` leaves over."""
    ra, ca = PLANE_AXES[axis]
    out = np.full((g, g), -np.inf)
    for r in range(g):
        for c in range(g):
            m = -np.inf
            for k in range(k0, k1):
                idx = [0, 0, 0]
                idx[ra], idx[ca], idx[axis] = r, c, k
                m = max(m, q[(idx[0] * g + idx[1]) * g + idx[2]])
            out[r, c] = m
    return out


def overlay_cell(u, axis, side, g):
    """Where ``overlaySubs`` puts a centre, expressed in image cells."""
    ra, ca = PLANE_AXES[axis]
    return (u[ra] * g / side, u[ca] * g / side)


# --- the JS still says what the transliteration assumes ---------------------

def test_js_plane_axes_table_is_the_one_transliterated(mod):
    assert "[[1,2],[0,2],[0,1]]" in mod.APP_JS.replace(" ", "")


def test_js_slab_index_expression_is_the_one_transliterated(mod):
    assert "(idx[0]*g+idx[1])*g+idx[2]" in mod.APP_JS.replace(" ", "")


def test_js_overlay_uses_row_from_ra_and_col_from_ca(mod):
    src = mod.APP_JS.replace(" ", "")
    assert "constx=u[ca]*px,y=u[ra]*px" in src


def test_js_overlay_scales_by_the_cube_extent_not_the_crop_side(mod):
    """The overlay must divide by what the drawn cube spans.

    ``block_reduce`` rounds the crop up to a whole number of blocks, so the
    cube covers ``vol_extent_sites``. Scaling by ``crop_side_sites`` instead
    drifts every circle outward by extent/side and pushes the outermost ones
    into the padded margin.
    """
    src = mod.APP_JS.replace(" ", "")
    assert "constextent=h.vol_extent_sites||side" in src
    assert "constpx=W/extent" in src
    assert "constd0=k0*extent/g,d1=k1*extent/g" in src
    assert "W/side" not in src, "an overlay is still scaled by the crop side"


def test_js_imagedata_row_stride_matches_the_projection(mod):
    """The ImageData is filled in the same ``r*g+c`` order the slab produces."""
    src = mod.APP_JS.replace(" ", "")
    assert "out[r*g+c]=m" in src
    assert "for(leti=0;i<g*g;i++)" in src


# --- the arithmetic itself -------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_single_voxel_projects_to_its_own_row_and_column(axis):
    g = 8
    q = np.zeros(g ** 3, dtype=np.uint8)
    site = (2, 5, 1)
    q[(site[0] * g + site[1]) * g + site[2]] = 255
    img = slab_max(q, g, axis, 0, g)
    ra, ca = PLANE_AXES[axis]
    assert np.unravel_index(int(np.argmax(img)), img.shape) == (site[ra], site[ca])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_subhalo_circle_lands_on_its_own_voxel(axis):
    """The property the page's whole claim rests on."""
    g, side = 8, 32
    site = (2, 5, 1)
    q = np.zeros(g ** 3, dtype=np.uint8)
    q[(site[0] * g + site[1]) * g + site[2]] = 255

    # A subhalo centred anywhere inside that voxel's native block.
    u = np.array(site, dtype=float) * side / g + 0.5 * side / g
    row, col = overlay_cell(u, axis, side, g)

    img = slab_max(q, g, axis, 0, g)
    assert np.unravel_index(int(np.argmax(img)), img.shape) == (int(row), int(col))


def test_slab_restricts_the_projection_to_its_own_depth():
    g = 8
    q = np.zeros(g ** 3, dtype=np.uint8)
    q[(3 * g + 3) * g + 6] = 255        # depth 6 along axis 2
    assert slab_max(q, g, 2, 0, 3).max() == 0
    assert slab_max(q, g, 2, 5, 8).max() == 255


# --- payload round trip ----------------------------------------------------

def _quant(v):
    lo, hi = float(v.min()), float(v.max())
    q = np.clip((v - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return np.concatenate(
        [np.array([lo, hi], dtype=np.float32).view(np.uint8), q.reshape(-1)])


def _toy(g=8, side=34, ids=(11, 22)):
    rng = np.random.default_rng(0)
    store, hosts = {}, []
    for hid in ids:
        for ch in ("sr2_rho", "hr_rho", "hr_sub"):
            store[f"h{hid}__{ch}"] = _quant(
                rng.normal(size=(g, g, g)).astype(np.float32))
        for sd in ("hr", "sr2"):
            store[f"h{hid}__scatter_{sd}"] = (
                rng.normal(size=(64, 3)) * 64).astype("<i2")
        hosts.append({
            "halo_id": hid, "mvir": 1e14, "log_mvir": 14.1, "num_p": 170000,
            "rvir_mpc_h": 1.15, "n_member_sites": 170000, "n_subhalos": 1,
            "n_subhalos_20p": 1, "crop_side_sites": side, "crop_side_mpc_h": 6.25,
            "crop_centre_site": [1.0, 2.0, 3.0], "rl_sites": 16.0, "rl_mpc_h": 3.1,
            "resample": {"native_side": side, "target_side": 96, "ratio": 3.0,
                         "native_sites": side ** 3, "target_sites": 96 ** 3},
            "grid_out": g, "vol_extent_sites": 40, "vol_pad_sites": 6,
            "block_factor": 5.0, "host_fill_frac": 0.4,
            "sub_frac_of_host": 0.07, "sr2_bound_frac_in_host": 0.8, "seconds": 1.0,
            "learnability": {"footprint": {
                "n_sites": 1000, "base_rate": 0.07,
                "auc_sr2_rho": {"sigma0": .6, "sigma1": .62, "sigma2": .65,
                                "sigma4": .63},
                "auc_hr_rho": {"sigma0": .9, "sigma1": .92, "sigma2": .95,
                               "sigma4": .9},
                "auc_radius": 0.55, "auc_sr2_bound": 0.58,
                "roc_sr2_rho": {"fpr": [0, .5, 1], "tpr": [0, .7, 1]},
                "roc_hr_rho": {"fpr": [0, .2, 1], "tpr": [0, .9, 1]}}},
            "hr_subhalos": [{"halo_id": 1, "num_p": 300, "mvir": 1e11,
                             "log_mvir": 11.0, "rvir_mpc_h": 0.05,
                             "u": [10.0, 12.0, 14.0], "d_mpc_h": [.3, -.2, .1],
                             "r_over_rvir": 0.4, "frac_in_crop": 1.0,
                             "rl_sites": 4.1, "sr2_bound_frac": 0.8,
                             "sr2_logrho": 2.1}],
            "sr2_objects": [],
        })
    summary = {"box": "setX", "ok": True, "n_hosts": len(ids), "grid": g,
               "crop_scale": 1.0, "k_neighbours": 32, "pad_sites": 8,
               "boxsize_mpc_h": 100.0, "ng_hr": 512,
               "particle_mass_msun_h": 5.8e8, "hr_field": "a", "sr2_field": "b",
               "seconds": 1.0, "hosts": hosts}
    return summary, store


def test_payload_volumes_decode_back_to_the_quantized_cube(mod):
    summary, store = _toy()
    p = mod.build_payload(summary, store, n_hosts=2)
    raw = base64.b64decode(p["hosts"][0]["vol"]["sr2_rho"])
    assert np.frombuffer(raw, dtype=np.uint8).tobytes() == store["h11__sr2_rho"].tobytes()
    lo, hi = np.frombuffer(raw[:8], dtype="<f4")
    assert hi > lo


def test_payload_scatter_is_little_endian_int16(mod):
    summary, store = _toy()
    p = mod.build_payload(summary, store, n_hosts=2)
    raw = base64.b64decode(p["hosts"][0]["scatter_hr"])
    assert np.array_equal(np.frombuffer(raw, dtype="<i2"),
                          store["h11__scatter_hr"].reshape(-1))


def test_payload_respects_the_host_cap(mod):
    summary, store = _toy()
    assert len(mod.build_payload(summary, store, n_hosts=1)["hosts"]) == 1


def test_page_renders_and_embeds_every_anchor_the_js_looks_up(tmp_path, mod):
    """Every ``getElementById`` in the script must have an element to find."""
    summary, store = _toy()
    out = mod.render(summary, store, tmp_path / "p.html", n_hosts=2)
    html = out.read_text()
    ids = set(re.findall(r'getElementById\("([^"]+)"\)', mod.APP_JS))
    ids |= set(re.findall(r'querySelector\("#([A-Za-z0-9_]+)', mod.APP_JS))
    missing = [i for i in ids if f'id="{i}"' not in html]
    assert not missing, f"the page never defines: {sorted(missing)}"
    assert "window.__CROPS__" in html


def test_embedded_payload_is_valid_json(tmp_path, mod):
    summary, store = _toy()
    out = mod.render(summary, store, tmp_path / "p.html", n_hosts=2)
    m = re.search(r"window\.__CROPS__ = (\{.*?\});</script>", out.read_text(),
                  re.S)
    assert m, "payload not embedded where the page expects it"
    data = json.loads(m.group(1).replace("<\\/", "</"))
    assert data["box"] == "setX" and len(data["hosts"]) == 2
