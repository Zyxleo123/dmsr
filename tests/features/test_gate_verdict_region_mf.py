import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_P = Path(__file__).resolve().parents[2] / "scripts/features/gate_verdict_vs_baseline.py"
_spec = importlib.util.spec_from_file_location("gate_verdict_vs_baseline", _P)
gv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gv)


def _cat(num_p, parent):
    return SimpleNamespace(num_p=np.asarray(num_p), parent_ids=np.asarray(parent))


def test_region_mass_function_bins_and_splits():
    # one per bin as a sub, plus a 300p top-level halo and a 10p sub below the floor
    cat = _cat([60, 150, 300, 900, 5000, 300, 10], [1, 1, 1, 1, 1, -1, 1])
    mask = np.ones(cat.num_p.size, dtype=bool)
    out = gv.region_mass_function(cat, mask)
    assert out["sub"] == [1, 1, 1, 1, 1]
    assert out["host"] == [0, 0, 1, 0, 0]
    # the 10p sub falls under the 50p floor and is counted nowhere
    assert sum(out["sub"]) + sum(out["host"]) == cat.num_p.size - 1


def test_region_mass_function_honours_the_mask():
    cat = _cat([60, 60], [1, 1])
    out = gv.region_mass_function(cat, np.array([True, False]))
    assert out["sub"] == [1, 0, 0, 0, 0]


def test_shape_score_is_zero_on_hr_itself_and_grows_with_mismatch():
    hr = [174, 97, 56, 34, 13]
    assert gv.shape_score(hr, hr) == 0.0
    near = gv.shape_score([170, 95, 55, 33, 13], hr)
    over = gv.shape_score([185, 113, 100, 60, 21], hr)   # nod6's measured host bins
    assert near < 0.05 < over
    # an empty bin stays finite
    assert np.isfinite(gv.shape_score([0, 0, 0, 0, 0], hr))


def _radial_cat(sub_offsets, host_rvir_kpc=1000.0):
    """One host at the origin-ish plus subs at given radial offsets (Mpc/h)."""
    n = len(sub_offsets)
    pos = np.zeros((n + 1, 3))
    pos[0] = [50.0, 50.0, 50.0]
    for i, dx in enumerate(sub_offsets):
        pos[i + 1] = [50.0 + dx, 50.0, 50.0]
    return SimpleNamespace(
        ids=np.arange(n + 1, dtype=np.int64),
        parent_ids=np.array([-1] + [0] * n, dtype=np.int64),
        num_p=np.full(n + 1, 100, dtype=np.int64),
        rvir=np.full(n + 1, host_rvir_kpc),
        pos=pos)


def _all_boxes():
    # one box covering the whole run, so in_tile_region keeps everything
    return np.array([[[0.0, 100.0], [0.0, 100.0], [0.0, 100.0]]])


def test_radial_distribution_bins_by_r_over_rvir():
    # host R_vir = 1 Mpc/h, so the offsets ARE r/R_vir
    cat = _radial_cat([0.1, 0.3, 0.8, 1.2, 4.0, 6.0])
    out = gv.radial_distribution(cat, _all_boxes(), dilate=0.0)
    assert out["n"] == 6 and out["n_orphan"] == 0
    # bins 0-.25 .25-.5 .5-.75 .75-1 1-1.5 1.5-2 2-3 3-5 >5
    assert out["counts"] == [1, 1, 0, 1, 1, 0, 0, 1, 1]
    assert abs(out["median_r_over_rvir"] - 1.0) < 1e-9
    assert abs(sum(out["frac"]) - 1.0) < 1e-12


def test_radial_distribution_uses_minimum_image_and_counts_orphans():
    cat = _radial_cat([0.2])
    cat.pos[1] = [49.8, 50.0, 50.0]          # the other side of the host
    assert gv.radial_distribution(cat, _all_boxes(), 0.0)["counts"][0] == 1
    cat.parent_ids[1] = 999                  # parent not in the catalog
    out = gv.radial_distribution(cat, _all_boxes(), 0.0)
    assert out["n_orphan"] == 1 and out["n"] == 0


def test_radial_score_is_zero_on_hr_and_ignores_abundance():
    hr = [0.5, 0.3, 0.2]
    assert gv.radial_score(hr, hr) == 0.0
    # fractions, so a uniformly sparser catalog with the SAME shape scores 0
    assert gv.radial_score([0.5, 0.3, 0.2], hr) == 0.0
    assert gv.radial_score([0.2, 0.3, 0.5], hr) > 0.3
