"""SUBFIND resolution settings reach the GADGET-4 parameter file and the readout."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts" / "reward", ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

snap = pytest.importorskip("subfind_snapshot")
cmp_ = pytest.importorskip("subfind_setting_compare")


def _param(**kw):
    return snap._param_txt(Path("/tmp/x"), "snapshot", 8, 100.0, max_mem_mb=100,
                           softening=0.005, time_limit_s=10, **kw)


def test_defaults_are_the_published_values():
    p = _param()
    assert "\nDesNumNgb                 64\n" in p
    assert "\nDesLinkNgb                20\n" in p


def test_settings_reach_the_param_file():
    p = _param(des_num_ngb=32, des_link_ngb=10)
    assert "\nDesNumNgb                 32\n" in p
    assert "\nDesLinkNgb                10\n" in p
    assert "{des_num_ngb}" not in p


def test_used_setting_reads_gadget_usedvalues(tmp_path):
    (tmp_path / "param.txt-usedvalues").write_text(
        "BoxSize    100\nDesNumNgb                                         32\n"
        "MaxNumNgbDeviation 1\nDesLinkNgb                                        10\n")
    npz = tmp_path / "subfind_catalog.npz"
    assert cmp_.used_setting(npz) == {"DesNumNgb": 32, "DesLinkNgb": 10}


def test_census_bands_and_sizes(tmp_path):
    # one host of R_vir 1 at the origin; subhalos at x = 0.1, 0.7, 2.0 with sizes
    npz = tmp_path / "subfind_catalog.npz"
    np.savez(npz, SubhaloLen=np.array([500, 30, 15, 60], dtype=np.int32),
             SubhaloPos=np.array([[10.1, 10, 10], [10.7, 10, 10], [10.7, 10, 10],
                                  [12.0, 10, 10]], dtype=np.float32),
             is_satellite=np.array([False, True, True, False]))
    r = cmp_.census(npz, np.array([[10.0, 10.0, 10.0]]), np.array([1.0]))
    assert r["all_ge20"] == [1, 0, 1, 0, 1]
    assert r["all_ge10"] == [1, 0, 2, 0, 1]
    assert r["sat_ge20"] == [0, 0, 1, 0, 0]
    assert r["outskirt_ge20"] == 2
    assert r["outskirt_sizes"] == [1, 1, 1, 0]
