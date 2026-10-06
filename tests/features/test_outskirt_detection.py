"""The decomposition that decides whether the outskirt failure is placement.

*Radii must be signed against the set's OWN parent.* A trained tile is
12.5 Mpc/h a side, so a task named for one host supervises subhalos of dozens
of others; ``host-profile-gate-passed`` records the same mistake inverting a
verdict. Pinned by a catalog where the two readings disagree by construction.

*The join must be by halo id.* The export and the comparison are written by
different jobs, days apart, and an order-based join that silently shifted by
one would read as a radial trend rather than as a bug.

*The conditional rates must reconstruct the unconditional one.* ``P(addressed)``
and ``P(hit | addressed)`` are the whole result; if they do not compose back to
the observed rate they are measuring two different populations.

*The offset must be in Mpc/h.* ``centre_offset_radii`` is normalised by
``max(r_vir, 0.15)``, and comparing it to the gate's 0.15 Mpc/h search radius
without multiplying back would silently compare a ratio to a length.

*A missing field must read as unavailable, not as zero.* Exports written before
``centre_offset_radii`` existed would otherwise produce an all-NaN table that
looks like a measured null.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cosmo_sr.features.outskirt_detection import (  # noqa: E402
    address_decomposition, build_set_table, hit_miss_contrast,
    radial_profile, reference_inflation,
)

BOX = 100.0


def _catalog():
    """Two parents 20 Mpc/h apart and four subhalos, two of each.

    Parent 0 sits at the origin with R_vir 1 Mpc/h; parent 1 at x = 20 with
    R_vir 2 Mpc/h. Sub 10 and 11 belong to parent 0 at 0.5 and 1.5 Mpc/h;
    sub 20 and 21 belong to parent 1 at 1.0 and 3.0 Mpc/h. Signed against
    their own parents the radii are 0.5, 1.5, 0.5, 1.5 -- and signed against
    parent 0 alone they would be 0.5, 1.5, 19.0, 17.0, which is the reading
    this pins against.
    """
    ids = np.array([0, 1, 10, 11, 20, 21], dtype=np.int64)
    pos = np.array([[0.0, 0, 0], [20.0, 0, 0],
                    [0.5, 0, 0], [1.5, 0, 0],
                    [21.0, 0, 0], [23.0, 0, 0]])
    rvir = np.array([1000.0, 2000.0, 100.0, 100.0, 200.0, 200.0])  # kpc/h
    num_p = np.array([100000, 100000, 400, 400, 400, 400], dtype=np.int64)
    parent = np.array([-1, -1, 0, 0, 1, 1], dtype=np.int64)
    return dict(ids=ids, pos=pos, rvir_kpc_h=rvir, num_p=num_p,
                parent_ids=parent)


def _rows(offsets_radii=(0.5, 0.5, 0.5, 0.5), with_offset=True):
    out = []
    for hid, off in zip((10, 11, 20, 21), offsets_radii):
        r = {"halo_id": hid, "num_p": 400, "n_live": 392,
             "bound_hard": 0.1, "virial": 30.0, "r_rms": 0.3,
             "sigma_v": 250.0, "parent_id": 0 if hid < 20 else 1}
        if with_offset:
            r["centre_offset_radii"] = off
        out.append(r)
    return out


def _cmp(hits=(True, False, True, False)):
    return [{"halo_id": hid, "hit": bool(h), "nearest_mpc_h": 0.1,
             "n_within_radius": 1 if h else 0, "searched_at": "hr_catalog"}
            for hid, h in zip((10, 11, 20, 21), hits)]


def test_radii_are_signed_against_each_sets_own_parent():
    t = build_set_table(_rows(), _cmp(), boxsize=BOX, **_catalog())
    assert t.n == 4 and t.n_unresolved == 0
    # 0.5/1.0, 1.5/1.0, 1.0/2.0, 3.0/2.0
    np.testing.assert_allclose(t.r_over_rvir, [0.5, 1.5, 0.5, 1.5], atol=1e-9)


def test_join_is_by_halo_id_not_by_order():
    rows, cmp_rows = _rows(), _cmp()
    shuffled = [cmp_rows[i] for i in (2, 0, 3, 1)]
    a = build_set_table(rows, cmp_rows, boxsize=BOX, **_catalog())
    b = build_set_table(rows, shuffled, boxsize=BOX, **_catalog())
    np.testing.assert_array_equal(a.hit, b.hit)
    np.testing.assert_array_equal(a.halo_id, b.halo_id)


def test_rows_without_a_compare_partner_are_dropped_not_matched():
    t = build_set_table(_rows(), _cmp()[:2], boxsize=BOX, **_catalog())
    assert t.n == 2
    np.testing.assert_array_equal(t.halo_id, [10, 11])


def test_offset_is_mpc_h_using_the_gate_floor_on_the_scale():
    """Both branches of ``max(r_vir, 0.15)``, because only one is exercised
    by the real pool: subs 10/11 have r_vir 0.1 Mpc/h so the floor binds and
    0.5 radii is 0.075 Mpc/h; subs 20/21 have 0.2 Mpc/h so it does not and the
    same 0.5 radii is 0.1. Reading the ratio as a length would put both
    outside the gate's 0.15 Mpc/h search radius."""
    t = build_set_table(_rows(), _cmp(), boxsize=BOX, **_catalog())
    np.testing.assert_allclose(t.offset_mpc_h,
                               [0.075, 0.075, 0.1, 0.1], atol=1e-9)
    dec = address_decomposition(t, tolerances=(0.15,), bins=((0.0, 2.0),))
    assert dec[0]["p_addressed"] == pytest.approx(1.0)


def test_periodic_wrap_across_the_box_face():
    cat = _catalog()
    cat["pos"] = cat["pos"].copy()
    cat["pos"][0] = [99.5, 0, 0]          # parent 0 just below the face
    cat["pos"][2] = [0.0, 0, 0]           # its sub just above it: 0.5 apart
    t = build_set_table(_rows(), _cmp(), boxsize=BOX, **cat)
    assert t.r_over_rvir[0] == pytest.approx(0.5)


def test_conditional_rates_reconstruct_the_unconditional_rate():
    # sub 10 and 20 hit; 10 is addressed (0.5 radii), 11 is not (9 radii).
    t = build_set_table(_rows(offsets_radii=(0.5, 9.0, 0.5, 9.0)), _cmp(),
                        boxsize=BOX, **_catalog())
    dec = address_decomposition(t, tolerances=(0.15,), bins=((0.0, 2.0),))
    row = [d for d in dec if not d["all_radii"]][0]
    p = row["p_addressed"]
    total = (p * row["hit_rate_given_addressed"]
             + (1 - p) * row["hit_rate_given_unaddressed"])
    prof = radial_profile(t, bins=((0.0, 2.0),))
    assert total == pytest.approx(prof[0]["hit_rate"])


def test_address_decomposition_is_empty_when_the_export_had_no_offset():
    t = build_set_table(_rows(with_offset=False), _cmp(), boxsize=BOX,
                        **_catalog())
    assert address_decomposition(t) == []
    assert "centre_offset_radii" in t.missing


def test_radial_profile_carries_the_mass_control_and_the_miss_anatomy():
    t = build_set_table(_rows(), _cmp(), boxsize=BOX, **_catalog())
    prof = radial_profile(t, bins=((0.0, 1.0), (1.0, 2.0)))
    inner, outer, allr = prof
    assert allr["all_radii"] and allr["n"] == 4
    assert inner["median_num_p"] == 400 and outer["median_num_p"] == 400
    assert inner["hit_rate"] == pytest.approx(1.0)
    assert outer["hit_rate"] == pytest.approx(0.0)
    # both outer rows are misses, and the fixture put nothing in their radius
    assert outer["n_miss"] == 2
    assert outer["miss_frac_something_in_radius"] == pytest.approx(0.0)


def test_hit_miss_contrast_reports_both_populations():
    t = build_set_table(_rows(), _cmp(), boxsize=BOX, **_catalog())
    con = hit_miss_contrast(t)
    assert con["hit"]["n"] == 2 and con["miss"]["n"] == 2
    # r_rms 0.3 Mpc/h against r_vir 0.1 (sub 10) and 0.2 (sub 20)
    assert con["hit"]["median_r_rms_over_rvir_hr"] == pytest.approx(2.25)


def test_reference_inflation_ratio_and_straggler_correlation():
    """Inflation that tracks the straggler fraction must read as correlated.

    Built so it does: each host's reference r_rms is HR's scaled by
    ``1 + 10 (1 - live_frac)``, which is the hypothesis the correlation is
    there to test rather than assume.
    """
    hosts = []
    for lf in (0.99, 0.97, 0.95, 0.93, 0.90):
        infl = 1.0 + 10.0 * (1.0 - lf)
        hosts.append({
            "median_live_frac": lf,
            "pure_hr_median": {"r_rms": 0.13, "sigma_v": 130.0,
                               "virial": 2.1, "bound_hard": 0.76},
            "reference_median": {"r_rms": 0.13 * infl, "sigma_v": 130.0 * infl,
                                 "virial": 2.1 * infl, "bound_hard": 0.67},
        })
    out = reference_inflation({"holdout_hosts": hosts})
    assert out["n_hosts"] == 5
    assert out["quantities"]["r_rms"]["ratio"] == pytest.approx(1.5)  # lf 0.95
    assert out["corr_straggler_frac_vs_r_rms_inflation"] == pytest.approx(1.0)


def test_reference_inflation_is_empty_without_a_pool():
    out = reference_inflation({})
    assert out["n_hosts"] == 0 and out["quantities"] == {}
