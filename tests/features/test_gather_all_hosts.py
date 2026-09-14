"""The all-hosts abundance rules, on catalogs whose answer is known by hand.

What is pinned here is exactly what could silently change the headline number:
which object in a sphere is dropped as "the host", whether the aperture wraps
through the periodic edge, what counts as a surviving host, and that pooling
across boxes sums counts rather than averaging ratios.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts" / "reward", ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.eval.rockstar import HaloCatalog  # noqa: E402

ah = pytest.importorskip("gather_all_hosts_abundance")
rep = pytest.importorskip("gather_all_hosts_report")
BOX = ah.BOX


def make_cat(pos, num_p, parent_ids=None) -> HaloCatalog:
    n = len(num_p)
    parent_ids = [-1] * n if parent_ids is None else parent_ids
    return HaloCatalog(
        ids=np.arange(n, dtype=np.int64),
        parent_ids=np.asarray(parent_ids, dtype=np.int64),
        mvir=np.asarray(num_p, dtype=np.float64) * 1e9,
        rvir=np.full(n, 100.0),
        vmax=np.full(n, 100.0),
        pos=np.asarray(pos, dtype=np.float64).reshape(n, 3),
        vel=np.zeros((n, 3)),
        num_p=np.asarray(num_p, dtype=np.int64),
    )


def test_central_is_dropped_and_the_cut_applies():
    """One 5000p host, three companions: 400p and 60p inside, 40p below the cut."""
    cat = make_cat(
        [[50, 50, 50], [50.2, 50, 50], [50, 50.3, 50], [49.8, 50, 50]],
        [5000, 400, 60, 40])
    n_sub, hist = ah.aperture_counts(
        cat, np.array([[50.0, 50.0, 50.0]]), np.array([1.0]), 50)
    assert n_sub.tolist() == [2]                       # 400p and 60p; not 40p
    assert hist.sum() == 2
    assert dict(zip(ah.BIN_LABELS, hist[0]))["200-500p"] == 1


def test_the_dropped_object_is_the_most_massive_not_the_nearest():
    """A satellite at the exact centre must not be mistaken for the host."""
    cat = make_cat([[50, 50, 50], [50.5, 50, 50]], [80, 900])
    n_sub, _ = ah.aperture_counts(
        cat, np.array([[50.0, 50.0, 50.0]]), np.array([1.0]), 50)
    assert n_sub.tolist() == [1]      # the 900p one is the host, the 80p counts


def test_aperture_wraps_through_the_periodic_edge():
    cat = make_cat([[0.1, 50, 50], [99.9, 50, 50]], [5000, 300])
    n_sub, _ = ah.aperture_counts(
        cat, np.array([[0.1, 50.0, 50.0]]), np.array([0.5]), 50)
    assert n_sub.tolist() == [1]


def test_empty_sphere_counts_zero_not_minus_one():
    cat = make_cat([[10, 10, 10]], [5000])
    n_sub, hist = ah.aperture_counts(
        cat, np.array([[50.0, 50.0, 50.0]]), np.array([1.0]), 50)
    assert n_sub.tolist() == [0] and hist.sum() == 0


def test_host_survival_needs_top_level_and_the_particle_cut():
    centre = np.array([[50.0, 50.0, 50.0]])
    radius = np.array([0.5])
    alive = make_cat([[50.1, 50, 50]], [5000])
    demoted = make_cat([[50.1, 50, 50]], [5000], parent_ids=[7])
    shrunk = make_cat([[50.1, 50, 50]], [150])
    gone = make_cat([[10, 10, 10]], [5000])
    assert ah.match_hosts(alive, centre, radius, 200).tolist() == [True]
    assert ah.match_hosts(demoted, centre, radius, 200).tolist() == [False]
    assert ah.match_hosts(shrunk, centre, radius, 200).tolist() == [False]
    assert ah.match_hosts(gone, centre, radius, 200).tolist() == [False]


def test_survival_takes_the_most_massive_in_range_not_the_nearest():
    """A small stripped clump nearer the HR centre must not mask the remnant."""
    cat = make_cat([[50.05, 50, 50], [50.2, 50, 50]], [150, 5000])
    assert ah.match_hosts(cat, np.array([[50.0, 50.0, 50.0]]),
                          np.array([0.5]), 200).tolist() == [True]


def test_discover_arms_strips_the_gate_scaffolding(tmp_path):
    d = tmp_path / "flow_rockstar" / "halos"
    for name in ("set9__candidate__mgho_all_blocks_self_set9",
                 "set9__candidate__mgho_frozen_set9",
                 "set10__candidate__mgho_all_blocks_self_set10"):
        (d / name).mkdir(parents=True)
    assert ah.discover_arms(tmp_path, "set9") == [
        ("all_blocks_self", "mgho_all_blocks_self_set9"),
        ("frozen", "mgho_frozen_set9"),
    ]


def _box(box, n_hosts, subs):
    """A minimal per-box result of the shape the abundance stage writes."""
    return {
        "ok": True, "box": box, "host_min_p": 200, "sub_min_p": 50,
        "n_hosts": n_hosts,
        "host_mass_edges_log10": [12.0, 13.0, 14.0],
        "mass_bin_labels": ah.BIN_LABELS,
        "arms": {
            name: {
                "subs_total": v, "hosts_alive": n_hosts,
                "mass_function": {k: 0 for k in ah.BIN_LABELS},
                "by_host_mass": [{"n_hosts": n_hosts, "subs_total": v},
                                 {"n_hosts": 0, "subs_total": 0}],
            } for name, v in subs.items()
        },
    }


def test_pooling_sums_counts_and_counts_each_box_host_once():
    """Two boxes with very different sizes: the pool must not average ratios."""
    arms = rep.pool([_box("set9", 100, {"hr": 1000, "self": 900}),
                     _box("set10", 300, {"hr": 3000, "self": 1500})])
    assert arms["hr"]["n_hosts"] == 400
    assert arms["self"]["subs_total"] == 2400
    # 2400/4000 = 60%, not the (90% + 50%)/2 = 70% a mean of ratios would give.
    assert arms["self"]["subs_total"] / arms["hr"]["subs_total"] == 0.6
    assert arms["self"]["subs_per_host"] == 6.0
    assert arms["self"]["boxes"] == ["set9", "set10"]


def test_an_arm_gated_on_one_box_only_pools_that_box():
    arms = rep.pool([_box("set9", 100, {"hr": 1000, "self": 900}),
                     _box("set10", 300, {"hr": 3000})])
    assert arms["self"]["boxes"] == ["set9"] and arms["self"]["n_hosts"] == 100
    assert arms["hr"]["n_hosts"] == 400


# --------------------------------------------------------------------------
# The talk figure: the mass-function bars must read the bins in PHYSICAL order
# (the json is a dict, so "2000+p" sorts second alphabetically) and label each
# bin with the tuned arm's share of HR.

plot = pytest.importorskip("gather_all_hosts_plot")


def _report_for_plot():
    labels = ah.BIN_LABELS
    def arm(mf, per):
        return {"mass_function": dict(zip(labels, mf)), "subs_per_host": per,
                "subs_total": sum(mf), "hosts_alive_frac": 0.06,
                "n_hosts": 10, "boxes": ["set9", "set10"],
                "by_host_mass": [{"n_hosts": 10, "subs_per_host": per}] * 2}
    return {"n_hosts": 10, "boxes": ["set9", "set10"],
            "host_mass_edges_log10": [12.0, 13.0, 14.0],
            "arms": {"hr": arm([100, 80, 40, 20, 10], 25.0),
                     "base": arm([10, 8, 4, 2, 1], 2.5),
                     "all_blocks_self": arm([50, 40, 20, 10, 5], 12.5)}}


def test_mass_function_bins_are_in_physical_order_not_alphabetical():
    assert ah.BIN_LABELS == plot.MASS_BIN_ORDER, (
        "the figure's bin order must track the abundance job's labels")
    assert plot.MASS_BIN_ORDER.index("2000+p") == len(ah.BIN_LABELS) - 1


def test_mass_function_figure_writes_both_scales(tmp_path):
    rp = _report_for_plot()
    plot.fig_sub_mass_function(rp, tmp_path)
    plot.fig_sub_mass_function(rp, tmp_path, yscale="log")
    assert (tmp_path / "sub_mass_function.png").exists()
    assert (tmp_path / "sub_mass_function_log.png").exists()


def test_mass_function_figure_skips_a_missing_arm(tmp_path):
    plot.fig_sub_mass_function(_report_for_plot(), tmp_path, tuned="nope")
    assert not (tmp_path / "sub_mass_function.png").exists()
