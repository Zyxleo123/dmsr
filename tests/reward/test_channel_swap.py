"""The channel-swap intervention's bookkeeping.

Everything this experiment concludes rests on one claim that a rendered number
cannot check: that the assembled field really carries *this* source's
displacement and *that* source's velocity. Swap the two slices by accident and
every downstream count is still perfectly plausible -- a catalog comes out, it
has halos, the ratio moves -- and the verdict is exactly backwards. So the
provenance of each channel group is pinned here against fields whose values
encode where they came from.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "reward" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def swap():
    return _load("channel_swap_rockstar")


@pytest.fixture(scope="module")
def report():
    return _load("report_channel_swap")


@pytest.fixture
def fields():
    """Two ``(6, 4, 4, 4)`` fields whose every value names its own source.

    HR values are +100 + channel, SR2 values are -100 - channel, so a channel
    taken from the wrong source is not merely a different number, it has the
    wrong sign.
    """
    ch = np.arange(6, dtype=np.float32).reshape(6, 1, 1, 1)
    ones = np.ones((6, 4, 4, 4), dtype=np.float32)
    return {"hr": ones * (100.0 + ch), "base": ones * (-100.0 - ch)}


# ---------------------------------------------------------------- assembly --

def test_swap_takes_each_group_from_its_named_source(swap, fields):
    out = swap.assemble_field(fields, swap.arm_spec("srpos_hrvel"))
    # displacement from SR2, velocity from HR -- and nothing else moved.
    assert np.array_equal(out[0:3], fields["base"][0:3])
    assert np.array_equal(out[3:6], fields["hr"][3:6])


def test_the_mirror_arm_is_the_other_way_round(swap, fields):
    out = swap.assemble_field(fields, swap.arm_spec("hrpos_srvel"))
    assert np.array_equal(out[0:3], fields["hr"][0:3])
    assert np.array_equal(out[3:6], fields["base"][3:6])


def test_a_swapped_field_equals_neither_source(swap, fields):
    """The guard against a 'swap' that silently returned one pure field."""
    for arm in ("srpos_hrvel", "hrpos_srvel"):
        out = swap.assemble_field(fields, swap.arm_spec(arm))
        assert not np.array_equal(out, fields["hr"])
        assert not np.array_equal(out, fields["base"])


def test_channels_keep_their_meaning_not_just_their_provenance(swap, fields):
    """Channel k of the output is channel k of *some* source, never channel j.

    A transposed slice (`out[0:3] = src[3:6]`) would still produce a field made
    of both sources, so the arm-level tests above would pass. This one fails on
    it: within a group the per-channel offsets must survive in order.
    """
    out = swap.assemble_field(fields, swap.arm_spec("srpos_hrvel"))
    for k in range(6):
        src = "base" if k < 3 else "hr"
        assert np.all(out[k] == fields[src][k][0, 0, 0]), f"channel {k}"


def test_pure_arms_reproduce_their_source_exactly(swap, fields):
    for arm, name in (("hr", "hr"), ("base", "base")):
        out = swap.assemble_field(fields, swap.arm_spec(arm))
        assert np.array_equal(out, fields[name])


def test_assemble_reads_only_the_requested_slice_of_each_source(swap, fields):
    """A source named for one group must not need to supply the other.

    The driver memmaps whole fields, so this is really a statement that the
    output is built group by group; if it ever copied a whole source first, a
    later group would overwrite it and the arm would collapse to a pure field.
    """
    out = swap.assemble_field(fields, {"disp": "base", "vel": "hr"})
    assert out[0].mean() < 0 and out[5].mean() > 0


@pytest.mark.parametrize("bad_spec", [
    {"disp": "hr"},                       # no source for vel
    {"disp": "hr", "vel": "nope"},        # unknown source
])
def test_bad_specs_are_refused(swap, fields, bad_spec):
    with pytest.raises(ValueError):
        swap.assemble_field(fields, bad_spec)


def test_mismatched_source_shapes_are_refused(swap, fields):
    fields = dict(fields, base=np.zeros((6, 8, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        swap.assemble_field(fields, {"disp": "base", "vel": "hr"})


def test_a_non_catnorm_field_is_refused(swap):
    bad = {"hr": np.zeros((3, 4, 4, 4), np.float32)}
    with pytest.raises(ValueError):
        swap.assemble_field(bad, {"disp": "hr", "vel": "hr"})


def test_unknown_arm_names_are_refused(swap):
    with pytest.raises(SystemExit):
        swap.arm_spec("hrvel_only")


# ------------------------------------------------------------------ counts --

class _Cat:
    def __init__(self, parents, num_p):
        self.parent_ids = np.asarray(parents, np.int64)
        self.num_p = np.asarray(num_p, np.int64)
        self.n = len(self.num_p)


def test_catalog_counts_splits_hosts_from_subhalos(swap):
    c = swap.catalog_counts(_Cat([-1, -1, 0, 0, 2], [100, 50, 30, 20, 10]))
    assert c["n_objects"] == 5
    assert c["n_hosts"] == 2
    assert c["n_subhalos"] == 3
    # num_p excludes substructure, so the total is each bound particle once.
    assert c["n_bound_particles"] == 210
    assert c["sub_particles"] == 60


# -------------------------------------------------------------- the scalar --

def test_recovery_fraction_anchors_at_the_two_controls(report):
    assert report.recovery_fraction(46067, 46067, 100599) == 0.0
    assert report.recovery_fraction(100599, 46067, 100599) == 1.0
    assert report.recovery_fraction(73333, 46067, 100599) == pytest.approx(0.5, abs=0.01)


def test_recovery_fraction_is_not_clipped(report):
    """Overshoot and undershoot are results, not numbers to tidy away."""
    assert report.recovery_fraction(120000, 46067, 100599) > 1.0
    assert report.recovery_fraction(30000, 46067, 100599) < 0.0


def test_recovery_fraction_is_none_when_the_controls_tie(report):
    assert report.recovery_fraction(5, 10, 10) is None


def test_counts_by_num_p_bins_are_half_open_and_cover_the_tail(report):
    num_p = np.array([19, 20, 49, 50, 199, 200, 499, 500, 10 ** 6])
    is_sub = np.ones(num_p.size, bool)
    got = report.counts_by_num_p(num_p, is_sub)
    assert got["20-50"]["subs"] == 2          # 20, 49 -- not 19, not 50
    assert got["50-100"]["subs"] == 1         # 50
    assert got["100-200"]["subs"] == 1        # 199
    assert got["200-500"]["subs"] == 2        # 200, 499
    assert got["500+"]["subs"] == 2           # 500, 1e6
    assert sum(v["subs"] for v in got.values()) == 8   # the 19 is below the floor


def test_counts_by_num_p_separates_hosts_from_subs(report):
    got = report.counts_by_num_p([25, 25, 600], [True, False, False])
    assert got["20-50"] == {"hosts": 1, "subs": 1}
    assert got["500+"] == {"hosts": 1, "subs": 0}


# ----------------------------------------------------------- the structure --

def _df(n_hosts, logm, vrms, t_over_u, extra_subs=0):
    import pandas as pd

    n = n_hosts + extra_subs
    return pd.DataFrame({
        "mvir": [10.0 ** logm] * n,
        "rvir": [1000.0] * n,
        "Rs": [100.0] * n,
        "vmax": [500.0] * n,
        "vrms": [vrms] * n,
        "Xoff": [50.0] * n,
        "T/|U|": [t_over_u] * n,
        "is_sub": [False] * n_hosts + [True] * extra_subs,
    })


def test_structure_by_mass_reports_medians_in_the_right_bin(report):
    got = report.structure_by_mass(_df(40, 13.2, 400.0, 0.55))
    assert list(got) == ["13.0-13.5"]
    row = got["13.0-13.5"]
    assert row["n"] == 40
    assert row["vrms"] == pytest.approx(400.0)
    assert row["t_over_u"] == pytest.approx(0.55)
    assert row["c"] == pytest.approx(10.0)          # rvir/Rs = 1000/100
    assert row["xoff"] == pytest.approx(0.05)       # Xoff/rvir = 50/1000


def test_structure_by_mass_drops_thin_bins(report):
    """Rockstar's NFW fit is noisy at low counts; a median over 5 is not a number."""
    assert report.structure_by_mass(_df(5, 13.2, 400.0, 0.55)) == {}
    assert report.structure_by_mass(_df(5, 13.2, 400.0, 0.55), min_objects=3) != {}


def test_structure_by_mass_ignores_subhalos(report):
    """The comparison is between *hosts* at fixed mass; satellites would skew it."""
    only_subs = _df(0, 13.2, 400.0, 0.55, extra_subs=40)
    assert report.structure_by_mass(only_subs) == {}


# ----------------------------------------------------------------- verdict --

def _rows(n_swap, n_mirror=None):
    out = {"hr": {"n_subhalos": 100599}, "base": {"n_subhalos": 46067},
           "srpos_hrvel": {"n_subhalos": n_swap}, "hrpos_srvel": None}
    if n_mirror is not None:
        out["hrpos_srvel"] = {"n_subhalos": n_mirror}
    return out


def test_verdict_blames_the_velocity_head_when_the_swap_recovers(report):
    assert "velocity" in report.verdict(_rows(95000)).lower()
    assert "velocity head" in report.verdict(_rows(95000))


def test_verdict_blames_the_displacement_field_when_it_does_not(report):
    text = report.verdict(_rows(47000))
    assert "not the constraint" in text
    assert "seeded-substructure" in text or "sr2_subhalo_deficit" in text


def test_verdict_refuses_to_choose_in_the_middle(report):
    assert "Neither half" in report.verdict(_rows(70000))


def test_verdict_reports_inconclusive_without_the_controls(report):
    rows = _rows(95000)
    rows["hr"] = None
    assert report.verdict(rows).startswith("INCONCLUSIVE")


def test_verdict_mentions_the_mirror_arm_when_it_ran(report):
    assert "HR pos + SR2 vel" in report.verdict(_rows(95000, 50000))
    assert "HR pos + SR2 vel" not in report.verdict(_rows(95000))


# ----------------------------------------------------- the subhalo test itself --

def test_parent_remap_matches_i_so_when_every_parent_is_printed(report):
    ids = [10, 11, 12]
    idx = [0, 1, 2]
    i_so = [-1, 0, 0]
    got = report.parent_ids_from_columns(ids, idx, i_so)
    assert got.tolist() == [-1, 10, 10]


def test_an_orphan_subhalo_counts_as_a_host(report):
    """`i_so >= 0` is not the same test as "is a subhalo".

    An object may name a parent index the catalog never prints. The project's
    convention -- load_rockstar_ascii's -- calls that a host, and on set8 the two
    readings differ by 993 SR2 subhalos, the same order as the effect this
    experiment measures.
    """
    got = report.parent_ids_from_columns([10, 11], [0, 1], [-1, 77])
    assert got.tolist() == [-1, -1]
    assert (got >= 0).sum() == 0
    # the naive test would have called row 1 a subhalo
    assert (np.asarray([-1, 77]) >= 0).sum() == 1


def test_parent_remap_uses_printed_ids_not_row_numbers(report):
    """`idx` is an internal index; the parent reported must be the printed id."""
    got = report.parent_ids_from_columns([500, 501], [3, 4], [-1, 3])
    assert got.tolist() == [-1, 500]


def test_parent_remap_handles_an_empty_catalog(report):
    assert report.parent_ids_from_columns([], [], []).tolist() == []


def test_parent_remap_rejects_ragged_columns(report):
    with pytest.raises(ValueError):
        report.parent_ids_from_columns([1, 2], [0], [-1, 0])
