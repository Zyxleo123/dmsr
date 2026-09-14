"""The LR-host conditioning channel, and the three things that must be true of it.

1. ZERO-INIT EQUIVALENCE. A ``cond_chan > 0`` generator loaded from the frozen
   checkpoint must be bit-identical to the frozen one at step 0, for any
   conditioning input. Every baseline comparison in this pipeline -- ``base`` in
   ``host_forward``, the ``frozen`` array in the gate's ``tiles.npz``, the
   low-k/lr-consistency guards -- assumes it. If it were only approximately
   true, every "the arm did X relative to frozen" number would carry an
   unmeasured offset.

2. CHANNEL / TILE ALIGNMENT. The conditioning crop and the LR crop must be the
   same tile, padding included. A half-cell or one-tile slip would hand the
   generator a coherent, plausible, WRONG host coordinate, and nothing
   downstream would raise.

3. REPLAY PARITY. ``export_gather_tiles`` rebuilds the channels from a run's
   stored config to score it. If the rebuild differs from what was trained, the
   Rockstar gate scores a different model and reports it under this arm's name.
   That bug has bitten this line before, which is why it is pinned here rather
   than argued in a comment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.data.crops import periodic_crop  # noqa: E402
from cosmo_sr.features.lr_host_channel import (  # noqa: E402
    CHANNEL_NAMES, LRHostChannelConfig, assign_hosts, build_channel_field,
    gate_hosts, load_channel_field, parse_channels, save_channel_field,
    tile_cond_crop,
)
from cosmo_sr.train.sr2_finetune_data import (  # noqa: E402
    SR2TileGeometry, tile_lr_crop,
)
from cosmo_sr.train.sr2_unfreeze import (  # noqa: E402
    RUNG_ORDER, group_of_parameter, parameter_groups, rung_names,
)
from cosmo_sr.tts.sampling import super_resolve_srs_seeded  # noqa: E402
from cosmo_sr.tts.srs_noise import (  # noqa: E402
    ControlledG, load_cond_state_dict, site_shapes,
)

# A small generator: scale_factor 8 (three blocks, the deployed geometry) with a
# narrow channel stack, so a forward pass is milliseconds rather than seconds.
SMALL = dict(chan_base=8, chan_min=4, chan_max=8)


def _pair(cond_chan: int, seed: int = 0):
    """``(frozen, conditioned)`` -- the same weights, one with cond columns."""
    torch.manual_seed(seed)
    frozen = ControlledG(6, 6, 8, **SMALL).eval()
    cond = ControlledG(6, 6, 8, cond_chan=cond_chan, **SMALL).eval()
    load_cond_state_dict(cond, frozen.state_dict())
    return frozen, cond


# --------------------------------------------------------------------------- #
# 1. Zero-init equivalence
# --------------------------------------------------------------------------- #
def test_cond_zero_init_matches_frozen_bit_for_bit():
    frozen, cond = _pair(3)
    x = torch.randn(2, 6, 14, 14, 14)
    # EXPLICIT noise, so the two forwards are the same realisation and any
    # difference is the conditioning rather than the RNG.
    noise = {k: torch.randn(*v) for k, v in site_shapes(14, 8, batch=2).items()}

    a = frozen(x, noise=noise)
    # A LOUD conditioning input: if any of it leaked through, the difference
    # would not be subtle.
    c = torch.randn(2, 3, 14, 14, 14) * 100.0
    b = cond(x, noise=noise, cond=c)
    assert torch.equal(a, b), "zero-initialised cond columns changed the output"


def test_cond_columns_stop_being_inert_once_trained():
    """The other half of the claim: the channel is WIRED IN, not decorative."""
    _frozen, cond = _pair(3)
    with torch.no_grad():
        cond.cond0.weight.normal_(0.0, 0.5)
    x = torch.randn(1, 6, 14, 14, 14)
    noise = {k: torch.randn(*v) for k, v in site_shapes(14, 8, batch=1).items()}
    a = cond(x, noise=noise, cond=torch.zeros(1, 3, 14, 14, 14))
    b = cond(x, noise=noise, cond=torch.ones(1, 3, 14, 14, 14))
    assert not torch.allclose(a, b)


def test_the_pretrained_keys_still_load_strictly():
    """``cond0`` may be missing. Nothing else may be."""
    frozen, _c = _pair(3)
    sd = frozen.state_dict()
    G = ControlledG(6, 6, 8, cond_chan=2, **SMALL)
    load_cond_state_dict(G, sd)                       # fine: only cond0 missing

    bad = {k: v for k, v in sd.items() if "block0" not in k}
    with pytest.raises(RuntimeError, match="missing"):
        load_cond_state_dict(ControlledG(6, 6, 8, cond_chan=2, **SMALL), bad)

    extra = dict(sd)
    extra["not_a_real_key"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        load_cond_state_dict(ControlledG(6, 6, 8, cond_chan=2, **SMALL), extra)


def test_a_trained_cond0_is_restored_not_re_zeroed():
    """Resuming or exporting a conditioned run must keep its trained columns."""
    _f, cond = _pair(3)
    with torch.no_grad():
        cond.cond0.weight.fill_(0.25)
    G = ControlledG(6, 6, 8, cond_chan=3, **SMALL)
    load_cond_state_dict(G, cond.state_dict())
    assert torch.equal(G.cond0.weight, cond.cond0.weight)


def test_a_missing_or_surplus_cond_is_refused_not_ignored():
    _frozen, cond = _pair(3)
    x = torch.randn(1, 6, 14, 14, 14)
    noise = {k: torch.randn(*v) for k, v in site_shapes(14, 8, batch=1).items()}
    with pytest.raises(ValueError, match="needs a conditioning stack"):
        cond(x, noise=noise)
    with pytest.raises(ValueError, match="cond_chan=0"):
        ControlledG(6, 6, 8, **SMALL)(x, noise=noise,
                                      cond=torch.zeros(1, 3, 14, 14, 14))
    with pytest.raises(ValueError, match="channels"):
        cond(x, noise=noise, cond=torch.zeros(1, 5, 14, 14, 14))
    with pytest.raises(ValueError, match="spatial shape"):
        cond(x, noise=noise, cond=torch.zeros(1, 3, 8, 8, 8))


# --------------------------------------------------------------------------- #
# The rung question: a frozen cond0 would run the arm as its own control
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rung", RUNG_ORDER)
def test_cond_columns_are_trainable_at_every_rung(rung):
    _f, cond = _pair(3)
    names = rung_names(cond, rung)
    assert "cond0.weight" in names, (
        f"rung {rung} freezes cond0, which would train the conditioned arm as "
        "the unconditioned control")
    groups = parameter_groups(cond, rung)
    assert cond.cond0.weight.requires_grad
    by_name = {g["name"]: g for g in groups}
    assert "cond" in by_name
    # Strictly the largest rate: cond0 starts at zero and has nothing to protect.
    assert by_name["cond"]["lr"] >= max(g["lr"] for g in groups)


def test_an_unconditioned_model_has_no_cond_group():
    frozen, _c = _pair(3)
    assert all(g["name"] != "cond" for g in parameter_groups(frozen, "fine"))
    assert group_of_parameter("cond0.weight") == "cond"


# --------------------------------------------------------------------------- #
# 2. Channel / tile alignment
# --------------------------------------------------------------------------- #
def test_the_cond_crop_is_the_same_tile_as_the_lr_crop():
    """Same start, same pad, same periodic wrap -- checked by construction.

    The LR field's channel 0 is copied into the conditioning stack, so the two
    crops must come back numerically identical. A one-tile or one-cell slip
    would show here and nowhere else downstream.
    """
    geom = SR2TileGeometry()
    rng = np.random.default_rng(0)
    lr = rng.normal(size=(6, geom.ng_lr, geom.ng_lr, geom.ng_lr)).astype(np.float32)
    cond = np.stack([lr[0], lr[1], lr[2]]).astype(np.float32)
    for tile in (0, 1, 7, 63, 64, 511):
        a = tile_lr_crop(lr, tile, geom)
        b = tile_cond_crop(cond, tile, geom)
        assert b.shape == (3,) + a.shape[1:]
        for ch in range(3):
            assert np.array_equal(b[ch], a[ch])


def test_the_whole_box_path_crops_the_channels_the_same_way():
    """``super_resolve_srs_seeded`` and the training crop must agree exactly.

    Not a similarity claim: the two are the same function of the same tile, and
    the gate compares a field made by one against a run trained by the other.
    """
    geom = SR2TileGeometry(ng_lr=16, nsplit=2, pad=3)
    rng = np.random.default_rng(1)
    cond_box = rng.normal(size=(2, 16, 16, 16)).astype(np.float32)
    for tile in range(geom.n_tiles):
        start = [(tile // 4) * 8, ((tile // 2) % 2) * 8, (tile % 2) * 8]
        want = periodic_crop(cond_box, start, geom.chunk, pad=geom.pad)
        got = tile_cond_crop(cond_box, tile, geom)
        assert np.array_equal(got, want)


def test_a_conditioned_generator_refuses_a_missing_whole_box_stack():
    _f, cond = _pair(2)
    lr = np.zeros((6, 16, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="cond_field"):
        super_resolve_srs_seeded(cond, lr, 0, scale_factor=8, nsplit=2, pad=3,
                                 device=torch.device("cpu"))


# --------------------------------------------------------------------------- #
# The field itself
# --------------------------------------------------------------------------- #
class _Cat:
    """The three columns :func:`gate_hosts` reads, as a Rockstar catalog would."""

    def __init__(self, parent_ids, num_p, rvir, pos=None, mvir=None, ids=None):
        self.parent_ids = np.asarray(parent_ids)
        self.num_p = np.asarray(num_p)
        self.rvir = np.asarray(rvir, dtype=float)
        n = len(self.num_p)
        self.pos = np.zeros((n, 3)) if pos is None else np.asarray(pos, dtype=float)
        self.mvir = np.ones(n) * 1e13 if mvir is None else np.asarray(mvir, dtype=float)
        self.ids = np.arange(n) if ids is None else np.asarray(ids)


def test_the_gate_is_in_hr_particles_not_lr_ones():
    """20000 HR particles is a ~39-particle LR floor at 512/64, not 20000."""
    cat = _Cat(parent_ids=[-1, -1, -1, 0],
               num_p=[38, 40, 1000, 5000],
               rvir=[500.0, 500.0, 900.0, 100.0])
    keep = gate_hosts(cat, min_hr_num_p=20000, ng_lr=64, ng_hr=512)
    assert keep.tolist() == [1, 2], "the gate is not converting by mass ratio"


def test_a_subhalo_is_never_conditioned_on_directly():
    cat = _Cat(parent_ids=[-1, 0], num_p=[1000, 1000], rvir=[500.0, 500.0])
    assert gate_hosts(cat, min_hr_num_p=20000).tolist() == [0]


def test_assignment_minimises_scaled_distance_not_raw_distance():
    """A site 3 Mpc from a cluster belongs to the cluster, not to the tiny
    group 2 Mpc away that it is nowhere near virialised with."""
    sites = np.array([[5.0, 0.0, 0.0]])
    hosts = np.array([[3.0, 0.0, 0.0],      # a small group, 2 Mpc away
                      [0.0, 0.0, 0.0]])     # a cluster, 5 Mpc away
    rvir = np.array([0.2, 4.0])
    idx, u = assign_hosts(sites, hosts, rvir, boxsize_mpc_h=100.0)
    assert idx.tolist() == [1]
    assert u[0] == pytest.approx(1.25)


def test_assignment_is_periodic():
    sites = np.array([[99.0, 0.0, 0.0]])
    hosts = np.array([[1.0, 0.0, 0.0]])
    idx, u = assign_hosts(sites, hosts, np.array([1.0]), boxsize_mpc_h=100.0)
    assert idx.tolist() == [0]
    assert u[0] == pytest.approx(2.0), "the minimum image was not taken"


def _tiny_field(cfg, ng=4, boxsize=100.0, rvir=30.0):
    """One central host on a 4^3 lattice. At rvir=30 the eight cells nearest the
    centre land inside 1 R_vir and the rest do not, so both branches of the
    valid mask are exercised."""
    q = (np.arange(ng) + 0.5) * (boxsize / ng)
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1)
    sites = grid.reshape(-1, 3)
    hosts = np.array([[boxsize / 2, boxsize / 2, boxsize / 2]])
    return build_channel_field(sites, hosts, np.array([float(rvir)]),
                               np.array([1e14]), cfg, ng_lr=ng,
                               boxsize_mpc_h=boxsize)


def test_invalid_sites_read_far_away_not_at_the_centre():
    """Fill values matter: 0 in ``r_over_rvir`` would read as "at the host
    centre", which is the opposite of what an unassigned site means."""
    cfg = LRHostChannelConfig(channels=("r_over_rvir", "log_mhost", "valid"),
                              r_max_rvir=1.0)
    field, stats = _tiny_field(cfg)
    r, logm, valid = field
    outside = valid == 0
    assert outside.any(), "the test needs some site beyond r_max"
    assert np.all(r[outside] == pytest.approx(cfg.r_max_rvir))
    assert np.all(logm[outside] == 0.0)
    assert 0.0 < stats["valid_fraction"] < 1.0


def test_the_stack_is_ordered_as_the_config_says():
    a = LRHostChannelConfig(channels=("r_over_rvir", "valid"))
    b = LRHostChannelConfig(channels=("valid", "r_over_rvir"))
    fa, _ = _tiny_field(a)
    fb, _ = _tiny_field(b)
    assert np.array_equal(fa[0], fb[1]) and np.array_equal(fa[1], fb[0])


def test_the_lattice_reshape_is_c_order_like_particle_ids():
    """Site k of ``field_to_particles`` is lattice cell ``unravel_index(k)``.

    If this were F order the channel would be transposed against the LR field
    and every host coordinate would be mirrored into the wrong octant.
    """
    cfg = LRHostChannelConfig(channels=("r_over_rvir",))
    ng, boxsize = 4, 100.0
    q = (np.arange(ng) + 0.5) * (boxsize / ng)
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1)
    sites = grid.reshape(-1, 3)
    hosts = np.array([[q[0], q[0], q[0]]])
    field, _ = build_channel_field(sites, hosts, np.array([100.0]),
                                   np.array([1e14]), cfg, ng_lr=ng,
                                   boxsize_mpc_h=boxsize)
    # Site 0 is cell (0,0,0), which is the host itself: r = 0 there and nowhere
    # else is smaller.
    assert field[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert np.argmin(field[0]) == 0


def test_rhat_points_from_the_site_toward_the_host():
    cfg = LRHostChannelConfig(
        channels=("rhat_x", "rhat_y", "rhat_z", "valid"), r_max_rvir=10.0)
    sites = np.array([[10.0, 50.0, 50.0]] + [[0.0, 0.0, 0.0]] * 63)
    hosts = np.array([[20.0, 50.0, 50.0]])
    field, _ = build_channel_field(sites, hosts, np.array([10.0]),
                                   np.array([1e14]), cfg, ng_lr=4,
                                   boxsize_mpc_h=100.0)
    flat = field.reshape(4, -1)
    assert flat[3, 0] == 1.0
    assert flat[0, 0] == pytest.approx(1.0)      # +x, toward the host
    assert flat[1, 0] == pytest.approx(0.0)
    assert flat[2, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 3. Replay parity: the cache refuses to be read as something it is not
# --------------------------------------------------------------------------- #
def test_a_cached_field_built_under_other_settings_is_refused(tmp_path):
    built = LRHostChannelConfig(channels=("r_over_rvir", "valid"),
                                min_hr_num_p=20000)
    field, _ = _tiny_field(built)
    p = save_channel_field(tmp_path / "set9.npz", field, built, {"box": "set9"})

    load_channel_field(p, built)                  # the matching read is fine
    wanted = LRHostChannelConfig(channels=("r_over_rvir", "valid"),
                                 min_hr_num_p=5000)
    with pytest.raises(ValueError, match="different settings"):
        load_channel_field(p, wanted)
    reordered = LRHostChannelConfig(channels=("valid", "r_over_rvir"))
    with pytest.raises(ValueError, match="different settings"):
        load_channel_field(p, reordered)


def test_a_missing_field_names_the_job_that_builds_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="submit_lr_host_channel.sh"):
        load_channel_field(tmp_path / "nope.npz")


def test_the_config_round_trips_through_json():
    cfg = LRHostChannelConfig(channels=("log_mhost", "valid"), mode="membership",
                              min_hr_num_p=1234, r_max_rvir=2.5)
    back = LRHostChannelConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back == cfg


def test_a_typo_in_a_channel_name_is_refused():
    with pytest.raises(ValueError, match="unknown conditioning channel"):
        parse_channels("r_over_rvir,log_mhsot")
    with pytest.raises(ValueError, match="repeated"):
        parse_channels("valid,valid")
    assert parse_channels(" r_over_rvir  valid ") == ("r_over_rvir", "valid")
    assert parse_channels("") == ()
    assert set(parse_channels(",".join(CHANNEL_NAMES))) == set(CHANNEL_NAMES)


# --------------------------------------------------------------------------- #
# The production path end to end: a built field -> a tile crop -> forward_tiles
# --------------------------------------------------------------------------- #
def test_forward_tiles_feeds_the_crop_and_still_matches_frozen_at_step_0(tmp_path):
    """The whole chain the trainer actually runs, on one synthetic box.

    Separately each link is pinned above. This is the composition, because the
    failure this line cannot afford is the one where every piece is right and
    the assembly hands the generator a stack it never asked for.
    """
    from overfit_host_mse import forward_tiles, stack_cond

    geom = SR2TileGeometry(ng_lr=16, nsplit=2, pad=3)
    cfg = LRHostChannelConfig(channels=("r_over_rvir", "log_mhost", "valid"))
    ng = geom.ng_lr
    q = (np.arange(ng) + 0.5) * (100.0 / ng)
    sites = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    field, stats = build_channel_field(
        sites, np.array([[50.0, 50.0, 50.0]]), np.array([20.0]),
        np.array([1e14]), cfg, ng_lr=ng, boxsize_mpc_h=100.0)
    assert 0.0 < stats["valid_fraction"] < 1.0
    p = save_channel_field(tmp_path / "setX.npz", field, cfg, stats)
    back, _c, _m = load_channel_field(p, cfg)
    assert np.array_equal(back, field)

    frozen, cond = _pair(cfg.n_chan)
    tiles = [0, 1]
    noise = {k: torch.randn(*v) for k, v in
             site_shapes(geom.lr_size, 8, batch=1).items()}
    data = {
        t: {"lr": torch.randn(6, geom.lr_size, geom.lr_size, geom.lr_size),
            "noise": {k: v.clone() for k, v in noise.items()},
            "cond": torch.from_numpy(
                np.ascontiguousarray(tile_cond_crop(back, t, geom))).float()}
        for t in tiles
    }
    c = stack_cond(data, tiles)
    assert c.shape == (len(tiles), cfg.n_chan) + (geom.lr_size,) * 3

    # THE SAME dict to both models, which is what a conditioned run actually
    # does on every step: one `data`, forwarded through the tuned candidate and
    # through the frozen reference. The frozen one has no conditioning columns
    # by design. An earlier version of this test hand-stripped "cond" before the
    # frozen call, which hid the bug that killed job 47460 on its first launch.
    with torch.no_grad():
        a = forward_tiles(frozen, data, tiles, geom)
        b = forward_tiles(cond, data, tiles, geom)
    assert torch.equal(a, b), (
        "the assembled conditioned path is not the frozen path at step 0")


def test_one_data_dict_serves_both_the_candidate_and_the_frozen_reference():
    """`stack_cond` decides by MODEL, not by data. Job 47460 died here.

    The frozen reference is deliberately `cond_chan=0` -- it is the deployed
    SR2, and every baseline is a difference against it -- so handing it the
    stack that the candidate consumes is an error, not a no-op.
    """
    from overfit_host_mse import stack_cond

    frozen, cond = _pair(3)
    tiles = [0]
    data = {0: {"cond": torch.randn(3, 14, 14, 14)}}
    assert stack_cond(data, tiles, cond) is not None
    assert stack_cond(data, tiles, frozen) is None, (
        "the frozen reference was handed conditioning channels it has no "
        "columns for")
    # And the safety property is NOT traded away: columns but no stack still
    # raises, so a conditioned arm cannot quietly run as its own control.
    assert stack_cond({0: {}}, tiles, cond) is None
    with pytest.raises(ValueError, match="needs a conditioning stack"):
        cond(torch.randn(1, 6, 14, 14, 14),
             noise={k: torch.randn(*v)
                    for k, v in site_shapes(14, 8, batch=1).items()},
             cond=None)
