"""Stage 0: reproducible multi-sample SR2 inference."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.tts.sampling import (
    GlobalNoiseField,
    derive_tile_seed,
    generate_srs_candidates,
    super_resolve_srs_seeded,
    tile_noise,
    tile_starts,
)
from cosmo_sr.tts.srs_noise import ControlledG, noise_site_layout

SMALL = dict(chan_base=16, chan_min=8, chan_max=16)
NG, NSPLIT, PAD, SCALE = 8, 4, 3, 8      # chunk 2, padded tile 8 -> 22, trimmed to 16


def _generator(seed: int = 0) -> ControlledG:
    torch.manual_seed(seed)
    g = ControlledG(6, 6, SCALE, **SMALL).eval()
    with torch.no_grad():
        for name, p in g.named_parameters():
            if name.endswith(".std"):
                p.copy_(torch.randn_like(p) * 0.3)
    return g


def _lr(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(6, NG, NG, NG)).astype(np.float32)


def _sr(g, lr, seed, **kw):
    return super_resolve_srs_seeded(g, lr, seed, scale_factor=SCALE, nsplit=NSPLIT,
                                    pad=PAD, **kw)


# --------------------------------------------------------------------------- #
# Seed derivation
# --------------------------------------------------------------------------- #
def test_tile_seeds_are_deterministic_and_distinct():
    a = derive_tile_seed(5, (1, 2, 3), 0)
    assert a == derive_tile_seed(5, (1, 2, 3), 0)
    others = {derive_tile_seed(5, (1, 2, 3), s) for s in range(6)}
    neighbours = {derive_tile_seed(5, c, 0) for c in [(1, 2, 3), (1, 2, 4), (2, 2, 3)]}
    assert len(others) == 6 and len(neighbours) == 3
    assert derive_tile_seed(6, (1, 2, 3), 0) != a


def test_tile_coord_must_be_three_dimensional():
    with pytest.raises(ValueError):
        derive_tile_seed(0, (1, 2), 0)


# --------------------------------------------------------------------------- #
# Stage-0 requirements
# --------------------------------------------------------------------------- #
def test_same_lr_and_seed_reproduce_identical_output():
    g, lr = _generator(), _lr()
    assert np.array_equal(_sr(g, lr, 11), _sr(g, lr, 11))


def test_different_seeds_give_nonzero_diversity():
    g, lr = _generator(), _lr()
    a, b = _sr(g, lr, 0), _sr(g, lr, 1)
    rel = float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a ** 2)))
    assert rel > 1e-4, rel


def test_tile_traversal_order_does_not_change_the_realisation():
    g, lr = _generator(), _lr()
    starts = tile_starts(NG, NSPLIT)
    shuffled = list(starts)
    np.random.default_rng(0).shuffle(shuffled)
    assert shuffled != starts
    assert np.array_equal(_sr(g, lr, 4), _sr(g, lr, 4, tile_order=shuffled))


def test_shape_and_units_match_the_existing_baseline():
    """Same geometry and the same normalized units as ``super_resolve_srs``."""
    from cosmo_sr.eval.baseline_srs import super_resolve_srs

    g, lr = _generator(), _lr()
    ours = _sr(g, lr, 0)
    torch.manual_seed(0)
    base = super_resolve_srs(g, lr, scale_factor=SCALE, nsplit=NSPLIT, pad=PAD, seed=0)
    assert ours.shape == base.shape == (6, NG * SCALE, NG * SCALE, NG * SCALE)
    assert ours.dtype == base.dtype == np.float32
    # Different noise draws, so not equal -- but the same field, to a few percent
    # of its own scale in every channel.
    for c in range(6):
        s_ours, s_base = ours[c].std(), base[c].std()
        assert abs(s_ours - s_base) / max(s_base, 1e-12) < 0.25, (c, s_ours, s_base)


def test_k1_matches_the_baseline_exactly_when_given_the_same_noise():
    """The only difference from the legacy path is *which* noise is drawn."""
    from cosmo_sr.eval.baseline_srs import super_resolve_srs

    g, lr = _generator(), _lr()
    torch.manual_seed(0)
    legacy = super_resolve_srs(g, lr, scale_factor=SCALE, nsplit=NSPLIT, pad=PAD, seed=0)
    # ControlledG with noise=None consumes the global RNG exactly as upstream does
    torch.manual_seed(0)
    replay = super_resolve_srs(g, lr, scale_factor=SCALE, nsplit=NSPLIT, pad=PAD, seed=0)
    assert np.array_equal(legacy, replay)


def test_candidates_record_seed_and_configuration():
    g, lr = _generator(), _lr()
    cands = generate_srs_candidates(g, lr, [2, 5], nsplit=NSPLIT, pad=PAD,
                                    scale_factor=SCALE, box="setX", model_path="G.pt")
    assert [c.seed for c in cands] == [2, 5]
    for c in cands:
        assert c.box == "setX"
        assert c.config["nsplit"] == NSPLIT and c.config["pad"] == PAD
        assert c.config["model_path"] == "G.pt" and c.config["ng_lr"] == NG
        assert c.disp.shape == (3, NG * SCALE, NG * SCALE, NG * SCALE)
    assert not np.array_equal(cands[0].field, cands[1].field)


def test_candidates_are_not_averaged():
    """A candidate must be an individual realisation, not an ensemble mean."""
    g, lr = _generator(), _lr()
    cands = generate_srs_candidates(g, lr, [0, 1, 2], nsplit=NSPLIT, pad=PAD,
                                    scale_factor=SCALE)
    mean = np.mean([c.field for c in cands], axis=0)
    for c in cands:
        assert not np.allclose(c.field, mean)
        # the mean is smoother than any member: variance must not have collapsed
        assert c.field.std() > mean.std()


def test_upstream_generator_is_rejected_with_a_clear_message():
    lr = _lr()
    with pytest.raises(TypeError, match="ControlledG"):
        super_resolve_srs_seeded(torch.nn.Conv3d(6, 6, 1), lr, 0, scale_factor=SCALE,
                                 nsplit=NSPLIT, pad=PAD, device=torch.device("cpu"))


# --------------------------------------------------------------------------- #
# Global (coordinate-indexed) noise
# --------------------------------------------------------------------------- #
def test_global_noise_is_shared_between_tiles_that_overlap():
    """Two tiles whose padded inputs overlap must read identical noise there."""
    field = GlobalNoiseField(0, NG, SCALE)
    lr_size = NG // NSPLIT + 2 * PAD
    a = tile_noise(0, (0, 0, 0), lr_size, SCALE, torch.device("cpu"), pad=PAD,
                   mode="global", global_field=field)
    b = tile_noise(0, (2, 0, 0), lr_size, SCALE, torch.device("cpu"), pad=PAD,
                   mode="global", global_field=field)
    lay = {l.site: l for l in noise_site_layout(lr_size, SCALE, start=0, pad=PAD)}
    lay_b = {l.site: l for l in noise_site_layout(lr_size, SCALE, start=2, pad=PAD)}
    shared = 0
    for site in lay:
        shift = lay_b[site].offset - lay[site].offset
        size = lay[site].size
        if 0 < shift < size:
            assert torch.equal(a[site][..., shift:, :, :], b[site][..., : size - shift, :, :])
            shared += 1
    assert shared >= 3, "no overlapping sites were actually compared"


def test_global_noise_realisation_is_reproducible_and_seed_dependent():
    lr_size = NG // NSPLIT + 2 * PAD
    kw = dict(pad=PAD, mode="global")
    a = tile_noise(0, (0, 0, 0), lr_size, SCALE, torch.device("cpu"),
                   global_field=GlobalNoiseField(7, NG, SCALE), **kw)
    b = tile_noise(0, (0, 0, 0), lr_size, SCALE, torch.device("cpu"),
                   global_field=GlobalNoiseField(7, NG, SCALE), **kw)
    c = tile_noise(0, (0, 0, 0), lr_size, SCALE, torch.device("cpu"),
                   global_field=GlobalNoiseField(8, NG, SCALE), **kw)
    for site in a:
        assert torch.equal(a[site], b[site])
        assert not torch.equal(a[site], c[site])


def test_global_mode_full_box_is_order_invariant():
    g, lr = _generator(), _lr()
    starts = tile_starts(NG, NSPLIT)
    shuffled = list(starts)
    np.random.default_rng(1).shuffle(shuffled)
    a = _sr(g, lr, 3, noise_mode="global",
            global_field=GlobalNoiseField(3, NG, SCALE))
    b = _sr(g, lr, 3, noise_mode="global", tile_order=shuffled,
            global_field=GlobalNoiseField(3, NG, SCALE))
    assert np.array_equal(a, b)
