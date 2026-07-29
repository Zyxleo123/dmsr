"""Stage 3: the checkpoint-compatible generator with explicit noise control."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from cosmo_sr.tts.srs_noise import (
    NOISE_SITES,
    STAGE_SITES,
    ControlledG,
    load_controlled_generator,
    noise_site_layout,
    site_shapes,
)

SMALL = dict(chan_base=16, chan_min=8, chan_max=16)
LR_SIZE = 10          # -> output 8 * 10 - 42 = 38


def _generator(seed: int = 0, noise_scale: float = 0.25) -> ControlledG:
    """A tiny SR2-shaped generator whose noise actually moves the output."""
    torch.manual_seed(seed)
    g = ControlledG(6, 6, 8, **SMALL).eval()
    with torch.no_grad():
        for name, p in g.named_parameters():
            if name.endswith(".std"):
                p.copy_(torch.randn_like(p) * noise_scale)
    return g


def _lr(seed: int = 1, size: int = LR_SIZE) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, 6, size, size, size)


# --------------------------------------------------------------------------- #
# Parity with the read-only upstream implementation
# --------------------------------------------------------------------------- #
def _upstream_models():
    srs = Path(__file__).resolve().parents[2] / "external" / "SRS-map2map"
    if not (srs / "map2map" / "models" / "srsgan.py").exists():
        pytest.skip("external/SRS-map2map not available")
    if str(srs) not in sys.path:
        sys.path.insert(0, str(srs))
    from map2map import models
    return models


def test_state_dict_keys_match_upstream():
    models = _upstream_models()
    up = models.G(6, 6, 8, **SMALL)
    ours = ControlledG(6, 6, 8, **SMALL)
    assert set(up.state_dict()) == set(ours.state_dict())
    for k, v in up.state_dict().items():
        assert v.shape == ours.state_dict()[k].shape


def test_matches_upstream_bit_for_bit_under_the_same_global_seed():
    """With no explicit noise we must draw the same values in the same order."""
    models = _upstream_models()
    up = models.G(6, 6, 8, **SMALL).eval()
    ours = ControlledG(6, 6, 8, **SMALL).eval()
    sd = up.state_dict()
    for k in list(sd):
        if k.endswith(".std"):
            sd[k] = torch.randn_like(sd[k]) * 0.25
    up.load_state_dict(sd)
    ours.load_state_dict(sd)

    x = _lr()
    for seed in (0, 7):
        torch.manual_seed(seed)
        with torch.no_grad():
            y_up = up(x)
        torch.manual_seed(seed)
        with torch.no_grad():
            y_ours = ours(x)
        assert torch.equal(y_up, y_ours)


def test_pretrained_checkpoint_loads_without_the_upstream_package():
    ckpt = Path(__file__).resolve().parents[2] / "external" / "SRS-map2map" / "SRmodel" / "G_z0.pt"
    if not ckpt.exists():
        pytest.skip("pretrained G_z0.pt not available")
    g = load_controlled_generator(ckpt, scale_factor=8, device=torch.device("cpu"))
    assert g.num_blocks == 3
    assert len(list(g.parameters())) == 26


# --------------------------------------------------------------------------- #
# Explicit noise
# --------------------------------------------------------------------------- #
def test_recorded_noise_replays_the_same_output():
    g, x = _generator(), _lr()
    with torch.no_grad():
        y, z = g(x, record=True)
        y_replay = g(x, noise=z)
    assert torch.equal(y, y_replay)
    assert set(z) == set(NOISE_SITES)


def test_stage_dict_and_sequence_forms_agree():
    g, x = _generator(), _lr()
    with torch.no_grad():
        y, z = g(x, record=True)
        by_stage = g(x, noise={s: [z[a], z[b]] for s, (a, b) in STAGE_SITES.items()})
        by_seq = g(x, noise=[z[s] for s in NOISE_SITES])
    assert torch.equal(y, by_stage)
    assert torch.equal(y, by_seq)


def test_noise_tensors_have_the_expected_spatial_shapes():
    g, x = _generator(), _lr()
    with torch.no_grad():
        _, z = g(x, record=True)
    expected = site_shapes(LR_SIZE, 8)
    assert {k: tuple(v.shape) for k, v in z.items()} == expected
    # coarse -> fine: resolution doubles every stage
    assert [lay.res for lay in noise_site_layout(LR_SIZE, 8)] == [1, 2, 2, 4, 4, 8]


def test_omitting_noise_preserves_the_stochastic_distribution():
    """Partial injection must leave the other sites sampling as before."""
    g, x = _generator(), _lr()
    with torch.no_grad():
        _, z = g(x, record=True)
        outs = [g(x, noise={"z0": z["z0"]}) for _ in range(6)]
    stack = torch.stack(outs)
    assert float(stack.std(dim=0).mean()) > 0, "unfixed sites stopped being random"


@pytest.mark.parametrize("site", list(NOISE_SITES))
def test_perturbing_one_site_changes_the_output(site):
    g, x = _generator(), _lr()
    with torch.no_grad():
        _, z = g(x, record=True)
        y = g(x, noise=z)
        z2 = dict(z)
        z2[site] = torch.randn_like(z2[site])
        y2 = g(x, noise=z2)
    assert float((y - y2).abs().max()) > 0


def _effect_radius(delta: torch.Tensor, frac: float = 0.9) -> float:
    """Radius around the peak response containing ``frac`` of the total |delta|."""
    d = delta.abs().sum(dim=(0, 1))
    n = d.shape[-1]
    flat_peak = int(torch.argmax(d))
    peak = torch.tensor([flat_peak // (n * n), (flat_peak // n) % n, flat_peak % n])
    ax = torch.arange(n)
    grids = torch.meshgrid(ax, ax, ax, indexing="ij")
    r = torch.sqrt(sum((gr - p).float() ** 2 for gr, p in zip(grids, peak)))
    order = torch.argsort(r.reshape(-1))
    cum = torch.cumsum(d.reshape(-1)[order], dim=0)
    cut = int(torch.searchsorted(cum, frac * cum[-1]))
    return float(r.reshape(-1)[order][min(cut, len(order) - 1)])


def test_coarse_noise_has_a_broader_spatial_effect_than_fine_noise():
    """A single coarse-lattice cell should influence a wider region than a fine one."""
    g, x = _generator(), _lr(size=14)
    with torch.no_grad():
        _, z = g(x, record=True)
        y0 = g(x, noise=z)
        radii = {}
        for site in ("z0", "z5"):
            z2 = {k: v.clone() for k, v in z.items()}
            c = z2[site].shape[-1] // 2
            z2[site][0, 0, c, c, c] += 25.0        # one cell, large kick
            radii[site] = _effect_radius(g(x, noise=z2) - y0)
    assert radii["z0"] > radii["z5"], radii


def test_gradients_reach_every_noise_tensor():
    g, x = _generator(), _lr()
    with torch.no_grad():
        _, z0 = g(x, record=True)
    z = {k: v.clone().requires_grad_(True) for k, v in z0.items()}
    y = g(x, noise=z)
    (y ** 2).mean().backward()
    for site, t in z.items():
        assert t.grad is not None, site
        assert float(t.grad.abs().sum()) > 0, site
