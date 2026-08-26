"""Pins for the gather fine-tune's objective wiring, verdict and eval figures.

The loop itself needs a GPU and a 3.2 GiB box, so what is pinned here is what
would make a run *wrong rather than absent*: that the four loss terms are the
functions they claim to be (in particular that the low-k anchor is the squared
form -- the un-squared one has an infinite derivative at step zero, which is
where every run starts), that the verdict cannot call a run a success while the
field is being distorted, and that the eval artifact really does redraw its
figures without the generator.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    for p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
              PROJECT_ROOT / "scripts" / "features"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    path = PROJECT_ROOT / "scripts" / "features" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("render_gather_slices")
M = _load("finetune_host_gather")


class _Args:
    """The argparse namespace the objective reads, with the shipped defaults."""

    w_gather = 1.0
    w_preserve = 1.0
    w_low = 1.0
    w_anchor = 0.1
    w_mse = 0.0
    gain_min = 0.05
    low_k_max = 0.02
    vdisp_tol = 0.15
    vbulk_tol = 0.5
    contrast_drop_max = 0.10


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #
def _fields(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(2, 6, n, n, n, generator=g) * 0.01
    hr = base + torch.randn(2, 6, n, n, n, generator=g) * 0.01
    return base, hr


class _Geom:
    scale_factor = 8


def test_low_k_term_is_the_squared_form_and_is_flat_at_zero():
    """At step zero the candidate IS the frozen field, and ``sqrt`` is not
    differentiable there -- the un-squared form produced NaN weights in three
    steps in the direct line. The squared form must be zero AND have zero
    gradient there."""
    base, _ = _fields()
    cand = base.clone().requires_grad_(True)
    a_c = M.block_average_torch(cand, 8)
    a_b = M.block_average_torch(base, 8).detach()
    low = (a_c - a_b).pow(2).mean() / a_b.pow(2).mean().clamp_min(1e-30)
    assert float(low.detach()) == pytest.approx(0.0, abs=1e-12)
    low.backward()
    assert torch.isfinite(cand.grad).all()
    assert float(cand.grad.abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_low_k_term_grows_with_an_lr_scale_distortion():
    base, _ = _fields()
    small = base + 1e-4
    big = base + 1e-2
    def low(c):
        a_c = M.block_average_torch(c, 8)
        a_b = M.block_average_torch(base, 8)
        return float((a_c - a_b).pow(2).mean() / a_b.pow(2).mean())
    assert low(small) < low(big)
    # The loop reports the gate-comparable RMS ratio, which is the square root
    # of the term it optimises -- the two must not drift apart.
    reported = float(max(low(big), 0.0) ** 0.5)
    assert reported == pytest.approx(low(big) ** 0.5, rel=1e-9)
    assert reported > float(max(low(small), 0.0) ** 0.5)


def test_mse_term_is_off_by_default():
    """The objective this experiment exists to replace must not be quietly on."""
    assert M.weights_of(_Args())["mse"] == 0.0
    assert M.weights_of(_Args())["gather"] == 1.0
    assert M.weights_of(_Args())["preserve"] == 1.0


def test_verdict_fails_a_run_that_blurs_the_unsupervised_field():
    """The gap that let an earlier run print ALL THREE HELD.

    low_k finished at 0.0187, inside its gate, while local peak structure away
    from the 43 supervised windows sat at 0.57 of the frozen generator's. A
    field gate that reads only the block-averaged scale cannot see that, and
    reported the run as clean.
    """
    frozen = {"compact_ratio": 0.20, "compact_ratio_median": 0.20}
    v = M.verdict(_history(0.55, 0.0187, preserve=0.57), frozen, _Args())
    assert v["moved"] and not v["field_preserved"]
    assert "BLURRED EVERYWHERE ELSE" in v["text"]
    ok = M.verdict(_history(0.55, 0.0187, preserve=0.97), frozen, _Args())
    assert ok["field_preserved"] and "ALL THREE HELD" in ok["text"]


def test_grad_norms_report_zero_for_disabled_terms_without_a_backward():
    p = torch.zeros(4, requires_grad=True)
    groups = [{"params": [p]}]
    terms = {"gather": (p * 2.0).sum(), "mse": (p * 5.0).sum()}
    out = M.grad_norms(terms, {"gather": 1.0, "mse": 0.0}, groups)
    assert out["gradnorm_mse"] == 0.0
    assert out["gradnorm_gather"] == pytest.approx(4.0)      # ||(2,2,2,2)||


# --------------------------------------------------------------------------- #
# The verdict: a gain bought with a distorted field is not a gain
# --------------------------------------------------------------------------- #
def _history(compact, low_k, vdisp=1.0, vbulk=0.1, preserve=1.0):
    return [{"step": 100, "compact_ratio": compact, "preserve_ratio": preserve,
             "compact_ratio_median": compact, "low_k_change": low_k,
             "highk_power_ratio": 0.9, "vdisp_ratio": vdisp,
             "vdisp_ratio_frozen": 0.6, "vbulk_offset": vbulk,
             "vbulk_offset_frozen": 1.2}]


def test_verdict_reads_all_three_axes_separately():
    frozen = {"compact_ratio": 0.20, "compact_ratio_median": 0.20}
    good = M.verdict(_history(0.55, 0.001), frozen, _Args())
    assert good["moved"] and good["field_preserved"] and good["kinematics_ok"]
    assert "ALL THREE HELD" in good["text"]

    bought = M.verdict(_history(0.55, 0.5), frozen, _Args())
    assert bought["moved"] and not bought["field_preserved"]
    assert "PAID FOR IT" in bought["text"]

    flat = M.verdict(_history(0.21, 0.001), frozen, _Args())
    assert not flat["moved"]
    assert "NO MOVEMENT" in flat["text"]


def test_verdict_refuses_to_call_a_density_only_gain_a_result():
    """The channel-swap failure mode: right places, wrong velocities."""
    frozen = {"compact_ratio": 0.20, "compact_ratio_median": 0.20}
    hot = M.verdict(_history(0.55, 0.001, vdisp=2.5), frozen, _Args())
    cold = M.verdict(_history(0.55, 0.001, vdisp=0.3), frozen, _Args())
    drifting = M.verdict(_history(0.55, 0.001, vbulk=3.0), frozen, _Args())
    for v in (hot, cold, drifting):
        assert v["moved"] and v["kinematics_ok"] is False
        assert "DENSITY WITHOUT KINEMATICS" in v["text"]


def test_verdict_says_so_when_velocity_was_not_measured():
    frozen = {"compact_ratio": 0.20, "compact_ratio_median": 0.20}
    h = _history(0.55, 0.001)
    h[0]["vdisp_ratio"] = None
    v = M.verdict(h, frozen, _Args())
    assert v["kinematics_ok"] is None
    assert "UNMEASURED" in v["text"]


def test_verdict_gain_is_relative_to_the_frozen_generator():
    v = M.verdict(_history(0.40, 0.0),
                  {"compact_ratio": 0.20, "compact_ratio_median": 0.20}, _Args())
    assert v["relative_gain"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The eval artifact redraws without the generator
# --------------------------------------------------------------------------- #
def _fake_npz(path, g=16, s=3):
    rng = np.random.default_rng(0)
    centre = rng.uniform(4, g - 4, size=(1, s, 3)).astype(np.float32)
    np.savez_compressed(
        path, step=700, tiles=np.array([37]),
        delta_out=rng.random((1, g, g, g), dtype=np.float32) * 10,
        delta_frozen=rng.random((1, g, g, g), dtype=np.float32) * 10,
        delta_hr=rng.random((1, g, g, g), dtype=np.float32) * 10,
        centre=centre, sigma=np.ones((1, s), dtype=np.float32),
        mask=np.ones((1, s), dtype=np.float32),
        num_p=np.full((1, s), 300), hr_compact=np.full((1, s), 200.0),
        halo_id=np.arange(s).reshape(1, s), host_id=271800, box="set8",
        cellsize_mpc_h=0.1953)


def test_eval_npz_renders_one_png_per_tile(tmp_path):
    npz = tmp_path / "step000700.npz"
    _fake_npz(npz)
    out = R.render_npz(npz, tmp_path / "slices", slab=3)
    assert len(out) == 1
    assert out[0].name == "step000700_tile037.png"
    assert out[0].stat().st_size > 1000


def test_projection_is_a_max_through_the_slab_not_a_slice():
    """A 366-particle subhalo is under one cell across; a single slice through
    the wrong plane misses it and would make a real gain look like nothing."""
    vol = np.zeros((8, 8, 8), dtype=np.float32)
    vol[4, 4, 6] = 999.0                       # off the mid-plane
    img = R._projection(vol, 2, 7)
    assert img[4, 4] == pytest.approx(np.log10(1000.0), rel=1e-6)
    assert img[0, 0] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# The loop body, end to end, without the generator
# --------------------------------------------------------------------------- #
def test_loop_body_runs_on_real_shapes_and_backpropagates(tmp_path):
    """Everything between the forward pass and the JSONL row, on 64^3 tiles.

    The generator needs a GPU and a 3.2 GiB box, but everything downstream of it
    is shape-sensitive plumbing across three modules -- the shared bulk offset,
    the padded target batch, the sub-batch selection, the density deposit and the
    eval artifact. This runs all of it on synthetic fields, so a shape or
    convention mistake fails here rather than eight hours into a GPU job.
    """
    from cosmo_sr.eval.density import valid_center_bulk
    from cosmo_sr.features.subhalo_gather import (
        GatherConfig, GatherTargets, attach_hr_reference, deposit_for_gather,
    )
    from cosmo_sr.reward.soft_structure import SoftStructureConfig

    n, b = 64, 2
    soft = SoftStructureConfig(region_fraction=0.5, grid_mult=1)
    cfg = GatherConfig.from_soft(soft, min_hr_compact=0.0)
    g = torch.Generator().manual_seed(3)
    base = torch.randn(b, 6, n, n, n, generator=g) * 0.02
    hr = base + torch.randn(b, 6, n, n, n, generator=g) * 0.01
    cand = base.clone().requires_grad_(True)

    # One shared origin for all three deposits -- the frozen field's.
    bulk = valid_center_bulk(base[:, 0:3], soft.cellsize_kpc_h, soft.dis_norm_kpc_h)
    hr_dep = deposit_for_gather(hr, soft, bulk=bulk)
    cand_dep = deposit_for_gather(cand, soft, bulk=bulk)
    cand_delta = cand_dep.delta
    assert cand_delta.shape == (b, 1, 32, 32, 32)

    centre = torch.tensor([[[12.0, 12.0, 12.0], [20.0, 18.0, 16.0]],
                           [[16.0, 16.0, 16.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    t = GatherTargets(centre=centre, sigma=torch.ones(b, 2), mask=mask,
                      hr_compact=torch.zeros(b, 2), hr_contrast=torch.zeros(b, 2),
                      hr_vbulk=torch.zeros(b, 2, 3), hr_vdisp=torch.ones(b, 2),
                      half_width=3, num_p=torch.full((b, 2), 300),
                      halo_id=torch.arange(2 * b).reshape(b, 2), tiles=[37, 38])
    t = attach_hr_reference(t, hr_dep, cfg)

    class A:
        n_bins, k_split = 8, 4.0
        w_gather, w_low, w_anchor, w_mse = 1.0, 1.0, 0.1, 0.0
        w_preserve = 1.0

    terms, diag = M.loss_terms(cand, base, hr, cand_dep, t, cfg, _Geom(), soft)
    # The addition that makes this a phase-space objective: the velocity
    # channels must receive gradient, not just the displacement ones.
    assert "vdisp_ratio" in diag
    assert set(terms) == {"gather", "preserve", "low", "anchor", "mse"}
    # No complement map was built here, so the preservation term must be inert
    # rather than guessing a reference.
    assert float(terms["preserve"]) == pytest.approx(0.0, abs=1e-9)
    assert diag["preserve_ratio"] == pytest.approx(1.0)
    total = sum(M.weights_of(A())[k] * v for k, v in terms.items())
    total.backward()
    assert torch.isfinite(cand.grad).all()
    assert float(cand.grad.abs().sum()) > 0.0
    assert diag["n_targets"] == int(t.mask.sum())

    # The padded slot must never contribute.
    one = t.select([1])
    assert float(one.mask.sum()) <= 1.0

    # And the eval artifact must render.
    npz = tmp_path / "step000100.npz"
    np.savez_compressed(
        npz, step=100, tiles=np.array([37, 38]),
        delta_out=cand_delta[:, 0].detach().numpy().astype(np.float32),
        delta_frozen=deposit_for_gather(base, soft, bulk=bulk).delta[:, 0]
        .numpy().astype(np.float32),
        delta_hr=hr_dep.delta[:, 0].numpy().astype(np.float32),
        centre=t.centre.numpy(), sigma=t.sigma.numpy(), mask=t.mask.numpy(),
        num_p=t.num_p.numpy(), hr_compact=t.hr_compact.numpy(),
        halo_id=t.halo_id.numpy(), host_id=271800, box="set8",
        cellsize_mpc_h=100.0 / 512.0)
    pngs = R.render_npz(npz, tmp_path / "slices", slab=4)
    assert len(pngs) == 2 and all(q.stat().st_size > 1000 for q in pngs)
