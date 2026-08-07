"""The zero-initialised head does NOT make a diffusion sample equal SR2.

``cosmo_sr.reward.model`` used to claim that because the output conv is
zero-initialised, "an untrained model predicts ``eps = 0`` and the composed
field starts exactly at the frozen SR2 output". The first half is true and the
second does not follow: DDIM starts from ``u_T ~ N(0, I)``, and with
``eps_hat = 0`` the step is ``u_next = (alpha_next / alpha_cur) * u``, so the
sample is the *initial noise rescaled* -- amplified, near ``t_max``, by
``1 / alpha(t_max)``. The only thing that recovers the frozen output bit for bit
is ``residual_scale = 0``, which short-circuits the composition.

The claim mattered because it licensed reading an untrained/failed sampler as a
harmless no-op. These tests measure the three facts instead of asserting them in
prose, and the docstring check keeps the wrong sentence from coming back.
"""
from __future__ import annotations

import re

import numpy as np
import pytest
import torch

from cosmo_sr.reward import model as model_mod
from cosmo_sr.reward.base import ResidualComposer, compose
from cosmo_sr.reward.diffusion import DiffusionConfig, ddim_sample
from cosmo_sr.reward.model import ResidualDenoiser


def tiny_model(**kw) -> ResidualDenoiser:
    """A CPU-sized denoiser with the real zero-init tail."""
    opts = dict(channels=2, scale_factor=2, width=4, num_levels=1,
                blocks_per_level=1, embed_dim=8, num_groups=2)
    opts.update(kw)
    return ResidualDenoiser(**opts).eval()


def test_zero_init_head_predicts_zero_eps():
    """The half of the old claim that IS true, so the test below is not vacuous."""
    m = tiny_model()
    u = torch.randn(1, 2, 8, 8, 8)
    out = m(u, torch.full((1,), 0.5),
            y_lr=torch.randn(1, 2, 4, 4, 4), psi_base=torch.randn(1, 2, 8, 8, 8))
    assert torch.count_nonzero(out) == 0


def test_zero_init_ddim_sample_is_not_zero():
    """eps_hat = 0 leaves the initial noise rescaled, not annihilated."""
    m = tiny_model()
    cond = {"y_lr": torch.randn(1, 2, 4, 4, 4), "psi_base": torch.randn(1, 2, 8, 8, 8)}
    cfg = DiffusionConfig(n_steps=8, x0_clip=4.0)
    gen = torch.Generator().manual_seed(0)
    u = ddim_sample(m, (1, 2, 8, 8, 8), cond, cfg, device=torch.device("cpu"),
                    generator=gen)

    assert torch.isfinite(u).all()
    # Not merely "nonzero at float noise level": the sample carries O(1)
    # whitened amplitude, i.e. a residual comparable to sigma_res.
    assert float(u.abs().max()) > 0.1
    assert float(u.pow(2).mean().sqrt()) > 0.05


def test_zero_init_ddim_amplitude_is_set_by_the_clip():
    """Without x0_clip the zero-head sample blows up: the bound sets the scale.

    This is the operational consequence of the corrected claim -- the amplitude
    of an untrained sample is a property of ``x0_clip``, not of the model -- so
    a run whose clip fraction is large is reporting the bound.
    """
    m = tiny_model()
    cond = {"y_lr": torch.zeros(1, 2, 4, 4, 4), "psi_base": torch.zeros(1, 2, 8, 8, 8)}
    shape = (1, 2, 8, 8, 8)
    dev = torch.device("cpu")

    clipped = ddim_sample(m, shape, cond, DiffusionConfig(n_steps=8, x0_clip=4.0),
                          device=dev, generator=torch.Generator().manual_seed(0))
    unclipped = ddim_sample(m, shape, cond, DiffusionConfig(n_steps=8, x0_clip=0.0),
                            device=dev, generator=torch.Generator().manual_seed(0))

    assert float(unclipped.abs().max()) > 4.0 * float(clipped.abs().max())


def test_clip_log_records_a_large_hit_fraction_for_a_zero_head():
    """The clip fraction is observable, and for a zero head it is not small."""
    m = tiny_model()
    cond = {"y_lr": torch.zeros(1, 2, 4, 4, 4), "psi_base": torch.zeros(1, 2, 8, 8, 8)}
    log: list = []
    ddim_sample(m, (1, 2, 8, 8, 8), cond, DiffusionConfig(n_steps=8, x0_clip=4.0),
                device=torch.device("cpu"),
                generator=torch.Generator().manual_seed(0), clip_log=log)
    assert log, "clip_log was requested but nothing was recorded"
    assert max(c["clip_fraction"] for c in log) > 0.5


@pytest.mark.parametrize("residual", [
    torch.full((1, 2, 4, 4, 4), 1e3),
    torch.full((1, 2, 4, 4, 4), float("nan")),
])
def test_only_residual_scale_zero_recovers_sr2_bit_exactly(residual):
    """``a = 0`` returns the SAME OBJECT, so no arithmetic can perturb it.

    NaN is in the parameter list deliberately: ``psi_base + 0 * NaN`` is NaN, so
    a short-circuit is the only implementation for which "residual disabled"
    means the frozen field even when the residual model has diverged.
    """
    base = torch.randn(1, 2, 4, 4, 4)
    out = compose(base, residual, residual_scale=0.0)
    assert out is base
    assert torch.equal(out, base)

    nonzero = compose(base, torch.ones_like(base), residual_scale=1e-6)
    assert not torch.equal(nonzero, base)


def test_composer_disabled_at_zero_scale():
    m = tiny_model()
    comp = ResidualComposer(residual=m, residual_scale=0.0)
    assert not comp.enabled
    base = torch.randn(1, 2, 8, 8, 8)
    assert comp.compose(base, torch.randn(1, 2, 8, 8, 8)) is base

    assert ResidualComposer(residual=m, residual_scale=1.0).enabled


# --------------------------------------------------------------------------- #
# Documentation regression
# --------------------------------------------------------------------------- #

#: Claims that are false as written. Each is a sentence pattern asserting that
#: the zero init (rather than ``residual_scale = 0``) recovers SR2.
BANNED = [
    r"zero[- ]init\w*[^.]{0,160}?composed field starts exactly at",
    r"eps\s*=\s*0[^.]{0,120}?composed field starts exactly at",
    r"zero[- ]init\w*[^.]{0,160}?(?<!not )(?<!does not )"
    r"(?:makes|means)[^.]{0,60}?(?:sample|residual)[^.]{0,40}?(?:is|equals?) zero",
]


def test_module_docstring_does_not_repeat_the_false_claim():
    doc = model_mod.__doc__ or ""
    for pat in BANNED:
        m = re.search(pat, doc, flags=re.IGNORECASE | re.DOTALL)
        assert m is None, (
            f"cosmo_sr.reward.model's docstring re-asserts that a zero-initialised "
            f"head makes the sample equal SR2:\n  ...{m.group(0)}...\n"
            f"Only residual_scale=0 does that; see this test module."
        )


def test_module_docstring_states_the_correct_condition():
    """The correction must be positively present, not merely the error absent."""
    doc = (model_mod.__doc__ or "").lower()
    assert "residual_scale" in doc
    assert re.search(r"only\s+``?residual_scale\s*=\s*0", doc), (
        "the docstring should say explicitly that only residual_scale=0 recovers "
        "the frozen SR2 output"
    )


def test_sampling_and_diffusion_docs_stay_consistent():
    """The two modules that already got this right must not regress either."""
    from cosmo_sr.reward import diffusion as diff_mod
    from cosmo_sr.reward import sampling as samp_mod

    for mod in (diff_mod.ddim_sample, samp_mod.sample_residual_box):
        doc = (mod.__doc__ or "").lower()
        assert "zero" in doc and ("clip" in doc or "residual_scale" in doc)
    for pat in BANNED:
        for doc in (diff_mod.__doc__ or "", samp_mod.__doc__ or ""):
            assert re.search(pat, doc, flags=re.IGNORECASE | re.DOTALL) is None
