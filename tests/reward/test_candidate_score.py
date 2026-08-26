"""The candidate score must reward collapse and reject dense-but-unbound clumps."""
from __future__ import annotations

import torch

from cosmo_sr.reward.candidate_score import (
    CandidateScoreConfig, SurvivalHead, candidate_features, feature_names,
    formation_score, unwrap_about,
)


def _clump(n: int, radius: float, sigma_v: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g) * radius + 50.0
    vel = torch.randn(n, 3, generator=g) * sigma_v
    center = torch.tensor([50.0, 50.0, 50.0])
    return pos, vel, center


def test_score_rises_as_the_clump_contracts():
    cfg = CandidateScoreConfig()
    center = torch.tensor([50.0, 50.0, 50.0])
    base, vel, _ = _clump(2000, radius=1.0, sigma_v=50.0)
    scores = []
    for shrink in (1.0, 0.7, 0.5, 0.3):
        pos = center.view(1, 3) + (base - center.view(1, 3)) * shrink
        s = formation_score(pos, vel, center, r_ref=0.5, cfg=cfg)
        scores.append(float(s))
    assert all(scores[i] < scores[i + 1] for i in range(len(scores) - 1)), scores


def test_binding_gate_prefers_cold_over_hot_at_equal_density():
    cfg = CandidateScoreConfig()
    pos, _, center = _clump(2000, radius=0.4, sigma_v=0.0)
    _, cold_v, _ = _clump(2000, radius=0.4, sigma_v=40.0, seed=1)
    _, hot_v, _ = _clump(2000, radius=0.4, sigma_v=600.0, seed=1)
    cold = float(formation_score(pos, cold_v, center, r_ref=0.4, cfg=cfg))
    hot = float(formation_score(pos, hot_v, center, r_ref=0.4, cfg=cfg))
    density_only_cold = float(
        formation_score(pos, cold_v, center, r_ref=0.4, cfg=cfg, use_binding=False))
    density_only_hot = float(
        formation_score(pos, hot_v, center, r_ref=0.4, cfg=cfg, use_binding=False))
    # Same positions -> density-only score is identical; the binding gate is the
    # only thing that can tell the cold clump from the hot one.
    assert abs(density_only_cold - density_only_hot) < 1e-5
    assert cold > hot


def test_score_is_differentiable_in_positions():
    cfg = CandidateScoreConfig()
    pos, vel, center = _clump(500, radius=0.6, sigma_v=50.0)
    pos = pos.clone().requires_grad_(True)
    s = formation_score(pos, vel, center, r_ref=0.5, cfg=cfg)
    (grad,) = torch.autograd.grad(s, pos)
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_unwrap_handles_the_periodic_boundary():
    box = 100.0
    center = torch.tensor([0.5, 0.5, 0.5])
    pos = torch.tensor([[99.5, 0.5, 0.5]])            # 1 Mpc away across the face
    uw = unwrap_about(pos, center.view(1, 3), box)
    assert torch.allclose(uw, torch.tensor([[-0.5, 0.5, 0.5]]), atol=1e-5)


def test_feature_vector_feeds_the_survival_head():
    cfg = CandidateScoreConfig()
    pos, vel, center = _clump(800, radius=0.5, sigma_v=80.0)
    f = candidate_features(pos, vel, center, r_ref=0.5, cfg=cfg)
    vec = torch.stack([f[n] for n in feature_names(cfg)]).unsqueeze(0)
    p = SurvivalHead(len(feature_names(cfg)))(vec)
    assert p.shape == (1,)
    assert 0.0 <= float(p) <= 1.0
