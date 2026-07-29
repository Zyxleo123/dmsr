"""Stages 2 and 4: test-time features, pairwise verifiers, noise refinement."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmo_sr.tts.features import (
    FEATURE_KEYS,
    HRReference,
    candidate_features,
    equivariance_features,
    feature_matrix,
    joint_density_velocity,
    noise_diagnostics,
    rotate_field,
)
from cosmo_sr.tts.metrics import DensityGeometry, cic_density_slabs
from cosmo_sr.tts.refine import (
    NoiseRegularizer,
    cem_refine_tile,
    default_schedule,
    noise_statistics,
    refine_tile_noise,
)
from cosmo_sr.tts.srs_noise import ControlledG
from cosmo_sr.tts.verifier import (
    FeatureRanker,
    PatchVerifier,
    Standardizer,
    make_pairs,
    rank_metrics,
    train_feature_ranker,
)

SMALL = dict(chan_base=16, chan_min=8, chan_max=16)
GEO = DensityGeometry(boxsize=100000.0, ng=16, dis_norm=6000.0)


def _generator(seed: int = 0, noise_scale: float = 0.3) -> ControlledG:
    torch.manual_seed(seed)
    g = ControlledG(6, 6, 8, **SMALL).eval()
    with torch.no_grad():
        for name, p in g.named_parameters():
            if name.endswith(".std"):
                p.copy_(torch.randn_like(p) * noise_scale)
    return g


# --------------------------------------------------------------------------- #
# Cubic symmetry helpers
# --------------------------------------------------------------------------- #
def test_rotation_is_order_four_and_invertible():
    x = torch.randn(1, 6, 4, 4, 4)
    y = x
    for _ in range(4):
        y = rotate_field(y, (-3, -2), 1)
    assert torch.allclose(x, y, atol=1e-6)
    assert torch.allclose(rotate_field(rotate_field(x, (-3, -1), 1), (-3, -1), -1), x, atol=1e-6)


def test_rotation_rotates_vector_components_not_just_the_lattice():
    """A field pointing along +x must point along +y after a rotation in (x, y)."""
    x = torch.zeros(1, 6, 2, 2, 2)
    x[:, 0] = 1.0                       # displacement along axis 0
    y = rotate_field(x, (-3, -2), 1)
    assert float(y[:, 1].mean()) == pytest.approx(1.0)
    assert float(y[:, 0].abs().max()) == pytest.approx(0.0)


def test_rotating_only_the_lattice_would_be_wrong():
    x = torch.zeros(1, 6, 2, 2, 2)
    x[:, 0] = 1.0
    lattice_only = torch.rot90(x, 1, dims=[-3, -2])
    assert not torch.allclose(lattice_only, rotate_field(x, (-3, -2), 1))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_hr_reference_distance_is_zero_for_its_own_reference():
    rho = cic_density_slabs(torch.randn(1, 3, 8, 8, 8) * 0.02, GEO.cellsize, GEO.dis_norm)
    ref = HRReference.fit([rho], n_bins=6)
    d = ref.distance(rho, n_bins=6)
    assert d["plaus_pk_logdist"] == pytest.approx(0.0, abs=1e-6)
    assert d["plaus_pdf_l1"] == pytest.approx(0.0, abs=1e-6)


def test_hr_reference_round_trips_through_disk(tmp_path):
    rho = cic_density_slabs(torch.randn(1, 3, 8, 8, 8) * 0.02, GEO.cellsize, GEO.dis_norm)
    ref = HRReference.fit([rho], n_bins=6)
    ref.save(tmp_path / "ref.npz")
    back = HRReference.load(tmp_path / "ref.npz")
    assert np.allclose(ref.pk, back.pk) and np.allclose(ref.pdf, back.pdf)


def test_joint_density_velocity_sees_velocity_density_coupling():
    """Fast material placed in dense cells must raise the coupling statistics."""
    n = 8
    rho = torch.zeros(1, 1, n, n, n)
    rho[0, 0, :4] = 2.0
    rho[0, 0, 4:] = -0.5
    coupled = torch.zeros(1, 6, n, n, n)
    coupled[0, 3, :4] = 3.0
    uncoupled = torch.zeros(1, 6, n, n, n)
    uncoupled[0, 3, :, :4] = 3.0
    c = joint_density_velocity(coupled, rho)
    u = joint_density_velocity(uncoupled, rho)
    assert c["jdv_speed_in_dense"] > u["jdv_speed_in_dense"]
    assert c["jdv_speed_rho_corr"] > u["jdv_speed_rho_corr"]


def test_candidate_features_cover_the_declared_feature_keys():
    torch.manual_seed(0)
    sr = torch.randn(1, 6, 16, 16, 16) * 0.05
    lr = torch.nn.functional.avg_pool3d(sr, 8)
    rho = cic_density_slabs(sr[:, 0:3], GEO.cellsize, GEO.dis_norm)
    ref = HRReference.fit([rho], n_bins=6)
    f = candidate_features(sr, lr, factor=8, geometry=GEO, reference=ref, tile_size=8,
                           n_bins=6, extra={"equiv_rotation_rel": 0.1, "equiv_flip_rel": 0.2})
    for key in FEATURE_KEYS:
        assert key in f and np.isfinite(f[key]), key


def test_feature_matrix_is_finite_even_with_missing_entries():
    x = feature_matrix([{"lr_recon_rel_disp": 1.0}, {}], FEATURE_KEYS)
    assert x.shape == (2, len(FEATURE_KEYS)) and np.isfinite(x).all()


def test_equivariance_features_transform_noise_with_the_input():
    """Without rotating the noise the residual would just be stochastic spread."""
    g = _generator()
    lr = np.random.default_rng(0).normal(size=(6, 16, 16, 16)).astype(np.float32)
    f = equivariance_features(g, lr, seed=0, nsplit=2, pad=3, scale_factor=8,
                              device=torch.device("cpu"), n_probes=1)
    assert 0.0 <= f["equiv_rotation_rel"] < 10.0
    assert f["equiv_n_probes"] == 1.0
    # a generator with zero noise scale is deterministic; its rotation residual
    # then measures only the network's own non-equivariance, and must be finite
    g0 = _generator(noise_scale=0.0)
    f0 = equivariance_features(g0, lr, seed=0, nsplit=2, pad=3, scale_factor=8,
                               device=torch.device("cpu"), n_probes=1)
    assert np.isfinite(f0["equiv_rotation_rel"])


def test_noise_diagnostics_flag_out_of_distribution_noise():
    ok = {"z0": torch.randn(1, 1, 8, 8, 8)}
    bad = {"z0": torch.randn(1, 1, 8, 8, 8) * 3.0 + 2.0}
    assert noise_diagnostics(ok)["noise_max_absmean"] < 0.2
    assert noise_diagnostics(bad)["noise_max_absmean"] > 1.0
    assert noise_diagnostics(bad)["noise_max_sigma_dev"] > 1.0


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
def test_pairs_are_within_group_and_ordered_best_first():
    groups = [[0, 1, 2], [3, 4]]
    targets = [0.1, 0.5, 0.9, 2.0, 1.0]
    pairs = make_pairs(groups, targets, max_pairs_per_group=10, rng=np.random.default_rng(0))
    for a, b in pairs:
        assert targets[a] < targets[b]
        assert ({a, b} <= {0, 1, 2}) or ({a, b} <= {3, 4})


def test_near_ties_can_be_dropped():
    groups = [[0, 1, 2]]
    targets = [1.0, 1.0001, 5.0]
    assert len(make_pairs(groups, targets, min_margin=0.0)) == 3
    assert len(make_pairs(groups, targets, min_margin=0.01)) == 2


def test_ranker_learns_a_monotone_synthetic_target():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(240, 4))
    target = x @ np.array([2.0, -1.0, 0.5, 0.0]) + 0.05 * rng.normal(size=240)
    groups = [list(range(i, i + 8)) for i in range(0, 160, 8)]
    groups_test = [list(range(i, i + 8)) for i in range(160, 240, 8)]
    model, hist = train_feature_ranker(x[:160], groups, target[:160], epochs=250, seed=0)
    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
    m = rank_metrics(pred, target, groups_test)
    assert m["pairwise_accuracy"] > 0.8, m
    assert m["spearman"] > 0.7, m


def test_rank_metrics_are_exact_for_a_perfect_and_a_reversed_predictor():
    target = np.array([0.1, 0.4, 0.2, 0.9])
    groups = [[0, 1, 2, 3]]
    good = rank_metrics(target, target, groups)
    assert good["spearman"] == pytest.approx(1.0)
    assert good["pairwise_accuracy"] == pytest.approx(1.0)
    assert good["regret"] == pytest.approx(0.0)
    bad = rank_metrics(-target, target, groups)
    assert bad["spearman"] == pytest.approx(-1.0)
    assert bad["regret"] > 0


def test_standardizer_handles_constant_columns(tmp_path):
    x = np.stack([np.arange(10.0), np.ones(10)], axis=1)
    std = Standardizer.fit(x, keys=("a", "b"))
    assert std.std[1] == 1.0
    assert np.isfinite(std(x)).all()
    std.save(tmp_path / "s.json")
    assert Standardizer.load(tmp_path / "s.json").keys == ("a", "b")


def test_patch_verifier_produces_one_score_per_sample():
    v = PatchVerifier(width=8, depth=2)
    lr_up = torch.randn(2, 6, 16, 16, 16)
    sr = torch.randn(2, 6, 16, 16, 16)
    rho = torch.randn(2, 1, 16, 16, 16)
    assert v(lr_up, sr, rho).shape == (2,)


def test_feature_ranker_is_lower_is_better_by_convention():
    m = FeatureRanker(3, hidden=())
    with torch.no_grad():
        m.net[0].weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
        m.net[0].bias.zero_()
    x = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
    with torch.no_grad():
        s = m(x)
    assert float(s[0]) < float(s[1])


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #
def _tile(size: int = 10) -> torch.Tensor:
    torch.manual_seed(4)
    return torch.randn(1, 6, size, size, size)


def test_gradient_refinement_reduces_a_differentiable_objective():
    g, x = _generator(), _tile()
    with torch.no_grad():
        _, z0 = g(x, record=True)
    target = torch.zeros(())

    def objective(y):
        return (y[:, 3:6].pow(2).mean() - target) ** 2

    res = refine_tile_noise(g, x, z0, objective,
                            schedule=[*default_schedule(steps=6, lr=0.1)],
                            regularizer=NoiseRegularizer(lam_mu=0.1, lam_sigma=0.1, lam_l2=0.01))
    assert res.score < res.initial_score, (res.initial_score, res.score)
    assert res.improvement > 0
    assert len(res.trajectory) > 1


def test_refinement_reports_noise_drift_and_rejects_out_of_distribution_noise():
    g, x = _generator(), _tile()
    with torch.no_grad():
        _, z0 = g(x, record=True)

    def runaway(y):          # unbounded: pushes the noise as far as it can go
        return -y.pow(2).mean()

    res = refine_tile_noise(g, x, z0, runaway,
                            schedule=[*default_schedule(steps=25, lr=0.5)],
                            regularizer=NoiseRegularizer(0.0, 0.0, 0.0),
                            max_absmean=0.05, max_sigma_dev=0.05)
    assert res.rejected and res.reject_reason
    assert res.stats["max_dist"] > 0


def test_the_prior_term_keeps_refined_noise_near_the_training_distribution():
    g, x = _generator(), _tile()
    with torch.no_grad():
        _, z0 = g(x, record=True)

    def runaway(y):
        return -y.pow(2).mean()

    loose = refine_tile_noise(g, x, z0, runaway, schedule=[*default_schedule(steps=15, lr=0.3)],
                              regularizer=NoiseRegularizer(0.0, 0.0, 0.0))
    tight = refine_tile_noise(g, x, z0, runaway, schedule=[*default_schedule(steps=15, lr=0.3)],
                              regularizer=NoiseRegularizer(10.0, 10.0, 10.0))
    assert tight.stats["max_dist"] < loose.stats["max_dist"]


def test_coarse_to_fine_schedule_unlocks_stages_in_order():
    sched = default_schedule(steps=5)
    assert [p.stages for p in sched] == [("coarse",), ("coarse", "middle"),
                                         ("coarse", "middle", "fine")]
    assert sched[0].lr > sched[1].lr > sched[2].lr


def test_only_the_unlocked_stages_move():
    g, x = _generator(), _tile()
    with torch.no_grad():
        _, z0 = g(x, record=True)
    from cosmo_sr.tts.refine import Phase

    res = refine_tile_noise(g, x, z0, lambda y: y.pow(2).mean(),
                            schedule=[Phase(("coarse",), 5, 0.1)],
                            regularizer=NoiseRegularizer(0.0, 0.0, 0.0))
    stats = noise_statistics(res.noise, z0)
    assert stats["z0_dist"] > 0 and stats["z1_dist"] > 0
    for site in ("z2", "z3", "z4", "z5"):
        assert stats[f"{site}_dist"] == pytest.approx(0.0, abs=1e-9), site


def test_cem_is_a_gradient_free_control_that_also_improves():
    g, x = _generator(), _tile()
    with torch.no_grad():
        _, z0 = g(x, record=True)
    res = cem_refine_tile(g, x, z0, lambda y: float(y[:, 3:6].pow(2).mean()),
                          stages=("fine",), iterations=3, population=6, sigma=0.4,
                          regularizer=NoiseRegularizer(0.0, 0.0, 0.0), seed=0)
    assert res.score <= res.initial_score
