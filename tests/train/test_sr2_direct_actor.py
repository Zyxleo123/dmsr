"""End-to-end behaviour of the direct fine-tuning step, on CPU with toy widths.

The claims pinned here are the ones that make the line safe to run at all:
step zero is bit-identical to the frozen checkpoint, the gradient really reaches
SR2's weights through CIC and the proxy, only the rung's parameters move, and a
reloaded checkpoint reproduces its predecessor exactly.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

# Fixtures live in a uniquely-named module rather than a conftest.py; see
# tests/train/sr2_fixtures.py for why a second conftest.py breaks the suite.
from sr2_fixtures import (  # noqa: F401
    chan_kwargs, generator, geom, hr_field, lr_field, model_path,
)

from cosmo_sr.reward.catalog import CatalogBins, ChunkSummary, pool
from cosmo_sr.reward.catalog_proxy import CatalogProxy, ProxyConfig, ProxyEnsemble
from cosmo_sr.reward.reward import fit_reward_model
from cosmo_sr.reward.soft_structure import (
    SoftStructureConfig, feature_names, paired_features,
)
from cosmo_sr.reward.torch_reward import (
    TorchRewardModel, TorchSummary, summary_from_ensemble,
)
from cosmo_sr.tts.sampling import super_resolve_srs_seeded
from cosmo_sr.train import sr2_unfreeze as U
from cosmo_sr.train.sr2_finetune_data import (
    SR2TileDataset, collate_tiles, fold_draws, trim_to_tile,
)
from cosmo_sr.train.train_sr2_direct import (
    DirectFinetuneConfig, DirectFinetuneTrainer, EMA, attach_summaries,
    block_average_torch,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def soft_cfg(geom) -> SoftStructureConfig:
    # ng_hr must match the test geometry or the CIC cell size is wrong.
    return SoftStructureConfig(ng_hr=geom.ng_hr, n_power_bands=3)


@pytest.fixture
def reward_and_box():
    bins = CatalogBins(sub_mass_edges=tuple(np.logspace(10.1, 13.1, 7).tolist()),
                       host_mass_edges=tuple(np.logspace(12.0, 14.5, 6).tolist()))
    rng = np.random.default_rng(0)
    chunks = []
    for b in range(6):
        off = float(rng.normal(0.0, 0.15))
        for c in range(8):
            s = float(np.exp(off + rng.normal(0.0, 0.05)))
            chunks.append(ChunkSummary(
                box=f"set{b}", chunk_id=c, source="hr",
                n_sub=np.round(np.array([400., 160., 60., 22., 8., 3.]) * s),
                n_host=np.round(np.array([160., 54., 16., 4., 1.]) * s) + 1,
                occ_numerator=np.round(np.array([200., 90., 40., 14., 5.]) * s),
                volume_mpc3=1562.5))
    model = fit_reward_model(chunks, bins,
                             active_dims=[i for i in range(11) if i != 10])
    return TorchRewardModel.from_numpy(model), pool(chunks)


@pytest.fixture
def proxies(soft_cfg):
    n_feat = 2 * len(feature_names(soft_cfg))
    cfg = ProxyConfig(n_features=n_feat, n_sub_bins=6, n_host_bins=5, hidden=(16,))
    return ProxyEnsemble([CatalogProxy(cfg, seed=s) for s in range(3)])


@pytest.fixture
def frozen_field(geom, generator, lr_field):
    return super_resolve_srs_seeded(
        generator, lr_field, 0, scale_factor=geom.scale_factor,
        nsplit=geom.nsplit, pad=geom.pad, device=torch.device("cpu"))


@pytest.fixture
def batch(geom, generator, lr_field, hr_field, frozen_field, reward_and_box):
    ds = SR2TileDataset(
        boxes=["setT"], lr_fields={"setT": lr_field}, hr_fields={"setT": hr_field},
        geom=geom, frozen_fields={"setT": frozen_field},
        frozen_generator=generator, noise_draws=2)
    b = collate_tiles([ds[0], ds[3]])
    _, box_ens = reward_and_box
    box = summary_from_ensemble(box_ens)
    tile = TorchSummary(n_sub=box.n_sub / 8.0, n_host=box.n_host / 8.0,
                        occ_numerator=box.occ_numerator / 8.0,
                        volume_mpc3=box.volume_mpc3 / 8.0)
    return attach_summaries(
        b, {"setT": box}, {("setT", 0): tile, ("setT", 3): tile})


@pytest.fixture
def trainer(model_path, proxies, reward_and_box, geom, soft_cfg, chan_kwargs):
    reward, _ = reward_and_box
    cfg = DirectFinetuneConfig(rung="proj_noise", amp=False,
                               reward_warmup_steps=0, density_power_bands=3)
    return DirectFinetuneTrainer(model_path, proxies, reward, cfg=cfg,
                                 geom=geom, soft_cfg=soft_cfg,
                                 device=torch.device("cpu"),
                                 chan_kwargs=chan_kwargs)


# --------------------------------------------------------------------------- #
# Step zero
# --------------------------------------------------------------------------- #
def test_step_zero_output_equals_the_frozen_checkpoint(trainer, batch, geom):
    """Before any update, the actor IS the frozen generator."""
    lr, noise, _, _ = fold_draws(batch)
    with torch.no_grad():
        a = trim_to_tile(trainer.actor(lr, noise=noise), geom)
        b = trim_to_tile(trainer.frozen(lr, noise=noise), geom)
    assert torch.equal(a, b)


def test_step_zero_matches_the_cached_full_box_frozen_field(trainer, batch, geom):
    lr, noise, nb, nd = fold_draws(batch)
    with torch.no_grad():
        out = trim_to_tile(trainer.actor(lr, noise=noise), geom)
    # Not bit-exact, and correctly so: this forward runs a batch of 4 while the
    # cached box was produced one tile at a time, and cuDNN/oneDNN pick
    # different reduction orders for different batch sizes. The bit-exact claim
    # is the single-sample one, pinned in test_sr2_finetune_data.py.
    assert torch.allclose(out, batch["frozen"].reshape(nb * nd, *out.shape[1:]),
                          atol=1e-6, rtol=1e-5)


def test_step_zero_reward_delta_uses_a_zero_feature_difference(trainer, batch, geom):
    lr, noise, _, _ = fold_draws(batch)
    with torch.no_grad():
        cand = trim_to_tile(trainer.actor(lr, noise=noise), geom).float()
        base = trim_to_tile(trainer.frozen(lr, noise=noise), geom).float()
        f = paired_features(cand, base, trainer.soft_cfg)
    n = len(feature_names(trainer.soft_cfg))
    # The difference block is exactly zero at step zero, which is the only state
    # in which "the actor has not changed anything" is visible in the features.
    assert torch.allclose(f[:, n:], torch.zeros_like(f[:, n:]), atol=1e-6)


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #
def test_gradient_reaches_sr2_weights_through_cic_and_the_proxy(trainer, batch):
    terms, _ = trainer.loss_terms(batch)
    params = [p for p in trainer.actor.parameters() if p.requires_grad]
    g = torch.autograd.grad(terms["reward"], params, allow_unused=True)
    total = 0.0
    for x in g:
        if x is None:
            continue
        assert torch.isfinite(x).all()
        total += float(x.abs().sum())
    assert total > 0.0, "the reward term produced no gradient into SR2"


def test_every_loss_term_is_finite_and_reported(trainer, batch):
    terms, diag = trainer.loss_terms(batch)
    assert set(terms) == {"reward", "density_power", "low_k", "prox", "diversity"}
    for k, v in terms.items():
        assert torch.isfinite(v).all(), k
    # All three rewards always logged, never just the combined one.
    for k in ("dR_cat", "dR_occ", "dR_abund", "q_mean", "q_std", "q_safe"):
        assert k in diag and np.isfinite(diag[k])


def test_low_k_and_prox_are_zero_at_step_zero(trainer, batch):
    terms, _ = trainer.loss_terms(batch)
    assert float(terms["low_k"]) == pytest.approx(0.0, abs=1e-6)
    assert float(terms["prox"]) == pytest.approx(0.0, abs=1e-12)


def test_per_term_grad_norms_are_measured(trainer, batch):
    terms, _ = trainer.loss_terms(batch)
    total, weights = trainer.weighted_sum(terms)
    norms = trainer._per_term_grad_norms(terms, weights)
    assert set(norms) == {f"gradnorm_{k}" for k in terms}
    for k, v in norms.items():
        assert np.isfinite(v) and v >= 0.0, k


def test_reward_warmup_ramps(model_path, proxies, reward_and_box, geom,
                             soft_cfg, chan_kwargs):
    reward, _ = reward_and_box
    cfg = DirectFinetuneConfig(rung="proj_noise", amp=False, lambda_reward=2.0,
                               reward_warmup_steps=10)
    t = DirectFinetuneTrainer(model_path, proxies, reward, cfg=cfg, geom=geom,
                              soft_cfg=soft_cfg, device=torch.device("cpu"),
                              chan_kwargs=chan_kwargs)
    assert t.reward_weight() == pytest.approx(0.0)
    t.step_index = 5
    assert t.reward_weight() == pytest.approx(1.0)
    t.step_index = 50
    assert t.reward_weight() == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# The step itself
# --------------------------------------------------------------------------- #
def test_one_step_moves_only_the_rung(trainer, batch):
    before = U.snapshot_parameters(trainer.actor)
    metrics = trainer.step(batch)
    deltas = U.assert_only_trainable_changed(trainer.actor, before, trainer.trainable)
    assert np.isfinite(metrics["loss"])
    assert metrics["step"] == 0.0
    assert trainer.step_index == 1
    assert any(deltas[n] > 0 for n in trainer.trainable), "nothing actually trained"


def test_frozen_twin_never_moves(trainer, batch):
    before = {n: p.detach().clone() for n, p in trainer.frozen.named_parameters()}
    trainer.step(batch)
    for n, p in trainer.frozen.named_parameters():
        assert torch.equal(p, before[n]), n
        assert not p.requires_grad


def test_proxy_parameters_never_move(trainer, batch):
    before = {n: p.detach().clone() for n, p in trainer.proxies.named_parameters()}
    trainer.step(batch)
    for n, p in trainer.proxies.named_parameters():
        assert torch.equal(p, before[n]), n


def test_metrics_row_is_flat_and_finite(trainer, batch):
    m = trainer.step(batch)
    assert all(isinstance(v, float) for v in m.values())
    assert np.isfinite(m["grad_norm_total"])
    for key in ("loss_reward", "loss_density_power", "loss_low_k", "loss_prox",
                "loss_diversity", "weight_reward", "div_d_struct"):
        assert key in m


def test_diversity_term_is_inactive_and_flagged_with_one_draw(
        geom, generator, lr_field, hr_field, frozen_field, reward_and_box,
        model_path, proxies, soft_cfg, chan_kwargs):
    reward, box_ens = reward_and_box
    ds = SR2TileDataset(
        boxes=["setT"], lr_fields={"setT": lr_field}, hr_fields={"setT": hr_field},
        geom=geom, frozen_fields={"setT": frozen_field}, noise_draws=1)
    b = collate_tiles([ds[1]])
    box = summary_from_ensemble(box_ens)
    tile = TorchSummary(n_sub=box.n_sub / 8.0, n_host=box.n_host / 8.0,
                        occ_numerator=box.occ_numerator / 8.0,
                        volume_mpc3=box.volume_mpc3 / 8.0)
    b = attach_summaries(b, {"setT": box}, {("setT", 1): tile})
    t = DirectFinetuneTrainer(
        model_path, proxies, reward,
        cfg=DirectFinetuneConfig(rung="proj_noise", amp=False,
                                 density_power_bands=3),
        geom=geom, soft_cfg=soft_cfg, device=torch.device("cpu"),
        chan_kwargs=chan_kwargs)
    terms, diag = t.loss_terms(b)
    assert float(terms["diversity"]) == 0.0
    assert diag.get("diversity_inactive") == 1.0
    assert np.isnan(diag["div_d_struct"])


# --------------------------------------------------------------------------- #
# Checkpoints and reproducibility
# --------------------------------------------------------------------------- #
def test_checkpoint_round_trip_reproduces_the_next_step(
        tmp_path, model_path, proxies, reward_and_box, geom, soft_cfg, batch,
        chan_kwargs):
    reward, _ = reward_and_box

    def fresh():
        return DirectFinetuneTrainer(
            model_path, proxies, reward,
            cfg=DirectFinetuneConfig(rung="proj_noise", amp=False,
                                     reward_warmup_steps=0,
                                     density_power_bands=3),
            geom=geom, soft_cfg=soft_cfg, device=torch.device("cpu"),
        chan_kwargs=chan_kwargs)

    a = fresh()
    a.step(batch)
    path = a.save(tmp_path / "ck.pt")
    m_a = a.step(batch)

    b = fresh()
    blob = b.load(path)
    assert blob["rung"] == "proj_noise"
    assert b.step_index == 1
    assert sorted(b.trainable) == sorted(a.trainable)
    m_b = b.step(batch)

    assert m_b["loss"] == pytest.approx(m_a["loss"], rel=1e-9, abs=1e-12)
    for n, p in b.actor.named_parameters():
        assert torch.allclose(p, dict(a.actor.named_parameters())[n], atol=0, rtol=0)


def test_two_trainers_with_the_same_seed_agree(
        model_path, proxies, reward_and_box, geom, soft_cfg, batch, chan_kwargs):
    reward, _ = reward_and_box

    def run():
        t = DirectFinetuneTrainer(
            model_path, proxies, reward,
            cfg=DirectFinetuneConfig(rung="fine", amp=False, seed=17,
                                     reward_warmup_steps=0,
                                     density_power_bands=3),
            geom=geom, soft_cfg=soft_cfg, device=torch.device("cpu"),
        chan_kwargs=chan_kwargs)
        return [t.step(batch)["loss"] for _ in range(2)]

    assert run() == pytest.approx(run(), rel=1e-9, abs=1e-12)


def test_ema_checkpoint_loads_as_an_ordinary_sr2_generator(
        tmp_path, trainer, batch, geom, lr_field, chan_kwargs):
    from cosmo_sr.tts.srs_noise import load_controlled_generator

    for _ in range(3):
        trainer.step(batch)
    path = trainer.save_ema_generator(tmp_path / "ema.pt")
    g = load_controlled_generator(path, scale_factor=geom.scale_factor,
                                  device=torch.device("cpu"), **chan_kwargs)
    # The whole point of the upstream layout: existing evaluation paths work.
    out = super_resolve_srs_seeded(
        g, lr_field, 0, scale_factor=geom.scale_factor, nsplit=geom.nsplit,
        pad=geom.pad, device=torch.device("cpu"))
    assert out.shape == (6, geom.ng_hr, geom.ng_hr, geom.ng_hr)
    assert np.isfinite(out).all()


def test_ema_tracks_and_restores(trainer, batch):
    for _ in range(3):
        trainer.step(batch)
    live = {n: p.detach().clone() for n, p in trainer.actor.named_parameters()}
    with trainer.ema.applied(trainer.actor) as m:
        swapped = {n: p.detach().clone() for n, p in m.named_parameters()}
    after = {n: p.detach().clone() for n, p in trainer.actor.named_parameters()}

    assert set(trainer.ema.shadow) == set(trainer.trainable)
    for n, p in trainer.actor.named_parameters():
        assert torch.isfinite(p).all(), f"{n} went non-finite during training"
    # The context manager swaps the EMA weights in and puts the live ones back.
    for n in trainer.trainable:
        assert torch.equal(swapped[n], trainer.ema.shadow[n])
        assert torch.equal(after[n], live[n])
    # EMA lags the live iterate, so at least one trainable tensor must differ.
    assert any(not torch.equal(swapped[n], live[n]) for n in trainer.trainable)


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #
def test_unknown_config_key_is_refused():
    with pytest.raises(KeyError, match="unknown actor config keys"):
        DirectFinetuneConfig.from_dict({"lambda_rewrd": 1.0})


def test_config_group_lr_merges_with_defaults():
    c = DirectFinetuneConfig.from_dict({"rung": "fine", "group_lr": {"fine": 1e-9}})
    assert c.group_lr["fine"] == pytest.approx(1e-9)
    assert c.group_lr["coarse"] == pytest.approx(U.DEFAULT_GROUP_LR["coarse"])


def test_block_average_matches_numpy():
    from cosmo_sr.reward.fields import block_average

    x = np.random.default_rng(0).normal(size=(1, 6, 16, 16, 16)).astype(np.float32)
    a = block_average(x[0], 8)
    b = block_average_torch(torch.from_numpy(x), 8)[0].numpy()
    assert np.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------- #
# Arm dispatch: the trainer computes each arm's own features
# --------------------------------------------------------------------------- #
def _arm_proxies(arm, soft_cfg):
    from cosmo_sr.reward.phase_space import (
        PhaseSpaceConfig, arm_paired_feature_names,
    )
    from cosmo_sr.reward.soft_rockstar import (
        SoftRockstarConfig, SoftRockstarProxy, SoftRockstarProxyConfig,
        token_feature_names,
    )

    pcfg = PhaseSpaceConfig()
    rcfg = SoftRockstarConfig()
    if arm == "c":
        cfg = SoftRockstarProxyConfig(
            n_token_features=len(token_feature_names(rcfg)),
            n_tokens=int(rcfg.tokens_per_axis) ** 3,
            n_sub_bins=6, n_host_bins=5, token_hidden=(8,), embed_dim=4)
        members = [SoftRockstarProxy(cfg, seed=s) for s in range(2)]
        g = torch.Generator().manual_seed(0)
        tok = torch.randn(8, 2, cfg.n_tokens, cfg.n_token_features, generator=g)
        for m in members:
            m.fit_standardizer(tok)
    else:
        n = len(arm_paired_feature_names(arm, soft_cfg, pcfg))
        cfg = ProxyConfig(n_features=n, n_sub_bins=6, n_host_bins=5, hidden=(16,))
        members = [CatalogProxy(cfg, seed=s) for s in range(2)]
    return ProxyEnsemble(members), pcfg, rcfg


@pytest.mark.parametrize("arm", ["b", "c"])
def test_reward_gradient_flows_through_the_arm_extractor(
        arm, model_path, reward_and_box, geom, soft_cfg, batch, chan_kwargs):
    """Arms B and C are real actor paths, not benchmark-only feature sets.

    The trainer must extract THIS arm's features differentiably and the reward
    gradient must reach SR2's weights through them -- for C that is the full
    field -> CIC deposit -> token grid -> DeepSets -> reward chain.
    """
    reward, _ = reward_and_box
    ens, pcfg, rcfg = _arm_proxies(arm, soft_cfg)
    t = DirectFinetuneTrainer(
        model_path, ens, reward,
        cfg=DirectFinetuneConfig(rung="proj_noise", amp=False,
                                 reward_warmup_steps=0, density_power_bands=3),
        geom=geom, soft_cfg=soft_cfg, device=torch.device("cpu"),
        chan_kwargs=chan_kwargs, arm=arm, phase_cfg=pcfg, rockstar_cfg=rcfg)
    terms, diag = t.loss_terms(batch)
    for k, v in terms.items():
        assert torch.isfinite(v).all(), k
    params = [p for p in t.actor.parameters() if p.requires_grad]
    g = torch.autograd.grad(terms["reward"], params, allow_unused=True)
    total = sum(float(x.abs().sum()) for x in g if x is not None)
    assert total > 0.0, f"arm {arm}: no reward gradient reached SR2"


def test_arm_c_extract_is_the_paired_token_grid(
        model_path, reward_and_box, geom, soft_cfg, chan_kwargs, batch):
    from cosmo_sr.train.sr2_finetune_data import fold_draws, trim_to_tile

    reward, _ = reward_and_box
    ens, pcfg, rcfg = _arm_proxies("c", soft_cfg)
    t = DirectFinetuneTrainer(
        model_path, ens, reward,
        cfg=DirectFinetuneConfig(rung="proj_noise", amp=False,
                                 density_power_bands=3),
        geom=geom, soft_cfg=soft_cfg, device=torch.device("cpu"),
        chan_kwargs=chan_kwargs, arm="c", phase_cfg=pcfg, rockstar_cfg=rcfg)
    lr, noise, _, _ = fold_draws(batch)
    with torch.no_grad():
        cand = trim_to_tile(t.actor(lr, noise=noise), geom).float()
        base = trim_to_tile(t.frozen(lr, noise=noise), geom).float()
        tok = t._extract(cand, base)
    T = int(rcfg.tokens_per_axis) ** 3
    assert tok.shape[1:] == (2, T, ens.members[0].cfg.n_token_features)
    # At step zero the actor IS the frozen generator: the difference block of
    # every token is exactly zero, same convention as the flat arms.
    assert torch.allclose(tok[:, 1], torch.zeros_like(tok[:, 1]), atol=1e-6)
