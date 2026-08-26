"""Field-only tests for the moment-constrained substructure module (step 5).

Covers the pieces that do not need the 512^3 whole-box artifacts: the tile
geometry, the local-scale normalization round-trip, the flow-matching loss and
Euler sampler, and a few-step CPU smoke of the trainer. The projector Pi itself
is tested in tests/features/.
"""
from __future__ import annotations

import numpy as np
import torch

from cosmo_sr.train import substructure_data as sd
from cosmo_sr.train.train_substructure import _build_model, train


def test_tile_geometry_partitions_and_aligns():
    # Every tile is a 64-HR / 8-LR block, and the raster covers the box once.
    seen_hr = np.zeros((sd.NG_HR,) * 3, dtype=bool)
    for t in range(sd.N_TILES):
        ix, iy, iz = sd.tile_coord(t)
        hx, hy, hz = sd.hr_block(ix, iy, iz)
        lx, ly, lz = sd.lr_block(ix, iy, iz)
        assert hx.stop - hx.start == sd.TILE_HR
        assert lx.stop - lx.start == sd.TILE_HR // sd.UPSAMPLE
        # HR block is exactly the upsample of the LR block.
        assert hx.start == lx.start * sd.UPSAMPLE
        assert not seen_hr[hx, hy, hz].any()
        seen_hr[hx, hy, hz] = True
    assert seen_hr.all()


def test_apply_scale_roundtrip():
    torch.manual_seed(0)
    field = torch.randn(2, 6, 8, 8, 8)
    s_disp = torch.rand(2, 1, 8, 8, 8) + 0.5
    s_vel = torch.rand(2, 1, 8, 8, 8) + 0.5
    norm = sd.apply_scale(field, s_disp, s_vel, undo=False)
    back = sd.apply_scale(norm, s_disp, s_vel, undo=True)
    assert torch.allclose(back, field, atol=1e-5)
    # disp and vel groups really used different scales.
    assert not torch.allclose(norm[:, 0:3], field[:, 0:3])


def test_scale_fields_positive_and_shaped():
    rng = np.random.default_rng(1)
    box = rng.normal(size=(6, 16, 16, 16)).astype(np.float32)
    s_disp, s_vel = sd.scale_fields(box, k=3, eps=1e-3)
    assert s_disp.shape == (16, 16, 16) and s_vel.shape == (16, 16, 16)
    assert np.all(s_disp > 0) and np.all(s_vel > 0)


def test_cfm_loss_finite_and_differentiable():
    model = _build_model({"width": 8, "num_levels": 2, "num_groups": 4}, torch.device("cpu"))
    n = 16
    x_in = torch.randn(2, 6, n, n, n)
    x1 = torch.randn(2, 6, n, n, n)
    ctx = torch.randn(2, sd.HOST_CHANNELS, n, n, n)
    loss = sd.cfm_loss(model, x_in, x1, ctx)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_integrate_tile_shape_and_finite():
    model = _build_model({"width": 8, "num_levels": 2, "num_groups": 4}, torch.device("cpu"))
    model.eval()
    n = 16
    x_in = torch.randn(1, 6, n, n, n)
    ctx = torch.randn(1, sd.HOST_CHANNELS, n, n, n)
    d = sd.integrate_tile(model, x_in, ctx, n_steps=4)
    assert d.shape == (1, 6, n, n, n)
    assert torch.isfinite(d).all()


def test_host_row_of_picks_id_then_argmax():
    from types import SimpleNamespace

    from cosmo_sr.train import substructure_eval as se

    table = SimpleNamespace(
        mvir=np.array([1e12, 5e14, 3e13]),
        row_of=lambda hid: {717: 2}.get(int(hid), -1))
    feat = SimpleNamespace(table=table)
    assert se.host_row_of(feat, 717) == 2          # id wins when present
    assert se.host_row_of(feat, 999) == 1          # unknown id -> most massive
    assert se.host_row_of(feat, None) == 1


def test_region_rockstar_eval_is_failure_tolerant(tmp_path):
    from types import SimpleNamespace

    from cosmo_sr.train import substructure_eval as se

    # A feat missing .grid raises inside the try; the eval must swallow it and
    # return {} rather than kill training.
    out = se.region_rockstar_eval(
        None, None, SimpleNamespace(), None, 0,
        work_dir=tmp_path, cache={}, device=torch.device("cpu"))
    assert out == {}


def test_train_smoke(tmp_path):
    cfg = {
        "model": {"width": 8, "num_levels": 2, "num_groups": 4},
        "train": {"seed": 0, "device": "cpu"},
        "output": {"run_dir": str(tmp_path / "run")},
        "wandb": {"mode": "disabled"},
    }
    result = train(cfg, smoke=True)
    assert np.isfinite(result["last"]["loss"])
    assert (tmp_path / "run" / "ckpt_last.pt").is_file()
