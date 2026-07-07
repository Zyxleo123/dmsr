import copy

import torch

from cosmo_sr.train import train_supervised, train_ambient
from cosmo_sr.models.wrappers import build_generator
from cosmo_sr.train.common import load_checkpoint


def _sup_cfg(run_dir, steps=20, scale=2, lr=1e-3):
    return {
        "data": {"crop_lr": 8, "scale_factor": scale},
        "model": {"name": "SimpleSRGenerator", "width": 8, "depth": 1},
        "loss": {},
        "train": {"batch_size": 1, "lr": lr, "steps": steps, "seed": 0,
                  "log_every": 5, "save_every": 0},
        "output": {"run_dir": str(run_dir)},
    }


def _amb_cfg(run_dir, steps=20, scale=2, lr=1e-3, l_amb=1.0, l_pair=1.0):
    return {
        "data": {"crop_lr": 8, "scale_factor": scale},
        "model": {"name": "SimpleSRGenerator", "width": 8, "depth": 1},
        "loss": {"lambda_ambient": l_amb, "lambda_pair": l_pair, "lambda_tv": 0.0},
        "degrader": {"mode": "average"},
        "train": {"batch_size_unpaired": 1, "batch_size_paired": 1, "lr": lr,
                  "steps": steps, "seed": 0, "log_every": 5, "save_every": 0},
        "output": {"run_dir": str(run_dir)},
    }


# ---------------------------------------------------------------- supervised

def test_supervised_smoke_completes(tmp_path):
    res = train_supervised.train(_sup_cfg(tmp_path / "sup", steps=20), smoke=True)
    assert res["steps"] == 20
    assert (tmp_path / "sup" / "ckpt_last.pt").exists()
    assert (tmp_path / "sup" / "config.yaml").exists()
    assert (tmp_path / "sup" / "env.json").exists()


def test_supervised_loss_decreases(tmp_path):
    res = train_supervised.train(_sup_cfg(tmp_path / "sup2", steps=150), smoke=False)
    assert res["last_loss"] < 0.9 * res["first_loss"]


def test_supervised_logs_have_columns(tmp_path):
    train_supervised.train(_sup_cfg(tmp_path / "sup3", steps=20), smoke=True)
    header = (tmp_path / "sup3" / "metrics.csv").read_text().splitlines()[0]
    for col in ("train_loss", "val_loss", "lr"):
        assert col in header


def test_supervised_checkpoint_reload_identical(tmp_path):
    run_dir = tmp_path / "sup4"
    train_supervised.train(_sup_cfg(run_dir, steps=20), smoke=True)
    model = build_generator("SimpleSRGenerator", scale_factor=2, width=8, depth=1)
    load_checkpoint(run_dir / "ckpt_last.pt", model, map_location="cpu")
    model.eval()
    x = torch.randn(1, 6, 8, 8, 8)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)


# ---------------------------------------------------------------- ambient

def test_ambient_unpaired_only_smoke(tmp_path):
    res = train_ambient.train(
        _amb_cfg(tmp_path / "amb1", steps=20, l_amb=1.0, l_pair=0.0), smoke=True
    )
    assert res["steps"] == 20
    assert "loss_ambient" in res["last"] and "loss_pair" not in res["last"]


def test_ambient_mixed_smoke(tmp_path):
    res = train_ambient.train(
        _amb_cfg(tmp_path / "amb2", steps=20, l_amb=1.0, l_pair=1.0), smoke=True
    )
    assert "loss_ambient" in res["last"] and "loss_pair" in res["last"]


def test_ambient_paired_only_smoke(tmp_path):
    res = train_ambient.train(
        _amb_cfg(tmp_path / "amb3", steps=20, l_amb=0.0, l_pair=1.0), smoke=True
    )
    assert "loss_pair" in res["last"] and "loss_ambient" not in res["last"]


def test_ambient_reduces_ambient_loss(tmp_path):
    res = train_ambient.train(
        _amb_cfg(tmp_path / "amb4", steps=150, l_amb=1.0, l_pair=0.0), smoke=False
    )
    assert res["last"]["loss_ambient"] < 0.9 * res["first"]["loss_ambient"]


def test_ambient_mixed_reduces_both(tmp_path):
    res = train_ambient.train(
        _amb_cfg(tmp_path / "amb5", steps=150, l_amb=1.0, l_pair=1.0), smoke=False
    )
    assert res["last"]["loss_ambient"] < res["first"]["loss_ambient"]
    assert res["last"]["loss_pair"] < res["first"]["loss_pair"]


def test_ambient_checkpoint_reload_identical(tmp_path):
    run_dir = tmp_path / "amb6"
    train_ambient.train(_amb_cfg(run_dir, steps=20), smoke=True)
    model = build_generator("SimpleSRGenerator", scale_factor=2, width=8, depth=1)
    load_checkpoint(run_dir / "ckpt_last.pt", model, map_location="cpu")
    model.eval()
    x = torch.randn(1, 6, 8, 8, 8)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)
