"""Shared CPU-only fixtures for the direct SR2 fine-tuning tests.

Why this is ``sr2_fixtures.py`` and not ``conftest.py``
------------------------------------------------------
There is no ``__init__.py`` anywhere under ``tests/``, so pytest imports every
``conftest.py`` under the top-level module name ``conftest`` and the last one
imported wins in ``sys.modules``. ``tests/reward/test_no_hr_leak.py`` and
``tests/reward/test_replay.py`` do a bare ``from conftest import ...``, which
resolves through ``sys.modules`` -- so adding a second ``conftest.py`` anywhere
that sorts after ``reward`` breaks them with an ImportError that names the wrong
file and points nowhere near the change that caused it. (Measured: it is the
only thing the full suite failed on.)

The fixtures therefore live in a uniquely-named module and are imported by name
into each test file; pytest resolves fixtures from the test module's namespace,
so an imported fixture function works exactly as a conftest one would.

Everything here is a *small* generator with the real architecture: same module
layout, same state-dict keys, same three blocks, same noise sites -- only the
channel widths are cut, so a 14^3 -> 70^3 forward pass runs in milliseconds on a
CPU. That is the point: the geometry, the noise plumbing and the parameter names
are exactly what production uses, and those are what these tests are about.

No test here touches a GPU, the real 512^3 boxes, or Rockstar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cosmo_sr.tts.srs_noise import ControlledG
from cosmo_sr.train.sr2_finetune_data import SR2TileGeometry

#: 2 tiles per axis over a 16-cell LR grid: chunk 8, pad 3, lr_size 14, tile 64.
#: The chunk and pad are the *production* values, so the crop geometry under
#: test is the deployed one; only the number of tiles is reduced.
TEST_GEOM = SR2TileGeometry(ng_lr=16, nsplit=2, pad=3, scale_factor=8)

#: Channel widths of the toy generator. They are a property of the checkpoint,
#: so anything that *loads* the fixture checkpoint has to be told them.
TINY_CHAN = {"chan_base": 16, "chan_min": 4, "chan_max": 16}


def tiny_generator(seed: int = 0) -> ControlledG:
    """A ControlledG with production layout, toy widths and *nonzero* noise scales.

    ``AddNoise.std`` initialises to zeros, so a freshly constructed generator is
    exactly deterministic in ``z`` -- every noise draw gives the same output.
    That is correct for a fresh model and completely wrong as a stand-in for
    ``G_z0.pt``, whose ``std`` is ~1e-3 at ``z0..z4`` and ~5e-2 at ``z5``. A
    fixture left at zero would make every diversity assertion vacuous, so the
    scales are set here to the pretrained order of magnitude.
    """
    torch.manual_seed(int(seed))
    g = ControlledG(6, 6, 8, **TINY_CHAN)
    with torch.no_grad():
        for name, p in g.named_parameters():
            if not name.endswith(".std"):
                continue
            fine_last = name.startswith("blocks.2.") and ".conv.4." in name
            p.fill_(5e-2 if fine_last else 1e-3)
    return g


@pytest.fixture
def chan_kwargs() -> dict:
    """Channel widths of the fixture checkpoint, for anything that loads it."""
    return dict(TINY_CHAN)


@pytest.fixture
def geom() -> SR2TileGeometry:
    return TEST_GEOM


@pytest.fixture
def generator() -> ControlledG:
    g = tiny_generator()
    g.eval()
    return g


@pytest.fixture
def model_path(tmp_path, generator) -> Path:
    """A checkpoint in the upstream ``{'model': state_dict}`` layout."""
    p = tmp_path / "G_tiny.pt"
    torch.save({"epoch": 0, "model": generator.state_dict()}, p)
    return p


@pytest.fixture
def lr_field(geom) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = geom.ng_lr
    return rng.normal(0.0, 0.1, size=(6, n, n, n)).astype(np.float32)


@pytest.fixture
def hr_field(geom) -> np.ndarray:
    rng = np.random.default_rng(11)
    n = geom.ng_hr
    return rng.normal(0.0, 0.1, size=(6, n, n, n)).astype(np.float32)
