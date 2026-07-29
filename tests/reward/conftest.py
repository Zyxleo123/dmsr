"""Shared fixtures for the reward-guided residual diffusion tests.

Everything here is CPU-only and synthetic. No test touches a GPU, the real
512^3 boxes, or Rockstar; the pieces that need those are covered by the audit
scripts and the GPU smoke jobs instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from cosmo_sr.eval.rockstar import HaloCatalog
from cosmo_sr.reward.catalog import CatalogBins, ChunkSummary


def synthetic_catalog(hosts, subs_per_host, *, boxsize=100.0, seed=0,
                      host_num_p=200, sub_num_p=50, sub_mass=5e10, rvir_kpc=200.0):
    """A catalog with known answers: ``hosts`` is a list of ``(pos, mvir)``.

    Ids are assigned host-first so a parser bug that confuses the internal index
    with the printed id would show up as a broken parent link, not as silence.
    """
    rng = np.random.default_rng(int(seed))
    ids, parent, mvir, rvir, vmax, pos, vel, num_p = [], [], [], [], [], [], [], []
    next_id = 0
    host_ids = []
    for hpos, hm in hosts:
        host_ids.append(next_id)
        ids.append(next_id)
        parent.append(-1)
        mvir.append(float(hm))
        rvir.append(float(rvir_kpc))
        vmax.append(200.0)
        pos.append(np.asarray(hpos, dtype=np.float64))
        vel.append(np.zeros(3))
        num_p.append(int(host_num_p))
        next_id += 1
    for hid, (hpos, hm), k in zip(host_ids, hosts, subs_per_host):
        for _ in range(int(k)):
            off = rng.normal(0.0, 0.05, size=3)
            ids.append(next_id)
            parent.append(int(hid))
            mvir.append(float(sub_mass))
            rvir.append(float(rvir_kpc) * 0.2)
            vmax.append(80.0)
            pos.append((np.asarray(hpos, dtype=np.float64) + off) % boxsize)
            vel.append(rng.normal(0.0, 100.0, size=3))
            num_p.append(int(sub_num_p))
            next_id += 1
    return HaloCatalog(
        ids=np.asarray(ids, dtype=np.int64),
        parent_ids=np.asarray(parent, dtype=np.int64),
        mvir=np.asarray(mvir, dtype=np.float64),
        rvir=np.asarray(rvir, dtype=np.float64),
        vmax=np.asarray(vmax, dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64).reshape(-1, 3),
        vel=np.asarray(vel, dtype=np.float64).reshape(-1, 3),
        num_p=np.asarray(num_p, dtype=np.int64),
    )


@pytest.fixture(autouse=True)
def reward_root(tmp_path, monkeypatch):
    """Redirect every artifact path into tmp_path so nothing lands on scratch."""
    monkeypatch.setenv("DMSR_REWARD_ROOT", str(tmp_path / "dmsr_reward"))
    return tmp_path / "dmsr_reward"


@pytest.fixture
def bins() -> CatalogBins:
    return CatalogBins(
        sub_mass_edges=tuple(np.logspace(10.0, 13.0, 4).tolist()),
        host_mass_edges=tuple(np.logspace(12.0, 14.0, 3).tolist()),
        min_sub_particles=20,
        min_host_particles=20,
        abundance_transform="log10_floor",
        occupation_transform="log10_floor",
    )


def make_chunk(box="set0", cid=0, source="hr", n_sub=(10, 5, 2),
               n_host=(4, 1), occ=(12, 5), volume=1000.0) -> ChunkSummary:
    return ChunkSummary(
        box=box, chunk_id=cid, source=source,
        n_sub=np.asarray(n_sub, dtype=np.float64),
        n_host=np.asarray(n_host, dtype=np.float64),
        occ_numerator=np.asarray(occ, dtype=np.float64),
        volume_mpc3=float(volume),
    )


@pytest.fixture
def hr_chunks(bins):
    """A small HR population with box-to-box scatter, for fitting a reward model."""
    rng = np.random.default_rng(0)
    out = []
    for b in range(4):
        # A per-box offset makes the bins correlated, which is the situation the
        # covariance is supposed to describe.
        off = float(rng.normal(0.0, 0.1))
        for c in range(8):
            scale = float(np.exp(off + rng.normal(0.0, 0.05)))
            out.append(make_chunk(
                box=f"set{b}", cid=c,
                n_sub=np.round(np.array([40.0, 12.0, 3.0]) * scale),
                n_host=np.round(np.array([6.0, 2.0]) * scale) + 1,
                occ=np.round(np.array([18.0, 6.0]) * scale),
                volume=900.0,
            ))
    return out
