"""Stage-0 particle export + Rockstar smoke (CPU)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cosmo_sr.eval.particles import field_to_particles, particle_mass_msun_h, write_gadget2_snapshot
from cosmo_sr.eval.rockstar import default_rockstar_binary, load_rockstar_ascii, run_rockstar_on_particles
from cosmo_sr.eval.halo_metrics import mass_function, nsub_vs_mhost
from cosmo_sr.eval.halo_match import match_hosts, classify_subhalos
from cosmo_sr.eval.rockstar import HaloCatalog


def test_particle_mass_formula():
    m = particle_mass_msun_h(0.2814, 100.0, 512 ** 3)
    assert 5.5e8 < m < 6.2e8


def test_field_to_particles_ids_and_shape():
    ng = 8
    rng = np.random.default_rng(0)
    field = rng.normal(size=(6, ng, ng, ng)).astype(np.float32) * 0.01
    p = field_to_particles(field, boxsize_kpc_h=100000.0, redshift=0.0)
    assert p.pos_mpc_h.shape == (ng ** 3, 3)
    assert p.ids[0] == 0 and p.ids[-1] == ng ** 3 - 1
    assert np.all((p.pos_mpc_h >= 0) & (p.pos_mpc_h < p.boxsize_mpc_h))


def test_gadget2_roundtrip_header(tmp_path):
    ng = 4
    rng = np.random.default_rng(1)
    field = rng.normal(size=(6, ng, ng, ng)).astype(np.float32) * 0.01
    p = field_to_particles(field, boxsize_kpc_h=100000.0)
    path = tmp_path / "t.gadget2"
    write_gadget2_snapshot(str(path), p)
    assert path.stat().st_size > 256


@pytest.mark.skipif(
    not default_rockstar_binary().is_file(),
    reason="Rockstar binary not built",
)
def test_rockstar_smoke_finds_a_halo(tmp_path):
    """Plant a compact overdensity so Rockstar returns ≥1 host."""
    from cosmo_sr.eval.particles import ParticleBox

    n = 2000
    rng = np.random.default_rng(2)
    # Main clump near centre + a few outliers
    pos = rng.normal(loc=50.0, scale=0.15, size=(n, 3)).astype(np.float32)
    pos = np.clip(pos, 0.01, 99.99)
    vel = rng.normal(scale=50.0, size=(n, 3)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    particles = ParticleBox(
        pos_mpc_h=pos, vel_kms=vel, ids=ids, boxsize_mpc_h=100.0,
        redshift=0.0, particle_mass_msun_h=1e10,
    )
    cat = run_rockstar_on_particles(
        particles, tmp_path / "rs", tag="smoke", overwrite=True,
    )
    assert cat.n >= 1
    assert cat.hosts().n >= 1


def test_match_and_classify_synthetic():
    # Two hosts, HR has 2 subs, SR recovers one
    hr = HaloCatalog(
        ids=np.array([0, 1, 10, 11]),
        parent_ids=np.array([-1, -1, 0, 0]),
        mvir=np.array([1e14, 5e13, 1e12, 8e11]),
        rvir=np.array([1000.0, 800.0, 100.0, 90.0]),  # kpc/h
        vmax=np.array([200.0, 150.0, 40.0, 35.0]),
        pos=np.array([[10, 10, 10], [50, 50, 50], [10.05, 10, 10], [10.2, 10, 10]], float),
        vel=np.zeros((4, 3)),
        num_p=np.array([1000, 500, 40, 30]),
    )
    sr = HaloCatalog(
        ids=np.array([0, 1, 10]),
        parent_ids=np.array([-1, -1, 0]),
        mvir=np.array([1.1e14, 4.5e13, 9e11]),
        rvir=np.array([1000.0, 800.0, 95.0]),
        vmax=np.array([200.0, 150.0, 38.0]),
        pos=np.array([[10.02, 10, 10], [50.1, 50, 50], [10.06, 10, 10]], float),
        vel=np.zeros((3, 3)),
        num_p=np.array([1000, 500, 35]),
    )
    hm = match_hosts(hr, sr, boxsize_mpc_h=100.0)
    assert int((hm.sr_ids >= 0).sum()) == 2
    classes = classify_subhalos(hr, sr, hm, boxsize_mpc_h=100.0)
    labels = {c["hr_id"]: c["class"] for c in classes}
    assert labels[10] in ("recovered", "recovered_biased", "spatially_shifted")
    assert labels[11] in ("missing", "merged_into_host")
    m_c, m_dn = mass_function(hr.hosts().mvir, 100.0)
    assert m_c.size == m_dn.size
    _, mean, _ = nsub_vs_mhost(hr)
    assert mean.shape[0] > 0
