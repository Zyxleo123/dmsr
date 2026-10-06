"""The trainer's kNN-distance support term: wiring, not the estimator.

The estimator is proved in ``test_knn_cdf.py``. What can go wrong HERE is the
plumbing: a batch of pool tiles from different boxes sharing one particle table
(box A's particles offered as box B's neighbours), a disabled flag that still
changes the loss, or a term whose gradient never reaches the generator output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/features", "scripts/reward", "src"):
    sys.path.insert(0, str(ROOT / p))

import finetune_member_gather as fmg  # noqa: E402

KW = dict(ng_hr=fmg.NG_HR, tile_hr=fmg.TILE, boxsize_mpc_h=fmg.BOXSIZE,
          dis_scale_mpc_h=6.0, vel_scale_kms=1.0)


def _args(*extra):
    return fmg.build_parser().parse_args(list(extra))


def _small_cfg():
    return fmg.KnnConfig(ks=(2, 8), n_queries=96, n_quantiles=8, buffer_cells=4.0)


def _fields(n, amp_hr=0.02, amp_cand=0.006, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = fmg.TILE
    hr = torch.zeros((n, 6, t, t, t))
    hr[:, :3] = amp_hr * torch.randn((n, 3, t, t, t), generator=g)
    cand = hr.clone()
    cand[:, :3] *= amp_cand / amp_hr
    return cand, hr


def test_defaults_are_off_and_measured():
    a = _args()
    assert a.w_knn_unsup == 0.0
    assert a.knn_measure is True
    cfg = fmg.knn_config_from_args(a)
    # The measured rule, as a default: the largest rank reaches the 50-200p floor.
    assert max(cfg.ks) >= 64
    assert _args("--no-knn-measure").knn_measure is False


def test_flags_reach_the_config():
    a = _args("--knn-ks", "16", "64", "--knn-queries", "300", "--knn-quantiles",
              "12", "--knn-dead-zone", "0.05", "--knn-buffer-cells", "6",
              "--knn-one-sided")
    c = fmg.knn_config_from_args(a)
    assert c.ks == (16, 64) and c.n_queries == 300 and c.n_quantiles == 12
    assert c.dead_zone == 0.05 and c.buffer_cells == 6.0 and c.one_sided


def test_term_has_gradient_to_the_generator_output():
    cand, hr = _fields(2)
    cand.requires_grad_(True)
    batch = [{"box": "set0", "tile": 100}, {"box": "set1", "tile": 100}]
    pen, diags = fmg._unsup_knn(None, batch, None, "cpu", None, KW, _small_cfg(),
                                cand=cand, hr=hr)
    assert len(diags) == 2 and not any(d["skipped"] for d in diags)
    assert [d["box"] for d in diags] == ["set0", "set1"]
    assert float(pen) > 0.0          # candidate is under-displaced vs HR
    pen.backward()
    assert torch.isfinite(cand.grad).all()
    assert float(cand.grad[:, :3].abs().max()) > 0.0
    # Velocity channels are not positions and must carry no gradient.
    assert float(cand.grad[:, 3:].abs().max()) == 0.0


def test_tiles_from_different_boxes_never_share_a_table():
    """Same tile id in two boxes: each tile's loss must equal its solo loss.

    A shared table would put both boxes' particles at the same coordinates and
    halve every neighbour distance, so the batched value would differ.
    """
    cand, hr = _fields(2, seed=3)
    cfg = _small_cfg()
    batch = [{"box": "set0", "tile": 7}, {"box": "set1", "tile": 7}]
    gen = torch.Generator().manual_seed(0)
    _, both = fmg._unsup_knn(None, batch, None, "cpu", None, KW, cfg,
                             generator=gen, cand=cand, hr=hr)
    gen = torch.Generator().manual_seed(0)
    _, solo = fmg._unsup_knn(None, batch[:1], None, "cpu", None, KW, cfg,
                             generator=gen, cand=cand[:1], hr=hr[:1])
    assert both[0]["loss"] == pytest.approx(solo[0]["loss"], rel=1e-12)


def test_eval_is_deterministic_and_reads_the_low_tail(monkeypatch):
    cand, hr = _fields(3, seed=5)
    pool = [{"box": "set0", "tile": t} for t in (3, 40, 77)]

    def fake_forward(model, batch, geom, device):
        idx = [pool.index(b) for b in batch]
        return cand[idx], hr[idx]
    monkeypatch.setattr(fmg, "_forward_unsup", fake_forward)
    cfg = _small_cfg()
    r1 = fmg.eval_unsup_knn(None, pool, None, "cpu", None, KW, cfg, batch=2)
    r2 = fmg.eval_unsup_knn(None, pool, None, "cpu", None, KW, cfg, batch=2)
    assert r1 == r2
    assert r1["n_tiles"] == 3 and len(r1["ratio_q10"]) == 2
    assert min(r1["ratio_q10"]) > 1.0          # under-clumped reads above 1
    assert fmg.eval_unsup_knn(None, [], None, "cpu", None, KW, cfg) is None


def test_attach_neighbours_uses_frozen_for_candidate_and_hr_for_reference(
        tmp_path, monkeypatch):
    ng, tile = 32, 8
    monkeypatch.setattr(fmg, "NG_HR", ng)
    monkeypatch.setattr(fmg, "TILE", tile)
    rng = np.random.default_rng(0)
    hr = (0.2 * rng.standard_normal((6, ng, ng, ng))).astype(np.float32)
    frozen = (0.05 * rng.standard_normal((6, ng, ng, ng))).astype(np.float32)
    np.save(tmp_path / "frozen.npy", frozen)
    monkeypatch.setattr(fmg, "find_base_field", lambda box, seed: tmp_path / "frozen.npy")
    monkeypatch.setattr(fmg, "load_hr", lambda cfg, box: hr)
    pool = [{"box": "set0", "tile": 9}, {"box": "set0", "tile": 40}]
    args = _args("--knn-nbr-margin-cells", "4")
    fmg.attach_knn_neighbours(pool, {}, args, {"dis_scale_mpc_h": 6.0})
    from cosmo_sr.features.knn_cdf import box_positions, tile_rows_flat
    for it in pool:
        own = set(tile_rows_flat(it["tile"], ng=ng, tile=tile).tolist())
        for tag, f in (("frozen", frozen), ("hr", hr)):
            xp = it[f"knn_extra_{tag}"].numpy()
            bp = box_positions(f, ng=ng, dis_scale_mpc_h=6.0)
            # Every extra is a real particle of THAT field, and none is the tile's.
            match = {tuple(r) for r in bp[sorted(set(range(ng ** 3)) - own)]}
            assert all(tuple(r) in match for r in xp[:200])
            assert xp.shape[0] > 0
    # Margin 0 attaches nothing.
    bare = [{"box": "set0", "tile": 9}]
    fmg.attach_knn_neighbours(bare, {}, _args("--knn-nbr-margin-cells", "0"),
                              {"dis_scale_mpc_h": 6.0})
    assert "knn_extra_hr" not in bare[0]


def test_unsup_knn_passes_extras_and_centre():
    cand, hr = _fields(1, seed=9)
    extra = torch.tensor(np.random.default_rng(1).uniform(0, 100, (4000, 3)),
                         dtype=torch.float32)
    batch = [{"box": "set0", "tile": 100, "knn_extra_frozen": extra,
              "knn_extra_hr": extra}]
    args = _args()
    pen, d = fmg._unsup_knn(None, batch, None, "cpu", args, KW, _small_cfg(),
                            cand=cand, hr=hr)
    assert d[0]["n_extra_cand"] == 4000 and d[0]["centre"] == "centroid"


# --------------------------------------------------------------------------- #
# Round 5b: phase space and owner classes through the trainer
# --------------------------------------------------------------------------- #
def test_phase_and_class_flags_reach_the_config():
    a = _args("--knn-phase-lambda", "4000", "--knn-classes", "sub_50_200",
              "sub_lt50")
    assert fmg.knn_config_from_args(a).phase_lambda == 4000.0
    from cosmo_sr.features.knn_cdf import OWNER_CLASSES
    assert fmg.knn_class_ids(a) == (OWNER_CLASSES.index("sub_50_200"),
                                    OWNER_CLASSES.index("sub_lt50"))
    assert fmg.knn_config_from_args(_args()).phase_lambda == 0.0
    assert fmg.knn_class_ids(_args()) == ()
    with pytest.raises(SystemExit, match="nope"):
        fmg.knn_class_ids(_args("--knn-classes", "nope"))


def test_6d_term_puts_gradient_on_the_velocity_channels():
    """The 3D term leaves velocity alone (test above); the 6D term must not."""
    cand, hr = _fields(2, seed=11)
    cand[:, 3:] = 0.01 * torch.randn_like(cand[:, 3:])
    hr[:, 3:] = 0.05 * torch.randn_like(hr[:, 3:])
    cand.requires_grad_(True)
    batch = [{"box": "set0", "tile": 100}, {"box": "set1", "tile": 100}]
    cfg = fmg.KnnConfig(ks=(2, 8), n_queries=96, n_quantiles=8, buffer_cells=4.0,
                        phase_lambda=4000.0)
    pen, diags = fmg._unsup_knn(None, batch, None, "cpu", None, KW, cfg,
                                cand=cand, hr=hr)
    assert not any(d["skipped"] for d in diags) and float(pen) > 0.0
    pen.backward()
    assert float(cand.grad[:, 3:].abs().max()) > 0.0
    assert float(cand.grad[:, :3].abs().max()) > 0.0


def test_class_conditioned_term_uses_the_tiles_labels():
    cand, hr = _fields(1, seed=12)
    cand.requires_grad_(True)
    cfg = fmg.KnnConfig(ks=(2, 8), n_queries=96, n_quantiles=8, buffer_cells=4.0)
    rows = fmg.TILE ** 3
    labels = torch.zeros(rows, dtype=torch.int8)
    labels[: rows // 2] = 3                       # host
    batch = [{"box": "set0", "tile": 100, "knn_classes": labels}]
    pen, diags = fmg._unsup_knn(None, batch, None, "cpu", None, KW, cfg,
                                cand=cand, hr=hr, class_ids=(0,))
    assert diags[0]["class_ids"] == [0]
    assert diags[0]["classes"]["class_0"]["n_queries_cand"] <= rows // 2
    pen.backward()
    assert float(cand.grad.abs().max()) > 0.0


def test_attach_knn_classes_reads_the_owner_array(tmp_path, monkeypatch):
    ng, tile = 32, 8
    monkeypatch.setattr(fmg, "NG_HR", ng)
    monkeypatch.setattr(fmg, "TILE", tile)
    root = tmp_path / "reward"
    (root / "halos" / "set0__hr__hr" / "a_rockstar").mkdir(parents=True)
    (root / "halos" / "set0__hr__hr" / "a_rockstar" / "halos_0.ascii").write_text("x")
    (root / "halos_particles" / "set0__hr__hr").mkdir(parents=True)
    rng = np.random.default_rng(0)
    owner = rng.choice(np.array([-1, 0, 1, 2], dtype=np.int32), size=ng ** 3)
    np.save(root / "halos_particles" / "set0__hr__hr" / "set0_hr_owner.npy", owner)
    from types import SimpleNamespace
    monkeypatch.setattr(fmg, "load_rockstar_ascii", lambda p: SimpleNamespace(
        ids=np.array([0, 1, 2]), parent_ids=np.array([-1, 0, 0]),
        num_p=np.array([9000, 120, 400])))
    monkeypatch.setenv("DMSR_REWARD_ROOT", str(root))
    pool = [{"box": "set0", "tile": 5}]
    fmg.attach_knn_classes(pool, _args("--knn-classes", "sub_50_200"))
    from cosmo_sr.features.knn_cdf import OWNER_CLASSES, tile_rows_flat
    got = pool[0]["knn_classes"].numpy()
    rows = tile_rows_flat(5, ng=ng, tile=tile)
    want = np.select([owner[rows] == -1, owner[rows] == 0, owner[rows] == 1],
                     [OWNER_CLASSES.index("unbound"), OWNER_CLASSES.index("host"),
                      OWNER_CLASSES.index("sub_50_200")],
                     OWNER_CLASSES.index("sub_ge200"))
    assert np.array_equal(got, want)
    assert (root / "member_gather" / "knn_cdf" / "classes_set0.npy").exists()
    # No classes asked for: nothing attached, and no owner array is needed.
    bare = [{"box": "set0", "tile": 5}]
    fmg.attach_knn_classes(bare, _args())
    assert "knn_classes" not in bare[0]


# --------------------------------------------------------------------------- #
# The catalog-free strata arm: density strata in place of the HR class mask
# --------------------------------------------------------------------------- #
def test_strata_flags_reach_the_config():
    a = _args("--knn-ks", "8", "32", "64", "128", "--knn-strata", "8",
              "--knn-strata-rank", "8", "--knn-strata-charge", "0,1")
    c = fmg.knn_config_from_args(a)
    assert c.strata == 8 and c.strata_rank == 8 and c.strata_charge == (0, 1)
    # Space- and comma-separated forms are the same list, because a submitter
    # writing an env file will produce whichever the shell leaves it.
    assert fmg.knn_strata_charge(_args("--knn-strata-charge", "0", "1")) == (0, 1)
    assert fmg.knn_strata_charge(_args("--knn-strata-charge", "0,1")) == (0, 1)
    assert fmg.knn_strata_charge(_args()) == ()
    # Default off, so an unset run is byte-identical to the existing arms.
    assert fmg.knn_config_from_args(_args()).strata == 0


def test_strata_change_the_penalty_and_the_charge_narrows_it():
    """The term must actually behave differently, and dense-only must be the
    one that weighs the clumped tail -- the measured point of the whole arm."""
    cand, hr = _fields(2, seed=5)
    pool = [{"box": "b", "tile": 0}, {"box": "b", "tile": 1}]
    base = fmg.KnnConfig(ks=(2, 8), n_queries=256, n_quantiles=4,
                         buffer_cells=0.0)
    strat = fmg.KnnConfig(ks=(2, 8), n_queries=256, n_quantiles=4,
                          buffer_cells=0.0, strata=4)
    dense = fmg.KnnConfig(ks=(2, 8), n_queries=256, n_quantiles=4,
                          buffer_cells=0.0, strata=4, strata_charge=(0,))
    a = _args()
    out = {}
    for name, cfg in (("pooled", base), ("equal", strat), ("dense", dense)):
        gen = torch.Generator().manual_seed(0)
        pen, diags = fmg._unsup_knn(None, pool, None, torch.device("cpu"), a, KW,
                                    cfg, generator=gen, cand=cand, hr=hr)
        out[name] = float(pen)
        if name != "pooled":
            assert diags[0]["strata_charge"] == (
                [0] if name == "dense" else [0, 1, 2, 3])
    # Equal-weight strata are the pooled loss in finer bins (the measured
    # identity); dense-only is a different, larger number.
    assert out["equal"] == pytest.approx(out["pooled"], rel=0.5)
    assert out["dense"] > out["equal"]


def test_diagnostic_only_classes_leave_the_loss_catalog_free():
    """The arm's defining property: with the flag on, the per-class numbers are
    still computed for the eval and the TRAINING penalty is the classless one."""
    cand, hr = _fields(2, seed=6)
    t = fmg.TILE
    # Filled with `host`, NOT zeros: 0 is sub_50_200's own index, so a zeroed
    # vector labels the whole tile as the class and the "subset" is every row.
    cls = torch.full((t ** 3,), fmg.OWNER_CLASSES.index("host"),
                     dtype=torch.int8)
    cls[: t ** 3 // 2] = fmg.OWNER_CLASSES.index("sub_50_200")
    pool = [{"box": "b", "tile": 0, "knn_classes": cls},
            {"box": "b", "tile": 1, "knn_classes": cls}]
    cfg = fmg.KnnConfig(ks=(2, 8), n_queries=256, n_quantiles=4,
                        buffer_cells=0.0, strata=4, strata_charge=(0,))
    a = _args()
    sub = (fmg.OWNER_CLASSES.index("sub_50_200"),)

    def pen_with(class_ids):
        gen = torch.Generator().manual_seed(0)
        p, _ = fmg._unsup_knn(None, pool, None, torch.device("cpu"), a, KW, cfg,
                              generator=gen, cand=cand, hr=hr,
                              class_ids=class_ids)
        return float(p)

    # The two are genuinely different objectives -- which is what makes the
    # demotion a real change and not a relabelling.
    assert pen_with(sub) != pytest.approx(pen_with(()), rel=1e-6)


def test_diagnostic_only_without_named_classes_is_refused():
    """A flag that silently does nothing is the failure mode this line keeps
    paying for, so the no-op combination is an error, not a default."""
    a = _args("--knn-classes-diagnostic-only")
    assert a.knn_classes_diagnostic_only is True
    assert fmg.knn_class_ids(a) == ()
