"""End-to-end plumbing for scripts/sr2/particle_identity.py.

A 16^3 stand-in for the 512^3 boxes: two catalogs, two `.particles` tables, two
displacement fields. The point is not the physics but that the script's answers
are the ones the construction puts in -- an identical box must score 1, a
rigidly shifted halo must score 1 with a nonzero bulk shift, and a halo rebuilt
from different Lagrangian ids must score low -- because on the real boxes each
of those looks the same in the logs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cosmo_sr.eval.particle_identity import stream_owner_assignment
from cosmo_sr.eval.rockstar import HaloCatalog

from test_particle_identity import write_particles_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NG = 16
N_PART = NG ** 3
BOX = 100.0
CELL = BOX / NG            # Mpc/h


def _load_script():
    path = PROJECT_ROOT / "scripts" / "sr2" / "particle_identity.py"
    spec = importlib.util.spec_from_file_location("sr2_particle_identity", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pi_script = _load_script()


def lagrangian_ids(coords) -> np.ndarray:
    c = np.asarray(coords, dtype=np.int64).reshape(-1, 3) % NG
    return (c[:, 0] * NG + c[:, 1]) * NG + c[:, 2]


def zero_positions() -> np.ndarray:
    """Positions of a zero-displacement field, indexed by particle id."""
    q = (np.arange(NG) + 0.5) * CELL
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


ASCII_HEADER = ("#id num_p mvir mbound_vir rvir vmax rvmax vrms x y z "
                "vx vy vz idx i_so i_ph num_cp mmetric")


def write_catalog(path: Path, ids, parent, num_p, pos, mvir, rvir_kpc) -> Path:
    """Minimal Rockstar ASCII in the column order load_rockstar_ascii parses."""
    idx_of = {int(h): k for k, h in enumerate(ids)}
    lines = [ASCII_HEADER]
    for k, h in enumerate(ids):
        i_so = idx_of[int(parent[k])] if int(parent[k]) >= 0 else -1
        lines.append(
            f"{int(h)} {int(num_p[k])} {mvir[k]:.6e} {mvir[k]:.6e} "
            f"{rvir_kpc[k]:.4f} 100.0 10.0 50.0 "
            f"{pos[k][0]:.6f} {pos[k][1]:.6f} {pos[k][2]:.6f} 0.0 0.0 0.0 "
            f"{k} {i_so} -1 0 0.0"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def build_side(root: Path, box: str, tag: str, members, parent, pos_lookup):
    """Write one side's catalog + owner array; return the dir and the catalog."""
    ids = sorted(members)
    num_p = [members[h].size for h in ids]
    centre = np.stack([pos_lookup[members[h]].mean(axis=0) for h in ids])
    mvir = [1e13 * n for n in num_p]
    rvir_kpc = [1.5 * CELL * 1000.0] * len(ids)
    d = root / "halos_particles" / f"{box}__{tag}__{tag}"
    write_catalog(d / f"{tag}_rockstar" / "halos_0.0.ascii",
                  ids, [parent[h] for h in ids], num_p, centre, mvir, rvir_kpc)
    cat = HaloCatalog(
        ids=np.asarray(ids, dtype=np.int64),
        parent_ids=np.asarray([parent[h] for h in ids], dtype=np.int64),
        mvir=np.asarray(mvir, dtype=np.float64),
        rvir=np.asarray(rvir_kpc), vmax=np.full(len(ids), 100.0),
        pos=centre, vel=np.zeros((len(ids), 3)),
        num_p=np.asarray(num_p, dtype=np.int64),
    )
    pf = write_particles_file(d / "tmp.particles", cat, members)
    np.save(d / f"{box}_{tag}_owner.npy", stream_owner_assignment(pf, N_PART))
    pf.unlink()
    return d


def blob(origin, n=6):
    """A compact run of Lagrangian sites, big enough to clear --min-particles."""
    ox, oy, oz = origin
    return lagrangian_ids([(ox + i, oy, oz) for i in range(n)])


@pytest.fixture
def world(tmp_path):
    """Host 0: the same object from the same ids in both boxes.

    Host 1 is the case the whole study is about -- B puts a halo at the same
    place as A's, but builds it out of a *different* Lagrangian patch, which it
    can only do by displacing those particles half a box. Nothing short of
    comparing ids can tell this apart from host 0.
    """
    from cosmo_sr.data.preprocess_srs import disnorm

    a_members = {0: blob((1, 1, 1)), 1: blob((1, 8, 8))}
    b_members = {0: blob((1, 1, 1)), 1: blob((9, 8, 8))}
    parent = {0: -1, 1: -1}

    a_pos = zero_positions()
    b_pos = zero_positions()
    b_pos[b_members[1], 0] = (b_pos[b_members[1], 0] - BOX / 2) % BOX

    field = np.zeros((6, NG, NG, NG), dtype=np.float32)
    b_field_arr = field.copy()
    moved = np.unravel_index(b_members[1], (NG, NG, NG))
    b_field_arr[0][moved] = float(disnorm(np.array([-BOX / 2 * 1000.0]), z=0.0)[0])

    a_dir = build_side(tmp_path, "set0", "hr", a_members, parent, a_pos)
    b_dir = build_side(tmp_path, "set0", "base", b_members, parent, b_pos)

    a_field = tmp_path / "a.npy"
    b_field = tmp_path / "b.npy"
    np.save(a_field, field)
    np.save(b_field, b_field_arr)
    return {"tmp": tmp_path, "a_dir": a_dir, "b_dir": b_dir,
            "a_field": a_field, "b_field": b_field, "pos": a_pos,
            "a_members": a_members, "b_members": b_members}


def run(world, out, extra=()):
    argv = [
        "--box", "set0", "--a", "hr", "--b", "base",
        "--a-dir", str(world["a_dir"]), "--b-dir", str(world["b_dir"]),
        "--a-field", str(world["a_field"]), "--b-field", str(world["b_field"]),
        "--reward-root", str(world["tmp"]), "--out", str(out),
        "--ng", str(NG), "--tile", "4", "--chunks", "4", "--boxsize", str(BOX),
        "--classes", "hosts", "--min-particles", "3",
        "--sample-particles", "500",
    ]
    assert pi_script.main(argv + list(extra)) == 0
    return json.loads((Path(out) / "summary.json").read_text())


def rows_of(out):
    return [json.loads(l) for l in (Path(out) / "pairs.jsonl").read_text().splitlines()]


def test_identical_and_rebuilt_halos_are_told_apart(world, tmp_path):
    out = tmp_path / "out"
    summary = run(world, out)
    rows = {r["a_id"]: r for r in rows_of(out)}
    assert set(rows) == {0, 1}

    # Halo 0 is the same object built from the same ids in both boxes.
    same = rows[0]
    assert same["matched"]
    assert same["set"]["jaccard"] == 1.0
    assert same["dest"]["top_fraction"] == 1.0
    assert same["dest"]["frac_unbound"] == 0.0
    assert same["tiles"]["intersection"] == pytest.approx(1.0)

    # Halo 1 is matched, sits at the same place, and has the same mass -- and
    # is made of entirely different mass elements. Every position-based metric
    # is happy; only the ids show it.
    rebuilt = rows[1]
    assert rebuilt["matched"]
    assert rebuilt["match_dist_mpc_h"] == pytest.approx(0.0, abs=1e-4)
    assert rebuilt["set"]["jaccard"] == 0.0
    assert rebuilt["dest"]["frac_unbound"] == 1.0     # B binds none of them
    assert rebuilt["tiles"]["intersection"] == pytest.approx(0.0)
    assert not rebuilt["tiles"]["same_dominant_tile"]
    # spatial slack does NOT rescue it: A's particles are sitting right there
    # in B, they are simply not what B bound into the halo.
    assert rebuilt["radius_a_in_b"]["rvir1"] > 0.5

    assert summary["hosts"]["n_analysed"] == 2
    assert summary["hosts"]["median_jaccard"] == pytest.approx(0.5)
    # The random-id null must sit far below any real overlap, or the Jaccards
    # above would not mean anything.
    assert summary["hosts"]["median_null_jaccard"] == 0.0
    assert summary["a_owner_consistency"]["ok"]
    assert summary["b_owner_consistency"]["ok"]


def test_translation_reads_as_bulk_not_reshuffle(world, tmp_path):
    """Same ids, moved rigidly: the residual only has to translate them."""
    shift_mpc = 0.5
    field = np.zeros((6, NG, NG, NG), dtype=np.float32)
    from cosmo_sr.data.preprocess_srs import disnorm
    # the field is catnorm displacement in kpc/h; disnorm(undo=False) is the
    # forward normalisation field_to_particles inverts
    field[0] = float(disnorm(np.array([shift_mpc * 1000.0]), z=0.0)[0])
    np.save(world["b_field"], field)

    out = tmp_path / "out_shift"
    run(world, out)
    row = {r["a_id"]: r for r in rows_of(out)}[0]
    d = row["disp_all"]
    assert d["bulk_mpc_h"] == pytest.approx(shift_mpc, rel=1e-3)
    assert d["residual_rms_mpc_h"] == pytest.approx(0.0, abs=1e-6)
    assert d["coherent_fraction"] == pytest.approx(1.0)
    # a rigid shift below the chunk size keeps almost everything in its chunk
    assert row["chunk_all"]["frac_same_or_adjacent"] == pytest.approx(1.0)


def test_a_catalog_without_its_member_table_is_not_a_cache(tmp_path):
    """The `.particles` table is deleted by a successful run, so a catalog
    directory without one must not short-circuit the halo finder -- doing so
    fails 20 seconds in with a message blaming the Rockstar config."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reward"))
    from rockstar_particles import must_rerun_halo_finder

    work = tmp_path / "set8__hr__hr"
    (work / "hr_rockstar").mkdir(parents=True)
    (work / "hr_rockstar" / "halos_0.0.ascii").write_text("#id\n")

    # catalog present, table already deleted -> must run again
    assert must_rerun_halo_finder(work, "hr", reuse=True)
    (work / "hr_rockstar" / "halos_0.0.particles").write_text("#x\n")
    assert not must_rerun_halo_finder(work, "hr", reuse=True)
    assert must_rerun_halo_finder(work, "hr", reuse=False)
    # a directory that was never populated
    assert must_rerun_halo_finder(tmp_path / "nope", "hr", reuse=True)


def test_metrics_npz_and_figures_are_self_contained(world, tmp_path):
    out = tmp_path / "out_plot"
    run(world, out)
    z = np.load(out / "metrics.npz", allow_pickle=False)
    assert z["hosts__jaccard"].shape == (2,)
    assert json.loads(str(z["summary_json"]))["box"] == "set0"

    # The plot stage must run off metrics.npz alone -- delete the inputs first,
    # since figures are supposed to be redrawable without recomputation.
    (out / "pairs.jsonl").unlink()
    assert pi_script.main(["--stage", "plot", "--out", str(out),
                           "--reward-root", str(world["tmp"])]) == 0
    figs = sorted(p.name for p in (out / "figures").glob("*.png"))
    assert figs == ["fig1_identity.png", "fig2_translation_vs_reshuffle.png",
                    "fig3_radius.png", "fig4_chunk.png", "fig5_fate_vs_mass.png"]
