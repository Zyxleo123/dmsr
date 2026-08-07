#!/usr/bin/env python
"""Do an SR2 halo and its HR counterpart consist of the SAME particles?

Both boxes are displacement fields on one Lagrangian lattice, so particle
``i`` in SR2 and particle ``i`` in HR are the same mass element by construction
(``cosmo_sr.eval.particles.field_to_particles`` sets ``id = arange(Ng**3)``).
Membership can therefore be compared as *sets of ids*, exactly, and the
comparison is run at three granularities:

  identity  exact id-set overlap of a matched pair (Jaccard / purity /
            completeness);
  radius    where the non-shared ids actually are -- the fraction of one
            object's members that sit within r of the other object's centre,
            in the other box;
  chunk     Lagrangian tile profiles (which patch of the ICs each object was
            built from) and Eulerian chunk crossings (whether the correction
            pushes a particle out of the region its generator tile covers).

Two measurements here need **no halo matching at all**, and they are the ones
to trust first, because matching is the weakest link in every SR2-vs-HR
comparison. For each SR2 object, ``owner_A[members_B]`` says what HR did with
exactly those particles: bound them to one object (and which), split them
across several, or left them unbound. The matched-pair numbers are reported
alongside so the two views can be checked against each other.

The residual question this exists to answer: a pointwise SR2->HR residual acts
on particle ``i`` whatever SR2 made of it. If a matched pair shares its ids and
differs by a coherent bulk shift, the residual only has to translate a valid
object. If the id sets disagree, the residual must take a self-consistent SR2
subhalo apart and rebuild an HR one out of different mass elements.
``coherent_fraction`` measures which of those is happening.

Controls, all computed by default:
  * the whole-field displacement baseline (what a *typical* particle does), so
    halo members can be judged against it rather than against zero;
  * a size-matched random id set, which fixes the null level of every overlap;
  * run the script with A and B both SR2 at different noise seeds to separate
    "the residual has to fix this" from "SR2's own sampling noise does this
    anyway" -- if seed-to-seed disagreement matches SR2-to-HR disagreement, no
    residual can be trained to remove it.

    python scripts/sr2/particle_identity.py --box set8 --a hr --b base:0
    python scripts/sr2/particle_identity.py --box set8 --a base:0 --b base:1
    python scripts/sr2/particle_identity.py --box set8 --stage plot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosmo_sr.eval.halo_match import match_hosts  # noqa: E402
from cosmo_sr.eval.particle_identity import (  # noqa: E402
    as_flat_catalog, build_owner_index, check_owner_consistency, child_map,
    displacement_stats, eulerian_chunk_shift, lagrangian_positions,
    profile_overlap, radius_fractions, remap_to_roots, set_metrics,
    tile_profile,
)
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402

REWARD_ROOT = Path("/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
PAIRED_ROOT = Path("/zfsauton/scratch/yixiz/DMSR/paired_catnorm")
RADII_RVIR = (0.5, 1.0, 2.0)
RADII_FIXED_MPC = (0.1, 0.5, 1.0, 2.0)


def banner(msg: str) -> None:
    print(f"=== {msg} ===", flush=True)


# --------------------------------------------------------------------------
# Locating one side of the comparison
# --------------------------------------------------------------------------

class Side:
    """One box: catalog, per-particle owner array, and positions."""

    def __init__(self, spec: str, box: str, reward_root: Path,
                 field_override: str = "", dir_override: str = ""):
        self.spec = spec
        source, _, seed = spec.partition(":")
        self.source = source
        self.seed = int(seed) if seed else 0
        self.box = box
        # Seed 0 keeps the plain `base` tag so the existing Experiment-0
        # catalogs are reused rather than recomputed; further seeds (the
        # noise-only control) get their own tag and their own directory.
        self.tag = (source if self.seed == 0 or source == "hr"
                    else f"{source}_seed{self.seed}")
        self.dir = (Path(dir_override) if dir_override else
                    reward_root / "halos_particles" / f"{box}__{source}__{self.tag}")
        self.field_override = field_override
        self.reward_root = reward_root

    def catalog_path(self) -> Path:
        p = self.dir / f"{self.tag}_rockstar" / "halos_0.0.ascii"
        if not p.is_file():
            raise SystemExit(
                f"no catalog at {p}\nRun the member-id Rockstar pass first:\n"
                f"  sbatch scripts/slurm/particle_identity_prep_cpu.sbatch "
                f"BOXES={self.box} SOURCES={self.source} BASE_SEED={self.seed}"
            )
        return p

    def owner_path(self) -> Path:
        p = self.dir / f"{self.box}_{self.tag}_owner.npy"
        if not p.is_file():
            raise SystemExit(
                f"no per-particle owner array at {p}\n"
                "It comes from scripts/reward/rockstar_particles.py "
                "--write-assignment; a run that predates that flag has the "
                "tile weights but not the ownership, and only a new "
                "halo-finder pass can produce it:\n"
                f"  sbatch scripts/slurm/particle_identity_prep_cpu.sbatch "
                f"BOXES={self.box} SOURCES={self.source} BASE_SEED={self.seed}"
            )
        return p

    def field_path(self) -> Path:
        if self.field_override:
            return Path(self.field_override)
        if self.source == "hr":
            return PAIRED_ROOT / "hr" / f"{self.box}.npy"
        hits = sorted((self.reward_root / "cache" / "sr2_base").glob(
            f"{self.box}_seed{self.seed}_*.npy"))
        if not hits:
            raise SystemExit(
                f"no cached SR2 field for {self.box} seed {self.seed} under "
                f"{self.reward_root / 'cache' / 'sr2_base'}; run "
                "scripts/slurm/cache_sr2_base.sbatch"
            )
        return hits[0]


def load_side(side: Side, boxsize: float, redshift: float) -> Dict:
    banner(f"loading {side.spec} ({side.box})")
    cat = load_rockstar_ascii(side.catalog_path())
    owner = np.load(side.owner_path(), mmap_mode=None)
    index = build_owner_index(owner)
    consistency = check_owner_consistency(cat, index)
    print(f"    {cat.n} objects, {index.n_part - index.n_unowned} bound "
          f"particles; len(members)==num_p for {consistency['n_exact']}/"
          f"{consistency['n_catalog_objects']}", flush=True)
    if not consistency["ok"]:
        # Not fatal: Rockstar can bind a particle to a clump it declined to
        # print, which leaves it unowned. Loud, because if the fraction is
        # large the id sets are not the objects' particle lists.
        print(f"    !! leaf attribution is not exact "
              f"(max |diff| {consistency['max_abs_diff']}, "
              f"{consistency['frac_particles_unowned']:.4f} of particles "
              f"unowned) -- treat set sizes as lower bounds", flush=True)
    t0 = time.time()
    field = np.load(side.field_path(), mmap_mode="r")
    pos = lagrangian_positions(field, boxsize_mpc_h=boxsize, redshift=redshift)
    del field
    print(f"    positions from {side.field_path().name} in "
          f"{time.time() - t0:.0f}s", flush=True)
    return {"cat": cat, "owner": owner, "index": index, "pos": pos,
            "owner_root": remap_to_roots(owner, cat),
            "consistency": consistency, "children": child_map(cat),
            "spec": side.spec, "field": str(side.field_path())}


# --------------------------------------------------------------------------
# Matching-free: what did box B do with box A's particles?
# --------------------------------------------------------------------------

def destination_spectrum(ids: np.ndarray, owner_other: np.ndarray) -> Dict:
    """Where the *other* box put exactly these particles.

    No halo matching involved: the ids are known, so the other box's ownership
    of them is a lookup. ``top_fraction`` near 1 means the object survives as a
    single object; a low ``top_fraction`` with many owners means it was split.
    """
    if ids.size == 0:
        return {"n": 0, "frac_unbound": 0.0, "top_fraction": 0.0,
                "top_owner": -1, "n_owners": 0, "frac_in_top3": 0.0}
    own = owner_other[ids]
    unbound = int(np.count_nonzero(own < 0))
    bound = own[own >= 0]
    if bound.size == 0:
        return {"n": int(ids.size), "frac_unbound": 1.0, "top_fraction": 0.0,
                "top_owner": -1, "n_owners": 0, "frac_in_top3": 0.0}
    uniq, cnt = np.unique(bound, return_counts=True)
    order = np.argsort(-cnt)
    return {
        "n": int(ids.size),
        "frac_unbound": float(unbound / ids.size),
        "top_owner": int(uniq[order[0]]),
        "top_fraction": float(cnt[order[0]] / ids.size),
        "frac_in_top3": float(cnt[order[:3]].sum() / ids.size),
        "n_owners": int(uniq.size),
    }


# --------------------------------------------------------------------------
# Per-pair analysis
# --------------------------------------------------------------------------

def select(cat, kind: str, min_particles: int):
    sel = cat.hosts() if kind == "hosts" else cat.subhalos()
    return sel._mask(np.asarray(sel.num_p, dtype=np.int64) >= int(min_particles))


def members_of(side: Dict, cat, hid: int, kind: str) -> np.ndarray:
    """A host's set includes its substructure; a subhalo's is its own list."""
    if kind == "hosts":
        return side["index"].members_with_substructure(
            side["cat"], hid, children=side["children"])
    return side["index"].members(hid)


def analyse_class(
    a: Dict, b: Dict, kind: str, *, boxsize: float, grid: TileGrid,
    min_particles: int, max_pairs: int, n_chunk_per_axis: int, rng,
) -> Tuple[List[Dict], Dict]:
    cat_a = select(a["cat"], kind, min_particles)
    cat_b = select(b["cat"], kind, min_particles)
    banner(f"{kind}: {cat_a.n} in A, {cat_b.n} in B (num_p >= {min_particles})")
    if cat_a.n == 0 or cat_b.n == 0:
        return [], {"n_a": int(cat_a.n), "n_b": int(cat_b.n)}

    t0 = time.time()
    res = match_hosts(as_flat_catalog(cat_a), as_flat_catalog(cat_b),
                      boxsize_mpc_h=boxsize)
    b_row = {int(i): k for k, i in enumerate(cat_b.ids)}
    print(f"    matched {int(np.count_nonzero(res.sr_ids >= 0))}/{cat_a.n} "
          f"in {time.time() - t0:.0f}s", flush=True)

    order = np.argsort(-np.asarray(cat_a.mvir))
    if max_pairs > 0 and order.size > max_pairs:
        # Most massive first, then a random tail: the heavy end is where the
        # residual matters and the tail keeps the sample representative.
        head = order[: max_pairs // 2]
        tail = rng.choice(order[max_pairs // 2:], size=max_pairs - head.size,
                          replace=False)
        order = np.concatenate([head, tail])
        print(f"    analysing {order.size} of {cat_a.n} objects "
              f"(--max-pairs)", flush=True)

    a_row = {int(i): k for k, i in enumerate(cat_a.ids)}
    rows: List[Dict] = []
    for k in order:
        hid = int(cat_a.ids[k])
        ids_a = members_of(a, cat_a, hid, kind)
        if ids_a.size == 0:
            continue
        rvir_mpc = float(cat_a.rvir[k]) * 1e-3
        rec: Dict = {
            "class": kind, "a_id": hid,
            "a_log10_mvir": float(np.log10(max(float(cat_a.mvir[k]), 1.0))),
            "a_num_p": int(cat_a.num_p[k]), "a_rvir_mpc_h": rvir_mpc,
            "a_n_members": int(ids_a.size),
        }
        # --- matching-free: what B did with exactly these particles.
        # Hosts are scored against B's *top-level* ownership: a satellite's
        # particles are owned by the satellite, so leaf ownership would score a
        # host whose material B binds perfectly -- into that host's own
        # subhalos -- as fragmented.
        rec["dest"] = destination_spectrum(
            ids_a, b["owner_root"] if kind == "hosts" else b["owner"])
        rec["disp_all"] = displacement_stats(ids_a, a["pos"], b["pos"], boxsize)
        rec["chunk_all"] = eulerian_chunk_shift(
            ids_a, a["pos"], b["pos"], boxsize, n_chunk_per_axis)
        # a size-matched random control for the overlap null
        ctrl = rng.integers(0, a["index"].n_part, size=ids_a.size)
        rec["null_jaccard"] = set_metrics(
            ids_a, np.unique(ctrl))["jaccard"]

        sid = int(res.sr_ids[a_row[hid]]) if hid in a_row else -1
        rec["b_id"] = sid
        rec["matched"] = bool(sid >= 0)
        if sid >= 0:
            j = b_row[sid]
            ids_b = members_of(b, cat_b, sid, kind)
            rec["b_log10_mvir"] = float(np.log10(max(float(cat_b.mvir[j]), 1.0)))
            rec["b_num_p"] = int(cat_b.num_p[j])
            rec["b_n_members"] = int(ids_b.size)
            rec["match_dist_mpc_h"] = float(np.linalg.norm(
                (np.asarray(cat_a.pos[k]) - np.asarray(cat_b.pos[j])
                 + boxsize / 2) % boxsize - boxsize / 2))
            # 1. identity
            rec["set"] = set_metrics(ids_a, ids_b)
            # 2. radius, both directions
            radii = [r * rvir_mpc for r in RADII_RVIR] + list(RADII_FIXED_MPC)
            names = [f"rvir{r:g}" for r in RADII_RVIR] + \
                    [f"mpc{r:g}" for r in RADII_FIXED_MPC]
            rec["radius_a_in_b"] = radius_fractions(
                ids_a, b["pos"], cat_b.pos[j], boxsize, radii, names)
            rec["radius_b_in_a"] = radius_fractions(
                ids_b, a["pos"], cat_a.pos[k], boxsize, radii, names)
            # 3. chunk: same patch of the ICs?
            rec["tiles"] = profile_overlap(tile_profile(ids_a, grid),
                                           tile_profile(ids_b, grid))
            # displacement of the shared material only
            shared = np.intersect1d(ids_a, ids_b, assume_unique=True)
            rec["disp_shared"] = displacement_stats(
                shared, a["pos"], b["pos"], boxsize)
        rows.append(rec)

    matched = [r for r in rows if r["matched"]]
    summary = {
        "n_a": int(cat_a.n), "n_b": int(cat_b.n), "n_analysed": len(rows),
        "n_matched": len(matched),
        "frac_matched": len(matched) / len(rows) if rows else 0.0,
        "n_b_unclaimed": int(cat_b.n - np.count_nonzero(res.sr_ids >= 0)),
    }
    if matched:
        def med(path):
            vals = [_dig(r, path) for r in matched]
            vals = [v for v in vals if v is not None]
            return float(np.median(vals)) if vals else float("nan")

        def wmean(path):
            """Per-particle mean: each object weighted by its member count.

            An unweighted average over objects is an average over *halos*, and
            the halo population is dominated by small ones. Where the question
            is about particles, weight by particles.
            """
            pairs = [(_dig(r, path), r["a_n_members"]) for r in matched]
            pairs = [(v, w) for v, w in pairs if v is not None and w]
            if not pairs:
                return float("nan")
            v = np.array([p[0] for p in pairs], dtype=np.float64)
            w = np.array([p[1] for p in pairs], dtype=np.float64)
            return float(np.sum(v * w) / np.sum(w))
        summary.update({
            "median_jaccard": med("set.jaccard"),
            "median_purity": med("set.purity"),
            "median_completeness": med("set.completeness"),
            "median_null_jaccard": float(np.median(
                [r["null_jaccard"] for r in matched])),
            "median_top_fraction": med("dest.top_fraction"),
            "median_frac_unbound_in_other": med("dest.frac_unbound"),
            "median_coherent_fraction": med("disp_all.coherent_fraction"),
            "median_bulk_mpc_h": med("disp_all.bulk_mpc_h"),
            "median_residual_rms_mpc_h": med("disp_all.residual_rms_mpc_h"),
            "median_rvir_mpc_h": float(np.median(
                [r["a_rvir_mpc_h"] for r in matched])),
            # Both, on purpose. A compact halo lying inside one chunk scores 1.0
            # whatever its members do, and those halos are the majority, so the
            # median over objects saturates at ~1 and hides the straddling tail:
            # measured 0.997 median against 0.884 particle-weighted on set8.
            # Chunk crossing is driven by proximity to a face, not by how far
            # the particles move, so the per-particle number is the one that
            # answers "does the correction leave the chunk".
            "median_frac_same_chunk": med("chunk_all.frac_same_chunk"),
            "weighted_frac_same_chunk": wmean("chunk_all.frac_same_chunk"),
            "weighted_frac_same_or_adjacent_chunk": wmean(
                "chunk_all.frac_same_or_adjacent"),
            "frac_objects_losing_over_10pct_from_chunk": float(np.mean(
                [1.0 if (v := _dig(r, "chunk_all.frac_same_chunk")) is not None
                 and v < 0.9 else 0.0 for r in matched])),
            "median_tile_intersection": med("tiles.intersection"),
            "median_radius_b_in_a_rvir1": med("radius_b_in_a.rvir1"),
            "median_radius_a_in_b_rvir1": med("radius_a_in_b.rvir1"),
        })
    return rows, summary


def _dig(rec: Dict, path: str):
    cur = rec
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

def field_baseline(a: Dict, b: Dict, boxsize: float, n_sample: int,
                   n_chunk_per_axis: int, rng) -> Dict:
    """What a *typical* particle does, so halo members have something to beat."""
    n = a["index"].n_part
    ids = np.unique(rng.integers(0, n, size=int(min(n_sample, n))))
    out = {"n_sample": int(ids.size)}
    out.update(displacement_stats(ids, a["pos"], b["pos"], boxsize))
    out["chunk"] = eulerian_chunk_shift(ids, a["pos"], b["pos"], boxsize,
                                        n_chunk_per_axis)
    out["frac_bound_a"] = float(np.mean(a["owner"][ids] >= 0))
    out["frac_bound_b"] = float(np.mean(b["owner"][ids] >= 0))
    return out


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_analyse(args) -> Dict:
    boxsize, redshift = float(args.boxsize), float(args.redshift)
    rng = np.random.default_rng(int(args.seed))
    root = Path(args.reward_root)
    a = load_side(Side(args.a, args.box, root, args.a_field, args.a_dir),
                  boxsize, redshift)
    b = load_side(Side(args.b, args.box, root, args.b_field, args.b_dir),
                  boxsize, redshift)
    grid = TileGrid(ng_hr=int(args.ng), tile_hr=int(args.tile),
                    boxsize_mpc_h=boxsize)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "box": args.box, "a": args.a, "b": args.b,
        "a_field": a["field"], "b_field": b["field"],
        "boxsize_mpc_h": boxsize, "redshift": redshift,
        "ng_hr": int(args.ng), "tile_hr": int(args.tile),
        "n_chunk_per_axis": int(args.chunks),
        "min_particles": int(args.min_particles),
        "a_owner_consistency": a["consistency"],
        "b_owner_consistency": b["consistency"],
    }

    banner("control: whole-field displacement baseline")
    summary["field_baseline"] = field_baseline(
        a, b, boxsize, int(args.sample_particles), int(args.chunks), rng)
    fb = summary["field_baseline"]
    print(f"    typical particle moves {fb['median_mpc_h']:.3f} Mpc/h "
          f"(rms {fb['rms_mpc_h']:.3f}); "
          f"{fb['chunk']['frac_same_chunk']:.3f} stay in their "
          f"{fb['chunk']['chunk_mpc_h']:.1f} Mpc/h chunk", flush=True)

    all_rows: List[Dict] = []
    for kind in [k.strip() for k in args.classes.split(",") if k.strip()]:
        rows, s = analyse_class(
            a, b, kind, boxsize=boxsize, grid=grid,
            min_particles=int(args.min_particles),
            max_pairs=int(args.max_pairs), n_chunk_per_axis=int(args.chunks),
            rng=rng)
        summary[kind] = s
        all_rows.extend(rows)
        for k, v in s.items():
            print(f"    {kind:9s} {k:32s} {v}", flush=True)

    jsonl = out / "pairs.jsonl"
    with open(jsonl, "w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, default=_json_default) + "\n")
    write_metrics_npz(out / "metrics.npz", all_rows, summary)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    banner(f"wrote {jsonl} ({len(all_rows)} rows), metrics.npz, summary.json")
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


FLAT_FIELDS = [
    ("a_log10_mvir", "a_log10_mvir"),
    ("a_num_p", "a_num_p"),
    ("a_n_members", "a_n_members"),
    ("a_rvir_mpc_h", "a_rvir_mpc_h"),
    ("matched", "matched"),
    ("null_jaccard", "null_jaccard"),
    ("jaccard", "set.jaccard"),
    ("purity", "set.purity"),
    ("completeness", "set.completeness"),
    ("top_fraction", "dest.top_fraction"),
    ("frac_unbound", "dest.frac_unbound"),
    ("n_owners", "dest.n_owners"),
    ("bulk", "disp_all.bulk_mpc_h"),
    ("resid_rms", "disp_all.residual_rms_mpc_h"),
    ("rms", "disp_all.rms_mpc_h"),
    ("coherent", "disp_all.coherent_fraction"),
    ("shared_coherent", "disp_shared.coherent_fraction"),
    ("shared_resid_rms", "disp_shared.residual_rms_mpc_h"),
    ("median_disp", "disp_all.median_mpc_h"),
    ("same_chunk", "chunk_all.frac_same_chunk"),
    ("same_or_adj_chunk", "chunk_all.frac_same_or_adjacent"),
    ("tile_intersection", "tiles.intersection"),
    ("tile_same_dominant", "tiles.same_dominant_tile"),
    ("match_dist", "match_dist_mpc_h"),
]
RADIUS_KEYS = [f"rvir{r:g}" for r in RADII_RVIR] + \
              [f"mpc{r:g}" for r in RADII_FIXED_MPC]


def write_metrics_npz(path: Path, rows: List[Dict], summary: Dict) -> Path:
    """Flatten to arrays so figures redraw without re-reading the catalogs."""
    arrays: Dict[str, np.ndarray] = {}
    for kind in sorted({r["class"] for r in rows}):
        sub = [r for r in rows if r["class"] == kind]
        for name, path_ in FLAT_FIELDS:
            vals = [_dig(r, path_) for r in sub]
            arrays[f"{kind}__{name}"] = np.array(
                [np.nan if v is None else float(v) for v in vals],
                dtype=np.float64)
        for direction in ("radius_a_in_b", "radius_b_in_a"):
            for rk in RADIUS_KEYS:
                vals = [_dig(r, f"{direction}.{rk}") for r in sub]
                arrays[f"{kind}__{direction}__{rk}"] = np.array(
                    [np.nan if v is None else float(v) for v in vals],
                    dtype=np.float64)
    arrays["summary_json"] = np.array(json.dumps(summary, default=_json_default))
    np.savez_compressed(path, **arrays)
    return path


def stage_plot(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    z = np.load(out / "metrics.npz", allow_pickle=False)
    summary = json.loads(str(z["summary_json"]))
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    kinds = sorted({k.split("__")[0] for k in z.files if "__" in k})
    label = f"{summary['box']}: A={summary['a']} vs B={summary['b']}"

    def col(kind, name):
        return z[f"{kind}__{name}"]

    # fig1 -- identity: overlap distributions against their null
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4.2),
                             squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        m = col(kind, "matched") > 0.5
        for name, style in (("jaccard", "-"), ("purity", "--"),
                            ("completeness", ":")):
            v = col(kind, name)[m]
            v = v[np.isfinite(v)]
            if v.size:
                ax.hist(v, bins=40, range=(0, 1), histtype="step",
                        linestyle=style, label=f"{name} (med {np.median(v):.2f})")
        nj = col(kind, "null_jaccard")
        nj = nj[np.isfinite(nj)]
        if nj.size:
            ax.axvline(float(np.median(nj)), color="k", lw=0.8,
                       label=f"random-id null ({np.median(nj):.3f})")
        ax.set_xlabel("id-set overlap of matched pair")
        ax.set_ylabel("objects")
        ax.set_title(f"{kind}: exact particle identity")
        ax.legend(fontsize=7)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig1_identity.png", dpi=130)
    plt.close(fig)

    # fig2 -- the residual question: coherent translation vs dispersal
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4.2),
                             squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        rvir = col(kind, "a_rvir_mpc_h")
        bulk, resid = col(kind, "bulk"), col(kind, "resid_rms")
        ok = np.isfinite(bulk) & np.isfinite(resid) & (rvir > 0)
        ax.scatter(bulk[ok] / rvir[ok], resid[ok] / rvir[ok], s=3, alpha=0.3)
        lim = max(1e-2, float(np.nanpercentile(
            np.concatenate([bulk[ok] / rvir[ok], resid[ok] / rvir[ok]]), 99)))
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="equal")
        ax.axhline(1.0, color="r", lw=0.8,
                   label="scatter = Rvir (object destroyed)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("bulk shift / Rvir  (translation)")
        ax.set_ylabel("residual scatter / Rvir  (reshuffle)")
        ax.set_title(f"{kind}: what the residual must do")
        ax.legend(fontsize=7)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig2_translation_vs_reshuffle.png", dpi=130)
    plt.close(fig)

    # fig3 -- radius granularity: recovery as the radius is relaxed
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4.2),
                             squeeze=False)
    xs = list(RADII_RVIR)
    for ax, kind in zip(axes[0], kinds):
        for direction, style in (("radius_b_in_a", "-o"),
                                 ("radius_a_in_b", "--s")):
            med, lo, hi = [], [], []
            for r in RADII_RVIR:
                v = z[f"{kind}__{direction}__rvir{r:g}"]
                v = v[np.isfinite(v)]
                med.append(np.median(v) if v.size else np.nan)
                lo.append(np.quantile(v, 0.25) if v.size else np.nan)
                hi.append(np.quantile(v, 0.75) if v.size else np.nan)
            ax.plot(xs, med, style, label=direction)
            ax.fill_between(xs, lo, hi, alpha=0.15)
        jac = col(kind, "jaccard")
        jac = jac[np.isfinite(jac)]
        if jac.size:
            ax.axhline(float(np.median(jac)), color="k", lw=0.8,
                       label="exact-membership Jaccard")
        ax.set_xlabel("radius / Rvir")
        ax.set_ylabel("fraction of members inside")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{kind}: identity with spatial slack")
        ax.legend(fontsize=7)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig3_radius.png", dpi=130)
    plt.close(fig)

    # fig4 -- chunk granularity, both meanings
    fig, axes = plt.subplots(1, 2 * len(kinds), figsize=(5.0 * 2 * len(kinds), 4.0),
                             squeeze=False)
    fb = summary.get("field_baseline", {}).get("chunk", {})
    for i, kind in enumerate(kinds):
        ax = axes[0][2 * i]
        v = col(kind, "same_chunk")
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(v, bins=40, range=(0, 1), histtype="step",
                    label=f"members (med {np.median(v):.3f})")
        if fb:
            ax.axvline(fb.get("frac_same_chunk", np.nan), color="k", lw=0.8,
                       label=f"all particles ({fb.get('frac_same_chunk', 0):.3f})")
        ax.set_xlabel(f"fraction staying in their "
                      f"{fb.get('chunk_mpc_h', 0):.1f} Mpc/h Eulerian chunk")
        ax.set_ylabel("objects")
        ax.set_title(f"{kind}: Eulerian chunk crossings")
        ax.legend(fontsize=7)

        ax = axes[0][2 * i + 1]
        v = col(kind, "tile_intersection")
        v = v[np.isfinite(v)]
        if v.size:
            ax.hist(v, bins=40, range=(0, 1), histtype="step",
                    label=f"med {np.median(v):.3f}")
        ax.set_xlabel("Lagrangian tile-profile overlap of the matched pair")
        ax.set_ylabel("objects")
        ax.set_title(f"{kind}: same patch of the ICs?")
        ax.legend(fontsize=7)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig4_chunk.png", dpi=130)
    plt.close(fig)

    # fig5 -- matching-free: what B did with A's particles, vs mass
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.5 * len(kinds), 4.2),
                             squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        mass = col(kind, "a_log10_mvir")
        for name, lab in (("top_fraction", "largest single destination"),
                          ("frac_unbound", "unbound in the other box")):
            v = col(kind, name)
            ok = np.isfinite(mass) & np.isfinite(v)
            if not ok.any():
                continue
            edges = np.linspace(np.nanmin(mass[ok]), np.nanmax(mass[ok]), 15)
            idx = np.digitize(mass[ok], edges) - 1
            xs_, ys_ = [], []
            for bcell in range(len(edges) - 1):
                sel = idx == bcell
                if np.count_nonzero(sel) >= 5:
                    xs_.append(0.5 * (edges[bcell] + edges[bcell + 1]))
                    ys_.append(np.median(v[ok][sel]))
            ax.plot(xs_, ys_, "-o", ms=3, label=lab)
        ax.set_xlabel(r"$\log_{10} M_{\rm vir}$ of the A object")
        ax.set_ylabel("fraction of members")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{kind}: fate of A's particles in B (no matching)")
        ax.legend(fontsize=7)
    fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig5_fate_vs_mass.png", dpi=130)
    plt.close(fig)

    banner(f"figures -> {figs}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", default="set8")
    ap.add_argument("--a", default="hr",
                    help="reference side: 'hr' or 'base[:seed]'")
    ap.add_argument("--b", default="base:0",
                    help="candidate side: 'hr' or 'base[:seed]'")
    ap.add_argument("--a-dir", default="", help="override A's halos_particles dir")
    ap.add_argument("--b-dir", default="", help="override B's halos_particles dir")
    ap.add_argument("--a-field", default="", help="override A's field .npy")
    ap.add_argument("--b-field", default="", help="override B's field .npy")
    ap.add_argument("--reward-root", default=str(REWARD_ROOT))
    ap.add_argument("--out", default="",
                    help="output directory (default: "
                         "$REWARD_ROOT/particle_identity/<box>__<a>__<b>)")
    ap.add_argument("--classes", default="hosts,subhalos")
    ap.add_argument("--min-particles", type=int, default=50,
                    help="skip objects below this num_p: below ~50 particles "
                         "Rockstar's membership is itself noisy, and the id "
                         "sets would measure the halo finder, not SR2")
    ap.add_argument("--max-pairs", type=int, default=20000,
                    help="0 = analyse every object")
    ap.add_argument("--sample-particles", type=int, default=2_000_000)
    ap.add_argument("--chunks", type=int, default=8,
                    help="Eulerian chunks per axis (8 = the nsplit=8 tiling)")
    ap.add_argument("--ng", type=int, default=512)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--boxsize", type=float, default=100.0)
    ap.add_argument("--redshift", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="both",
                    choices=("analyse", "plot", "both"))
    args = ap.parse_args(argv)

    if not args.out:
        tagname = f"{args.box}__{args.a}__{args.b}".replace(":", "")
        args.out = str(Path(args.reward_root) / "particle_identity" / tagname)

    if args.stage in ("analyse", "both"):
        stage_analyse(args)
    if args.stage in ("plot", "both"):
        stage_plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
