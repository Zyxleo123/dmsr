#!/usr/bin/env python
"""Build the catalog-proxy training set: candidate fields, features, real labels.

Three stages, run as separate jobs because they want different hardware:

``generate`` (GPU)
    Produce a candidate full-box field and, in the same pass while the field is
    still resident, its **per-tile soft-structure features**. Computing features
    here rather than at training time is the difference between the trainer
    reading a few MB per candidate and re-reading a 3.2 GB array per candidate.
``label`` (CPU)
    Run Rockstar with ``FULL_PARTICLE_CHUNKS = 1`` on an assembled periodic box,
    stream the ~7 GB ``.particles`` table into exact fractional tile summaries
    through :func:`member_weights_from_particles`, and **delete the table in the
    same job**. Nothing bulky survives the allocation.
``index`` (CPU, seconds)
    Join features and labels into one training table and report what is present.

Candidate sources (``--source``)
--------------------------------
``hr``
    The paired HR field. The positive structural anchor: what the catalog looks
    like when the structure is right.
``frozen``
    Frozen SR2 at the base seed. The baseline every ``dR`` is measured against.
``frozen_seed``
    Frozen SR2 at *other* seeds. A negative control: the catalog differences
    between these are what the generator produces for free, so a proxy that
    cannot tell them apart from a real improvement is measuring noise.
``intervention``
    The existing targeted HR interventions
    (:mod:`cosmo_sr.reward.oracle_hr`), regenerated on the proxy-fit boxes at
    several ``alpha``. These are the only rows that vary *locally and
    monotonically* in the thing the actor is trying to change, so they carry
    most of the within-tile ranking signal.
``actor``
    Outputs of early actor checkpoints, appended by the DAgger loop.

Why not "HR versus SR2" alone
-----------------------------
A proxy trained on that pair learns an easy binary distinction -- SR2's high-k
density bias separates the classes perfectly -- and nothing about which local
change helps. It would score well on any aggregate metric and be useless as a
gradient. The ``intervention`` and ``frozen_seed`` sources exist precisely to
make the task be "which of these two versions of *this tile* is better".

    python scripts/reward/collect_catalog_proxy_data.py --stage generate \
        --box set0 --source frozen_seed --seed 7
    python scripts/reward/collect_catalog_proxy_data.py --stage label \
        --box set0 --source frozen_seed --seed 7
    python scripts/reward/collect_catalog_proxy_data.py --stage index
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from _sr2_direct import (  # noqa: E402
    PROJECT_ROOT, add_direct_args, array_sha, assert_not_sealed, banner,
    bins_of, direct_root, file_sha, geometry_of, load_direct_config,
    load_hr, load_lr, manifest_row, model_path_of, read_jsonl, soft_config_of,
    tile_grid_of, write_json,
)

from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.eval.rockstar import run_rockstar_on_particles  # noqa: E402
from cosmo_sr.reward.soft_structure import paired_features, paired_feature_names  # noqa: E402
from cosmo_sr.reward.tiles import (  # noqa: E402
    check_member_consistency, direct_full_box_stats, member_weights_from_particles,
    tile_summaries, write_tile_summaries,
)
from cosmo_sr.tts.sampling import super_resolve_srs_seeded  # noqa: E402
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402

SOURCES = ("hr", "frozen", "frozen_seed", "intervention", "actor")


def candidate_tag(source: str, seed: int, alpha: Optional[float],
                  checkpoint: str = "") -> str:
    if source == "intervention":
        return f"intervention_a{float(alpha):.3f}_seed{int(seed)}"
    if source == "actor":
        return f"actor_{Path(checkpoint).stem or 'ck'}_seed{int(seed)}"
    return f"{source}_seed{int(seed)}"


def candidate_dir(box: str, tag: str, *, create: bool = False) -> Path:
    return direct_root("candidates", f"{box}__{tag}", create=create)


def manifest_path() -> Path:
    return direct_root("candidates", create=True) / "manifest.jsonl"


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def generate_field(cfg, box: str, source: str, seed: int, *, alpha: float,
                   checkpoint: str, device) -> np.ndarray:
    geom = geometry_of(cfg)
    lr = np.asarray(load_lr(cfg, box), dtype=np.float32)

    if source == "hr":
        return np.asarray(load_hr(cfg, box), dtype=np.float32)

    if source in ("frozen", "frozen_seed", "actor"):
        path = Path(checkpoint) if (source == "actor" and checkpoint) \
            else model_path_of(cfg)
        g = load_controlled_generator(path, scale_factor=geom.scale_factor,
                                      device=device, eval_mode=True)
        return super_resolve_srs_seeded(
            g, lr, int(seed), scale_factor=geom.scale_factor,
            nsplit=geom.nsplit, pad=geom.pad, device=device,
            noise_mode="per_tile", progress=True)

    if source == "intervention":
        # The existing Experiment-1 machinery, regenerated on the proxy-fit
        # boxes. Reused rather than reimplemented: the interventions are only
        # useful as ranking signal if they are the SAME operation the oracle
        # study measured.
        from cosmo_sr.reward.oracle_hr import apply_intervention

        base = generate_field(cfg, box, "frozen", seed, alpha=0.0,
                              checkpoint="", device=device)
        hr = np.asarray(load_hr(cfg, box), dtype=np.float32)
        mask_npz = direct_root("intervention_masks") / f"{box}.npz"
        if not mask_npz.is_file():
            raise SystemExit(
                f"no intervention masks for {box} at {mask_npz}; produce them "
                "with scripts/reward/oracle_select_targets.py + "
                "scripts/reward/extract_editor_members.py on the proxy-fit boxes"
            )
        z = np.load(mask_npz)
        return apply_intervention(base, hr, z["mask"], float(alpha), "both")

    raise SystemExit(f"unknown source {source!r}; expected one of {list(SOURCES)}")


def tile_features(cfg, field: np.ndarray, frozen: np.ndarray, device) -> np.ndarray:
    """``(n_tiles, 2F)`` paired soft-structure features, one row per tile.

    Batched over tiles because the CIC deposit is the cost, and 512 separate
    launches of a 64^3 deposit is dominated by launch overhead.
    """
    geom = geometry_of(cfg)
    grid = tile_grid_of(cfg)
    scfg = soft_config_of(cfg)
    rows: List[np.ndarray] = []
    batch = 8
    ids = list(range(grid.n_tiles))
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        sl = [grid.slices(t) for t in chunk]
        cand = np.stack([np.asarray(field[:, s[0], s[1], s[2]]) for s in sl])
        base = np.stack([np.asarray(frozen[:, s[0], s[1], s[2]]) for s in sl])
        c = torch.from_numpy(np.ascontiguousarray(cand, dtype=np.float32)).to(device)
        b = torch.from_numpy(np.ascontiguousarray(base, dtype=np.float32)).to(device)
        with torch.no_grad():
            rows.append(paired_features(c, b, scfg).cpu().numpy())
        del c, b
    return np.concatenate(rows, axis=0)


def stage_generate(cfg, args, device) -> int:
    box, source, seed = args.box, args.source, int(args.seed)
    tag = candidate_tag(source, seed, args.alpha, args.checkpoint)
    work = candidate_dir(box, tag, create=True)
    field_npy = work / "field.npy"
    feats_npz = work / "features.npz"

    if feats_npz.is_file() and (field_npy.is_file() or args.drop_field) and args.reuse:
        banner(f"{box}/{tag}: already generated -> {work}")
        return 0

    t0 = time.time()
    banner(f"{box}/{tag}: generating")
    field = generate_field(cfg, box, source, seed, alpha=args.alpha,
                           checkpoint=args.checkpoint, device=device)
    lr = np.asarray(load_lr(cfg, box), dtype=np.float32)

    # The frozen reference at the SAME seed. The difference block of the feature
    # vector is the part the actor can move; against a different seed it would
    # be measuring the noise draw instead.
    if source == "frozen":
        frozen = field
    else:
        frozen = generate_field(cfg, box, "frozen", seed if source == "frozen_seed"
                                else int(args.base_seed), alpha=0.0,
                                checkpoint="", device=device)

    feats = tile_features(cfg, field, frozen, device)
    np.savez_compressed(
        feats_npz, features=feats.astype(np.float32),
        tile_id=np.arange(feats.shape[0], dtype=np.int64),
        feature_names=np.array(paired_feature_names(soft_config_of(cfg))),
    )
    if not args.drop_field:
        tmp = field_npy.with_suffix(".tmp.npy")
        np.save(tmp, field.astype(np.float32))
        tmp.replace(field_npy)

    model_p = Path(args.checkpoint) if (source == "actor" and args.checkpoint) \
        else model_path_of(cfg)
    row = manifest_row(
        box=box, source=source, tag=tag, seed=int(seed),
        alpha=(None if args.alpha is None else float(args.alpha)),
        field_path=str(field_npy if not args.drop_field else ""),
        features_path=str(feats_npz),
        model_sha=("hr_field" if source == "hr" else file_sha(model_p)),
        model_path=("" if source == "hr" else str(model_p)),
        lr_sha=array_sha(lr),
        n_tiles=int(feats.shape[0]), n_features=int(feats.shape[1]),
        seconds=round(time.time() - t0, 1),
    )
    from _sr2_direct import append_jsonl
    append_jsonl(manifest_path(), row)
    print(f"  features {feats.shape} -> {feats_npz}", flush=True)
    print(f"  {time.time() - t0:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# label
# --------------------------------------------------------------------------- #
def stage_label(cfg, args) -> int:
    box, source, seed = args.box, args.source, int(args.seed)
    tag = candidate_tag(source, seed, args.alpha, args.checkpoint)
    work = candidate_dir(box, tag, create=True)
    field_npy = work / "field.npy"
    jsonl = work / "tile_summaries.jsonl"
    npz = work / "tile_weights.npz"

    if jsonl.is_file() and npz.is_file() and args.reuse:
        banner(f"{box}/{tag}: already labelled -> {jsonl}")
        return 0
    if not field_npy.is_file():
        print(f">>> MISSING INPUT: {field_npy}")
        print(">>> produced by: --stage generate")
        return 0

    grid = tile_grid_of(cfg)
    bins = bins_of(cfg["_reward"])
    d = cfg["_reward"]["data"]

    t0 = time.time()
    banner(f"{box}/{tag}: Rockstar with member-particle ids")
    field = np.load(field_npy, mmap_mode="r")
    particles = field_to_particles(
        np.asarray(field, dtype=np.float32),
        boxsize_kpc_h=float(d.get("boxsize_mpc_h", 100.0)) * 1000.0,
        redshift=float(d.get("redshift", 0.0)))
    del field

    rk = dict(cfg.get("rockstar", {}))
    halo_dir = work / "rockstar"
    cat = run_rockstar_on_particles(
        particles, halo_dir, tag=tag,
        binary=PROJECT_ROOT / rk.get("binary", "external/rockstar/rockstar"),
        cfg=PROJECT_ROOT / rk.get("config",
                                  "configs/sr2_baseline/rockstar_particles.cfg"),
        overwrite=not args.reuse)
    del particles

    tables = sorted(Path(halo_dir, f"{tag}_rockstar").glob("*.particles"))
    if not tables:
        raise SystemExit(
            f"Rockstar wrote no .particles table under {halo_dir}; the config at "
            f"{rk.get('config')} must set FULL_PARTICLE_CHUNKS = 1")

    weights = member_weights_from_particles(
        tables[0], grid, chunk_rows=int(rk.get("chunk_rows", 8_000_000)))
    consistency = check_member_consistency(cat, weights)
    weights.to_npz(npz)

    summaries = tile_summaries(cat, weights, bins, grid, box=box, source=tag)
    write_tile_summaries(jsonl, [summaries[t] for t in sorted(summaries)])

    # Summing tiles must reproduce the direct full-box statistics. Checked here,
    # per candidate, because it is the identity every label downstream rests on.
    direct = direct_full_box_stats(cat, bins)
    pooled = {
        "n_host": np.sum([summaries[t].n_host for t in summaries], axis=0),
        "n_sub": np.sum([summaries[t].n_sub for t in summaries], axis=0),
        "occ_numerator": np.sum([summaries[t].occ_numerator for t in summaries], axis=0),
    }
    residual = {k: float(np.max(np.abs(pooled[k] - direct[k]))) for k in pooled}

    if bool(rk.get("delete_particles", True)) and not args.keep_particles:
        for p in tables:
            size_gb = p.stat().st_size / 1e9
            p.unlink()
            print(f"  deleted {p.name} ({size_gb:.1f} GB) in the job that made it",
                  flush=True)

    write_json(work / "label_report.json", {
        "box": box, "tag": tag, "n_objects": int(cat.n),
        "member_consistency": consistency,
        "tile_sum_vs_direct_max_abs_residual": residual,
        "seconds": round(time.time() - t0, 1),
    })
    from _sr2_direct import append_jsonl
    append_jsonl(manifest_path(), {
        "box": box, "tag": tag, "stage": "label",
        "catalog_path": str(getattr(cat, "path", "")),
        "tile_summaries_path": str(jsonl),
        "tile_weights_path": str(npz),
        "member_consistency_ok": bool(consistency["ok"]),
        "code_commit": manifest_row(box=box, source=source, field_path=str(field_npy),
                                    model_sha="x", lr_sha="x", seed=seed)["code_commit"],
    })
    print(f"  {jsonl}  ({time.time() - t0:.0f}s)", flush=True)
    if not consistency["ok"]:
        print(">>> member consistency FAILED; this candidate must not be used",
              flush=True)
    return 0


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
def stage_index(cfg, args) -> int:
    from cosmo_sr.reward.tiles import read_tile_summaries

    rows = read_jsonl(manifest_path()) if manifest_path().is_file() else []
    have_feats = {(r["box"], r["tag"]): r for r in rows if "features_path" in r}
    have_label = {(r["box"], r["tag"]): r for r in rows if r.get("stage") == "label"}

    out: List[Dict] = []
    for key, gen in sorted(have_feats.items()):
        lab = have_label.get(key)
        if lab is None or not lab.get("member_consistency_ok", False):
            continue
        feats = np.load(gen["features_path"])
        summaries = {int(s.tile_id): s for s in
                     read_tile_summaries(lab["tile_summaries_path"])}
        for i, tid in enumerate(feats["tile_id"].tolist()):
            s = summaries.get(int(tid))
            if s is None:
                continue
            out.append({
                "box": gen["box"], "tag": gen["tag"], "source": gen["source"],
                "seed": int(gen["seed"]), "alpha": gen.get("alpha"),
                "tile_id": int(tid),
                "features": feats["features"][i].tolist(),
                "n_sub": [float(x) for x in s.n_sub],
                "n_host": [float(x) for x in s.n_host],
                "occ_numerator": [float(x) for x in s.occ_numerator],
                "volume_mpc3": float(s.volume_mpc3),
                "model_sha": gen["model_sha"], "lr_sha": gen["lr_sha"],
                "code_commit": gen["code_commit"],
            })

    table = direct_root("proxy_data", create=True) / "rows.jsonl"
    with open(table, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    by_source: Dict[str, int] = {}
    by_box: Dict[str, int] = {}
    for r in out:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_box[r["box"]] = by_box.get(r["box"], 0) + 1
    report = {
        "rows": len(out), "table": str(table),
        "candidates_with_features": len(have_feats),
        "candidates_with_labels": len(have_label),
        "rows_by_source": by_source, "rows_by_box": by_box,
        # Named explicitly because "HR vs SR2 only" is the failure mode this
        # dataset is designed to avoid, and a count is how you see it happening.
        "has_within_tile_variation": bool(
            sum(v for k, v in by_source.items()
                if k in ("frozen_seed", "intervention", "actor")) > 0),
    }
    write_json(direct_root("proxy_data") / "index_report.json", report)
    banner(json.dumps(report, indent=2))
    if not report["has_within_tile_variation"]:
        print(">>> WARNING: every row is 'hr' or 'frozen'. A proxy fitted on this "
              ">>> learns an easy binary distinction, not a local improvement "
              ">>> direction. Generate frozen_seed and intervention candidates.",
              flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--stage", required=True, choices=("generate", "label", "index"))
    ap.add_argument("--box", default="")
    ap.add_argument("--source", default="frozen", choices=SOURCES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=0,
                    help="the frozen reference seed for the feature difference")
    ap.add_argument("--alpha", type=float, default=None,
                    help="intervention strength (source=intervention)")
    ap.add_argument("--checkpoint", default="", help="actor checkpoint (source=actor)")
    ap.add_argument("--device", default="")
    ap.add_argument("--reuse", action="store_true", default=True)
    ap.add_argument("--overwrite", dest="reuse", action="store_false")
    ap.add_argument("--drop-field", action="store_true",
                    help="do not keep the 3.2 GB field (features only)")
    ap.add_argument("--keep-particles", action="store_true",
                    help="do NOT delete the ~7 GB ASCII table (debugging only)")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    if args.stage == "index":
        return stage_index(cfg, args)

    if not args.box:
        raise SystemExit("--box is required for --stage generate/label")
    assert_not_sealed(cfg, [args.box])
    if args.source == "intervention" and args.alpha is None:
        raise SystemExit("--alpha is required for --source intervention")

    if args.stage == "generate":
        dev = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        return stage_generate(cfg, args, dev)
    return stage_label(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
