#!/usr/bin/env python
"""Build the catalog-proxy training set: candidate fields, features, real labels.

Three stages, run as separate jobs because they want different hardware and,
more importantly, because they must be ordered:

``generate`` (GPU)
    Produce a candidate full-box field and, in the same pass while the field is
    still resident, the **per-tile features of both proxy arms**. Computing them
    here rather than at training time is the difference between the trainer
    reading a few MB per candidate and re-reading a 3.2 GB array per candidate;
    computing *both arms* here is what makes the arm comparison a comparison of
    feature vectors and nothing else -- they are derived from one pass over one
    field, so they cannot disagree about which tile they describe.
``label`` (CPU)
    Run Rockstar with ``FULL_PARTICLE_CHUNKS = 1`` on the assembled periodic
    box, stream the ~7 GB ``.particles`` table into exact fractional tile
    summaries through :func:`member_weights_from_particles`, check the additivity
    identity, and **delete the table in the same job**. Nothing bulky survives
    the allocation. This stage writes only inside its own candidate directory.
``index`` (CPU, seconds)
    Join features and labels into one training table, **once**, after every
    labelling job. Writes ``rows.jsonl`` and ``labels_complete.json`` atomically.

Why indexing is a separate, single job
--------------------------------------
It used to run at the end of every labelling job, on the theory that it is cheap
and idempotent. It is neither, concurrently: 120 labelling jobs finishing in
overlapping windows all rewrite one ``rows.jsonl`` in place, so a trainer that
starts on the heels of the last one can read a file another job is halfway
through truncating -- and, worse, a trainer ordered ``afterok`` on the *last
submitted* label job is not ordered after the *slowest* one, so it can fit on a
table missing whole boxes and report a validation number for a dataset that no
longer exists. One indexing job depending on all of them, plus a
``labels_complete.json`` the trainer refuses to start without, removes both.

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
    Targeted HR interventions (:mod:`cosmo_sr.reward.oracle_hr`) at several
    ``alpha`` and, on the predeclared diagnostic boxes, in three channel modes.
    ``mode`` is the part that makes arm B falsifiable: a ``vel`` intervention
    moves velocities and leaves displacement alone, so a density-only proxy
    cannot see it at all, and an arm-B win there is attributable rather than
    incidental.
``actor``
    Outputs of early actor checkpoints, appended by the DAgger loop. Not part of
    this milestone; the source is kept because the labelling path is identical.

Why not "HR versus SR2" alone
-----------------------------
A proxy trained on that pair learns an easy binary distinction -- SR2's high-k
density bias separates the classes perfectly -- and nothing about which local
change helps. It would score well on any aggregate metric and be useless as a
gradient. The ``intervention`` and ``frozen_seed`` sources exist precisely to
make the task be "which of these two versions of *this tile* is better".

    python scripts/reward/collect_catalog_proxy_data.py --stage generate \
        --box set0 --source intervention --alpha 0.5 --mode vel
    python scripts/reward/collect_catalog_proxy_data.py --stage label \
        --box set0 --source intervention --alpha 0.5 --mode vel
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
    PROJECT_ROOT, add_direct_args, array_sha, assert_not_sealed,
    banner, bins_of, candidate_matrix, candidate_tag, direct_root, file_sha,
    geometry_of, labels_complete_path, load_direct_config, load_hr, load_lr,
    manifest_row, model_path_of, phase_space_config_of, rockstar_provenance,
    soft_config_of, tile_grid_of, write_json, write_json_atomic,
)

from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.eval.rockstar import run_rockstar_on_particles  # noqa: E402
from cosmo_sr.reward.oracle_hr import CHANNEL_MODES  # noqa: E402
from cosmo_sr.reward.phase_space import ARMS, arm_features, arm_paired_feature_names  # noqa: E402
from cosmo_sr.reward.soft_structure import feature_names  # noqa: E402
from cosmo_sr.reward.tiles import (  # noqa: E402
    check_member_consistency, direct_full_box_stats, member_weights_from_particles,
    stream_member_ids, tile_summaries, write_tile_summaries,
)
from cosmo_sr.tts.sampling import super_resolve_srs_seeded  # noqa: E402
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402

SOURCES = ("hr", "frozen", "frozen_seed", "intervention", "actor")

#: The tile decomposition must reproduce the direct whole-box counts. These are
#: sums of float64 weights over 512 tiles and a few 1e4 objects, so the floating
#: point error is ~1e-10; anything above this is a real disagreement between the
#: member table and the catalog, not rounding.
IDENTITY_TOL = 1e-6


def candidate_dir(box: str, tag: str, *, create: bool = False) -> Path:
    return direct_root("candidates", f"{box}__{tag}", create=create)


def spec_tag(args) -> str:
    return candidate_tag(args.source, seed=int(args.seed), alpha=args.alpha,
                         mode=args.mode, checkpoint=args.checkpoint)


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def cached_frozen_field(box: str, seed: int) -> Optional[np.ndarray]:
    """The frozen candidate's field, if the labelled one is already on disk.

    Every use of a "frozen reference" in this file means *the* frozen candidate
    of that box and seed -- the one whose catalog the labels' ``dR`` is measured
    against. Regenerating it instead of loading it makes that an assumption
    about bit-reproducible seeded sampling across nodes and GPU models, and if
    the assumption ever fails the failure is silent: an intervention's alpha=0
    anchor stops being the box its reward is compared to. Loading is also a
    minute of a6000 time saved per candidate.
    """
    p = candidate_dir(box, candidate_tag("frozen", seed=int(seed))) / "field.npy"
    return np.load(p, mmap_mode="r") if p.is_file() else None


def generate_field(cfg, box: str, source: str, seed: int, *, alpha: float,
                   mode: str, checkpoint: str, device) -> np.ndarray:
    geom = geometry_of(cfg)

    if source == "hr":
        return np.asarray(load_hr(cfg, box), dtype=np.float32)

    if source in ("frozen", "frozen_seed"):
        cached = cached_frozen_field(box, seed)
        if cached is not None:
            print(f"    reusing the labelled frozen field for {box} seed {seed}",
                  flush=True)
            return np.asarray(cached, dtype=np.float32)

    if source in ("frozen", "frozen_seed", "actor"):
        lr = np.asarray(load_lr(cfg, box), dtype=np.float32)
        path = Path(checkpoint) if (source == "actor" and checkpoint) \
            else model_path_of(cfg)
        g = load_controlled_generator(path, scale_factor=geom.scale_factor,
                                      device=device, eval_mode=True)
        return super_resolve_srs_seeded(
            g, lr, int(seed), scale_factor=geom.scale_factor,
            nsplit=geom.nsplit, pad=geom.pad, device=device,
            noise_mode="per_tile", progress=True)

    if source == "intervention":
        # The existing Experiment-1 machinery, regenerated on the proxy boxes.
        # Reused rather than reimplemented: the interventions are only useful as
        # ranking signal if they are the SAME operation the oracle study measured.
        from cosmo_sr.reward.oracle_hr import apply_intervention

        # generate_field reuses the labelled frozen field when it is on disk,
        # which by the submitter's ordering it always is: interventions are
        # submitted behind their box's mask job, which is behind its HR label.
        base = generate_field(cfg, box, "frozen", seed, alpha=0.0, mode=mode,
                              checkpoint="", device=device)
        hr = np.asarray(load_hr(cfg, box), dtype=np.float32)
        mask_npz = direct_root("intervention_masks") / f"{box}.npz"
        if not mask_npz.is_file():
            raise SystemExit(
                f"no intervention masks for {box} at {mask_npz}; produce them "
                "with scripts/reward/build_intervention_masks.py, which needs "
                "the HR candidate of this box to have been LABELLED first (that "
                "is the job that extracts the Lagrangian member ids)"
            )
        z = np.load(mask_npz)
        return apply_intervention(base, hr, z["mask"], float(alpha), mode)

    raise SystemExit(f"unknown source {source!r}; expected one of {list(SOURCES)}")


def tile_features(cfg, field: np.ndarray, frozen: np.ndarray,
                  device) -> Dict[str, np.ndarray]:
    """``{arm: (n_tiles, 2F)}`` paired features, one row per tile, in one pass.

    Both arms come out of a single evaluation of the *larger* arm's base block,
    which arm A is a prefix of. That is not an optimisation: it is what makes
    "the arms share their density coordinates exactly" a fact about the code
    rather than a claim about two code paths that look similar. Batched over
    tiles because the CIC deposit is the cost, and 512 separate launches of a
    64^3 deposit is dominated by launch overhead.
    """
    grid = tile_grid_of(cfg)
    scfg, pcfg = soft_config_of(cfg), phase_space_config_of(cfg)
    n_dens = len(feature_names(scfg))
    cand_rows: List[np.ndarray] = []
    base_rows: List[np.ndarray] = []
    batch = 8
    ids = list(range(grid.n_tiles))
    for i in range(0, len(ids), batch):
        sl = [grid.slices(t) for t in ids[i:i + batch]]
        cand = np.stack([np.asarray(field[:, s[0], s[1], s[2]]) for s in sl])
        base = np.stack([np.asarray(frozen[:, s[0], s[1], s[2]]) for s in sl])
        c = torch.from_numpy(np.ascontiguousarray(cand, dtype=np.float32)).to(device)
        b = torch.from_numpy(np.ascontiguousarray(base, dtype=np.float32)).to(device)
        with torch.no_grad():
            cand_rows.append(arm_features(c, "b", scfg, pcfg).cpu().numpy())
            base_rows.append(arm_features(b, "b", scfg, pcfg).cpu().numpy())
        del c, b
    cand_f = np.concatenate(cand_rows, axis=0)
    base_f = np.concatenate(base_rows, axis=0)

    widths = {"a": n_dens, "b": cand_f.shape[1]}
    return {arm: np.concatenate(
        [cand_f[:, :w], cand_f[:, :w] - base_f[:, :w]], axis=1).astype(np.float32)
        for arm, w in widths.items()}


def _generate_inputs(cfg, args) -> Dict:
    """The provenance of everything that goes *into* this candidate.

    Computed before any work so that ``--reuse`` can compare it against what is
    on disk: a candidate whose LR, weights, Rockstar build or intervention mask
    have changed is not the candidate in the directory, however identical the
    file names are.
    """
    box, source = args.box, args.source
    model_p = Path(args.checkpoint) if (source == "actor" and args.checkpoint) \
        else model_path_of(cfg)
    mask = direct_root("intervention_masks") / f"{box}.npz"
    row = {
        "box": box, "source": source, "tag": spec_tag(args),
        "seed": int(args.seed),
        "alpha": (None if args.alpha is None else float(args.alpha)),
        "mode": str(args.mode),
        "lr_sha": array_sha(np.asarray(load_lr(cfg, box), dtype=np.float32)),
        "hr_sha": array_sha(np.asarray(load_hr(cfg, box), dtype=np.float32)),
        "model_sha": ("hr_field" if source == "hr" else file_sha(model_p)),
        "model_path": ("" if source == "hr" else str(model_p)),
        "base_seed": int(args.base_seed),
        "mask_sha": (file_sha(mask) if (source == "intervention" and mask.is_file())
                     else ""),
        "phase_space": {
            "vel_norm_km_s": float(phase_space_config_of(cfg).vel_norm_km_s),
        },
    }
    row.update(rockstar_provenance(cfg))
    return row


def stage_generate(cfg, args, device) -> int:
    box, source, seed = args.box, args.source, int(args.seed)
    tag = spec_tag(args)
    work = candidate_dir(box, tag, create=True)
    field_npy = work / "field.npy"
    feats_npz = work / "features.npz"
    man_path = work / "manifest.json"

    inputs = _generate_inputs(cfg, args)
    if args.reuse and man_path.is_file() and feats_npz.is_file() \
            and (field_npy.is_file() or args.drop_field):
        old = json.loads(man_path.read_text())
        # `phase_space` is compared like everything else: a changed velocity
        # normalisation changes arm B's features, so the candidate on disk is
        # not the candidate this call would produce, however identical the
        # field would be.
        stale = [k for k, v in inputs.items() if old.get(k) != v]
        if not stale:
            banner(f"{box}/{tag}: already generated and input-identical -> {work}")
            return 0
        print(f">>> regenerating {box}/{tag}: inputs changed since the last run "
              f"({', '.join(stale)})", flush=True)

    t0 = time.time()
    banner(f"{box}/{tag}: generating")
    field = generate_field(cfg, box, source, seed, alpha=args.alpha,
                           mode=args.mode, checkpoint=args.checkpoint,
                           device=device)

    # The frozen reference at the SAME seed. The difference block of the feature
    # vector is the part the actor can move; against a different seed it would be
    # measuring the noise draw instead.
    if source == "frozen":
        frozen = field
    else:
        frozen = generate_field(
            cfg, box, "frozen",
            seed if source == "frozen_seed" else int(args.base_seed),
            alpha=0.0, mode=args.mode, checkpoint="", device=device)

    feats = tile_features(cfg, field, frozen, device)
    scfg, pcfg = soft_config_of(cfg), phase_space_config_of(cfg)
    payload = {"tile_id": np.arange(feats["a"].shape[0], dtype=np.int64)}
    for arm in ARMS:
        payload[f"features_{arm}"] = feats[arm]
        payload[f"feature_names_{arm}"] = np.array(
            arm_paired_feature_names(arm, scfg, pcfg))
    np.savez_compressed(feats_npz, **payload)

    if not args.drop_field:
        tmp = field_npy.with_suffix(".tmp.npy")
        np.save(tmp, field.astype(np.float32))
        tmp.replace(field_npy)

    row = manifest_row(
        field_path=str(field_npy if not args.drop_field else ""),
        features_path=str(feats_npz),
        field_sha=(file_sha(field_npy) if not args.drop_field else ""),
        n_tiles=int(feats["a"].shape[0]),
        n_features={arm: int(feats[arm].shape[1]) for arm in ARMS},
        seconds=round(time.time() - t0, 1),
        **inputs)
    write_json_atomic(man_path, row)
    print(f"  features " + ", ".join(f"{a}:{feats[a].shape}" for a in ARMS)
          + f" -> {feats_npz}", flush=True)
    print(f"  {time.time() - t0:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# label
# --------------------------------------------------------------------------- #
def _extract_hr_members(cfg, box: str, table: Path, work: Path) -> Optional[Path]:
    """Lagrangian member ids of the intervention targets, in the same pass.

    Only for the HR candidate, and only when target selection has already asked
    for specific halos. This is what lets the interventions exist at all on the
    ten boxes that have no member extraction of their own: the HR labelling job
    is already streaming the 7 GB table, so pulling two dozen halos' ids out of
    it costs one more pass instead of one more full Rockstar run.
    """
    ids_json = direct_root("intervention_targets") / f"halo_ids_{box}.json"
    if not ids_json.is_file():
        print(f"  (no {ids_json.name}; skipping member extraction. Run "
              "build_intervention_masks.py --stage targets first if this box "
              "needs interventions.)", flush=True)
        return None
    wanted = [int(i) for i in json.loads(ids_json.read_text())["halo_ids"]]
    if not wanted:
        return None
    rk = dict(cfg.get("rockstar", {}))
    members = stream_member_ids(table, wanted,
                                chunk_rows=int(rk.get("chunk_rows", 8_000_000)))
    ids = np.asarray(sorted(members), dtype=np.int64)
    offs = np.zeros(ids.size + 1, dtype=np.int64)
    parts = []
    for k, h in enumerate(ids):
        p = np.asarray(members[int(h)], dtype=np.int64)
        parts.append(p)
        offs[k + 1] = offs[k] + p.size
    out = work / "hr_members.npz"
    np.savez_compressed(
        out, halo_id=ids, offset=offs,
        particle_id=(np.concatenate(parts) if parts else np.zeros(0, np.int64)))
    empty = [int(h) for h, p in zip(ids, parts) if p.size == 0]
    print(f"  member ids for {ids.size} halos -> {out.name}"
          + (f" ({len(empty)} empty: {empty[:5]})" if empty else ""), flush=True)
    return out


def stage_label(cfg, args) -> int:
    box, source = args.box, args.source
    tag = spec_tag(args)
    work = candidate_dir(box, tag, create=True)
    field_npy = work / "field.npy"
    jsonl = work / "tile_summaries.jsonl"
    npz = work / "tile_weights.npz"
    gen_manifest = work / "manifest.json"
    report_path = work / "label_report.json"

    if not gen_manifest.is_file():
        print(f">>> MISSING INPUT: {gen_manifest}")
        print(">>> produced by: --stage generate")
        return 0
    gen = json.loads(gen_manifest.read_text())
    if not field_npy.is_file():
        print(f">>> MISSING INPUT: {field_npy}")
        print(">>> produced by: --stage generate (without --drop-field)")
        return 0

    if args.reuse and report_path.is_file() and jsonl.is_file() and npz.is_file():
        old = json.loads(report_path.read_text())
        if old.get("field_sha") == gen.get("field_sha"):
            banner(f"{box}/{tag}: already labelled from this exact field -> {jsonl}")
            return 0
        print(">>> relabelling: the field on disk is not the one this label "
              ">>> describes (field_sha changed since --stage generate).", flush=True)

    # The catalog must describe the field the features were computed from. A
    # regenerated field with a stale label is the one failure mode that produces
    # a perfectly well-formed, entirely meaningless training row.
    on_disk = file_sha(field_npy)
    if gen.get("field_sha") and on_disk != gen["field_sha"]:
        print(f">>> GATE FAILED: {field_npy} hashes {on_disk}, but the manifest "
              f">>> records {gen['field_sha']}. Re-run --stage generate.")
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

    members_path = (_extract_hr_members(cfg, box, tables[0], work)
                    if source == "hr" else None)

    summaries = tile_summaries(cat, weights, bins, grid, box=box, source=tag)
    write_tile_summaries(jsonl, [summaries[t] for t in sorted(summaries)])

    # Summing tiles must reproduce the direct full-box statistics. Checked here,
    # per candidate, because it is the identity every label downstream rests on --
    # and a candidate that fails it is INVALID, not merely noted: its tile rows
    # would be a decomposition of something other than its own catalog.
    direct = direct_full_box_stats(cat, bins)
    pooled = {
        "n_host": np.sum([summaries[t].n_host for t in summaries], axis=0),
        "n_sub": np.sum([summaries[t].n_sub for t in summaries], axis=0),
        "occ_numerator": np.sum([summaries[t].occ_numerator for t in summaries], axis=0),
    }
    residual = {k: float(np.max(np.abs(pooled[k] - direct[k]))) for k in pooled}
    identity_ok = all(v <= IDENTITY_TOL for v in residual.values())
    label_ok = bool(consistency["ok"] and identity_ok)

    if bool(rk.get("delete_particles", True)) and not args.keep_particles:
        # The ASCII member table AND the GADGET2 snapshot Rockstar was fed. Both
        # are regenerable from field.npy in minutes and neither is evidence; at
        # ~7 GB and ~3.5 GB per candidate they are 1.2 TB across the matrix if
        # they are allowed to accumulate. The snapshot is easy to forget because
        # it is written by run_rockstar_on_field rather than by this script --
        # oracle_intervene.py and splice_verify.py each delete it separately.
        doomed = list(tables) + sorted(Path(halo_dir).glob("*.gadget2"))
        for p in doomed:
            size_gb = p.stat().st_size / 1e9
            p.unlink()
            print(f"  deleted {p.name} ({size_gb:.1f} GB) in the job that made it",
                  flush=True)

    full_box = {k: [float(x) for x in direct[k]]
                for k in ("n_sub", "n_host", "occ_numerator", "occupation")}
    write_json_atomic(report_path, {
        "box": box, "tag": tag, "source": source,
        "seed": int(args.seed), "alpha": gen.get("alpha"), "mode": gen.get("mode"),
        "n_objects": int(cat.n),
        "field_sha": gen.get("field_sha", ""),
        "catalog_path": str(getattr(cat, "path", "")),
        "tile_summaries_path": str(jsonl),
        "tile_weights_path": str(npz),
        "hr_members_path": (str(members_path) if members_path else ""),
        "member_consistency": consistency,
        "tile_sum_vs_direct_max_abs_residual": residual,
        "identity_tolerance": IDENTITY_TOL,
        "additivity_ok": bool(identity_ok),
        "member_consistency_ok": bool(consistency["ok"]),
        "label_ok": label_ok,
        "full_box": full_box,
        "seconds": round(time.time() - t0, 1),
        **rockstar_provenance(cfg),
    })
    print(f"  {jsonl}  ({time.time() - t0:.0f}s)", flush=True)
    if not label_ok:
        print(">>> LABEL INVALID: this candidate is excluded from the table.")
        print(f">>>   member consistency ok: {consistency['ok']}")
        print(f">>>   additivity residual:   {residual} (tol {IDENTITY_TOL})")
    return 0


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
def _scan_candidates() -> List[Dict]:
    """Every candidate directory that has a generate manifest, with its label.

    Reads per-candidate files rather than one shared append-only manifest: 120
    concurrent jobs appending to one JSONL is a race even when each line is
    small, and a torn line there is indistinguishable from a candidate that was
    never generated.
    """
    root = direct_root("candidates")
    out: List[Dict] = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        man = d / "manifest.json"
        if not man.is_file():
            continue
        try:
            gen = json.loads(man.read_text())
        except json.JSONDecodeError:
            print(f">>> unreadable manifest, skipping: {man}", flush=True)
            continue
        rep = d / "label_report.json"
        lab = None
        if rep.is_file():
            try:
                lab = json.loads(rep.read_text())
            except json.JSONDecodeError:
                print(f">>> unreadable label report, skipping: {rep}", flush=True)
        out.append({"dir": d, "generate": gen, "label": lab})
    return out


def stage_index(cfg, args) -> int:
    from cosmo_sr.reward.tiles import read_tile_summaries

    expected = {(c["box"], c["tag"]): c for c in candidate_matrix(cfg)}
    found = _scan_candidates()
    rows: List[Dict] = []
    labelled, invalid, unlabelled = [], [], []

    for entry in found:
        gen, lab = entry["generate"], entry["label"]
        key = (gen["box"], gen["tag"])
        if lab is None:
            unlabelled.append(key)
            continue
        if not lab.get("label_ok", False):
            invalid.append({"key": list(key),
                            "member_consistency_ok": lab.get("member_consistency_ok"),
                            "additivity_ok": lab.get("additivity_ok")})
            continue
        if lab.get("field_sha") and lab["field_sha"] != gen.get("field_sha"):
            invalid.append({"key": list(key), "reason": "label describes a stale field"})
            continue
        labelled.append(key)

        feats = np.load(gen["features_path"])
        summaries = {int(s.tile_id): s
                     for s in read_tile_summaries(lab["tile_summaries_path"])}
        blocks = {arm: feats[f"features_{arm}"] for arm in ARMS}
        for i, tid in enumerate(feats["tile_id"].tolist()):
            s = summaries.get(int(tid))
            if s is None:
                continue
            row = {
                "box": gen["box"], "tag": gen["tag"], "source": gen["source"],
                "seed": int(gen["seed"]), "alpha": gen.get("alpha"),
                "mode": gen.get("mode", "both"), "tile_id": int(tid),
                "n_sub": [float(x) for x in s.n_sub],
                "n_host": [float(x) for x in s.n_host],
                "occ_numerator": [float(x) for x in s.occ_numerator],
                "volume_mpc3": float(s.volume_mpc3),
                "model_sha": gen["model_sha"], "lr_sha": gen["lr_sha"],
                "field_sha": gen.get("field_sha", ""),
                "code_commit": gen["code_commit"],
            }
            for arm in ARMS:
                row[f"features_{arm}"] = blocks[arm][i].tolist()
            rows.append(row)

    table = direct_root("proxy_data", create=True) / "rows.jsonl"
    tmp = table.with_name(table.name + ".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(table)

    by_source: Dict[str, int] = {}
    by_box: Dict[str, int] = {}
    by_mode: Dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_box[r["box"]] = by_box.get(r["box"], 0) + 1
        if r["source"] == "intervention":
            by_mode[r["mode"]] = by_mode.get(r["mode"], 0) + 1

    done = set(labelled)
    missing = sorted(f"{b}/{t}" for (b, t) in expected if (b, t) not in done)
    complete = not missing
    report = {
        "rows": len(rows), "table": str(table),
        "candidates_expected": len(expected),
        "candidates_generated": len(found),
        "candidates_labelled_ok": len(labelled),
        "candidates_unlabelled": [f"{b}/{t}" for b, t in sorted(unlabelled)],
        "candidates_invalid": invalid,
        "missing_from_predeclared_matrix": missing,
        "complete": bool(complete),
        "rows_by_source": by_source, "rows_by_box": by_box,
        "intervention_rows_by_mode": by_mode,
        "n_arms": {arm: (len(rows[0][f"features_{arm}"]) if rows else 0)
                   for arm in ARMS},
        # Named explicitly because "HR vs SR2 only" is the failure mode this
        # dataset is designed to avoid, and a count is how you see it happening.
        "has_within_tile_variation": bool(
            sum(v for k, v in by_source.items()
                if k in ("frozen_seed", "intervention", "actor")) > 0),
        "code_commit": manifest_row(box="index", source="index", field_path="x",
                                    model_sha="x", lr_sha="x",
                                    seed=0)["code_commit"],
    }
    write_json_atomic(direct_root("proxy_data") / "index_report.json", report)
    banner(json.dumps({k: v for k, v in report.items()
                       if k not in ("candidates_unlabelled",
                                    "missing_from_predeclared_matrix")}, indent=2))

    marker = labels_complete_path()
    if complete and rows:
        write_json_atomic(marker, {
            "complete": True, "rows": len(rows),
            "candidates": sorted(f"{b}/{t}" for b, t in expected),
            "table": str(table), "index_report": str(
                direct_root("proxy_data") / "index_report.json"),
        })
        banner(f"labels COMPLETE -> {marker}")
    else:
        marker.unlink(missing_ok=True)
        print(f">>> labels INCOMPLETE: {len(missing)} of {len(expected)} "
              f"predeclared candidates are not labelled ok.")
        for m in missing[:20]:
            print(f">>>   {m}")
        if len(missing) > 20:
            print(f">>>   ... and {len(missing) - 20} more")
        print(">>> No labels_complete.json is written, so no proxy trainer will")
        print(">>> start on this partial table. That is the intended behaviour.")
    if not report["has_within_tile_variation"]:
        print(">>> WARNING: every row is 'hr' or 'frozen'. A proxy fitted on this")
        print(">>> learns an easy binary distinction, not a local improvement")
        print(">>> direction. Generate frozen_seed and intervention candidates.",
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
    ap.add_argument("--mode", default="both", choices=tuple(sorted(CHANNEL_MODES)),
                    help="intervention channels: disp, vel or both")
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
    if args.source != "intervention" and args.mode != "both":
        raise SystemExit(
            f"--mode {args.mode} only means something for --source intervention; "
            "for every other source the whole field is what it is")

    if args.stage == "generate":
        dev = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        return stage_generate(cfg, args, dev)
    return stage_label(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
