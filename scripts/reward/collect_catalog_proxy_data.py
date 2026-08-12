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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from _sr2_direct import (  # noqa: E402
    PROJECT_ROOT, add_direct_args, array_sha, assert_not_sealed,
    banner, bins_of, candidate_matrix, candidate_tag, direct_root, file_sha,
    geometry_of, labels_complete_path, load_direct_config, load_hr, load_lr,
    manifest_row, model_path_of, phase_space_config_of, rockstar_provenance,
    soft_config_of, soft_rockstar_config_of, tile_grid_of, write_json,
    write_json_atomic,
)

from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.eval.rockstar import run_rockstar_on_particles  # noqa: E402
from cosmo_sr.reward.arms import (  # noqa: E402
    ARMS, FEATURE_SCHEMA_VERSION, FIELD_CHANGED_KEY, features_key,
    owned_sidecar_arms, sidecar_file,
)
from cosmo_sr.reward.oracle_hr import CHANNEL_MODES  # noqa: E402
from cosmo_sr.reward.phase_space import (  # noqa: E402
    FLAT_ARMS, arm_features, arm_paired_feature_names, phase_space_grid_channel_names,
    phase_space_paired_grid,
)
from cosmo_sr.reward.soft_rockstar import (  # noqa: E402
    paired_token_feature_names, soft_rockstar_tokens,
)

#: A tile counts as changed from its frozen reference when ANY of the six raw
#: displacement/velocity channels differs by more than this, anywhere in the
#: crop. The raw field is copied byte-for-byte outside an edit (an intervention
#: touches only its masked region; a frozen candidate IS its own reference), so a
#: touched tile differs by >> this while an untouched one differs by exactly 0 --
#: the threshold only has to separate "copied" from "modified", not measure a
#: magnitude. Read off all six channels so a velocity-only edit, invisible to a
#: density feature, is still detected.
FIELD_CHANGED_ABS_TOL = 1e-6
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
    """Per-tile features of the CACHED arms, one row per tile, in one pass.

    Arms ``a``/``b`` are ``(n_tiles, 2F)`` paired flat vectors; arm ``c`` is a
    ``(n_tiles, 2, T, F_tok)`` paired token grid (arm ``d`` reuses it); arm ``e``
    is a ``(n_tiles, 2, 5, 32, 32, 32)`` paired dense phase-space grid, cached as
    float16. Arm ``f`` has no cached block -- it streams raw fields at train time
    -- so it is absent here. Everything comes from ONE pass over one field: the
    flat arms from the larger arm's base block (arm A is a prefix), the tokens and
    the grid from the same CIC deposit
    (:func:`cosmo_sr.reward.phase_space.deposit_phase_space`). That is what makes
    "the arms share their density coordinates exactly" a fact about the code.

    Also returns the arm-neutral ``field_changed`` flag per tile, read off the
    raw six-channel candidate-versus-frozen crop (see ``FIELD_CHANGED_ABS_TOL``)
    so it catches a velocity-only edit a density feature would miss.
    """
    grid = tile_grid_of(cfg)
    scfg, pcfg = soft_config_of(cfg), phase_space_config_of(cfg)
    rcfg = soft_rockstar_config_of(cfg)
    n_dens = len(feature_names(scfg))
    cand_rows: List[np.ndarray] = []
    base_rows: List[np.ndarray] = []
    cand_tok: List[np.ndarray] = []
    base_tok: List[np.ndarray] = []
    grid_blocks: List[np.ndarray] = []
    changed: List[np.ndarray] = []
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
            cand_tok.append(soft_rockstar_tokens(c, scfg, pcfg, rcfg).cpu().numpy())
            base_tok.append(soft_rockstar_tokens(b, scfg, pcfg, rcfg).cpu().numpy())
            # Arm E: the paired dense grid, cached as float16 (train/eval upcast).
            grid_blocks.append(
                phase_space_paired_grid(c, b, scfg, pcfg).cpu().to(torch.float16).numpy())
            # Arm-neutral changed flag: max abs raw difference over all 6 channels.
            changed.append((c - b).abs().amax(dim=(1, 2, 3, 4)).cpu().numpy())
        del c, b
    cand_f = np.concatenate(cand_rows, axis=0)
    base_f = np.concatenate(base_rows, axis=0)
    ct = np.concatenate(cand_tok, axis=0)
    bt = np.concatenate(base_tok, axis=0)

    widths = {"a": n_dens, "b": cand_f.shape[1]}
    out: Dict[str, np.ndarray] = {arm: np.concatenate(
        [cand_f[:, :w], cand_f[:, :w] - base_f[:, :w]], axis=1).astype(np.float32)
        for arm, w in widths.items()}
    # Same slot convention as the flat arms: [candidate, candidate - frozen].
    out["c"] = np.stack([ct, ct - bt], axis=1).astype(np.float32)
    out["e"] = np.concatenate(grid_blocks, axis=0).astype(np.float16)
    out[FIELD_CHANGED_KEY] = (
        np.concatenate(changed, axis=0) > FIELD_CHANGED_ABS_TOL).astype(np.int8)
    return out


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


def _paired_grid_channel_names() -> np.ndarray:
    base = phase_space_grid_channel_names()
    return np.array(list(base) + [f"d_{n}" for n in base])


def _format_feat_shapes(feats: Dict[str, np.ndarray]) -> str:
    """Human-readable shapes for the logged feature dump.

    ``tile_features`` only materialises the *owned* blocks (``a``/``b``/``c``/``e``
    plus ``field_changed``). Arm ``d`` reuses ``c``'s tokens and arm ``f`` streams
    raw fields, so iterating ``ARMS`` and indexing ``feats[a]`` KeyErrors on the
    first features-only backfill -- which is exactly the schema-3 path.
    """
    parts: List[str] = []
    for arm in ARMS:
        if arm == "d":
            parts.append("d:reuses_c")
        elif arm == "f":
            parts.append("f:streamed")
        elif arm in feats:
            parts.append(f"{arm}:{tuple(feats[arm].shape)}")
        else:
            parts.append(f"{arm}:missing")
    if FIELD_CHANGED_KEY in feats:
        parts.append(f"{FIELD_CHANGED_KEY}:{tuple(feats[FIELD_CHANGED_KEY].shape)}")
    return ", ".join(parts)


def _write_features(cfg, work: Path, feats: Dict[str, np.ndarray]) -> Path:
    """``features.npz`` with every CACHED arm's blocks and names, written atomically.

    Only the arms that own a stored block are written: the flat arms ``a``/``b``,
    and the owned-sidecar arms ``c`` (tokens, shared with ``d``) and ``e`` (the
    dense grid). Arm ``f`` streams raw fields and has no block. The arm-neutral
    ``field_changed`` flag rides along as its own column.
    """
    scfg, pcfg = soft_config_of(cfg), phase_space_config_of(cfg)
    feats_npz = work / "features.npz"
    payload = {"tile_id": np.arange(feats["a"].shape[0], dtype=np.int64),
               "feature_schema_version": np.int64(FEATURE_SCHEMA_VERSION),
               FIELD_CHANGED_KEY: feats[FIELD_CHANGED_KEY]}
    for arm in FLAT_ARMS:
        payload[features_key(arm)] = feats[arm]
        payload[f"feature_names_{arm}"] = np.array(
            arm_paired_feature_names(arm, scfg, pcfg))
    for arm in owned_sidecar_arms():
        payload[features_key(arm)] = feats[arm]
        payload[f"feature_names_{arm}"] = (
            _paired_grid_channel_names() if arm == "e"
            else np.array(paired_token_feature_names(soft_rockstar_config_of(cfg))))
    tmp = feats_npz.with_name(feats_npz.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(feats_npz)
    return feats_npz


def _feature_manifest_keys(feats: Dict[str, np.ndarray]) -> Dict:
    def shape_of(arm: str):
        if arm == "f":
            return "streamed_20ch_no_cache"      # built from field.npy at train time
        block = feats["c"] if arm == "d" else feats.get(arm)  # d reuses c's tokens
        if block is None:
            return None
        return int(block.shape[1]) if block.ndim == 2 else list(block.shape[1:])

    return {
        "feature_schema_version": int(FEATURE_SCHEMA_VERSION),
        "n_features": {arm: shape_of(arm) for arm in ARMS},
        "n_tiles": int(feats["a"].shape[0]),
    }


def _backfill_features(cfg, args, work: Path, old: Dict, device) -> bool:
    """Recompute ``features.npz`` from the saved field. No GPU pass, no Rockstar.

    This is what :data:`FEATURE_SCHEMA_VERSION` buys: a new feature block
    (arm C's tokens) changes nothing about the field or its label, so a
    candidate whose inputs are identical but whose features predate the schema
    is repaired in place from ``field.npy`` -- ``field_sha`` is untouched and
    the existing label stays valid. Needs the frozen reference on disk for the
    difference block; returns False (fall through to a full regenerate) when
    either field is missing.
    """
    field_npy = work / "field.npy"
    if not field_npy.is_file():
        return False
    source, seed = args.source, int(args.seed)
    if source == "frozen":
        frozen = np.load(field_npy, mmap_mode="r")
    else:
        frozen = cached_frozen_field(
            args.box, seed if source == "frozen_seed" else int(args.base_seed))
        if frozen is None:
            return False
    t0 = time.time()
    banner(f"{args.box}/{spec_tag(args)}: features-only backfill "
           f"(schema {old.get('feature_schema_version', 1)} -> "
           f"{FEATURE_SCHEMA_VERSION}); the field and its label are untouched")
    field = np.load(field_npy, mmap_mode="r")
    feats = tile_features(cfg, field, frozen, device)
    _write_features(cfg, work, feats)
    row = dict(old)
    row.update(_feature_manifest_keys(feats))
    row["features_backfilled_seconds"] = round(time.time() - t0, 1)
    write_json_atomic(work / "manifest.json", row)
    print(f"  features {_format_feat_shapes(feats)}  ({time.time() - t0:.0f}s)",
          flush=True)
    return True


def stage_generate(cfg, args, device) -> int:
    box, source, seed = args.box, args.source, int(args.seed)
    tag = spec_tag(args)
    work = candidate_dir(box, tag, create=True)
    field_npy = work / "field.npy"
    man_path = work / "manifest.json"

    inputs = _generate_inputs(cfg, args)
    if args.reuse and man_path.is_file() and (work / "features.npz").is_file() \
            and (field_npy.is_file() or args.drop_field):
        old = json.loads(man_path.read_text())
        # `phase_space` is compared like everything else: a changed velocity
        # normalisation changes arm B's features, so the candidate on disk is
        # not the candidate this call would produce, however identical the
        # field would be.
        stale = [k for k, v in inputs.items() if old.get(k) != v]
        if not stale:
            if int(old.get("feature_schema_version", 1)) == FEATURE_SCHEMA_VERSION:
                banner(f"{box}/{tag}: already generated and input-identical -> {work}")
                return 0
            # Input-identical but the features predate the current schema:
            # repair the features from disk rather than redoing the field.
            if _backfill_features(cfg, args, work, old, device):
                return 0
            print(">>> features are schema-stale and the frozen reference is "
                  "not on disk; regenerating in full.", flush=True)
        else:
            print(f">>> regenerating {box}/{tag}: inputs changed since the last "
                  f"run ({', '.join(stale)})", flush=True)

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
    feats_npz = _write_features(cfg, work, feats)

    if not args.drop_field:
        tmp = field_npy.with_suffix(".tmp.npy")
        np.save(tmp, field.astype(np.float32))
        tmp.replace(field_npy)

    row = manifest_row(
        field_path=str(field_npy if not args.drop_field else ""),
        features_path=str(feats_npz),
        field_sha=(file_sha(field_npy) if not args.drop_field else ""),
        seconds=round(time.time() - t0, 1),
        **_feature_manifest_keys(feats),
        **inputs)
    write_json_atomic(man_path, row)
    print(f"  features {_format_feat_shapes(feats)} -> {feats_npz}", flush=True)
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
    # Reuse a previous Rockstar run ONLY if that run got as far as writing a
    # label report. Halo-finder output left behind by a job that died between
    # Rockstar and the summaries is not a cache -- measured here, a crashed
    # attempt left an ascii catalog of 209,757 objects next to a .particles
    # table missing 18,155 of them, and reusing the pair produced a tile
    # decomposition of a catalog that never existed. A complete attempt always
    # leaves label_report.json, so its absence is the exact signal that the
    # leftovers are debris.
    reuse_rockstar = bool(args.reuse) and report_path.is_file()
    if args.reuse and not reuse_rockstar and Path(halo_dir).exists():
        print(">>> discarding Rockstar output from an attempt that never "
              ">>> finished labelling; re-running the halo finder.", flush=True)
    cat = run_rockstar_on_particles(
        particles, halo_dir, tag=tag,
        binary=PROJECT_ROOT / rk.get("binary", "external/rockstar/rockstar"),
        cfg=PROJECT_ROOT / rk.get("config",
                                  "configs/sr2_baseline/rockstar_particles.cfg"),
        overwrite=not reuse_rockstar)
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

    # tile_summaries RAISES on objects with no member rows, which is right --
    # their weight is nowhere and the partition of unity is broken. But a raise
    # here is a traceback where a verdict belongs: the candidate is invalid, the
    # run should say so in the report the indexer reads, and the job should exit
    # 0 so dependents report the same instead of stranding.
    if not consistency["ok"]:
        write_json_atomic(report_path, {
            "box": box, "tag": tag, "source": source, "label_ok": False,
            "member_consistency_ok": False, "member_consistency": consistency,
            "field_sha": gen.get("field_sha", ""),
            "reason": (
                f"{consistency['n_missing']} of {consistency['n_catalog_objects']} "
                "catalog objects have no member-particle rows"),
            "seconds": round(time.time() - t0, 1),
        })
        print(f">>> INVALID CANDIDATE: {consistency['n_missing']} of "
              f"{consistency['n_catalog_objects']} catalog objects have no "
              ">>> member-particle rows, so their weight is nowhere and the tile")
        print(">>> sums cannot reproduce the box. The .particles table and the")
        print(">>> ascii catalog do not describe the same run.")
        print(">>> Recorded as label_ok=false; the indexer will exclude it and")
        print(">>> withhold labels_complete.json. Re-run this candidate with")
        print(">>> OVERWRITE=1 to redo the halo finding from scratch.")
        return 0

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


def _arm_report_shape(arm: str, rows: List[Dict], sidecars: Dict[str, Dict]):
    """The per-arm feature shape for the index report, robust to storage kind."""
    if not rows:
        return 0
    if arm in FLAT_ARMS:
        return len(rows[0][features_key(arm)])
    if arm == "f":
        return "streamed_20ch_no_cache"
    owner = "c" if arm in ("c", "d") else arm      # d reuses c's sidecar
    return sidecars.get(owner, {}).get("shape", [0])[1:]


def stage_index(cfg, args) -> int:
    from cosmo_sr.reward.tiles import read_tile_summaries

    from numpy.lib.format import open_memmap

    expected = {(c["box"], c["tag"]): c for c in candidate_matrix(cfg)}
    found = _scan_candidates()
    rows: List[Dict] = []
    # Arm C's tokens are small enough to hold and stack; arm E's dense grid is
    # ~32 GB across the table, so it is never held -- each row records a reference
    # (features file, tile index) and a second streaming pass writes it straight
    # into a memory-mapped float16 sidecar.
    token_blocks: List[np.ndarray] = []
    e_refs: List[Tuple[str, int]] = []
    labelled, invalid, unlabelled, stale_features = [], [], [], []
    leftover_particles: List[str] = []

    for entry in found:
        gen, lab = entry["generate"], entry["label"]
        key = (gen["box"], gen["tag"])
        # `.particles` / `.gadget2` debris means a label job died mid-flight or
        # forgot to clean up; at ~10 GB per candidate it must not accumulate,
        # so it blocks the completeness marker until someone looks.
        for p in list(Path(entry["dir"]).rglob("*.particles")) + \
                list(Path(entry["dir"]).rglob("*.gadget2")):
            leftover_particles.append(str(p))
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
        if int(gen.get("feature_schema_version", 1)) != FEATURE_SCHEMA_VERSION:
            # The label is fine; the FEATURES predate the current schema, so
            # this candidate's rows cannot join the table (they would leave
            # holes in the newer arms' blocks). --stage generate repairs it
            # from field.npy without re-running Rockstar.
            stale_features.append(key)
            continue
        feats = np.load(gen["features_path"])
        # Schema version alone is not enough: a hand-stamped manifest (or a
        # half-written features.npz) can claim schema 3 while omitting arm E's
        # grid / field_changed. Treat that as stale so we backfill rather than
        # KeyError mid-index after every candidate has already been accepted.
        required = [features_key(a) for a in (*FLAT_ARMS, "c", "e")] + [FIELD_CHANGED_KEY]
        missing_keys = [k for k in required if k not in feats.files]
        if missing_keys:
            stale_features.append(key)
            print(f">>> {key[0]}/{key[1]}: features.npz missing {missing_keys}; "
                  "treating as schema-stale", flush=True)
            continue
        labelled.append(key)

        summaries = {int(s.tile_id): s
                     for s in read_tile_summaries(lab["tile_summaries_path"])}
        # Only the small blocks are read here; arm E's grid is left in the file
        # and streamed in a second pass (see e_refs). Arm D reads arm C's tokens
        # and arm F streams raw fields, so neither has a block to gather.
        flat_blocks = {arm: feats[features_key(arm)] for arm in FLAT_ARMS}
        c_block = feats[features_key("c")]
        changed = np.asarray(feats[FIELD_CHANGED_KEY], dtype=np.int8)
        for i, tid in enumerate(feats["tile_id"].tolist()):
            s = summaries.get(int(tid))
            if s is None:
                continue
            row = {
                # The row's position in the full table, which is what joins it
                # to the sidecar arrays after any filtering.
                "row_id": len(rows),
                "box": gen["box"], "tag": gen["tag"], "source": gen["source"],
                "seed": int(gen["seed"]), "alpha": gen.get("alpha"),
                "mode": gen.get("mode", "both"), "tile_id": int(tid),
                # Arm-neutral changed-from-frozen flag, from the raw six-channel
                # field, so it catches velocity-only edits (see collect's
                # FIELD_CHANGED_ABS_TOL).
                FIELD_CHANGED_KEY: int(changed[i]),
                "n_sub": [float(x) for x in s.n_sub],
                "n_host": [float(x) for x in s.n_host],
                "occ_numerator": [float(x) for x in s.occ_numerator],
                "volume_mpc3": float(s.volume_mpc3),
                "model_sha": gen["model_sha"], "lr_sha": gen["lr_sha"],
                "field_sha": gen.get("field_sha", ""),
                "code_commit": gen["code_commit"],
            }
            for arm in FLAT_ARMS:
                row[features_key(arm)] = flat_blocks[arm][i].tolist()
            token_blocks.append(np.asarray(c_block[i], dtype=np.float32))
            e_refs.append((str(gen["features_path"]), int(i)))
            rows.append(row)

    table = direct_root("proxy_data", create=True) / "rows.jsonl"
    tmp = table.with_name(table.name + ".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(table)

    sidecars: Dict[str, Dict] = {}
    # Arm C: tokens, held in memory and stacked (float32, ~1.6 GB at full size).
    c_path = table.parent / sidecar_file("c")
    c_stack = (np.stack(token_blocks).astype(np.float32) if token_blocks
               else np.zeros((0, 2, 1, 1), dtype=np.float32))
    tmp_c = c_path.with_name(c_path.name + ".tmp.npy")
    np.save(tmp_c, c_stack)
    tmp_c.replace(c_path)
    sidecars["c"] = {"path": str(c_path), "shape": list(c_stack.shape),
                     "dtype": "float32"}
    del c_stack, token_blocks

    # Arm E: the dense grid, written straight into a memory-mapped float16 .npy,
    # one candidate's file loaded at a time. Never held in memory, never float64.
    e_path = table.parent / sidecar_file("e")
    tmp_e = e_path.with_name(e_path.name + ".tmp.npy")
    if e_refs:
        block_shape = np.load(e_refs[0][0])[features_key("e")].shape[1:]
        mm = open_memmap(tmp_e, mode="w+", dtype=np.float16,
                         shape=(len(e_refs), *block_shape))
        r = 0
        while r < len(e_refs):
            path = e_refs[r][0]
            arr = np.load(path)[features_key("e")]         # (n_tiles, 2, 5, D, D, D)
            j = r
            while j < len(e_refs) and e_refs[j][0] == path:
                mm[j] = arr[e_refs[j][1]]
                j += 1
            del arr
            r = j
        mm.flush()
        del mm
        e_shape = [len(e_refs), *block_shape]
    else:
        np.save(tmp_e, np.zeros((0, 2, 5, 1, 1, 1), dtype=np.float16))
        e_shape = [0, 2, 5, 1, 1, 1]
    tmp_e.replace(e_path)
    sidecars["e"] = {"path": str(e_path), "shape": [int(x) for x in e_shape],
                     "dtype": "float16"}

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
    # Leftover halo-finder debris blocks the marker too: it means a label job
    # died between Rockstar and the summaries (or cleanup failed), and at ~10 GB
    # per candidate "someone will clean it later" is how a TB disappears.
    complete = not missing and not leftover_particles
    report = {
        "rows": len(rows), "table": str(table),
        "feature_schema_version": int(FEATURE_SCHEMA_VERSION),
        "sidecars": sidecars,
        "candidates_expected": len(expected),
        "candidates_generated": len(found),
        "candidates_labelled_ok": len(labelled),
        "candidates_unlabelled": [f"{b}/{t}" for b, t in sorted(unlabelled)],
        "candidates_stale_features": [f"{b}/{t}" for b, t in sorted(stale_features)],
        "candidates_invalid": invalid,
        "leftover_particles": leftover_particles,
        "missing_from_predeclared_matrix": missing,
        "complete": bool(complete),
        "rows_by_source": by_source, "rows_by_box": by_box,
        "intervention_rows_by_mode": by_mode,
        "n_arms": {arm: _arm_report_shape(arm, rows, sidecars) for arm in ARMS},
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
                                    "missing_from_predeclared_matrix",
                                    "leftover_particles")}, indent=2))

    marker = labels_complete_path()
    if complete and rows:
        write_json_atomic(marker, {
            "complete": True, "rows": len(rows),
            "feature_schema_version": int(FEATURE_SCHEMA_VERSION),
            "sidecars": sidecars,
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
        if stale_features:
            print(f">>> {len(stale_features)} candidates hold valid labels but "
                  "schema-stale features; repair them with --stage generate "
                  "(features-only backfill, no Rockstar re-run).")
        if leftover_particles:
            print(f">>> {len(leftover_particles)} leftover .particles/.gadget2 "
                  "files block completeness; a relabel (OVERWRITE=1) or manual "
                  "cleanup removes them:")
            for p in leftover_particles[:10]:
                print(f">>>   {p}")
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
