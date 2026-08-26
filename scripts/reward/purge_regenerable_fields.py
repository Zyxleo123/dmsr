#!/usr/bin/env python
"""Reclaim quota by deleting 3.2 GB fields that can be rebuilt. Dry run by default.

The per-user quota on ``/zfsauton/scratch`` is the pipeline's recurring failure
mode, and it fails in the least readable way there is: a job does all of its
work, dies on the final small write, and cannot even write its own traceback to
the log, because the log is on the same quota. On 2026-08-25 that took out three
gate jobs whose 3.2 GB fields were already complete on disk.

Two groups qualify, and both are REBUILDABLE, which is the only reason they are
here:

``spliced``
    ``flow_rockstar/fields/<box>__<tag>__seed<n>.npy`` -- an assembled candidate
    box. It is deleted only when its Rockstar catalog already exists (the
    measurement is banked) AND the ``tiles.npz`` it was spliced from is still on
    disk (~2 min of CPU to rebuild via ``splice_gather_field.py``).

``candidate``
    ``sr2_direct/candidates/<box>__frozen_seed<n>/field.npy`` -- a frozen
    generator forward. It is deleted only when its ``manifest.json`` records the
    model, the inputs and a ``field_sha``, so the rebuild is deterministic and
    checkable. Everything DERIVED from it -- features.npz, the Rockstar catalog,
    the labels, the tile summaries -- is kept; that is what the proxy line reads.

Nothing is deleted without ``--apply``.

    python scripts/reward/purge_regenerable_fields.py                # dry run
    python scripts/reward/purge_regenerable_fields.py --group spliced --apply
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.reward import paths  # noqa: E402

GB = 1024 ** 3


def _tiles_npz_index(root: Path) -> set:
    """Every ``tiles.npz`` on disk, as a set of run-dir names -> presence."""
    return {p.parent for p in root.glob("*_gather/**/tiles.npz")}


def spliced_candidates(root: Path, keep_tags: List[str]) -> List[Tuple[Path, str]]:
    """Spliced fields whose catalog is banked and whose tiles.npz survives."""
    out = []
    have_tiles = bool(list(root.glob("*_gather/**/tiles.npz")))
    for f in sorted((root / "flow_rockstar" / "fields").glob("*.npy")):
        stem = f.stem                      # <box>__<tag>__seed<n>
        box, rest = stem.split("__", 1)
        tag = rest.rsplit("__seed", 1)[0]
        if tag in keep_tags:
            continue
        cat = glob.glob(str(root / "flow_rockstar" / "halos"
                            / f"{box}__candidate__{tag}" / "*" / "halos_*.ascii"))
        if not cat:
            continue                       # nothing banked -- this field is live
        if not have_tiles:
            continue
        out.append((f, f"catalog banked ({box}/{tag}), rebuild from tiles.npz"))
    return out


def candidate_fields(root: Path) -> List[Tuple[Path, str]]:
    """Frozen generator forwards whose manifest makes the rebuild checkable."""
    out = []
    for d in sorted((root / "sr2_direct" / "candidates").glob("*")):
        f = d / "field.npy"
        if not f.is_file() or f.is_symlink():
            continue
        man = d / "manifest.json"
        if not man.is_file():
            continue
        m = json.loads(man.read_text())
        if not (m.get("field_sha") and m.get("model_sha")
                and m.get("seed") is not None and m.get("box")):
            continue                       # not reproducible -- leave it alone
        out.append((f, f"sha {m['field_sha']}, {m['box']} seed {m['seed']} "
                       f"({m.get('source', '?')})"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="all",
                    choices=("all", "spliced", "candidate"))
    ap.add_argument("--keep-tag", action="append", default=[],
                    help="a spliced tag to spare, repeatable. Use it for the "
                         "fields a queued job is about to read.")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete. Without it nothing is touched.")
    a = ap.parse_args(argv)

    root = paths.reward_root()
    victims: List[Tuple[Path, str]] = []
    if a.group in ("all", "spliced"):
        victims += spliced_candidates(root, a.keep_tag)
    if a.group in ("all", "candidate"):
        victims += candidate_fields(root)

    if not victims:
        print("nothing qualifies -- no field is both spent and rebuildable.")
        return 0

    total = 0
    print(f"{'size':>8}  path / why")
    for f, why in victims:
        sz = f.stat().st_size
        total += sz
        print(f"{sz / GB:7.1f}G  {f.relative_to(root)}\n"
              f"{'':>9} {why}")
    print(f"\n{len(victims)} files, {total / GB:.1f} GB")

    if not a.apply:
        print("\nDRY RUN -- nothing deleted. Re-run with --apply.")
        return 0

    freed = 0
    for f, _ in victims:
        sz = f.stat().st_size
        f.unlink()
        freed += sz
    print(f"\ndeleted {len(victims)} files, freed {freed / GB:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
