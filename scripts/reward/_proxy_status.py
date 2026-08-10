#!/usr/bin/env python
"""What does each predeclared candidate still need? Torch-free, for the submitter.

``submit_proxy_labels.sh`` runs this on the LOGIN node to decide which jobs to
submit, so it must import nothing heavy: no torch, no ``_sr2_direct`` (which
pulls torch through :mod:`cosmo_sr.reward.phase_space`). The schema version
comes from :mod:`cosmo_sr.reward.arms`, which exists precisely so this check
does not need a GPU stack to read a list of strings; the candidate matrix comes
from :mod:`_proxy_matrix`, the same function the indexer's completeness check
calls, so the submitter cannot disagree with it about what the dataset is.

The verdict per candidate is an ACTION, not a status, because the submitter's
question is "which sbatch lines do I emit":

``skip``
    Labelled ok and features at the current schema. Nothing to do.
``generate_only``
    Labelled ok, but the features predate the current schema
    (:data:`cosmo_sr.reward.arms.FEATURE_SCHEMA_VERSION`). The generate stage
    will repair ``features.npz`` from the saved ``field.npy`` -- no generator
    forward pass if the frozen reference is on disk, no Rockstar re-run, label
    untouched.
``label_only``
    Generated at the current schema but not labelled. Only the CPU Rockstar
    job is needed.
``generate_label``
    Not generated (or generated under an older schema AND unlabelled). The
    full pair.
``invalid``
    A label exists and says ``label_ok: false``. NOT auto-resubmitted: the
    label stage's reuse check would skip it, and the recorded failure means a
    human should look before burning another 12-hour Rockstar run. Re-run that
    candidate with ``OVERWRITE=1`` once understood.

    python scripts/reward/_proxy_status.py <config.yaml> <sr2_direct_root> [box ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

from _proxy_matrix import candidate_matrix

from cosmo_sr.reward.arms import FEATURE_SCHEMA_VERSION

__all__ = ["candidate_action"]


def _read_json(p: Path) -> Optional[Dict]:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def candidate_action(cand_dir: Path) -> str:
    """One of skip / generate_only / label_only / generate_label / invalid."""
    man = _read_json(cand_dir / "manifest.json")
    generated = (man is not None and (cand_dir / "features.npz").is_file())
    schema_ok = bool(generated and
                     int(man.get("feature_schema_version", 1)) == FEATURE_SCHEMA_VERSION)
    lab = _read_json(cand_dir / "label_report.json")
    labelled_ok = bool(
        lab is not None and lab.get("label_ok", False)
        and (not lab.get("field_sha") or man is None
             or lab.get("field_sha") == man.get("field_sha")))

    if lab is not None and not lab.get("label_ok", True):
        return "invalid"
    if not generated:
        return "generate_label"
    if labelled_ok:
        return "skip" if schema_ok else "generate_only"
    if not schema_ok:
        return "generate_label"
    # Generated at the current schema, unlabelled. The label stage needs the
    # field itself, which --drop-field candidates do not keep.
    return "label_only" if (cand_dir / "field.npy").is_file() else "generate_label"


def main(argv=None) -> int:
    import yaml

    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 2
    cfg = yaml.safe_load(open(argv[0]))
    root = Path(argv[1])
    boxes = [b for b in argv[2:] if b.strip()] or None
    for c in candidate_matrix(cfg, boxes=boxes):
        action = candidate_action(root / "candidates" / f"{c['box']}__{c['tag']}")
        print(c["box"], c["source"], c["seed"],
              "-" if c["alpha"] is None else f"{c['alpha']:.4f}",
              c["mode"], c["tag"], action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
