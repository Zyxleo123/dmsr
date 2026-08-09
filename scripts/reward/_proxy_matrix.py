"""The predeclared candidate matrix, in ONE place and with no heavy imports.

Kept out of :mod:`_sr2_direct` -- which pulls in torch through the reward
package -- so the submitter can enumerate the dataset on a login node in
milliseconds. That matters for more than speed: the submitter, the indexer and
the completeness check must agree on what the dataset *is*, and the cheapest way
to guarantee that is for all three to call the same function. A submitter that
had to reimplement the enumeration because importing the real one was expensive
is a submitter that will disagree with it eventually.

:mod:`_sr2_direct` re-exports both names, so scripts import them from there as
usual.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

__all__ = ["candidate_matrix", "candidate_tag", "dataset_of"]


def dataset_of(cfg: Mapping) -> Dict:
    return dict(cfg.get("proxy_dataset", {}))


def candidate_tag(source: str, *, seed: int = 0, alpha: Optional[float] = None,
                  mode: str = "both", checkpoint: str = "") -> str:
    """The directory name of one candidate, and its identity everywhere else.

    The intervention tag carries ``mode`` because the diagnostic subset runs the
    same box, seed and alpha in three channel modes; without it they would
    collide on one directory and the last one written would silently become all
    three.
    """
    s = str(source)
    if s == "hr":
        return "hr"
    if s == "intervention":
        if alpha is None:
            raise ValueError("an intervention candidate needs an alpha")
        return f"intervention_{mode}_a{float(alpha):.2f}_seed{int(seed)}"
    if s == "actor":
        return f"actor_{Path(checkpoint).stem or 'ck'}_seed{int(seed)}"
    if s in ("frozen", "frozen_seed"):
        return f"frozen_seed{int(seed)}"
    raise ValueError(f"unknown candidate source {source!r}")


def candidate_matrix(cfg: Mapping, *,
                     boxes: Optional[Sequence[str]] = None) -> List[Dict]:
    """The full predeclared list of candidates, one dict per full-box Rockstar run.

    Phase 2 of the plan: per box, one HR anchor, four frozen seeds and the alpha
    ladder; plus, on the predeclared diagnostic boxes, the same ladder in the
    other channel modes. Each entry is ``{box, source, seed, alpha, mode, tag}``.
    """
    d = dataset_of(cfg)
    box_list = [str(b) for b in (boxes if boxes is not None else d.get("boxes", []))]
    seeds = [int(s) for s in d.get("frozen_seeds", [0, 1, 2, 3])]
    alphas = [float(a) for a in d.get("alphas", [0.25, 0.5, 1.0])]
    core_mode = str(d.get("core_intervention_mode", "both"))
    diag_boxes = {str(b) for b in d.get("diagnostic_boxes", [])}
    diag_modes = [str(m) for m in d.get("diagnostic_modes", [])]
    base_seed = seeds[0] if seeds else 0

    out: List[Dict] = []
    for box in box_list:
        def add(source, seed=0, alpha=None, mode="both"):
            out.append({
                "box": box, "source": source, "seed": int(seed),
                "alpha": (None if alpha is None else float(alpha)), "mode": mode,
                "tag": candidate_tag(source, seed=seed, alpha=alpha, mode=mode),
            })

        add("hr")
        for s in seeds:
            add("frozen" if s == base_seed else "frozen_seed", seed=s)
        for a in alphas:
            add("intervention", seed=base_seed, alpha=a, mode=core_mode)
        if box in diag_boxes:
            for m in diag_modes:
                if m == core_mode:
                    continue
                for a in alphas:
                    add("intervention", seed=base_seed, alpha=a, mode=m)
    return out
