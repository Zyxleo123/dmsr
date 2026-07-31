"""Offline elite replay: per-chunk credit, selection, and the replay dataset.

Credit assignment
-----------------
For ensemble ``E_k`` containing corrected chunk ``i``,

    A_{k,i} = R(E_k) - R(E_k^{i -> base})

where ``E_k^{i -> base}`` swaps chunk ``i``'s *catalog summary* for the summary
of the same chunk under the frozen SR2 baseline. Because summaries pool by
summing numerators and denominators, this counterfactual is arithmetic on cached
vectors -- **the halo finder is never re-run**, which is what makes a per-chunk
credit affordable at all.

``A`` is a leave-one-out difference, not a Shapley value: with ``B = 16`` chunks
the interactions are small (the reward is a smooth quadratic in a sum of 16
roughly independent contributions) and the exact version costs ``2^B``
evaluations. The approximation is stated rather than hidden.

Selection
---------
A chunk enters the replay buffer only if **both** hold:

* its ensemble is feasible and in the top ``elite_quantile`` of feasible rewards;
* its own marginal contribution is positive.

The first condition keeps the context honest (a chunk that only looks good
inside a bad ensemble is not evidence); the second stops a mediocre chunk from
riding along inside a good one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog import ChunkSummary, pool
from .reward import RewardModel

__all__ = [
    "ReplayEntry",
    "elite_weights",
    "marginal_contributions",
    "read_replay",
    "select_elites",
    "sha256_file",
    "write_replay",
]


def sha256_file(path: str | Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def marginal_contributions(
    model: RewardModel,
    ensemble: Mapping[int, ChunkSummary],
    baseline: Mapping[int, ChunkSummary],
    *,
    score: str = "R_occ",
) -> Dict[int, float]:
    """``A_{k,i}`` for every chunk of one ensemble, from cached summaries only.

    ``ensemble`` and ``baseline`` are keyed by chunk id and must cover the same
    chunks; ``baseline[i]`` is chunk ``i`` measured on the frozen SR2 field.

    ``score`` picks which reward the credit is measured in: ``R_occ`` (default),
    ``R_abund`` or the joint ``R_cat``. It must match what selection was decided
    on. Credit in ``R_cat`` while Gate B is decided on occupation up-weights
    chunks that improved *abundance* -- so the elite set trains the model toward
    a statistic nobody selected for, and a distilled model can move R_cat while
    leaving occupation exactly as flat as SR2 left it.
    """
    fn = {
        "R_cat": model.reward,
        "R_occ": model.reward_occupation,
        "R_abund": model.reward_abundance,
    }.get(str(score))
    if fn is None:
        raise ValueError(f"unknown score {score!r}; use R_occ, R_abund or R_cat")
    keys = list(ensemble)
    missing = [k for k in keys if k not in baseline]
    if missing:
        raise KeyError(f"no baseline summary for chunks {missing}")
    full = fn(pool([ensemble[k] for k in keys]))
    out: Dict[int, float] = {}
    for k in keys:
        swapped = [baseline[k] if kk == k else ensemble[kk] for kk in keys]
        out[k] = float(full - fn(pool(swapped)))
    return out


@dataclass
class ReplayEntry:
    """One selected chunk: the residual that was generated and why it was kept."""

    box: str
    chunk_id: int
    ensemble_id: str
    residual_path: str
    residual_sha256: str
    lr_id: str                       # LR field identifier (path or box name)
    base_id: str                     # cached SR2 base field identifier
    base_seed: int
    residual_seed: int
    residual_scale: float
    ensemble_reward: float
    marginal_contribution: float
    constraint_values: Dict[str, float] = dc_field(default_factory=dict)
    feasible: bool = True
    generation_checkpoint: str = ""
    hr_origin: Tuple[int, int, int] = (0, 0, 0)
    chunk_hr: int = 128
    weight: float = 1.0
    meta: Dict = dc_field(default_factory=dict)

    def key(self) -> str:
        return f"{self.ensemble_id}:{self.box}/c{self.chunk_id}"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["hr_origin"] = list(self.hr_origin)
        return d

    @staticmethod
    def from_dict(d: Dict) -> "ReplayEntry":
        d = dict(d)
        d["hr_origin"] = tuple(int(x) for x in d.get("hr_origin", (0, 0, 0)))
        return ReplayEntry(**d)


def select_elites(
    rows: Sequence[Dict],
    *,
    elite_quantile: float = 0.2,
    require_positive_marginal: bool = True,
) -> List[Dict]:
    """Filter scored ensemble/chunk rows down to the elite set.

    Each row needs ``feasible``, ``ensemble_reward``, ``marginal_contribution``
    and ``ensemble_id``. The quantile is taken over **ensembles**, not chunks, so
    a large ensemble cannot dominate the cut.
    """
    feasible = [r for r in rows if bool(r.get("feasible", False))]
    if not feasible:
        return []
    by_ens: Dict[str, float] = {}
    for r in feasible:
        by_ens[str(r["ensemble_id"])] = float(r["ensemble_reward"])
    rewards = np.asarray(sorted(by_ens.values()), dtype=np.float64)
    cut = float(np.quantile(rewards, 1.0 - float(elite_quantile)))
    keep_ens = {e for e, v in by_ens.items() if v >= cut}
    out = [r for r in feasible if str(r["ensemble_id"]) in keep_ens]
    if require_positive_marginal:
        out = [r for r in out if float(r.get("marginal_contribution", 0.0)) > 0.0]
    return out


def elite_weights(
    marginals: Sequence[float],
    *,
    tau: float = 1.0,
    a_max: float = 5.0,
    w_max: float = 10.0,
    normalize: str = "mean",
) -> np.ndarray:
    """``w_i = exp(clip(A_i, 0, A_max) / tau)``, clipped and normalised.

    Two guards, both needed. ``a_max`` bounds the exponent so one lucky ensemble
    cannot produce a weight of ``e^40``; ``w_max`` bounds the result after
    normalisation so no single example can own a batch even if every other
    weight is tiny.
    """
    a = np.clip(np.asarray(marginals, dtype=np.float64), 0.0, float(a_max))
    w = np.exp(a / max(float(tau), 1e-12))
    if normalize == "mean":
        w = w / max(float(w.mean()), 1e-30)
    elif normalize == "sum":
        w = w * (len(w) / max(float(w.sum()), 1e-30))
    elif normalize != "none":
        raise ValueError(f"unknown normalize {normalize!r}")
    return np.clip(w, 0.0, float(w_max))


def write_replay(path: str | Path, entries: Iterable[ReplayEntry]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e.to_dict(), sort_keys=True) + "\n")
    return p


def read_replay(path: str | Path, *, verify: bool = False) -> List[ReplayEntry]:
    out: List[ReplayEntry] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = ReplayEntry.from_dict(json.loads(line))
            if verify:
                rp = Path(e.residual_path)
                if not rp.is_file():
                    raise FileNotFoundError(f"replay entry {e.key()} -> missing {rp}")
                if e.residual_sha256 and sha256_file(rp) != e.residual_sha256:
                    raise RuntimeError(f"checksum mismatch for {rp}")
            out.append(e)
    return out
