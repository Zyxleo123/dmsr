"""Bounded, resumable cross-entropy method over the local-editor action space.

CEM: sample actions from a distribution, keep the best ones, refit the
distribution around them. It is the right search here for two reasons that are
both about cost rather than elegance. The objective needs a full-box Rockstar
run per candidate box (~10-20 min), so the budget is measured in *hundreds* of
evaluations, not millions -- ruling out anything gradient-free and sample-hungry
like ES with large populations. And the objective is discontinuous: an object
either appears or it does not, so there is no gradient to estimate in the first
place.

Everything happens in the **unconstrained** coordinates of
:class:`cosmo_sr.reward.local_editor.ActionCodec`. The sampling distribution is
therefore an ordinary Gaussian on ``R^d`` that never has to be truncated,
rejected or clipped, and the bounds are enforced exactly by the squashing map
rather than approximately by the search.

Design decisions that are not defaults
--------------------------------------
*Diagonal covariance.* A round is 24-32 candidates against 12 dimensions. A full
covariance estimated from ~6 elites is dominated by the sample, and its
off-diagonal structure would be noise the next round then samples along.
Per-parameter variances are what the population can actually support, and the
plan asks for exactly that.

*A variance floor.* Classic CEM's failure mode is premature collapse: one lucky
elite cluster shrinks sigma, the next round samples only there, and the search
is over after two rounds. ``sigma_floor`` bounds that from below.

*An explicit exploration mixture.* A fixed fraction of every round is drawn from
the *initial* distribution, not the current one. This is stronger than the
variance floor -- it keeps global coverage even after the mean has moved far --
and it means a round is never wholly wasted if the mean has wandered somewhere
useless.

*Refusing to update.* Two situations produce a "best" candidate that carries no
information: no candidate was feasible, and every candidate scored the same
(overwhelmingly likely early on, when nothing creates an object and every reward
is 0). Fitting to an arbitrary subset of ties is how a search convinces itself
it is making progress. Both cases leave the distribution untouched and say so in
the manifest.

Resumability
------------
Each round writes one JSON manifest holding the state that *produced* it, the
sampled vectors, and (after scoring) the rewards. :meth:`CEMRun.resume` replays
the completed rounds' updates; because sampling is seeded from
``(seed, round_index)`` alone, a resumed run emits bit-identical candidates to
one that never stopped. ``tests/reward/test_local_cem.py`` pins that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["CEMRun", "CEMState", "elite_threshold"]

_TIE_TOL = 1e-12


@dataclass
class CEMState:
    """A diagonal Gaussian over unconstrained action coordinates, plus its rules."""

    dim: int
    mean: np.ndarray
    std: np.ndarray
    init_mean: np.ndarray
    init_std: np.ndarray
    round_index: int = 0
    seed: int = 0
    n_samples: int = 32
    elite_frac: float = 0.2
    sigma_floor: float = 0.25
    explore_mix: float = 0.25
    min_elites: int = 3
    history: List[Dict] = dc_field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("mean", "std", "init_mean", "init_std"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64).reshape(-1))
        if not (self.mean.size == self.std.size == self.init_mean.size
                == self.init_std.size == int(self.dim)):
            raise ValueError("mean/std/init_* must all have length dim")
        if np.any(self.std <= 0) or np.any(self.init_std <= 0):
            raise ValueError("standard deviations must be positive")
        if not 0.0 < float(self.elite_frac) <= 1.0:
            raise ValueError(f"elite_frac must be in (0, 1], got {self.elite_frac}")
        if not 0.0 <= float(self.explore_mix) < 1.0:
            raise ValueError(f"explore_mix must be in [0, 1), got {self.explore_mix}")

    # -- construction --------------------------------------------------------

    @staticmethod
    def initial(dim: int, *, seed: int = 0, sigma: float = 1.5, **kw) -> "CEMState":
        """Standard start: zero mean in ``z``, which is the centre of every box.

        ``sigma = 1.5`` puts roughly the central 80% of each bounded parameter
        inside one standard deviation, so the first round samples the interior
        broadly without piling up on the bounds (a Gaussian with large sigma in
        ``z`` degenerates onto the two endpoints after squashing).
        """
        m = np.zeros(int(dim), dtype=np.float64)
        s = np.full(int(dim), float(sigma), dtype=np.float64)
        return CEMState(dim=int(dim), mean=m.copy(), std=s.copy(),
                        init_mean=m.copy(), init_std=s.copy(), seed=int(seed), **kw)

    # -- sampling ------------------------------------------------------------

    def rng(self) -> np.random.Generator:
        """Seeded from ``(seed, round_index)`` only -- never from wall clock or state."""
        return np.random.default_rng([int(self.seed), int(self.round_index)])

    def n_explore(self, n: Optional[int] = None) -> int:
        n = int(self.n_samples if n is None else n)
        return int(round(float(self.explore_mix) * n))

    def sample(self, n: Optional[int] = None) -> np.ndarray:
        """``(n, dim)`` unconstrained action vectors for this round.

        The exploration draws are the *last* rows, and every normal deviate is
        drawn in one call before any transformation, so changing ``explore_mix``
        between runs cannot silently reshuffle which candidate is which.
        """
        n = int(self.n_samples if n is None else n)
        xi = self.rng().standard_normal((n, int(self.dim)))
        z = self.mean[None, :] + self.std[None, :] * xi
        k = self.n_explore(n)
        if k > 0:
            z[n - k:] = self.init_mean[None, :] + self.init_std[None, :] * xi[n - k:]
        return z

    def is_explore(self, n: Optional[int] = None) -> np.ndarray:
        n = int(self.n_samples if n is None else n)
        out = np.zeros(n, dtype=bool)
        k = self.n_explore(n)
        if k > 0:
            out[n - k:] = True
        return out

    # -- update --------------------------------------------------------------

    def update(
        self,
        z: np.ndarray,
        rewards: np.ndarray,
        feasible: Optional[np.ndarray] = None,
    ) -> Tuple["CEMState", Dict]:
        """Refit around the elites; return ``(next_state, info)``.

        Never raises on a degenerate round. A search that stops because nothing
        worked is worse than one that keeps sampling the same distribution and
        records why, since the manifest is then a readable account of what the
        action space did rather than a traceback.
        """
        z = np.asarray(z, dtype=np.float64).reshape(-1, int(self.dim))
        r = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if r.shape[0] != z.shape[0]:
            raise ValueError(f"{z.shape[0]} vectors but {r.shape[0]} rewards")
        ok = np.isfinite(r)
        if feasible is not None:
            ok &= np.asarray(feasible, dtype=bool).reshape(-1)

        info: Dict = {
            "round_index": int(self.round_index),
            "n_candidates": int(z.shape[0]),
            "n_feasible": int(np.count_nonzero(ok)),
            "reward_max": float(np.max(r[ok])) if np.any(ok) else float("nan"),
            "reward_mean": float(np.mean(r[ok])) if np.any(ok) else float("nan"),
            "updated": False,
            "reason": "",
        }
        nxt = CEMState(dim=self.dim, mean=self.mean.copy(), std=self.std.copy(),
                       init_mean=self.init_mean.copy(), init_std=self.init_std.copy(),
                       round_index=int(self.round_index) + 1, seed=int(self.seed),
                       n_samples=int(self.n_samples), elite_frac=float(self.elite_frac),
                       sigma_floor=float(self.sigma_floor),
                       explore_mix=float(self.explore_mix),
                       min_elites=int(self.min_elites),
                       history=list(self.history))

        if not np.any(ok):
            info["reason"] = "no_feasible_candidate"
        elif float(np.max(r[ok]) - np.min(r[ok])) <= _TIE_TOL:
            info["reason"] = "all_candidates_equivalent"
        else:
            zv, rv = z[ok], r[ok]
            n_elite = int(max(int(self.min_elites),
                              int(np.ceil(float(self.elite_frac) * zv.shape[0]))))
            n_elite = int(min(n_elite, zv.shape[0]))
            elite = zv[np.argsort(-rv, kind="stable")[:n_elite]]
            nxt.mean = elite.mean(axis=0)
            sd = (elite.std(axis=0, ddof=1) if n_elite > 1
                  else np.zeros(int(self.dim)))
            nxt.std = np.maximum(sd, float(self.sigma_floor))
            info.update(updated=True, reason="refit", n_elites=int(n_elite),
                        elite_reward_min=float(rv[np.argsort(-rv)[n_elite - 1]]),
                        elite_mean=nxt.mean.tolist(), elite_std=nxt.std.tolist(),
                        raw_elite_std=sd.tolist(),
                        n_floored=int(np.count_nonzero(sd < float(self.sigma_floor))))
        nxt.history = list(self.history) + [info]
        return nxt, info

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "dim": int(self.dim),
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "init_mean": self.init_mean.tolist(), "init_std": self.init_std.tolist(),
            "round_index": int(self.round_index), "seed": int(self.seed),
            "n_samples": int(self.n_samples), "elite_frac": float(self.elite_frac),
            "sigma_floor": float(self.sigma_floor),
            "explore_mix": float(self.explore_mix),
            "min_elites": int(self.min_elites),
            "history": list(self.history),
        }

    @staticmethod
    def from_dict(d: Dict) -> "CEMState":
        return CEMState(
            dim=int(d["dim"]), mean=np.asarray(d["mean"]), std=np.asarray(d["std"]),
            init_mean=np.asarray(d["init_mean"]), init_std=np.asarray(d["init_std"]),
            round_index=int(d.get("round_index", 0)), seed=int(d.get("seed", 0)),
            n_samples=int(d.get("n_samples", 32)),
            elite_frac=float(d.get("elite_frac", 0.2)),
            sigma_floor=float(d.get("sigma_floor", 0.25)),
            explore_mix=float(d.get("explore_mix", 0.25)),
            min_elites=int(d.get("min_elites", 3)),
            history=list(d.get("history", [])),
        )


def elite_threshold(rewards: Sequence[float], elite_frac: float,
                    min_elites: int = 3) -> float:
    """Reward of the worst elite -- the cut the replay set of stage 5 inherits."""
    r = np.sort(np.asarray(rewards, dtype=np.float64))[::-1]
    if r.size == 0:
        return float("nan")
    k = int(min(r.size, max(int(min_elites), int(np.ceil(float(elite_frac) * r.size)))))
    return float(r[k - 1])


# ---------------------------------------------------------------------------
# Round manifests on disk
# ---------------------------------------------------------------------------


@dataclass
class CEMRun:
    """One search, as a directory of per-round JSON manifests.

    A round has two writes: ``propose`` (state + sampled vectors) before the
    expensive evaluation, and ``record`` (rewards) after it. Splitting them is
    what makes a run resumable at the granularity the cluster actually fails at
    -- a time limit hit halfway through a scoring array loses that array, not the
    search.
    """

    root: Path
    name: str = "cem"

    @property
    def dir(self) -> Path:
        return Path(self.root) / self.name

    def round_path(self, i: int) -> Path:
        return self.dir / f"round_{int(i):03d}.json"

    def write_round(self, state: CEMState, z: np.ndarray,
                    extra: Optional[Dict] = None) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.round_path(state.round_index)
        doc = {
            "round_index": int(state.round_index),
            "state": state.to_dict(),
            "z": np.asarray(z, dtype=np.float64).tolist(),
            "is_explore": state.is_explore(np.asarray(z).shape[0]).tolist(),
            "rewards": None, "feasible": None, "scored": False,
        }
        doc.update(extra or {})
        p.write_text(json.dumps(doc, indent=2, sort_keys=True))
        return p

    def record_rewards(self, i: int, rewards: Sequence[float],
                       feasible: Optional[Sequence[bool]] = None,
                       extra: Optional[Dict] = None) -> Path:
        p = self.round_path(i)
        doc = json.loads(p.read_text())
        doc["rewards"] = [float(x) for x in rewards]
        doc["feasible"] = (None if feasible is None
                           else [bool(x) for x in feasible])
        doc["scored"] = True
        doc.update(extra or {})
        p.write_text(json.dumps(doc, indent=2, sort_keys=True))
        return p

    def read_round(self, i: int) -> Optional[Dict]:
        p = self.round_path(i)
        return json.loads(p.read_text()) if p.is_file() else None

    def rounds(self) -> List[Dict]:
        out = []
        for p in sorted(self.dir.glob("round_*.json")):
            out.append(json.loads(p.read_text()))
        return out

    def resume(self, initial: CEMState) -> Tuple[CEMState, List[Dict]]:
        """State for the next round, by replaying every *scored* round in order.

        Replaying the updates rather than trusting a stored "current state" means
        a manifest edited by hand, or a round rescored after a bug fix, changes
        the search the way it should. The stored state is still written for
        provenance and is compared against the replay by the tests.
        """
        state = initial
        infos: List[Dict] = []
        for doc in self.rounds():
            if not doc.get("scored"):
                break
            if int(doc["round_index"]) != int(state.round_index):
                raise ValueError(
                    f"manifest gap: round {doc['round_index']} found while "
                    f"replaying at round {state.round_index}"
                )
            z = np.asarray(doc["z"], dtype=np.float64)
            r = np.asarray(doc["rewards"], dtype=np.float64)
            f = (None if doc.get("feasible") is None
                 else np.asarray(doc["feasible"], dtype=bool))
            state, info = state.update(z, r, f)
            infos.append(info)
        return state, infos
