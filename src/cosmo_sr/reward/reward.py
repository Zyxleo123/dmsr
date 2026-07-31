"""Ensemble-level catalog reward.

    R_cat(E) = - (s(E) - mu_HR)^T C_reg^{-1} (s(E) - mu_HR),
    C_reg    = C + lambda I .

``s(E)`` is the pooled summary vector of :mod:`cosmo_sr.reward.catalog`; ``mu_HR``
and ``C`` are estimated from HR chunk summaries by drawing ensembles of the same
size and stratification as the ones being scored, so the covariance describes
the sampling noise of an ensemble rather than of a single chunk.

The ridge ``lambda I`` is not cosmetic. With ``J + I ~ 11`` correlated bins and
only a few hundred bootstrap draws, the raw ``C`` is close to singular and its
inverse would put essentially unbounded weight on the least-constrained
direction -- a reward optimiser would find that direction and nothing else. The
shrinkage is reported alongside the condition number of ``C_reg`` so the
regularisation is always visible in the manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .catalog import CatalogBins, ChunkSummary, EnsembleSummary, pool, summary_vector

__all__ = ["RewardModel", "fit_reward_model", "stratified_ensembles"]


@dataclass
class RewardModel:
    """HR target mean, regularized covariance, and the reward it defines."""

    mu: np.ndarray                    # (D,)
    cov: np.ndarray                   # (D, D) raw bootstrap covariance
    lam: float                        # ridge added to the diagonal
    bins: CatalogBins
    ensemble_size: int
    n_draws: int
    labels: Tuple[str, ...] = ()
    meta: Dict = dc_field(default_factory=dict)
    # Which of the ``D`` binned statistics the reward is actually a function of.
    # ``mu`` and ``cov`` stay full-dimensional -- an excluded bin is still
    # *reported*, it just carries no gradient and no Mahalanobis weight -- but
    # ``R_cat`` is the quadratic form on this sub-block alone. Empty by
    # convention means "all dimensions active".
    active_dims: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.cov = np.asarray(self.cov, dtype=np.float64)
        d = self.mu.shape[0]
        if self.cov.shape != (d, d):
            raise ValueError(f"cov {self.cov.shape} does not match mu {self.mu.shape}")
        self.active_dims = tuple(int(i) for i in (self.active_dims or range(d)))
        bad = [i for i in self.active_dims if not 0 <= i < d]
        if bad:
            raise ValueError(f"active_dims {bad} out of range for dim {d}")
        if len(set(self.active_dims)) != len(self.active_dims):
            raise ValueError(f"active_dims has duplicates: {self.active_dims}")
        self._active = np.asarray(self.active_dims, dtype=np.int64)
        self._cov_reg = self.cov + float(self.lam) * np.eye(d)
        self._prec = np.linalg.inv(self._cov_reg)
        # Symmetrise: inv() of a symmetric matrix can drift at the 1e-16 level,
        # and an asymmetric precision would make the reward depend on the
        # arbitrary ordering of the quadratic form.
        self._prec = 0.5 * (self._prec + self._prec.T)

    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])

    @property
    def active_dim(self) -> int:
        """Number of dimensions the reward is actually scored on."""
        return int(self._active.size)

    @property
    def cov_reg(self) -> np.ndarray:
        return self._cov_reg

    @property
    def precision(self) -> np.ndarray:
        """Precision of the ACTIVE block, embedded back into the full ``D x D``.

        Inactive rows/columns are exactly zero, so ``d @ precision @ d`` is the
        reward's quadratic form no matter what an excluded bin contains.
        """
        cache = self.__dict__.setdefault("_prec_active", {})
        if "full" not in cache:
            p = np.zeros((self.dim, self.dim), dtype=np.float64)
            p[np.ix_(self._active, self._active)] = self._block_precision(self._active)
            cache["full"] = p
        return cache["full"]

    @property
    def condition_number(self) -> float:
        """cond of the ACTIVE regularized covariance -- what the reward inverts."""
        return float(np.linalg.cond(self._cov_reg[np.ix_(self._active, self._active)]))

    @property
    def condition_number_full(self) -> float:
        return float(np.linalg.cond(self._cov_reg))

    def vector(self, ens: EnsembleSummary) -> Tuple[np.ndarray, np.ndarray]:
        return summary_vector(ens, self.bins, empty_fill=self.mu)

    def mahalanobis2(self, ens: EnsembleSummary) -> float:
        return self.block_mahalanobis2(ens, self._active)

    def reward(self, ens: EnsembleSummary) -> float:
        """``-D^2``. Maximal (0) exactly when ``s(E) == mu_HR``."""
        return -self.mahalanobis2(ens)

    def reward_of_chunks(self, chunks: Sequence[ChunkSummary]) -> float:
        return self.reward(pool(chunks))

    # ------------------------------------------------------------------
    # Sub-rewards.
    #
    # The joint 11-d Mahalanobis distance can fall because abundance improved
    # while occupation stayed flat, and occupation is the primary scientific
    # target. So the two blocks are also scored separately.
    #
    # A block score is the *marginal* Mahalanobis distance: the sub-vector
    # against the corresponding sub-block of ``C_reg``, inverted on its own.
    # It is deliberately not the partial (conditional) distance and not a
    # slice of the joint precision -- both of those mix in the other block's
    # residual through the cross-covariance, which is exactly what we are
    # trying to separate. The three numbers therefore do not add up to the
    # joint distance, and are not meant to.
    # ------------------------------------------------------------------

    @property
    def abundance_index(self) -> np.ndarray:
        """Active abundance dimensions."""
        return np.asarray(
            [i for i in self._active if i < self.bins.n_sub_bins], dtype=np.int64
        )

    @property
    def occupation_index(self) -> np.ndarray:
        """Active occupation dimensions (excluded host bins are not scored)."""
        j = self.bins.n_sub_bins
        return np.asarray(
            [i for i in self._active if j <= i < j + self.bins.n_host_bins],
            dtype=np.int64,
        )

    def _block_precision(self, idx: np.ndarray) -> np.ndarray:
        key = tuple(int(i) for i in idx)
        cache = self.__dict__.setdefault("_block_cache", {})
        if key not in cache:
            sub = self._cov_reg[np.ix_(idx, idx)]
            p = np.linalg.inv(sub)
            cache[key] = 0.5 * (p + p.T)
        return cache[key]

    def block_mahalanobis2(
        self, ens: EnsembleSummary, idx: Sequence[int]
    ) -> float:
        idx = np.asarray(list(idx), dtype=np.int64)
        if idx.size == 0:
            return float("nan")
        s, _ = self.vector(ens)
        d = (s - self.mu)[idx]
        return float(d @ self._block_precision(idx) @ d)

    def reward_occupation(self, ens: EnsembleSummary,
                          idx: Optional[Sequence[int]] = None) -> float:
        """``R_occ``: occupation-only covariance-normalized score."""
        return -self.block_mahalanobis2(
            ens, self.occupation_index if idx is None else idx
        )

    def reward_abundance(self, ens: EnsembleSummary,
                         idx: Optional[Sequence[int]] = None) -> float:
        """``R_abund``: abundance-only covariance-normalized score."""
        return -self.block_mahalanobis2(
            ens, self.abundance_index if idx is None else idx
        )

    def occupation_curve(self, ens: EnsembleSummary) -> np.ndarray:
        """``<N_sub | M_host>`` per host bin, NaN where the bin has no hosts."""
        return np.asarray(ens.occupation(), dtype=np.float64)

    def occupation_gap(self, ens: EnsembleSummary) -> np.ndarray:
        """Per-host-bin ``|s_i - mu_i|`` in whitened (per-bin sigma) units.

        The sign is dropped deliberately: occupation can in principle overshoot,
        and "closer to HR" is the improvement we mean. Bins with no hosts give
        NaN rather than a fabricated zero.

        Reported for **every** host bin, including ones excluded from the reward
        by ``active_dims``: the gate indexes this by host-bin number, and an
        excluded bin is still evidence about what the residual did.
        """
        s, valid = self.vector(ens)
        j = self.bins.n_sub_bins
        idx = np.arange(j, j + self.bins.n_host_bins, dtype=np.int64)
        sd = np.sqrt(np.diag(self._cov_reg)[idx])
        gap = np.abs(s[idx] - self.mu[idx]) / np.where(sd > 0, sd, np.nan)
        return np.where(valid[idx], gap, np.nan)

    def scores(self, ens: EnsembleSummary,
               reliable_host_bins: Optional[Sequence[int]] = None) -> Dict[str, float]:
        """``R_cat``, ``R_occ``, ``R_abund`` and the reliable-bin variant."""
        out = {
            "R_cat": self.reward(ens),
            "R_occ": self.reward_occupation(ens),
            "R_abund": self.reward_abundance(ens),
        }
        if reliable_host_bins is not None:
            j = self.bins.n_sub_bins
            active = set(int(i) for i in self._active)
            idx = [j + int(i) for i in reliable_host_bins if j + int(i) in active]
            out["R_occ_reliable"] = -self.block_mahalanobis2(ens, idx)
        return out

    def components(self, ens: EnsembleSummary) -> Dict[str, float]:
        """Per-bin contribution ``d_i * (C^-1 d)_i`` for diagnosis (sums to D^2).

        Bins outside ``active_dims`` contribute exactly zero, because they are
        not in the quadratic form at all.
        """
        s, valid = self.vector(ens)
        d = s - self.mu
        contrib = d * (self.precision @ d)
        names = self.labels or tuple(self.bins.labels())
        out = {f"contrib_{n}": float(v) for n, v in zip(names, contrib)}
        out["mahalanobis2"] = float(contrib.sum())
        out["n_valid_bins"] = float(np.count_nonzero(valid))
        out["n_active_bins"] = float(self.active_dim)
        return out

    def to_dict(self) -> Dict:
        return {
            "mu": self.mu.tolist(),
            "cov": self.cov.tolist(),
            "lam": float(self.lam),
            "cov_reg_condition_number": self.condition_number,
            "cov_reg_condition_number_full": self.condition_number_full,
            "cov_condition_number": float(np.linalg.cond(self.cov))
            if np.linalg.matrix_rank(self.cov) == self.dim else float("inf"),
            "bins": self.bins.to_dict(),
            "ensemble_size": int(self.ensemble_size),
            "n_draws": int(self.n_draws),
            "labels": list(self.labels or self.bins.labels()),
            "active_dims": [int(i) for i in self.active_dims],
            "active_labels": [
                (self.labels or tuple(self.bins.labels()))[i] for i in self.active_dims
            ],
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict) -> "RewardModel":
        from .catalog import load_bins

        b = d["bins"]
        bins = CatalogBins(
            sub_mass_edges=tuple(b["sub_mass_edges"]),
            host_mass_edges=tuple(b["host_mass_edges"]),
            min_sub_particles=int(b["min_sub_particles"]),
            min_host_particles=int(b["min_host_particles"]),
            min_purity=float(b["min_purity"]),
            radius_mult=float(b["radius_mult"]),
            abundance_transform=str(b["abundance_transform"]),
            occupation_transform=str(b["occupation_transform"]),
            abundance_floor_halos=float(b["abundance_floor_halos"]),
            occupation_floor=float(b["occupation_floor"]),
        )
        return RewardModel(
            mu=np.asarray(d["mu"], dtype=np.float64),
            cov=np.asarray(d["cov"], dtype=np.float64),
            lam=float(d["lam"]),
            bins=bins,
            ensemble_size=int(d["ensemble_size"]),
            n_draws=int(d["n_draws"]),
            labels=tuple(d.get("labels", ())),
            active_dims=tuple(int(i) for i in d.get("active_dims", ())),
            meta=dict(d.get("meta", {})),
        )


def stratified_ensembles(
    chunks: Sequence[ChunkSummary],
    ensemble_size: int,
    n_draws: int,
    *,
    strata: Optional[Sequence[str]] = None,
    seed: int = 0,
    replace_boxes: bool = True,
) -> List[List[int]]:
    """Index lists of ``ensemble_size`` chunks, balanced over strata.

    ``strata[i]`` labels chunk ``i`` (box, host-mass class, environment class...).
    Each draw takes as evenly as possible from every stratum, so one massive-host
    chunk cannot dominate every ensemble.

    With ``replace_boxes`` the draw is a **box-level bootstrap**: boxes are
    resampled with replacement and a box drawn twice contributes twice as many
    chances, exactly as a bootstrap of the independent unit requires. Chunks
    inside a box are not independent, so resampling chunks directly would
    understate the ensemble covariance -- which is the whole quantity being
    estimated. (An earlier version resampled boxes only to build a *set* of
    eligible strata and then drew chunks uniformly across it; that discarded the
    multiplicity and was a chunk bootstrap wearing a box bootstrap's name.)
    """
    rng = np.random.default_rng(int(seed))
    n = len(chunks)
    if n == 0:
        raise ValueError("no chunk summaries to draw from")
    if strata is None:
        strata = [c.box for c in chunks]
    strata = list(strata)
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(str(s), []).append(i)
    keys = sorted(groups)
    boxes = sorted({c.box for c in chunks})
    by_box: Dict[str, List[str]] = {}
    for k in keys:
        for b in sorted({chunks[i].box for i in groups[k]}):
            by_box.setdefault(b, []).append(k)

    draws: List[List[int]] = []
    for _ in range(int(n_draws)):
        if replace_boxes:
            picked = list(rng.choice(boxes, size=len(boxes), replace=True))
            # Within a resampled box only that box's chunks are eligible, and
            # there is one slot per (resampled box, stratum) pair -- so a box
            # drawn twice really does contribute its chunks twice.
            eligible = {
                (b, k): [i for i in groups[k] if chunks[i].box == b]
                for b in set(picked) for k in by_box.get(b, [])
            }
            slots = [(b, k) for b in picked for k in by_box.get(b, [])]
        else:
            eligible = {(None, k): groups[k] for k in keys}
            slots = [(None, k) for k in keys]
        slots = [s for s in slots if eligible[s]]
        if not slots:
            raise ValueError("bootstrap resampling produced no eligible chunks")
        rng.shuffle(slots)
        take: List[int] = []
        k = 0
        while len(take) < ensemble_size:
            g = eligible[slots[k % len(slots)]]
            if g:
                take.append(int(rng.choice(g)))
            k += 1
        draws.append(take[:ensemble_size])
    return draws


def fit_reward_model(
    hr_chunks: Sequence[ChunkSummary],
    bins: CatalogBins,
    *,
    ensemble_size: int = 16,
    n_draws: int = 400,
    shrinkage: float = 0.1,
    strata: Optional[Sequence[str]] = None,
    seed: int = 0,
    min_lambda: float = 1e-8,
    active_dims: Optional[Sequence[int]] = None,
) -> RewardModel:
    """Estimate ``mu_HR`` and ``C_reg`` from HR chunk summaries by box bootstrap.

    ``shrinkage`` is relative: ``lambda = shrinkage * mean(diag(C))``, floored at
    ``min_lambda`` so a degenerate ``C`` (e.g. a synthetic test with identical
    draws) still inverts.

    ``active_dims`` restricts the scored sub-block (see
    :attr:`RewardModel.active_dims`); ``mu`` and ``C`` are still estimated on
    every dimension so an excluded bin remains reportable.
    """
    draws = stratified_ensembles(
        hr_chunks, ensemble_size, n_draws, strata=strata, seed=seed
    )
    vectors = []
    for idx in draws:
        ens = pool([hr_chunks[i] for i in idx])
        s, _ = summary_vector(ens, bins)
        vectors.append(s)
    v = np.asarray(vectors, dtype=np.float64)
    finite = np.all(np.isfinite(v), axis=1)
    if not finite.any():
        raise RuntimeError("every bootstrap draw produced a non-finite summary vector")
    v = v[finite]
    mu = v.mean(axis=0)
    cov = np.cov(v, rowvar=False) if v.shape[0] > 1 else np.zeros((v.shape[1],) * 2)
    cov = np.atleast_2d(cov)
    lam = max(float(shrinkage) * float(np.mean(np.diag(cov))), float(min_lambda))
    return RewardModel(
        mu=mu, cov=cov, lam=lam, bins=bins,
        ensemble_size=int(ensemble_size), n_draws=int(v.shape[0]),
        labels=tuple(bins.labels()),
        active_dims=tuple(int(i) for i in active_dims) if active_dims is not None else (),
        meta={
            "shrinkage": float(shrinkage),
            "n_hr_chunks": int(len(hr_chunks)),
            "n_boxes": int(len({c.box for c in hr_chunks})),
            "n_draws_requested": int(n_draws),
            "n_draws_finite": int(v.shape[0]),
            "box_bootstrap": True,
        },
    )
