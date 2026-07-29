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

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.cov = np.asarray(self.cov, dtype=np.float64)
        d = self.mu.shape[0]
        if self.cov.shape != (d, d):
            raise ValueError(f"cov {self.cov.shape} does not match mu {self.mu.shape}")
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
    def cov_reg(self) -> np.ndarray:
        return self._cov_reg

    @property
    def precision(self) -> np.ndarray:
        return self._prec

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self._cov_reg))

    def vector(self, ens: EnsembleSummary) -> Tuple[np.ndarray, np.ndarray]:
        return summary_vector(ens, self.bins, empty_fill=self.mu)

    def mahalanobis2(self, ens: EnsembleSummary) -> float:
        s, _ = self.vector(ens)
        d = s - self.mu
        return float(d @ self._prec @ d)

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
        return np.arange(self.bins.n_sub_bins, dtype=np.int64)

    @property
    def occupation_index(self) -> np.ndarray:
        j = self.bins.n_sub_bins
        return np.arange(j, j + self.bins.n_host_bins, dtype=np.int64)

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
        """
        s, valid = self.vector(ens)
        idx = self.occupation_index
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
            idx = [j + int(i) for i in reliable_host_bins]
            out["R_occ_reliable"] = -self.block_mahalanobis2(ens, idx)
        return out

    def components(self, ens: EnsembleSummary) -> Dict[str, float]:
        """Per-bin contribution ``d_i * (C^-1 d)_i`` for diagnosis (sums to D^2)."""
        s, valid = self.vector(ens)
        d = s - self.mu
        contrib = d * (self._prec @ d)
        names = self.labels or tuple(self.bins.labels())
        out = {f"contrib_{n}": float(v) for n, v in zip(names, contrib)}
        out["mahalanobis2"] = float(d @ self._prec @ d)
        out["n_valid_bins"] = float(np.count_nonzero(valid))
        return out

    def to_dict(self) -> Dict:
        return {
            "mu": self.mu.tolist(),
            "cov": self.cov.tolist(),
            "lam": float(self.lam),
            "cov_reg_condition_number": self.condition_number,
            "cov_condition_number": float(np.linalg.cond(self.cov))
            if np.linalg.matrix_rank(self.cov) == self.dim else float("inf"),
            "bins": self.bins.to_dict(),
            "ensemble_size": int(self.ensemble_size),
            "n_draws": int(self.n_draws),
            "labels": list(self.labels or self.bins.labels()),
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
    chunk cannot dominate every ensemble. With ``replace_boxes`` the draw is a
    box-level bootstrap: boxes are resampled with replacement first, which is the
    correct independent unit here (chunks inside a box are not independent).
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

    draws: List[List[int]] = []
    for _ in range(int(n_draws)):
        if replace_boxes:
            boxes = sorted({chunks[i].box for i in range(n)})
            picked = rng.choice(boxes, size=len(boxes), replace=True)
            pool_keys = [k for k in keys if any(
                chunks[i].box in picked for i in groups[k]
            )] or keys
        else:
            pool_keys = keys
        take: List[int] = []
        order = list(pool_keys)
        rng.shuffle(order)
        k = 0
        while len(take) < ensemble_size:
            g = groups[order[k % len(order)]]
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
) -> RewardModel:
    """Estimate ``mu_HR`` and ``C_reg`` from HR chunk summaries by box bootstrap.

    ``shrinkage`` is relative: ``lambda = shrinkage * mean(diag(C))``, floored at
    ``min_lambda`` so a degenerate ``C`` (e.g. a synthetic test with identical
    draws) still inverts.
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
        meta={
            "shrinkage": float(shrinkage),
            "n_hr_chunks": int(len(hr_chunks)),
            "n_boxes": int(len({c.box for c in hr_chunks})),
            "n_draws_requested": int(n_draws),
            "n_draws_finite": int(v.shape[0]),
        },
    )
