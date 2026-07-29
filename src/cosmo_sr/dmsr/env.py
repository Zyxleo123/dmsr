"""Environment descriptors, balanced sampling, and the source-classifier check.

The Stage C-vs-D comparison has a built-in trap. The critic sees **real** examples
only from the 3 paired HR boxes, but in Stage D its **fake** examples come from
hundreds of LR-only boxes. If those two pools differ in their LR environment
statistics, the critic can separate real from fake by reading the *environment*
rather than the *quality* -- and Stage D would look different from Stage C for a
reason that has nothing to do with the hypothesis under test.

So Stage D samples LR-only crops to match the paired environment distribution,
and we verify the match with an adversarial diagnostic: a classifier trying to
predict paired-vs-unpaired *from the descriptors alone* should be at chance.

    target: source_classifier_AUC <= 0.60

ROC-AUC rather than accuracy because the two pools have very different sizes and
accuracy is trivially gamed by predicting the majority class.

A note on which descriptors are usable
--------------------------------------
Several natural descriptors are **identically constant** here and carry no
information, so they cannot help balancing and only add noise to the classifier:

* ``mean_density`` -- :func:`cosmo_sr.eval.density.cic_density` wraps particles
  within the crop (``g % n``) and normalises by the crop mean, so the mean
  overdensity is exactly 0 for every crop by construction.
* a periodic-FD ``mean(div v)`` would likewise vanish by the divergence theorem.

They are still computed and reported (the design asks for them), but
:class:`DescriptorStandardizer` automatically drops any descriptor whose training
standard deviation is below ``min_std``, and records which were dropped. That
keeps the descriptor space low-dimensional, which is what actually makes the
histogram matching work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..eval.density import cic_density

DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "redshift",
    "mean_density",
    "var_density",
    "mean_div_v",
    "var_div_v",
    "disp_rms",
    "tidal_I2",
    "tidal_I3",
)


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #
def _divergence(vec: torch.Tensor) -> torch.Tensor:
    """Central-difference divergence of a ``(B, 3, n, n, n)`` field (periodic)."""
    d = torch.zeros_like(vec[:, :1])
    for i, dim in enumerate((-3, -2, -1)):
        comp = vec[:, i : i + 1]
        d = d + 0.5 * (torch.roll(comp, -1, dims=dim) - torch.roll(comp, 1, dims=dim))
    return d


def _tidal_invariants(delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean 2nd and 3rd invariants of the tidal tensor ``t_ij = k_i k_j / k^2 delta``."""
    n = delta.shape[-1]
    dev = delta.device
    kx = torch.fft.fftfreq(n, device=dev) * n
    kz = torch.fft.rfftfreq(n, device=dev) * n
    KX, KY, KZ = torch.meshgrid(kx, kx, kz, indexing="ij")
    k = torch.stack([KX, KY, KZ])
    k2 = (k ** 2).sum(0)
    k2 = torch.where(k2 == 0, torch.ones_like(k2), k2)

    fd = torch.fft.rfftn(delta.float(), dim=(-3, -2, -1))
    t = torch.empty(delta.shape[0], 3, 3, *delta.shape[-3:], device=dev)
    for i in range(3):
        for j in range(i, 3):
            tij = torch.fft.irfftn(
                fd * (k[i] * k[j] / k2), s=delta.shape[-3:], dim=(-3, -2, -1)
            )
            t[:, i, j] = tij[:, 0]
            t[:, j, i] = tij[:, 0]

    tr = t[:, 0, 0] + t[:, 1, 1] + t[:, 2, 2]
    tr_t2 = (t * t.transpose(1, 2)).sum(dim=(1, 2))
    i2 = 0.5 * (tr ** 2 - tr_t2)
    i3 = torch.linalg.det(t.permute(0, 3, 4, 5, 1, 2))
    return i2.mean(dim=(-3, -2, -1)), i3.mean(dim=(-3, -2, -1))


@torch.no_grad()
def environment_descriptors(
    y: torch.Tensor,
    redshift: float = 0.0,
    disp_channels: Sequence[int] = (0, 1, 2),
    vel_channels: Optional[Sequence[int]] = (3, 4, 5),
    cellsize: float = 1000.0 * 1000.0 / 64.0,
    dis_norm: float = 6000.0,
    use_tidal: bool = False,
) -> torch.Tensor:
    """Per-crop environment descriptors from the LR field only.

    Returns ``(B, len(DESCRIPTOR_NAMES))``. Entries for unavailable channels
    (e.g. ``vel_channels=None`` in a displacement-only run) and for disabled
    tidal invariants are filled with zeros, and get dropped downstream by
    :class:`DescriptorStandardizer` as constant.
    """
    if y.dim() != 5:
        raise ValueError(f"expected (B, C, n, n, n), got {tuple(y.shape)}")
    b = y.shape[0]
    dev = y.device
    out = torch.zeros(b, len(DESCRIPTOR_NAMES), device=dev)

    disp = y.index_select(1, torch.tensor(list(disp_channels), device=dev))
    delta = cic_density(disp, cellsize, dis_norm)

    out[:, 0] = float(redshift)
    out[:, 1] = delta.mean(dim=(1, 2, 3, 4))
    out[:, 2] = delta.var(dim=(1, 2, 3, 4))
    if vel_channels is not None and max(vel_channels) < y.shape[1]:
        vel = y.index_select(1, torch.tensor(list(vel_channels), device=dev))
        div = _divergence(vel)
        out[:, 3] = div.mean(dim=(1, 2, 3, 4))
        out[:, 4] = div.var(dim=(1, 2, 3, 4))
    out[:, 5] = disp.pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    if use_tidal:
        i2, i3 = _tidal_invariants(delta)
        out[:, 6] = i2
        out[:, 7] = i3
    return out


# --------------------------------------------------------------------------- #
# Standardization (fit on TRAINING data only)
# --------------------------------------------------------------------------- #
@dataclass
class DescriptorStandardizer:
    """Z-score descriptors, dropping those that are constant on training data."""

    mean: np.ndarray
    std: np.ndarray
    keep: np.ndarray  # boolean mask over DESCRIPTOR_NAMES
    min_std: float = 1e-6

    @classmethod
    def fit(cls, desc: np.ndarray, min_std: float = 1e-6) -> "DescriptorStandardizer":
        desc = np.asarray(desc, dtype=np.float64)
        mean = desc.mean(axis=0)
        std = desc.std(axis=0)
        keep = std > float(min_std)
        return cls(mean=mean, std=std, keep=keep, min_std=float(min_std))

    def transform(self, desc: np.ndarray) -> np.ndarray:
        desc = np.asarray(desc, dtype=np.float64)
        z = (desc - self.mean) / np.where(self.std > self.min_std, self.std, 1.0)
        return z[:, self.keep]

    @property
    def kept_names(self) -> List[str]:
        return [n for n, k in zip(DESCRIPTOR_NAMES, self.keep) if k]

    @property
    def dropped_names(self) -> List[str]:
        return [n for n, k in zip(DESCRIPTOR_NAMES, self.keep) if not k]

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "keep": self.keep.tolist(),
            "min_std": self.min_std,
            "kept_names": self.kept_names,
            "dropped_names": self.dropped_names,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DescriptorStandardizer":
        return cls(
            mean=np.asarray(d["mean"]),
            std=np.asarray(d["std"]),
            keep=np.asarray(d["keep"], dtype=bool),
            min_std=float(d.get("min_std", 1e-6)),
        )


# --------------------------------------------------------------------------- #
# Balanced sampling
# --------------------------------------------------------------------------- #
@dataclass
class BalanceReport:
    """Bookkeeping from :class:`EnvironmentBalancedSampler`, saved with the run."""

    n_paired: int
    n_unpaired: int
    n_in_support: int
    n_rejected: int
    n_bins_occupied: int
    auc_before: float
    auc_after: float
    kept_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


class EnvironmentBalancedSampler:
    """Sample LR-only crops so their descriptor histogram matches the paired one.

    Method: standardize (training statistics only), project onto the leading
    ``n_dims`` PCA directions **of the paired pool**, bin that space with
    quantile edges taken from the paired pool (so the paired histogram is uniform
    by construction), then weight each unpaired crop by
    ``target_prob(bin) / n_unpaired_in(bin)``.

    Unpaired crops landing in a bin with **no paired support** get weight 0 and
    are counted as rejected. We do not claim generalization to those
    environments; they are excluded, not extrapolated into.
    """

    def __init__(
        self,
        paired_desc: np.ndarray,
        unpaired_desc: np.ndarray,
        standardizer: DescriptorStandardizer,
        n_dims: int = 2,
        n_bins: int = 8,
        seed: int = 0,
    ):
        self.standardizer = standardizer
        self.n_dims = int(n_dims)
        self.n_bins = int(n_bins)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        zp = standardizer.transform(paired_desc)
        zu = standardizer.transform(unpaired_desc)
        self._n_paired = int(len(zp))
        if zp.shape[1] == 0:
            raise ValueError(
                "all descriptors were dropped as constant; balancing is impossible. "
                "Check that the descriptor computation is receiving varied crops."
            )
        d = min(self.n_dims, zp.shape[1])

        # PCA basis from the paired pool (the distribution we are matching *to*).
        self._pca_mean = zp.mean(axis=0)
        _, _, vt = np.linalg.svd(zp - self._pca_mean, full_matrices=False)
        self._basis = vt[:d].T                       # (n_kept, d)

        pp = (zp - self._pca_mean) @ self._basis
        pu = (zu - self._pca_mean) @ self._basis

        # Quantile bin edges per PCA axis, from the paired pool.
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        self._edges = [np.quantile(pp[:, j], qs) for j in range(d)]

        bp = self._bin_index(pp)
        bu = self._bin_index(pu)

        # Explicit support box, checked *before* the bin counts. Quantile bins
        # have *unbounded* outer edges, so a crop arbitrarily far outside the
        # paired distribution still lands in the last (populated) bin and would
        # otherwise be treated as supported. Anything beyond the observed paired
        # range on any PCA axis is extrapolation, and we do not claim it.
        self._support_lo = pp.min(axis=0)
        self._support_hi = pp.max(axis=0)
        in_box = np.all((pu >= self._support_lo) & (pu <= self._support_hi), axis=1)

        n_bin_total = self.n_bins ** d
        paired_counts = np.bincount(bp, minlength=n_bin_total).astype(np.float64)
        # Count only in-support crops: they are the only ones that can be drawn,
        # so they are the population the per-bin weight must normalise over.
        unpaired_counts = np.bincount(bu[in_box], minlength=n_bin_total).astype(np.float64)

        target = paired_counts / max(paired_counts.sum(), 1.0)
        supported = (paired_counts > 0) & (unpaired_counts > 0)

        w = np.zeros(len(bu), dtype=np.float64)
        ok = supported[bu] & in_box
        w[ok] = target[bu[ok]] / unpaired_counts[bu[ok]]
        total = w.sum()
        if total <= 0:
            raise ValueError(
                "no unpaired crop fell in a bin with paired support; the two pools "
                "do not overlap in environment space at all"
            )
        self.weights = w / total
        self.in_support = ok
        self._bins_unpaired = bu
        self._n_bins_occupied = int(supported.sum())

        self.auc_before = source_classifier_auc(zp, zu, seed=self.seed)
        idx = self.sample(min(len(zp) * 4, max(len(zu), 1)))
        self.auc_after = source_classifier_auc(zp, zu[idx], seed=self.seed)

    def _bin_index(self, p: np.ndarray) -> np.ndarray:
        idx = np.zeros(len(p), dtype=np.int64)
        for j in range(p.shape[1]):
            b = np.searchsorted(self._edges[j], p[:, j], side="right")
            idx = idx * self.n_bins + np.clip(b, 0, self.n_bins - 1)
        return idx

    def sample(self, n: int) -> np.ndarray:
        """Draw ``n`` indices into the unpaired pool, matched to paired support."""
        return self._rng.choice(len(self.weights), size=int(n), p=self.weights)

    def report(self) -> BalanceReport:
        return BalanceReport(
            n_paired=self._n_paired,
            n_unpaired=int(len(self.weights)),
            n_in_support=int(self.in_support.sum()),
            n_rejected=int((~self.in_support).sum()),
            n_bins_occupied=self._n_bins_occupied,
            auc_before=float(self.auc_before),
            auc_after=float(self.auc_after),
            kept_names=self.standardizer.kept_names,
        )


# --------------------------------------------------------------------------- #
# Source classifier diagnostic
# --------------------------------------------------------------------------- #
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC (ties get half credit). ``labels`` in ``{0, 1}``."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def source_classifier_auc(
    paired_z: np.ndarray,
    unpaired_z: np.ndarray,
    seed: int = 0,
    steps: int = 300,
    lr: float = 0.1,
) -> float:
    """ROC-AUC of a logistic classifier predicting paired vs unpaired from descriptors.

    Trained on a class-balanced split and scored on a held-out half, so a model
    that merely memorises cannot inflate the number. ``<= 0.60`` after balancing
    is the pre-registered target.
    """
    paired_z = np.asarray(paired_z, dtype=np.float64)
    unpaired_z = np.asarray(unpaired_z, dtype=np.float64)
    if len(paired_z) < 4 or len(unpaired_z) < 4:
        return float("nan")

    rng = np.random.default_rng(seed)
    x = np.concatenate([paired_z, unpaired_z], axis=0)
    y = np.concatenate([np.ones(len(paired_z)), np.zeros(len(unpaired_z))])
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    cut = len(x) // 2
    x_tr, y_tr, x_te, y_te = x[:cut], y[:cut], x[cut:], y[cut:]
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan")

    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    # Class weights: the pools are deliberately different sizes, and an
    # unweighted fit would just learn the prior.
    w_pos = float(len(yt) / (2 * max(yt.sum().item(), 1.0)))
    w_neg = float(len(yt) / (2 * max((1 - yt).sum().item(), 1.0)))
    weight = torch.where(yt > 0.5, torch.tensor(w_pos), torch.tensor(w_neg))

    model = torch.nn.Linear(xt.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss(weight=weight)
    for _ in range(int(steps)):
        opt.zero_grad()
        loss = lossf(model(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        scores = model(torch.tensor(x_te, dtype=torch.float32)).squeeze(-1).numpy()
    return roc_auc(scores, y_te)
