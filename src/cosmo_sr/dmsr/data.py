"""Datasets and box-level splits for the DMSR stage.

**Splits are by simulation box, never by spatial crop.** Crops from one box are
strongly correlated (they share the same initial conditions, the same large-scale
modes and the same realisation of every statistic we report), so a crop-level
split would leak the test set into training and make every confidence interval
meaningless. :func:`resolve_split` asserts the three file sets are disjoint and
writes the resolved manifest into the run directory.

Paired and LR-only crops both reuse the existing, tested
:class:`~cosmo_sr.data.datasets.FieldCropDataset` (with and without ``hr_paths``),
so the on-disk format, the channel-slicing behaviour and the ``disp+vel``
augmentation conventions are unchanged.

The one genuinely new piece is :class:`LRCropPool`. Environment-balanced sampling
needs descriptors for a *fixed, enumerable* set of candidate crops, but
``FieldCropDataset`` draws a fresh random crop on every ``__getitem__``. The pool
therefore materialises a finite list of ``(box, start)`` locations up front,
computes their descriptors once, and lets the sampler weight them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.crops import periodic_crop
from ..data.datasets import FieldCropDataset, GridCropDataset, list_fields
from ..data.field_io import load_field
from .env import (
    DescriptorStandardizer,
    EnvironmentBalancedSampler,
    environment_descriptors,
)


# --------------------------------------------------------------------------- #
# Box-level split
# --------------------------------------------------------------------------- #
@dataclass
class BoxSplit:
    """Resolved simulation-box split. All paths are absolute and sorted."""

    train_lr: List[str] = _field(default_factory=list)
    train_hr: List[str] = _field(default_factory=list)
    val_lr: List[str] = _field(default_factory=list)
    val_hr: List[str] = _field(default_factory=list)
    test_lr: List[str] = _field(default_factory=list)
    test_hr: List[str] = _field(default_factory=list)
    lr_only: List[str] = _field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return dict(self.__dict__)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _stem_set(paths: Sequence[str]) -> set:
    return {Path(p).stem for p in paths}


def resolve_split(data_cfg: Dict) -> BoxSplit:
    """Resolve globs into a :class:`BoxSplit`, asserting box-level disjointness.

    Required keys: ``train_hr_glob``/``train_lr_glob``, ``val_*``, ``test_*``.
    Optional: ``lr_only_glob`` (the unpaired boxes used by Stage D and by SSL).
    """
    split = BoxSplit(
        train_lr=list_fields(data_cfg.get("train_lr_glob")),
        train_hr=list_fields(data_cfg.get("train_hr_glob")),
        val_lr=list_fields(data_cfg.get("val_lr_glob")),
        val_hr=list_fields(data_cfg.get("val_hr_glob")),
        test_lr=list_fields(data_cfg.get("test_lr_glob")),
        test_hr=list_fields(data_cfg.get("test_hr_glob")),
        lr_only=list_fields(data_cfg.get("lr_only_glob")),
    )
    for name in ("train", "val", "test"):
        lr = getattr(split, f"{name}_lr")
        hr = getattr(split, f"{name}_hr")
        if len(lr) != len(hr):
            raise ValueError(
                f"{name}: {len(lr)} LR boxes but {len(hr)} HR boxes; the globs must "
                "pair index-for-index"
            )
        if _stem_set(lr) != _stem_set(hr):
            raise ValueError(f"{name}: LR and HR box names differ ({_stem_set(lr) ^ _stem_set(hr)})")
    if not split.train_hr:
        raise ValueError("no training boxes matched train_hr_glob")

    tr, va, te = (_stem_set(split.train_hr), _stem_set(split.val_hr), _stem_set(split.test_hr))
    for a, b, an, bn in ((tr, va, "train", "val"), (tr, te, "train", "test"), (va, te, "val", "test")):
        if a & b:
            raise ValueError(f"{an}/{bn} boxes overlap: {sorted(a & b)} -- splits must be by box")

    # LR-only boxes must not include held-out paired boxes, even though LR
    # fields exist for them: the encoder and the critic would then have seen the
    # evaluation conditions.
    held_out = va | te
    leaked = [p for p in split.lr_only if Path(p).stem in held_out]
    if leaked:
        raise ValueError(
            f"lr_only_glob matches held-out boxes {sorted(_stem_set(leaked))}; exclude "
            "every validation and test box from the unpaired pool"
        )
    return split


# --------------------------------------------------------------------------- #
# Paired / LR-only datasets
# --------------------------------------------------------------------------- #
def build_paired_dataset(
    split: BoxSplit,
    crop_lr: int,
    scale_factor: int,
    seed: int = 0,
    augment: bool = True,
    channels: int = 6,
    use_channels: Optional[Sequence[int]] = None,
    mmap: bool = True,
    length: Optional[int] = None,
) -> FieldCropDataset:
    """Random paired crops from the training boxes."""
    return FieldCropDataset(
        lr_paths=split.train_lr,
        hr_paths=split.train_hr,
        crop_lr=crop_lr,
        scale_factor=scale_factor,
        seed=seed,
        augment=augment,
        channels=channels,
        use_channels=use_channels,
        mmap=mmap,
        length=length,
    )


def box_names(split, which: str = "test"):
    """Box stems in the order ``GridCropDataset`` indexes them (sorted paths)."""
    from pathlib import Path
    return [Path(p).stem for p in getattr(split, f"{which}_hr")]


def build_val_dataset(
    split: BoxSplit,
    crop_lr: int,
    scale_factor: int,
    channels: int = 6,
    use_channels: Optional[Sequence[int]] = None,
    mmap: bool = True,
    max_crops: Optional[int] = 32,
    which: str = "val",
) -> GridCropDataset:
    """Deterministic tiling of the held-out boxes (same crops at every eval)."""
    lr = getattr(split, f"{which}_lr")
    hr = getattr(split, f"{which}_hr")
    if not lr:
        raise ValueError(f"no {which} boxes in the split")
    return GridCropDataset(
        lr_paths=lr,
        hr_paths=hr,
        crop_lr=crop_lr,
        scale_factor=scale_factor,
        channels=channels,
        use_channels=use_channels,
        mmap=mmap,
        max_crops=max_crops,
    )


# --------------------------------------------------------------------------- #
# Enumerable LR crop pool (for environment balancing)
# --------------------------------------------------------------------------- #
class LRCropPool:
    """A fixed pool of LR crops with precomputed environment descriptors.

    Parameters
    ----------
    paths:
        LR box files to draw from.
    n_crops:
        Pool size. Crop locations are drawn once with ``seed`` and then frozen,
        so descriptors, sampler weights and the source-classifier AUC all refer
        to the same concrete set of crops.
    """

    def __init__(
        self,
        paths: Sequence[str],
        crop_lr: int,
        n_crops: int = 2048,
        seed: int = 0,
        channels: int = 6,
        use_channels: Optional[Sequence[int]] = None,
        mmap: bool = True,
        descriptor_kwargs: Optional[Dict] = None,
    ):
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("LRCropPool requires at least one LR box")
        self.crop_lr = int(crop_lr)
        self.channels = int(channels)
        self.use_channels = list(use_channels) if use_channels is not None else None
        self.mmap = bool(mmap)
        self.seed = int(seed)
        self._cache: Dict[str, np.ndarray] = {}
        self._descriptor_kwargs = dict(descriptor_kwargs or {})

        rng = np.random.default_rng(self.seed)
        probe = self._load(self.paths[0])
        ng = probe.shape[1]
        self.index: List[Tuple[int, Tuple[int, int, int]]] = [
            (
                int(rng.integers(0, len(self.paths))),
                tuple(int(v) for v in rng.integers(0, ng, size=3)),
            )
            for _ in range(int(n_crops))
        ]
        self._descriptors: Optional[np.ndarray] = None

    def _load(self, path: str) -> np.ndarray:
        f = self._cache.get(path)
        if f is None:
            f = load_field(path, mmap=self.mmap)
            if f.shape[0] != self.channels:
                raise ValueError(f"expected {self.channels} channels, got {f.shape[0]} in {path}")
            if self.use_channels is not None:
                f = np.ascontiguousarray(f[self.use_channels])
            elif self.mmap:
                f = f.copy()
            self._cache[path] = f
        return f

    def __len__(self) -> int:
        return len(self.index)

    def crop(self, i: int) -> torch.Tensor:
        file_idx, start = self.index[int(i)]
        arr = periodic_crop(self._load(self.paths[file_idx]), start, self.crop_lr, pad=0)
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    def descriptors(self, batch_size: int = 64, device=None) -> np.ndarray:
        """Environment descriptors for every crop in the pool (computed once)."""
        if self._descriptors is not None:
            return self._descriptors
        out = []
        for lo in range(0, len(self), int(batch_size)):
            batch = torch.stack([self.crop(i) for i in range(lo, min(lo + batch_size, len(self)))])
            if device is not None:
                batch = batch.to(device)
            out.append(environment_descriptors(batch, **self._descriptor_kwargs).cpu().numpy())
        self._descriptors = np.concatenate(out, axis=0)
        return self._descriptors


class BalancedLRDataset(Dataset):
    """Draws LR-only crops from a pool via an :class:`EnvironmentBalancedSampler`.

    ``__getitem__`` ignores its index and returns a sampler-drawn crop, matching
    the streaming behaviour of ``FieldCropDataset`` so the two can be used
    interchangeably in the training loop.
    """

    def __init__(
        self,
        pool: LRCropPool,
        sampler: EnvironmentBalancedSampler,
        length: int = 4096,
        seed: int = 0,
    ):
        self.pool = pool
        self.sampler = sampler
        self._length = int(length)
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(self.sampler.sample(1)[0])
        return {"lr": self.pool.crop(i)}


class UnbalancedLRDataset(Dataset):
    """Uniform sampling from the same pool -- the control for the balancing test."""

    def __init__(self, pool: LRCropPool, length: int = 4096, seed: int = 0):
        self.pool = pool
        self._length = int(length)
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"lr": self.pool.crop(int(self._rng.integers(0, len(self.pool))))}


def build_balanced_sampler(
    paired_pool: LRCropPool,
    unpaired_pool: LRCropPool,
    n_dims: int = 2,
    n_bins: int = 8,
    seed: int = 0,
) -> Tuple[EnvironmentBalancedSampler, DescriptorStandardizer]:
    """Fit the standardizer on **paired training** descriptors and build the sampler."""
    paired_desc = paired_pool.descriptors()
    unpaired_desc = unpaired_pool.descriptors()
    standardizer = DescriptorStandardizer.fit(paired_desc)
    sampler = EnvironmentBalancedSampler(
        paired_desc=paired_desc,
        unpaired_desc=unpaired_desc,
        standardizer=standardizer,
        n_dims=n_dims,
        n_bins=n_bins,
        seed=seed,
    )
    return sampler, standardizer
