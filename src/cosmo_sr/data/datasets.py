"""Datasets for training: file-based periodic crops and synthetic SR pairs.

Conventions:
  * Fields on disk are canonical ``(6, Ng, Ng, Ng)`` ``.npy`` arrays.
  * A paired sample provides an LR crop ``(C, crop_lr, ...)`` and an aligned HR
    crop ``(C, crop_lr*scale, ...)``.
"""
from __future__ import annotations

import glob as _glob
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .field_io import load_field
from .crops import make_crop_grid, periodic_crop


def list_fields(pattern: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Sorted list of files matching glob pattern(s) (empty list if falsy).

    ``pattern`` may be a single glob string or a list of glob strings (e.g. to
    combine several ``set[0-9].npy``-style character classes); matches from all
    patterns are deduplicated and sorted together.
    """
    if not pattern:
        return []
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files = set()
    for p in patterns:
        files.update(_glob.glob(p))
    return sorted(files)


def _sample_augment_axes(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a random flip mask + axis permutation (cubic Oh symmetry, order 48)."""
    flip_axes = np.flatnonzero(rng.integers(0, 2, size=3))
    perm_axes = rng.permutation(3)
    return flip_axes, perm_axes


def _augment_field(field: np.ndarray, flip_axes: np.ndarray, perm_axes: np.ndarray) -> np.ndarray:
    """Flip/permute the 3 spatial axes of a ``(C, N, N, N)`` field.

    Mirrors map2map's ``data/fields.py`` flip/perm augmentation for the box's
    cubic point-group symmetry. Channels are assumed to be stacked 3-vectors
    (our canonical layout is disp[0:3] + vel[3:6]), so each triple's
    components are flipped/permuted along with the spatial axes -- otherwise a
    flipped/rotated displacement or velocity field would no longer be a
    physically self-consistent configuration.
    """
    c = field.shape[0]
    field = field.copy()
    if flip_axes.size:
        if c % 3 == 0:
            for t in range(0, c, 3):
                field[t + flip_axes] = -field[t + flip_axes]
        field = np.flip(field, axis=tuple(int(a) + 1 for a in flip_axes))
    field = np.transpose(field, (0,) + tuple(int(a) + 1 for a in perm_axes))
    if c % 3 == 0:
        field = field.copy()
        for t in range(0, c, 3):
            field[t : t + 3] = field[t : t + 3][perm_axes]
    return np.ascontiguousarray(field)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# Nominal per-"epoch" dataset length used when `length` isn't given explicitly.
# __getitem__ in both crop datasets below ignores `idx` and draws an
# independently random file + crop location on every call, so `len(dataset)`
# has no real meaning here -- it only bounds how large `batch_size` can be
# before `infinite_loader`'s `drop_last=True` starts dropping every batch (see
# that function's docstring). Defaulting it to `len(paths)` (as few as 3-15
# for the real paired boxes) silently caps the usable batch_size far below
# what training actually needs.
_DEFAULT_EPOCH_LENGTH = 4096


class FieldCropDataset(Dataset):
    """Random periodic crops from a list of field files.

    If ``hr_paths`` is provided it must align index-for-index with ``lr_paths``;
    each item then contains aligned ``lr`` and ``hr`` crops. Otherwise only ``lr``
    is returned (ambient / LR-only data).

    ``fixed_crops`` (memorization / overfit mode): when set to an int ``N > 0``,
    the dataset draws ``N`` crops once at init (using ``seed``) and forever
    cycles through that closed set. Without this, every ``__getitem__`` draws a
    fresh random crop from the full boxes -- so a "3-box overfit" config with
    ``crop_lr=16`` on a 64^3 LR grid still sees ``~3 * 64^3`` unique examples
    and train loss tracks per-crop ``||A(hr)-lr||^2`` difficulty (often
    ~0.03-0.6) instead of collapsing.
    """

    def __init__(
        self,
        lr_paths: Sequence[str],
        hr_paths: Optional[Sequence[str]] = None,
        crop_lr: int = 16,
        scale_factor: int = 8,
        length: Optional[int] = None,
        seed: int = 0,
        channels: int = 6,
        mmap: bool = False,
        augment: bool = False,
        fixed_crops: Optional[int] = None,
        use_channels: Optional[Sequence[int]] = None,
    ):
        self.lr_paths = list(lr_paths)
        self.hr_paths = list(hr_paths) if hr_paths is not None else None
        self.mmap = bool(mmap)
        if not self.lr_paths:
            raise ValueError("FieldCropDataset requires at least one LR file.")
        if self.hr_paths is not None and len(self.hr_paths) != len(self.lr_paths):
            raise ValueError("lr_paths and hr_paths must have equal length.")
        self.crop_lr = int(crop_lr)
        self.scale_factor = int(scale_factor)
        # `channels` is what must be on disk (6); `use_channels` selects a subset
        # to actually train on. Displacement-only (use_channels [0,1,2]) is the
        # useful case: real-space halo positions are q + Psi, so they depend on
        # displacement alone, and the velocity channels carry ~99% of the MSE
        # while being nearly unpredictable from the HR field.
        self.channels = int(channels)
        self.use_channels = (list(int(c) for c in use_channels)
                             if use_channels is not None else None)
        if self.use_channels is not None:
            if len(self.use_channels) % 3 != 0:
                # _augment_field flips/permutes channels in 3-vector triples;
                # a non-multiple-of-3 selection would silently break that.
                raise ValueError(
                    "use_channels must select whole 3-vectors (len % 3 == 0), "
                    f"got {self.use_channels}"
                )
            bad = [c for c in self.use_channels if not 0 <= c < self.channels]
            if bad:
                raise ValueError(f"use_channels {bad} out of range for {self.channels}")
        self.seed = int(seed)
        self.augment = bool(augment)
        if fixed_crops is not None and int(fixed_crops) <= 0:
            raise ValueError(f"fixed_crops must be a positive int, got {fixed_crops!r}")
        self.fixed_crops = int(fixed_crops) if fixed_crops is not None else None
        # A single mutable generator advanced on every __getitem__ call, rather
        # than re-seeded from `seed + idx`. With few files (as few as 3 for the
        # real paired boxes), `idx` cycles through a tiny fixed range, so
        # reseeding per-idx returned the exact same crop every time that idx
        # came up -- effectively `len(paths)` unique training examples no
        # matter how many steps were trained. `num_workers=0` everywhere in
        # this codebase (see `infinite_loader`), so a single instance-level
        # generator is safe (no multi-process state duplication).
        self._rng = _rng(self.seed)
        # `mmap=True` only makes *opening* a file cheap; every still-unpaged
        # region of a multi-GB file on network storage costs several seconds
        # to fault in (measured ~5-9s per never-before-touched 128^3 crop of
        # a 3.2GB box here), and nothing previously cached that page-in across
        # calls. With as few as 15 files reused for the entire run, reading
        # each one fully into RAM once (~12s/file measured) and slicing crops
        # out of that in-process cache (~0.15s/crop measured, vs 5-9s cold)
        # is a strict win and comfortably fits in memory (15 boxes * 3.2GB).
        self._field_cache: Dict[str, np.ndarray] = {}
        self._fixed_samples: Optional[List[Dict[str, torch.Tensor]]] = None
        if self.fixed_crops is not None:
            # Augmentation would defeat memorization (each draw would still be
            # a new Oh-orbited view). Pre-sample once with aug off.
            was_aug = self.augment
            self.augment = False
            self._fixed_samples = [self._draw_crop() for _ in range(self.fixed_crops)]
            self.augment = was_aug
            if was_aug:
                # Freeze the OH transforms too: one fixed view per memorized crop.
                aug_rng = _rng(self.seed + 17)
                for i, sample in enumerate(self._fixed_samples):
                    flip_axes, perm_axes = _sample_augment_axes(aug_rng)
                    lr_np = sample["lr"].numpy()
                    sample["lr"] = torch.from_numpy(
                        _augment_field(lr_np, flip_axes, perm_axes)
                    ).float()
                    if "hr" in sample:
                        hr_np = sample["hr"].numpy()
                        sample["hr"] = torch.from_numpy(
                            _augment_field(hr_np, flip_axes, perm_axes)
                        ).float()
                    self._fixed_samples[i] = sample
            self._length = int(length) if length is not None else max(
                self.fixed_crops, _DEFAULT_EPOCH_LENGTH
            )
        else:
            self._length = int(length) if length is not None else max(
                len(self.lr_paths), _DEFAULT_EPOCH_LENGTH
            )

    def __len__(self) -> int:
        return self._length

    def _load_cached(self, path: str) -> np.ndarray:
        field = self._field_cache.get(path)
        if field is None:
            field = load_field(path, mmap=self.mmap)
            if field.shape[0] != self.channels:
                raise ValueError(
                    f"Expected {self.channels} channels, got {field.shape[0]} in {path}"
                )
            if self.use_channels is not None:
                # Slice channels HERE, not per-crop. Fancy-indexing the memmap
                # pages in only the channels we asked for, so a displacement-only
                # run reads and caches 3 of every 6 channels -- halving both the
                # network read (1.6 GB vs 3.2 GB per box) and the resident cache
                # (22 GB vs 45 GB across the 14 training boxes).
                field = np.ascontiguousarray(field[self.use_channels])
            elif self.mmap:
                # `load_field(mmap=True)` returns a lazy np.memmap; np.asarray()
                # on it is a no-op (same underlying mmap buffer, still lazy --
                # confirmed via np.shares_memory), so an explicit .copy() is
                # required to actually materialize it into real RAM once.
                field = field.copy()
            self._field_cache[path] = field
        return field

    def warm_cache(self) -> None:
        """Materialise every box into the in-process cache.

        Called before forking DataLoader workers so they inherit the boxes
        copy-on-write rather than each re-reading them from network storage.
        """
        for p in self.lr_paths:
            self._load_cached(p)
        for p in self.hr_paths or []:
            self._load_cached(p)

    def _draw_crop(self) -> Dict[str, torch.Tensor]:
        """Draw one (possibly augmented) crop using ``self._rng``."""
        rng = self._rng
        file_idx = int(rng.integers(0, len(self.lr_paths)))
        lr_field = self._load_cached(self.lr_paths[file_idx])
        Ng_lr = lr_field.shape[1]
        start = tuple(int(rng.integers(0, Ng_lr)) for _ in range(3))
        lr_crop = periodic_crop(lr_field, start, self.crop_lr, pad=0)

        hr_crop = None
        if self.hr_paths is not None:
            hr_field = self._load_cached(self.hr_paths[file_idx])
            Ng_hr = hr_field.shape[1]
            if Ng_hr != Ng_lr * self.scale_factor:
                raise ValueError(
                    f"HR/LR alignment mismatch: Ng_hr={Ng_hr} != "
                    f"Ng_lr*scale={Ng_lr * self.scale_factor}"
                )
            hr_start = tuple(s * self.scale_factor for s in start)
            hr_crop = periodic_crop(
                hr_field, hr_start, self.crop_lr * self.scale_factor, pad=0
            )

        if self.augment:
            flip_axes, perm_axes = _sample_augment_axes(rng)
            lr_crop = _augment_field(lr_crop, flip_axes, perm_axes)
            if hr_crop is not None:
                hr_crop = _augment_field(hr_crop, flip_axes, perm_axes)

        sample = {"lr": torch.from_numpy(np.ascontiguousarray(lr_crop)).float()}
        if hr_crop is not None:
            sample["hr"] = torch.from_numpy(np.ascontiguousarray(hr_crop)).float()
        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._fixed_samples is not None:
            # Clone so downstream code / collate can't mutate the cached tensors.
            sample = self._fixed_samples[int(idx) % len(self._fixed_samples)]
            return {k: v.clone() for k, v in sample.items()}
        return self._draw_crop()


class GridCropDataset(Dataset):
    """Deterministic tiling of paired LR/HR boxes into a fixed crop grid.

    ``FieldCropDataset`` draws a *fresh random* crop on every ``__getitem__``,
    which is right for training and wrong for validation: per-crop MSE against
    the ``A`` baseline swings by ~10x across crops (0.05 to 0.53 on the real
    paired boxes), so a val metric averaged over a couple of random crops is
    dominated by which crops happened to come up, not by the model. Here index
    ``i`` always maps to the same ``(file, start)``, so iterating the dataset
    yields the identical crop set every eval and val numbers are comparable
    across steps *and* across runs.

    ``stride`` defaults to ``crop_lr`` (non-overlapping tiling: a 64^3 LR box at
    ``crop_lr=16`` gives 4^3 = 64 crops per file). ``max_crops`` evenly
    subsamples the grid when the full tiling is more than you want to spend.
    """

    def __init__(
        self,
        lr_paths: Sequence[str],
        hr_paths: Sequence[str],
        crop_lr: int = 16,
        scale_factor: int = 8,
        channels: int = 6,
        mmap: bool = False,
        stride: Optional[int] = None,
        max_crops: Optional[int] = None,
        use_channels: Optional[Sequence[int]] = None,
    ):
        self.lr_paths = list(lr_paths)
        self.hr_paths = list(hr_paths)
        if not self.lr_paths:
            raise ValueError("GridCropDataset requires at least one LR file.")
        if len(self.hr_paths) != len(self.lr_paths):
            raise ValueError("lr_paths and hr_paths must have equal length.")
        self.crop_lr = int(crop_lr)
        self.scale_factor = int(scale_factor)
        self.channels = int(channels)
        self.use_channels = (list(int(c) for c in use_channels)
                             if use_channels is not None else None)
        self.mmap = bool(mmap)
        self.stride = int(stride) if stride is not None else self.crop_lr
        self._field_cache: Dict[str, np.ndarray] = {}

        # Build the (file_idx, start) index once, from the first box's grid size.
        probe = self._load_cached(self.lr_paths[0])
        Ng_lr = probe.shape[1]
        starts = make_crop_grid((Ng_lr, Ng_lr, Ng_lr), self.crop_lr, self.stride)
        self._index: List[Tuple[int, Tuple[int, int, int]]] = [
            (f, s) for f in range(len(self.lr_paths)) for s in starts
        ]
        if max_crops is not None and 0 < int(max_crops) < len(self._index):
            n = int(max_crops)
            step = len(self._index) / n
            self._index = [self._index[int(i * step)] for i in range(n)]

    def __len__(self) -> int:
        return len(self._index)

    def _load_cached(self, path: str) -> np.ndarray:
        field = self._field_cache.get(path)
        if field is None:
            field = load_field(path, mmap=self.mmap)
            if field.shape[0] != self.channels:
                raise ValueError(
                    f"Expected {self.channels} channels, got {field.shape[0]} in {path}"
                )
            if self.use_channels is not None:
                # See FieldCropDataset._load_cached: slice channels at load so a
                # disp-only run pages in and caches half as much.
                field = np.ascontiguousarray(field[self.use_channels])
            elif self.mmap:
                field = field.copy()  # materialize once; see FieldCropDataset
            self._field_cache[path] = field
        return field

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start = self._index[int(idx)]
        lr_field = self._load_cached(self.lr_paths[file_idx])
        hr_field = self._load_cached(self.hr_paths[file_idx])
        Ng_lr, Ng_hr = lr_field.shape[1], hr_field.shape[1]
        if Ng_hr != Ng_lr * self.scale_factor:
            raise ValueError(
                f"HR/LR alignment mismatch: Ng_hr={Ng_hr} != "
                f"Ng_lr*scale={Ng_lr * self.scale_factor}"
            )
        lr_crop = periodic_crop(lr_field, start, self.crop_lr, pad=0)
        hr_start = tuple(s * self.scale_factor for s in start)
        hr_crop = periodic_crop(hr_field, hr_start, self.crop_lr * self.scale_factor, pad=0)
        return {
            "lr": torch.from_numpy(np.ascontiguousarray(lr_crop)).float(),
            "hr": torch.from_numpy(np.ascontiguousarray(hr_crop)).float(),
            # Which simulation box this crop came from. Required for box-level
            # bootstrap resampling: crops within a box share initial conditions and
            # large-scale modes, so resampling crops (rather than boxes) understates
            # the variance badly and manufactures significance.
            "box": torch.tensor(file_idx, dtype=torch.long),
        }


def finite_loader(dataset: Dataset, batch_size: int, num_workers: int = 0):
    """One deterministic pass over ``dataset`` (no shuffle, no drop_last).

    The counterpart to :func:`infinite_loader` for evaluation: every sample is
    visited exactly once, in index order. Safe to parallelise -- ``GridCropDataset``
    has no RNG state (index -> fixed crop), so workers cannot desynchronise it.
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        persistent_workers=int(num_workers) > 0,
    )


def _worker_init(worker_id: int) -> None:
    """Give each DataLoader worker its own crop stream.

    ``FieldCropDataset.__getitem__`` ignores ``idx`` and draws from a mutable
    ``self._rng``. Workers each get a *fork copy* of the dataset, so without
    re-seeding, every worker would replay the identical crop sequence and an
    N-worker batch would be N copies of the same crops -- silently cutting the
    effective batch by N with no error. Re-seed per worker.

    Combined with ``persistent_workers=True`` (see :func:`infinite_loader`) the
    workers are created once and keep advancing their own streams, so crops do
    not repeat across epochs either.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if hasattr(ds, "_rng") and hasattr(ds, "seed"):
        ds._rng = _rng(int(ds.seed) + 10_000 * (int(worker_id) + 1))


class SyntheticSRDataset(Dataset):
    """On-the-fly synthetic SR pairs with ``lr == A(hr)`` exactly.

    A smooth low-frequency HR field is generated deterministically per index, and
    the LR field is its average-pool degradation, so ambient consistency is
    exactly satisfiable and the mapping is learnable by a small model.
    """

    def __init__(
        self,
        num_samples: int = 8,
        channels: int = 6,
        crop_lr: int = 8,
        scale_factor: int = 8,
        seed: int = 0,
        smooth: bool = True,
    ):
        self.num_samples = int(num_samples)
        self.channels = int(channels)
        self.crop_lr = int(crop_lr)
        self.scale_factor = int(scale_factor)
        self.seed = int(seed)
        self.smooth = bool(smooth)

    def __len__(self) -> int:
        return self.num_samples

    def _make_hr(self, idx: int) -> torch.Tensor:
        rng = _rng(self.seed + idx)
        Ng_hr = self.crop_lr * self.scale_factor
        if self.smooth:
            # low-res random field upsampled -> smooth, learnable HR
            coarse = rng.standard_normal(
                (self.channels, self.crop_lr, self.crop_lr, self.crop_lr)
            ).astype(np.float32)
            coarse_t = torch.from_numpy(coarse).unsqueeze(0)
            hr = torch.nn.functional.interpolate(
                coarse_t, scale_factor=self.scale_factor, mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            hr = torch.from_numpy(
                rng.standard_normal(
                    (self.channels, Ng_hr, Ng_hr, Ng_hr)
                ).astype(np.float32)
            )
        return hr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr = self._make_hr(idx)
        lr = torch.nn.functional.avg_pool3d(
            hr.unsqueeze(0), kernel_size=self.scale_factor, stride=self.scale_factor
        ).squeeze(0)
        return {"lr": lr.contiguous(), "hr": hr.contiguous()}


class PyramidCropDataset(Dataset):
    """Random crops from HR boxes, returned as a factor-2 resolution pyramid.

    From each HR field (grid ``Ng_hr``, e.g. 512) a cubic crop of size
    ``crop_hr`` is taken, then repeatedly block-averaged to build coarser levels.
    The returned dict maps a resolution label to the aligned crop::

        {"r512": (C, crop_hr, ...), "r256": (C, crop_hr/2, ...),
         "r128": (C, crop_hr/4, ...), "r64":  (C, crop_hr/8, ...)}

    Adjacent levels ``(rR, r2R)`` form the flow-matching training pairs. The
    absolute resolution labels assume the finest crop sits at ``full_res`` (the
    HR box grid size); coarser labels follow by halving.
    """

    def __init__(
        self,
        hr_paths: Sequence[str],
        crop_hr: int = 64,
        n_levels: int = 4,
        full_res: int = 512,
        length: Optional[int] = None,
        seed: int = 0,
        channels: int = 6,
        mmap: bool = False,
    ):
        self.hr_paths = list(hr_paths)
        if not self.hr_paths:
            raise ValueError("PyramidCropDataset requires at least one HR file.")
        self.crop_hr = int(crop_hr)
        self.n_levels = int(n_levels)
        self.full_res = int(full_res)
        if self.crop_hr % (2 ** (self.n_levels - 1)) != 0:
            raise ValueError(
                f"crop_hr={self.crop_hr} must be divisible by 2**(n_levels-1)="
                f"{2 ** (self.n_levels - 1)}"
            )
        self.channels = int(channels)
        self.seed = int(seed)
        self.mmap = bool(mmap)
        self._length = int(length) if length is not None else max(
            len(self.hr_paths), _DEFAULT_EPOCH_LENGTH
        )
        # See FieldCropDataset for why this is a single mutable generator
        # advanced per call rather than reseeded from `seed + idx`.
        self._rng = _rng(self.seed)
        # See FieldCropDataset._load_cached for why this per-file cache
        # matters: without it, every crop cold-faults a fresh region of a
        # multi-GB network-mounted file (~5-9s) instead of reusing an
        # already-materialized in-RAM copy (~0.15s).
        self._field_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self._length

    def _load_cached(self, path: str) -> np.ndarray:
        field = self._field_cache.get(path)
        if field is None:
            field = load_field(path, mmap=self.mmap)
            if self.mmap:
                # `load_field(mmap=True)` returns a lazy np.memmap; np.asarray()
                # on it is a no-op (same underlying mmap buffer, still lazy --
                # confirmed via np.shares_memory), so an explicit .copy() is
                # required to actually materialize it into real RAM once.
                field = field.copy()
            self._field_cache[path] = field
        return field

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import torch.nn.functional as F

        rng = self._rng
        file_idx = int(rng.integers(0, len(self.hr_paths)))
        hr_field = self._load_cached(self.hr_paths[file_idx])
        if hr_field.shape[0] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {hr_field.shape[0]} in "
                f"{self.hr_paths[file_idx]}"
            )
        Ng = hr_field.shape[1]
        start = tuple(int(rng.integers(0, Ng)) for _ in range(3))
        crop = periodic_crop(hr_field, start, self.crop_hr, pad=0)
        finest = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0)

        levels = [finest]
        for _ in range(self.n_levels - 1):
            levels.append(F.avg_pool3d(levels[-1], kernel_size=2, stride=2))

        sample: Dict[str, torch.Tensor] = {}
        for lvl, tensor in enumerate(levels):
            res = self.full_res // (2 ** lvl)
            sample[f"r{res}"] = tensor.squeeze(0).contiguous()
        return sample


class SyntheticPyramidDataset(Dataset):
    """On-the-fly synthetic resolution pyramids (for smoke tests)."""

    def __init__(
        self,
        num_samples: int = 8,
        channels: int = 6,
        crop_hr: int = 32,
        n_levels: int = 4,
        full_res: int = 512,
        seed: int = 0,
    ):
        self.num_samples = int(num_samples)
        self.channels = int(channels)
        self.crop_hr = int(crop_hr)
        self.n_levels = int(n_levels)
        self.full_res = int(full_res)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import torch.nn.functional as F

        rng = _rng(self.seed + idx)
        coarse = rng.standard_normal(
            (self.channels, self.crop_hr // 4, self.crop_hr // 4, self.crop_hr // 4)
        ).astype(np.float32)
        finest = F.interpolate(
            torch.from_numpy(coarse).unsqueeze(0), size=self.crop_hr, mode="trilinear",
            align_corners=False,
        )
        levels = [finest]
        for _ in range(self.n_levels - 1):
            levels.append(F.avg_pool3d(levels[-1], kernel_size=2, stride=2))
        sample: Dict[str, torch.Tensor] = {}
        for lvl, tensor in enumerate(levels):
            res = self.full_res // (2 ** lvl)
            sample[f"r{res}"] = tensor.squeeze(0).contiguous()
        return sample


def infinite_loader(dataset: Dataset, batch_size: int, shuffle: bool = True, seed: int = 0,
                    num_workers: int = 0):
    """Yield batches forever from a dataset (for step-based training).

    ``drop_last=True`` below silently yields *zero* batches per epoch if
    ``batch_size > len(dataset)`` -- the ``while True`` loop then spins
    forever without ever reaching ``yield``, hanging with no error and no
    log output. Fail loudly instead.

    ``num_workers > 0`` parallelises cropping, which is the training bottleneck
    on the real boxes (a 128^3 HR crop is a ~25 MB strided gather out of a
    512^3 box, ~45 ms, and an optimizer step needs ``batch_size * accum_steps``
    of them). Two things make it safe:

    * :func:`_worker_init` re-seeds each worker's crop RNG, so workers don't all
      replay the same crop sequence;
    * we warm the dataset's box cache in the parent *before* forking, so the
      workers inherit the already-materialised boxes copy-on-write instead of
      each re-reading 22 GB from network storage. They only ever read the boxes,
      so the pages stay shared and RAM does not multiply by ``num_workers``.
    """
    from torch.utils.data import DataLoader

    if batch_size > len(dataset):
        raise ValueError(
            f"batch_size={batch_size} > len(dataset)={len(dataset)}: with "
            "drop_last=True this yields zero batches per epoch and hangs "
            "forever. Increase the dataset's `length` or lower batch_size."
        )

    num_workers = max(0, int(num_workers))
    if num_workers > 0 and hasattr(dataset, "warm_cache"):
        dataset.warm_cache()   # must happen BEFORE fork, or COW buys us nothing

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    while True:
        for batch in loader:
            yield batch
