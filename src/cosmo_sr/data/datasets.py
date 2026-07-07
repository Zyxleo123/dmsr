"""Datasets for training: file-based periodic crops and synthetic SR pairs.

Conventions:
  * Fields on disk are canonical ``(6, Ng, Ng, Ng)`` ``.npy`` arrays.
  * A paired sample provides an LR crop ``(C, crop_lr, ...)`` and an aligned HR
    crop ``(C, crop_lr*scale, ...)``.
"""
from __future__ import annotations

import glob as _glob
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .field_io import load_field
from .crops import periodic_crop


def list_fields(pattern: Optional[str]) -> List[str]:
    """Sorted list of files matching a glob pattern (empty list if ``None``)."""
    if not pattern:
        return []
    return sorted(_glob.glob(pattern))


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


class FieldCropDataset(Dataset):
    """Random periodic crops from a list of field files.

    If ``hr_paths`` is provided it must align index-for-index with ``lr_paths``;
    each item then contains aligned ``lr`` and ``hr`` crops. Otherwise only ``lr``
    is returned (ambient / LR-only data).
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
        self.channels = int(channels)
        self.seed = int(seed)
        self._length = int(length) if length is not None else len(self.lr_paths)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = _rng(self.seed + idx)
        file_idx = int(rng.integers(0, len(self.lr_paths)))
        lr_field = load_field(self.lr_paths[file_idx], mmap=self.mmap)
        if lr_field.shape[0] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {lr_field.shape[0]} in "
                f"{self.lr_paths[file_idx]}"
            )
        Ng_lr = lr_field.shape[1]
        start = tuple(int(rng.integers(0, Ng_lr)) for _ in range(3))
        lr_crop = periodic_crop(lr_field, start, self.crop_lr, pad=0)
        sample = {"lr": torch.from_numpy(np.ascontiguousarray(lr_crop)).float()}

        if self.hr_paths is not None:
            hr_field = load_field(self.hr_paths[file_idx], mmap=self.mmap)
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
            sample["hr"] = torch.from_numpy(np.ascontiguousarray(hr_crop)).float()
        return sample


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
        self._length = int(length) if length is not None else len(self.hr_paths)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import torch.nn.functional as F

        rng = _rng(self.seed + idx)
        file_idx = int(rng.integers(0, len(self.hr_paths)))
        hr_field = load_field(self.hr_paths[file_idx], mmap=self.mmap)
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


def infinite_loader(dataset: Dataset, batch_size: int, shuffle: bool = True, seed: int = 0):
    """Yield batches forever from a dataset (for step-based training)."""
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    while True:
        for batch in loader:
            yield batch
