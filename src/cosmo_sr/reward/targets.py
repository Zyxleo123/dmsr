"""Paired residual targets ``dPsi* = Psi_HR - Psi_base`` and their crop dataset.

The residual target is a *difference of two fields in the same catnorm units*, so
it needs no normalization of its own and ``Psi_base + dPsi* == Psi_HR`` exactly
(up to float32 round-off). The cache therefore stores the **base field**, not the
difference: the base is already needed by every other stage, the HR field is
already on scratch, and materialising ``dPsi*`` as well would add 3.2 GB per box
for no information. :func:`residual_target` reconstructs it on demand, and
:class:`ResidualTargetCache` can still materialise it when a job wants a single
memory-mapped array.

Splits are box-level and come from ``configs/sr2_baseline/freeze.yaml``, so no
crop of a validation or test box can reach a training loader.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.crops import periodic_crop
from ..data.field_io import load_field
from . import paths

__all__ = [
    "PairedResidualCrops",
    "ResidualTargetCache",
    "BoxPaths",
    "residual_stats",
    "residual_target",
    "resolve_boxes",
]


def residual_target(hr, psi_base, channels: Optional[Sequence[int]] = None):
    """``Psi_HR - Psi_base`` on the selected channels (all by default)."""
    if channels is None:
        return hr - psi_base
    idx = list(channels)
    out = np.zeros_like(np.asarray(hr)) if not torch.is_tensor(hr) else torch.zeros_like(hr)
    out[..., idx, :, :, :] = hr[..., idx, :, :, :] - psi_base[..., idx, :, :, :]
    return out


@dataclass
class BoxPaths:
    """Where one box's LR, HR and cached base field live."""

    box: str
    lr: Path
    hr: Path
    base: Optional[Path] = None
    seed: int = 0

    def exists(self) -> bool:
        return self.lr.is_file() and self.hr.is_file() and (
            self.base is None or Path(self.base).is_file()
        )


def resolve_boxes(
    boxes: Sequence[str],
    data_root: str | Path,
    *,
    base_dir: Optional[str | Path] = None,
    base_seed: int = 0,
    base_pattern: str = "{box}_seed{seed}_*.npy",
) -> List[BoxPaths]:
    """Locate LR/HR/base files for named boxes; base is optional."""
    root = Path(data_root)
    base_root = Path(base_dir) if base_dir else paths.SR2_BASE_CACHE()
    out: List[BoxPaths] = []
    for b in boxes:
        hits = sorted(Path(base_root).glob(base_pattern.format(box=b, seed=base_seed)))
        out.append(
            BoxPaths(
                box=b,
                lr=root / "lr" / f"{b}.npy",
                hr=root / "hr" / f"{b}.npy",
                base=hits[0] if hits else None,
                seed=int(base_seed),
            )
        )
    return out


class ResidualTargetCache:
    """Optional materialisation of ``dPsi*`` plus the manifest that tracks it.

    The manifest records the split each box belongs to and a checksum of the
    inputs, so a training job can assert it never opened a validation box.
    """

    def __init__(self, root: Optional[str | Path] = None):
        self.root = Path(root) if root else paths.RESIDUAL_CACHE(create=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, box: str, seed: int) -> Path:
        return self.root / f"resid_{box}_seed{seed}.npy"

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def materialize(self, bp: BoxPaths, split: str, *, chunk: int = 64,
                    overwrite: bool = False) -> Path:
        """Write ``HR - base`` for one box in HR-slab chunks (bounded memory)."""
        if bp.base is None:
            raise ValueError(f"box {bp.box} has no cached SR2 base field")
        out = self.path(bp.box, bp.seed)
        if out.is_file() and not overwrite:
            return out
        hr = load_field(bp.hr, mmap=True)
        base = np.load(bp.base, mmap_mode="r")
        if hr.shape != base.shape:
            raise ValueError(f"shape mismatch HR {hr.shape} vs base {base.shape}")
        tmp = out.with_suffix(".tmp.npy")
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=hr.shape)
        for i in range(0, hr.shape[1], chunk):
            arr[:, i:i + chunk] = np.asarray(hr[:, i:i + chunk], dtype=np.float32) - \
                np.asarray(base[:, i:i + chunk], dtype=np.float32)
        arr.flush()
        del arr
        tmp.replace(out)
        self.record(bp, split, out)
        return out

    def record(self, bp: BoxPaths, split: str, resid_path: Optional[Path]) -> None:
        man = self.load_manifest()
        man["boxes"][bp.box] = {
            "split": str(split),
            "seed": int(bp.seed),
            "lr": str(bp.lr),
            "hr": str(bp.hr),
            "base": str(bp.base) if bp.base else None,
            "residual": str(resid_path) if resid_path else None,
            "key": hashlib.blake2b(
                f"{bp.box}|{bp.seed}|{bp.base}".encode(), digest_size=8
            ).hexdigest(),
        }
        self.manifest_path().write_text(json.dumps(man, indent=2, sort_keys=True))

    def load_manifest(self) -> Dict:
        p = self.manifest_path()
        if p.is_file():
            return json.loads(p.read_text())
        return {"boxes": {}}

    def assert_split(self, boxes: Sequence[str], split: str) -> None:
        """Raise if any box is recorded under a different split."""
        man = self.load_manifest()
        bad = [
            b for b in boxes
            if b in man["boxes"] and man["boxes"][b]["split"] != split
        ]
        if bad:
            raise RuntimeError(
                f"split leak: {bad} are cached as "
                f"{[man['boxes'][b]['split'] for b in bad]} but requested as {split!r}"
            )


class PairedResidualCrops(Dataset):
    """Random periodic HR crops of ``(dPsi*, Psi_base, y_lr)`` from paired boxes.

    Crops are aligned to the LR lattice (``hr_start`` is a multiple of
    ``scale_factor``) so the LR conditioning window covers exactly the same
    region -- otherwise ``U(y_lr)`` and ``Psi_base`` would be offset by a
    sub-LR-cell shift that the model would have to undo.
    """

    def __init__(
        self,
        boxes: Sequence[BoxPaths],
        *,
        crop_hr: int = 64,
        scale_factor: int = 8,
        length: int = 10000,
        seed: int = 0,
        channels: Optional[Sequence[int]] = None,
        use_cached_residual: bool = False,
        redshift: float = 0.0,
    ):
        missing = [b.box for b in boxes if not b.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing LR/HR/base for {missing}; run the SR2 base cache job first"
            )
        if crop_hr % scale_factor:
            raise ValueError(
                f"crop_hr={crop_hr} must be a multiple of scale_factor={scale_factor}"
            )
        self.boxes = list(boxes)
        self.crop_hr = int(crop_hr)
        self.crop_lr = self.crop_hr // int(scale_factor)
        self.scale_factor = int(scale_factor)
        self.length = int(length)
        self.seed = int(seed)
        self.channels = None if channels is None else tuple(int(c) for c in channels)
        self.use_cached_residual = bool(use_cached_residual)
        self.redshift = float(redshift)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.length

    def _open(self, bp: BoxPaths) -> Dict[str, np.ndarray]:
        if bp.box not in self._cache:
            entry = {
                "lr": load_field(bp.lr, mmap=True),
                "hr": load_field(bp.hr, mmap=True),
                "base": np.load(bp.base, mmap_mode="r"),
            }
            if self.use_cached_residual:
                rp = ResidualTargetCache().path(bp.box, bp.seed)
                if rp.is_file():
                    entry["residual"] = np.load(rp, mmap_mode="r")
            self._cache[bp.box] = entry
        return self._cache[bp.box]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        bi = int(rng.integers(len(self.boxes)))
        bp = self.boxes[bi]
        f = self._open(bp)
        ng_hr = int(f["hr"].shape[1])
        ng_lr = ng_hr // self.scale_factor
        lr_start = tuple(int(rng.integers(ng_lr)) for _ in range(3))
        hr_start = tuple(s * self.scale_factor for s in lr_start)

        y_lr = periodic_crop(np.asarray(f["lr"]), lr_start, self.crop_lr, pad=0)
        base = periodic_crop(np.asarray(f["base"]), hr_start, self.crop_hr, pad=0)
        if "residual" in f:
            resid = periodic_crop(np.asarray(f["residual"]), hr_start, self.crop_hr, pad=0)
        else:
            hr = periodic_crop(np.asarray(f["hr"]), hr_start, self.crop_hr, pad=0)
            resid = residual_target(hr, base, self.channels)
        return {
            "residual": torch.from_numpy(np.ascontiguousarray(resid, dtype=np.float32)),
            "psi_base": torch.from_numpy(np.ascontiguousarray(base, dtype=np.float32)),
            "y_lr": torch.from_numpy(np.ascontiguousarray(y_lr, dtype=np.float32)),
            "redshift": torch.tensor(self.redshift, dtype=torch.float32),
            "box_index": torch.tensor(bi, dtype=torch.long),
            "hr_start": torch.tensor(hr_start, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Audit statistics
# --------------------------------------------------------------------------- #
def _radial_power(field: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Isotropic power of one cubic scalar field, in mode units (k = 1 .. N/2)."""
    x = np.asarray(field, dtype=np.float32)
    n = x.shape[-1]
    f = np.fft.rfftn(x, axes=(-3, -2, -1))
    p = (f.real ** 2 + f.imag ** 2) / float(n ** 3)
    kx = np.fft.fftfreq(n) * n
    kz = np.fft.rfftfreq(n) * n
    kmag = np.sqrt(
        kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2
    )
    kmax = n / 2.0
    edges = np.linspace(0.0, kmax, n_bins + 1)
    which = np.clip(np.digitize(kmag.ravel(), edges) - 1, 0, n_bins - 1)
    sums = np.bincount(which, weights=p.ravel(), minlength=n_bins)
    cnt = np.bincount(which, minlength=n_bins).astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, sums / np.maximum(cnt, 1.0)


def residual_stats(
    hr: np.ndarray,
    base: np.ndarray,
    *,
    scale_factor: int = 8,
    n_bins: int = 24,
    channels: Sequence[int] = (0, 1, 2),
    max_block: int = 64,
) -> Dict[str, object]:
    """Everything the residual-target audit reports, for one box.

    ``channels`` selects which channels get spectra (displacement by default);
    means, stds and RMS are reported for all channels regardless.
    """
    hr = np.asarray(hr)
    base = np.asarray(base)
    if hr.shape != base.shape:
        raise ValueError(f"HR {hr.shape} != base {base.shape}")
    c, n = int(hr.shape[0]), int(hr.shape[1])

    # Streaming moments so a 512^3 x 6 pair never lands in memory twice.
    acc = {k: np.zeros(c, dtype=np.float64) for k in ("r_sum", "r_sq", "hr_sq", "b_sq")}
    rmax = np.zeros(c, dtype=np.float64)
    max_per_slab: List[float] = []
    count = 0
    for i in range(0, n, max_block):
        h = np.asarray(hr[:, i:i + max_block], dtype=np.float32)
        b = np.asarray(base[:, i:i + max_block], dtype=np.float32)
        r = h - b
        acc["r_sum"] += r.sum(axis=(1, 2, 3))
        acc["r_sq"] += (r.astype(np.float64) ** 2).sum(axis=(1, 2, 3))
        acc["hr_sq"] += (h.astype(np.float64) ** 2).sum(axis=(1, 2, 3))
        acc["b_sq"] += (b.astype(np.float64) ** 2).sum(axis=(1, 2, 3))
        m = np.abs(r).max(axis=(1, 2, 3))
        rmax = np.maximum(rmax, m)
        max_per_slab.append(float(np.abs(r[0:3]).max()))
        count += r[0].size

    mean = acc["r_sum"] / count
    var = acc["r_sq"] / count - mean ** 2
    out: Dict[str, object] = {
        "n_grid": n,
        "residual_mean": mean.tolist(),
        "residual_std": np.sqrt(np.maximum(var, 0.0)).tolist(),
        "residual_rms": np.sqrt(acc["r_sq"] / count).tolist(),
        "hr_rms": np.sqrt(acc["hr_sq"] / count).tolist(),
        "base_rms": np.sqrt(acc["b_sq"] / count).tolist(),
        "residual_rms_over_hr": (
            np.sqrt(acc["r_sq"] / count) / np.maximum(np.sqrt(acc["hr_sq"] / count), 1e-30)
        ).tolist(),
        "residual_rms_over_base": (
            np.sqrt(acc["r_sq"] / count) / np.maximum(np.sqrt(acc["b_sq"] / count), 1e-30)
        ).tolist(),
        "residual_absmax": rmax.tolist(),
        "residual_absmax_slab_distribution": {
            "p50": float(np.percentile(max_per_slab, 50)),
            "p90": float(np.percentile(max_per_slab, 90)),
            "p99": float(np.percentile(max_per_slab, 99)),
            "max": float(np.max(max_per_slab)),
        },
    }

    k_nyq_lr = n / (2.0 * float(scale_factor))
    spectra: Dict[str, object] = {}
    for ch in channels:
        h = np.asarray(hr[ch], dtype=np.float32)
        b = np.asarray(base[ch], dtype=np.float32)
        kc, p_hr = _radial_power(h, n_bins)
        _, p_base = _radial_power(b, n_bins)
        _, p_res = _radial_power(h - b, n_bins)
        low = kc < 0.5 * k_nyq_lr
        trans = (kc >= 0.5 * k_nyq_lr) & (kc < 2.0 * k_nyq_lr)
        high = kc >= 2.0 * k_nyq_lr
        # Power is per-mode; weight by shell volume ~ k^2 for a fair fraction.
        w = kc ** 2
        tot = float((p_res * w).sum())
        below = float((p_res * w)[kc < k_nyq_lr].sum())
        spectra[f"ch{ch}"] = {
            "k": kc.tolist(),
            "P_hr": p_hr.tolist(),
            "P_base": p_base.tolist(),
            "P_residual": p_res.tolist(),
            "residual_power_low_k": float(p_res[low].mean()) if low.any() else float("nan"),
            "residual_power_transition_k": float(p_res[trans].mean()) if trans.any() else float("nan"),
            "residual_power_high_k": float(p_res[high].mean()) if high.any() else float("nan"),
            "frac_residual_power_below_lr_nyquist": below / max(tot, 1e-30),
        }
    out["k_nyquist_lr_modes"] = float(k_nyq_lr)
    out["spectra"] = spectra
    return out
