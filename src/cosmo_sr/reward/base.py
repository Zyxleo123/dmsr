"""Frozen SR2 baseline interface and residual composition.

    Psi_base = G_SR2(y, z)              (frozen, deterministic given a seed)
    Psi_hat  = Psi_base + a * dPsi      (a = residual_scale)

Three properties matter and are enforced here rather than left to the caller:

* **Frozen.** Every SR2 parameter has ``requires_grad = False`` and the module is
  in ``eval()``; :meth:`FrozenSR2Base.assert_frozen` re-checks it, and the base
  field is produced under ``torch.no_grad``.
* **No null-space projection.** ``Psi_hat`` is the plain sum. The exact
  projection ``P_null`` is *not* applied: enforcing ``A(Psi_hat) = y`` cell by
  cell fights the phase-coherent collapse that produces halos, which earlier
  runs here showed damages density and halo statistics. LR consistency is
  instead a *measured constraint* (:mod:`cosmo_sr.reward.constraints`), free to
  be violated slightly if the catalog improves.
* **Normalization untouched.** The composition is a sum in catnorm units, so
  ``disnorm`` / ``velnorm`` and everything downstream (CIC, particles, Rockstar)
  see exactly the units they already expect.

Caching: a 512^3 six-channel base field is 3.2 GB, and every candidate for a box
shares it, so it is written once to ``$ZFS/dmsr_reward/cache/sr2_base/`` and
memory-mapped thereafter.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..data.crops import periodic_crop
from ..tts.sampling import tile_noise, tile_starts
from ..tts.srs_noise import ControlledG, load_controlled_generator
from . import paths

__all__ = ["BaseFieldSpec", "FrozenSR2Base", "ResidualComposer", "compose",
           "find_base_field", "lr_sha"]

# Bump when anything that changes the produced field changes: the freeze
# convention, the tiling/trim, the noise construction. It is part of the cache
# key, so an old field cannot be silently reused under new code.
BASE_FIELD_CODE_VERSION = "1"


def lr_sha(lr_field: np.ndarray) -> str:
    """Short checksum of an LR conditioning field.

    In the cache key because the key's job is to identify *the field that was
    produced*, and that is a function of the LR input as much as of the weights.
    Without it, regenerating ``lr/set8.npy`` -- a preprocessing fix, a different
    degrader -- leaves a stale 3.2 GB base field sitting at exactly the filename
    the new one would claim, and every downstream residual is computed against
    a baseline that no longer corresponds to its own conditioning.
    """
    a = np.ascontiguousarray(np.asarray(lr_field), dtype=np.float32)
    return hashlib.blake2b(a.tobytes(), digest_size=8).hexdigest()


def find_base_field(box: str, seed: int, root=None, *, pattern: Optional[str] = None):
    """The one cached base field for ``(box, seed)``; refuse if there are several.

    Every consumer used to ``sorted(glob(...))[0]``, which silently picks the
    lexicographically first key when two differ -- different weights, different
    LR field, different code version. Two candidates mean the caller has to say
    which; picking one is how a run mixes baselines.
    """
    root = Path(root) if root else paths.SR2_BASE_CACHE()
    pat = pattern or f"{box}_seed{int(seed)}_*.npy"
    hits = sorted(p for p in Path(root).glob(pat) if not p.name.endswith(".tmp.npy"))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(
            f"{len(hits)} cached base fields for {box} seed {seed} in {root}:\n  "
            + "\n  ".join(h.name for h in hits)
            + "\nThey differ in weights, LR field or code version (the hash in "
              "the filename). Delete the stale ones or name the file explicitly; "
              "picking the first would mix baselines across candidates."
        )
    return hits[0]


def compose(
    psi_base: torch.Tensor | np.ndarray,
    residual: torch.Tensor | np.ndarray | None,
    residual_scale: float = 1.0,
    channels: Optional[Sequence[int]] = None,
):
    """``Psi_base + a * dPsi``, with ``a = 0`` returning ``Psi_base`` untouched.

    The zero case short-circuits rather than adding ``0 * dPsi``: "residual
    disabled" must mean the frozen output bit for bit, including when the
    residual model emits a NaN.
    """
    if residual is None or float(residual_scale) == 0.0:
        return psi_base
    a = float(residual_scale)
    if channels is None:
        return psi_base + a * residual
    out = psi_base.clone() if torch.is_tensor(psi_base) else psi_base.copy()
    idx = list(channels)
    if torch.is_tensor(out):
        out[..., idx, :, :, :] = out[..., idx, :, :, :] + a * residual[..., idx, :, :, :]
    else:
        out[..., idx, :, :, :] = out[..., idx, :, :, :] + a * residual[..., idx, :, :, :]
    return out


@dataclass(frozen=True)
class BaseFieldSpec:
    """Everything that determines one frozen SR2 field, hashed into its filename."""

    box: str
    seed: int
    model_sha: str
    scale_factor: int = 8
    nsplit: int = 8
    pad: int = 3
    noise_mode: str = "per_tile"
    redshift: float = 0.0
    # Identity of the LR conditioning field and of the code that consumed it.
    # Both are part of what determines the output, so both are in the key.
    lr_sha: str = ""
    code_version: str = BASE_FIELD_CODE_VERSION

    def key(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()

    def filename(self) -> str:
        return f"{self.box}_seed{self.seed}_{self.key()}.npy"


class FrozenSR2Base(nn.Module):
    """The frozen SR2 generator, with caching and sub-region generation."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        scale_factor: int = 8,
        nsplit: int = 8,
        pad: int = 3,
        noise_mode: str = "per_tile",
        redshift: float = 0.0,
        device=None,
        cache_dir: Optional[str | os.PathLike] = None,
        generator: Optional[ControlledG] = None,
    ):
        super().__init__()
        self.model_path = str(model_path)
        self.scale_factor = int(scale_factor)
        self.nsplit = int(nsplit)
        self.pad = int(pad)
        self.noise_mode = str(noise_mode)
        self.redshift = float(redshift)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.generator = generator if generator is not None else load_controlled_generator(
            self.model_path, scale_factor=self.scale_factor, device=self._device
        )
        self.freeze()

    # ---------------------------------------------------------------- frozen
    def freeze(self) -> "FrozenSR2Base":
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad_(False)
        for b in self.generator.buffers():
            b.requires_grad_(False)
        return self

    def assert_frozen(self) -> None:
        bad = [n for n, p in self.generator.named_parameters() if p.requires_grad]
        if bad:
            raise RuntimeError(f"SR2 parameters are trainable: {bad[:5]}")
        if self.generator.training:
            raise RuntimeError("SR2 generator is in train() mode")

    def train(self, mode: bool = True) -> "FrozenSR2Base":   # noqa: D102
        super().train(mode)
        self.generator.eval()
        return self

    def model_sha(self) -> str:
        h = hashlib.sha256()
        with open(self.model_path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:16]

    # ------------------------------------------------------------ generation
    def spec(self, box: str, seed: int, lr_field: Optional[np.ndarray] = None,
             *, lr_sha_hex: str = "") -> BaseFieldSpec:
        if not lr_sha_hex and lr_field is not None:
            lr_sha_hex = lr_sha(lr_field)
        return BaseFieldSpec(
            box=str(box), seed=int(seed), model_sha=self.model_sha(),
            scale_factor=self.scale_factor, nsplit=self.nsplit, pad=self.pad,
            noise_mode=self.noise_mode, redshift=self.redshift,
            lr_sha=str(lr_sha_hex),
        )

    def cache_path(self, box: str, seed: int, lr_field: Optional[np.ndarray] = None,
                   *, lr_sha_hex: str = "") -> Path:
        root = self.cache_dir or paths.SR2_BASE_CACHE(create=True)
        Path(root).mkdir(parents=True, exist_ok=True)
        return Path(root) / self.spec(
            box, seed, lr_field, lr_sha_hex=lr_sha_hex
        ).filename()

    @torch.no_grad()
    def base_region(
        self,
        lr_field: np.ndarray,
        seed: int,
        tile_ids: Optional[Sequence[Tuple[int, int, int]]] = None,
        out_dtype=np.float32,
    ) -> Dict[Tuple[int, int, int], np.ndarray]:
        """SR2 output for a subset of LR tiles, keyed by the tile's LR start.

        Every tile is produced exactly as ``super_resolve_srs_seeded`` would
        (periodic pad, per-tile hashed noise, centre trim), so a region assembled
        from tiles is bit-identical to the corresponding slice of the full box.
        """
        self.assert_frozen()
        lr = np.ascontiguousarray(np.asarray(lr_field), dtype=np.float32)
        ng = int(lr.shape[1])
        chunk = ng // self.nsplit
        lr_size = chunk + 2 * self.pad
        tgt = chunk * self.scale_factor
        starts = list(tile_ids) if tile_ids is not None else tile_starts(ng, self.nsplit)
        out: Dict[Tuple[int, int, int], np.ndarray] = {}
        for start in starts:
            crop = periodic_crop(lr, tuple(start), chunk, pad=self.pad)
            t = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0).to(self._device)
            noise = tile_noise(
                int(seed), tuple(start), lr_size, self.scale_factor, self._device,
                pad=self.pad, mode=self.noise_mode,
            )
            sr = self.generator(t, noise=noise).squeeze(0)
            w = sr.shape[-1] - tgt
            hw = w // 2
            sr = sr[..., hw:hw + tgt, hw:hw + tgt, hw:hw + tgt]
            out[tuple(int(s) for s in start)] = sr.to("cpu").numpy().astype(out_dtype, copy=False)
        return out

    @torch.no_grad()
    def base_field(
        self,
        lr_field: np.ndarray,
        seed: int,
        *,
        box: str = "",
        use_cache: bool = True,
        mmap: bool = True,
        progress: bool = False,
        overwrite: bool = False,
    ) -> np.ndarray:
        """The full-box frozen SR2 field, cached on scratch.

        Deterministic in ``(lr_field, seed)``: identical inputs always return an
        identical array, cached or not.

        ``overwrite`` regenerates even when the cache hits. It has to be handled
        *here*: a caller that only refused to skip the call still landed on the
        ``path.is_file()`` return below and got the stale field back, so
        ``--overwrite`` was a no-op wherever it was passed.
        """
        self.assert_frozen()
        path = self.cache_path(box, seed, lr_field) if (use_cache and box) else None
        if path is not None and path.is_file() and not overwrite:
            return np.load(path, mmap_mode="r" if mmap else None)

        from ..tts.sampling import super_resolve_srs_seeded

        field = super_resolve_srs_seeded(
            self.generator, lr_field, int(seed), scale_factor=self.scale_factor,
            nsplit=self.nsplit, pad=self.pad, device=self._device,
            noise_mode=self.noise_mode, progress=progress,
        )
        if path is not None:
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, field)
            os.replace(tmp, path)
            meta = path.with_suffix(".json")
            meta.write_text(json.dumps(
                self.spec(box, seed, lr_field).__dict__, indent=2))
            if mmap:
                return np.load(path, mmap_mode="r")
        return field


class ResidualComposer(nn.Module):
    """Frozen SR2 plus a trainable stochastic residual.

    Holds no SR2 weights of its own -- ``base`` is optional so training jobs that
    consume cached base fields never pay for loading the generator.
    """

    def __init__(
        self,
        residual: Optional[nn.Module] = None,
        base: Optional[FrozenSR2Base] = None,
        residual_scale: float = 1.0,
        residual_channels: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.residual = residual
        self.base = base
        self.residual_scale = float(residual_scale)
        self.residual_channels = (
            None if residual_channels is None else tuple(int(c) for c in residual_channels)
        )
        if base is not None:
            base.assert_frozen()

    @property
    def enabled(self) -> bool:
        return self.residual is not None and self.residual_scale != 0.0

    def compose(self, psi_base, residual):
        return compose(psi_base, residual, self.residual_scale, self.residual_channels)

    def trainable_parameters(self):
        """Only the residual model's parameters; SR2 is never in the optimizer."""
        return [] if self.residual is None else list(self.residual.parameters())

    @torch.no_grad()
    def sample_residual(
        self,
        psi_base: torch.Tensor,
        y_lr: torch.Tensor,
        *,
        seed: int,
        cfg=None,
        redshift: float = 0.0,
        device=None,
    ) -> torch.Tensor:
        """One residual realisation for a fixed ``psi_base``; reproducible in ``seed``.

        Calling this repeatedly with the same ``psi_base`` and different ``seed``
        is the "multiple residual samples for one fixed SR2 output" mode the
        best-of-K oracle needs.
        """
        from .diffusion import DiffusionConfig, unwhiten

        if self.residual is None:
            return torch.zeros_like(psi_base)
        cfg = cfg or DiffusionConfig()
        device = device or psi_base.device
        # The initial noise is always drawn on the CPU so a realisation depends on
        # the seed alone and not on which device produced it -- the same argument
        # as tts.sampling._randn.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(tuple(psi_base.shape), generator=gen, dtype=torch.float32)
        u = _ddim_from_noise(
            self.residual, noise.to(device), cfg,
            cond={"y_lr": y_lr.to(device), "psi_base": psi_base.to(device),
                  "redshift": redshift},
            seed=int(seed),
        )
        return unwhiten(u, self.residual.sigma_res).to(psi_base.device)


@torch.no_grad()
def _ddim_from_noise(model, u: torch.Tensor, cfg, cond: dict, seed: int,
                     stats: Optional[dict] = None) -> torch.Tensor:
    """DDIM starting from a caller-supplied noise tensor (device-independent seed).

    ``stats``, if given, collects the ``x0_clip`` hit fraction per step. Gate A
    criterion 5 is "the clip, not the model, must not be what sets the
    amplitude", and that is unanswerable unless the fraction is recorded while
    sampling.
    """
    from ..models.operator_denoiser import CosineSchedule

    sched = CosineSchedule()
    ts = torch.linspace(cfg.t_max, cfg.t_min, int(cfg.n_steps) + 1, device=u.device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)
    clipped: list = []
    for i in range(int(cfg.n_steps)):
        t_cur = ts[i].expand(u.shape[0])
        t_nxt = ts[i + 1].expand(u.shape[0])
        a_c, s_c = sched.broadcast(t_cur, ndim=5)
        a_n, s_n = sched.broadcast(t_nxt, ndim=5)
        eps_hat = model(u, t_cur, **cond)
        u0_hat = (u - s_c * eps_hat) / a_c.clamp_min(1e-8)
        if getattr(cfg, "x0_clip", 0.0) > 0:
            if stats is not None:
                clipped.append(float((u0_hat.abs() >= cfg.x0_clip).float().mean()))
            u0_hat = u0_hat.clamp(-cfg.x0_clip, cfg.x0_clip)
        if cfg.churn <= 0.0:
            u = a_n * u0_hat + s_n * eps_hat
        else:
            keep = (max(0.0, 1.0 - cfg.churn ** 2)) ** 0.5
            fresh = torch.randn(u.shape, generator=gen, dtype=torch.float32).to(u.device)
            u = a_n * u0_hat + s_n * (keep * eps_hat + cfg.churn * fresh)
    if stats is not None:
        stats["x0_clip_fraction_per_step"] = clipped
        stats["x0_clip_fraction_max"] = max(clipped) if clipped else 0.0
        stats["x0_clip_fraction_last"] = clipped[-1] if clipped else 0.0
    return u
