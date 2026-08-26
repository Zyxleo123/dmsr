"""Data + normalization for the moment-constrained substructure module (step 5).

This is the data side of pilot step 5 of ``docs/sr2_substructure_module.md``
(Option B, native Lagrangian tiles) with the high-pass replaced by the moment
target of ``docs/sr2_moment_constraint.md``.

The flow generates the projected residual ``d`` (6 channels: disp+vel) on one
``64^3`` Lagrangian tile, conditioned on the frozen SR2 tile and the LR host
channels. The whole-box target ``x1 = Pi(Psi_HR - Psi_SR2)`` is precomputed by
``scripts/features/build_moment_target.py``; here it is sliced into tiles.

Two whole-box arrays are held in RAM (fp16, ~1.6 GB each): the frozen SR2 field
and the projected target. A tile is one ``[i*64:(i+1)*64]`` block on each axis --
the same raster ``build_moment_target.generate_sr2_box`` scatters tiles into, so
the SR2 box, the target box and the LR host stack all address the same physical
region by ``(ix,iy,iz)`` with no tile-id bookkeeping.

Normalization (``docs/sr2_substructure_module.md`` section 4.2). A native tile
still spans the full dynamic range -- ~6.7 Mpc/h across a cluster patch against
~1 in a void -- so a cluster would dominate an unweighted loss and section 1.1's
disease returns. We normalize the *values* pointwise by a locally-estimated
scale ``s`` (smoothed rms of the SR2 field, derived from the field itself, never
from a host catalog, so it is defined everywhere and available at inference by
construction). Both the conditioning and the target are divided by ``s``; the
flow therefore works in equalized units and the loss needs no extra ``1/s^2``
weight. ``s`` is recomputed the same way at inference, so nothing is saved.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..features.lagrangian_host import LagrangianHostFeatures

# One SR2 output tile: nsplit=8 over ng_hr=512 -> 64^3 HR, 8^3 LR children.
TILE_HR = 64
N_PER_AXIS = 8
N_TILES = N_PER_AXIS ** 3
NG_HR = 512
NG_LR = 64
UPSAMPLE = NG_HR // NG_LR          # 8

# Conditioning channel count from LagrangianHostFeatures.stack_lr(): host_member,
# log_host_mass, dq_over_rl(x3), host_fraction_per_tile, subhalo_budget = 7.
HOST_CHANNELS = 7

# Fixed light scalings so the host conditioning channels sit near O(1) alongside
# the s-normalized SR2 input (log_host_mass is ~11-15; everything else is already
# O(1) or a 0..1 fraction). Applied in `host_context_stack`.
_LOGM_SCALE = 0.1


# --------------------------------------------------------------------------- #
# Tile geometry (raster, matching build_moment_target.generate_sr2_box)
# --------------------------------------------------------------------------- #
def tile_coord(tile_id: int) -> Tuple[int, int, int]:
    """``(ix, iy, iz)`` of a raster tile id in ``[0, 512)``."""
    n = N_PER_AXIS
    return tile_id // (n * n), (tile_id // n) % n, tile_id % n


def hr_block(ix: int, iy: int, iz: int) -> Tuple[slice, slice, slice]:
    s = TILE_HR
    return (slice(ix * s, (ix + 1) * s),
            slice(iy * s, (iy + 1) * s),
            slice(iz * s, (iz + 1) * s))


def lr_block(ix: int, iy: int, iz: int) -> Tuple[slice, slice, slice]:
    s = TILE_HR // UPSAMPLE            # 8 LR sites per tile edge
    return (slice(ix * s, (ix + 1) * s),
            slice(iy * s, (iy + 1) * s),
            slice(iz * s, (iz + 1) * s))


# --------------------------------------------------------------------------- #
# Whole-box SR2 generation (the one GPU cost; cached and reused by the sampler)
# --------------------------------------------------------------------------- #
def generate_sr2_box(model, lr, geom, device, seed: int, batch: int) -> np.ndarray:
    """``(6, 512, 512, 512)`` frozen SR2 displacement+velocity, on-disk units.

    The same per-tile forward and raster scatter as
    ``scripts/features/build_moment_target.generate_sr2_box`` -- kept in sync so
    the SR2 box aligns with the cached moment target tile for tile.
    """
    from .sr2_finetune_data import tile_lr_crop, tile_noise_stack, trim_to_tile

    n = NG_HR // TILE_HR
    lr_np = np.asarray(lr)
    box = np.empty((6, NG_HR, NG_HR, NG_HR), dtype=np.float32)
    tiles = list(range(n ** 3))
    t0 = time.time()
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        lr_b = torch.stack([
            torch.from_numpy(np.ascontiguousarray(tile_lr_crop(lr_np, t, geom)))
            .float() for t in chunk]).to(device)
        noise_list = [tile_noise_stack([seed], t, geom, device=device) for t in chunk]
        keys = list(noise_list[0].keys())
        noise = {k: torch.cat([nl[k] for nl in noise_list], dim=0) for k in keys}
        with torch.no_grad():
            out = trim_to_tile(model(lr_b, noise=noise), geom).float().cpu().numpy()
        for j, t in enumerate(chunk):
            ix, iy, iz = tile_coord(t)
            sx, sy, sz = hr_block(ix, iy, iz)
            box[:, sx, sy, sz] = out[j]
        if (i // batch) % 8 == 0:
            print(f"    sr2 tiles {i + len(chunk)}/{len(tiles)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return box


def load_or_make_sr2_box(model, lr, geom, device, seed: int, batch: int,
                         cache_path: Optional[Path], force: bool = False) -> np.ndarray:
    """Frozen SR2 box, from ``cache_path`` if present else generated and cached.

    Cached as fp16 (1.6 GB) so both the trainer and the sampler form it once.
    """
    if cache_path is not None and Path(cache_path).is_file() and not force:
        print(f"[sr2] cached -> {cache_path}", flush=True)
        return np.asarray(np.load(cache_path)).astype(np.float32)
    box = generate_sr2_box(model, lr, geom, device, seed, batch)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(cache_path).with_suffix(".tmp.npy")
        np.save(tmp, box.astype(np.float16))
        tmp.replace(cache_path)
        print(f"[sr2] wrote {cache_path} (fp16)", flush=True)
    return box


# --------------------------------------------------------------------------- #
# Local scale (section 4.2)
# --------------------------------------------------------------------------- #
def scale_fields(sr2_box: np.ndarray, k: int = 3, eps: float = 1e-3
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """``(s_disp, s_vel)``, each ``(512,512,512)`` smoothed rms of the SR2 field.

    ``s = sqrt(<|Psi|^2>_k + eps^2)`` with a ``k^3`` periodic box filter over the
    three components of the group. Periodic (``mode='wrap'``) because the
    Lagrangian lattice is; ``eps`` floors the void so the division is stable.
    """
    from scipy.ndimage import uniform_filter

    def one(group: np.ndarray) -> np.ndarray:
        sq = np.sum(np.asarray(group, dtype=np.float64) ** 2, axis=0)  # (512^3)
        sm = uniform_filter(sq, size=k, mode="wrap")
        return np.sqrt(sm + eps * eps).astype(np.float32)

    return one(sr2_box[0:3]), one(sr2_box[3:6])


def apply_scale(field6: torch.Tensor, s_disp: torch.Tensor, s_vel: torch.Tensor,
                undo: bool = False) -> torch.Tensor:
    """Divide (``undo=False``) or multiply (``undo=True``) a 6-ch field by ``s``.

    ``field6`` is ``(..., 6, D, H, W)``; ``s_disp`` / ``s_vel`` are ``(...,1,D,H,W)``
    or broadcastable. Displacement channels 0:3 use ``s_disp``, velocity 3:6 use
    ``s_vel`` -- the two groups have different physical scale.
    """
    disp, vel = field6[..., 0:3, :, :, :], field6[..., 3:6, :, :, :]
    if undo:
        disp, vel = disp * s_disp, vel * s_vel
    else:
        disp, vel = disp / s_disp, vel / s_vel
    return torch.cat([disp, vel], dim=-4)


# --------------------------------------------------------------------------- #
# Host conditioning + tile importance
# --------------------------------------------------------------------------- #
def host_context_stack(feat: LagrangianHostFeatures) -> np.ndarray:
    """``(HOST_CHANNELS, 64, 64, 64)`` LR host channels, lightly rescaled.

    Missing ``subhalo_budget`` is filled with zeros so the channel count is
    fixed at build time (a model trained with 7 context channels must always be
    fed 7).
    """
    stack = feat.stack_lr().astype(np.float32)          # (6 or 7, 64,64,64)
    if stack.shape[0] == HOST_CHANNELS - 1:             # no subhalo_budget
        stack = np.concatenate([stack, np.zeros((1,) + stack.shape[1:], np.float32)])
    if stack.shape[0] != HOST_CHANNELS:
        raise ValueError(
            f"host stack has {stack.shape[0]} channels, expected {HOST_CHANNELS}")
    stack = stack.copy()
    stack[1] *= _LOGM_SCALE                             # log_host_mass -> ~O(1)
    return stack


def tile_weights(feat: LagrangianHostFeatures, floor: float = 1e-3) -> np.ndarray:
    """``(512,)`` importance weights over tiles, by host-mass content.

    Section 4.3: ~24 clusters per box means most of 512 tiles are field and
    void, which SR2 already gets right; sampling uniformly spends the compute
    there. Weight by each tile's summed ``subhalo_budget`` (``lambda``, the share
    of the box's subhalo budget it owes), falling back to ``host_member`` count
    when the budget channel is absent. A small ``floor`` keeps every tile
    reachable so the loss never fully ignores a region.
    """
    field = feat.subhalo_budget if feat.subhalo_budget is not None else feat.host_member
    field = np.asarray(field, dtype=np.float64)
    w = np.zeros(N_TILES, dtype=np.float64)
    for t in range(N_TILES):
        ix, iy, iz = tile_coord(t)
        lx, ly, lz = lr_block(ix, iy, iz)
        w[t] = float(field[lx, ly, lz].sum())
    w = w / max(w.sum(), 1e-12)
    w = w + floor / N_TILES
    return (w / w.sum()).astype(np.float64)


# --------------------------------------------------------------------------- #
# The box container
# --------------------------------------------------------------------------- #
@dataclass
class SubstructureBoxes:
    """Whole-box SR2, target and scale fields, sliced into normalized tiles."""

    sr2: np.ndarray            # (6,512,512,512) on-disk units, fp16/fp32
    target: np.ndarray         # (6,512,512,512) = Pi(Psi_HR - Psi_SR2)
    s_disp: np.ndarray         # (512,512,512)
    s_vel: np.ndarray          # (512,512,512)
    host: np.ndarray           # (HOST_CHANNELS,64,64,64) LR, rescaled
    weights: np.ndarray        # (512,)

    @staticmethod
    def build(sr2_box: np.ndarray, target_box: np.ndarray,
              feat: LagrangianHostFeatures, k: int = 3, eps: float = 1e-3
              ) -> "SubstructureBoxes":
        if sr2_box.shape != (6, NG_HR, NG_HR, NG_HR):
            raise ValueError(f"sr2 box is {sr2_box.shape}, expected (6,512,512,512)")
        if target_box.shape != sr2_box.shape:
            raise ValueError(
                f"target {target_box.shape} != sr2 {sr2_box.shape}")
        s_disp, s_vel = scale_fields(sr2_box, k=k, eps=eps)
        return SubstructureBoxes(
            sr2=sr2_box, target=target_box, s_disp=s_disp, s_vel=s_vel,
            host=host_context_stack(feat), weights=tile_weights(feat))

    # -- tile access ------------------------------------------------------- #
    def _host_tile(self, ix: int, iy: int, iz: int) -> np.ndarray:
        lx, ly, lz = lr_block(ix, iy, iz)
        crop = self.host[:, lx, ly, lz]                 # (C,8,8,8)
        f = UPSAMPLE
        return crop.repeat(f, axis=1).repeat(f, axis=2).repeat(f, axis=3)

    def tile_tensors(self, tile_id: int, device) -> Dict[str, torch.Tensor]:
        """Normalized tensors for one tile (no batch dim), on ``device``.

        Keys: ``x_in`` (6, normalized SR2), ``x1`` (6, normalized target),
        ``context`` (HOST_CHANNELS), ``s_disp`` / ``s_vel`` (1,64,64,64) for
        un-normalizing a generated ``d`` back to on-disk units.
        """
        ix, iy, iz = tile_coord(tile_id)
        sx, sy, sz = hr_block(ix, iy, iz)
        sr2 = torch.from_numpy(np.asarray(self.sr2[:, sx, sy, sz], dtype=np.float32))
        tgt = torch.from_numpy(np.asarray(self.target[:, sx, sy, sz], dtype=np.float32))
        sd = torch.from_numpy(self.s_disp[sx, sy, sz][None].astype(np.float32))
        sv = torch.from_numpy(self.s_vel[sx, sy, sz][None].astype(np.float32))
        ctx = torch.from_numpy(np.ascontiguousarray(self._host_tile(ix, iy, iz)))
        sr2, tgt, sd, sv, ctx = (t.to(device) for t in (sr2, tgt, sd, sv, ctx))
        x_in = apply_scale(sr2, sd, sv, undo=False)
        x1 = apply_scale(tgt, sd, sv, undo=False)
        return {"x_in": x_in, "x1": x1, "context": ctx, "s_disp": sd, "s_vel": sv}

    def sample_batch(self, rng: np.random.Generator, batch: int, device
                     ) -> Dict[str, torch.Tensor]:
        """A batch of importance-sampled tiles: ``x_in``, ``x1``, ``context``."""
        ids = rng.choice(N_TILES, size=int(batch), p=self.weights)
        cols = {"x_in": [], "x1": [], "context": []}
        for t in ids:
            tt = self.tile_tensors(int(t), device)
            for k in cols:
                cols[k].append(tt[k])
        return {k: torch.stack(v, dim=0) for k, v in cols.items()}


# --------------------------------------------------------------------------- #
# Flow matching: loss and sampling (shared by trainer and inference)
# --------------------------------------------------------------------------- #
def cfm_loss(model, x_in: torch.Tensor, x1: torch.Tensor, context: torch.Tensor
             ) -> torch.Tensor:
    """Conditional (rectified/OT) flow-matching loss on one batch of tiles.

    Linear path ``r_t = t*x1 + (1-t)*z`` with ``z ~ N(0,I)``; the target velocity
    is the constant ``x1 - z``. The SR2 tile ``x_in`` rides the model's ``y_R``
    slot (upsampled by ``factor=1`` -> identity) and the host channels ride
    ``context``; flow time ``t`` enters via FiLM.
    """
    b = x1.shape[0]
    z = torch.randn_like(x1)
    t = torch.rand(b, device=x1.device, dtype=x1.dtype)
    tt = t.view(b, 1, 1, 1, 1)
    r_t = tt * x1 + (1.0 - tt) * z
    r_id = torch.ones(b, device=x1.device, dtype=x1.dtype)
    v = model(r_t, t, x_in, r_id, context=context)
    return torch.mean((v - (x1 - z)) ** 2)


@torch.no_grad()
def integrate_tile(model, x_in: torch.Tensor, context: torch.Tensor,
                   n_steps: int, generator: Optional[torch.Generator] = None
                   ) -> torch.Tensor:
    """Euler-integrate the flow from ``z ~ N(0,I)`` to ``d_norm`` for a batch.

    Returns the generated field in *normalized* units (still divided by ``s``);
    the caller un-normalizes with :func:`apply_scale`. ``x_in`` and ``context``
    are ``(B, C, 64, 64, 64)`` on the model's device.
    """
    device = x_in.device
    z = torch.randn(x_in.shape, device=device, dtype=x_in.dtype, generator=generator)
    x = z
    r_id = torch.ones(x_in.shape[0], device=device, dtype=x_in.dtype)
    dt = 1.0 / int(n_steps)
    for i in range(int(n_steps)):
        t = torch.full((x_in.shape[0],), i * dt, device=device, dtype=x_in.dtype)
        v = model(x, t, x_in, r_id, context=context)
        x = x + v * dt
    return x
