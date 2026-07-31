"""Full-box residual sampling by exact valid-core tiling.

A 512^3 six-channel forward pass does not fit on one GPU, so each denoising step
is evaluated tile by tile. The tiling is **exact**, not a blend: each tile is
``core + 2 * margin`` wide, the model runs on the whole tile, and only the
central ``core`` is written back. Provided ``margin`` is at least the network's
receptive-field half-width, the written values do not depend on the tiling, so no
seam correction (MultiDiffusion weights, overlap averaging) is needed and none is
applied.

This holds only because :class:`~cosmo_sr.reward.model.ResidualDenoiser` uses the
pointwise :class:`~cosmo_sr.models.flow_unet.ChannelGroupNorm3d`. With
``nn.GroupNorm`` the statistics are taken over the whole tile, every output voxel
depends on the tile size, and no margin makes the tiling exact -- measured here,
the tiled and untiled results then differ by the full signal amplitude.

Different tilings agree to float32 rounding (~1e-7 measured), not bit-exactly:
torch selects different convolution algorithms for different input sizes, so a
whole-box pass is not a bit-exact reference for a tiled one.

:func:`measure_receptive_field` measures the half-width numerically instead of
deriving it from the layer stack, :func:`tile_margin_for` turns it into a margin,
and :func:`check_margin` refuses to sample with one that is too small -- a
silently-too-small margin produces exactly the tile seams that show up later as
spurious small-scale power. For the configured model (``num_levels=2``,
``blocks_per_level=2``, ``scale_factor=8``) the half-width is 41 HR cells and the
margin is 48; ``scripts/reward/measure_receptive_field.py`` prints the table for
other depths along with the compute overhead each implies.

Tiles are cut with :func:`cosmo_sr.data.crops.periodic_crop`, the same periodic
crop the rest of the pipeline uses, so the box stays periodic throughout.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.crops import periodic_crop
from ..models.operator_denoiser import CosineSchedule
from .diffusion import DiffusionConfig, _clip_record, unwhiten

__all__ = [
    "RF_SAFETY_CELLS",
    "TileSpec",
    "check_margin",
    "measure_receptive_field",
    "receptive_field_halfwidth",
    "sample_residual_box",
    "tile_margin_for",
    "tile_origins",
]

# The measured half-width is a lower bound (see measure_receptive_field), so every
# margin carries a couple of cells of slack.
RF_SAFETY_CELLS = 2


class TileSpec:
    """Core/margin tiling of a cubic box on the HR grid."""

    def __init__(self, ng_hr: int, core: int = 128, margin: int = 32, scale_factor: int = 8):
        if ng_hr % core:
            raise ValueError(f"core={core} does not divide ng_hr={ng_hr}")
        if (core + 2 * margin) % scale_factor:
            raise ValueError(
                f"tile size {core + 2 * margin} must be a multiple of "
                f"scale_factor={scale_factor} so the LR window is exact"
            )
        if margin % scale_factor:
            raise ValueError(
                f"margin={margin} must be a multiple of scale_factor={scale_factor}"
            )
        self.ng_hr = int(ng_hr)
        self.core = int(core)
        self.margin = int(margin)
        self.scale_factor = int(scale_factor)

    @property
    def size(self) -> int:
        return self.core + 2 * self.margin

    @property
    def n_per_axis(self) -> int:
        return self.ng_hr // self.core

    @property
    def n_tiles(self) -> int:
        return self.n_per_axis ** 3

    def origins(self) -> List[Tuple[int, int, int]]:
        n = self.n_per_axis
        c = self.core
        return [(i * c, j * c, k * c) for i in range(n) for j in range(n) for k in range(n)]


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else max(abs(a), abs(b), 1)


def tile_origins(ng_hr: int, core: int) -> List[Tuple[int, int, int]]:
    n = ng_hr // core
    return [(i * core, j * core, k * core) for i in range(n) for j in range(n) for k in range(n)]


@torch.no_grad()
def receptive_field_halfwidth(
    model,
    *,
    size: int = 64,
    channels: int = 6,
    scale_factor: int = 8,
    device=None,
    threshold: float = 0.0,
) -> int:
    """Numerically measure how far one input voxel can influence the output.

    Perturbs the centre voxel and reports the largest axis distance at which the
    output changes at all (``threshold=0``) or by more than ``threshold`` times
    the largest change. Cheaper and much harder to get wrong than counting
    convolutions by hand; used by :func:`check_margin`.

    The measurement is made on a *probe copy* of the model, not on the model
    itself, for three reasons:

    * every weight is replaced by the same **positive** constant. The trained
      weights are not the point -- the support is -- and a zero-initialised output
      head would report a half-width of zero and wave through any margin, while
      random weights can cancel and hide real support. All-positive weights with
      a zero background and a positive perturbation admit no cancellation, so a
      nonzero output really does mean "in the support".
    * biases are zeroed, so the unperturbed background stays exactly zero and the
      support is just the nonzero set of a single forward pass.
    * any spatially-reducing normalisation is replaced by identity.
      ``nn.GroupNorm`` takes its statistics over the whole volume and so couples
      every voxel to every other -- a strictly infinite receptive field of
      amplitude ~1/N^3 that no margin can cover. The model itself defaults to the
      pointwise :class:`~cosmo_sr.models.flow_unet.ChannelGroupNorm3d` for
      exactly that reason, so this substitution is normally a no-op, but the
      channel norm is dropped too because at group size 1 it collapses a channel
      to its own sign and would truncate the measured support.

    Because the probe is width-independent, the half-width of an expensive model
    can be measured on a narrow one with the same depth, kernel size and
    ``scale_factor``.

    All three inputs are probed, not just ``u_t``. One LR voxel is block-upsampled
    to a ``scale_factor`` cube, so the conditioning path reaches roughly
    ``scale_factor`` voxels further than the noise path -- measured, the LR reach
    was the binding constraint for every configuration tried here.

    The returned half-width is the convolutional support: how far tile-boundary
    padding can reach into the tile, which is exactly what :func:`check_margin`
    needs. A return value equal to ``size // 2`` means the probe box was too
    small and the true half-width is larger; :func:`measure_receptive_field`
    grows the box until that is not the case.
    """
    device = device or torch.device("cpu")
    probe = _structural_probe(model).to(device).eval()
    n = int(size)
    sf = int(scale_factor)
    # The perturbation is placed at the *start* of an LR block, and on the
    # downsampling grid. Otherwise the answer depends on where the centre of the
    # probe box happens to fall inside its block: measured on the levels=2 model,
    # a centre at block offset 4 reported 37 cells where a centre at offset 0
    # reported 41, because a block-upsampled LR voxel reaches scale_factor cells
    # further on one side than the other.
    align = _lcm(sf, int(getattr(model, "divisor", 1)))
    c = max(n // 2 // align, 1) * align
    t = torch.full((1,), 0.5, device=device)
    n_lr = n // sf
    inputs = {
        "u_t": torch.zeros(1, channels, n, n, n, device=device),
        "psi_base": torch.zeros(1, channels, n, n, n, device=device),
        "y_lr": torch.zeros(1, channels, n_lr, n_lr, n_lr, device=device),
    }

    def _run(over: Dict[str, torch.Tensor]) -> torch.Tensor:
        f = {**inputs, **over}
        return probe(f["u_t"], t, y_lr=f["y_lr"], psi_base=f["psi_base"], redshift=0.0)

    y0 = _run({})
    reach = 0
    for name in ("u_t", "psi_base", "y_lr"):
        pert = inputs[name].clone()
        j = c // sf if name == "y_lr" else c
        pert[:, :, j, j, j] = 1.0
        diff = (_run({name: pert}) - y0).abs().amax(dim=(0, 1))
        peak = float(diff.max())
        if peak <= 0.0:
            continue
        idx = torch.nonzero(diff > float(threshold) * peak)
        if idx.numel():
            reach = max(reach, int((idx - c).abs().max().item()))
    return reach


def measure_receptive_field(
    model,
    *,
    channels: int = 6,
    scale_factor: int = 8,
    device=None,
    start: int = 64,
    max_size: int = 256,
) -> int:
    """:func:`receptive_field_halfwidth`, with the probe box grown until it fits.

    A single-shot probe under-reports whenever the support runs off the edge of the
    probe box, which is the one failure mode that would let a too-small tile margin
    through. So the box grows until the answer fits inside it, and then one larger
    probe is taken and the larger of the two returned.

    The two probes are not expected to agree exactly. The outermost shells of the
    support are at the float32 noise floor, so a shell resolved in one box can
    truncate to zero in another -- measured on the ``levels=2, blocks=2`` model:
    40, 41, 40 cells at probe 96, 144, 216. Treat the result as a lower bound and
    add :data:`RF_SAFETY_CELLS`; :func:`tile_margin_for` does.

    The box grows by 1.5x rather than doubling: a probe costs O(size^3) memory, and
    for the deepest configuration tried here doubling stepped straight from a box
    that was too small to one that would not fit in RAM. ``max_size`` is the same
    guard -- a 320^3 confirmation probe was killed by the OOM reaper on a login
    node, so the confirmation is skipped rather than attempted beyond it.
    """
    div = int(getattr(model, "divisor", 1)) * int(scale_factor)
    size = max(int(start), div) // div * div

    def grow(s: int) -> int:
        return max(s + div, -(-(s * 3 // 2) // div) * div)

    def probe(s: int) -> int:
        return receptive_field_halfwidth(model, size=s, channels=channels,
                                         scale_factor=scale_factor, device=device)

    rf = 0
    while size <= int(max_size):
        rf = probe(size)
        if rf < size // 2:
            bigger = grow(size)
            return max(rf, probe(bigger)) if bigger <= int(max_size) else rf
        size = grow(size)
    raise RuntimeError(
        f"receptive field is at least {rf} and still fills a probe box of "
        f"{max_size}; a model this wide-reaching cannot be tiled cheaply -- "
        f"reduce num_levels or blocks_per_level"
    )


def tile_margin_for(rf: int, scale_factor: int, safety: int = RF_SAFETY_CELLS) -> int:
    """Smallest admissible tile margin: ``rf + safety``, rounded up to ``scale_factor``.

    Rounding to ``scale_factor`` is required (the LR window must be exact), and the
    safety term covers the float32 tail truncation described in
    :func:`measure_receptive_field`.
    """
    need = int(rf) + int(safety)
    return -(-need // int(scale_factor)) * int(scale_factor)


def _structural_probe(model):
    """A copy of ``model`` with positive weights, zero biases and no norm layers."""
    import copy

    import torch.nn as nn

    from ..models.flow_unet import ChannelGroupNorm3d

    probe = copy.deepcopy(model)
    norms = (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm3d, nn.InstanceNorm3d,
             ChannelGroupNorm3d)
    for mod in probe.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, norms):
                setattr(mod, name, nn.Identity())
    with torch.no_grad():
        for name, p in probe.named_parameters():
            if p.dim() >= 2:
                # Gain ~1 per layer, so a deep stack neither overflows nor
                # underflows to zero and truncates the measured support.
                fan_in = int(p.numel() // p.shape[0])
                p.fill_(1.0 / max(fan_in, 1))
            else:
                p.zero_()
    return probe


def check_margin(model, margin: int, *, probe_size: int = 64, **kw) -> int:
    """Raise if ``margin`` is smaller than the measured receptive-field half-width.

    The probe box starts at ``probe_size`` and grows until the measurement is both
    unclipped and stable, so a margin can never be accepted merely because the
    probe was too small to see past it.
    """
    rf = measure_receptive_field(model, start=probe_size, **kw)
    sf = int(kw.get("scale_factor", 8))
    need = tile_margin_for(rf, sf)
    if margin < need:
        raise ValueError(
            f"tile margin {margin} < receptive-field half-width {rf} "
            f"(+{RF_SAFETY_CELLS} safety, rounded to {sf}); valid-core tiling "
            f"would leak tile-boundary padding into the written region. "
            f"Use --tile-margin {need}."
        )
    return rf


def _crop_pair(
    u: torch.Tensor,
    psi_base: torch.Tensor,
    y_lr: torch.Tensor,
    origin: Tuple[int, int, int],
    spec: TileSpec,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One tile of each field, as ``(1, C, ...)`` ready to concatenate."""
    start_hr = tuple(int(o) - spec.margin for o in origin)
    start_lr = tuple(s // spec.scale_factor for s in start_hr)
    uc = periodic_crop(u, start_hr, spec.size, pad=0)
    bc = periodic_crop(psi_base, start_hr, spec.size, pad=0)
    yc = periodic_crop(y_lr, start_lr, spec.size // spec.scale_factor, pad=0)
    return uc.unsqueeze(0), bc.unsqueeze(0), yc.unsqueeze(0)


def _write_core(dst: torch.Tensor, src: torch.Tensor, origin: Tuple[int, int, int],
                spec: TileSpec) -> None:
    m, c, n = spec.margin, spec.core, spec.ng_hr
    core = src[..., m:m + c, m:m + c, m:m + c]
    ox, oy, oz = (int(o) % n for o in origin)
    dst[..., ox:ox + c, oy:oy + c, oz:oz + c] = core


@torch.no_grad()
def sample_residual_box(
    model,
    psi_base: np.ndarray | torch.Tensor,
    y_lr: np.ndarray | torch.Tensor,
    *,
    seed: int,
    cfg: Optional[DiffusionConfig] = None,
    spec: Optional[TileSpec] = None,
    device=None,
    redshift: float = 0.0,
    tile_batch: int = 1,
    verify_margin: bool = True,
    progress: bool = False,
    clip_log: Optional[list] = None,
    init_noise: Optional[np.ndarray | torch.Tensor] = None,
) -> np.ndarray:
    """One full-box residual realisation ``dPsi`` in catnorm units.

    The initial noise is drawn on the CPU from ``seed`` so a realisation depends
    on the seed alone, never on the device or the tile order -- same guarantee
    (and the same reason) as :func:`cosmo_sr.tts.sampling.super_resolve_srs_seeded`.

    ``init_noise`` overrides that draw with an explicit tensor, which is what
    lets a search (CEM) perturb a good noise vector instead of only jumping to
    an unrelated seed. ``seed`` still drives the ``churn`` draws, so an explicit
    noise vector only fully determines the sample when ``cfg.churn == 0``.

    ``clip_log`` collects the per-step ``x0_clip`` statistics in standardized
    residual units; see :func:`cosmo_sr.reward.diffusion.ddim_sample`. A
    zero-init head does **not** make this sample equal ``psi_base`` -- only
    ``residual_scale = 0`` does -- so the clip fraction is the honest measure of
    how much of the residual amplitude is the model and how much is the bound.
    """
    cfg = cfg or DiffusionConfig()
    device = device or next(model.parameters()).device
    # The box stays 4D ``(C, N, N, N)``; the batch axis exists only per tile.
    base = torch.as_tensor(np.asarray(psi_base), dtype=torch.float32)
    lr = torch.as_tensor(np.asarray(y_lr), dtype=torch.float32)
    if base.dim() == 5:
        base = base.squeeze(0)
    if lr.dim() == 5:
        lr = lr.squeeze(0)
    ng = int(base.shape[-1])
    spec = spec or TileSpec(ng, core=min(128, ng), margin=32)
    if verify_margin:
        check_margin(model, spec.margin, probe_size=max(32, 2 * spec.margin + 8),
                     channels=int(base.shape[0]), scale_factor=spec.scale_factor,
                     device=device)

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    if init_noise is None:
        u = torch.randn(tuple(base.shape), generator=gen, dtype=torch.float32)
    else:
        u = torch.as_tensor(np.asarray(init_noise), dtype=torch.float32).clone()
        if tuple(u.shape) != tuple(base.shape):
            raise ValueError(
                f"init_noise shape {tuple(u.shape)} != field shape "
                f"{tuple(base.shape)}"
            )
    sched = CosineSchedule()
    ts = torch.linspace(cfg.t_max, cfg.t_min, int(cfg.n_steps) + 1)
    origins = spec.origins()

    for i in range(int(cfg.n_steps)):
        t_cur = ts[i].reshape(1)
        t_nxt = ts[i + 1].reshape(1)
        eps_full = torch.empty_like(u)
        for b0 in range(0, len(origins), max(1, int(tile_batch))):
            group = origins[b0:b0 + max(1, int(tile_batch))]
            uc, bc, yc = zip(*[_crop_pair(u, base, lr, o, spec) for o in group])
            uc = torch.cat(uc, 0).to(device, non_blocking=True)
            bc = torch.cat(bc, 0).to(device, non_blocking=True)
            yc = torch.cat(yc, 0).to(device, non_blocking=True)
            eps = model(
                uc, t_cur.to(device).expand(uc.shape[0]),
                y_lr=yc, psi_base=bc, redshift=redshift,
            ).to("cpu")
            for j, o in enumerate(group):
                _write_core(eps_full, eps[j], o, spec)
            del uc, bc, yc, eps
        a_c, s_c = sched.broadcast(t_cur, ndim=4)
        a_n, s_n = sched.broadcast(t_nxt, ndim=4)
        u0 = (u - s_c * eps_full) / a_c.clamp_min(1e-8)
        if cfg.x0_clip > 0:
            if clip_log is not None:
                clip_log.append(_clip_record(u0.unsqueeze(0), float(cfg.x0_clip),
                                             int(i), float(ts[i])))
            u0 = u0.clamp(-cfg.x0_clip, cfg.x0_clip)
        if cfg.churn <= 0.0:
            u = a_n * u0 + s_n * eps_full
        else:
            keep = (max(0.0, 1.0 - cfg.churn ** 2)) ** 0.5
            fresh = torch.randn(u.shape, generator=gen, dtype=torch.float32)
            u = a_n * u0 + s_n * (keep * eps_full + cfg.churn * fresh)
        del eps_full, u0
        if progress:
            print(f"    ddim step {i + 1}/{cfg.n_steps}", flush=True)

    resid = unwhiten(u.unsqueeze(0), model.sigma_res.cpu()).squeeze(0)
    return resid.numpy().astype(np.float32, copy=False)
