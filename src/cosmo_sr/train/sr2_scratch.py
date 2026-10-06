"""Training SR2's own architecture from a RANDOM initialisation.

Every other entry point in this repository fine-tunes ``SRmodel/G_z0.pt``. This
module exists because that checkpoint is itself the hypothesis under test:

    `radial-placement-is-inverted` measured HR putting subhalos at a median
    1.13 host ``R_vir`` while every tuned arm sits at 0.30-0.35, *worse than the
    frozen checkpoint's own 0.53*. `docs/sr2_soft_membership.md` section 7.9
    then measured that the membership solve is **aiming** correctly -- its
    candidate centres sit at median 1.42 ``R_vir`` against HR's 1.19 -- and the
    objects still condense at 0.35. Right target, no object.

So the material is already piled in the host centre before any loss term speaks,
and no fine-tune has moved it. The experiment this module serves is: **start
from noise instead, and put the placement-sensitive terms in the objective from
the beginning.** If a from-scratch generator trained with the same base recipe
reproduces the same inverted radial profile, the defect is the architecture or
the base objective and no loss term will fix it. If it does not, the frozen
checkpoint was the obstacle.

What "the same base recipe" means
---------------------------------
SR2 was trained by ``map2map`` (``external/map2map``, pinned in
``external/COMMITS.txt``) with ``--criterion MSELoss`` plus a conditional
adversary. Both halves are reproduced here:

``MSE``
    :func:`base_mse` on all six channels against the paired HR tile. Faithful,
    and *suspect*: a pointwise loss to one realisation is this repository's
    recorded cause of mean-collapse (`dmsr-flow-is-the-fresh-train-answer`), and
    is a plausible cause of the central concentration itself. It is a weight, not
    a constant, so the ``w_mse = 0`` arm is a one-flag experiment.

``adversarial``
    WGAN-GP on :class:`cosmo_sr.reward.sr2_adversarial.SR2Critic`, which already
    reproduces SR2's discriminator **input contract**: 6 upsampled LR channels +
    6 candidate channels + **8 channels of inverse-pixel-shuffled CIC density**.
    The density channels are the point -- they are what makes this critic judge
    the mass field directly rather than the displacement that produced it.

The density channels only work at ``valid_center > 0``
------------------------------------------------------
This is the one setting in this module that is not a preference. A training
example is a 64^3 *tile*, not a periodic box, and :func:`cic_density`'s ``% ng``
deposit wraps escaping particles back inside it. Only ~9.75% of a tile's
particles stay put (median displacement ~36 HR cells against a 64-cell tile), so
the wrapped density correlates **r = 0.08** with the tile's true density --
measured, ``docs/density_collapse_investigation.md``. Eight of the critic's
twenty channels would be noise, and the critic would learn to ignore them.

:func:`scratch_critic_input` therefore defaults to the valid-centre deposit at
``region = tile / 2`` (r = 1.0000 against an exactly-buffered reference) and
:func:`check_density_channels` refuses ``valid_center = 0`` unless the caller
says ``allow_wrapped=True``. The same convention is what ``critic.valid_center:
32`` fixed for the DMSR line.

The critic input must be normalised, and the density tail transformed
------------------------------------------------------------------
*Measured* on real HR tiles of set0 (batch 4), the raw 20-channel input:

    upsampled LR   (ch 0-5)    RMS 0.778
    field disp+vel (ch 6-11)   RMS 0.841
    DENSITY        (ch 12-19)  RMS 4.911   <- 5.8x the field channels
    density max 2786.1   p99.9 21.3   p50 -1.000

Two separate problems, and the second is the one that breaks WGAN-GP.

**Group imbalance.** Density enters at 5.8x the field channels, so the field
half starves. This is what :class:`GatherCriticNorm` and DMSR's
:class:`CriticInputNormalizer` exist for -- the latter's docstring records the
same near-miss at 138x. :func:`fit_critic_norm` reuses ``GatherCriticNorm``
(it is channel-count agnostic) with the same discipline both of those insist
on: **the scales are estimated from REAL HR tiles only, held fixed, and applied
identically to real and fake.** Per-batch statistics would normalise real and
fake differently and hand the critic a discriminative signal that has nothing
to do with sample quality.

**The heavy tail.** Cosmological overdensity is unbounded above: max 2786
against an RMS of 4.9, a 570x spread, because a halo core is a halo core. A
per-channel RMS scale divides that down to ~570 and no further, and asking
``||grad_x D|| = 1`` over a space where a handful of coordinates swing by
hundreds makes the gradient penalty ill-conditioned -- dominated by the density
directions and effectively unenforced everywhere else.

*Measured*, jobs 62067/62068 with neither fix: ``d_real - d_fake`` ran 93 at
step 2k, 298 at 5k, **2548 at 30k**, monotone, with the penalty at 1414, i.e.
``||grad_x D|| ~ 4`` against the target of 1. That is a runaway.

:func:`apply_density_transform` therefore puts the density channels through
``asinh`` before the scale is fitted. ``asinh`` and not ``log1p``: overdensity
is bounded below by -1 exactly, where ``log1p`` is ``-inf`` and the median voxel
sits (p50 = -1.000, most of a tile is empty). ``asinh`` is signed, smooth at 0,
logarithmic in the tail and needs no clamp; it maps 2786 to 8.6.

``--density-transform none`` restores the raw channels for a
faithful-to-upstream arm. It is not the default, because the default that
reproduces upstream exactly is the one measured above to diverge.

The low-k anchor is referenced to HR, not to the frozen field
-------------------------------------------------------------
``free_field_gather.guards`` anchors the block-averaged candidate to the
**frozen SR2 field**, because in a fine-tune the candidate starts as that field
and the guard is a drift detector. From a random initialisation there is no
drift to detect and the frozen field is not the target, so :func:`lowk_anchor`
takes the same block average against **HR**. The operator is identical; only the
reference changes, and the returned keys match ``guards``' so the two are
readable side by side in one ``metrics.jsonl``.

Warm-up is not optional for the structure terms
-----------------------------------------------
The member-gather soft-membership solve chooses particles *from the current SR
field* (`soft-membership-relaxation`), and the kNN term compares the candidate's
neighbour distances to HR's. At step 0 from scratch that field is noise: the
solve would select arbitrary material and the kNN term would charge a distance
distribution that reflects nothing. :func:`term_ramp` holds those terms at zero
for ``warmup`` steps and then ramps them linearly over ``ramp``, which is the
same shape the gather critic's ``adv_weight_at`` already uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..reward.sr2_adversarial import (
    SR2Critic,
    critic_input,
    gradient_penalty,
    wgan_critic_loss,
    wgan_generator_loss,
)
from ..dmsr.critic import LazyR1
from ..features.gather_critic import GatherCriticNorm
from ..tts.srs_noise import ControlledG
from .sr2_finetune_data import (
    SR2TileGeometry,
    lr_start_of_tile_id,
    tile_lr_crop,
    tile_noise_stack,
    trim_to_tile,
)

__all__ = [
    "ADV_LOSSES",
    "DENSITY_TRANSFORMS",
    "add_instance_noise",
    "bce_critic_update",
    "bce_generator_loss",
    "instance_noise_std",
    "ScratchBatch",
    "PairedTileSampler",
    "apply_density_transform",
    "base_mse",
    "fit_critic_norm",
    "build_scratch_critic",
    "build_scratch_generator",
    "check_density_channels",
    "critic_update",
    "generator_adv_loss",
    "lowk_anchor",
    "noise_std_report",
    "scratch_critic_input",
    "term_ramp",
    "tile_hr_target",
]


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
def build_scratch_generator(
    *,
    in_chan: int = 6,
    out_chan: int = 6,
    scale_factor: int = 8,
    cond_chan: int = 0,
    init_weight_std: float = 0.0,
    init_noise_std: float = 0.0,
    seed: int = 0,
    device=None,
    **chan_kwargs,
) -> ControlledG:
    """A randomly initialised :class:`ControlledG` -- SR2's architecture, no weights.

    ``init_weight_std``
        ``> 0`` reproduces ``map2map.train.init_weights``: every convolution and
        linear weight drawn from ``N(0, std)``. ``0`` (the default) keeps
        PyTorch's own Kaiming-uniform init, which is what upstream uses when
        ``--init-weight-std`` is not passed.

    ``init_noise_std``
        The ``AddNoise.std`` parameters. Upstream initialises them at **zero**,
        which makes a fresh generator deterministic until the gradient grows
        them; the pretrained ``G_z0`` ended at ~1e-3 (sites z0-z4) and ~5e-2
        (z5). Zero is faithful and is the default. A positive value seeds all
        six sites at that amplitude, which is worth having because SR2's
        stochasticity is overwhelmingly what writes fine-scale structure -- a
        run whose ``std`` never leaves zero has no noise to make substructure
        out of. :func:`noise_std_report` is how a run checks.

    The generator is returned in ``train`` mode with every parameter requiring
    grad: the rung machinery of :mod:`cosmo_sr.train.sr2_unfreeze` is about
    protecting a pretrained function and has nothing to protect here.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.random.fork_rng(devices=[]):
        torch.random.manual_seed(int(seed))
        model = ControlledG(int(in_chan), int(out_chan), int(scale_factor),
                            cond_chan=int(cond_chan), **chan_kwargs)

    std = float(init_weight_std)
    if std > 0:
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                with torch.no_grad():
                    m.weight.copy_(torch.randn(m.weight.shape, generator=gen) * std)

    ns = float(init_noise_std)
    if ns != 0.0:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith(".std"):
                    p.fill_(ns)

    model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def noise_std_report(model: nn.Module) -> Dict[str, float]:
    """``{parameter name: mean |std|}`` for every ``AddNoise.std`` in the model.

    SR2's fine-scale stochasticity lives entirely in these six tensors. A
    from-scratch run that trains for ten hours with all of them still at ~0 has
    produced a deterministic upsampler, whatever its power spectrum says, and
    this is the cheapest place to see that.
    """
    out: Dict[str, float] = {}
    for name, p in model.named_parameters():
        if name.endswith(".std"):
            out[name] = float(p.detach().abs().mean())
    return out


# --------------------------------------------------------------------------- #
# The paired-tile stream
# --------------------------------------------------------------------------- #
def tile_hr_target(hr_field, tile_id: int, *, ng_hr: int = 512,
                   tile: int = 64) -> np.ndarray:
    """``(6, tile, tile, tile)`` HR sub-cube of one tile.

    A copy of ``overfit_host_mse.tile_hr_target`` with the grid as arguments
    rather than as module constants, so this module imports nothing from
    ``scripts/``. ``tests/train/test_sr2_scratch.py`` pins the two against each
    other; a divergence would mean the from-scratch run and the fine-tune runs
    are supervised on different sub-cubes of the same box.
    """
    n = int(ng_hr) // int(tile)
    ix, iy, iz = tile_id // (n * n), (tile_id // n) % n, tile_id % n
    s = int(tile)
    # A COPY, not a view. `hr_field` is normally a read-only memmap of a 3.2 GB
    # box, and `torch.from_numpy` on a non-writable array warns and documents
    # the result as undefined behaviour if anything ever writes to it.
    return np.array(hr_field[:, ix * s:(ix + 1) * s, iy * s:(iy + 1) * s,
                             iz * s:(iz + 1) * s], dtype=np.float32, copy=True)


@dataclass
class ScratchBatch:
    """One minibatch of paired tiles, on the training device."""

    lr: torch.Tensor                       # (B, 6, lr_size, ...)
    hr: torch.Tensor                       # (B, 6, tile_hr, ...)
    noise: Dict[str, torch.Tensor]         # {site: (B, 1, s, s, s)}
    boxes: List[str] = _field(default_factory=list)
    tiles: List[int] = _field(default_factory=list)

    def __len__(self) -> int:
        return int(self.lr.shape[0])


class PairedTileSampler:
    """Random paired ``(LR crop, HR tile, noise)`` examples, streamed from memmaps.

    The host pool that ``finetune_member_gather.py`` builds holds every tile it
    will ever use as a resident CPU tensor, which is right for ~56 hosts and
    impossible for the 8,192 ``(box, tile)`` pairs a from-scratch run needs: at
    6 MB per HR tile that pool alone is 48 GB. Here the HR and LR boxes stay
    memory-mapped and one tile is sliced per example, so host RSS is flat in the
    number of tiles.

    ``exclude``
        ``{(box, tile)}`` pairs to keep out of the stream. The trainer passes the
        supervised host tiles so the base objective's data and the member-gather
        objective's data do not silently become the same tiles -- the held-out
        gate would then be the only honest number in the run.

    ``noise_seed_stride``
        A tile's noise is a function of ``(root seed, tile)``, exactly as in
        deployed inference. Drawing a *fresh* root seed per visit (the default,
        ``stride > 0``) is what stops the generator from memorising one
        realisation per tile; ``stride = 0`` pins every visit to ``base_seed``,
        which is the replayable setting the diagnostics use.
    """

    def __init__(
        self,
        boxes: Sequence[str],
        lr_fields: Dict[str, np.ndarray],
        hr_fields: Dict[str, np.ndarray],
        *,
        geom: Optional[SR2TileGeometry] = None,
        ng_hr: int = 512,
        tile_hr: int = 64,
        exclude: Optional[Sequence[Tuple[str, int]]] = None,
        base_seed: int = 0,
        noise_seed_stride: int = 1_000_003,
        seed: int = 0,
        device=None,
    ):
        self.geom = geom or SR2TileGeometry()
        self.boxes = [str(b) for b in boxes]
        if not self.boxes:
            raise ValueError("PairedTileSampler needs at least one box")
        self.lr_fields = {str(k): v for k, v in lr_fields.items()}
        self.hr_fields = {str(k): v for k, v in hr_fields.items()}
        missing = [b for b in self.boxes
                   if b not in self.lr_fields or b not in self.hr_fields]
        if missing:
            raise KeyError(f"no LR/HR field for {missing}")
        self.ng_hr = int(ng_hr)
        self.tile_hr = int(tile_hr)
        self.base_seed = int(base_seed)
        self.noise_seed_stride = int(noise_seed_stride)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(int(seed))

        n_tot = (self.ng_hr // self.tile_hr) ** 3
        drop = {(str(b), int(t)) for b, t in (exclude or ())}
        self.index: List[Tuple[str, int]] = [
            (b, t) for b in self.boxes for t in range(n_tot)
            if (b, t) not in drop
        ]
        if not self.index:
            raise ValueError(
                "every (box, tile) pair was excluded; the base objective would "
                "have no data")
        self.n_excluded = len(drop)
        self._visits = 0

    def __len__(self) -> int:
        return len(self.index)

    def _seed_for(self, tile: int) -> int:
        if self.noise_seed_stride == 0:
            return self.base_seed
        return int(self.base_seed
                   + self.noise_seed_stride * int(self.rng.integers(1 << 30)))

    def example(self, box: str, tile: int) -> Dict:
        """One example as CPU tensors, for tests and for deterministic replay."""
        lr = tile_lr_crop(np.asarray(self.lr_fields[box]), int(tile), self.geom)
        hr = tile_hr_target(self.hr_fields[box], int(tile),
                            ng_hr=self.ng_hr, tile=self.tile_hr)
        noise = tile_noise_stack([self._seed_for(int(tile))], int(tile), self.geom,
                                 device=torch.device("cpu"))
        return {
            "box": str(box), "tile": int(tile),
            "lr": torch.from_numpy(np.ascontiguousarray(lr)).float(),
            "hr": torch.from_numpy(hr).float(),
            # tile_noise_stack carries a draw axis of length 1; the base
            # objective has no use for draws, so it is squeezed here rather
            # than folded, and the batch axis below is the only one.
            "noise": {s: v[0] for s, v in noise.items()},
        }

    def batch(self, size: int) -> ScratchBatch:
        """``size`` random examples, without replacement inside the batch."""
        n = int(size)
        if n < 1:
            raise ValueError("batch size must be >= 1")
        pick = self.rng.choice(len(self.index), size=min(n, len(self.index)),
                               replace=False)
        rows = [self.example(*self.index[int(i)]) for i in pick]
        self._visits += len(rows)
        sites = list(rows[0]["noise"].keys())
        return ScratchBatch(
            lr=torch.stack([r["lr"] for r in rows]).to(self.device),
            hr=torch.stack([r["hr"] for r in rows]).to(self.device),
            noise={s: torch.stack([r["noise"][s] for r in rows]).to(self.device)
                   for s in sites},
            boxes=[r["box"] for r in rows],
            tiles=[r["tile"] for r in rows],
        )


def forward_batch(model: nn.Module, batch: ScratchBatch,
                  geom: SR2TileGeometry) -> torch.Tensor:
    """``(B, 6, tile_hr, ...)`` centre-trimmed candidate for a batch."""
    return trim_to_tile(model(batch.lr, noise=batch.noise), geom).float()


def lr_core(batch_lr: torch.Tensor, geom: SR2TileGeometry) -> torch.Tensor:
    """The unpadded LR core of a padded generator input, for the critic."""
    p, c = int(geom.pad), int(geom.chunk)
    return batch_lr[:, :, p:p + c, p:p + c, p:p + c]


# --------------------------------------------------------------------------- #
# The base objective
# --------------------------------------------------------------------------- #
def base_mse(cand: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """``map2map``'s default ``--criterion MSELoss``, on all six channels."""
    if cand.shape != hr.shape:
        raise ValueError(f"cand {tuple(cand.shape)} != hr {tuple(hr.shape)}")
    return (cand - hr).pow(2).mean()


def check_density_channels(valid_center: int, tile_hr: int, *,
                           allow_wrapped: bool = False) -> int:
    """Validate the critic's density deposit mode. Returns ``valid_center``.

    ``0`` is the wrapped ``% ng`` deposit, which on a single tile correlates
    ``r = 0.08`` with that tile's true density. Eight of the critic's twenty
    channels are then noise. This refuses it rather than warning, because the
    resulting run looks healthy from every log line it writes.
    """
    r = int(valid_center)
    if r == 0 and not allow_wrapped:
        raise ValueError(
            "valid_center=0 is the wrapped CIC deposit. On a "
            f"{tile_hr}^3 tile it correlates r=0.08 with the tile's true "
            "density (docs/density_collapse_investigation.md), so the critic's "
            "8 density channels would carry no density. Use "
            f"valid_center={tile_hr // 2} (exact), or pass allow_wrapped=True "
            "to run the broken deposit deliberately.")
    if r < 0 or r > int(tile_hr):
        raise ValueError(f"valid_center {r} outside 0..{tile_hr}")
    return r


#: Transforms available for the density half of the critic input.
DENSITY_TRANSFORMS: Tuple[str, ...] = ("none", "asinh")


def apply_density_transform(x: torch.Tensor, n_density: int,
                            kind: str = "asinh") -> torch.Tensor:
    """Transform the trailing ``n_density`` channels of a critic input.

    ``asinh`` (the default) compresses the overdensity tail: *measured* max
    2786 -> 8.6, while leaving the bulk near-linear (``asinh(x) ~ x`` for small
    ``x``). It is signed, smooth at 0 and finite at ``delta = -1``, which
    ``log1p`` is not -- and ``delta = -1`` is the MEDIAN voxel of a tile, not an
    edge case.

    The head channels are untouched, and the operation is functional rather
    than in-place so it composes with autograd.
    """
    k = str(kind)
    if k not in DENSITY_TRANSFORMS:
        raise ValueError(f"density transform {k!r} not in {DENSITY_TRANSFORMS}")
    if k == "none":
        return x
    n = int(n_density)
    if not 0 < n < int(x.shape[1]):
        raise ValueError(
            f"n_density {n} must be inside the channel count {int(x.shape[1])}")
    head, rho = x[:, :-n], x[:, -n:]
    return torch.cat([head, torch.asinh(rho)], dim=1)


def scratch_critic_input(
    lr: torch.Tensor,
    field: torch.Tensor,
    *,
    cellsize_kpc_h: float,
    dis_norm_kpc_h: float = 6000.0,
    grid_mult: int = 2,
    valid_center: int = 32,
    allow_wrapped: bool = False,
    density_transform: str = "asinh",
    norm: Optional[GatherCriticNorm] = None,
) -> torch.Tensor:
    """SR2's 20-channel critic input, deposit checked, tail tamed, scaled.

    Wraps :func:`cosmo_sr.reward.sr2_adversarial.critic_input` and adds the
    three things a WGAN-GP critic on these channels cannot do without:

    * ``valid_center`` is explicit at every call site and defaults to the
      tile/2 region rather than the wrapped deposit (``allow_wrapped`` is the
      deliberate escape hatch; see :func:`check_density_channels`);
    * ``density_transform`` tames the overdensity tail
      (:func:`apply_density_transform`);
    * ``norm`` divides by fixed per-channel scales.

    ``norm=None`` returns the transformed but UNSCALED tensor. That is not a
    convenience default -- it is what :func:`fit_critic_norm` consumes to
    estimate the scales in the first place. Every training call site passes a
    fitted ``norm``, and the trainer refuses to build a critic without one.
    """
    check_density_channels(int(valid_center), int(field.shape[-1]),
                           allow_wrapped=bool(allow_wrapped))
    x = critic_input(lr, field, cellsize_kpc_h=float(cellsize_kpc_h),
                     dis_norm_kpc_h=float(dis_norm_kpc_h),
                     grid_mult=int(grid_mult),
                     valid_center=int(valid_center))
    x = apply_density_transform(x, int(grid_mult) ** 3, str(density_transform))
    return x if norm is None else norm(x)


@torch.no_grad()
def fit_critic_norm(
    sampler: "PairedTileSampler",
    geom: SR2TileGeometry,
    *,
    cellsize_kpc_h: float,
    dis_norm_kpc_h: float = 6000.0,
    grid_mult: int = 2,
    valid_center: int = 32,
    allow_wrapped: bool = False,
    density_transform: str = "asinh",
    n_tiles: int = 32,
    batch: int = 4,
    device=None,
) -> GatherCriticNorm:
    """Per-channel scales for the critic input, from **real HR tiles only**.

    The discipline is the one :class:`GatherCriticNorm` and DMSR's
    :class:`CriticInputNormalizer` both spell out: estimate once from real
    data, hold fixed, apply identically to real and fake. Per-batch statistics
    would normalise the two sides differently and hand the critic a shortcut
    that has nothing to do with sample quality.

    The HR field is fed as ``field``, so the density channels are HR's own -- the
    generator is not involved and the scales do not drift with it.
    """
    tiles = []
    got = 0
    while got < int(n_tiles):
        b = sampler.batch(min(int(batch), int(n_tiles) - got))
        tiles.append(scratch_critic_input(
            lr_core(b.lr, geom), b.hr, cellsize_kpc_h=float(cellsize_kpc_h),
            dis_norm_kpc_h=float(dis_norm_kpc_h), grid_mult=int(grid_mult),
            valid_center=int(valid_center), allow_wrapped=bool(allow_wrapped),
            density_transform=str(density_transform), norm=None).float())
        got += len(b)
    return GatherCriticNorm.fit(tiles).to(
        device or (tiles[0].device if tiles else "cpu"))


def instance_noise_std(step: int, *, sigma: float, batches: float) -> float:
    """``map2map``'s annealed instance noise: ``sigma`` at step 0, 0 at ``batches``.

    Upstream adds it to the discriminator's inputs -- both real and fake -- to
    stop D separating them perfectly, which is exactly the failure that ended
    run 3: ``d_real`` +8.5 against ``d_fake`` -8.2, ``d_loss`` 0.0005, and a
    generator free to walk to 1e11 because a saturated D gave it no gradient
    back.
    """
    if sigma <= 0 or batches <= 0:
        return 0.0
    return float(sigma) * max(0.0, 1.0 - float(step) / float(batches))


def add_instance_noise(x: torch.Tensor, std: float,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Add ``N(0, std)`` to a critic input. Applied IDENTICALLY to real and fake.

    A different draw per side is fine and is what upstream does; what must not
    differ is the *scale*, or the critic gets a cue that has nothing to do with
    sample quality.
    """
    if std <= 0:
        return x
    return x + float(std) * torch.randn(x.shape, device=x.device, dtype=x.dtype,
                                        generator=generator)


def build_scratch_critic(*, width: int = 64, depth: int = 4, seed: int = 0,
                         spectral_norm: bool = False, inner_norm: str = "group",
                         device=None) -> SR2Critic:
    """A fresh :class:`SR2Critic`.

    ``cosmo_sr.reward.sr2_adversarial.build_critic`` refuses a fresh critic
    unless ``allow_fresh_critic`` is set, and stamps the provenance
    ``fresh_random_init_ABLATION_ONLY``. That caveat is about *continuing* the
    pretrained generator's training, where a random critic defines a random
    objective applied to weights worth protecting. Here the generator is random
    too: a fresh critic is the only possible critic and is not an ablation of
    anything. The provenance is still recorded by the trainer's manifest.

    ``inner_norm="none"`` drops the critic's GroupNorms. That is not a tuning
    detail: GroupNorm makes :class:`SR2Critic` EXACTLY scale-invariant, which is
    the measured mechanism of the handoff divergence (see the class docstring
    and :func:`scale_anchor`). Use it together with ``spectral_norm=True`` --
    the SNGAN pairing -- so the critic is bounded without being blinded.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.random.fork_rng(devices=[]):
        torch.random.manual_seed(int(seed))
        critic = SR2Critic(width=int(width), depth=int(depth),
                           inner_norm=str(inner_norm))
    if spectral_norm:
        # map2map's --adv-model-spectral-norm. It bounds the critic's Lipschitz
        # constant structurally, which is a different and more robust mechanism
        # than a penalty term: R1 and WGAN-GP only *charge* for a large
        # gradient, they cannot prevent one.
        from torch.nn.utils.parametrizations import spectral_norm as _sn
        for name, m in list(critic.named_modules()):
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                parent = critic
                *path, leaf = name.split(".")
                for q in path:
                    parent = getattr(parent, q)
                setattr(parent, leaf, _sn(m))
    return critic.to(device)


#: Adversarial objectives. ``bce`` is what ``map2map`` actually trains with.
ADV_LOSSES: Tuple[str, ...] = ("bce", "wgan_gp")


def generator_adv_loss(critic: nn.Module, fake_in: torch.Tensor) -> torch.Tensor:
    """``-E[D(fake)]`` -- the WGAN generator loss."""
    return wgan_generator_loss(critic(fake_in))


def bce_generator_loss(critic: nn.Module, fake_in: torch.Tensor) -> torch.Tensor:
    """The non-saturating GAN generator loss: ``BCE(D(fake), 1)``.

    ``map2map/train.py`` line 412, verbatim in form:
    ``adv_criterion(score_out, real.expand_as(score_out))`` with
    ``--adv-criterion BCEWithLogitsLoss`` (its default).
    """
    score = critic(fake_in)
    return F.binary_cross_entropy_with_logits(score, torch.ones_like(score))


def bce_critic_update(
    critic: nn.Module,
    opt_d: torch.optim.Optimizer,
    real_in: torch.Tensor,
    fake_in: torch.Tensor,
    *,
    lazy_r1: Optional[LazyR1] = None,
    label_smoothing: float = 1.0,
) -> Dict[str, float]:
    """One discriminator step in ``map2map``'s form.

    ``BCE(D(fake), 0) + BCE(D(real), label_smoothing)``, plus lazy R1 on the
    **real** batch only. That R1 -- a gradient penalty on real samples, applied
    every ``interval`` steps with the coefficient scaled by the interval -- is
    what ``--adv-r1-reg-interval`` does upstream, and it is NOT WGAN-GP: there
    is no interpolate between real and fake anywhere in ``map2map``.

    ``label_smoothing`` is ``--adv-label-smoothing`` (upstream default 1.0,
    i.e. off); below 1 it softens the real target to stop D saturating.
    """
    real_in = real_in.detach()
    fake_in = fake_in.detach()
    opt_d.zero_grad(set_to_none=True)

    s_fake = critic(fake_in)
    l_fake = F.binary_cross_entropy_with_logits(s_fake, torch.zeros_like(s_fake))
    l_fake.backward()

    s_real = critic(real_in)
    l_real = F.binary_cross_entropy_with_logits(
        s_real, torch.full_like(s_real, float(label_smoothing)))
    l_real.backward()

    r1 = 0.0
    if lazy_r1 is not None:
        pen, m = lazy_r1(critic, real_in)
        if pen is not None:
            pen.backward()
            r1 = float(m.get("loss_R1", 0.0))
    opt_d.step()
    return {
        "d_loss": float((l_fake + l_real).detach()),
        "d_real": float(s_real.mean().detach()),
        "d_fake": float(s_fake.mean().detach()),
        "d_r1": r1,
    }


def critic_update(
    critic: nn.Module,
    opt_d: torch.optim.Optimizer,
    real_in: torch.Tensor,
    fake_in: torch.Tensor,
    *,
    gp_lambda: float = 10.0,
    do_gp: bool = True,
    gp_every: int = 16,
) -> Dict[str, float]:
    """One WGAN-GP critic step on detached inputs. Returns the logged scalars.

    ``do_gp`` carries the upstream schedule: the gradient penalty is the
    expensive term and SR2 paid for it once per ``gp_every`` critic batches, so
    the coefficient is scaled by ``gp_every`` on the steps it does run (lazy
    regularisation, the same arithmetic ``LazyR1`` uses for the DMSR critic).
    """
    real_in = real_in.detach()
    fake_in = fake_in.detach()
    opt_d.zero_grad(set_to_none=True)
    real_score = critic(real_in)
    fake_score = critic(fake_in)
    loss = wgan_critic_loss(real_score, fake_score)
    gp = torch.zeros((), device=real_in.device)
    if do_gp:
        gp = gradient_penalty(critic, real_in, fake_in,
                              gp_lambda=float(gp_lambda) * float(gp_every))
    (loss + gp).backward()
    opt_d.step()
    return {
        "d_loss": float(loss.detach()),
        "d_real": float(real_score.mean().detach()),
        "d_fake": float(fake_score.mean().detach()),
        "d_gp": float(gp.detach()),
    }


# --------------------------------------------------------------------------- #
# Guards and schedules
# --------------------------------------------------------------------------- #
def _block_average(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Mean over non-overlapping ``factor^3`` blocks, ``(B, C, N, N, N)``."""
    if x.dim() != 5:
        raise ValueError(f"expected (B, C, N, N, N), got {tuple(x.shape)}")
    b, c, n = x.shape[0], x.shape[1], x.shape[-1]
    f = int(factor)
    if n % f:
        raise ValueError(f"grid {n} not divisible by factor {f}")
    m = n // f
    return x.reshape(b, c, m, f, m, f, m, f).mean(dim=(3, 5, 7))


def lowk_anchor(cand: torch.Tensor, hr: torch.Tensor,
                scale_factor: int) -> Dict[str, torch.Tensor]:
    """``free_field_gather.guards``' LR-scale anchor, referenced to **HR**.

    Same operator, same three splits (all six channels, displacement only,
    velocity only) and the same ``vel_rms_ratio``, so a scratch run's
    ``metrics.jsonl`` lines up column-for-column with a fine-tune's. The one
    change is the reference: a from-scratch candidate has not drifted from the
    frozen field, it has never been there, and the quantity worth constraining
    is agreement with the truth at LR scale.

    Squared relative error, never its square root: at a perfect match the
    gradient of ``sqrt`` is infinite, which is how this term NaN'd a fine-tune.
    """
    out: Dict[str, torch.Tensor] = {}
    for name, sl in (("low", slice(0, 6)), ("low_dis", slice(0, 3)),
                     ("low_vel", slice(3, 6))):
        a = _block_average(cand[:, sl], scale_factor)
        b = _block_average(hr[:, sl], scale_factor).detach()
        out[name] = (a - b).pow(2).mean() / b.pow(2).mean().clamp_min(1e-30)
    out["vel_rms_ratio"] = cand[:, 3:6].std() / hr[:, 3:6].std().detach()
    return out


#: Channel groups the amplitude anchor constrains, as ``(name, slice)``.
SCALE_GROUPS: Tuple[Tuple[str, slice], ...] = (
    ("dis", slice(0, 3)), ("vel", slice(3, 6)))


def scale_anchor(cand: torch.Tensor, hr: torch.Tensor) -> Dict[str, torch.Tensor]:
    """``log(rms_cand / rms_hr)^2`` per channel group -- an AMPLITUDE-only leash.

    WHY THIS EXISTS. With ``--w-mse 0`` and ``--w-base-low 0`` the from-scratch
    objective is the adversary alone, and the adversary cannot see amplitude:
    :class:`SR2Critic`'s GroupNorms make it exactly scale-invariant (MEASURED --
    ``D`` constant to 4 digits from 1e3x to 1e6x, ``||dD/dx||`` falling as
    ``1/a``). Nothing in the objective then constrains the output's overall
    scale, and the field random-walks in log-amplitude: bracket probes C and D
    reached ``vel_rms_ratio`` 5.8e5 within one eval interval of the handoff, and
    THRASHED across five decades rather than climbing, which is what an
    unconstrained direction looks like -- not what a diverging one does.

    WHY IT IS SAFE FOR THE RADIAL QUESTION. The whole point of dropping MSE is
    that a pointwise loss to one realisation is suspected of flattening the
    radial distribution of subhalo mass. This term is two scalars per batch --
    the RMS of the displacement channels and of the velocity channels -- and
    carries NO spatial information whatsoever. It cannot prefer one radial
    profile over another; it can only say "the field is a thousand times too
    loud". It is the ``vel_rms_ratio`` that ``lowk_anchor`` already reports,
    promoted from a diagnostic to a (small-weight) objective.

    Log-squared, not ratio-squared, so over- and under-shooting by the same
    factor cost the same and the term is finite as ``rms_cand -> 0``. The HR
    side is detached; the clamp keeps ``log`` away from zero at a random init.
    """
    out: Dict[str, torch.Tensor] = {}
    total = None
    for name, sl in SCALE_GROUPS:
        a = cand[:, sl].pow(2).mean().clamp_min(1e-20).sqrt()
        b = hr[:, sl].pow(2).mean().clamp_min(1e-20).sqrt().detach()
        term = torch.log(a / b).pow(2)
        out[f"scale_{name}"] = term
        out[f"rms_ratio_{name}"] = (a / b).detach()
        total = term if total is None else total + term
    out["scale"] = total
    return out


def mse_handoff_weight(step: int, w_mse: float, *, adv_start: int,
                       handoff: int) -> float:
    """``w_mse`` up to ``adv_start``, then linearly to 0 over ``handoff`` steps.

    ``handoff=0`` is the cliff -- ``map2map``'s own form and what probes C and D
    ran. A cliff swaps the generator's entire objective between two consecutive
    steps, which also leaves Adam's second-moment estimate calibrated to the
    gradient of a loss that no longer exists. That is an aggravator rather than
    the cause (the cause is that nothing downstream of the cliff constrains
    amplitude at all), but it is free to remove.
    """
    if step <= int(adv_start):
        return float(w_mse)
    if int(handoff) <= 0:
        return 0.0
    frac = (step - int(adv_start)) / float(handoff)
    return float(w_mse) * max(0.0, 1.0 - frac)


def term_ramp(step: int, *, warmup: int, ramp: int) -> float:
    """``0`` before ``warmup``, then linear to ``1`` over ``ramp`` steps.

    The structure terms need a field with structure in it before they mean
    anything: the soft-membership solve selects particles from the *current* SR
    field, and the kNN term matches neighbour distances against HR. Charged from
    step 0 against noise they would select arbitrary material and dominate the
    gradient while doing so.
    """
    w, r = int(warmup), int(ramp)
    s = int(step)
    if s < w:
        return 0.0
    if r <= 0:
        return 1.0
    return float(min(1.0, (s - w) / r))
