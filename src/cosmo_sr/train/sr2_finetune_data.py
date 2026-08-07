"""Training crops for **direct** SR2 fine-tuning: one generator tile per example.

The residual line trains a separate network on top of a frozen SR2, so its data
loader is free to crop wherever it likes. This line fine-tunes ``G_theta(Y, z)``
itself, and that removes the freedom: an example has to be *exactly* one forward
pass of the deployed generator, or the gradient is computed for a function the
box is never produced by.

"Exactly one forward pass" is a concrete geometry, fixed by
``configs/sr2_baseline/freeze.yaml`` (``nsplit = 8``, ``pad = 3``,
``scale_factor = 8``) and by :func:`cosmo_sr.tts.sampling.super_resolve_srs_seeded`:

* the LR input is an ``8^3`` core plus 3 LR cells of periodic padding on every
  side, i.e. ``14^3``;
* the six noise tensors come from ``tile_noise(root_seed, lr_start, ...)``, so a
  tile's realisation is a function of ``(seed, tile)`` and of nothing else --
  not of traversal order, not of batching, not of the device;
* the ``70^3`` output is centre-trimmed to ``64^3``, which is precisely one
  :class:`cosmo_sr.reward.tiles.TileGrid` tile.

So a crop from this module and the corresponding slice of a full-box seeded
inference are the *same tensor*, not merely similar ones, and
``tests/train/test_sr2_finetune_data.py`` pins that as an exact equality rather
than a tolerance. That identity is what lets a per-tile catalog summary -- which
only exists for whole boxes, because only whole boxes can be run through a halo
finder -- be used as the label for a single training example.

Tile ordering
-------------
:func:`cosmo_sr.tts.sampling.tile_starts` walks ``(ix, iy, iz)`` in C order over
``nsplit^3``, and :meth:`cosmo_sr.reward.tiles.TileGrid.index` maps the same
triple to ``(ix * n + iy) * n + iz``. The two orderings therefore agree, and
:func:`tile_id_of_lr_start` / :func:`lr_start_of_tile_id` are exact inverses;
they exist so that no caller has to re-derive the correspondence and get it
subtly wrong.

Seed diversity, not tile diversity
----------------------------------
``noise_draws = 2`` returns two independent noise realisations **of the same LR
tile**. The diversity guard in the actor objective is a statement about
``z``: "different ``z`` must still give different structure". Substituting two
different LR tiles would satisfy any such measure trivially -- different inputs
produce different outputs whatever the generator does with its noise -- and
would silently turn a collapse detector into a no-op. Hence the two draws share
``lr`` and differ only in ``noise``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.crops import periodic_crop
from ..reward.tiles import TileGrid, TileSummary, read_tile_summaries
from ..tts.sampling import tile_noise, tile_starts
from ..tts.srs_noise import NOISE_SITES, ControlledG, noise_site_layout

__all__ = [
    "SR2TileDataset",
    "SR2TileGeometry",
    "collate_tiles",
    "fold_draws",
    "frozen_tile_forward",
    "load_frozen_summaries",
    "lr_start_of_tile_id",
    "tile_id_of_lr_start",
    "tile_lr_crop",
    "tile_noise_stack",
    "trim_to_tile",
]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SR2TileGeometry:
    """The deployed tiling, as numbers rather than as scattered constants.

    Defaults are the frozen SR2 inference settings. Nothing here is a tunable:
    changing ``pad`` or ``nsplit`` changes what a "tile" is, and the per-tile
    catalog labels stop applying.
    """

    ng_lr: int = 64
    nsplit: int = 8
    pad: int = 3
    scale_factor: int = 8

    @property
    def chunk(self) -> int:
        """LR cells per tile core."""
        if self.ng_lr % self.nsplit:
            raise ValueError(f"nsplit={self.nsplit} does not divide ng_lr={self.ng_lr}")
        return self.ng_lr // self.nsplit

    @property
    def lr_size(self) -> int:
        """Spatial size of the tensor actually fed to the generator."""
        return self.chunk + 2 * self.pad

    @property
    def tile_hr(self) -> int:
        """HR cells per tile: the centre-trimmed output size."""
        return self.chunk * self.scale_factor

    @property
    def ng_hr(self) -> int:
        return self.ng_lr * self.scale_factor

    @property
    def n_tiles(self) -> int:
        return self.nsplit ** 3

    def tile_grid(self, boxsize_mpc_h: float = 100.0) -> TileGrid:
        return TileGrid(ng_hr=self.ng_hr, tile_hr=self.tile_hr,
                        boxsize_mpc_h=float(boxsize_mpc_h))

    def site_sizes(self) -> Dict[str, int]:
        """``{site: spatial size}`` of the six noise tensors for one tile."""
        return {lay.site: int(lay.size)
                for lay in noise_site_layout(self.lr_size, self.scale_factor)}

    def starts(self) -> List[Tuple[int, int, int]]:
        return tile_starts(self.ng_lr, self.nsplit)


def tile_id_of_lr_start(start: Sequence[int], geom: SR2TileGeometry) -> int:
    """Tile id of the tile whose *unpadded* LR core begins at ``start``."""
    c = geom.chunk
    n = geom.nsplit
    ix, iy, iz = (int(s) for s in start)
    if ix % c or iy % c or iz % c:
        raise ValueError(f"LR start {tuple(start)} is not a multiple of chunk {c}")
    return int(((ix // c) % n) * n * n + ((iy // c) % n) * n + ((iz // c) % n))


def lr_start_of_tile_id(tile_id: int, geom: SR2TileGeometry) -> Tuple[int, int, int]:
    """Inverse of :func:`tile_id_of_lr_start`."""
    n = geom.nsplit
    t = int(tile_id)
    if not 0 <= t < geom.n_tiles:
        raise IndexError(f"tile {tile_id} outside 0..{geom.n_tiles - 1}")
    c = geom.chunk
    return (t // (n * n) * c, (t // n) % n * c, (t % n) * c)


def tile_lr_crop(lr_field: np.ndarray, tile_id: int,
                 geom: SR2TileGeometry) -> np.ndarray:
    """``(C, lr_size, lr_size, lr_size)`` periodically padded LR input of one tile."""
    start = lr_start_of_tile_id(tile_id, geom)
    return periodic_crop(np.asarray(lr_field), start, geom.chunk, pad=geom.pad)


def trim_to_tile(sr: torch.Tensor, geom: SR2TileGeometry) -> torch.Tensor:
    """Centre-trim a generator output to the ``tile_hr`` cube it contributes.

    Identical arithmetic to ``sampling._narrow_like``; duplicated here only so a
    training step does not have to import a ``@torch.no_grad`` module's private
    helper (and so a shape mistake raises here rather than silently mis-aligning
    a label).
    """
    tgt = geom.tile_hr
    width = int(sr.shape[-1]) - tgt
    if width < 0:
        raise ValueError(f"generator output {sr.shape[-1]} smaller than tile {tgt}")
    if width % 2:
        raise ValueError(
            f"generator output {sr.shape[-1]} and tile {tgt} differ by an odd "
            "number of cells; the centre trim would be off by half a cell"
        )
    h = width // 2
    return sr[..., h:h + tgt, h:h + tgt, h:h + tgt]


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def tile_noise_stack(
    seeds: Sequence[int],
    tile_id: int,
    geom: SR2TileGeometry,
    *,
    device=None,
    noise_mode: str = "per_tile",
) -> Dict[str, torch.Tensor]:
    """``{site: (D, 1, s, s, s)}`` -- one noise realisation per root seed.

    Each realisation is exactly what ``super_resolve_srs_seeded`` would give the
    tile at that seed, so a draw can be replayed on the full box. ``D`` is the
    *draw* axis and is folded into the batch by :func:`collate_tiles`' consumer;
    it exists so a step can see one LR tile under several ``z`` at once, which is
    what the structural-diversity term needs.
    """
    start = lr_start_of_tile_id(tile_id, geom)
    dev = device or torch.device("cpu")
    per_seed = [
        tile_noise(int(s), start, geom.lr_size, geom.scale_factor, dev,
                   pad=geom.pad, mode=str(noise_mode))
        for s in seeds
    ]
    sites = [s for s in NOISE_SITES if s in per_seed[0]]
    return {
        site: torch.stack([n[site].squeeze(0) for n in per_seed], dim=0)
        for site in sites
    }


@torch.no_grad()
def frozen_tile_forward(
    generator: ControlledG,
    lr: torch.Tensor,
    noise: Dict[str, torch.Tensor],
    geom: SR2TileGeometry,
) -> torch.Tensor:
    """The frozen tile output for an explicit ``(Y, z)``, centre-trimmed.

    ``lr`` is ``(B, C, lr_size, ...)`` and every noise tensor is ``(B, 1, s, ...)``
    -- i.e. the draw axis has already been folded into the batch. Wrapped in
    ``no_grad`` because this is the *baseline*: a gradient into it would let the
    optimiser move the thing the guards measure against.
    """
    was_training = generator.training
    generator.eval()
    out = trim_to_tile(generator(lr, noise=noise), geom)
    if was_training:
        generator.train()
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SR2TileDataset(torch.utils.data.Dataset):
    """One example = one ``(box, tile)`` pair, with everything a step needs.

    Parameters
    ----------
    boxes:
        Box names. Every array below is looked up by box name, and a box missing
        from any required mapping is an error at construction time rather than at
        step 400 of a training run.
    lr_fields / hr_fields:
        ``{box: (6, ng_lr, ...)}`` and ``{box: (6, ng_hr, ...)}``. HR may be
        memory-mapped; a ``512^3`` six-channel box is 3.2 GB and only one tile of
        it is ever touched per example.
    frozen_fields:
        ``{box: (6, ng_hr, ...)}`` full-box **frozen SR2** output at
        ``base_seed``. Optional, and preferred when present: slicing the cached
        box is free and is by construction the field every earlier stage of this
        project measured. Draws at other seeds are generated on the fly, which
        needs ``frozen_generator``.
    frozen_summaries:
        ``{box: {tile_id: TileSummary}}`` from the frozen box's real Rockstar
        run. This is the label side of the problem: ``s_{0,j}``, the frozen
        tile's contribution, and ``S_0 = sum_j s_{0,j}`` the pooled box summary.
    tile_ids:
        Restrict to a subset of tiles (the host-rich overfit in section 10 uses
        a handful). ``None`` means all ``nsplit^3``.
    noise_draws:
        Independent ``z`` per example, all for the **same** LR tile.
    """

    def __init__(
        self,
        boxes: Sequence[str],
        lr_fields: Dict[str, np.ndarray],
        hr_fields: Dict[str, np.ndarray],
        *,
        geom: Optional[SR2TileGeometry] = None,
        frozen_fields: Optional[Dict[str, np.ndarray]] = None,
        frozen_summaries: Optional[Dict[str, Dict[int, TileSummary]]] = None,
        frozen_generator: Optional[ControlledG] = None,
        tile_ids: Optional[Sequence[int]] = None,
        base_seed: int = 0,
        noise_draws: int = 1,
        diversity_seed_stride: int = 1_000_003,
        noise_mode: str = "per_tile",
    ):
        self.geom = geom or SR2TileGeometry()
        self.boxes = [str(b) for b in boxes]
        if not self.boxes:
            raise ValueError("SR2TileDataset needs at least one box")
        self.lr_fields = {str(k): v for k, v in lr_fields.items()}
        self.hr_fields = {str(k): v for k, v in hr_fields.items()}
        self.frozen_fields = {str(k): v for k, v in (frozen_fields or {}).items()}
        self.frozen_summaries = {
            str(k): {int(t): s for t, s in v.items()}
            for k, v in (frozen_summaries or {}).items()
        }
        self.frozen_generator = frozen_generator
        self.base_seed = int(base_seed)
        self.noise_draws = int(noise_draws)
        if self.noise_draws < 1:
            raise ValueError("noise_draws must be >= 1")
        self.diversity_seed_stride = int(diversity_seed_stride)
        self.noise_mode = str(noise_mode)

        missing_lr = [b for b in self.boxes if b not in self.lr_fields]
        missing_hr = [b for b in self.boxes if b not in self.hr_fields]
        if missing_lr or missing_hr:
            raise KeyError(
                f"missing LR fields for {missing_lr} and HR fields for {missing_hr}"
            )
        for b in self.boxes:
            lr = self.lr_fields[b]
            if int(lr.shape[-1]) != self.geom.ng_lr:
                raise ValueError(
                    f"{b}: LR grid {lr.shape[-1]} != geometry ng_lr {self.geom.ng_lr}"
                )
            hr = self.hr_fields[b]
            if int(hr.shape[-1]) != self.geom.ng_hr:
                raise ValueError(
                    f"{b}: HR grid {hr.shape[-1]} != geometry ng_hr {self.geom.ng_hr}"
                )
        if self.noise_draws > 1 and not self.frozen_fields and self.frozen_generator is None:
            raise ValueError(
                "noise_draws > 1 needs a frozen_generator: only the base seed has "
                "a cached full-box frozen field to slice"
            )

        all_ids = list(range(self.geom.n_tiles))
        self.tile_ids = [int(t) for t in (all_ids if tile_ids is None else tile_ids)]
        bad = [t for t in self.tile_ids if not 0 <= t < self.geom.n_tiles]
        if bad:
            raise IndexError(f"tile ids outside 0..{self.geom.n_tiles - 1}: {bad[:5]}")
        self.index: List[Tuple[str, int]] = [
            (b, t) for b in self.boxes for t in self.tile_ids
        ]

    # -- provenance -------------------------------------------------------- #
    def seeds_for(self, tile_id: int) -> List[int]:
        """Root seeds of the ``noise_draws`` realisations of one tile.

        Draw 0 is always ``base_seed``, so it is the realisation the cached
        frozen field, the frozen catalog and every earlier measurement describe.
        Further draws are offset by a large stride rather than by ``+1`` because
        ``derive_tile_seed`` hashes ``(root, coord, site)`` and consecutive roots
        are perfectly fine there -- but a stride keeps the draw index legible in
        a log, and makes an accidental collision with another run's seed list
        obvious rather than plausible.
        """
        return [self.base_seed + d * self.diversity_seed_stride
                for d in range(self.noise_draws)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict:
        box, tile_id = self.index[int(i)]
        geom = self.geom
        seeds = self.seeds_for(tile_id)

        lr_start = lr_start_of_tile_id(tile_id, geom)
        hr_start = tuple(int(s) * geom.scale_factor for s in lr_start)
        lr = np.ascontiguousarray(
            tile_lr_crop(self.lr_fields[box], tile_id, geom), dtype=np.float32
        )
        sl = geom.tile_grid().slices(tile_id)
        hr = np.ascontiguousarray(
            np.asarray(self.hr_fields[box][:, sl[0], sl[1], sl[2]]), dtype=np.float32
        )

        noise = tile_noise_stack(seeds, tile_id, geom, noise_mode=self.noise_mode)
        frozen = self._frozen(box, tile_id, lr, noise, seeds)

        out: Dict = {
            "box": box,
            "tile_id": int(tile_id),
            "lr_start": torch.as_tensor(lr_start, dtype=torch.int64),
            "hr_start": torch.as_tensor(hr_start, dtype=torch.int64),
            "seeds": torch.as_tensor(seeds, dtype=torch.int64),
            "lr": torch.from_numpy(lr),
            "hr": torch.from_numpy(hr),
            "noise": noise,
            "frozen": frozen,
        }
        summary = self.frozen_summaries.get(box, {}).get(int(tile_id))
        if summary is not None:
            out["frozen_n_sub"] = torch.as_tensor(
                np.asarray(summary.n_sub, dtype=np.float64))
            out["frozen_n_host"] = torch.as_tensor(
                np.asarray(summary.n_host, dtype=np.float64))
            out["frozen_occ_numerator"] = torch.as_tensor(
                np.asarray(summary.occ_numerator, dtype=np.float64))
            out["frozen_volume_mpc3"] = float(summary.volume_mpc3)
        return out

    def _frozen(self, box: str, tile_id: int, lr: np.ndarray,
                noise: Dict[str, torch.Tensor], seeds: Sequence[int]) -> torch.Tensor:
        """``(D, C, tile_hr, ...)`` frozen output for each draw's ``(Y, z)``."""
        geom = self.geom
        cached = self.frozen_fields.get(box)
        out: List[torch.Tensor] = []
        for d, seed in enumerate(seeds):
            if cached is not None and int(seed) == self.base_seed:
                sl = geom.tile_grid().slices(tile_id)
                out.append(torch.from_numpy(np.ascontiguousarray(
                    np.asarray(cached[:, sl[0], sl[1], sl[2]]), dtype=np.float32)))
                continue
            if self.frozen_generator is None:
                raise RuntimeError(
                    f"{box}/t{tile_id} draw {d} (seed {seed}) has no cached frozen "
                    "field and no frozen_generator to produce one"
                )
            t = torch.from_numpy(lr).unsqueeze(0)
            z = {k: v[d:d + 1] for k, v in noise.items()}
            out.append(frozen_tile_forward(self.frozen_generator, t, z, geom).squeeze(0))
        return torch.stack(out, dim=0)


def collate_tiles(batch: Sequence[Dict]) -> Dict:
    """Stack examples, keeping the ``(B, D, ...)`` draw axis explicit.

    ``torch.utils.data.default_collate`` would do the tensors correctly but would
    turn ``box`` into a list and choke on nothing else -- it is written out here
    so the shape contract is visible where the trainer reads it, and so the
    string/scalar fields are handled deliberately instead of by accident.
    """
    items = list(batch)
    if not items:
        raise ValueError("cannot collate an empty batch")
    out: Dict = {
        "box": [str(b["box"]) for b in items],
        "tile_id": torch.as_tensor([int(b["tile_id"]) for b in items], dtype=torch.int64),
    }
    for key in ("lr_start", "hr_start", "seeds", "lr", "hr", "frozen"):
        out[key] = torch.stack([b[key] for b in items], dim=0)
    out["noise"] = {
        site: torch.stack([b["noise"][site] for b in items], dim=0)
        for site in items[0]["noise"]
    }
    for key in ("frozen_n_sub", "frozen_n_host", "frozen_occ_numerator"):
        if key in items[0]:
            out[key] = torch.stack([b[key] for b in items], dim=0)
    if "frozen_volume_mpc3" in items[0]:
        out["frozen_volume_mpc3"] = torch.as_tensor(
            [float(b["frozen_volume_mpc3"]) for b in items], dtype=torch.float64)
    return out


def fold_draws(batch: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int, int]:
    """``(lr, noise, B, D)`` with the draw axis folded into the batch axis.

    The generator has no notion of draws: it wants ``(B*D, C, ...)`` inputs and
    ``(B*D, 1, ...)`` noise. ``lr`` is *repeated*, not expanded, because the
    noise tensors are real distinct memory and mixing a view with a copy through
    a checkpointed forward is a class of bug that only shows up under AMP.
    """
    lr = batch["lr"]
    noise = batch["noise"]
    b = int(lr.shape[0])
    d = int(next(iter(noise.values())).shape[1])
    lr_rep = lr.unsqueeze(1).repeat(1, d, *([1] * (lr.dim() - 1)))
    lr_rep = lr_rep.reshape(b * d, *lr.shape[1:])
    z = {k: v.reshape(b * d, *v.shape[2:]) for k, v in noise.items()}
    return lr_rep, z, b, d


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def load_frozen_summaries(path: str | Path) -> Dict[int, TileSummary]:
    """``{tile_id: TileSummary}`` from a ``write_tile_summaries`` JSONL file."""
    return {int(s.tile_id): s for s in read_tile_summaries(path)}
