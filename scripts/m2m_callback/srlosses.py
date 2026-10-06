"""Our loss terms, in the shape map2map's ``--extra-criterion`` expects.

``import_attr(args.extra_criterion, nn, models, callback_at=...)`` resolves a
name in this module, constructs it with ``--extra-criterion-args``, and the
training loop calls it as ``f(output, target, style=style)`` on the **raw
narrowed 6-channel fields**, before ``lag2eul``/``--cgan`` concatenation.

Why an extra criterion rather than ``--criterion``: the generator steps on the
adversarial loss ALONE once ``epoch >= adv_start`` (``train.py`` ~448-460), so
``--criterion`` is computed and logged but never backpropagated. A term plugged
in there would silently do nothing for every epoch that matters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cosmo_sr.features.knn_cdf import KnnConfig          # noqa: E402
from cosmo_sr.features.knn_crop import (                 # noqa: E402
    CELL_MPC_H, PHASE_LAMBDA_KMS_PER_MPC_H, knn_crop_loss,
)


class KnnCdf(nn.Module):
    """The kNN-CDF term on the training crop. See ``features/knn_crop.py``.

    Index-free by construction: map2map's ``flip``/``perm`` are proper
    isometries, so a kNN log-distance distribution is invariant under
    ``--augment``, and the reference is the TARGET of the same batch element,
    which went through the same flip and permutation as the candidate. There is
    nothing stored to fall out of step, and no constraint on ``--crop``,
    ``--crop-step`` or ``--aug-shift``.

    ``phase_lambda`` defaults to 4000 km/s per Mpc/h rather than ``KnnConfig``'s
    0, and the reason is measured: round 5b found the POOLED position-only
    statistic ranks arms at rho +0.91 against region subhalo counts -- the wrong
    sign, it is dominated by host and unbound particles -- while ``v / 4000``
    flips it to -0.65 (p 0.002) and needs no labels, which is the only kind of
    fix available on an unlabelled crop.

    ``n_queries`` is the only knob that matters for step time: cost is
    ``O(n_queries * N)`` for the search, with ``N = 64^3 = 262144`` per crop and
    two fields per call. The default is 256, a quarter of ``KnnConfig``'s 1024,
    and that is a COST decision, not a statistical one.

    ``n_quantiles`` defaults to 16, not ``KnnConfig``'s 64, and that is
    MEASURED too: job 68938 found one field supplying **33** interior queries
    against the other's 256. ``interior_queries`` cuts on EULERIAN position and
    at ``buffer_cells=8`` the interior half-width is 4.69 Mpc/h, while real
    z = 0 displacements run to several Mpc/h -- so most particles legitimately
    leave, and the candidate and target lose different fractions. ``knn_crop``
    clamps the bin count to the smaller draw, but a clamp that fires on most
    crops would let the bin count (and so the loss scale) vary crop to crop. 16
    is below the observed floor, so it binds instead.

    MEASURED, and it is affordable. On an a6000, inside the adversarial epoch
    so both arms do the same work: baseline **171 ms/step**, with this term
    **281 ms/step** -- a **1.64x** slowdown at 256 queries (jobs 68916 / 68949,
    from tensorboard wall-clock). For contrast the same forward takes 16.0 s on
    CPU, so this is only usable GPU-side; the backward is 0.23 s either way,
    i.e. the SEARCH dominates, not the tape.

    If a future config does more than double step time, drop ``n_queries``
    before dropping ``ks``: the ranks set what gets BUILT (max(ks) at or above
    the smallest object's particle count, so 64 at the very least for the
    50-200p deficit), whereas the query count only sets the sampling error of a
    distribution over draws.
    """

    def __init__(self, *, ks=(8, 32, 64, 128), n_queries: int = 256,
                 n_quantiles: int = 16, dead_zone: float = 0.02,
                 buffer_cells: float = 8.0, one_sided: bool = False,
                 phase_lambda: float = PHASE_LAMBDA_KMS_PER_MPC_H,
                 cell_mpc_h: float = CELL_MPC_H):
        super().__init__()
        self.cfg = KnnConfig(
            ks=tuple(int(k) for k in ks), n_queries=int(n_queries),
            n_quantiles=int(n_quantiles), dead_zone=float(dead_zone),
            buffer_cells=float(buffer_cells), one_sided=bool(one_sided),
        )
        self.phase_lambda = float(phase_lambda)
        self.cell_mpc_h = float(cell_mpc_h)

    def forward(self, output, target, style=None):
        del style                      # z = 0 only; the term is scale-free
        loss, _ = knn_crop_loss(
            output, target, self.cfg, cell_mpc_h=self.cell_mpc_h,
            phase_lambda=self.phase_lambda)
        return loss

    def extra_repr(self) -> str:
        return (f"ks={self.cfg.ks}, n_queries={self.cfg.n_queries}, "
                f"phase_lambda={self.phase_lambda}")


# --------------------------------------------------------------------------- #
# The member-gather term
# --------------------------------------------------------------------------- #
#: Undo factors of ``preprocess_srs.disnorm``/``velnorm`` at z = 0, which is the
#: only redshift these boxes have. The layout on disk is already normalised
#: (``paired_catnorm``), and the training command omits ``--*-norms`` entirely,
#: so these are the numbers that turn a stored field back into Mpc/h and km/s.
DIS_SCALE_MPC_H = 6.0
VEL_SCALE_KMS = 313.42210244571896
PARTICLE_MASS_MSUN_H = 581881454.8686146
NG_HR, TILE, BOXSIZE_MPC_H, SCALE_FACTOR = 512, 64, 100.0, 8


class MemberGather(nn.Module):
    """The member-gather loss, minimal set, on a FIXED batch of known tiles.

    Why this cannot use ``(output, target)`` like :class:`KnnCdf` does
    ----------------------------------------------------------------
    ``member_gather_loss`` gathers HR member particles **by flat Lagrangian
    id**: row ``b * 64^3 + (i * 64 + j) * 64 + k`` is the particle whose
    Lagrangian site is local cell ``(i, j, k)`` of ``tiles[b]``. That addressing
    needs to know which tile the field is and which way round it is. map2map
    hands the criterion a random 72^3 crop at an arbitrary offset, passed
    through a random ``flip``/``perm``, carrying no tile id -- so the rows would
    address different particles every step. ``KnnCdf`` survives this because a
    kNN distance distribution is invariant under those isometries
    (``knn-cdf-is-augmentation-invariant``); a catalog-indexed loss is not.

    So this term runs its own forward on its own canonical, unaugmented batch,
    and ignores the ``output``/``target`` it is handed. The gradient still
    reaches the generator -- it is the same parameters -- which is the whole
    point.

    The geometry, which is derived and then ASSERTED
    -----------------------------------------------
    ``srsgan.G`` is three ``SkipBlock``s at ``scale_factor=8``; each one
    upsamples by 2 (narrowing 1 per side) and applies two unpadded 3^3 convs
    (2 more per side), so ``n -> 2n - 6`` and ``out(n) = 8n - 42``. The training
    run's own log is the check: ``crop=9``, ``pad=3`` gives ``in = 15`` and
    ``8 * 15 - 42 = 78``, which is the output shape it printed.

    For a 64^3 tile that inverts to ``in = 14``: ``8 * 14 - 42 = 70``, then
    ``narrow_by(_, 3)`` trims to exactly 64. The LR window therefore starts at
    ``8 * t - pad`` on each axis (periodic), which puts the tile's own 8 LR
    cells at local ``[3, 11)`` -- the central 8 of 14, so the trimmed 64^3 is
    the tile and not a neighbour. ``_check_geometry`` asserts the output shape
    on the first call rather than trusting the arithmetic.

    Two terms are HR-referenced here, and in the fine-tune they were not
    --------------------------------------------------------------------
    ``loss-term-ablation`` measured the minimal set as **virial + centre +
    low-k guard** while fine-tuning the FROZEN SR2 field, where both the centre
    target (``centre_mode="self"``, the set's centroid under the frozen field)
    and the guard (don't drift from frozen at LR scale) are statements about a
    reference this run does not have. A from-scratch generator has not drifted
    from the frozen field; it has never been there. So:

    * ``centre_mode="full"`` charges ``||cbar - centre_target||``, the offset to
      the reachable HR centroid -- the quantity the Rockstar gate actually
      scores, and the same one ``"self"`` reports as ``off`` while charging
      something else.
    * the guard is :func:`sr2_scratch.lowk_anchor`, which is that module's
      from-scratch spelling of the same operator against HR.

    ``virial`` is unchanged: it was always referenced to HR's own value on the
    same set.

    Supervision is on **set8**, which is in ``M2M_TRAIN_BOXES``; the gate boxes
    set9/set10 are not touched here, so the held-out score stays honest. The
    supervision is in-sample by construction and no claim about generalisation
    comes from the training curve -- only from that gate.
    """

    def __init__(self, *, box: str = "set8", n_tiles: int = 4, pad: int = 3,
                 w_virial: float = 1.0, w_centre: float = 1.0,
                 w_low: float = 100.0, centre_mode: str = "full",
                 sets_per_step: int = 16, min_num_p: int = 200,
                 max_sets: int = 256, seed: int = 0, log_every: int = 200,
                 layout: str = None, split: str = "train"):
        super().__init__()
        import glob as _glob
        import os
        import numpy as np

        from cosmo_sr.eval.particle_identity import build_owner_index
        from cosmo_sr.eval.rockstar import load_rockstar_ascii
        from cosmo_sr.features.member_gather import (
            MemberGatherConfig, build_member_sets)

        self.box, self.pad = str(box), int(pad)
        self.w_low = float(w_low)
        self.sets_per_step = int(sets_per_step)
        self._model = None
        self._checked = False
        self._nstep = 0
        self._log_every = int(log_every)
        self._gen = torch.Generator().manual_seed(int(seed))

        layout = layout or os.environ.get(
            "M2M_LAYOUT", "/zfsauton/scratch/yixiz/DMSR/m2m_layout")
        root = os.environ.get(
            "DMSR_REWARD_ROOT", "/zfsauton/scratch/yixiz/DMSR/dmsr_reward")

        # --- the catalog, and which tiles hold the most supervised subhalos ---
        cat_hits = sorted(_glob.glob(
            f"{root}/halos/{box}__hr__hr/*_rockstar/halos_*.ascii"))
        if not cat_hits:
            raise SystemExit(f"MemberGather: no HR catalog for {box}")
        cat = load_rockstar_ascii(cat_hits[0])
        owner_p = f"{root}/halos_particles/{box}__hr__hr/{box}_hr_owner.npy"
        if not os.path.isfile(owner_p):
            raise SystemExit(f"MemberGather: no owner array at {owner_p}")
        oidx = build_owner_index(np.load(owner_p))

        from cosmo_sr.features.subhalo_gather import subhalo_home_tiles
        home = subhalo_home_tiles(cat, oidx, ng_hr=NG_HR, tile_hr=TILE,
                                  min_num_p=int(min_num_p), population="sub")
        # Rank tiles by how many resolved subhalos call them home. A tile with
        # no supervised set contributes nothing but a forward pass.
        counts = np.bincount(np.asarray(home["tile"], dtype=np.int64),
                             minlength=(NG_HR // TILE) ** 3)
        self.tiles = [int(t) for t in np.argsort(counts)[::-1][:int(n_tiles)]]
        if counts[self.tiles].sum() == 0:
            raise SystemExit("MemberGather: no supervised subhalos in any tile")

        # --- the fields: HR + frozen tiles, and the LR windows that make them -
        hr = self._tiles_from_box(
            f"{layout}/{split}/hr/{box}/disp.npy",
            f"{layout}/{split}/hr/{box}/vel.npy", self.tiles)
        froz_hits = sorted(_glob.glob(f"{root}/cache/sr2_base/{box}_seed0_*.npy"))
        if not froz_hits:
            raise SystemExit(f"MemberGather: no frozen box for {box} in cache")
        frozen_box = np.load(froz_hits[0], mmap_mode="r")
        base = torch.from_numpy(np.stack(
            [self._slice_tile(frozen_box, t) for t in self.tiles])).float()

        self.register_buffer("lr", self._lr_windows(
            f"{layout}/{split}/lr/{box}/disp.npy",
            f"{layout}/{split}/lr/{box}/vel.npy", self.tiles), persistent=False)
        self.register_buffer("hr", hr, persistent=False)

        # --- the sets ---------------------------------------------------------
        # Every weight not in the minimal set is zero, including the ones whose
        # dataclass defaults are 1.0 (`mgtrain-defaults-are-not-the-winning-arm`
        # is the same trap: an unset weight is not an off weight).
        self.cfg = MemberGatherConfig(
            w_virial=float(w_virial), w_centre=float(w_centre),
            centre_mode=str(centre_mode),
            w_bound=0.0, w_d6=0.0, w_rrms=0.0, w_sigmav=0.0,
            w_radius=0.0, w_shell=0.0, w_isolation=0.0,
            min_num_p=int(min_num_p), max_sets=int(max_sets),
            population="sub")
        self.kw = dict(ng_hr=NG_HR, tile_hr=TILE, boxsize_mpc_h=BOXSIZE_MPC_H,
                       dis_scale_mpc_h=DIS_SCALE_MPC_H,
                       vel_scale_kms=VEL_SCALE_KMS)
        report = {}
        self.sets = build_member_sets(
            cat, oidx, self.tiles, hr, base, self.cfg,
            particle_mass_msun_h=PARTICLE_MASS_MSUN_H,
            frozen_box=frozen_box, host_pos=None, report=report, **self.kw)
        if self.sets.n_sets == 0:
            raise SystemExit("MemberGather: build_member_sets returned 0 sets")
        print(f"[MemberGather] {box} tiles={self.tiles} "
              f"sets={self.sets.n_sets} centre_mode={centre_mode} "
              f"w_virial={w_virial} w_centre={w_centre} w_low={w_low}",
              flush=True)

    # -- data helpers ---------------------------------------------------------
    @staticmethod
    def _tile_origin(tid: int):
        n = NG_HR // TILE
        return (tid // (n * n)) * TILE, ((tid // n) % n) * TILE, (tid % n) * TILE

    @staticmethod
    def _slice_tile(box, tid: int):
        import numpy as np
        ox, oy, oz = MemberGather._tile_origin(int(tid))
        return np.asarray(box[:6, ox:ox + TILE, oy:oy + TILE, oz:oz + TILE])

    def _tiles_from_box(self, disp_p: str, vel_p: str, tiles):
        import numpy as np
        d = np.load(disp_p, mmap_mode="r")
        v = np.load(vel_p, mmap_mode="r")
        out = []
        for t in tiles:
            ox, oy, oz = self._tile_origin(int(t))
            sl = (slice(ox, ox + TILE), slice(oy, oy + TILE), slice(oz, oz + TILE))
            out.append(np.concatenate([np.asarray(d[(slice(None),) + sl]),
                                       np.asarray(v[(slice(None),) + sl])], 0))
        return torch.from_numpy(np.stack(out)).float()

    def _lr_windows(self, disp_p: str, vel_p: str, tiles):
        """``(B, 6, 14, 14, 14)``, periodic, centred on each tile's 8 LR cells."""
        import numpy as np
        d, v = np.load(disp_p), np.load(vel_p)
        lr = np.concatenate([d, v], axis=0)                 # (6, 64, 64, 64)
        n_lr, w = lr.shape[-1], TILE // SCALE_FACTOR + 2 * self.pad
        out = []
        for t in tiles:
            ox, oy, oz = self._tile_origin(int(t))
            ix = (np.arange(w) + ox // SCALE_FACTOR - self.pad) % n_lr
            iy = (np.arange(w) + oy // SCALE_FACTOR - self.pad) % n_lr
            iz = (np.arange(w) + oz // SCALE_FACTOR - self.pad) % n_lr
            out.append(lr[:, ix[:, None, None], iy[None, :, None],
                          iz[None, None, :]])
        return torch.from_numpy(np.stack(out)).float()

    def _apply(self, fn, *a, **k):
        """Carry ``MemberSets`` along with the buffers on ``.to(device)``.

        The sets are plain tensors in a dataclass, not parameters or buffers, so
        ``nn.Module.to`` does not see them -- and ``train.py`` moves the
        criterion with exactly that call. Without this the loss would index a
        CUDA particle table with CPU rows on the first step.
        """
        out = super()._apply(fn, *a, **k)
        self.sets = self.sets.to(self.lr.device)
        return out

    # -- the term -------------------------------------------------------------
    def set_model(self, model):
        """The generator to run the member batch through.

        Unwrapped from DDP deliberately: this is a SECOND forward inside one
        training step, and going through the wrapper would re-arm the reducer's
        per-iteration bookkeeping. The parameters are the same objects either
        way, so the single ``backward()`` still produces the gradients DDP
        synchronises.
        """
        self._model = getattr(model, "module", model)

    def _check_geometry(self, cand):
        if self._checked:
            return
        self._checked = True
        exp = (len(self.tiles), 6, TILE, TILE, TILE)
        if tuple(cand.shape) != exp:
            raise RuntimeError(
                f"MemberGather geometry: narrowed candidate is "
                f"{tuple(cand.shape)}, expected {exp}. The LR window is "
                f"{tuple(self.lr.shape)} and out(n) = 8n - 42 should give "
                f"{8 * self.lr.shape[-1] - 42} before narrow_by({self.pad}). "
                "A different generator or scale_factor invalidates the "
                "Lagrangian row addressing, so this refuses rather than "
                "supervising the wrong particles.")

    def forward(self, output, target, style=None):
        del output, target, style      # see the class docstring
        from map2map.models.narrow import narrow_by

        from cosmo_sr.features.member_gather import (
            member_gather_loss, tile_particles)
        from cosmo_sr.train.sr2_scratch import lowk_anchor

        if self._model is None:
            raise RuntimeError(
                "MemberGather needs the generator: train.py must call "
                "extra_criterion.set_model(model) before the loop.")
        cand = narrow_by(self._model(self.lr), self.pad)
        self._check_geometry(cand)

        pos, vel = tile_particles(cand, self.tiles, **self.kw)
        n = self.sets.n_sets
        idx = (torch.randperm(n, generator=self._gen)[:self.sets_per_step]
               .tolist() if self.sets_per_step and self.sets_per_step < n
               else None)
        gather, _ = member_gather_loss(pos, vel, self.sets, self.cfg, idx)
        g = lowk_anchor(cand, self.hr, SCALE_FACTOR)
        total = gather + self.w_low * g["low"]
        # The SPLIT, not just the sum. `w_low` is 100 against terms normalised
        # to O(1), so the guard can carry most of the term while the member
        # objective -- the thing the arm exists to test -- contributes a few
        # percent. A single logged total cannot distinguish that from a
        # balanced term, and the whole arm would be mislabelled.
        self._nstep += 1
        if self._log_every and self._nstep % self._log_every == 1:
            print(f"[MemberGather] step={self._nstep} "
                  f"gather={float(gather):.4g} low={float(g['low']):.4g} "
                  f"w_low*low={self.w_low * float(g['low']):.4g} "
                  f"total={float(total):.4g}", flush=True)
        return total

    def extra_repr(self) -> str:
        return (f"box={self.box}, tiles={self.tiles}, "
                f"sets={self.sets.n_sets}, w_low={self.w_low}")
