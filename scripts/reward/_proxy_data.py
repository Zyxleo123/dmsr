"""Shared plumbing for the two-arm proxy comparison: table, context, metrics.

The trainer and the benchmark must measure the *same* things the same way, or
the gate is scoring a different quantity from the one that was optimised. They
therefore both import from here rather than the benchmark reaching into the
trainer's private helpers.

The one structure worth explaining is :class:`RowContext`. Every reward here is
a whole-box quantity, so a tile's ``dR`` is "what happens to this box's reward
when this tile's summary is swapped in for the frozen one". Evaluating that for
``n`` rows needs, for each row, its own box's pooled frozen summary and its own
``(box, tile)``'s frozen summary. Those are *fixed* -- they come from the labels,
not from the model -- so they are built once as tensors and reused for every
epoch and every ensemble member. Rebuilding them inside the training loop, as an
earlier version did, made one epoch cost a Python pass over fifty thousand rows
and put the model selection out of reach.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import json

import numpy as np
import torch

from _sr2_direct import direct_root, labels_complete_path, read_jsonl  # noqa: E402

from cosmo_sr.reward.arms import (  # noqa: E402
    FEATURE_SCHEMA_VERSION, arm_storage, features_key, sidecar_file, tokens_key,
)
from cosmo_sr.reward.catalog_proxy import spearman, tie_aware_agreement  # noqa: E402
from cosmo_sr.reward.phase_space import ARMS  # noqa: E402
from cosmo_sr.reward.regions import region_ids_from_tile_ids  # noqa: E402
from cosmo_sr.reward.torch_reward import TorchSummary  # noqa: E402

COUNT_KEYS = ("n_sub", "n_host", "occ_numerator")


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def load_rows(path: Path, *, require_complete: bool = True) -> List[Dict]:
    """The joined table, refusing a partial one unless explicitly allowed.

    ``labels_complete.json`` is written by the single indexing job once every
    predeclared candidate has a valid label. Fitting without it means fitting on
    whatever happened to have finished, which changes the dataset between runs
    and makes two fits incomparable -- including the two arms, if they were
    started at different times.

    When the marker is required it must also carry the current
    ``FEATURE_SCHEMA_VERSION``. A schema-2 marker with a schema-3 trainer would
    silently train A–D without ``field_changed`` weights and crash (or skip) E.
    """
    if not path.is_file():
        raise SystemExit(
            f"no proxy table at {path}; run "
            "scripts/reward/collect_catalog_proxy_data.py --stage index first")
    marker = labels_complete_path()
    if require_complete and not marker.is_file():
        raise SystemExit(
            f"{marker} is absent: labelling is not complete, so "
            f"{path} is a partial table. Re-run the indexing job once every "
            "candidate has been labelled, or pass --allow-incomplete if you "
            "really mean to fit on a subset (nothing gated may use the result).")
    if require_complete and marker.is_file():
        try:
            doc = json.loads(marker.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"unreadable {marker}: {exc}") from exc
        got = int(doc.get("feature_schema_version", 1))
        if got != int(FEATURE_SCHEMA_VERSION):
            raise SystemExit(
                f"{marker} is feature schema {got}, but the code expects "
                f"{FEATURE_SCHEMA_VERSION}. Re-run the features-only backfill "
                "(`bash scripts/slurm/submit_proxy_labels.sh all`) and re-index "
                "before fitting; do not train on a stale table.")
    return read_jsonl(path)


def unit_ids_of(tile_ids: Sequence[int], *, width: int = 1,
                n_per_axis: int = 8) -> np.ndarray:
    """The supervision-unit id of each row, from its tile id.

    THE one grouping helper: the training pairs and every evaluation metric
    must group by the same array, so both call this rather than deriving the
    mapping twice. At the deployed ``width = 1`` it is the identity
    (``region_id == tile_id``); at any coarser width it is the origin-0
    periodic partition of :func:`cosmo_sr.reward.regions.region_ids_from_tile_ids`.
    """
    ids = np.asarray(tile_ids, dtype=np.int64)
    if int(width) == 1:
        return ids
    return region_ids_from_tile_ids(ids, width=int(width),
                                    n_per_axis=int(n_per_axis))


def _sidecar_features(rows: Sequence[Mapping], arm: str,
                      table_dir: Optional[Path]) -> np.ndarray:
    """One sidecar arm's ``(n, 2, T, F)`` token blocks for these rows.

    The sidecar is one memory-mappable array over the FULL table, indexed by
    each row's ``row_id`` (its position in ``rows.jsonl``, stamped by the
    indexer). Rows are usually a filtered subset -- a box split -- so the
    ``row_id`` is what keeps a row and its tokens paired through any filter.
    """
    d = Path(table_dir) if table_dir else direct_root("proxy_data")
    side = d / tokens_key(arm)
    if not side.is_file():
        raise SystemExit(
            f"arm '{arm}' stores its token features in {side}, which does not "
            "exist. Re-run collect_catalog_proxy_data.py --stage index with "
            "the current code (it writes the sidecar next to rows.jsonl).")
    if rows and "row_id" not in rows[0]:
        raise SystemExit(
            "the table's rows carry no 'row_id', so they cannot be joined to "
            f"the {side.name} sidecar. The table predates the arm-C schema; "
            "re-run collect_catalog_proxy_data.py --stage index.")
    tokens = np.load(side, mmap_mode="r")
    idx = np.asarray([int(r["row_id"]) for r in rows], dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= tokens.shape[0]):
        raise SystemExit(
            f"row_id range [{idx.min()}, {idx.max()}] does not fit the "
            f"{tokens.shape[0]}-row sidecar {side}; the table and the sidecar "
            "are from different index runs. Re-run --stage index.")
    return np.asarray(tokens[idx], dtype=np.float64)


def as_arrays(rows: Sequence[Mapping], arm: str = "a",
              table_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Metadata columns and (for a cached arm) the per-tile feature array.

    The ``features`` array is materialised for the inline arms ``a``/``b`` and the
    sidecar arms ``c``/``d``/``e`` -- but for the dense arm ``e`` that is a float64
    copy of the whole grid, which is ~160 GB on the fit rows. Callers that iterate
    the full table (the trainer, gate, benchmark) must therefore use
    :func:`make_arm_features` and stream chunks; ``as_arrays`` features are for the
    metadata and for small-set callers. The field arm ``f`` has no cached block
    and raises here -- it is only reachable through :func:`make_arm_features`.
    """
    storage = arm_storage(arm)
    if storage == "field":
        raise SystemExit(
            f"arm {arm!r} streams its input from raw fields; it has no cached "
            "feature array. Use make_arm_features(arm, rows, table_dir, cfg).get(idx).")
    key = features_key(arm)
    if storage == "sidecar":
        feats = _sidecar_features(rows, arm, table_dir)
    else:
        if rows and key not in rows[0]:
            raise SystemExit(
                f"the table has no '{key}' column. It was written before the arm "
                "comparison existed; re-run collect_catalog_proxy_data.py "
                "--stage index against candidates generated by the current code.")
        feats = np.asarray([r[key] for r in rows], dtype=np.float64)
    return {
        "features": feats,
        "n_sub": np.asarray([r["n_sub"] for r in rows], dtype=np.float64),
        "n_host": np.asarray([r["n_host"] for r in rows], dtype=np.float64),
        "occ_numerator": np.asarray([r["occ_numerator"] for r in rows],
                                    dtype=np.float64),
        "volume": np.asarray([r["volume_mpc3"] for r in rows], dtype=np.float64),
        "box": np.asarray([str(r["box"]) for r in rows]),
        "tag": np.asarray([str(r["tag"]) for r in rows]),
        "tile_id": np.asarray([int(r["tile_id"]) for r in rows], dtype=np.int64),
        "source": np.asarray([str(r["source"]) for r in rows]),
        "mode": np.asarray([str(r.get("mode", "both")) for r in rows]),
        "alpha": np.asarray([np.nan if r.get("alpha") is None else float(r["alpha"])
                             for r in rows], dtype=np.float64),
        # The arm-neutral changed-from-frozen flag (schema >= 3); absent rows
        # (an older table) read as "unchanged", which is the safe default.
        "field_changed": np.asarray([bool(int(r.get("field_changed", 0)))
                                     for r in rows], dtype=bool),
    }


# --------------------------------------------------------------------------- #
# Lazy per-arm feature access
# --------------------------------------------------------------------------- #
class ArmFeatures:
    """Index a chunk of rows -> a feature tensor, without materialising the whole.

    The arms differ in where their features live: an in-memory array (``a``/``b``),
    a memory-mapped sidecar (``c``/``d``/``e``), or built on the fly from raw
    fields (``f``). The trainer, gate and benchmark all want the same thing -- give
    me these rows' features on this device -- and must never hold the full tensor
    (arm ``e`` is ~160 GB as float64, arm ``f`` ~800 GB). This hides the storage.
    """

    def __len__(self) -> int:                       # pragma: no cover - interface
        raise NotImplementedError

    @property
    def per_row_shape(self) -> Tuple[int, ...]:     # pragma: no cover - interface
        raise NotImplementedError

    def get(self, idx: np.ndarray, device=None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def standardizer_sample(self, idx: Optional[np.ndarray] = None, *,
                            max_rows: int = 2048, seed: int = 0) -> np.ndarray:
        """A capped subset of rows' features (float32 ndarray) for fitting stats.

        The full grid is too large to fit a standardiser on; a few thousand rows
        estimate a per-channel mean/std perfectly well. Drawn from the given rows
        (the drawn bootstrap rows) so the standardiser is fitted on the training
        sample, exactly as the flat arms fit theirs.
        """
        pool = (np.arange(len(self)) if idx is None
                else np.asarray(idx, dtype=np.int64))
        if pool.size > int(max_rows):
            pool = np.random.default_rng(int(seed)).choice(
                pool, size=int(max_rows), replace=False)
        return self.get(np.sort(pool)).cpu().numpy()


class InlineArmFeatures(ArmFeatures):
    """Features held in an in-memory ``(n, F)`` array (arms ``a``/``b``)."""

    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.array.shape[0])

    @property
    def per_row_shape(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in self.array.shape[1:])

    def get(self, idx: np.ndarray, device=None) -> torch.Tensor:
        t = torch.as_tensor(self.array[np.asarray(idx, dtype=np.int64)],
                            dtype=torch.float32)
        return t.to(device) if device is not None else t


class SidecarArmFeatures(ArmFeatures):
    """Features memory-mapped from a global sidecar, indexed by ``row_id``.

    The sidecar spans the FULL table; the rows here are usually a filtered subset
    (a box split), so each local index maps through ``row_ids`` to the global row.
    Never upcast beyond float32 -- arm E's grid is float16 on disk and stays small.
    """

    def __init__(self, path, row_ids: Sequence[int]):
        self.path = str(path)
        self.row_ids = np.asarray(row_ids, dtype=np.int64)
        self._mm: Optional[np.ndarray] = None

    @property
    def mm(self) -> np.ndarray:
        if self._mm is None:
            if not Path(self.path).is_file():
                raise SystemExit(
                    f"sidecar {self.path} is missing; re-run "
                    "collect_catalog_proxy_data.py --stage index.")
            self._mm = np.load(self.path, mmap_mode="r")
        return self._mm

    def __len__(self) -> int:
        return int(self.row_ids.size)

    @property
    def per_row_shape(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in self.mm.shape[1:])

    def get(self, idx: np.ndarray, device=None) -> torch.Tensor:
        gids = self.row_ids[np.asarray(idx, dtype=np.int64)]
        block = np.asarray(self.mm[gids], dtype=np.float32)
        t = torch.as_tensor(block, dtype=torch.float32)
        return t.to(device) if device is not None else t


class FieldArmFeatures(ArmFeatures):
    """Arm F: build the 20-channel SR2 critic input per tile from raw fields.

    No cache: for each row it slices the candidate ``field.npy`` and the LR box,
    upsamples the LR sub-region, CIC-deposits the fine density, and assembles the
    exact :func:`cosmo_sr.reward.sr2_adversarial.critic_input`. Rows are grouped by
    candidate so each 3.2 GB ``field.npy`` is opened once per chunk, and one LR box
    is cached at a time. Built on ``device`` when given, so the CIC density runs on
    the GPU rather than being copied to it.
    """

    def __init__(self, rows: Sequence[Mapping], *, field_path, lr_loader, tile_grid,
                 cellsize_kpc_h: float, dis_norm_kpc_h: float, grid_mult: int,
                 scale_factor: int, in_channels: int = 20):
        self.box = [str(r["box"]) for r in rows]
        self.tag = [str(r["tag"]) for r in rows]
        self.tile_id = [int(r["tile_id"]) for r in rows]
        self.field_path = field_path            # (box, tag) -> Path
        self.lr_loader = lr_loader              # box -> (6, ng_lr, ng_lr, ng_lr)
        self.grid = tile_grid
        self.cellsize = float(cellsize_kpc_h)
        self.dis_norm = float(dis_norm_kpc_h)
        self.grid_mult = int(grid_mult)
        self.scale = int(scale_factor)
        self.in_channels = int(in_channels)
        self._lr_box: Optional[str] = None
        self._lr: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.tile_id)

    @property
    def per_row_shape(self) -> Tuple[int, ...]:
        s = int(self.grid.tile_hr)
        return (self.in_channels, s, s, s)

    def _lr_for(self, box: str) -> np.ndarray:
        if self._lr_box != box:
            self._lr = np.asarray(self.lr_loader(box), dtype=np.float32)
            self._lr_box = box
        return self._lr

    def get(self, idx: np.ndarray, device=None) -> torch.Tensor:
        from cosmo_sr.reward.sr2_adversarial import critic_input

        idx = np.asarray(idx, dtype=np.int64)
        dev = device if device is not None else torch.device("cpu")
        out: List[Optional[torch.Tensor]] = [None] * idx.size
        # Group by candidate so each field.npy opens once.
        order = sorted(range(idx.size),
                       key=lambda k: (self.box[idx[k]], self.tag[idx[k]]))
        cur: Optional[Tuple[str, str]] = None
        field = None
        for k in order:
            r = int(idx[k])
            b, t, tid = self.box[r], self.tag[r], self.tile_id[r]
            if (b, t) != cur:
                field = np.load(self.field_path(b, t), mmap_mode="r")
                cur = (b, t)
            sx, sy, sz = self.grid.slices(tid)
            ftile = torch.as_tensor(
                np.asarray(field[:, sx, sy, sz], dtype=np.float32), device=dev)
            lr = self._lr_for(b)
            lslice = tuple(slice(s.start // self.scale, s.stop // self.scale)
                           for s in (sx, sy, sz))
            lrtile = torch.as_tensor(
                np.asarray(lr[:, lslice[0], lslice[1], lslice[2]], dtype=np.float32),
                device=dev)
            ci = critic_input(lrtile.unsqueeze(0), ftile.unsqueeze(0),
                              cellsize_kpc_h=self.cellsize,
                              dis_norm_kpc_h=self.dis_norm, grid_mult=self.grid_mult)
            out[k] = ci[0]
        return torch.stack(out, dim=0)


def make_arm_features(arm: str, rows: Sequence[Mapping],
                      table_dir: Optional[Path] = None,
                      cfg: Optional[Mapping] = None) -> ArmFeatures:
    """The right :class:`ArmFeatures` for an arm's storage kind.

    ``cfg`` is required only for the field arm ``f`` (it needs the geometry and
    the LR/field roots to build the critic input); the cached arms ignore it.
    """
    storage = arm_storage(arm)
    if storage == "inline":
        return InlineArmFeatures(as_arrays(rows, arm, table_dir)["features"])
    if storage == "sidecar":
        d = Path(table_dir) if table_dir else direct_root("proxy_data")
        if rows and "row_id" not in rows[0]:
            raise SystemExit(
                "the table's rows carry no 'row_id', so they cannot be joined to "
                f"the {sidecar_file(arm)} sidecar. Re-run --stage index.")
        return SidecarArmFeatures(d / sidecar_file(arm),
                                  [int(r["row_id"]) for r in rows])
    if cfg is None:
        raise SystemExit(f"arm {arm!r} streams raw fields; make_arm_features needs cfg")
    from _sr2_direct import (geometry_of, load_lr, soft_config_of,  # noqa: E402
                             tile_grid_of)
    scfg = soft_config_of(cfg)
    grid_mult = int(dict(cfg.get("adversarial", {})).get("density_grid_mult", 2))
    return FieldArmFeatures(
        rows,
        field_path=lambda b, t: direct_root("candidates", f"{b}__{t}") / "field.npy",
        lr_loader=lambda b: load_lr(cfg, b),
        tile_grid=tile_grid_of(cfg),
        cellsize_kpc_h=float(scfg.cellsize_kpc_h),
        dis_norm_kpc_h=float(scfg.dis_norm_kpc_h),
        grid_mult=grid_mult,
        scale_factor=int(geometry_of(cfg).scale_factor),
        in_channels=12 + grid_mult ** 3)


# --------------------------------------------------------------------------- #
# Reward context
# --------------------------------------------------------------------------- #
@dataclass
class RowContext:
    """Per-row frozen and whole-box references, as tensors, built once."""

    box: TorchSummary
    frozen: TorchSummary
    measured: TorchSummary
    keep: np.ndarray

    @property
    def tile_volume(self) -> torch.Tensor:
        """The row's own tile volume -- what a *tile* summary must be labelled with.

        ``swap_summary`` keeps the box volume and ignores the predicted tile's,
        so this makes no numerical difference to any ``dR``. It is still the
        right value: a per-tile summary carrying the whole box's volume is a
        number density wrong by 512, waiting for the first piece of code that
        reads it.
        """
        return self.measured.volume_mpc3

    def index(self, idx: np.ndarray) -> "RowContext":
        t = torch.as_tensor(np.asarray(idx, dtype=np.int64),
                            device=self.box.n_sub.device)

        def take(s: TorchSummary) -> TorchSummary:
            return TorchSummary(s.n_sub[t], s.n_host[t], s.occ_numerator[t],
                                s.volume_mpc3[t])

        return RowContext(take(self.box), take(self.frozen), take(self.measured),
                          self.keep[np.asarray(idx, dtype=np.int64)])

    def to(self, device) -> "RowContext":
        """Move the reference summaries to ``device`` (the reward math runs there).

        The proxy predicts on ``device``, and ``delta_reward_swap`` mixes the
        prediction with these frozen/box references -- so on CUDA they must live
        there too or every swap raises a device mismatch. ``keep`` stays a NumPy
        mask; it only ever indexes host-side.
        """
        return RowContext(self.box.to(device), self.frozen.to(device),
                          self.measured.to(device), self.keep)


def build_row_context(rows: Sequence[Mapping]) -> RowContext:
    """``S_box``, ``s_frozen`` and ``s_measured`` for every row, aligned.

    A row whose ``(box, tile)`` has no frozen candidate cannot have a
    baseline-relative reward at all. Those rows are kept in place -- dropping
    them would misalign every array against ``rows`` -- with placeholder
    references and ``keep = False``, and every ``dR`` computed for them comes
    out NaN and is excluded downstream.
    """
    frozen_row: Dict[Tuple[str, int], Mapping] = {}
    for r in rows:
        if r["source"] == "frozen":
            frozen_row[(r["box"], int(r["tile_id"]))] = r

    box_acc: Dict[str, Dict[str, np.ndarray]] = {}
    for r in rows:
        if r["source"] != "frozen":
            continue
        a = box_acc.setdefault(r["box"], {k: 0.0 for k in COUNT_KEYS} | {"vol": 0.0})
        for k in COUNT_KEYS:
            a[k] = a[k] + np.asarray(r[k], dtype=np.float64)
        a["vol"] = a["vol"] + float(r["volume_mpc3"])

    cols = {n: {k: [] for k in COUNT_KEYS} | {"vol": []}
            for n in ("box", "frozen", "measured")}
    keep = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        f = frozen_row.get((r["box"], int(r["tile_id"])))
        b = box_acc.get(r["box"])
        keep[i] = f is not None and b is not None
        if f is None:
            f = {k: np.zeros_like(np.asarray(r[k], dtype=np.float64))
                 for k in COUNT_KEYS} | {"volume_mpc3": r["volume_mpc3"]}
        if b is None:
            b = {k: np.ones_like(np.asarray(r[k], dtype=np.float64))
                 for k in COUNT_KEYS} | {"vol": r["volume_mpc3"]}
        for k in COUNT_KEYS:
            cols["box"][k].append(np.asarray(b[k], dtype=np.float64))
            cols["frozen"][k].append(np.asarray(f[k], dtype=np.float64))
            cols["measured"][k].append(np.asarray(r[k], dtype=np.float64))
        cols["box"]["vol"].append(float(b["vol"]))
        cols["frozen"]["vol"].append(float(f["volume_mpc3"]))
        cols["measured"]["vol"].append(float(r["volume_mpc3"]))

    def stack(name: str) -> TorchSummary:
        d = cols[name]
        return TorchSummary(
            n_sub=torch.tensor(np.asarray(d["n_sub"]), dtype=torch.float64),
            n_host=torch.tensor(np.asarray(d["n_host"]), dtype=torch.float64),
            occ_numerator=torch.tensor(np.asarray(d["occ_numerator"]),
                                       dtype=torch.float64),
            volume_mpc3=torch.tensor(np.asarray(d["vol"]), dtype=torch.float64))

    return RowContext(stack("box"), stack("frozen"), stack("measured"), keep)


def delta_of_summary(pred: TorchSummary, ctx: RowContext, reward_t, *,
                     w_joint: float, w_occ: float,
                     key: str = "dR_combined") -> torch.Tensor:
    """``(n,)`` ``dR`` from swapping ``pred`` into each row's box, NaN where invalid."""
    out = reward_t.delta_reward_swap(ctx.box, ctx.frozen, pred,
                                     w_joint=w_joint, w_occ=w_occ)[key]
    keep = torch.as_tensor(ctx.keep, device=out.device)
    return torch.where(keep, out, torch.full_like(out, float("nan")))


def predicted_delta(member, feats: torch.Tensor, ctx: RowContext, reward_t, *,
                    w_joint: float, w_occ: float,
                    key: str = "dR_combined") -> torch.Tensor:
    # ctx.frozen is the measured frozen tile summary the residual head reconstructs
    # against; passing it is what makes a residual proxy predict a *change* from
    # frozen rather than an absolute count (an absolute-count checkpoint ignores it).
    return delta_of_summary(member.summary(feats, ctx.tile_volume, ctx.frozen), ctx,
                            reward_t, w_joint=w_joint, w_occ=w_occ, key=key)


def true_delta_rewards(ctx: RowContext, reward_t, *, w_joint: float,
                       w_occ: float, key: str = "dR_combined") -> np.ndarray:
    """The ranking target: the **real** ``dR`` this candidate's tile would give.

    Computed from the measured tile summary, not from a prediction, so a proxy
    that ranks these correctly has learned the thing the actor is scored on.
    """
    with torch.no_grad():
        return delta_of_summary(ctx.measured, ctx, reward_t, w_joint=w_joint,
                                w_occ=w_occ, key=key).detach().cpu().numpy()


def ensemble_delta(members, feats: torch.Tensor, ctx: RowContext, reward_t, *,
                   w_joint: float, w_occ: float,
                   key: str = "dR_combined") -> Tuple[np.ndarray, np.ndarray]:
    """``(mean, std)`` over ensemble members of the predicted ``dR``.

    The spread is the quantity the actor's lower confidence bound subtracts, so
    it is returned next to the mean rather than recomputed somewhere else with a
    different ``unbiased`` convention.
    """
    with torch.no_grad():
        per = torch.stack([
            predicted_delta(m, feats, ctx, reward_t, w_joint=w_joint,
                            w_occ=w_occ, key=key) for m in members])
    return (per.mean(dim=0).detach().cpu().numpy(),
            per.std(dim=0, unbiased=False).detach().cpu().numpy() if per.shape[0] > 1
            else np.zeros(per.shape[1]))


def stream_ensemble_delta(provider: "ArmFeatures", members, ctx: RowContext,
                          reward_t, *, w_joint: float, w_occ: float,
                          key: str = "dR_combined", chunk_rows: int = 2048,
                          device=None, rows: Optional[np.ndarray] = None,
                          transform=None) -> Tuple[np.ndarray, np.ndarray]:
    """``(mean, std)`` predicted ``dR`` over the ensemble, streamed in chunks.

    The provider-and-chunk version of :func:`ensemble_delta`: it never holds more
    than ``chunk_rows`` rows of features, so it is the only form usable for the
    dense arm ``e`` and the streamed arm ``f``. ``ctx`` must already be on
    ``device`` (see :meth:`RowContext.to`).

    ``rows`` restricts evaluation to a subset (the rest come back NaN); ``transform``
    is an optional ``(xi) -> xi`` applied to each chunk's features, used by the
    feature-ablation check to replace one input channel with its mean.
    """
    n = len(provider)
    order = (np.arange(n) if rows is None else np.asarray(rows, dtype=np.int64))
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, order.size, int(chunk_rows)):
            idx = order[s:s + int(chunk_rows)]
            xi = provider.get(idx, device=device)
            if transform is not None:
                xi = transform(xi)
            sub = ctx.index(idx)
            per = torch.stack([
                predicted_delta(m, xi, sub, reward_t, w_joint=w_joint,
                                w_occ=w_occ, key=key) for m in members])
            mean[idx] = per.mean(dim=0).double().cpu().numpy()
            std[idx] = (per.std(dim=0, unbiased=False).double().cpu().numpy()
                        if per.shape[0] > 1 else 0.0)
    return mean, std


def channel_mean_transform(provider: "ArmFeatures", channel: int, storage: str, *,
                           rows: Optional[np.ndarray] = None, sample: int = 256,
                           device=None):
    """A ``(xi) -> xi`` that replaces one input channel with its global mean.

    The unit of "one feature" per storage: a column of a flat vector; a
    (candidate/difference slot, token-feature) channel of the ``(2, T, F)`` token
    grid; a channel of the ``(2, C, D, D, D)`` grid; a channel of the 20-channel
    critic input. The mean is estimated from a small sample of rows so the
    ablation costs one extra pass, not one per channel.
    """
    pool = (np.arange(len(provider)) if rows is None
            else np.asarray(rows, dtype=np.int64))
    if pool.size > int(sample):
        pool = np.random.default_rng(0).choice(pool, size=int(sample), replace=False)
    x = provider.get(np.sort(pool), device=device)

    if storage == "inline":
        val = x[:, channel].mean()

        def tf(xi):
            xi = xi.clone(); xi[:, channel] = val; return xi
    elif storage == "sidecar" and x.dim() == 4:            # tokens (B, 2, T, F)
        f = x.shape[-1]
        slot, feat = divmod(int(channel), int(f))
        val = x[:, slot, :, feat].mean()

        def tf(xi):
            xi = xi.clone(); xi[:, slot, :, feat] = val; return xi
    elif storage == "sidecar":                             # grid (B, 2, C, D,D,D)
        c = x.shape[2]
        slot, ch = divmod(int(channel), int(c))
        val = x[:, slot, ch].mean()

        def tf(xi):
            xi = xi.clone(); xi[:, slot, ch] = val; return xi
    else:                                                  # field (B, 20, N,N,N)
        val = x[:, channel].mean()

        def tf(xi):
            xi = xi.clone(); xi[:, channel] = val; return xi
    return tf


def stream_pred_counts(provider: "ArmFeatures", members, ctx: RowContext, *,
                       chunk_rows: int = 2048, device=None) -> Dict[str, np.ndarray]:
    """Mean-over-members predicted counts per row, streamed. ``{key: (n, bins)}``.

    Feeds the per-candidate pooled metric without materialising the features. The
    residual head reconstructs against ``ctx.frozen``, so ``ctx`` must be indexed
    (and on the model's device) per chunk, which :meth:`RowContext.index` does.
    """
    n = len(provider)
    out: Dict[str, Optional[np.ndarray]] = {k: None for k in COUNT_KEYS}
    with torch.no_grad():
        for s in range(0, n, int(chunk_rows)):
            idx = np.arange(s, min(s + int(chunk_rows), n))
            xi = provider.get(idx, device=device)
            sub = ctx.index(idx)
            preds = [m.summary(xi, sub.tile_volume, sub.frozen) for m in members]
            for k in COUNT_KEYS:
                stacked = torch.stack([getattr(p, k) for p in preds]).mean(0)
                arr = stacked.double().cpu().numpy()
                if out[k] is None:
                    out[k] = np.empty((n, arr.shape[1]), dtype=np.float64)
                out[k][idx] = arr
    return {k: v for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Row weighting
# --------------------------------------------------------------------------- #
def changed_tile_mask(features: np.ndarray, *, threshold: float) -> np.ndarray:
    """Which rows differ from their frozen reference by more than noise.

    Read off the *difference* half of the feature vector -- the second half by
    construction -- in units of each difference coordinate's own spread, so it
    means the same thing for an intervention (whose mask says which tiles it
    touched) and for an actor candidate (which has no mask at all). A tile the
    edit did not reach is a row whose label is the frozen label and whose
    ranking pair is a tie; the point of finding them is to stop them being most
    of the training signal.
    """
    x = np.asarray(features, dtype=np.float64)
    half = x.shape[1] // 2
    diff = x[:, half:]
    sd = diff.std(axis=0)
    sd = np.where(sd < 1e-12, np.inf, sd)
    return np.max(np.abs(diff) / sd, axis=1) > float(threshold)


def row_weights(source: np.ndarray, changed: np.ndarray, *,
                balance_sources: bool = True,
                changed_weight: float = 3.0) -> Dict[str, np.ndarray]:
    """Per-row training weights, normalised to mean 1.

    Two corrections, both aimed at the same failure: HR and frozen are five of
    every eight candidates and are the easy ones, so an unweighted loss is
    mostly the statement "HR has more substructure than SR2" -- true, already
    known, and not a direction the actor can move in.

    ``changed`` is the arm-neutral ``field_changed`` flag (read off the raw
    six-channel field, so it catches velocity-only edits a density feature would
    miss). Changed rows are up-weighted by ``changed_weight`` (3x) and then each
    source is renormalised back to its balanced total, so the 3x redistributes a
    source's weight *toward the tiles an edit actually reached* without changing
    how much that source contributes overall. A source whose rows are uniformly
    changed (HR, frozen_seed) is therefore unaffected -- the emphasis only bites
    where a source mixes touched and untouched tiles (interventions, actor).
    """
    src = np.asarray(source)
    changed = np.asarray(changed, dtype=bool)
    n = src.size
    w = np.ones(n, dtype=np.float64)
    if not n:
        return {"weight": w, "changed": changed}

    uniq, counts = np.unique(src, return_counts=True)
    if balance_sources:
        per = {s: (n / len(uniq)) / c for s, c in zip(uniq, counts)}
        w *= np.asarray([per[s] for s in src], dtype=np.float64)

    if changed_weight != 1.0:
        w = w * np.where(changed, float(changed_weight), 1.0)
        if balance_sources:
            # Restore each source to the equal total it had before the 3x, so the
            # up-weighting is purely within-source.
            target = n / len(uniq)
            for s in uniq:
                m = src == s
                ssum = float(w[m].sum())
                if ssum > 0:
                    w[m] *= target / ssum

    w = w / max(float(w.mean()), 1e-12)
    return {"weight": w, "changed": changed}


def pair_weights(pairs: np.ndarray, kinds: np.ndarray, mult: np.ndarray,
                 changed: np.ndarray, *, type_weights: Mapping[int, float],
                 changed_weight: float = 3.0) -> np.ndarray:
    """``w_pair = box_bootstrap_mult * pair_type * (changed_weight if either changed)``.

    Replaces the old ``min(weight_a, weight_b)``. Both endpoints of a within-
    ``(box, unit)`` pair share a box, so ``mult[a] == mult[b]`` -- the box's
    bootstrap draw count (0 excludes the pair) -- and the ranking loss should not
    fold in the whole-table source balance the count loss uses, only the
    bootstrap, the pair type, and the same 3x emphasis on a pair the edit
    actually reached.
    """
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.float64)
    a, b = pairs[:, 0], pairs[:, 1]
    chg = np.asarray(changed, dtype=bool)[a] | np.asarray(changed, dtype=bool)[b]
    tw = np.asarray([float(type_weights.get(int(k), 1.0)) for k in kinds],
                    dtype=np.float64)
    return (np.asarray(mult, dtype=np.float64)[a] * tw
            * np.where(chg, float(changed_weight), 1.0))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def robust_scale(x: np.ndarray, *, floor: float = 1e-6) -> float:
    """``1.4826 * MAD`` of the finite values -- the ``sigma_R`` for dR regression.

    A robust scale, not the standard deviation: the true ``dR`` is heavy-tailed
    (the HR anchor and the strong interventions sit far from the near-frozen
    bulk), and a standard deviation dominated by those tails would normalise the
    residual so the many small, informative changes vanish under the Huber knee.
    Computed on the fit boxes ONLY, so the normaliser never sees held-out data.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    mad = np.median(np.abs(x - np.median(x)))
    return float(max(1.4826 * mad, floor))


def group_indices(box: np.ndarray, tile_id: np.ndarray) -> Dict[Tuple[str, int], List[int]]:
    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, (b, t) in enumerate(zip(box, tile_id)):
        groups.setdefault((str(b), int(t)), []).append(i)
    return groups


def rank_metrics(pred: np.ndarray, true: np.ndarray, box: np.ndarray,
                 tile_id: np.ndarray) -> Dict[str, float]:
    """Within-``(box, tile)`` ranking quality.

    Grouped and never pooled: a correlation across tiles rewards knowing which
    tiles are host-rich, which is true and useless -- the actor changes what
    happens *in* a tile.
    """
    rhos, accs, n_pairs = [], [], 0
    for idx in group_indices(box, tile_id).values():
        if len(idx) < 2:
            continue
        p, t = pred[idx], true[idx]
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() < 2:
            continue
        pi, ti = p[ok], t[ok]
        rhos.append(spearman(pi, ti))
        # Tie-aware: a prediction tie on a non-tied true pair scores 0.5, not the
        # accidental "correct" a plain ``==`` gives it on a descending true pair.
        acc, npair = tie_aware_agreement(pi, ti)
        if npair:
            accs.append(acc)
            n_pairs += npair
    return {
        "within_tile_spearman": float(np.nanmean(rhos)) if rhos else float("nan"),
        "pairwise_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "n_groups": len(rhos), "n_pairs": int(n_pairs), "n_rows": int(pred.size),
    }


def pooled_count_error(pred: Mapping[str, np.ndarray],
                       true: Mapping[str, np.ndarray]) -> Dict[str, object]:
    """Whole-box ``N``, ``H``, ``S`` and ``O``, summed over tiles, in dex.

    Occupation is formed **after** pooling, ``O = sum S / sum H``, which is what
    it means. A per-tile occupancy averaged over tiles is a different number and
    a wrong one: most tiles hold a fraction of one host.
    """
    out: Dict[str, object] = {}
    errs = []
    for k in COUNT_KEYS:
        p = np.asarray(pred[k], dtype=np.float64).sum(axis=0)
        t = np.asarray(true[k], dtype=np.float64).sum(axis=0)
        ok = (p > 0) & (t > 0)
        e = np.full(p.shape, np.nan)
        e[ok] = np.abs(np.log10(p[ok]) - np.log10(t[ok]))
        out[f"pooled_{k}_pred"] = p.tolist()
        out[f"pooled_{k}_true"] = t.tolist()
        out[f"{k}_log_error"] = e.tolist()
        errs.append(e)
    ph = np.asarray(pred["n_host"], dtype=np.float64).sum(axis=0)
    ps = np.asarray(pred["occ_numerator"], dtype=np.float64).sum(axis=0)
    th = np.asarray(true["n_host"], dtype=np.float64).sum(axis=0)
    ts = np.asarray(true["occ_numerator"], dtype=np.float64).sum(axis=0)
    po = np.divide(ps, ph, out=np.full_like(ps, np.nan), where=ph > 0)
    to = np.divide(ts, th, out=np.full_like(ts, np.nan), where=th > 0)
    ok = np.isfinite(po) & np.isfinite(to) & (po > 0) & (to > 0)
    oe = np.full(po.shape, np.nan)
    oe[ok] = np.abs(np.log10(po[ok]) - np.log10(to[ok]))
    out["predicted_occupation"] = po.tolist()
    out["true_occupation"] = to.tolist()
    out["occupation_log_error"] = oe.tolist()
    stacked = np.concatenate(errs + [oe])
    out["mean_log_error"] = float(np.nanmean(stacked))
    out["max_log_error"] = float(np.nanmax(stacked))
    out["occupation_log_error_mean"] = float(np.nanmean(oe))
    out["occupation_log_error_max"] = float(np.nanmax(oe))
    return out


def pooled_count_error_by_candidate(
    pred: Mapping[str, np.ndarray], true: Mapping[str, np.ndarray],
    box: np.ndarray, tag: np.ndarray) -> Dict[str, object]:
    """Pooled ``(N, H, S, O)`` error computed PER CANDIDATE, then averaged by box.

    The all-row :func:`pooled_count_error` sums every row of every candidate into
    one box -- so one candidate over-predicting a bin cancels another under-
    predicting it, and the metric flatters a proxy that is wrong candidate by
    candidate. The right unit is the candidate: sum its 512 tiles on their own,
    score that whole-box catalog, and only then average. Averaging is by BOX
    (mean over a box's candidates, then mean over boxes) because a box's
    candidates share its modes and are not independent -- the same reason the
    ranking bootstrap resamples boxes.

    Returns the box-averaged point estimates the gate reads (``mean_log_error``,
    the per-bin ``occupation_log_error`` and its max) plus the ``per_candidate``
    table, so a caller can bootstrap over boxes without recomputing.
    """
    box = np.asarray(box)
    tag = np.asarray(tag)
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, (b, t) in enumerate(zip(box, tag)):
        groups.setdefault((str(b), str(t)), []).append(i)

    per_cand: List[Dict] = []
    for (b, t), idx in sorted(groups.items()):
        ix = np.asarray(idx, dtype=np.int64)
        pc = pooled_count_error({k: np.asarray(pred[k])[ix] for k in COUNT_KEYS},
                                {k: np.asarray(true[k])[ix] for k in COUNT_KEYS})
        per_cand.append({
            "box": b, "tag": t, "n_tiles": int(ix.size),
            "mean_log_error": float(pc["mean_log_error"]),
            "occupation_log_error": [float(x) for x in pc["occupation_log_error"]],
            "occupation_log_error_max": float(pc["occupation_log_error_max"]),
        })

    boxes = sorted({c["box"] for c in per_cand})
    n_occ = (len(per_cand[0]["occupation_log_error"]) if per_cand else 0)

    def box_mean(select) -> float:
        vals = [np.nanmean([select(c) for c in per_cand if c["box"] == b])
                for b in boxes]
        return float(np.nanmean(vals)) if vals else float("nan")

    occ_per_bin = []
    for j in range(n_occ):
        vals = [np.nanmean([c["occupation_log_error"][j]
                            for c in per_cand if c["box"] == b]) for b in boxes]
        occ_per_bin.append(float(np.nanmean(vals)) if vals else float("nan"))

    return {
        "unit": "per_candidate_pooled_averaged_by_box",
        "mean_log_error": box_mean(lambda c: c["mean_log_error"]),
        "occupation_log_error": occ_per_bin,
        "occupation_log_error_max": (float(np.nanmax(occ_per_bin))
                                     if occ_per_bin else float("nan")),
        "occupation_log_error_mean": box_mean(
            lambda c: float(np.nanmean(c["occupation_log_error"]))),
        "n_candidates": len(per_cand), "n_boxes": len(boxes),
        "per_candidate": per_cand,
    }


def slice_rows(arrays: Mapping[str, np.ndarray], spec: Mapping) -> np.ndarray:
    """Row indices of one predeclared reporting slice."""
    n = arrays["source"].size
    keep = np.ones(n, dtype=bool)
    src = spec.get("sources")
    if src:
        keep &= np.isin(arrays["source"], [str(s) for s in src])
    modes = spec.get("modes")
    if modes:
        keep &= np.isin(arrays["mode"], [str(m) for m in modes])
    alphas = spec.get("alphas")
    if alphas:
        keep &= np.isin(np.round(arrays["alpha"], 6),
                        [round(float(a), 6) for a in alphas])
    return np.nonzero(keep)[0]


def group_kfold_by_box(boxes: Sequence[str], n_folds: int,
                       seed: int = 0) -> List[np.ndarray]:
    """Fold assignments over BOXES, not rows.

    Two tiles of one box share its large-scale modes, so a row-level fold leaks
    the answer and reports a generalisation the model does not have.
    """
    uniq = sorted({str(b) for b in boxes})
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(uniq))
    k = max(1, min(int(n_folds), len(uniq)))
    return [np.asarray([uniq[i] for i in order[f::k]], dtype=object)
            for f in range(k)]


__all__ = [
    "ARMS", "ArmFeatures", "COUNT_KEYS", "FieldArmFeatures", "InlineArmFeatures",
    "RowContext", "SidecarArmFeatures", "as_arrays", "build_row_context",
    "changed_tile_mask", "channel_mean_transform", "delta_of_summary",
    "ensemble_delta", "group_indices",
    "group_kfold_by_box", "load_rows", "make_arm_features", "pair_weights",
    "pooled_count_error", "pooled_count_error_by_candidate", "predicted_delta",
    "rank_metrics", "robust_scale", "row_weights", "slice_rows", "stream_ensemble_delta",
    "stream_pred_counts", "true_delta_rewards", "unit_ids_of",
]
