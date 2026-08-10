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

import numpy as np
import torch

from _sr2_direct import direct_root, labels_complete_path, read_jsonl  # noqa: E402

from cosmo_sr.reward.arms import arm_storage, features_key, tokens_key  # noqa: E402
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
    """
    if not path.is_file():
        raise SystemExit(
            f"no proxy table at {path}; run "
            "scripts/reward/collect_catalog_proxy_data.py --stage index first")
    if require_complete and not labels_complete_path().is_file():
        raise SystemExit(
            f"{labels_complete_path()} is absent: labelling is not complete, so "
            f"{path} is a partial table. Re-run the indexing job once every "
            "candidate has been labelled, or pass --allow-incomplete if you "
            "really mean to fit on a subset (nothing gated may use the result).")
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
    key = features_key(arm)
    if arm_storage(arm) == "sidecar":
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
        "tile_id": np.asarray([int(r["tile_id"]) for r in rows], dtype=np.int64),
        "source": np.asarray([str(r["source"]) for r in rows]),
        "mode": np.asarray([str(r.get("mode", "both")) for r in rows]),
        "alpha": np.asarray([np.nan if r.get("alpha") is None else float(r["alpha"])
                             for r in rows], dtype=np.float64),
    }


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
        t = torch.as_tensor(np.asarray(idx, dtype=np.int64))

        def take(s: TorchSummary) -> TorchSummary:
            return TorchSummary(s.n_sub[t], s.n_host[t], s.occ_numerator[t],
                                s.volume_mpc3[t])

        return RowContext(take(self.box), take(self.frozen), take(self.measured),
                          self.keep[np.asarray(idx, dtype=np.int64)])


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
    return delta_of_summary(member.summary(feats, ctx.tile_volume), ctx,
                            reward_t, w_joint=w_joint, w_occ=w_occ, key=key)


def true_delta_rewards(ctx: RowContext, reward_t, *, w_joint: float,
                       w_occ: float, key: str = "dR_combined") -> np.ndarray:
    """The ranking target: the **real** ``dR`` this candidate's tile would give.

    Computed from the measured tile summary, not from a prediction, so a proxy
    that ranks these correctly has learned the thing the actor is scored on.
    """
    with torch.no_grad():
        return delta_of_summary(ctx.measured, ctx, reward_t, w_joint=w_joint,
                                w_occ=w_occ, key=key).numpy()


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
    return (per.mean(dim=0).numpy(),
            per.std(dim=0, unbiased=False).numpy() if per.shape[0] > 1
            else np.zeros(per.shape[1]))


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


def row_weights(source: np.ndarray, features: np.ndarray, *,
                balance_sources: bool = True, changed_weight: float = 1.0,
                changed_threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """Per-row training weights, normalised to mean 1.

    Two corrections, both aimed at the same failure: HR and frozen are five of
    every eight candidates and are the easy ones, so an unweighted loss is
    mostly the statement "HR has more substructure than SR2" -- true, already
    known, and not a direction the actor can move in.
    """
    src = np.asarray(source)
    w = np.ones(src.size, dtype=np.float64)
    if balance_sources and src.size:
        uniq, counts = np.unique(src, return_counts=True)
        per = {s: (src.size / len(uniq)) / c for s, c in zip(uniq, counts)}
        w *= np.asarray([per[s] for s in src], dtype=np.float64)
    changed = changed_tile_mask(features, threshold=changed_threshold)
    if changed_weight != 1.0:
        w = w * np.where(changed, float(changed_weight), 1.0)
    w = w / max(float(w.mean()), 1e-12)
    return {"weight": w, "changed": changed}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
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
    "ARMS", "COUNT_KEYS", "RowContext", "as_arrays", "build_row_context",
    "changed_tile_mask", "delta_of_summary", "ensemble_delta",
    "group_indices", "group_kfold_by_box", "load_rows", "pooled_count_error",
    "predicted_delta", "rank_metrics", "row_weights", "slice_rows",
    "true_delta_rewards", "unit_ids_of",
]
