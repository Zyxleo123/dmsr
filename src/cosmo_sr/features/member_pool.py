"""Select many hosts across many boxes for the member-gather fine-tune.

``docs/sr2_member_gather.md`` established the objective on **one** host of one
box with a free field, and section 7 items 3 and 4 name exactly what that
leaves open: a free field is not a generator, and one in-sample host supports no
generalisation claim. This module is the selection half of closing item 4 -- it
turns "host 271800 of set8" into a train pool and a disjoint held-out pool drawn
from every box that has an HR ``owner`` array.

Two things it does that a loop over :func:`overfit_host_mse.host_tiles` cannot
-----------------------------------------------------------------------------
**One owner-array load per box.** ``host_tiles`` reloads the 537 MB owner array
and rebuilds its CSR index on every call, which is right for a one-host
experiment and quadratic nonsense for forty. Here the catalog and the index are
loaded once and every host of that box is selected against them.

**Train/held-out tile disjointness, enforced.** Two clusters within ~12 Mpc/h of
each other can rank the same Lagrangian tile among their top few. If such a pair
were split across the train and held-out pools, the held-out host would sit in a
tile the run had been supervised on and its recovery number would not be a
held-out number at all. :func:`split_pool` refuses that by construction rather
than reporting it, because a contaminated held-out score is not a weaker result
-- it is a wrong one, and it looks exactly like a strong one.

What is deliberately NOT done here
----------------------------------
No box is dropped for being "easy" or "hard", and hosts are ranked by mass alone
-- the same order ``host_tiles`` uses. Selecting hosts on any property of the
frozen generator's output (how bad SR2 already is there, how many subhalos it is
missing) would make the training pool a function of the thing being measured.

``tests/features/test_member_pool.py`` pins :func:`select_hosts` against
``host_tiles``' answer for the top host of a box: the two must agree exactly on
the chosen tiles, or there are two definitions of "which tiles are this host's"
in the repository and the fine-tune is not measuring the oracle's problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "HostSelection",
    "PoolSplit",
    "select_hosts",
    "split_pool",
    "summarise_pool",
]


@dataclass
class HostSelection:
    """One cluster and the Lagrangian tiles the fine-tune may move for it."""

    box: str
    halo_id: int
    log_mvir: float
    num_p: int
    n_member_sites: int
    tiles: List[int]
    tile_member_sites: List[int]
    #: fraction of the host's Lagrangian sites inside ``tiles`` -- the oracle
    #: ran at 0.424, and the reachable reference is built from exactly this.
    site_coverage: float

    @property
    def key(self) -> str:
        return f"{self.box}:h{self.halo_id}"


@dataclass
class PoolSplit:
    train: List[HostSelection] = field(default_factory=list)
    holdout: List[HostSelection] = field(default_factory=list)
    #: hosts dropped because their tiles touched the other side of the split
    rejected: List[Tuple[str, str]] = field(default_factory=list)


def select_hosts(
    cat, owner_index, box: str, *,
    n_tiles: int = 4,
    max_hosts: int = 32,
    min_log_mvir: float = 13.5,
    ng_hr: int = 512,
    tile_hr: int = 64,
    children: Optional[Dict] = None,
) -> List[HostSelection]:
    """The most massive hosts of one box, each with its top ``n_tiles`` tiles.

    ``n_tiles`` stays at the oracle's 4 by default and should stay there. The
    measured ceiling at 4 tiles is 151/154 (98.1%): the ~5.4% of members that
    start outside the trained tiles cost **three** targets, so widening buys
    almost nothing while halving how many hosts fit in the same compute. Section
    6.1 of ``docs/sr2_member_gather.md`` is the measurement; changing this
    invalidates comparison with every calibrated number in the line and requires
    a fresh ceiling run per host before any result is readable.
    """
    from ..eval.particle_identity import child_map
    from .host_crops import flat_to_sites

    if children is None:
        children = child_map(cat)
    n_side = int(ng_hr) // int(tile_hr)
    hosts = np.flatnonzero(cat.parent_ids < 0)
    if hosts.size == 0:
        return []
    keep = hosts[np.log10(np.maximum(cat.mvir[hosts], 1.0)) >= float(min_log_mvir)]
    order = keep[np.argsort(-cat.mvir[keep])]

    out: List[HostSelection] = []
    for row in order:
        if len(out) >= int(max_hosts):
            break
        hid = int(cat.ids[row])
        members = owner_index.members_with_substructure(cat, hid, children=children)
        if members.size == 0:
            continue
        sites = flat_to_sites(members, ng_hr) // int(tile_hr)
        tid = (sites[:, 0] * n_side + sites[:, 1]) * n_side + sites[:, 2]
        counts = np.bincount(tid, minlength=n_side ** 3)
        top = np.argsort(-counts)[: int(n_tiles)].astype(int)
        top = [int(t) for t in top if counts[int(t)] > 0]
        if not top:
            continue
        held = int(sum(counts[t] for t in top))
        out.append(HostSelection(
            box=box, halo_id=hid,
            log_mvir=float(np.log10(max(cat.mvir[row], 1.0))),
            num_p=int(cat.num_p[row]),
            n_member_sites=int(members.size),
            tiles=top,
            tile_member_sites=[int(counts[t]) for t in top],
            site_coverage=float(held) / float(max(members.size, 1)),
        ))
    return out


def split_pool(
    hosts: Sequence[HostSelection], *,
    train_boxes: Iterable[str],
    holdout_boxes: Iterable[str],
    holdout_keys: Iterable[str] = (),
) -> PoolSplit:
    """Split by box and by explicit host key, then enforce tile disjointness.

    Two held-out axes, and they answer different questions:

    ``holdout_boxes``
        A different realisation whose LR field the fine-tune has never seen.
        This is the strong axis and the one a generalisation claim should rest
        on.
    ``holdout_keys``
        Named hosts (``"set3:h12345"``) carved out of a *training* box. Weaker --
        same realisation, same LR field -- but it separates "generalises to new
        environments" from "generalises to new realisations", and it is free:
        the whole-box gate scores every host in the box anyway.

    The second axis is exactly why the tile check below is not decoration. Two
    clusters within ~12 Mpc/h can rank the same Lagrangian tile among their top
    few, and a held-out host sitting in a tile the run was supervised on would
    report a training number under a held-out name.

    On a clash it is the **held-out** host that is dropped, with its reason
    recorded, and the training host that is kept. That asymmetry is deliberate:
    the defect is a held-out score computed on supervised material, so removing
    the held-out host removes it entirely, and discarding the training host as
    well would throw away usable supervision for nothing. A dropped host is never
    quietly reassigned to the training pool -- that would make the training pool
    a function of the held-out selection.
    """
    tr_boxes, ho_boxes = set(train_boxes), set(holdout_boxes)
    ho_keys = set(holdout_keys)
    overlap = tr_boxes & ho_boxes
    if overlap:
        raise ValueError(
            f"boxes {sorted(overlap)} are in both the train and held-out lists. "
            "A box cannot be both; the held-out score would not be held out.")

    split = PoolSplit()
    train = [h for h in hosts if h.box in tr_boxes and h.key not in ho_keys]
    hold = [h for h in hosts
            if h.box in ho_boxes or (h.box in tr_boxes and h.key in ho_keys)]

    # Tiles are per-box, so only same-box pairs can collide. With a pure box
    # split this loop finds nothing, which is the point: it costs nothing and it
    # is the only thing standing between a same-box split and a silent leak.
    train_tiles: Dict[str, Set[int]] = {}
    for h in train:
        train_tiles.setdefault(h.box, set()).update(h.tiles)

    for h in hold:
        clash = train_tiles.get(h.box, set()) & set(h.tiles)
        if clash:
            split.rejected.append(
                (h.key, f"tiles {sorted(clash)} are also supervised in training"))
        else:
            split.holdout.append(h)

    # No second pass over `train`: every surviving held-out host was checked
    # against the union of ALL training tiles above, so by construction nothing
    # in `split.holdout` overlaps anything in `train`. A reverse check would be
    # unreachable code that reads like a safeguard.
    split.train.extend(train)
    return split


def summarise_pool(hosts: Sequence[HostSelection]) -> Dict:
    """Counts and coverage, for the shakeout print and the run summary."""
    if not hosts:
        return {"n_hosts": 0, "n_boxes": 0, "n_tiles_total": 0}
    boxes = sorted({h.box for h in hosts})
    tiles = sum(len(h.tiles) for h in hosts)
    lm = np.array([h.log_mvir for h in hosts], dtype=np.float64)
    cov = np.array([h.site_coverage for h in hosts], dtype=np.float64)
    per_box = {b: sum(1 for h in hosts if h.box == b) for b in boxes}
    return {
        "n_hosts": len(hosts),
        "n_boxes": len(boxes),
        "boxes": boxes,
        "hosts_per_box": per_box,
        "n_tiles_total": int(tiles),
        "log_mvir_min": float(lm.min()),
        "log_mvir_max": float(lm.max()),
        "log_mvir_median": float(np.median(lm)),
        "site_coverage_median": float(np.median(cov)),
        "site_coverage_min": float(cov.min()),
    }
