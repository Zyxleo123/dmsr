"""Why does Rockstar report the inner supervised sets and not the outskirt ones?

The claim this module exists to test is that the member-gather arms build
detectable objects near the host centre and nothing at 1 R_vir, *at fixed
subhalo mass*. That is a statement about a conditional rate, so it needs the
confound controlled and the failure decomposed, not a picture.

The decomposition, and it is the whole point
--------------------------------------------
A supervised target is "recovered" when the candidate catalog holds a halo
within ``max(r_vir, 0.15)`` Mpc/h of it carrying >= 25% of its particles
(``compare_gather_catalog.target_hit_rate``). That can fail two ways, and they
need different fixes:

``address``
    the gathered members' centroid is nowhere near the HR subhalo, so the gate
    searches an empty sphere. Nothing about the object is being tested.
``object``
    the centroid IS within the search sphere and Rockstar still reports
    nothing -- the material there is too diffuse, too hot, or too light.

So every rate here is reported three ways: unconditional, conditional on the
centroid being within a tolerance of the target, and conditional on it not
being. ``P(addressed)`` and ``P(hit | addressed)`` multiply to the observed
rate, and which of the two carries the radial trend is the result.

Signing radii
-------------
Against each set's **own parent**, never against the task's host.
``host-profile-gate-passed`` records what the other choice costs: a trained tile
is 12.5 Mpc/h a side, so a task named for one host supervises subhalos of
dozens of others, and signing them all against the task host inverted that
gate's verdict. ``parent_id`` is on the row when the export wrote one and is
read from the catalog otherwise.

Everything here is a pure function of artefacts already on disk -- the HR
catalog, the arm's ``subhalos.json``, its ``compare/*.json`` and its
``pool.json``. No field, no model, no GPU, and the figures redraw from the
JSON this writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RADIAL_BINS",
    "ADDRESS_TOLERANCES",
    "SetTable",
    "build_set_table",
    "radial_profile",
    "address_decomposition",
    "hit_miss_contrast",
    "reference_inflation",
]

#: Host-centric radial bins in ``R_vir`` of the set's own parent. The edges
#: match ``docs/sr2_member_gather_training.md`` section 16e so the rows can be
#: read straight against the radial tables already on disk.
RADIAL_BINS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 1e9))

#: Address tolerances in Mpc/h. 0.15 is the gate's own ``min_radius`` -- the
#: honest threshold. The looser two exist because at 0.15 the conditional
#: sample is a handful of sets per radial bin and a rate on n=1 is not a
#: measurement; they trade a little chance coincidence for a readable n.
ADDRESS_TOLERANCES: Tuple[float, ...] = (0.15, 0.30, 0.50)

#: ``compare_gather_catalog.target_hit_rate``'s ``min_radius``, and the scale
#: ``member_gather`` normalises ``centre_offset_radii`` by
#: (``MemberSets.centre_scale = max(r_vir, 0.15)``). Both, deliberately: the
#: offset in Mpc/h is only recoverable from the row by multiplying back.
MIN_SEARCH_RADIUS_MPC_H = 0.15


def periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    """Minimum-image ``a - b`` in a periodic box of side ``box``."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return (d + 0.5 * box) % box - 0.5 * box


@dataclass
class SetTable:
    """One row per supervised set, joined across the three artefacts.

    ``ok`` is the mask of rows whose target and parent were both found in the
    HR catalog; every statistic below is reported over ``ok`` only. A row that
    drops out is counted rather than silently vanishing -- ``n_unresolved`` --
    because a join that quietly loses a third of the pool reads as a result.
    """

    halo_id: np.ndarray
    parent_id: np.ndarray
    r_over_rvir: np.ndarray        # host-centric radius, own parent
    rvir_hr_mpc_h: np.ndarray      # the target's OWN virial radius
    num_p: np.ndarray              # HR particle count
    hit: np.ndarray                # bool, from the compare JSON
    nearest_mpc_h: np.ndarray
    n_within_radius: np.ndarray
    offset_mpc_h: np.ndarray       # |centroid - target|, NaN when not exported
    bound_hard: np.ndarray
    virial: np.ndarray
    r_rms: np.ndarray
    sigma_v: np.ndarray
    live_frac: np.ndarray
    searched_at: str = ""
    n_unresolved: int = 0
    missing: List[str] = field(default_factory=list)

    @property
    def ok(self) -> np.ndarray:
        return np.isfinite(self.r_over_rvir)

    @property
    def n(self) -> int:
        return int(self.halo_id.size)


def _catalog_index(ids: np.ndarray) -> Dict[int, int]:
    return {int(h): i for i, h in enumerate(ids)}


def build_set_table(
    rows: Sequence[Dict], compare_rows: Sequence[Dict], *,
    ids: np.ndarray, pos: np.ndarray, rvir_kpc_h: np.ndarray,
    num_p: np.ndarray, parent_ids: np.ndarray, boxsize: float = 100.0,
) -> SetTable:
    """Join ``subhalos.json`` rows to the compare rows and the HR catalog.

    The two row lists are matched **by halo id**, not by position: an export and
    a comparison written days apart can disagree on ordering, and an off-by-one
    join here would read as a radial trend. Ids present in one and not the other
    are dropped and counted.

    ``rvir_kpc_h`` is the catalog column as Rockstar prints it; the conversion
    to Mpc/h happens here, once.
    """
    by_id = _catalog_index(np.asarray(ids))
    cmp_by_id = {int(r["halo_id"]): r for r in compare_rows}

    keep: List[Tuple[Dict, Dict]] = []
    for r in rows:
        c = cmp_by_id.get(int(r["halo_id"]))
        if c is not None:
            keep.append((r, c))
    n = len(keep)

    def col(getter, dtype=np.float64):
        return np.array([getter(r, c) for r, c in keep], dtype=dtype)

    hid = col(lambda r, c: int(r["halo_id"]), np.int64)
    # `parent_id` landed on the row only after the small-halo arm; fall back to
    # the catalog so older exports still resolve rather than dropping out.
    pid = np.array([int(r["parent_id"]) if "parent_id" in r
                    else int(parent_ids[by_id[int(r["halo_id"])]])
                    if int(r["halo_id"]) in by_id else -1
                    for r, _ in keep], dtype=np.int64)

    rad = np.full(n, np.nan)
    rv = np.full(n, np.nan)
    npp = np.zeros(n, dtype=np.int64)
    unresolved = 0
    for k in range(n):
        i = by_id.get(int(hid[k]))
        j = by_id.get(int(pid[k]))
        if i is None or j is None:
            unresolved += 1
            continue
        rv[k] = float(rvir_kpc_h[i]) / 1000.0
        npp[k] = int(num_p[i])
        d = periodic_delta(pos[i], pos[j], boxsize)
        rp = float(rvir_kpc_h[j]) / 1000.0
        if rp <= 0.0:
            unresolved += 1
            continue
        rad[k] = float(np.linalg.norm(d)) / rp

    # `centre_offset_radii` is in units of centre_scale = max(r_vir, 0.15), so
    # the Mpc/h offset the gate cares about is the product. Rows written before
    # that field existed report NaN rather than a guess.
    scale = np.maximum(rv, MIN_SEARCH_RADIUS_MPC_H)
    off_radii = col(lambda r, c: float(r.get("centre_offset_radii", np.nan)))
    missing = [k for k in ("centre_offset_radii", "parent_id")
               if keep and k not in keep[0][0]]

    live = col(lambda r, c: (float(r["n_live"]) / max(float(r["num_p"]), 1.0)
                             if "n_live" in r and "num_p" in r else np.nan))
    return SetTable(
        halo_id=hid, parent_id=pid, r_over_rvir=rad, rvir_hr_mpc_h=rv,
        num_p=npp,
        hit=col(lambda r, c: bool(c["hit"]), bool),
        nearest_mpc_h=col(lambda r, c: float(c["nearest_mpc_h"])),
        n_within_radius=col(lambda r, c: float(c["n_within_radius"])),
        offset_mpc_h=off_radii * scale,
        bound_hard=col(lambda r, c: float(r.get("bound_hard", np.nan))),
        virial=col(lambda r, c: float(r.get("virial", np.nan))),
        r_rms=col(lambda r, c: float(r.get("r_rms", np.nan))),
        sigma_v=col(lambda r, c: float(r.get("sigma_v", np.nan))),
        live_frac=live,
        searched_at=str(keep[0][1].get("searched_at") or "hr_catalog") if keep else "",
        n_unresolved=unresolved, missing=missing)


def _med(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _rate(mask: np.ndarray, hit: np.ndarray) -> float:
    return float(hit[mask].mean()) if int(mask.sum()) else float("nan")


def radial_profile(t: SetTable, bins=RADIAL_BINS) -> List[Dict]:
    """Hit rate and the per-set statistics, binned by host-centric radius.

    ``num_p`` is carried in every row on purpose. The inner/outer gap is only a
    radial result if the mass distribution is flat across the bins, and a table
    that does not print the mass cannot be read that way.
    """
    out: List[Dict] = []
    for lo, hi in list(bins) + [(0.0, 1e9)]:
        m = t.ok & (t.r_over_rvir >= lo) & (t.r_over_rvir < hi)
        miss = m & ~t.hit
        out.append({
            "r_lo": float(lo), "r_hi": float(min(hi, 1e9)),
            "all_radii": bool(lo == 0.0 and hi >= 1e9),
            "n": int(m.sum()),
            "hit_rate": _rate(m, t.hit),
            "median_num_p": _med(t.num_p[m]),
            "median_bound_hard": _med(t.bound_hard[m]),
            "median_virial": _med(t.virial[m]),
            "median_r_rms_mpc_h": _med(t.r_rms[m]),
            "median_r_rms_over_rvir_hr": _med(t.r_rms[m] / t.rvir_hr_mpc_h[m]),
            "median_sigma_v": _med(t.sigma_v[m]),
            "median_offset_mpc_h": _med(t.offset_mpc_h[m]),
            "median_live_frac": _med(t.live_frac[m]),
            # The miss anatomy: "nothing is there" and "something too light is
            # there" are different failures and the hit flag cannot tell them
            # apart.
            "n_miss": int(miss.sum()),
            "miss_nearest_mpc_h": _med(t.nearest_mpc_h[miss]),
            "miss_frac_something_in_radius": (
                float((t.n_within_radius[miss] > 0).mean())
                if int(miss.sum()) else float("nan")),
        })
    return out


def address_decomposition(t: SetTable, tolerances=ADDRESS_TOLERANCES,
                          bins=RADIAL_BINS) -> List[Dict]:
    """``P(addressed)`` and ``P(hit | addressed)`` per radial bin, per tolerance.

    Returns an empty list when the arm's export carried no
    ``centre_offset_radii``: the decomposition is undefined without it, and an
    all-NaN table would read as "measured, and null".
    """
    if not np.isfinite(t.offset_mpc_h).any():
        return []
    out: List[Dict] = []
    for tol in tolerances:
        addressed = t.offset_mpc_h < float(tol)
        for lo, hi in list(bins) + [(0.0, 1e9)]:
            m = t.ok & (t.r_over_rvir >= lo) & (t.r_over_rvir < hi)
            a, b = m & addressed, m & ~addressed
            out.append({
                "tol_mpc_h": float(tol),
                "r_lo": float(lo), "r_hi": float(min(hi, 1e9)),
                "all_radii": bool(lo == 0.0 and hi >= 1e9),
                "n": int(m.sum()),
                "p_addressed": (float(a.sum()) / float(m.sum())
                                if int(m.sum()) else float("nan")),
                "n_addressed": int(a.sum()),
                "hit_rate_given_addressed": _rate(a, t.hit),
                "n_unaddressed": int(b.sum()),
                "hit_rate_given_unaddressed": _rate(b, t.hit),
            })
    return out


def hit_miss_contrast(t: SetTable) -> Dict:
    """What separates a recovered target from a lost one, pooled over radius.

    Reported as medians rather than a fitted discriminator: the point is which
    axes move at all, and a two-column table says that without importing a
    model the rest of this line does not use.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, m in (("hit", t.ok & t.hit), ("miss", t.ok & ~t.hit)):
        out[name] = {
            "n": int(m.sum()),
            "median_r_over_rvir": _med(t.r_over_rvir[m]),
            "median_offset_mpc_h": _med(t.offset_mpc_h[m]),
            "median_r_rms_over_rvir_hr": _med(t.r_rms[m] / t.rvir_hr_mpc_h[m]),
            "median_bound_hard": _med(t.bound_hard[m]),
            "median_virial": _med(t.virial[m]),
            "median_sigma_v": _med(t.sigma_v[m]),
            "median_num_p": _med(t.num_p[m]),
        }
    return out


def reference_inflation(pool: Dict, split: str = "holdout_hosts") -> Dict:
    """How far the loss's *reachable* reference sits from HR's actual object.

    ``build_member_sets`` measures the reference on **HR inside the trained
    tiles, frozen outside**, because members whose Lagrangian site is outside
    those tiles cannot be moved by the run or by the splice. It records HR's own
    value alongside as ``pure_hr_median``, and this is the comparison that
    artefact exists for.

    It matters because ``r_rms`` is an **rms**: a straggler tail of a couple of
    percent sitting a Mpc/h away enters the square, so the reference can be
    twice HR's radius while 98% of the members are where HR put them. The
    correlation with ``1 - live_frac`` is reported precisely so that reading is
    testable rather than asserted -- if the inflation were not the stragglers,
    it would be near zero.
    """
    hosts = list(pool.get(split, []))
    keys = ("r_rms", "sigma_v", "virial", "bound_hard")
    out: Dict = {"split": split, "n_hosts": len(hosts), "quantities": {}}
    if not hosts:
        return out
    lf = np.array([h.get("median_live_frac", np.nan) for h in hosts], float)
    out["median_live_frac"] = _med(lf)
    out["min_live_frac"] = float(np.nanmin(lf)) if np.isfinite(lf).any() else float("nan")
    # Always the HYBRID against pure HR, never `reference_median`. Under
    # `--stats-members live` the reference IS pure HR, so comparing the two
    # would report ratio 1.00 -- "no inflation" -- for the arm that fixed it,
    # and the reader could not tell that apart from an arm with no stragglers.
    # `hybrid_reference_median` is written under both policies for exactly
    # this; pools predating it fall back to `reference_median`, which was the
    # hybrid then.
    policy = str(hosts[0].get("stats_members") or "all")
    out["stats_members"] = policy
    ref_key = ("hybrid_reference_median"
               if any("hybrid_reference_median" in h for h in hosts)
               else "reference_median")
    out["compared_against"] = ref_key
    for k in keys:
        pure = np.array([h.get("pure_hr_median", {}).get(k, np.nan)
                         for h in hosts], float)
        ref = np.array([(h.get(ref_key) or {}).get(k, np.nan)
                        for h in hosts], float)
        mp, mr = _med(pure), _med(ref)
        out["quantities"][k] = {
            "pure_hr": mp, "reference": mr,
            "ratio": (mr / mp) if (np.isfinite(mp) and mp != 0.0) else float("nan"),
        }
    pure = np.array([h.get("pure_hr_median", {}).get("r_rms", np.nan)
                     for h in hosts], float)
    ref = np.array([(h.get(ref_key) or {}).get("r_rms", np.nan)
                    for h in hosts], float)
    good = np.isfinite(pure) & np.isfinite(ref) & np.isfinite(lf) & (pure != 0)
    if int(good.sum()) > 2 and np.std(lf[good]) > 0:
        out["corr_straggler_frac_vs_r_rms_inflation"] = float(
            np.corrcoef(1.0 - lf[good], ref[good] / pure[good])[0, 1])
    else:
        out["corr_straggler_frac_vs_r_rms_inflation"] = float("nan")
    return out
