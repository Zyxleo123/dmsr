"""Variable-cardinality proposals by bootstrapping whole training-host catalogs.

Stage 6 needs ``C_h ~ p(C_h | h)``: how many subhalos a host of this mass should
have, at what mass ratios, at what radii. A learned point process or set-flow is
the eventual answer. It is not the first answer, because the empirical bootstrap
below gets the same three things for free and gets one of them *better*:

* the count ``K_h`` and the mass function come out with the right joint
  distribution rather than a factorised approximation;
* the radial distribution is conditioned on mass ratio, because a donor host's
  satellites are copied together;
* correlations we have not thought to model survive automatically.

The method: index every training host by ``log10 Mvir``; to generate for host
``h``, draw a donor of similar mass and copy its **normalised** child list
(``log10(M_sub/M_host)`` and ``r/Rvir``), then subtract what SR2 already has.
Directions are resampled isotropically -- a donor's angular arrangement is a
property of its particular filament, not of its mass.

What this is allowed to read
----------------------------
Aggregate catalogs from **training boxes**, which stage 3 explicitly permits for
choosing plausible mass and radial ranges. Never the target box's HR catalog,
never HR member ids, never a paired residual. :meth:`HostTokenLibrary.from_catalogs`
records the boxes it was built from in the artifact so the provenance travels
with the numbers, and ``scripts/reward/*`` refuse to build one from a box in the
dev or final-eval split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..eval.rockstar import HaloCatalog
from .local_editor import SubhaloToken

__all__ = ["HostTokenLibrary", "subtract_existing"]


def _min_image(d: np.ndarray, box: float) -> np.ndarray:
    return d - float(box) * np.round(d / float(box))


@dataclass
class HostTokenLibrary:
    """Normalised satellite lists of many training hosts, indexed by host mass."""

    host_log_mvir: np.ndarray       # (H,)
    offsets: np.ndarray             # (H + 1,) into the flat child arrays
    log_mass_ratio: np.ndarray      # (S,)
    radius_rvir: np.ndarray         # (S,)
    boxes: Tuple[str, ...] = ()
    source: str = "training_catalogs"
    min_sub_particles: int = 20

    @property
    def n_hosts(self) -> int:
        return int(self.host_log_mvir.shape[0])

    def children(self, h: int) -> Tuple[np.ndarray, np.ndarray]:
        a, b = int(self.offsets[h]), int(self.offsets[h + 1])
        return self.log_mass_ratio[a:b], self.radius_rvir[a:b]

    # -- construction --------------------------------------------------------

    @staticmethod
    def from_catalogs(
        catalogs: Dict[str, HaloCatalog],
        *,
        boxsize_mpc_h: float = 100.0,
        min_sub_particles: int = 20,
        min_host_particles: int = 200,
        host_log_mvir_range: Tuple[float, float] = (12.0, 15.0),
    ) -> "HostTokenLibrary":
        """Index every qualifying host of every supplied (training) catalog.

        ``min_host_particles`` is well above the catalog's own resolution floor:
        a host with 20 particles has no meaningful satellite population to copy,
        and including it would flood the low-mass end of the index with hosts
        whose ``K_h`` is zero for resolution reasons rather than physical ones.
        """
        hl: List[float] = []
        offs: List[int] = [0]
        lmr: List[np.ndarray] = []
        rr: List[np.ndarray] = []
        for box in sorted(catalogs):
            cat = catalogs[box]
            hosts, subs = cat.hosts(), cat.subhalos()
            by_parent: Dict[int, List[int]] = {}
            for k, pid in enumerate(subs.parent_ids):
                by_parent.setdefault(int(pid), []).append(k)
            for i in range(hosts.n):
                if int(hosts.num_p[i]) < int(min_host_particles):
                    continue
                lm = float(np.log10(max(hosts.mvir[i], 1e-30)))
                if not host_log_mvir_range[0] <= lm <= host_log_mvir_range[1]:
                    continue
                kids = by_parent.get(int(hosts.ids[i]), [])
                kids = [k for k in kids if int(subs.num_p[k]) >= int(min_sub_particles)]
                rv = max(float(hosts.rvir[i]) * 1e-3, 1e-9)
                if kids:
                    ki = np.asarray(kids, dtype=np.int64)
                    ratio = np.log10(np.maximum(subs.mvir[ki], 1e-30)
                                     / max(hosts.mvir[i], 1e-30))
                    d = np.linalg.norm(
                        _min_image(subs.pos[ki] - hosts.pos[i], boxsize_mpc_h), axis=1)
                    lmr.append(ratio.astype(np.float64))
                    rr.append((d / rv).astype(np.float64))
                    offs.append(offs[-1] + len(kids))
                else:
                    offs.append(offs[-1])
                hl.append(lm)
        return HostTokenLibrary(
            host_log_mvir=np.asarray(hl, dtype=np.float64),
            offsets=np.asarray(offs, dtype=np.int64),
            log_mass_ratio=(np.concatenate(lmr) if lmr else np.zeros(0)),
            radius_rvir=(np.concatenate(rr) if rr else np.zeros(0)),
            boxes=tuple(sorted(catalogs)),
            min_sub_particles=int(min_sub_particles),
        )

    # -- generation ----------------------------------------------------------

    def sample_tokens(
        self,
        *,
        host_id: int,
        host_mvir: float,
        rng: np.random.Generator,
        existing_log_mass_ratio: Sequence[float] = (),
        donor_dex: float = 0.15,
        log_mass_ratio_range: Tuple[float, float] = (-3.0, -1.3),
        radius_range: Tuple[float, float] = (0.08, 0.90),
        max_tokens: int = 8,
    ) -> List[SubhaloToken]:
        """Desired-minus-present satellites for one host, as tokens.

        The subtraction is the step that makes this an *edit* generator rather
        than a catalog generator: SR2 already produced some of the population, so
        proposing the whole donor list would ask the editor to duplicate objects
        that exist. What is left over is exactly the deficit the occupation
        statistic is complaining about.
        """
        if self.n_hosts == 0:
            return []
        lm = float(np.log10(max(host_mvir, 1e-30)))
        near = np.nonzero(np.abs(self.host_log_mvir - lm) <= float(donor_dex))[0]
        if near.size == 0:
            near = np.asarray([int(np.argmin(np.abs(self.host_log_mvir - lm)))])
        donor = int(near[rng.integers(near.size)])
        ratio, radius = self.children(donor)

        keep = ((ratio >= log_mass_ratio_range[0]) & (ratio <= log_mass_ratio_range[1])
                & (radius >= radius_range[0]) & (radius <= radius_range[1]))
        ratio, radius = ratio[keep], radius[keep]
        want_idx = subtract_existing(ratio, existing_log_mass_ratio)
        ratio, radius = ratio[want_idx], radius[want_idx]

        # Most massive first: the objects Rockstar is most likely to resolve, so
        # a truncated budget spends itself on detectable proposals.
        order = np.argsort(-ratio)[: int(max_tokens)]
        out: List[SubhaloToken] = []
        for k in order:
            ct = float(rng.uniform(-1.0, 1.0))
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            st = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
            out.append(SubhaloToken(
                host_id=int(host_id),
                log_mass_ratio=float(ratio[k]),
                radius_rvir=float(radius[k]),
                direction=(st * np.cos(phi), st * np.sin(phi), ct),
            ))
        return out

    # -- io ------------------------------------------------------------------

    def to_npz(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, host_log_mvir=self.host_log_mvir, offsets=self.offsets,
            log_mass_ratio=self.log_mass_ratio, radius_rvir=self.radius_rvir,
            meta=np.asarray([json.dumps({
                "boxes": list(self.boxes), "source": self.source,
                "min_sub_particles": int(self.min_sub_particles)})]),
        )
        return p

    @staticmethod
    def from_npz(path: str | Path) -> "HostTokenLibrary":
        z = np.load(str(path), allow_pickle=False)
        meta = json.loads(str(z["meta"][0]))
        return HostTokenLibrary(
            host_log_mvir=z["host_log_mvir"], offsets=z["offsets"],
            log_mass_ratio=z["log_mass_ratio"], radius_rvir=z["radius_rvir"],
            boxes=tuple(meta.get("boxes", ())), source=str(meta.get("source", "")),
            min_sub_particles=int(meta.get("min_sub_particles", 20)),
        )


def subtract_existing(
    desired: Sequence[float], existing: Sequence[float], *, tol_dex: float = 0.25
) -> np.ndarray:
    """Indices of ``desired`` with no counterpart in ``existing``, matched 1:1.

    Greedy nearest in log mass ratio. One present object cancels at most one
    desired object, which is what keeps the proposal count equal to the actual
    deficit rather than to the total.
    """
    d = np.asarray(desired, dtype=np.float64).reshape(-1)
    e = list(np.asarray(existing, dtype=np.float64).reshape(-1))
    keep: List[int] = []
    for i, x in enumerate(d):
        if e:
            j = int(np.argmin([abs(x - y) for y in e]))
            if abs(x - e[j]) <= float(tol_dex):
                e.pop(j)
                continue
        keep.append(i)
    return np.asarray(keep, dtype=np.int64)
