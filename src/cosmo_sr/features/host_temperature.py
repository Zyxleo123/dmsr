"""Host-centric local temperature: a population constraint on the velocity field.

Why this term exists (measured 2026-10-15, docs/sr2_placement_forensics.md):

* Every SR2 field -- frozen and fine-tuned -- has HR's host DENSITY profile to
  ~10%, but the wrong host-centric velocity profile: the k=32 local dispersion
  is 0.68-0.75x HR inside 0.1 R_vir (0.34x in the tuned arm's >=100k-particle
  hosts) and 1.3-1.75x HR beyond 0.75 R_vir.
* The member-gather objective cools host cores further (vel_rms 0.71; its
  virial term is cheapest to satisfy by cooling), and Rockstar fragments a cold
  core: 12-19 fragments per core structure against frozen's 3.6, tracking the
  6D host phase-space collapse at Spearman -0.97 over 33 arms.
* Rescaling the velocities to HR's profile, positions untouched (the velocity
  oracle), brings the tuned arm's SUBFIND core satellites 672 -> 292 (HR 252)
  and the SUBFIND median r/R_vir 0.69 -> 1.02 (HR 0.88) without losing hosts.
* HR's profile is universal: 0.3-2% box-to-box scatter over seven boxes, so it
  is a rule a shared local operator can learn, unlike every per-object radius
  target this line tried.

What it measures, per host: particles within ``reach`` R_vir of the field's own
host centre (re-found by shrinking spheres, detached), queries sampled per
r/R_vir bin, each query's ``k`` nearest neighbours found WITHOUT gradient, and
the rms dispersion of those neighbours' velocities about their mean -- the same
statistic ``scripts/features/host_phase_profile.py`` reports as ``sigma_loc``.
The per-bin mean of log sigma is differentiable in the VELOCITY outputs only
(positions enter through the detached selection), so the term cannot move the
density field the oracle showed is already right.

The target is HR's value on the SAME tiles' particles, measured once when the
task is built: a tile subset misses some neighbours, and measuring both sides on
the same subset makes that bias cancel rather than become part of the target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

#: r / R_vir bin edges, host_phase_profile.EDGES_X (one definition of the shells
#: the forensics measured the defect on).
TEMP_EDGES: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75,
                                 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
EPS_KMS = 1.0


@dataclass
class TempConfig:
    k: int = 32
    queries_per_bin: int = 256
    #: bins whose upper edge exceeds this are not charged (0.5 = the core-only
    #: arm: the oracle validated the core half, the outskirt half gave +18-37%)
    max_x: float = 3.0
    min_x: float = 0.0
    reach: float = 3.3
    min_queries: int = 32
    search_max_elems: int = 1 << 26
    boxsize_mpc_h: float = 100.0
    edges: Tuple[float, ...] = TEMP_EDGES

    def __post_init__(self):
        if self.k < 2 or self.queries_per_bin < 1:
            raise ValueError("k >= 2 and queries_per_bin >= 1 are required")
        if not (0.0 <= self.min_x < self.max_x):
            raise ValueError(f"need 0 <= min_x < max_x, got {self.min_x}, {self.max_x}")


def _wrap(d: torch.Tensor, box: float) -> torch.Tensor:
    return d - box * torch.round(d / box)


@torch.no_grad()
def shrinking_centre(pos: torch.Tensor, guess: torch.Tensor, rvir: float, *,
                     box: float = 100.0, factor: float = 0.8,
                     min_n: int = 200) -> torch.Tensor:
    """Shrinking-sphere centre of ``pos`` near ``guess`` (absolute, periodic)."""
    d = _wrap(pos - guess, box)
    c = torch.zeros(3, device=pos.device, dtype=pos.dtype)
    r = 0.5 * float(rvir)
    while r > 0.05 * float(rvir):
        m = ((d - c) ** 2).sum(dim=1) < r * r
        if int(m.sum()) < min_n:
            break
        c = d[m].mean(dim=0)
        r *= factor
    return torch.remainder(guess + c, box)


@torch.no_grad()
def neighbour_rows(pos: torch.Tensor, query_rows: torch.Tensor, k: int,
                   max_elems: int = 1 << 26) -> torch.Tensor:
    """``(Q, k)`` rows of each query's k nearest neighbours in ``pos``, self removed.

    ``pos`` is already centre-relative (no wrapping needed inside a few R_vir).
    The self-match is removed by row identity, not by dropping the first column
    -- coincident particles at a collapsed site would otherwise lose a real
    neighbour (knn_cdf.knn_neighbour_ids makes the same argument).
    """
    n = int(pos.shape[0])
    if n < k + 2:
        raise ValueError(f"{n} particles cannot supply {k} neighbours")
    q = query_rows.to(pos.device, dtype=torch.long)
    per = max(1, min(int(q.numel()), int(max_elems) // max(n, 1)))
    out = []
    for s in range(0, int(q.numel()), per):
        qi = q[s:s + per]
        d2 = torch.cdist(pos[qi], pos)
        idx = d2.topk(k + 1, dim=1, largest=False, sorted=True).indices
        keep = idx != qi[:, None]
        # exactly one self-match per row; if ties pushed it out, drop the last
        first = keep.cumsum(dim=1) <= k
        out.append(idx[keep & first].reshape(qi.numel(), k))
    return torch.cat(out, dim=0)


def local_log_sigma(vel: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
    """log rms dispersion of each neighbourhood's velocities. Differentiable in ``vel``."""
    vn = vel[nbr]                                   # (Q, k, 3)
    vn = vn - vn.mean(dim=1, keepdim=True)
    return 0.5 * torch.log((vn * vn).sum(dim=-1).mean(dim=1) + EPS_KMS ** 2)


def temperature_profile(pos: torch.Tensor, vel: torch.Tensor,
                        centre_guess: torch.Tensor, rvir: float, cfg: TempConfig,
                        generator: Optional[torch.Generator] = None,
                        queries_per_bin: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Per-bin mean log local dispersion about the field's own host centre.

    Returns ``mean`` (n_bins,) -- differentiable in ``vel`` -- and ``n`` (queries
    per bin), plus the centre used. Bins with no queries hold NaN.
    """
    box = float(cfg.boxsize_mpc_h)
    c = shrinking_centre(pos.detach(), centre_guess.to(pos), rvir, box=box)
    with torch.no_grad():
        d_all = _wrap(pos.detach() - c, box)
        r_all = torch.sqrt((d_all * d_all).sum(dim=1))
        pool = torch.nonzero(r_all < float(cfg.reach) * float(rvir)).squeeze(1)
    d = d_all[pool]
    x = r_all[pool] / float(rvir)
    edges = torch.as_tensor(cfg.edges, device=pos.device, dtype=x.dtype)
    nb = len(cfg.edges) - 1
    b = torch.bucketize(x, edges[1:-1], right=True)      # 0..nb-1 inside reach
    b = torch.where(x < edges[-1], b, torch.full_like(b, -1))
    qpb = int(queries_per_bin or cfg.queries_per_bin)
    rows, bins = [], []
    for i in range(nb):
        m = torch.nonzero(b == i).squeeze(1)
        if m.numel() == 0:
            continue
        if m.numel() > qpb:
            perm = torch.randperm(m.numel(), generator=generator,
                                  device="cpu")[:qpb].to(m.device)
            m = m[perm]
        rows.append(m)
        bins.append(torch.full((m.numel(),), i, device=m.device, dtype=torch.long))
    mean = torch.full((nb,), float("nan"), device=pos.device, dtype=vel.dtype)
    n = torch.zeros(nb, device=pos.device, dtype=torch.long)
    if not rows:
        return {"mean": mean, "n": n, "centre": c}
    q = torch.cat(rows)
    qb = torch.cat(bins)
    nbr = neighbour_rows(d, q, int(cfg.k), int(cfg.search_max_elems))
    ls = local_log_sigma(vel[pool], nbr)
    n = torch.bincount(qb, minlength=nb)
    s = torch.zeros(nb, device=pos.device, dtype=ls.dtype).index_add(0, qb, ls)
    mean = torch.where(n > 0, s / n.clamp_min(1), torch.full_like(s, float("nan")))
    return {"mean": mean, "n": n, "centre": c}


def temperature_loss(prof: Dict[str, torch.Tensor], target_mean: torch.Tensor,
                     target_n: torch.Tensor, cfg: TempConfig
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Mean over charged bins of (mean log sigma_cand - mean log sigma_HR)^2.

    Two-sided: the core is too cold and the outskirts too hot in every SR2 field,
    and a one-sided hinge would charge only one of them. A bin is charged when
    both sides have ``min_queries`` queries and it lies inside [min_x, max_x).
    The info dict carries the core (< 0.25 R_vir) and outskirt (0.75-3 R_vir)
    sigma ratios cand/HR, for logging at any weight.
    """
    edges = torch.as_tensor(cfg.edges)
    lo, hi = edges[:-1], edges[1:]
    t = target_mean.to(prof["mean"])
    ok = ((prof["n"] >= int(cfg.min_queries)) & (target_n.to(prof["n"]) >= int(cfg.min_queries))
          & torch.isfinite(prof["mean"]) & torch.isfinite(t))
    charged = ok & (lo.to(ok.device) >= float(cfg.min_x)) & (hi.to(ok.device) <= float(cfg.max_x) + 1e-9)
    delta = prof["mean"] - t
    if bool(charged.any()):
        loss = (delta[charged] ** 2).mean()
    else:
        loss = prof["mean"].new_zeros(())
    with torch.no_grad():
        dd = delta.detach()
        core = ok & (hi.to(ok.device) <= 0.25 + 1e-9)
        out = ok & (lo.to(ok.device) >= 0.75 - 1e-9)
        info = {"temp_core_ratio": float(torch.exp(dd[core].mean())) if bool(core.any())
                else float("nan"),
                "temp_out_ratio": float(torch.exp(dd[out].mean())) if bool(out.any())
                else float("nan"),
                "temp_n_bins": int(charged.sum())}
    return loss, info


@torch.no_grad()
def target_profile(hr_pos: torch.Tensor, hr_vel: torch.Tensor, centre: torch.Tensor,
                   rvir: float, cfg: TempConfig, queries_per_bin: int = 2048,
                   seed: int = 0) -> Dict[str, torch.Tensor]:
    """HR's per-bin mean log sigma on the task's own particles (built once)."""
    g = torch.Generator().manual_seed(int(seed))
    prof = temperature_profile(hr_pos, hr_vel, centre, rvir, cfg, generator=g,
                               queries_per_bin=queries_per_bin)
    return {"mean": prof["mean"].detach().cpu(), "n": prof["n"].cpu()}
