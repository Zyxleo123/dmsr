"""Experiment 1: the targeted HR-residual oracle.

The question
------------
SR2's occupation function is flat where HR rises by two decades: ~14 subhalos in
a ``3.16e13 Msun/h`` host against HR's ~125. Before spending anything on search
or reward training, one thing is worth knowing: **can a missing subhalo be put
back by editing the displacement/velocity field locally at all?**

This module builds that oracle. Starting from the frozen SR2 field, it injects a
*localised* fraction of the true HR correction around a selected missing
subhalo,

    x_alpha = x_SR2 + alpha * w * (x_HR - x_SR2),

runs the complete periodic box through the halo finder, and asks whether the
target came back. It is not a deployable method -- it uses the answer -- but it
separates three failure modes that no amount of sampling can:

* recovery grows smoothly with ``alpha``  -> the reward landscape has an
  accessible direction, and search is worth funding;
* recovery only at ``alpha = 1``          -> the representation works but
  exploration will be hard;
* no recovery at any ``alpha``            -> localised residual editing cannot
  realise the missing structure, and the action representation is the problem.

Where the mask lives
--------------------
The residual ``x_HR - x_SR2`` is a **Lagrangian** field: index ``(i, j, k)`` is
the particle that started at lattice site ``(i, j, k)``. A missing subhalo is an
**Eulerian** object, at a position. The two are related by the displacement, and
by nothing simpler -- the median ``|Psi|`` is ~36 HR cells, so "the Lagrangian
neighbourhood of an Eulerian position" is not a cube around that position.

So the mask is built where the information actually is: from the HR member
particle ids of the target subhalo. Those ids *are* Lagrangian lattice indices
(:mod:`cosmo_sr.reward.tiles`), so the set of sites that formed the subhalo is
known exactly. :func:`lagrangian_mask` takes that site set, dilates it, and
smooths it, giving a mask that is localised in the space the residual is
indexed by.

An earlier design smoothed a sphere around the Eulerian position instead. That
is wrong in a way that would have produced a quiet null result: the Lagrangian
patch of a subhalo inside a cluster is nowhere near the cluster's Eulerian
position, so the injected correction would have landed on unrelated material.

Controls
--------
Any intervention that adds power near a host will change its catalog. The
control that makes the result mean something is
:func:`random_control_sites`: the **same number of Lagrangian sites**, drawn
from the same host's own particles, excluding the target's. If the targeted
edit recovers subhalos and the equal-count random edit does not, the recovery is
structural rather than a generic consequence of adding fluctuation power.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..eval.halo_match import classify_subhalos, match_hosts
from ..eval.rockstar import HaloCatalog

__all__ = [
    "InterventionSpec",
    "Target",
    "apply_intervention",
    "channel_slice",
    "ids_to_lattice",
    "lagrangian_mask",
    "random_control_sites",
    "recovery_report",
    "select_targets",
]

CHANNEL_MODES = {
    "disp": (0, 1, 2),
    "vel": (3, 4, 5),
    "both": (0, 1, 2, 3, 4, 5),
}


def channel_slice(mode: str) -> Tuple[int, ...]:
    if mode not in CHANNEL_MODES:
        raise ValueError(f"mode must be one of {sorted(CHANNEL_MODES)}, got {mode!r}")
    return CHANNEL_MODES[mode]


def ids_to_lattice(ids: np.ndarray, ng: int) -> np.ndarray:
    """``(N, 3)`` Lagrangian lattice coordinates from flat particle ids."""
    a = np.asarray(ids, dtype=np.int64)
    return np.stack([a // (ng * ng), (a // ng) % ng, a % ng], axis=1)


# --------------------------------------------------------------------------
# Target selection
# --------------------------------------------------------------------------


@dataclass
class Target:
    """One clearly-missing HR subhalo inside a matched host."""

    hr_sub_id: int
    hr_host_id: int
    sr_host_id: int
    host_mvir: float
    host_bin: int
    sub_mvir: float
    sub_num_p: int
    sub_pos_mpc_h: Tuple[float, float, float]
    host_pos_mpc_h: Tuple[float, float, float]
    host_rvir_kpc_h: float
    r_rvir: float
    n_sites: int = 0
    meta: Dict = dc_field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["sub_pos_mpc_h"] = list(map(float, self.sub_pos_mpc_h))
        d["host_pos_mpc_h"] = list(map(float, self.host_pos_mpc_h))
        return d

    @staticmethod
    def from_dict(d: Dict) -> "Target":
        d = dict(d)
        d["sub_pos_mpc_h"] = tuple(d["sub_pos_mpc_h"])
        d["host_pos_mpc_h"] = tuple(d["host_pos_mpc_h"])
        return Target(**d)


def _periodic_dist(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    d = np.asarray(a) - np.asarray(b)
    d -= box * np.round(d / box)
    return np.linalg.norm(d, axis=-1)


def select_targets(
    hr: HaloCatalog,
    sr: HaloCatalog,
    *,
    host_mass_edges: Sequence[float],
    host_bins: Sequence[int] = (2, 3),
    boxsize_mpc_h: float = 100.0,
    min_sub_particles: int = 50,
    max_targets: int = 32,
    min_separation_mpc_h: float = 6.0,
    one_per_host: bool = True,
    seed: int = 0,
) -> List[Target]:
    """Clearly-missing HR subhalos in matched hosts of the requested mass bins.

    "Clearly missing" means the repaired periodic matcher
    (:func:`cosmo_sr.eval.halo_match.classify_subhalos`) puts the HR subhalo in
    the ``missing`` class -- no SR counterpart at all, as opposed to shifted,
    mass-biased or velocity-incoherent. Those other classes are interesting but
    they are not what this oracle is testing.

    ``min_separation_mpc_h`` is what makes the experiment affordable. Every
    intervention needs a full-box Rockstar run, so targets are batched into one
    box; that is only sound if their edits cannot interact. Targets are selected
    greedily in descending host mass subject to a minimum pairwise separation,
    and the separation is enforced on **host** positions because the injected
    correction spans a host's Lagrangian patch, not just the subhalo's.

    Selecting the most massive hosts first is deliberate: bins 2 and 3 are where
    Gate B has to be decided, and the sparse ``1e14`` bin is excluded from
    ``host_bins`` by default for the reason the covariance audit gives.
    """
    edges = np.asarray(host_mass_edges, dtype=np.float64)
    hosts = hr.hosts()
    subs = hr.subhalos()
    hb = np.digitize(hosts.mvir, edges) - 1
    want = set(int(b) for b in host_bins)

    m = match_hosts(hr, sr, boxsize_mpc_h)
    classes = classify_subhalos(hr, sr, m, boxsize_mpc_h)

    host_row = {int(i): k for k, i in enumerate(hosts.ids)}
    sub_row = {int(i): k for k, i in enumerate(subs.ids)}

    cands: List[Target] = []
    for rec in classes:
        if rec["class"] != "missing":
            continue
        hrow = host_row.get(int(rec["host_hr_id"]))
        srow = sub_row.get(int(rec["hr_id"]))
        if hrow is None or srow is None:
            continue
        if int(hb[hrow]) not in want:
            continue
        if int(subs.num_p[srow]) < int(min_sub_particles):
            continue
        cands.append(Target(
            hr_sub_id=int(rec["hr_id"]),
            hr_host_id=int(rec["host_hr_id"]),
            sr_host_id=int(rec["host_sr_id"]),
            host_mvir=float(hosts.mvir[hrow]),
            host_bin=int(hb[hrow]),
            sub_mvir=float(subs.mvir[srow]),
            sub_num_p=int(subs.num_p[srow]),
            sub_pos_mpc_h=tuple(map(float, subs.pos[srow])),
            host_pos_mpc_h=tuple(map(float, hosts.pos[hrow])),
            host_rvir_kpc_h=float(hosts.rvir[hrow]),
            r_rvir=float(rec["r_rvir"]),
        ))

    # Most massive subhalo first inside the most massive host: the easiest
    # targets to detect if they come back, so a null result is not a detection
    # threshold artefact.
    cands.sort(key=lambda t: (-t.host_mvir, -t.sub_mvir))
    if one_per_host:
        seen = set()
        uniq = []
        for t in cands:
            if t.hr_host_id in seen:
                continue
            seen.add(t.hr_host_id)
            uniq.append(t)
        cands = uniq

    chosen: List[Target] = []
    for t in cands:
        if len(chosen) >= int(max_targets):
            break
        if chosen:
            d = _periodic_dist(
                np.array([c.host_pos_mpc_h for c in chosen]),
                np.array(t.host_pos_mpc_h), boxsize_mpc_h,
            )
            if float(d.min()) < float(min_separation_mpc_h):
                continue
        chosen.append(t)
    _ = seed
    return chosen


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------


def lagrangian_mask(
    sites: np.ndarray,
    ng: int,
    *,
    dilate: int = 2,
    smooth: float = 1.5,
    peak_normalize: bool = True,
) -> np.ndarray:
    """A smooth periodic mask over the Lagrangian lattice from a set of sites.

    ``sites`` is ``(N, 3)`` integer lattice coordinates -- in practice the
    Lagrangian origins of the HR particles that formed the target subhalo.

    The mask is built as: indicator on the sites, dilated by ``dilate`` cells so
    the correction is not applied to a set of measure zero, then smoothed with a
    periodic Gaussian of width ``smooth`` cells so the injected field has no
    hard edge (a step in displacement is a delta in density, and the low-k
    constraint would be measuring the step rather than the physics).

    ``peak_normalize`` rescales to ``max = 1`` so that ``alpha = 1`` really does
    apply the *full* HR correction somewhere, which is the only reading of
    ``alpha`` that makes the recovery-vs-alpha curve interpretable.

    Returns ``float32`` of shape ``(ng, ng, ng)``.
    """
    s = np.asarray(sites, dtype=np.int64).reshape(-1, 3) % int(ng)
    m = np.zeros((ng, ng, ng), dtype=np.float32)
    if s.size == 0:
        return m
    m[s[:, 0], s[:, 1], s[:, 2]] = 1.0

    d = int(dilate)
    if d > 0:
        acc = np.zeros_like(m)
        for ox in range(-d, d + 1):
            for oy in range(-d, d + 1):
                for oz in range(-d, d + 1):
                    if ox * ox + oy * oy + oz * oz > d * d:
                        continue
                    acc += np.roll(m, (ox, oy, oz), axis=(0, 1, 2))
        m = np.minimum(acc, 1.0)

    if smooth and float(smooth) > 0:
        m = _periodic_gaussian(m, float(smooth))

    if peak_normalize:
        peak = float(m.max())
        if peak > 0:
            m /= peak
    return m.astype(np.float32, copy=False)


def _periodic_gaussian(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Separable periodic Gaussian smoothing, done in Fourier space.

    A direct convolution over a 512^3 volume would dominate the cost of building
    a mask; an FFT of one real volume is seconds.
    """
    n = field.shape[0]
    k = np.fft.fftfreq(n) * n
    g1 = np.exp(-0.5 * (2.0 * np.pi * k / n * sigma_cells) ** 2)
    f = np.fft.rfftn(field.astype(np.float64))
    f *= g1[:, None, None]
    f *= g1[None, :, None]
    f *= g1[None, None, : f.shape[2]]
    return np.fft.irfftn(f, s=field.shape, axes=(0, 1, 2)).astype(np.float32)


def random_control_sites(
    host_sites: np.ndarray,
    target_sites: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Equal-count control: random sites from the same host, excluding the target.

    Same host, same number of Lagrangian sites, same intervention machinery --
    only the *location within the host* is randomised. That isolates "the
    correction had to be where the subhalo is" from "adding any HR correction
    inside a massive host helps".
    """
    host = np.asarray(host_sites, dtype=np.int64).reshape(-1, 3)
    tgt = np.asarray(target_sites, dtype=np.int64).reshape(-1, 3)
    if host.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    key_h = (host[:, 0] * (1 << 20)) + (host[:, 1] * (1 << 10)) + host[:, 2]
    key_t = (tgt[:, 0] * (1 << 20)) + (tgt[:, 1] * (1 << 10)) + tgt[:, 2]
    free = host[~np.isin(key_h, key_t)]
    if free.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    n = min(int(tgt.shape[0]), int(free.shape[0]))
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(free.shape[0], size=n, replace=False)
    return free[pick]


# --------------------------------------------------------------------------
# Applying an intervention
# --------------------------------------------------------------------------


@dataclass
class InterventionSpec:
    """One cell of the Experiment-1 grid."""

    box: str
    alpha: float
    mode: str                       # disp | vel | both
    kind: str = "targeted"          # targeted | control | exact_particles
    dilate: int = 2
    smooth: float = 1.5
    seed: int = 0
    target_ids: Tuple[int, ...] = ()

    @property
    def tag(self) -> str:
        a = f"{self.alpha:g}".replace(".", "p")
        return f"{self.kind}_{self.mode}_a{a}"

    def to_dict(self) -> Dict:
        d = dict(self.__dict__)
        d["target_ids"] = list(map(int, self.target_ids))
        return d


def apply_intervention(
    base: np.ndarray,
    hr: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    mode: str,
) -> np.ndarray:
    """``x_alpha = x_SR2 + alpha * w * (x_HR - x_SR2)`` on the selected channels.

    ``alpha = 0`` returns the frozen SR2 field bit-exactly, which is the only
    baseline the recovery-vs-alpha curve may be anchored on; ``alpha = 1`` with
    a mask that reaches 1 replaces the field with HR exactly there.

    Channels outside ``mode`` are copied unchanged, so "displacement only" really
    does leave the velocities frozen -- the distinction the interpretation table
    turns on.
    """
    out = np.array(base, dtype=np.float32, copy=True)
    a = float(alpha)
    if a == 0.0:
        return out
    ch = list(channel_slice(mode))
    w = np.asarray(mask, dtype=np.float32)
    if w.shape != out.shape[1:]:
        raise ValueError(f"mask {w.shape} does not match field {out.shape[1:]}")
    src = np.asarray(hr)
    for c in ch:
        out[c] += (a * w * (src[c].astype(np.float32) - out[c])).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# Did it work?
# --------------------------------------------------------------------------


def recovery_report(
    targets: Sequence[Target],
    hr: HaloCatalog,
    sr_after: HaloCatalog,
    *,
    boxsize_mpc_h: float = 100.0,
    pos_tol_rvir: float = 0.5,
    mass_tol_dex: float = 0.5,
) -> List[Dict]:
    """Per target: is there now an SR subhalo where the HR one is?

    Recovery is judged **directly against the HR target's own position and
    mass**, not by re-running the class machinery, so the answer does not depend
    on whether the host happened to re-match after the intervention. The
    tolerance is a fraction of the *host's* virial radius, which is the scale
    over which a subhalo's position is meaningful.

    An object is counted as recovered only if it is a *subhalo* in the new
    catalog. A target that reappears as an independent host is reported
    separately (``as_host``): that is a real change in the field but it is not
    the occupation statistic Gate B is about.
    """
    subs = hr.subhalos()
    sub_row = {int(i): k for k, i in enumerate(subs.ids)}
    new_subs = sr_after.subhalos()
    new_hosts = sr_after.hosts()

    out: List[Dict] = []
    for t in targets:
        row = sub_row.get(int(t.hr_sub_id))
        if row is None:
            out.append({"hr_sub_id": int(t.hr_sub_id), "recovered": False,
                        "reason": "target not in HR catalog"})
            continue
        pos = subs.pos[row]
        mass = float(subs.mvir[row])
        tol = max(float(pos_tol_rvir) * float(t.host_rvir_kpc_h) * 1e-3, 0.05)

        def _best(cat):
            if cat.n == 0:
                return None
            d = _periodic_dist(cat.pos, pos, boxsize_mpc_h)
            dm = np.abs(np.log10(np.maximum(cat.mvir, 1e-30) / max(mass, 1e-30)))
            ok = (d <= tol) & (dm <= float(mass_tol_dex))
            if not ok.any():
                return None
            j = int(np.nonzero(ok)[0][np.argmin(d[ok])])
            return {"id": int(cat.ids[j]), "d_mpc_h": float(d[j]),
                    "dlog10m": float(dm[j]), "mvir": float(cat.mvir[j])}

        hit = _best(new_subs)
        as_host = _best(new_hosts) if hit is None else None
        out.append({
            "hr_sub_id": int(t.hr_sub_id),
            "hr_host_id": int(t.hr_host_id),
            "host_bin": int(t.host_bin),
            "host_mvir": float(t.host_mvir),
            "sub_mvir": mass,
            "tol_mpc_h": float(tol),
            "recovered": hit is not None,
            "match": hit,
            "as_host": as_host,
        })
    return out
