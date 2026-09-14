"""The LR-host conditioning channel: what the SR operator is missing.

Why this exists
---------------
MEASURED on held-out ``set9`` (``radial-profile-is-flat-in-host-mass``): HR's
median subhalo host-centric radius scales with host mass over a factor of 6.5
(0.181 Mpc/h at 200-1k particles to 1.186 at >=100k), because ``r ~ R_vir ~
M^(1/3)``. Every candidate catalog this line has produced is **flat** at ~0.1
Mpc/h. That is not a tuning failure. ``G_theta(Y, z)`` is local and
translation-equivariant: fed only a 14^3 LR crop it cannot distinguish "1.2 Mpc
from a cluster" from "0.3 Mpc from a group" -- the two look alike locally -- so
it emits the conditional mean, and the conditional mean IS a flat profile.

The fix is a non-local input carrying the host's centre and scale. This module
builds it, on the LR grid, from an LR-only Rockstar catalog:

* ``r_over_rvir`` -- the site's distance to its owning host, in that host's
  R_vir. The coordinate the radial profile is a function of.
* ``log_mhost``   -- the host's mass, offset and scaled to O(1). The scale the
  operator has no way to infer locally.
* ``valid``       -- 1 where a gated host owns the site, 0 where none does.
  Without it the network cannot tell "at 3 R_vir of a cluster" from "no host
  here at all", since both carry the same fill values in the other channels.
* ``rhat_x/y/z``  -- optional unit vector toward the host centre. A radius
  alone says how far in to move material, never which way; see
  ``centre-offset-is-an-address``. Off by default.

Not leakage
-----------
Every channel is a function of the LR box alone. The catalog is Rockstar run on
the 64^3 LR particles, which exists at inference for any LR box that the
generator can be run on at all. Nothing here reads HR.

The completeness cliff, and why the default is ``derived``
----------------------------------------------------------
LR is 64^3 (m_p 2.98e11 Msun/h), so a 20-particle Rockstar floor is ~6e12
Msun/h: the catalog falls off a cliff rather than degrading, and
``lr-host-channel-fidelity`` measured the channel usable only for hosts above
~20k HR-equivalent particles. Hence ``min_hr_num_p``, which gates in HR
particle units and converts by the mass ratio, so the gate reads the same
whatever grid it is applied on.

Two assignment modes:

* ``derived`` (default) -- every site is assigned to the gated host minimising
  ``|x - x_host| / R_vir``, valid out to ``r_max_rvir``.
* ``membership`` -- strict Rockstar membership from the ``.particles`` table,
  resolved to the top-level ancestor.

These are CLOSE, and the default is a judgement call rather than a settled
question. ``lr_membership_coverage.py`` priced membership on set8 and set9, and
the fear it was written to test did not materialise: membership is COVERED out
to 2 R_vir, with a median correct-assignment fraction of 0.79-0.81 in the
1.0-1.5 R_vir shell where HR's subhalos peak (0.87 at 0.75-1, 0.67-0.70 at
1.5-2), and it assigns 31.6-31.9% of LR sites against ``derived``'s 29% valid
fraction. It is only past 2 R_vir that it falls to PARTIAL (0.45-0.49) and then
collapses (0.15-0.25 beyond 3).

``derived`` is the default for the reasons that survive that measurement, none
of them decisive:

* it is defined everywhere, with no PARTIAL tail past 2 R_vir -- which is
  exactly the range ``r_max_rvir = 3`` is trying to cover;
* it degrades continuously in ``r/R_vir`` rather than categorically;
* it needs no ``halos_*.particles`` table. That file is what filled the per-user
  quota once already (``rockstar-label-fails-are-quota``).

``--cond-mode membership`` is a genuine alternative arm, not a straw man.

Indexed by Lagrangian site
--------------------------
The operator consumes a Lagrangian tile, so the channel is indexed by
``particle_id`` -- the LR lattice site -- not by Eulerian position.
:func:`cosmo_sr.eval.particles.field_to_particles` numbers particles in C order
over the ``(ng, ng, ng)`` lattice, so an ``(N,)`` per-site array reshapes
directly onto the LR field's own spatial axes and the tile crop is byte-for-byte
the crop ``tile_lr_crop`` already takes.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CHANNEL_NAMES",
    "DEFAULT_CHANNELS",
    "LRHostChannelConfig",
    "assign_hosts",
    "build_channel_field",
    "channel_field_path",
    "gate_hosts",
    "load_channel_field",
    "parse_channels",
    "save_channel_field",
    "tile_cond_crop",
]

#: Every channel this module knows how to build, in canonical order.
CHANNEL_NAMES: Tuple[str, ...] = (
    "r_over_rvir", "log_mhost", "valid", "rhat_x", "rhat_y", "rhat_z",
)

#: What an arm gets unless it asks for more: radius, scale, and the mask that
#: makes "far away" distinguishable from "nothing here".
DEFAULT_CHANNELS: Tuple[str, ...] = ("r_over_rvir", "log_mhost", "valid")


def parse_channels(spec) -> Tuple[str, ...]:
    """``"r_over_rvir,log_mhost,valid"`` (or a sequence) -> a checked tuple.

    Order is the caller's, not canonical: it is the channel order of the tensor
    the generator will be trained with, so it has to round-trip exactly through
    a run's ``summary.json`` into the Rockstar gate's replay. A silently
    reordered stack would score a different model with no error anywhere.
    """
    if isinstance(spec, str):
        items = [c for c in spec.replace(",", " ").split() if c]
    else:
        items = [str(c) for c in (spec or [])]
    bad = [c for c in items if c not in CHANNEL_NAMES]
    if bad:
        raise ValueError(
            f"unknown conditioning channel(s) {bad}; known: {list(CHANNEL_NAMES)}")
    dup = sorted({c for c in items if items.count(c) > 1})
    if dup:
        raise ValueError(f"conditioning channel(s) repeated: {dup}")
    return tuple(items)


@dataclass(frozen=True)
class LRHostChannelConfig:
    """Everything that decides what the channel stack contains.

    Every field lands in the cached field's sidecar and in the run config, and
    :func:`check_compatible` refuses a cached field built under different
    settings. That is the whole defence against the failure this line has hit
    before: a gate that replays the config, rebuilds the channels differently,
    and scores a different model without raising.
    """

    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    mode: str = "derived"
    #: Gate in HR-equivalent particles; converted to the LR floor by mass ratio.
    min_hr_num_p: int = 20000
    #: Sites beyond this many R_vir of every gated host are marked invalid.
    r_max_rvir: float = 3.0
    #: ``log_mhost = (log10 Mvir - ref) / scale``.
    log_mhost_ref: float = 13.0
    log_mhost_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", parse_channels(self.channels))
        if self.mode not in ("derived", "membership"):
            raise ValueError(f"unknown mode {self.mode!r}; expected derived|membership")
        if self.r_max_rvir <= 0:
            raise ValueError("r_max_rvir must be positive")
        if self.log_mhost_scale == 0:
            raise ValueError("log_mhost_scale must be non-zero")

    @property
    def n_chan(self) -> int:
        return len(self.channels)

    def to_dict(self) -> Dict:
        return {"channels": list(self.channels), "mode": str(self.mode),
                "min_hr_num_p": int(self.min_hr_num_p),
                "r_max_rvir": float(self.r_max_rvir),
                "log_mhost_ref": float(self.log_mhost_ref),
                "log_mhost_scale": float(self.log_mhost_scale)}

    @classmethod
    def from_dict(cls, d: Dict) -> "LRHostChannelConfig":
        d = dict(d or {})
        return cls(channels=parse_channels(d.get("channels", DEFAULT_CHANNELS)),
                   mode=str(d.get("mode", "derived")),
                   min_hr_num_p=int(d.get("min_hr_num_p", 20000)),
                   r_max_rvir=float(d.get("r_max_rvir", 3.0)),
                   log_mhost_ref=float(d.get("log_mhost_ref", 13.0)),
                   log_mhost_scale=float(d.get("log_mhost_scale", 1.0)))

    def check_compatible(self, other: Dict, where: str = "") -> None:
        """Raise unless a cached field's sidecar config matches this one."""
        mine, theirs = self.to_dict(), LRHostChannelConfig.from_dict(other).to_dict()
        diff = {k: (theirs.get(k), v) for k, v in mine.items() if theirs.get(k) != v}
        if diff:
            raise ValueError(
                f"cached conditioning field{' ' + where if where else ''} was "
                f"built under different settings (cached, wanted): {diff}. "
                "Rebuild it with scripts/features/build_lr_host_channel.py "
                "rather than training on a stack the config does not describe."
            )


# --------------------------------------------------------------------------- #
# Host selection
# --------------------------------------------------------------------------- #
def gate_hosts(cat, *, min_hr_num_p: int, ng_lr: int = 64,
               ng_hr: int = 512) -> np.ndarray:
    """Row indices of the TOP-LEVEL LR hosts massive enough to condition on.

    The gate is stated in HR particles because that is the unit every
    measurement on this line is in, and converted here by the mass ratio
    ``(ng_hr / ng_lr)^3`` -- 512 at the deployed grids, so a 20k HR floor is a
    ~39-particle LR floor. Subhalos are dropped, not resolved upward: a site
    inside an LR subhalo is handled by whichever top-level host wins the
    ``r / R_vir`` assignment, which is the cluster.
    """
    ratio = (float(ng_hr) / float(ng_lr)) ** 3
    if ratio <= 0:
        raise ValueError("ng_hr / ng_lr must be positive")
    floor = float(min_hr_num_p) / ratio
    parent = np.asarray(cat.parent_ids)
    num_p = np.asarray(cat.num_p, dtype=np.float64)
    rvir = np.asarray(cat.rvir, dtype=np.float64)
    keep = (parent < 0) & (num_p >= floor) & (rvir > 0)
    return np.flatnonzero(keep).astype(np.int64)


def assign_hosts(site_pos_mpc: np.ndarray, host_pos_mpc: np.ndarray,
                 host_rvir_mpc: np.ndarray, *, boxsize_mpc_h: float,
                 chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """``(host_index, r_over_rvir)`` of the host minimising ``r / R_vir``.

    Minimising the SCALED distance rather than the raw one is the point: a site
    3 Mpc from a cluster belongs to the cluster, not to the 10^12 group 2 Mpc
    away that it is nowhere near virialised with. Brute force in chunks over a
    host list of hundreds to a few thousand rows, exactly as
    ``lr_host_fidelity.nearest`` does -- no tree, no dependency, and periodic
    by minimum image.

    Returns ``(-1, inf)`` rows when the host list is empty.
    """
    n = int(site_pos_mpc.shape[0])
    if host_pos_mpc.shape[0] == 0:
        return np.full(n, -1, dtype=np.int64), np.full(n, np.inf)
    box = float(boxsize_mpc_h)
    inv_r = 1.0 / np.maximum(np.asarray(host_rvir_mpc, dtype=np.float64), 1e-12)
    ref = np.asarray(host_pos_mpc, dtype=np.float64)
    out_i = np.empty(n, dtype=np.int64)
    out_u = np.empty(n, dtype=np.float64)
    for s in range(0, n, int(chunk)):
        q = np.asarray(site_pos_mpc[s:s + int(chunk)], dtype=np.float64)
        d = q[:, None, :] - ref[None, :, :]
        d -= box * np.round(d / box)
        u = np.sqrt((d ** 2).sum(axis=2)) * inv_r[None, :]
        j = np.argmin(u, axis=1)
        out_i[s:s + q.shape[0]] = j
        out_u[s:s + q.shape[0]] = u[np.arange(q.shape[0]), j]
    return out_i, out_u


# --------------------------------------------------------------------------- #
# The field
# --------------------------------------------------------------------------- #
def build_channel_field(
    site_pos_mpc: np.ndarray,
    host_pos_mpc: np.ndarray,
    host_rvir_mpc: np.ndarray,
    host_mvir: np.ndarray,
    cfg: LRHostChannelConfig,
    *,
    ng_lr: int,
    boxsize_mpc_h: float,
    site_host_index: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """``((C, ng, ng, ng) float32, stats)`` on the LR Lagrangian lattice.

    ``site_pos_mpc`` is ``(ng^3, 3)`` in ``particle_id`` order -- the C-order
    lattice ordering of :func:`cosmo_sr.eval.particles.field_to_particles`, and
    the reason the result can be reshaped straight onto the LR field's axes.

    ``site_host_index`` supplies an assignment instead of deriving one (mode
    ``membership``); ``-1`` marks an unassigned site. The radius is then still
    measured to THAT host, so both modes put the same physical quantity in the
    same channel and differ only in who owns what.

    Invalid sites get ``r_over_rvir = r_max_rvir`` (the far-field value, not a
    misleading 0, which would read as "at the centre of a host") and 0 in every
    other channel. ``valid`` is what distinguishes them.
    """
    n_sites = int(ng_lr) ** 3
    site_pos = np.asarray(site_pos_mpc, dtype=np.float64)
    if site_pos.shape != (n_sites, 3):
        raise ValueError(
            f"site_pos_mpc has shape {site_pos.shape}, expected ({n_sites}, 3) "
            f"for ng_lr={ng_lr}")
    hp = np.asarray(host_pos_mpc, dtype=np.float64).reshape(-1, 3)
    hr = np.asarray(host_rvir_mpc, dtype=np.float64).reshape(-1)
    hm = np.asarray(host_mvir, dtype=np.float64).reshape(-1)
    if not (hp.shape[0] == hr.shape[0] == hm.shape[0]):
        raise ValueError("host_pos / host_rvir / host_mvir disagree in length")

    if site_host_index is None:
        idx, u = assign_hosts(site_pos, hp, hr, boxsize_mpc_h=boxsize_mpc_h)
    else:
        idx = np.asarray(site_host_index, dtype=np.int64).reshape(-1)
        if idx.shape[0] != n_sites:
            raise ValueError(
                f"site_host_index has {idx.shape[0]} entries, expected {n_sites}")
        u = np.full(n_sites, np.inf)
        has = idx >= 0
        if has.any() and hp.shape[0]:
            d = site_pos[has] - hp[idx[has]]
            d -= float(boxsize_mpc_h) * np.round(d / float(boxsize_mpc_h))
            u[has] = np.sqrt((d ** 2).sum(axis=1)) / np.maximum(hr[idx[has]], 1e-12)

    valid = (idx >= 0) & np.isfinite(u) & (u <= float(cfg.r_max_rvir))
    u_fill = np.where(valid, np.minimum(u, cfg.r_max_rvir), cfg.r_max_rvir)

    # The direction channels, only computed if asked for: a (N, 3) minimum-image
    # delta over 262k sites is cheap but not free, and it is off by default.
    rhat = None
    if any(c.startswith("rhat_") for c in cfg.channels):
        rhat = np.zeros((n_sites, 3), dtype=np.float64)
        if hp.shape[0] and valid.any():
            d = hp[idx[valid]] - site_pos[valid]          # site -> host centre
            d -= float(boxsize_mpc_h) * np.round(d / float(boxsize_mpc_h))
            norm = np.linalg.norm(d, axis=1, keepdims=True)
            rhat[valid] = np.divide(d, np.maximum(norm, 1e-12))

    log_m = np.zeros(n_sites, dtype=np.float64)
    if hm.shape[0] and valid.any():
        m = np.maximum(hm[idx[valid]], 1.0)
        log_m[valid] = (np.log10(m) - cfg.log_mhost_ref) / cfg.log_mhost_scale

    per_channel = {
        "r_over_rvir": u_fill,
        "log_mhost": log_m,
        "valid": valid.astype(np.float64),
    }
    if rhat is not None:
        for a, name in enumerate(("rhat_x", "rhat_y", "rhat_z")):
            per_channel[name] = rhat[:, a]

    stack = np.stack([per_channel[c] for c in cfg.channels], axis=0)
    field = np.ascontiguousarray(
        stack.reshape(cfg.n_chan, ng_lr, ng_lr, ng_lr), dtype=np.float32)

    stats = {
        "n_sites": n_sites,
        "n_hosts_gated": int(hp.shape[0]),
        "valid_fraction": float(valid.mean()),
        "median_r_over_rvir_valid": (float(np.median(u[valid])) if valid.any()
                                     else float("nan")),
        "median_log10_mhost_valid": (
            float(np.median(np.log10(np.maximum(hm[idx[valid]], 1.0))))
            if (valid.any() and hm.shape[0]) else float("nan")),
        "per_channel_mean": {c: float(per_channel[c].mean()) for c in cfg.channels},
        "per_channel_std": {c: float(per_channel[c].std()) for c in cfg.channels},
    }
    return field, stats


def tile_cond_crop(field: np.ndarray, tile_id: int, geom) -> np.ndarray:
    """``(C, lr_size, lr_size, lr_size)`` conditioning crop of one tile.

    Deliberately the SAME call ``tile_lr_crop`` makes, on the same start and the
    same padding, so the conditioning stack and the LR input are the same tile
    by construction rather than by a matching pair of index expressions.
    """
    from ..data.crops import periodic_crop
    from ..train.sr2_finetune_data import lr_start_of_tile_id

    start = lr_start_of_tile_id(int(tile_id), geom)
    return periodic_crop(np.asarray(field), start, geom.chunk, pad=geom.pad)


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #
def channel_field_path(box: str, root=None) -> Path:
    """Where a box's built channel stack lives.

    Under the reward root, not the repo: it is a 64^3 float32 array per channel
    and home has the quota that ``rockstar-label-fails-are-quota`` is about.
    """
    from ..reward import paths
    base = Path(root) if root is not None else paths.subdir("lr_host_channel")
    return base / f"{box}.npz"


def save_channel_field(path, field: np.ndarray, cfg: LRHostChannelConfig,
                       meta: Optional[Dict] = None) -> Path:
    """Write the stack and everything needed to prove what it is."""
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p, field=np.asarray(field, dtype=np.float32),
        config=np.array(json.dumps(cfg.to_dict())),
        meta=np.array(json.dumps(dict(meta or {}))))
    return p


def load_channel_field(path, cfg: Optional[LRHostChannelConfig] = None
                       ) -> Tuple[np.ndarray, LRHostChannelConfig, Dict]:
    """``(field, config, meta)``, checked against ``cfg`` when one is given."""
    import json

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"no conditioning field at {p}. Build it first:\n"
            f"    bash scripts/slurm/submit_lr_host_channel.sh")
    with np.load(p, allow_pickle=False) as z:
        field = np.asarray(z["field"], dtype=np.float32)
        stored = json.loads(str(z["config"]))
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
    stored_cfg = LRHostChannelConfig.from_dict(stored)
    if cfg is not None:
        cfg.check_compatible(stored, where=str(p))
        if field.shape[0] != cfg.n_chan:
            raise ValueError(
                f"{p} holds {field.shape[0]} channels, config wants {cfg.n_chan}")
    return field, stored_cfg, meta
