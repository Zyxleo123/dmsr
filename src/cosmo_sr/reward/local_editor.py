"""A deployment-legal analytic editor that adds one subhalo to a frozen host.

What this module is for
-----------------------
The full-field residual prior failed: a six-channel diffusion model asked to
correct a 512^3 displacement field never learned to put a subhalo anywhere. The
targeted HR oracle (:mod:`cosmo_sr.reward.oracle_hr`, ``docs/catalog_reward_oracle.md``)
established the complementary fact -- that *localised* editing of the frozen SR2
field does have causal leverage on the halo catalog: displacement-only
intervention took subhalo recovery from 37.5% to 72.9%, displacement plus
velocity to 95.8%.

That oracle is not deployable, because it reads the HR answer. This module is
the deployable analogue of the same intervention:

    Psi_out = Psi_SR2 + E(Psi_SR2, C, a)

where ``C`` is a set of proposed subhalo tokens, ``a`` a low-dimensional
stochastic editing action, and ``E`` a fixed analytic operator that moves only
particles it has explicitly claimed. Nothing here loads the HR field, HR
residuals, HR subhalo positions or HR member ids; ``tests/reward/test_no_hr_leak.py``
pins that as an import-level property, not a convention.

The representation
------------------
A proposal is split into *what* and *how*:

``SubhaloToken`` (what)
    Host, mass ratio, radial location and direction inside the host, and
    optionally a desired relative velocity. This is the object a catalog
    generator eventually samples (:mod:`cosmo_sr.reward.token_bootstrap`).
``EditorAction`` (how)
    Eight bounded numbers describing the local transformation: where exactly
    the centre sits, how wide the edit reaches, how hard it contracts, how hard
    it cools, what it cools *towards*, and how soft the boundary is.

Both are searched by CEM in a single unconstrained vector through
:class:`ActionCodec`, which squashes ``R^d`` onto the boxed parameter space.
Search therefore never has to handle constraints, and the conditional flow of
stage 5 can be an ordinary Euclidean density.

The transformation
------------------
For each claimed particle ``i`` with smooth window weight ``w_i``:

    dx_i = -kappa_x w_i MI(x_i - c)                       (contraction)
    v_i' = v_ref + (1 - kappa_v w_i) (v_i - v_ref)        (cooling)

``MI`` is the periodic minimum image, so a proposal that straddles the box face
behaves exactly like one in the middle. ``w`` is 1 in the core and reaches
**exactly** zero at ``r = source_radius``, which is what makes "only claimed
particles change" true rather than approximately true.

Why contraction can work at all: Rockstar finds a subhalo as an overdense,
kinematically coherent phase-space clump inside a host. Pulling a few hundred
smooth-host particles towards a point raises the local density; cooling shrinks
their velocity dispersion so the clump is bound rather than transient. Neither
operation creates or destroys particles, so the host's mass is untouched by
construction and only its *profile* moves -- which is what the host-damage term
of :mod:`cosmo_sr.reward.local_reward` measures.

Units
-----
Fields are ``(6, ng, ng, ng)`` catnorm, channels ``0:3`` displacement and
``3:6`` velocity, exactly as everywhere else in the repo. All geometry in this
module is in **Mpc/h** (matching ``HaloCatalog.pos``) and velocities in km/s.
Conversion goes through :func:`cosmo_sr.data.preprocess_srs.disnorm` /
:func:`~cosmo_sr.data.preprocess_srs.velnorm`; the constants are never rewritten
here, and :func:`check_norm_convention` fails loudly if a config disagrees with
them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.preprocess_srs import disnorm, growth_D, velnorm

__all__ = [
    "ACTION_PARAMS",
    "ActionCodec",
    "EditPlan",
    "EditorAction",
    "HostPool",
    "ParamSpec",
    "SEARCH_PARAMS",
    "SubhaloToken",
    "TOKEN_PARAMS",
    "action_from_values",
    "add_displacement_mpc",
    "add_velocity_kms",
    "apply_edits",
    "apply_plan",
    "build_host_pool",
    "check_norm_convention",
    "direction_from_angles",
    "edge_window",
    "min_image",
    "n_particles_for_token",
    "particle_positions_mpc",
    "particle_velocities_kms",
    "plan_edit",
    "plan_edits",
    "proposal_center_mpc",
    "search_codec",
    "token_from_values",
]

# Modes, and what each one is allowed to move. Budget is split by
# `mode_weights` in the YAML; see `search_codec` for why velocity turned out to
# be the dominant lever on this selection rule rather than the junior partner
# the HR oracle suggested.
EDIT_MODES = ("disp", "both", "vel")


# ---------------------------------------------------------------------------
# Bounded parameter space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """One searched parameter and the interval it is confined to.

    ``lo == hi`` is a *pinned* parameter: the codec ignores its coordinate
    entirely and always decodes the constant. That is how a mode disables a
    channel (``disp`` pins ``velocity_cooling`` to 0) without changing the
    dimensionality of the search vector, so a CEM state, a manifest and a flow
    checkpoint stay comparable across modes.
    """

    name: str
    lo: float
    hi: float
    doc: str = ""

    @property
    def pinned(self) -> bool:
        return float(self.hi) - float(self.lo) <= 0.0


# The token block: *what* object is being asked for and roughly where.
TOKEN_PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec("log_mass_ratio", -3.0, -1.3,
              "log10(M_sub / M_host); with ~1.7e4 members in a 1e13 host this "
              "spans ~17-850 particles, bracketing the 50-300 the plan asks for"),
    ParamSpec("radius_rvir", 0.08, 0.90,
              "radial location of the proposal, in host Rvir"),
    ParamSpec("dir_cos_theta", -1.0, 1.0,
              "cos(polar angle); uniform in cos so the direction prior is "
              "isotropic rather than pole-heavy"),
    ParamSpec("dir_phi", 0.0, 2.0 * np.pi, "azimuth, radians"),
)

# The action block: *how* the edit is carried out. This is the vector the
# conditional flow of stage 5 models, conditioned on (host features, token).
ACTION_PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec("center_offset_x", -0.15, 0.15, "centre nudge, host Rvir units"),
    ParamSpec("center_offset_y", -0.15, 0.15, ""),
    ParamSpec("center_offset_z", -0.15, 0.15, ""),
    ParamSpec("source_radius_rvir", 0.08, 0.30,
              "support radius of the edit window, in host Rvir. The lower bound "
              "is a counting argument, not a taste: a 1e13 Msun/h host holds "
              "~1.7e4 particles inside Rvir ~ 0.5 Mpc/h, so a window at 0.02 "
              "Rvir contains ~0.1 particles and the edit is a silent no-op that "
              "still costs a full Rockstar run. See the YAML for the table."),
    ParamSpec("contraction", 0.0, 0.95, "kappa_x"),
    ParamSpec("velocity_cooling", 0.0, 0.95, "kappa_v"),
    ParamSpec("bulk_velocity_mix", 0.0, 1.0,
              "0 = cool towards the claimed pool's own mean velocity, "
              "1 = towards the velocity the token asks for"),
    ParamSpec("edge_softness", 0.05, 1.0,
              "fraction of the support radius over which w tapers to zero"),
)

SEARCH_PARAMS: Tuple[ParamSpec, ...] = TOKEN_PARAMS + ACTION_PARAMS

_SIGMOID_CLIP = 30.0
_P_EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64),
                                        -_SIGMOID_CLIP, _SIGMOID_CLIP)))


@dataclass(frozen=True)
class ActionCodec:
    """Squashing map between an unconstrained ``R^d`` vector and bounded params.

    ``decode`` is ``lo + (hi - lo) * sigmoid(z)``, so *every* real vector is a
    valid action -- CEM samples Gaussians and the flow transports Gaussians, and
    neither ever has to reject or clip a proposal. ``encode`` is the exact
    inverse on the open interval; a value pinned at a bound comes back as a
    large finite ``z`` rather than an infinity, because a manifest with ``inf``
    in it poisons every downstream mean.
    """

    params: Tuple[ParamSpec, ...]

    @property
    def dim(self) -> int:
        return len(self.params)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def decode(self, z: np.ndarray) -> Dict[str, float]:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.shape[0] != self.dim:
            raise ValueError(f"expected {self.dim} coordinates, got {z.shape[0]}")
        out: Dict[str, float] = {}
        for k, p in enumerate(self.params):
            if p.pinned:
                out[p.name] = float(p.lo)
            else:
                out[p.name] = float(p.lo + (p.hi - p.lo) * _sigmoid(z[k]))
        return out

    def decode_batch(self, z: np.ndarray) -> List[Dict[str, float]]:
        z = np.asarray(z, dtype=np.float64).reshape(-1, self.dim)
        return [self.decode(row) for row in z]

    def encode(self, values: Dict[str, float]) -> np.ndarray:
        z = np.zeros(self.dim, dtype=np.float64)
        for k, p in enumerate(self.params):
            if p.pinned:
                continue
            if p.name not in values:
                raise KeyError(f"missing parameter {p.name!r}")
            frac = (float(values[p.name]) - p.lo) / (p.hi - p.lo)
            frac = float(np.clip(frac, _P_EPS, 1.0 - _P_EPS))
            z[k] = float(np.log(frac / (1.0 - frac)))
        return z

    def clip_to_bounds(self, values: Dict[str, float]) -> Dict[str, float]:
        out = dict(values)
        for p in self.params:
            if p.name in out:
                out[p.name] = float(np.clip(out[p.name], p.lo, p.hi))
        return out

    def to_dict(self) -> Dict:
        return {"params": [{"name": p.name, "lo": p.lo, "hi": p.hi} for p in self.params]}

    @staticmethod
    def from_dict(d: Dict) -> "ActionCodec":
        return ActionCodec(tuple(
            ParamSpec(str(p["name"]), float(p["lo"]), float(p["hi"]))
            for p in d["params"]
        ))


def _apply_overrides(params: Sequence[ParamSpec],
                     bounds: Optional[Dict[str, Sequence[float]]]) -> Tuple[ParamSpec, ...]:
    b = dict(bounds or {})
    out = []
    for p in params:
        if p.name in b:
            lo, hi = (float(x) for x in b[p.name])
            if hi < lo:
                raise ValueError(f"{p.name}: hi={hi} < lo={lo}")
            out.append(ParamSpec(p.name, lo, hi, p.doc))
        else:
            out.append(p)
    return tuple(out)


def search_codec(
    mode: str = "both",
    *,
    bounds: Optional[Dict[str, Sequence[float]]] = None,
    action_only: bool = False,
    both_cooling_cap: float = 0.92,
) -> ActionCodec:
    """The codec for one edit mode, with optional YAML bound overrides.

    The three modes differ only in which channels they are allowed to touch, and
    they are implemented by *pinning* bounds rather than by branching in the
    editor -- so the "displacement only returns velocities bit-for-bit" claim is
    a property of one number being zero, which is far easier to keep true than a
    conditional scattered through the transformation code.

    ``both`` keeps cooling capped below the contraction ceiling, so the search
    cannot quietly turn into velocity-only and report it as the winning mode.

    That cap is a *measured* number, not a stylistic one, and it started out far
    too low. Measured on the selected set8 hosts: the 400 Eulerian-nearest smooth
    particles to a proposal centre already span r90 ~ 100 kpc -- about the right
    size for the ~2.3e11 Msun/h object being asked for (target ~128 kpc) -- but
    carry sigma_3D ~ 613 km/s against a target of ~89 km/s. So the set is the
    right *shape* and seven times too hot, and the lever that matters is cooling,
    not contraction: kappa_v ~ 0.82-0.91 is needed, while kappa_x ~ 0 suffices.
    An earlier cap of 0.60 put the entire physically viable region outside the
    search space.

    Note this is a real disagreement with the plan's prior that velocity is the
    junior partner. That prior comes from the HR oracle, whose mask selects a
    real subhalo's *Lagrangian* member sites -- particles that are already
    kinematically coherent, so simply moving them together works. Selecting the
    Eulerian-nearest particles instead hands the editor a set with the host's
    full velocity dispersion, and that difference is what makes cooling the
    dominant lever here.
    """
    if mode not in EDIT_MODES:
        raise ValueError(f"mode must be one of {EDIT_MODES}, got {mode!r}")
    params = ACTION_PARAMS if action_only else SEARCH_PARAMS
    params = _apply_overrides(params, bounds)
    mode_pins: Dict[str, Tuple[float, float]] = {}
    if mode == "disp":
        mode_pins = {"velocity_cooling": (0.0, 0.0), "bulk_velocity_mix": (0.0, 0.0)}
    elif mode == "vel":
        mode_pins = {"contraction": (0.0, 0.0)}
    else:  # both: contraction ceiling stays above the cooling ceiling
        cur = {p.name: p for p in params}
        cap = min(float(both_cooling_cap), float(cur["velocity_cooling"].hi))
        mode_pins = {"velocity_cooling": (float(cur["velocity_cooling"].lo), cap)}
    return ActionCodec(_apply_overrides(params, mode_pins))


# ---------------------------------------------------------------------------
# Tokens and actions
# ---------------------------------------------------------------------------


@dataclass
class SubhaloToken:
    """The object a proposal asks for, in host-relative units.

    Host-relative on purpose: a token sampled from a ``1e13`` host is then
    directly reusable on a ``3e13`` one, which is what makes the empirical
    bootstrap of stage 6 a sensible generator and what the conditional flow
    conditions on.
    """

    host_id: int
    log_mass_ratio: float                                  # log10(M_sub / M_host)
    radius_rvir: float                                     # |r| / Rvir_host
    direction: Tuple[float, float, float]                  # unit vector
    rel_velocity_vvir: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict:
        return {
            "host_id": int(self.host_id),
            "log_mass_ratio": float(self.log_mass_ratio),
            "radius_rvir": float(self.radius_rvir),
            "direction": [float(x) for x in self.direction],
            "rel_velocity_vvir": (None if self.rel_velocity_vvir is None
                                  else [float(x) for x in self.rel_velocity_vvir]),
        }

    @staticmethod
    def from_dict(d: Dict) -> "SubhaloToken":
        rv = d.get("rel_velocity_vvir")
        return SubhaloToken(
            host_id=int(d["host_id"]),
            log_mass_ratio=float(d["log_mass_ratio"]),
            radius_rvir=float(d["radius_rvir"]),
            direction=tuple(float(x) for x in d["direction"]),
            rel_velocity_vvir=None if rv is None else tuple(float(x) for x in rv),
        )


@dataclass
class EditorAction:
    """How the edit is carried out. Eight bounded numbers, nothing else."""

    center_offset: Tuple[float, float, float]   # host Rvir units
    source_radius_rvir: float
    contraction: float                          # kappa_x in [0, 1)
    velocity_cooling: float                     # kappa_v in [0, 1)
    bulk_velocity_mix: float
    edge_softness: float

    def to_dict(self) -> Dict:
        return {
            "center_offset": [float(x) for x in self.center_offset],
            "source_radius_rvir": float(self.source_radius_rvir),
            "contraction": float(self.contraction),
            "velocity_cooling": float(self.velocity_cooling),
            "bulk_velocity_mix": float(self.bulk_velocity_mix),
            "edge_softness": float(self.edge_softness),
        }

    @staticmethod
    def from_dict(d: Dict) -> "EditorAction":
        return EditorAction(
            center_offset=tuple(float(x) for x in d["center_offset"]),
            source_radius_rvir=float(d["source_radius_rvir"]),
            contraction=float(d["contraction"]),
            velocity_cooling=float(d["velocity_cooling"]),
            bulk_velocity_mix=float(d["bulk_velocity_mix"]),
            edge_softness=float(d["edge_softness"]),
        )

    @property
    def is_noop(self) -> bool:
        """No contraction and no cooling: the editor must return SR2 unchanged."""
        return float(self.contraction) == 0.0 and float(self.velocity_cooling) == 0.0


def direction_from_angles(cos_theta: float, phi: float) -> Tuple[float, float, float]:
    ct = float(np.clip(cos_theta, -1.0, 1.0))
    st = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
    return (st * float(np.cos(phi)), st * float(np.sin(phi)), ct)


def token_from_values(host_id: int, values: Dict[str, float],
                      rel_velocity_vvir: Optional[Sequence[float]] = None) -> SubhaloToken:
    return SubhaloToken(
        host_id=int(host_id),
        log_mass_ratio=float(values["log_mass_ratio"]),
        radius_rvir=float(values["radius_rvir"]),
        direction=direction_from_angles(values["dir_cos_theta"], values["dir_phi"]),
        rel_velocity_vvir=(None if rel_velocity_vvir is None
                           else tuple(float(x) for x in rel_velocity_vvir)),
    )


def action_from_values(values: Dict[str, float]) -> EditorAction:
    return EditorAction(
        center_offset=(float(values["center_offset_x"]),
                       float(values["center_offset_y"]),
                       float(values["center_offset_z"])),
        source_radius_rvir=float(values["source_radius_rvir"]),
        contraction=float(values["contraction"]),
        velocity_cooling=float(values["velocity_cooling"]),
        bulk_velocity_mix=float(values["bulk_velocity_mix"]),
        edge_softness=float(values["edge_softness"]),
    )


# ---------------------------------------------------------------------------
# Catnorm <-> physical, for a subset of particles
# ---------------------------------------------------------------------------


def check_norm_convention(dis_norm_kpc_h: float, redshift: float = 0.0) -> None:
    """Fail if the config's ``dis_norm_kpc_h`` is not the one ``disnorm`` uses.

    ``disnorm`` closes over ``6000 * D(z)``. The reward YAML repeats 6000 as a
    data-block constant, and every other consumer takes it from there. If the
    two ever diverge, every position this module computes is silently wrong by a
    constant factor -- which would look like a physics result, not a bug. So the
    check is an assertion at the top of every entry point, not a comment.
    """
    ref = float(disnorm(np.ones(1), z=float(redshift), undo=True)[0])
    want = float(dis_norm_kpc_h) * float(growth_D(float(redshift)))
    if not np.isclose(ref, want, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"config dis_norm_kpc_h={dis_norm_kpc_h} implies {want} kpc/h at "
            f"z={redshift}, but cosmo_sr.data.preprocess_srs.disnorm uses {ref}. "
            "Fix the config; do not re-derive the constant here."
        )


def _lattice_mpc(ids: np.ndarray, ng: int, boxsize_mpc_h: float) -> np.ndarray:
    """``(N, 3)`` Lagrangian lattice positions of flat particle ids, in Mpc/h.

    The id convention is the repo-wide one: ``id = (ix * ng + iy) * ng + iz``
    from ``cosmo_sr.eval.particles.field_to_particles`` building ``arange(ng**3)``
    against a ``(ng, ng, ng)`` C-order array.
    """
    ids = np.asarray(ids, dtype=np.int64)
    cell = float(boxsize_mpc_h) / int(ng)
    ix = ids // (ng * ng)
    iy = (ids // ng) % ng
    iz = ids % ng
    return (np.stack([ix, iy, iz], axis=1).astype(np.float64) + 0.5) * cell


def particle_positions_mpc(
    field: np.ndarray,
    ids: np.ndarray,
    *,
    boxsize_mpc_h: float = 100.0,
    redshift: float = 0.0,
) -> np.ndarray:
    """``(N, 3)`` Eulerian positions in Mpc/h for the named particles only.

    Deliberately *not* :func:`cosmo_sr.eval.particles.field_to_particles`: that
    materialises 512^3 positions and velocities (3.2 GB) to answer a question
    about a few hundred thousand particles. The arithmetic is identical --
    ``tests/reward/test_local_editor.py`` asserts agreement to float32 on a
    small box -- but peak memory is ``O(len(ids))``.
    """
    ids = np.asarray(ids, dtype=np.int64)
    ng = int(field.shape[1])
    if ids.size and (ids.min() < 0 or ids.max() >= ng ** 3):
        raise ValueError(f"particle id outside 0..{ng ** 3 - 1}")
    disp = np.asarray(field[0:3].reshape(3, -1)[:, ids], dtype=np.float64)
    disp_mpc = disnorm(disp, z=float(redshift), undo=True).T * 1e-3
    return (_lattice_mpc(ids, ng, boxsize_mpc_h) + disp_mpc) % float(boxsize_mpc_h)


def particle_velocities_kms(
    field: np.ndarray, ids: np.ndarray, *, redshift: float = 0.0
) -> np.ndarray:
    """``(N, 3)`` peculiar velocities in km/s for the named particles only."""
    ids = np.asarray(ids, dtype=np.int64)
    vel = np.asarray(field[3:6].reshape(3, -1)[:, ids], dtype=np.float64)
    return velnorm(vel, z=float(redshift), undo=True).T


def add_displacement_mpc(
    field: np.ndarray, ids: np.ndarray, delta_mpc: np.ndarray, *, redshift: float = 0.0
) -> None:
    """In-place ``Psi += delta`` on the displacement channels of named particles.

    The edit is expressed as an *increment* to the existing displacement, never
    as a recomputed absolute position. That is what makes the periodic behaviour
    trivially right: whatever wrapping convention the frozen field already
    encodes is preserved, and a zero increment is a genuine no-op instead of a
    round trip through ``mod``.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return
    d = np.asarray(delta_mpc, dtype=np.float64).reshape(-1, 3)
    if d.shape[0] != ids.shape[0]:
        raise ValueError(f"delta {d.shape} does not match {ids.shape[0]} ids")
    field[0:3].reshape(3, -1)[:, ids] += disnorm(
        d.T * 1e3, z=float(redshift), undo=False
    ).astype(field.dtype, copy=False)


def add_velocity_kms(
    field: np.ndarray, ids: np.ndarray, delta_kms: np.ndarray, *, redshift: float = 0.0
) -> None:
    """In-place ``v += delta`` on the velocity channels of named particles."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return
    d = np.asarray(delta_kms, dtype=np.float64).reshape(-1, 3)
    if d.shape[0] != ids.shape[0]:
        raise ValueError(f"delta {d.shape} does not match {ids.shape[0]} ids")
    field[3:6].reshape(3, -1)[:, ids] += velnorm(
        d.T, z=float(redshift), undo=False
    ).astype(field.dtype, copy=False)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def min_image(delta: np.ndarray, boxsize: float) -> np.ndarray:
    """Periodic minimum-image displacement, wrapped into ``[-L/2, L/2)``."""
    d = np.asarray(delta, dtype=np.float64)
    return d - float(boxsize) * np.round(d / float(boxsize))


def edge_window(u: np.ndarray, softness: float) -> np.ndarray:
    """Smooth radial window: 1 in the core, **exactly** 0 at ``u >= 1``.

    ``u = r / R_source``. The taper occupies the outer ``softness`` fraction of
    the support and uses the cubic smoothstep, so ``w`` and ``dw/dr`` are both
    continuous at both ends. Continuity matters for a reason the low-k
    constraint makes concrete: a step in displacement is a delta function in
    density, and the feasibility filter would then be measuring the window's
    edge rather than the physics of the proposal.

    Reaching exactly zero (rather than asymptotically) is what makes "only the
    claimed particles changed" a checkable statement.
    """
    s = float(np.clip(softness, 1e-6, 1.0))
    t = np.clip((np.asarray(u, dtype=np.float64) - (1.0 - s)) / s, 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# The particle pool of one frozen host
# ---------------------------------------------------------------------------


@dataclass
class HostPool:
    """Editable ("smooth") particles of one frozen-SR2 host, with their state.

    Built once per host per box from the *frozen* field, because it is identical
    for every candidate that box produces -- rebuilding it per candidate would
    dominate the cost of everything except Rockstar itself.

    The pool is host members minus the members of every subhalo the frozen
    catalog already found, minus everything sitting inside an existing subhalo's
    exclusion sphere. Both subtractions matter and they are not the same one:
    the first removes particles Rockstar has already bound elsewhere, the second
    removes smooth particles that merely *sit near* an existing subhalo and
    would let an edit take credit for an object that was there all along.
    """

    host_id: int
    center_mpc: np.ndarray            # (3,)
    rvir_mpc: float
    mvir: float
    vmax: float
    n_members: int                    # host members incl. substructure
    ids: np.ndarray                   # (P,) int64 editable particle ids
    pos_mpc: np.ndarray               # (P, 3)
    vel_kms: np.ndarray               # (P, 3)
    host_mean_vel_kms: np.ndarray     # (3,) bulk velocity of all host members
    boxsize_mpc_h: float = 100.0
    n_excluded_sub_members: int = 0
    n_excluded_near_sub: int = 0

    @property
    def n_pool(self) -> int:
        return int(self.ids.shape[0])

    @property
    def vvir_kms(self) -> float:
        """``sqrt(G M / Rvir)`` in km/s -- the natural velocity unit of a token."""
        # G in (Mpc/h) (km/s)^2 / (Msun/h)
        g = 4.30091727e-9
        return float(np.sqrt(g * max(self.mvir, 1e-30) / max(self.rvir_mpc, 1e-9)))

    def summary(self) -> Dict:
        return {
            "host_id": int(self.host_id),
            "center_mpc": [float(x) for x in self.center_mpc],
            "rvir_mpc": float(self.rvir_mpc),
            "mvir": float(self.mvir),
            "vmax": float(self.vmax),
            "vvir_kms": self.vvir_kms,
            "n_members": int(self.n_members),
            "n_pool": self.n_pool,
            "smooth_fraction": self.n_pool / max(int(self.n_members), 1),
            "n_excluded_sub_members": int(self.n_excluded_sub_members),
            "n_excluded_near_sub": int(self.n_excluded_near_sub),
        }


def build_host_pool(
    field: np.ndarray,
    *,
    host_id: int,
    host_member_ids: np.ndarray,
    subhalo_member_ids: Sequence[np.ndarray] = (),
    subhalo_centers_mpc: Optional[np.ndarray] = None,
    subhalo_radii_mpc: Optional[np.ndarray] = None,
    center_mpc: Sequence[float],
    rvir_mpc: float,
    mvir: float,
    vmax: float = 0.0,
    boxsize_mpc_h: float = 100.0,
    redshift: float = 0.0,
    exclusion_mult: float = 1.0,
) -> HostPool:
    """Assemble the editable pool of one host from frozen catalog information.

    ``subhalo_centers_mpc`` / ``subhalo_radii_mpc`` come from the frozen SR2
    catalog; a smooth particle within ``exclusion_mult * Rvir_sub`` of an
    existing subhalo is dropped. Nothing here reads HR.
    """
    host = np.unique(np.asarray(host_member_ids, dtype=np.int64))
    n_members = int(host.size)
    if subhalo_member_ids:
        subs = np.unique(np.concatenate(
            [np.asarray(s, dtype=np.int64).reshape(-1) for s in subhalo_member_ids]
        ))
        smooth = host[~np.isin(host, subs, assume_unique=True)]
    else:
        smooth = host
    n_sub_removed = n_members - int(smooth.size)

    pos = particle_positions_mpc(field, smooth, boxsize_mpc_h=boxsize_mpc_h,
                                 redshift=redshift)
    vel = particle_velocities_kms(field, smooth, redshift=redshift)

    n_near = 0
    if subhalo_centers_mpc is not None and len(subhalo_centers_mpc):
        c = np.asarray(subhalo_centers_mpc, dtype=np.float64).reshape(-1, 3)
        r = np.asarray(subhalo_radii_mpc, dtype=np.float64).reshape(-1)
        keep = np.ones(pos.shape[0], dtype=bool)
        for k in range(c.shape[0]):
            d = np.linalg.norm(min_image(pos - c[k], boxsize_mpc_h), axis=1)
            keep &= d > float(exclusion_mult) * float(r[k])
        n_near = int(np.count_nonzero(~keep))
        smooth, pos, vel = smooth[keep], pos[keep], vel[keep]

    # The bulk velocity is the *whole* host's, substructure included: it is the
    # frame the host moves in, and an edit that cools towards the smooth
    # component's mean alone would inherit whatever streaming the smooth
    # component happens to have.
    host_vel = particle_velocities_kms(field, host, redshift=redshift)
    return HostPool(
        host_id=int(host_id),
        center_mpc=np.asarray(center_mpc, dtype=np.float64).reshape(3),
        rvir_mpc=float(rvir_mpc), mvir=float(mvir), vmax=float(vmax),
        n_members=n_members, ids=smooth, pos_mpc=pos, vel_kms=vel,
        host_mean_vel_kms=host_vel.mean(axis=0) if host_vel.size else np.zeros(3),
        boxsize_mpc_h=float(boxsize_mpc_h),
        n_excluded_sub_members=int(n_sub_removed),
        n_excluded_near_sub=n_near,
    )


def n_particles_for_token(
    token: SubhaloToken, pool: HostPool, *, n_min: int = 40, n_max: int = 400
) -> int:
    """Particle count implied by the token's mass ratio, clamped to the search box.

    Particles are equal-mass, so the mass ratio *is* the count ratio. Clamping
    is reported rather than silent: a token asking for more particles than the
    smooth pool holds is a fact about the host, and the reward has to see the
    count that was actually used.
    """
    n = int(round((10.0 ** float(token.log_mass_ratio)) * max(pool.n_members, 1)))
    return int(np.clip(n, int(n_min), min(int(n_max), max(pool.n_pool, 1))))


# ---------------------------------------------------------------------------
# Planning and applying one edit
# ---------------------------------------------------------------------------


@dataclass
class EditPlan:
    """Exactly which particles an edit claims, and with what weight.

    Separated from application so that (a) disjointness across proposals can be
    settled before a single number is written, and (b) the audit and the tests
    can inspect a claim without touching a 3.2 GB field.
    """

    host_id: int
    center_mpc: np.ndarray            # (3,)
    source_radius_mpc: float
    ids: np.ndarray                   # (n,) int64, claimed particles
    weights: np.ndarray               # (n,) float64 in [0, 1]
    pos_mpc: np.ndarray               # (n, 3) pre-edit
    vel_kms: np.ndarray               # (n, 3) pre-edit
    n_requested: int = 0
    n_short: int = 0                  # requested minus available

    @property
    def active(self) -> np.ndarray:
        """Particles the edit actually moves (``w > 0``)."""
        return self.weights > 0.0

    def summary(self) -> Dict:
        w = self.weights
        return {
            "host_id": int(self.host_id),
            "center_mpc": [float(x) for x in self.center_mpc],
            "source_radius_mpc": float(self.source_radius_mpc),
            "n_claimed": int(self.ids.size),
            "n_active": int(np.count_nonzero(self.active)),
            "n_requested": int(self.n_requested),
            "n_short": int(self.n_short),
            "weight_mean": float(w.mean()) if w.size else 0.0,
        }


def proposal_center_mpc(pool: HostPool, token: SubhaloToken,
                        action: EditorAction) -> np.ndarray:
    """Host centre + (radius * direction + offset) * Rvir, wrapped periodically."""
    d = np.asarray(token.direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    d = d / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    off = np.asarray(action.center_offset, dtype=np.float64)
    r = (float(token.radius_rvir) * d + off) * float(pool.rvir_mpc)
    return (pool.center_mpc + r) % float(pool.boxsize_mpc_h)


def plan_edit(
    pool: HostPool,
    token: SubhaloToken,
    action: EditorAction,
    *,
    n_particles: Optional[int] = None,
    n_min: int = 40,
    n_max: int = 400,
    claimed: Optional[np.ndarray] = None,
) -> EditPlan:
    """Claim the ``n`` pool particles nearest the proposed centre.

    Selection and support are two separate decisions and are kept that way:

    * *which* particles -- the ``n`` nearest to ``c``, where ``n`` comes from the
      token's mass ratio. Deterministic, and ties broken by particle id so the
      claim does not depend on numpy's sort stability or on pool ordering.
    * *how strongly* -- the window ``w(r / R_source)``, with ``R_source`` from the
      action. A claimed particle outside ``R_source`` gets ``w = 0`` and is left
      untouched.

    Coupling the two (e.g. "take everything inside R") would make particle count
    a dependent variable of the radius, and the CEM search would then be unable
    to move mass and concentration independently -- which are precisely the two
    things that decide whether Rockstar sees a bound clump.

    ``claimed`` is the set of particle ids already taken by earlier proposals in
    this candidate box; they are removed from the pool before selection, which
    is what makes multi-proposal assignment disjoint. Because removal happens
    before any distance is computed, two proposals with disjoint claims produce
    the same plans in either order.
    """
    n_req = int(n_particles) if n_particles is not None else n_particles_for_token(
        token, pool, n_min=n_min, n_max=n_max)

    ids, pos, vel = pool.ids, pool.pos_mpc, pool.vel_kms
    if claimed is not None and len(claimed):
        free = ~np.isin(ids, np.asarray(claimed, dtype=np.int64))
        ids, pos, vel = ids[free], pos[free], vel[free]

    center = proposal_center_mpc(pool, token, action)
    if ids.size == 0:
        return EditPlan(pool.host_id, center,
                        float(action.source_radius_rvir) * pool.rvir_mpc,
                        np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3)),
                        np.zeros((0, 3)), n_requested=n_req, n_short=n_req)

    d = np.linalg.norm(min_image(pos - center, pool.boxsize_mpc_h), axis=1)
    n_take = int(min(n_req, ids.size))
    # lexsort: last key is primary. Distance first, particle id as the tie-break,
    # so the claim is a pure function of the pool contents, not of their order.
    order = np.lexsort((ids, d))[:n_take]
    sel = np.sort(order)  # keep pool order for cache-friendly gathers

    r = d[sel]
    r_src = float(action.source_radius_rvir) * float(pool.rvir_mpc)
    w = edge_window(r / max(r_src, 1e-12), action.edge_softness)
    return EditPlan(
        host_id=pool.host_id, center_mpc=center, source_radius_mpc=r_src,
        ids=ids[sel], weights=w, pos_mpc=pos[sel], vel_kms=vel[sel],
        n_requested=n_req, n_short=int(max(0, n_req - n_take)),
    )


def plan_edits(
    pools: Dict[int, HostPool],
    proposals: Sequence[Tuple[SubhaloToken, EditorAction]],
    *,
    n_min: int = 40,
    n_max: int = 400,
) -> List[EditPlan]:
    """Plan several proposals with globally disjoint particle claims.

    Claims accumulate across *all* hosts, not per host. Rockstar's host member
    sets are already disjoint so cross-host collision cannot happen in practice,
    but relying on that would make a silent aliasing bug possible the first time
    two selected hosts overlap; here it is impossible by construction.
    """
    claimed: List[np.ndarray] = []
    out: List[EditPlan] = []
    for token, action in proposals:
        pool = pools[int(token.host_id)]
        taken = np.concatenate(claimed) if claimed else None
        plan = plan_edit(pool, token, action, n_min=n_min, n_max=n_max, claimed=taken)
        out.append(plan)
        if plan.ids.size:
            claimed.append(plan.ids)
    return out


def apply_plan(
    field: np.ndarray,
    plan: EditPlan,
    action: EditorAction,
    pool: HostPool,
    *,
    boxsize_mpc_h: float = 100.0,
    redshift: float = 0.0,
) -> Dict:
    """Apply one planned edit to ``field`` in place; return what it did.

    ``field`` must be a writable ``(6, ng, ng, ng)`` catnorm array. A no-op
    action returns without writing anything at all: adding an exact zero would
    still be a write, and on a channel holding ``-0.0`` it would flip the sign
    bit -- so "zero contraction and zero cooling return SR2 bit-for-bit" is
    enforced by not touching the array rather than by trusting IEEE addition.
    """
    stats: Dict = {"n_moved": 0, "n_cooled": 0,
                   "max_dx_mpc": 0.0, "max_dv_kms": 0.0}
    act = plan.active
    if not np.any(act) or action.is_noop:
        return stats
    ids = plan.ids[act]
    w = plan.weights[act][:, None]

    kx = float(action.contraction)
    if kx > 0.0:
        dx = -kx * w * min_image(plan.pos_mpc[act] - plan.center_mpc, boxsize_mpc_h)
        add_displacement_mpc(field, ids, dx, redshift=redshift)
        stats["n_moved"] = int(ids.size)
        stats["max_dx_mpc"] = float(np.abs(dx).max())

    kv = float(action.velocity_cooling)
    if kv > 0.0:
        v = plan.vel_kms[act]
        v_pool = v.mean(axis=0)
        v_target = np.asarray(pool.host_mean_vel_kms, dtype=np.float64)
        # A token may name the bulk velocity it wants, in Vvir units, relative
        # to the host. When it does not, "what the token wants" degenerates to
        # the host frame, which is the physically neutral choice.
        rel = getattr(plan, "rel_velocity_kms", None)
        if rel is not None:
            v_target = v_target + np.asarray(rel, dtype=np.float64)
        mix = float(action.bulk_velocity_mix)
        v_ref = (1.0 - mix) * v_pool + mix * v_target
        dv = -kv * w * (v - v_ref)
        add_velocity_kms(field, ids, dv, redshift=redshift)
        stats["n_cooled"] = int(ids.size)
        stats["max_dv_kms"] = float(np.abs(dv).max())
    return stats


def apply_edits(
    base_field: np.ndarray,
    pools: Dict[int, HostPool],
    proposals: Sequence[Tuple[SubhaloToken, EditorAction]],
    *,
    boxsize_mpc_h: float = 100.0,
    redshift: float = 0.0,
    n_min: int = 40,
    n_max: int = 400,
    plans: Optional[Sequence[EditPlan]] = None,
) -> Tuple[np.ndarray, List[EditPlan], List[Dict]]:
    """``(edited_field, plans, per-proposal stats)``. The output stays catnorm.

    The candidate field is composed in memory from a copy of the frozen base and
    handed straight to the halo finder; nothing writes a 3.2 GB candidate to
    disk unless a debugging flag asks for it.
    """
    check_norm_convention(6000.0, redshift)
    out = np.array(base_field, dtype=np.float32, copy=True)
    if plans is None:
        plans = plan_edits(pools, proposals, n_min=n_min, n_max=n_max)
    stats = []
    for plan, (token, action) in zip(plans, proposals):
        rel = None
        if token.rel_velocity_vvir is not None:
            rel = np.asarray(token.rel_velocity_vvir, dtype=np.float64) * \
                pools[int(token.host_id)].vvir_kms
        object.__setattr__(plan, "rel_velocity_kms", rel)
        stats.append(apply_plan(out, plan, action, pools[int(token.host_id)],
                                boxsize_mpc_h=boxsize_mpc_h, redshift=redshift))
    return out, list(plans), stats
