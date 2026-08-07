"""One correction transform ``u -> delta``, with the coarse allowance as a knob.

Every proposal that edits the frozen SR2 field goes through this module, so
"which parts of the field may this policy touch" is one configured object rather
than an assumption spread across samplers.

Geometry
--------
``A`` block-averages the HR grid down to the LR grid and ``A^dagger`` broadcasts
back (nearest / block-constant). They satisfy ``A A^dagger = I`` exactly, so

    P_R = A^dagger A        (the blockwise coarse component)
    P_N = I - P_R           (the within-block, zero-mean component)

are complementary orthogonal projections: ``P_R + P_N = I``, ``A P_N = 0``, and
each is idempotent. ``P_R u`` is exactly the part of ``u`` the LR field can see;
``P_N u`` is exactly the part it cannot.

Modes
-----
``none``          ``delta = u``                        -- unconstrained.
``block_null``    ``delta = P_N u``                    -- LR-invisible edits only.
``block_leaky``   ``delta = P_N u + alpha * P_R u``    -- a bounded coarse allowance.
``split``         ``delta = P_N u + A^dagger c``       -- coarse part from its own
                                                          LR-grid head ``c``.

``block_null`` is deliberately **not** the default. Forcing ``A delta = 0`` is
the exact null-space projection that :mod:`cosmo_sr.reward.base` already refuses
to apply to ``Psi_hat``: enforcing LR consistency cell by cell fights the
phase-coherent collapse that makes halos, and earlier runs here measured that it
damages density and halo statistics. Whether a hard null projection is
affordable is an empirical question, and it is the question
``scripts/reward/audit_projection_oracle.py`` exists to answer -- so the default
is ``block_leaky`` with ``alpha`` supplied by that audit, not a hard zero
asserted in advance. In ``split`` mode the coarse component is not projected
away either; it is simply made explicit, and ``A delta = c`` holds exactly.

Bounded actions
---------------
Raw network outputs are bounded *before* composition,

    u = s * tanh(h) ,

so no proposal can exceed ``s`` in magnitude whatever the network does. ``s`` is
a per-channel amplitude with **separate values for displacement and velocity**
(they are different physical quantities in different catnorm units; one shared
bound would be a bound on one of them and a formality for the other).

``tanh`` saturates, and a policy pinned against its own bound is a policy whose
gradient has vanished and whose samples are set by ``s`` rather than by the
network. :meth:`CorrectionTransform.forward` therefore *measures* the saturated
fraction per group and returns it, exactly as the diffusion sampler reports its
clip fraction; a run that does not look at it cannot tell the two apart.

Scales come from measurement, never from a literal
--------------------------------------------------
``s`` is the amplitude of a real physical residual, so it is measured from
paired data (``scripts/reward/calibrate_correction_scales.py``) and loaded as a
:class:`CorrectionScales` carrying its own provenance. A hardcoded default would
silently set the size of every edit the policy can make. :class:`CorrectionScales`
therefore starts ``calibrated=False``, and
:func:`require_calibrated_scales` is what a production stage calls to refuse an
uncalibrated one -- the same pattern as ``constraints.calibrated``.

Bit-exact fallback
------------------
``scale = 0`` (or ``mode="none"`` with a zero amplitude) short-circuits to a
zero correction, and :func:`cosmo_sr.reward.base.compose` with
``residual_scale = 0`` returns ``psi_base`` itself. That is the *only* path that
recovers frozen SR2 bit for bit; see ``tests/reward/test_zero_init_claim.py``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..operators.multiscale import block_average, block_upsample

__all__ = [
    "MODES",
    "CorrectionConfig",
    "CorrectionScales",
    "CorrectionTransform",
    "bounded_action",
    "coarse_projection",
    "leaky_transform",
    "load_correction_scales",
    "null_projection",
    "remove_group_mean",
    "require_calibrated_scales",
    "saturation_fraction",
]

MODES: Tuple[str, ...] = ("none", "block_null", "block_leaky", "split")

# |tanh(h)| above this counts as saturated. tanh(2.65) ~ 0.99, at which point the
# local gradient is ~2e-2 of its value at the origin -- the number is a reporting
# convention, so it is named rather than inlined.
SATURATED_ABS = 0.99


# --------------------------------------------------------------------------- #
# Projections
# --------------------------------------------------------------------------- #
def coarse_projection(u: torch.Tensor, factor: int) -> torch.Tensor:
    """``P_R u = A^dagger A u``: the blockwise mean, broadcast back to the HR grid."""
    return block_upsample(block_average(u, factor), factor)


def null_projection(u: torch.Tensor, factor: int) -> torch.Tensor:
    """``P_N u = u - P_R u``: the within-block, zero-mean part (``A P_N u = 0``)."""
    return u - coarse_projection(u, factor)


def leaky_transform(
    r: torch.Tensor, alpha: torch.Tensor | float, factor: int
) -> torch.Tensor:
    """``T_alpha(r) = P_N r + alpha * P_R r``.

    ``alpha`` may be a scalar or a per-channel tensor, which is how displacement
    and velocity get independent coarse allowances. ``alpha = 1`` is the identity
    and ``alpha = 0`` is the hard null projection, both exactly.
    """
    coarse = coarse_projection(r, factor)
    a = _as_channel_tensor(alpha, r)
    return (r - coarse) + a * coarse


def remove_group_mean(u: torch.Tensor, channels: Sequence[int]) -> torch.Tensor:
    """Subtract each selected channel's box-wide mean, leaving others untouched.

    A uniform displacement translates the whole box and a uniform velocity boosts
    it; both are exact symmetries of a periodic simulation, so they change no
    halo statistic while spending amplitude budget and moving ``low_k_change``.
    Optional because removing them is a *choice about the action space*, and a
    silently mean-free action space is one whose bound means something different
    from what the config says.
    """
    idx = list(channels)
    if not idx:
        return u
    out = u.clone()
    sel = out[:, idx]
    out[:, idx] = sel - sel.mean(dim=(-3, -2, -1), keepdim=True)
    return out


# --------------------------------------------------------------------------- #
# Bounded actions
# --------------------------------------------------------------------------- #
def bounded_action(h: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    """``u = s * tanh(h)``, elementwise, with ``s`` broadcast over channels."""
    return _as_channel_tensor(scale, h) * torch.tanh(h)


def saturation_fraction(
    h: torch.Tensor, channels: Optional[Sequence[int]] = None,
    threshold: float = SATURATED_ABS,
) -> float:
    """Fraction of entries with ``|tanh(h)| >= threshold``.

    Computed from ``h`` under ``no_grad``: it is a diagnostic, and it must never
    contribute a gradient path of its own.
    """
    with torch.no_grad():
        x = h if channels is None else h[:, list(channels)]
        if x.numel() == 0:
            return float("nan")
        return float((torch.tanh(x).abs() >= float(threshold)).to(torch.float32).mean())


def _as_channel_tensor(value, ref: torch.Tensor) -> torch.Tensor:
    """Scalar or ``(C,)`` -> a tensor broadcastable over ``(B, C, D, H, W)``."""
    if torch.is_tensor(value):
        v = value.to(device=ref.device, dtype=ref.dtype)
        if v.dim() == 0:
            return v
        if v.dim() == 1:
            return v.view(1, -1, *([1] * (ref.dim() - 2)))
        return v
    return torch.as_tensor(float(value), device=ref.device, dtype=ref.dtype)


# --------------------------------------------------------------------------- #
# Calibrated scales
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CorrectionScales:
    """Amplitude bounds ``s``, measured from paired data.

    ``fine_*`` bound the within-block (``P_N``) part and ``coarse_*`` the coarse
    (``P_R`` / ``c``) part; they are separate because the paired residual's coarse
    and fine components differ by roughly an order of magnitude, so one number
    would over-bound one of them.

    ``calibrated=False`` means these are placeholders. Nothing in the pipeline may
    treat a placeholder as a measurement -- see :func:`require_calibrated_scales`.
    """

    fine_disp: float = 1.0
    fine_vel: float = 1.0
    coarse_disp: float = 1.0
    coarse_vel: float = 1.0
    calibrated: bool = False
    source: str = ""
    boxes: Tuple[str, ...] = ()
    meta: Dict = dc_field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["boxes"] = list(self.boxes)
        return d

    @staticmethod
    def from_dict(d: Dict) -> "CorrectionScales":
        d = dict(d or {})
        return CorrectionScales(
            fine_disp=float(d.get("fine_disp", 1.0)),
            fine_vel=float(d.get("fine_vel", 1.0)),
            coarse_disp=float(d.get("coarse_disp", 1.0)),
            coarse_vel=float(d.get("coarse_vel", 1.0)),
            calibrated=bool(d.get("calibrated", False)),
            source=str(d.get("source", "")),
            boxes=tuple(str(b) for b in d.get("boxes", ())),
            meta=dict(d.get("meta", {})),
        )


def load_correction_scales(path: str | Path) -> CorrectionScales:
    """Read the JSON written by ``scripts/reward/calibrate_correction_scales.py``."""
    d = json.loads(Path(path).read_text())
    return CorrectionScales.from_dict(d.get("scales", d))


def require_calibrated_scales(scales: CorrectionScales) -> Optional[str]:
    """``None`` if the amplitudes were measured, else why they were not."""
    if scales.calibrated:
        return None
    return (
        "correction scales are placeholders (calibrated: false). The amplitude "
        "bound sets the size of every edit the policy can propose, so it has to "
        "be measured, not guessed. Run "
        "scripts/reward/calibrate_correction_scales.py and point "
        "correction.scales_path at its output."
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CorrectionConfig:
    """Which correction transform, with what allowance and what bounds."""

    mode: str = "block_leaky"
    scale_factor: int = 8
    channels: int = 6
    disp_channels: Tuple[int, ...] = (0, 1, 2)
    vel_channels: Tuple[int, ...] = (3, 4, 5)
    # Coarse allowance, independently for displacement and velocity. 1.0 = no
    # constraint, 0.0 = hard null projection. The projection oracle chooses these.
    alpha_disp: float = 1.0
    alpha_vel: float = 1.0
    # Overall multiplier on the calibrated amplitudes: the curriculum knob.
    # 0.0 short-circuits to an exactly zero correction.
    amplitude: float = 1.0
    remove_mean_disp: bool = False
    remove_mean_vel: bool = False
    saturated_abs: float = SATURATED_ABS
    scales: CorrectionScales = dc_field(default_factory=CorrectionScales)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown correction mode {self.mode!r}; use one of {MODES}")
        if int(self.scale_factor) < 1:
            raise ValueError(f"scale_factor must be >= 1, got {self.scale_factor}")
        overlap = set(self.disp_channels) & set(self.vel_channels)
        if overlap:
            raise ValueError(
                f"displacement and velocity channel sets overlap at {sorted(overlap)}; "
                f"they carry independent alphas and bounds, so a channel cannot be in both"
            )
        bad = [c for c in (*self.disp_channels, *self.vel_channels)
               if not 0 <= int(c) < int(self.channels)]
        if bad:
            raise ValueError(f"channel indices {bad} out of range for channels={self.channels}")
        for name in ("alpha_disp", "alpha_vel"):
            a = float(getattr(self, name))
            if not 0.0 <= a <= 1.0:
                raise ValueError(
                    f"{name}={a} outside [0, 1]; alpha is the FRACTION of the "
                    f"coarse component that is allowed through"
                )
        if float(self.amplitude) < 0.0:
            raise ValueError(f"amplitude must be >= 0, got {self.amplitude}")

    # -- per-channel vectors ------------------------------------------------ #
    def _vector(self, disp_value: float, vel_value: float, fill: float) -> torch.Tensor:
        v = torch.full((int(self.channels),), float(fill), dtype=torch.float32)
        for c in self.disp_channels:
            v[int(c)] = float(disp_value)
        for c in self.vel_channels:
            v[int(c)] = float(vel_value)
        return v

    def alpha_vector(self) -> torch.Tensor:
        """``(C,)`` coarse allowance. Channels in neither group keep ``alpha = 1``."""
        return self._vector(self.alpha_disp, self.alpha_vel, 1.0)

    def fine_scale_vector(self) -> torch.Tensor:
        """``(C,)`` amplitude bound of the within-block part, times ``amplitude``."""
        s = self._vector(self.scales.fine_disp, self.scales.fine_vel, 0.0)
        return s * float(self.amplitude)

    def coarse_scale_vector(self) -> torch.Tensor:
        """``(C,)`` amplitude bound of the LR-grid head in ``split`` mode."""
        s = self._vector(self.scales.coarse_disp, self.scales.coarse_vel, 0.0)
        return s * float(self.amplitude)

    @property
    def uses_coarse_head(self) -> bool:
        return self.mode == "split"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["disp_channels"] = list(self.disp_channels)
        d["vel_channels"] = list(self.vel_channels)
        d["scales"] = self.scales.to_dict()
        return d

    @staticmethod
    def from_dict(cfg: Dict, scales: Optional[CorrectionScales] = None) -> "CorrectionConfig":
        """Build from a config dict; ``scales`` overrides an inline ``scales`` block.

        ``scales_path`` in the dict is read when no explicit object is passed, so
        a YAML can point at the calibration output instead of copying numbers
        that would then drift from it.
        """
        c = dict(cfg or {})
        sc = scales
        if sc is None:
            if c.get("scales_path"):
                sc = load_correction_scales(c["scales_path"])
            else:
                sc = CorrectionScales.from_dict(c.get("scales", {}))
        return CorrectionConfig(
            mode=str(c.get("mode", "block_leaky")),
            scale_factor=int(c.get("scale_factor", 8)),
            channels=int(c.get("channels", 6)),
            disp_channels=tuple(int(x) for x in c.get("disp_channels", (0, 1, 2))),
            vel_channels=tuple(int(x) for x in c.get("vel_channels", (3, 4, 5))),
            alpha_disp=float(c.get("alpha_disp", 1.0)),
            alpha_vel=float(c.get("alpha_vel", 1.0)),
            amplitude=float(c.get("amplitude", 1.0)),
            remove_mean_disp=bool(c.get("remove_mean_disp", False)),
            remove_mean_vel=bool(c.get("remove_mean_vel", False)),
            saturated_abs=float(c.get("saturated_abs", SATURATED_ABS)),
            scales=sc,
        )


# --------------------------------------------------------------------------- #
# The transform
# --------------------------------------------------------------------------- #
class CorrectionTransform(nn.Module):
    """``(h, c) -> (delta, stats)``: bound, project, optionally de-mean.

    Stateless and parameter-free -- a module only so it moves with ``.to(device)``
    and can sit inside a policy. ``forward`` returns the diagnostics alongside the
    field because the saturated fraction has to be logged per call, not
    reconstructed later from a checkpoint.
    """

    def __init__(self, cfg: CorrectionConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("alpha", cfg.alpha_vector(), persistent=False)
        self.register_buffer("fine_scale", cfg.fine_scale_vector(), persistent=False)
        self.register_buffer("coarse_scale", cfg.coarse_scale_vector(), persistent=False)

    # -- pieces, exposed so a caller can reuse them without the bookkeeping -- #
    def bound(self, h: torch.Tensor) -> torch.Tensor:
        return bounded_action(h, self.fine_scale)

    def project(self, u: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the configured mode to an already-bounded ``u``."""
        f = int(self.cfg.scale_factor)
        mode = self.cfg.mode
        if mode == "none":
            return u
        if mode == "block_null":
            return null_projection(u, f)
        if mode == "block_leaky":
            return leaky_transform(u, self.alpha, f)
        # split
        delta = null_projection(u, f)
        if c is None:
            return delta
        return delta + block_upsample(c, f)

    def demean(self, delta: torch.Tensor) -> torch.Tensor:
        if self.cfg.remove_mean_disp:
            delta = remove_group_mean(delta, self.cfg.disp_channels)
        if self.cfg.remove_mean_vel:
            delta = remove_group_mean(delta, self.cfg.vel_channels)
        return delta

    def forward(
        self,
        h: torch.Tensor,
        c_raw: Optional[torch.Tensor] = None,
        *,
        stats: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """``h`` is the raw HR-grid action; ``c_raw`` the raw LR-grid one (split mode).

        ``amplitude = 0`` returns exact zeros without touching the network output,
        so the zero-amplitude arm of a curriculum is bit-exact rather than
        numerically small.
        """
        if h.dim() != 5:
            raise ValueError(f"h must be (B, C, D, H, W), got {tuple(h.shape)}")
        if h.shape[1] != int(self.cfg.channels):
            raise ValueError(
                f"h has {h.shape[1]} channels, config says {self.cfg.channels}"
            )
        n = int(h.shape[-1])
        f = int(self.cfg.scale_factor)
        if self.cfg.mode != "none" and n % f:
            raise ValueError(
                f"spatial size {n} is not a multiple of scale_factor={f}; the "
                f"block projections are only defined on whole blocks"
            )

        out: Dict[str, float] = {}
        if float(self.cfg.amplitude) == 0.0:
            delta = torch.zeros_like(h)
            if stats:
                out = self._stats(h, delta, c_raw, zero=True)
            return delta, out

        if self.cfg.uses_coarse_head and c_raw is not None:
            self._check_coarse(c_raw, h)

        u = self.bound(h)
        c = None
        if self.cfg.uses_coarse_head and c_raw is not None:
            c = bounded_action(c_raw, self.coarse_scale)
        delta = self.demean(self.project(u, c))
        if stats:
            out = self._stats(h, delta, c_raw)
        return delta, out

    # -- diagnostics -------------------------------------------------------- #
    def _check_coarse(self, c_raw: torch.Tensor, h: torch.Tensor) -> None:
        f = int(self.cfg.scale_factor)
        want = (h.shape[0], int(self.cfg.channels)) + tuple(s // f for s in h.shape[2:])
        if tuple(c_raw.shape) != want:
            raise ValueError(
                f"coarse head output {tuple(c_raw.shape)} != expected {want} "
                f"(the LR grid at 1/{f} of the HR crop)"
            )

    @torch.no_grad()
    def _stats(self, h: torch.Tensor, delta: torch.Tensor,
               c_raw: Optional[torch.Tensor], zero: bool = False) -> Dict[str, float]:
        cfg = self.cfg
        f = int(cfg.scale_factor)
        out: Dict[str, float] = {
            "mode": cfg.mode,
            "alpha_disp": float(cfg.alpha_disp),
            "alpha_vel": float(cfg.alpha_vel),
            "amplitude": float(cfg.amplitude),
            "zero_amplitude": bool(zero),
        }
        thr = float(cfg.saturated_abs)
        out["tanh_saturated_fraction"] = saturation_fraction(h, None, thr)
        out["tanh_saturated_fraction_disp"] = saturation_fraction(h, cfg.disp_channels, thr)
        out["tanh_saturated_fraction_vel"] = saturation_fraction(h, cfg.vel_channels, thr)
        if c_raw is not None and cfg.uses_coarse_head:
            out["tanh_saturated_fraction_coarse"] = saturation_fraction(c_raw, None, thr)

        d = delta.float()
        out["delta_rms"] = float(d.pow(2).mean().sqrt())
        out["delta_absmax"] = float(d.abs().max())
        for name, chans in (("disp", cfg.disp_channels), ("vel", cfg.vel_channels)):
            if not chans:
                continue
            g = d[:, list(chans)]
            out[f"delta_rms_{name}"] = float(g.pow(2).mean().sqrt())
            out[f"delta_absmax_{name}"] = float(g.abs().max())
        # How much of the edit the LR field can actually see. This is the
        # quantity `alpha` controls, so it is reported next to it.
        if d.shape[-1] % f == 0:
            coarse = coarse_projection(d, f)
            num = float(coarse.pow(2).mean().sqrt())
            out["delta_coarse_rms"] = num
            out["delta_null_rms"] = float((d - coarse).pow(2).mean().sqrt())
            out["coarse_fraction"] = num / max(out["delta_rms"], 1e-30)
        return out

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        c = self.cfg
        return (f"mode={c.mode}, alpha=({c.alpha_disp}, {c.alpha_vel}), "
                f"amplitude={c.amplitude}, scale_factor={c.scale_factor}")
