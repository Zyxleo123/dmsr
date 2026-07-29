"""Part 3 -- non-mutating Fourier-band diagnostic for the flow-matching loss.

This measures **how the ordinary spatial flow-matching MSE is distributed across
Fourier shells and channels**, so the decision to spectrally reweight (Part 5) is
made from evidence rather than from an assumption that a non-uniform target
spectrum is a training problem. It never back-propagates, never draws its own
randomness (the caller supplies the fields), and never touches ``A``/``A_plus``/
``P_A`` -- those stay in real space; the FFT is applied only to the already-real,
already-projected velocity error.

Normalisation
-------------
The transform is the **full complex** ``fftn`` with ``norm='ortho'`` (not ``rfftn``).
Two reasons:

* **Parseval is exact and trivial to reason about.** With ``norm='ortho'``,
  ``sum_q |X_q|^2 = sum_x |x_x|^2``, and the number of modes equals the number of
  voxels, so ``mean_x |dv|^2 == mean_q |DV_q|^2``. Hence *constant unit spectral
  weights reproduce the spatial MSE numerically* -- the property the Part-5 loss
  must reduce to, and a test here pins it.
* **Hermitian counting is consistent by construction.** A real field's spectrum
  has ``X_{-q} = conj(X_q)``, so every physical mode and its conjugate are both
  counted once; the shell sums are therefore over a symmetric mode set and no
  half-plane bookkeeping (as ``rfftn`` would require) can silently double- or
  half-count DC/Nyquist. The cost (2x memory vs ``rfftn``) is negligible at 64^3.

Shells
------
Integer radial shells ``k = round(|q|)`` with ``|q|`` in integer mode units
(``fftfreq(n)*n``). Shell ``0`` is exactly the DC mode. Shells are aggregated into
the same ``low`` / ``transition`` / ``high`` bands used by evaluation, via
``k_LR = N_hr / (2 s)``.

Quantities (per channel ``c``, per shell ``k``)
-----------------------------------------------
``n_modes``               number of modes in the shell (geometry only)
``target_power_per_mode`` mean ``|FFT(v_target)|^2`` over the shell
``target_shell_energy``   sum ``|FFT(v_target)|^2`` over the shell
``error_power_per_mode``  mean ``|FFT(dv)|^2`` over the shell
``error_shell_energy``    sum ``|FFT(dv)|^2`` over the shell
``loss_fraction``         shell error energy / total error energy (all c,k)
``relative_shell_error``  shell error energy / (shell target energy + eps)
``shell_cosine``          shell-summed Re<V_pred, conj V_target> correlation

Per-mode quantities reveal *amplitude* imbalance; shell totals fold in the growing
number of 3D modes at larger ``k``. Both are reported, by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SHELL_CACHE: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, int]] = {}


# --------------------------------------------------------------------------- #
# FFT + shell geometry
# --------------------------------------------------------------------------- #
def fft_ortho(x: torch.Tensor) -> torch.Tensor:
    """Full complex ``fftn`` over the last three axes with ``norm='ortho'``."""
    return torch.fft.fftn(x.float(), dim=(-3, -2, -1), norm="ortho")


def parseval_residual(x: torch.Tensor) -> float:
    """``|sum|X|^2 - sum|x|^2| / sum|x|^2`` -- 0 iff Parseval holds (a self-check)."""
    e_real = float(x.float().pow(2).sum())
    X = fft_ortho(x)
    e_spec = float((X.real ** 2 + X.imag ** 2).sum())
    return abs(e_spec - e_real) / max(e_real, 1e-30)


def shell_index(n: int, device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """``(idx[N,N,N], n_modes[S], n_shells)`` for integer radial shells ``round(|k|)``."""
    key = (n, str(device))
    cached = _SHELL_CACHE.get(key)
    if cached is not None:
        return cached
    k = torch.fft.fftfreq(n, device=device) * n
    KX, KY, KZ = torch.meshgrid(k, k, k, indexing="ij")
    kmag = torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    n_shells = int(kmag.max().round().item()) + 1
    idx = kmag.round().long().clamp(max=n_shells - 1)
    n_modes = torch.zeros(n_shells, device=device).index_add_(
        0, idx.reshape(-1), torch.ones(idx.numel(), device=device)
    )
    out = (idx, n_modes, n_shells)
    _SHELL_CACHE[key] = out
    return out


def band_of_shells(shell_k: np.ndarray, factor: int, low_frac: float, high_frac: float) -> List[str]:
    """Assign each integer shell to ``low`` / ``transition`` / ``high`` (evaluation bands).

    ``k_LR = n/(2 s)`` is the LR Nyquist in integer mode units; the grid ``n`` is
    recovered as ``2 * factor * k_LR`` is unknown here, so callers pass ``factor``
    and we compare against ``k_LR`` derived from the max shell (which is the HR
    Nyquist ``n/2``): ``k_LR = (max_shell) / factor``.
    """
    k_ny_hr = float(shell_k.max())
    k_lr = k_ny_hr / factor
    bands = []
    for k in shell_k:
        if k <= low_frac * k_lr:
            bands.append("low")
        elif k <= high_frac * k_lr:
            bands.append("transition")
        else:
            bands.append("high")
    return bands


# --------------------------------------------------------------------------- #
# Accumulator
# --------------------------------------------------------------------------- #
@dataclass
class BandDiagnosticAccumulator:
    """Sums shell energies over batches so ratios use pooled statistics.

    ``add(v_pred, v_target, operator)`` is ``no_grad`` and detaches; nothing here
    can alter training. Call :meth:`result` for the finished diagnostic.
    """

    factor: int
    low_frac: float = 0.5
    high_frac: float = 1.5
    channel_names: Optional[List[str]] = None
    flow_time_bin: str = "all"
    # accumulated (C, S) sums, lazily shaped on first add
    _err: Optional[torch.Tensor] = field(default=None, repr=False)
    _tgt: Optional[torch.Tensor] = field(default=None, repr=False)
    _pred: Optional[torch.Tensor] = field(default=None, repr=False)
    _cross: Optional[torch.Tensor] = field(default=None, repr=False)
    _n_modes: Optional[torch.Tensor] = field(default=None, repr=False)
    _n_samples: int = 0
    _grid: int = 0

    @torch.no_grad()
    def add(self, v_pred: torch.Tensor, v_target: torch.Tensor, operator) -> None:
        vp = operator.P_A(v_pred.detach())
        vt = operator.P_A(v_target.detach())
        dv = vp - vt                                  # == P_A(v_pred - v_target)
        n = dv.shape[-1]
        c = dv.shape[1]
        idx, n_modes, n_shells = shell_index(n, dv.device)
        flat = idx.reshape(-1)

        DV = fft_ortho(dv)
        VT = fft_ortho(vt)
        VP = fft_ortho(vp)
        err = (DV.real ** 2 + DV.imag ** 2).sum(0)            # (C, N, N, N)
        tgt = (VT.real ** 2 + VT.imag ** 2).sum(0)
        prd = (VP.real ** 2 + VP.imag ** 2).sum(0)
        crs = (VP.real * VT.real + VP.imag * VT.imag).sum(0)

        def shell_sum(x):  # (C, N, N, N) -> (C, S)
            out = torch.zeros(c, n_shells, device=x.device)
            out.index_add_(1, flat, x.reshape(c, -1))
            return out

        e, tg, pr, cr = shell_sum(err), shell_sum(tgt), shell_sum(prd), shell_sum(crs)
        if self._err is None:
            self._err = torch.zeros_like(e)
            self._tgt = torch.zeros_like(tg)
            self._pred = torch.zeros_like(pr)
            self._cross = torch.zeros_like(cr)
            self._n_modes = n_modes.clone()
            self._grid = n
        self._err += e
        self._tgt += tg
        self._pred += pr
        self._cross += cr
        self._n_samples += dv.shape[0]

    def result(self) -> "ShellDiagnostic":
        if self._err is None:
            raise RuntimeError("BandDiagnosticAccumulator.add was never called")
        eps = 1e-30
        err = self._err.cpu().numpy()
        tgt = self._tgt.cpu().numpy()
        prd = self._pred.cpu().numpy()
        crs = self._cross.cpu().numpy()
        n_modes = self._n_modes.cpu().numpy()                # (S,)
        ns = max(self._n_samples, 1)
        C, S = err.shape
        modes_cs = np.maximum(n_modes[None, :], 1.0)

        shell_k = np.arange(S, dtype=np.float64)
        bands = band_of_shells(shell_k, self.factor, self.low_frac, self.high_frac)
        names = self.channel_names or [f"ch{c}" for c in range(C)]
        return ShellDiagnostic(
            factor=self.factor, grid=self._grid, n_samples=self._n_samples,
            n_batches_channels=C, flow_time_bin=self.flow_time_bin,
            channel_names=list(names), shell_k=shell_k, band_of_shell=bands,
            n_modes=n_modes,
            target_power_per_mode=(tgt / ns) / modes_cs,
            target_shell_energy=tgt / ns,
            error_power_per_mode=(err / ns) / modes_cs,
            error_shell_energy=err / ns,
            loss_fraction=err / max(err.sum(), eps),
            relative_shell_error=err / (tgt + eps),
            shell_cosine=crs / np.sqrt(np.maximum(prd, eps) * np.maximum(tgt, eps)),
        )


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class ShellDiagnostic:
    """Finished per-channel, per-shell diagnostic (all arrays are numpy, detached)."""

    factor: int
    grid: int
    n_samples: int
    n_batches_channels: int
    flow_time_bin: str
    channel_names: List[str]
    shell_k: np.ndarray                 # (S,)
    band_of_shell: List[str]            # length S
    n_modes: np.ndarray                 # (S,)
    target_power_per_mode: np.ndarray   # (C, S)
    target_shell_energy: np.ndarray
    error_power_per_mode: np.ndarray
    error_shell_energy: np.ndarray
    loss_fraction: np.ndarray
    relative_shell_error: np.ndarray
    shell_cosine: np.ndarray

    # -- band aggregation -------------------------------------------------- #
    def band_totals(self) -> Dict[str, Dict[str, float]]:
        """Per-band pooled quantities (summed over shells, then over channels)."""
        out: Dict[str, Dict[str, float]] = {}
        total_err = float(self.error_shell_energy.sum()) or 1e-30
        for band in ("low", "transition", "high"):
            sel = np.array([b == band for b in self.band_of_shell])
            err = float(self.error_shell_energy[:, sel].sum())
            tgt = float(self.target_shell_energy[:, sel].sum())
            out[band] = {
                "error_shell_energy": err,
                "target_shell_energy": tgt,
                "loss_fraction": err / total_err,
                "relative_shell_error": err / (tgt + 1e-30),
                "n_modes": float(self.n_modes[sel].sum()),
            }
        return out

    def per_channel_band_totals(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        total_err = float(self.error_shell_energy.sum()) or 1e-30
        for ci, name in enumerate(self.channel_names):
            out[name] = {}
            for band in ("low", "transition", "high"):
                sel = np.array([b == band for b in self.band_of_shell])
                err = float(self.error_shell_energy[ci, sel].sum())
                tgt = float(self.target_shell_energy[ci, sel].sum())
                out[name][band] = {
                    "loss_fraction": err / total_err,
                    "relative_shell_error": err / (tgt + 1e-30),
                }
        return out

    def to_dict(self) -> Dict:
        """JSON-serialisable machine-readable form."""
        return {
            "factor": self.factor,
            "grid": self.grid,
            "n_samples": self.n_samples,
            "flow_time_bin": self.flow_time_bin,
            "channel_names": self.channel_names,
            "shell_k": self.shell_k.tolist(),
            "band_of_shell": self.band_of_shell,
            "n_modes": self.n_modes.tolist(),
            "per_channel": {
                name: {
                    "target_power_per_mode": self.target_power_per_mode[c].tolist(),
                    "target_shell_energy": self.target_shell_energy[c].tolist(),
                    "error_power_per_mode": self.error_power_per_mode[c].tolist(),
                    "error_shell_energy": self.error_shell_energy[c].tolist(),
                    "loss_fraction": self.loss_fraction[c].tolist(),
                    "relative_shell_error": self.relative_shell_error[c].tolist(),
                    "shell_cosine": self.shell_cosine[c].tolist(),
                }
                for c, name in enumerate(self.channel_names)
            },
            "band_totals": self.band_totals(),
            "per_channel_band_totals": self.per_channel_band_totals(),
        }


# --------------------------------------------------------------------------- #
# Part 4 -- the diagnostic decision gate
# --------------------------------------------------------------------------- #
@dataclass
class GateThresholds:
    """Conservative, explicit thresholds for the imbalance gate. All documented so
    the verdict is a transparent function of measured quantities, not a judgement
    call after the fact."""

    # (1) concentration: <= this fraction of shells holds >= dominant_frac of loss
    small_subset_frac: float = 0.25
    dominant_frac: float = 0.80
    # (2) transition/high relative error is this-x the low-band relative error
    rel_error_ratio: float = 1.5
    # (3) transition+high shells carry LESS than this share of the total loss
    starved_loss_fraction: float = 0.20
    # (4) a criterion is "stable" if it holds in >= this fraction of diagnostics
    min_stable_frac: float = 0.80
    # (5) shells driving (2) must carry target power >= this x the peak shell energy
    near_zero_power_frac: float = 1e-3


@dataclass
class GateReport:
    verdict: str                       # NO_IMBALANCE | IMBALANCE_SUPPORTED | INCONCLUSIVE
    criteria: Dict[str, bool]
    measurements: Dict[str, float]
    n_diagnostics: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict,
            "criteria": self.criteria,
            "measurements": self.measurements,
            "n_diagnostics": self.n_diagnostics,
            "notes": self.notes,
        }

    def human_summary(self) -> str:
        lines = [f"DIAGNOSTIC GATE VERDICT: {self.verdict}",
                 f"(from {self.n_diagnostics} diagnostic(s))", ""]
        labels = {
            "c1_concentrated": "1. loss concentrated in a small subset of shells",
            "c2_high_rel_error": "2. transition/high shells retain larger relative error",
            "c3_gradient_starved": "3. those shells get too little gradient under spatial MSE",
            "c4_stable": "4. pattern stable across diagnostics",
            "c5_not_artifact": "5. not an FFT-normalisation / near-zero-power artifact",
        }
        for k, label in labels.items():
            lines.append(f"  [{'PASS' if self.criteria.get(k) else 'fail'}] {label}")
        lines.append("")
        for k, v in self.measurements.items():
            lines.append(f"    {k} = {v:.4g}")
        if self.notes:
            lines.append("")
            lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _single_diag_flags(d: ShellDiagnostic, th: GateThresholds) -> Dict[str, float]:
    """Per-diagnostic scalar signals used by the gate (pooled over channels)."""
    bands = d.band_totals()
    # (1) concentration over shells (pooled channels)
    shell_err = d.error_shell_energy.sum(axis=0)              # (S,)
    total = float(shell_err.sum()) or 1e-30
    order = np.argsort(shell_err)[::-1]
    cum = np.cumsum(shell_err[order]) / total
    k_needed = int(np.searchsorted(cum, th.dominant_frac) + 1)
    conc_frac = k_needed / max(len(shell_err), 1)

    low_rel = bands["low"]["relative_shell_error"]
    th_rel = max(bands["transition"]["relative_shell_error"],
                 bands["high"]["relative_shell_error"])
    rel_ratio = th_rel / max(low_rel, 1e-30)

    starved = bands["transition"]["loss_fraction"] + bands["high"]["loss_fraction"]

    # (5) target power actually present in the transition/high bands
    peak_shell_energy = float(d.target_shell_energy.sum(axis=0).max()) or 1e-30
    th_tgt = (bands["transition"]["target_shell_energy"]
              + bands["high"]["target_shell_energy"]) / peak_shell_energy

    return {
        "concentration_frac": conc_frac,
        "rel_error_ratio": rel_ratio,
        "transition_high_loss_fraction": starved,
        "transition_high_target_power_frac": th_tgt,
    }


def gate_decision(
    diagnostics: Sequence[ShellDiagnostic],
    thresholds: Optional[GateThresholds] = None,
) -> GateReport:
    """Apply the Part-4 gate to one or more diagnostics.

    A non-uniform target spectrum is *expected* and is not, on its own, a problem.
    ``IMBALANCE_SUPPORTED`` requires ALL five criteria; a clear absence of the
    imbalance signature returns ``NO_IMBALANCE``; anything mixed or unstable, or an
    artifact of near-zero-power shells, returns ``INCONCLUSIVE``. Conservative by
    construction: the default is not to reweight.
    """
    th = thresholds or GateThresholds()
    if not diagnostics:
        return GateReport("INCONCLUSIVE", {}, {}, 0, ["no diagnostics provided"])

    flags = [_single_diag_flags(d, th) for d in diagnostics]
    n = len(flags)

    def frac_true(pred) -> float:
        return sum(1 for f in flags if pred(f)) / n

    c1_rate = frac_true(lambda f: f["concentration_frac"] <= th.small_subset_frac)
    c2_rate = frac_true(lambda f: f["rel_error_ratio"] >= th.rel_error_ratio)
    c3_rate = frac_true(lambda f: f["transition_high_loss_fraction"] <= th.starved_loss_fraction)
    c5_rate = frac_true(lambda f: f["transition_high_target_power_frac"] >= th.near_zero_power_frac)

    c1 = c1_rate >= th.min_stable_frac
    c2 = c2_rate >= th.min_stable_frac
    c3 = c3_rate >= th.min_stable_frac
    c5 = c5_rate >= th.min_stable_frac
    # (4) stability: the *directional* imbalance signal (c2) must be consistent
    c4 = c2_rate >= th.min_stable_frac and c3_rate >= th.min_stable_frac

    criteria = {
        "c1_concentrated": bool(c1),
        "c2_high_rel_error": bool(c2),
        "c3_gradient_starved": bool(c3),
        "c4_stable": bool(c4),
        "c5_not_artifact": bool(c5),
    }
    measurements = {
        "concentration_frac_mean": float(np.mean([f["concentration_frac"] for f in flags])),
        "rel_error_ratio_mean": float(np.mean([f["rel_error_ratio"] for f in flags])),
        "transition_high_loss_fraction_mean": float(np.mean([f["transition_high_loss_fraction"] for f in flags])),
        "transition_high_target_power_frac_mean": float(np.mean([f["transition_high_target_power_frac"] for f in flags])),
        "c1_rate": c1_rate, "c2_rate": c2_rate, "c3_rate": c3_rate, "c5_rate": c5_rate,
    }
    notes: List[str] = []
    if n < 2:
        notes.append("only one diagnostic: criterion 4 (stability) cannot be established; "
                     "treat any IMBALANCE_SUPPORTED as INCONCLUSIVE until repeated over "
                     "batches/crops/seeds")

    if not c5:
        verdict = "INCONCLUSIVE"
        notes.append("high relative error sits in near-zero target-power shells -- likely "
                     "an FFT/normalisation artifact, not a learnable imbalance")
    elif c1 and c2 and c3 and c4 and c5 and n >= 2:
        verdict = "IMBALANCE_SUPPORTED"
    elif not (c2 or c3):
        verdict = "NO_IMBALANCE"
    else:
        verdict = "INCONCLUSIVE"

    return GateReport(verdict, criteria, measurements, n, notes)

