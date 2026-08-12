"""Arm C: a soft-Rockstar / DeepSets region proxy on a local token grid.

What this is, and is not
------------------------
Arms A and B reduce a tile to one flat vector of crop-level summaries, so a
tile with one hot massive clump and a tile with three cold small ones can look
alike. Rockstar does not summarise -- it links locally in phase space -- and
this arm is the smallest differentiable gesture at that: keep the *where*, at
token resolution, and let a shared per-token encoder plus permutation-invariant
pooling decide what to aggregate.

It is deliberately **not** an all-pairs particle graph and not a differentiable
re-implementation of Rockstar's FOF hierarchy. The token grid is 8^3 per tile,
the encoder is a two-layer MLP shared across tokens, and the whole model is
small enough to sit inside an SR2 fine-tuning step next to the generator --
that constraint is binding, because the passing arm's extractor becomes the
actor's reward path.

The pipeline, in order
----------------------
1. Deposit mass, momentum and squared speed on the valid-centre grid, in the
   **same single CIC pass arm B uses**
   (:func:`cosmo_sr.reward.phase_space.deposit_phase_space`): the arms must not
   disagree about which particles landed in which cell.
2. Reduce the deposit to a fixed ``t^3`` token grid (``t = 8``; a 64^3 crop's
   32^3 valid-centre region gives 4^3 cells per token) with mass-weighted
   pooling.
3. Give each token: local density, multiscale density contrast, velocity
   dispersion, velocity coherence and (negative) divergence, and a
   kinetic-to-binding proxy. All dimensionless and O(1).
4. Encode each token with a shared MLP.
5. Pool with mean, max and log-sum-exp over tokens.
6. The token grid stores the candidate block and the candidate-minus-frozen
   block, exactly the pairing arms A/B use, so "improved relative to frozen
   SR2" is visible per token.
7. Predict exactly the same catalog sufficient statistics as arms A and B:
   ``n_sub``, ``n_host``, ``occ_numerator``, through the shared
   :class:`cosmo_sr.reward.catalog_proxy.ProxyBase` head.

Storage
-------
A tile's arm-C features are a ``(2, T, F)`` token grid (``T = t^3``), not a
flat vector, and are stored in a sidecar ``tokens_c.npy`` next to the proxy
table rather than inline in ``rows.jsonl`` (see
:data:`cosmo_sr.reward.arms.ARM_STORAGE`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .catalog_proxy import ProxyBase, register_proxy
from .phase_space import (
    G_MPC_KMS2_PER_MSUN, PhaseSpaceConfig, _central_divergence,
    deposit_phase_space,
)
from .soft_structure import SoftStructureConfig, _smooth

__all__ = [
    "SoftRockstarConfig",
    "SoftRockstarProxy",
    "SoftRockstarProxyConfig",
    "paired_token_feature_names",
    "soft_rockstar_paired_tokens",
    "soft_rockstar_tokens",
    "token_feature_names",
]

#: Valid pooling names for :class:`SoftRockstarProxy`.
POOLS = ("mean", "max", "lse")


# --------------------------------------------------------------------------- #
# Feature configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SoftRockstarConfig:
    """The token-grid feature choices. Serialized into every candidate.

    Fixed before any candidate is generated: the token features live in
    ``features.npz`` on disk, so changing any of this is a feature-schema bump
    (:data:`cosmo_sr.reward.arms.FEATURE_SCHEMA_VERSION`), not a tunable.
    """

    #: Tokens per axis. 8 over a 32^3 valid-centre region means 4^3 cells per
    #: token -- ~0.8 Mpc/h, the Lagrangian size of the subhalos occupation is
    #: decided on.
    tokens_per_axis: int = 8
    #: Density-contrast smoothing scales, matched to
    #: ``SoftStructureConfig.peak_scales`` so a token contrast and a crop peak
    #: feature describe the same smoothing.
    contrast_scales: Tuple[int, ...] = (1, 2, 4)
    #: One coherence scale, not the ladder: the token grid already localises,
    #: so one scale keeps the token vector short.
    coherence_scale: int = 2

    def __post_init__(self) -> None:
        if int(self.tokens_per_axis) <= 0:
            raise ValueError("tokens_per_axis must be positive")
        if not self.contrast_scales:
            raise ValueError("need at least one contrast scale")
        if int(self.coherence_scale) <= 0:
            raise ValueError("coherence_scale must be positive")

    @property
    def n_tokens(self) -> int:
        return int(self.tokens_per_axis) ** 3

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["contrast_scales"] = [int(s) for s in self.contrast_scales]
        return d


def token_feature_names(rcfg: Optional[SoftRockstarConfig] = None) -> List[str]:
    rcfg = rcfg or SoftRockstarConfig()
    return (
        ["tok_log_mass"]
        + [f"tok_contrast_s{s}" for s in rcfg.contrast_scales]
        + ["tok_vdisp", "tok_neg_div_v", "tok_vcoh", "tok_kinetic_binding"]
    )


def paired_token_feature_names(
    rcfg: Optional[SoftRockstarConfig] = None,
) -> List[str]:
    """Names of the ``2F`` channels a token presents to the encoder.

    The first ``F`` are the candidate block, the second the candidate-minus-
    frozen block -- the same convention as ``arm_paired_feature_names``.
    """
    base = token_feature_names(rcfg)
    return list(base) + [f"d_{n}" for n in base]


# --------------------------------------------------------------------------- #
# The token grid
# --------------------------------------------------------------------------- #
def _pool_weighted(x: torch.Tensor, w: torch.Tensor, p: int) -> torch.Tensor:
    """Mass-weighted mean of ``x`` over non-overlapping ``p^3`` blocks.

    Weighted, not plain: a token is mostly empty cells, and an unweighted mean
    of a per-cell *ratio* (velocity, dispersion) would be dominated by cells
    whose ratio is noise over a near-zero mass.
    """
    num = F.avg_pool3d(w * x, int(p))
    den = F.avg_pool3d(w, int(p)).clamp_min(1e-8)
    return num / den


def soft_rockstar_tokens(
    field: torch.Tensor,
    cfg: Optional[SoftStructureConfig] = None,
    ps: Optional[PhaseSpaceConfig] = None,
    rcfg: Optional[SoftRockstarConfig] = None,
) -> torch.Tensor:
    """``(B, T, F)`` token features of a ``(B, C>=6, N, N, N)`` crop.

    Differentiable in the field. ``T = tokens_per_axis^3`` and ``F`` is
    ``len(token_feature_names())``; the token order is the C order of the token
    grid and is therefore deterministic -- the *model* is what must not depend
    on it, which is the permutation-invariance test's job.
    """
    cfg = cfg or SoftStructureConfig()
    ps = ps or PhaseSpaceConfig()
    rcfg = rcfg or SoftRockstarConfig()
    if field.dim() != 5:
        raise ValueError(f"expected (B, C, N, N, N), got {tuple(field.shape)}")
    if field.shape[1] < 6:
        raise ValueError(
            f"arm 'c' needs 6 channels (displacement + velocity), got "
            f"{field.shape[1]}; a 3-channel field has no phase space to token-ise")

    m, vbar, sigma2 = deposit_phase_space(
        field[:, 0:3], field[:, 3:6], cfg, ps)
    b, r = int(m.shape[0]), int(m.shape[-1])
    t = int(rcfg.tokens_per_axis)
    if r % t:
        raise ValueError(
            f"valid-centre region {r}^3 is not divisible into {t}^3 tokens; "
            "tokens_per_axis must divide region_of(crop)")
    p = r // t
    dims = (1, 2, 3, 4)

    # The crop's own bulk flow is a large-scale mode -- identical in candidate
    # and frozen, and not something fine-tuning can move -- so it is removed
    # exactly as arm B removes it.
    total = m.sum(dim=dims).clamp_min(1e-12)
    v0 = (m * vbar).sum(dim=(2, 3, 4)) / total.unsqueeze(1)
    dv = vbar - v0.view(-1, 3, 1, 1, 1)
    dv2 = (dv ** 2).sum(dim=1, keepdim=True)

    u = torch.log(m.clamp_min(1e-6))

    # 1. Local density: log of the token's mean (1 + delta).
    tok_log_mass = torch.log(F.avg_pool3d(m, p).clamp_min(1e-6))

    # 2. Multiscale density contrast: how far the token's cells rise above
    # their own smoothed neighbourhood, mass-weighted -- the token-local
    # analogue of the crop peak-contrast ladder, threshold-free.
    contrasts = [_pool_weighted(u - torch.log(_smooth(m, int(s)).clamp_min(1e-6)),
                                m, p)
                 for s in rcfg.contrast_scales]

    vref = float(ps.v_ref_km_s)
    # 3. Velocity dispersion of the token's material.
    tok_vdisp = _pool_weighted(sigma2, m, p).clamp_min(0.0).sqrt() / vref

    # 4. Convergence of the bulk-free flow: positive where material streams in.
    tok_neg_div = _pool_weighted(-_central_divergence(dv), m, p) / vref

    # 5. Coherence: fraction of the token's kinetic energy that survives box
    # smoothing of the (mass-weighted) velocity field. Coherent infall keeps
    # it; virialised motion loses it.
    s = int(rcfg.coherence_scale)
    vs = _smooth(m * dv, s) / _smooth(m, s).clamp_min(1e-6)
    ke = dv2 + sigma2
    tok_vcoh = (_pool_weighted((vs ** 2).sum(dim=1, keepdim=True), m, p)
                / _pool_weighted(ke, m, p).clamp_min(1e-6))

    # 6. Kinetic against binding energy of the token as one lump: sigma^2 + v^2
    # against G M_tok / r_tok, with r_tok the fixed token half-width. A crude
    # virial estimator on purpose -- it is a feature, not a measurement.
    cell_mpc = float(cfg.boxsize_mpc_h) / float(cfg.ng_hr) / float(cfg.grid_mult)
    r_tok_mpc = 0.5 * cell_mpc * p
    mass_msun = (F.avg_pool3d(m, p) * float(p ** 3)
                 / float(int(cfg.grid_mult) ** 3)
                 * float(ps.particle_mass_msun_h))
    binding = G_MPC_KMS2_PER_MSUN * mass_msun / r_tok_mpc
    tok_kb = torch.log10((_pool_weighted(ke, m, p) + 1e-6) / (binding + 1e-6))

    cols = [tok_log_mass, *contrasts, tok_vdisp, tok_neg_div, tok_vcoh, tok_kb]
    # (B, F, t, t, t) -> (B, T, F), C-ordered tokens.
    grid = torch.cat(cols, dim=1)
    return grid.reshape(b, grid.shape[1], -1).transpose(1, 2).contiguous()


def soft_rockstar_paired_tokens(
    candidate: torch.Tensor,
    frozen: torch.Tensor,
    cfg: Optional[SoftStructureConfig] = None,
    ps: Optional[PhaseSpaceConfig] = None,
    rcfg: Optional[SoftRockstarConfig] = None,
) -> torch.Tensor:
    """``(B, 2, T, F)``: candidate tokens and candidate-minus-frozen tokens.

    Same contract as :func:`cosmo_sr.reward.phase_space.arm_paired_features`,
    token-wise. The frozen branch is detached: a gradient through the reference
    would let the optimiser shrink the difference by moving a baseline whose
    weights never change. Slots pair token-by-token because the grids share
    their geometry -- token ``k`` of the candidate IS token ``k`` of frozen.
    """
    cand = soft_rockstar_tokens(candidate, cfg, ps, rcfg)
    with torch.no_grad():
        base = soft_rockstar_tokens(frozen, cfg, ps, rcfg)
    return torch.stack([cand, cand - base.detach()], dim=1)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass
class SoftRockstarProxyConfig:
    """Shape of one arm-C proxy. One fixed configuration; there is no sweep."""

    n_token_features: int = 8
    n_tokens: int = 512
    n_sub_bins: int = 6
    n_host_bins: int = 5
    token_hidden: Tuple[int, ...] = (64, 64)
    embed_dim: int = 32
    pools: Tuple[str, ...] = ("mean", "max", "lse")
    dropout: float = 0.0
    #: Legacy ``softplus(raw) * scale`` scale, kept for absolute-count checkpoints.
    output_scale: Tuple[float, ...] = ()
    #: Predict a signed frozen-relative log-count residual (see
    #: :class:`cosmo_sr.reward.catalog_proxy.ProxyConfig`). False for a checkpoint
    #: saved before this field existed.
    residual_head: bool = True

    def __post_init__(self) -> None:
        bad = [p for p in self.pools if p not in POOLS]
        if bad or not self.pools:
            raise ValueError(f"pools must be a non-empty subset of {POOLS}, "
                             f"got {list(self.pools)}")

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["token_hidden"] = [int(h) for h in self.token_hidden]
        d["pools"] = [str(p) for p in self.pools]
        d["output_scale"] = [float(x) for x in self.output_scale]
        return d

    @staticmethod
    def from_dict(d: Mapping) -> "SoftRockstarProxyConfig":
        return SoftRockstarProxyConfig(
            n_token_features=int(d["n_token_features"]),
            n_tokens=int(d["n_tokens"]),
            n_sub_bins=int(d["n_sub_bins"]),
            n_host_bins=int(d["n_host_bins"]),
            token_hidden=tuple(int(h) for h in d.get("token_hidden", (64, 64))),
            embed_dim=int(d.get("embed_dim", 32)),
            pools=tuple(str(p) for p in d.get("pools", ("mean", "max", "lse"))),
            dropout=float(d.get("dropout", 0.0)),
            output_scale=tuple(float(x) for x in d.get("output_scale", ())),
            residual_head=bool(d.get("residual_head", False)),
        )

    @property
    def n_outputs(self) -> int:
        return int(self.n_sub_bins) + 2 * int(self.n_host_bins)


@register_proxy
class SoftRockstarProxy(ProxyBase):
    """Token grid -> ``(N, H, S)``. Arm C.

    A DeepSets model: a shared MLP encodes each token's ``2F`` channels, the
    embeddings are pooled over tokens with mean, max and log-sum-exp, and the
    shared :class:`ProxyBase` head turns the pooled vector into the same three
    count blocks arms A and B predict. Permutation-invariant over tokens by
    construction -- nothing between the encoder and the head sees the token
    index.
    """

    CONFIG = SoftRockstarProxyConfig

    def __init__(self, cfg: Optional[SoftRockstarProxyConfig] = None, *,
                 seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or SoftRockstarProxyConfig()
        if seed is not None:
            torch.manual_seed(int(seed))
        f2 = 2 * int(self.cfg.n_token_features)
        dims = [f2, *[int(h) for h in self.cfg.token_hidden],
                int(self.cfg.embed_dim)]
        layers: List[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                if self.cfg.dropout > 0:
                    layers.append(nn.Dropout(float(self.cfg.dropout)))
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(len(self.cfg.pools) * int(self.cfg.embed_dim),
                      self.cfg.n_outputs),
        )
        self._zero_init_head(self.head[-1])
        self.seed = None if seed is None else int(seed)

        scale = list(self.cfg.output_scale) or [1.0] * self.cfg.n_outputs
        if len(scale) != self.cfg.n_outputs:
            raise ValueError(
                f"output_scale has {len(scale)} entries, expected "
                f"{self.cfg.n_outputs}")
        self.register_buffer("output_scale",
                             torch.as_tensor(scale, dtype=torch.float64))
        self.register_buffer("feat_mean", torch.zeros(f2, dtype=torch.float64))
        self.register_buffer("feat_std", torch.ones(f2, dtype=torch.float64))

    # -- feature scaling ----------------------------------------------------- #
    @property
    def n_features(self) -> int:
        """Channels per token (2F). Named for parity with :class:`ProxyConfig`."""
        return 2 * int(self.cfg.n_token_features)

    def _flat_channels(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, T, 2F)`` from ``(B, 2, T, F)``: the encoder's per-token input."""
        if tokens.dim() != 4 or tokens.shape[1] != 2 \
                or tokens.shape[-1] != int(self.cfg.n_token_features):
            raise ValueError(
                f"expected (B, 2, T, {self.cfg.n_token_features}) tokens, got "
                f"{tuple(tokens.shape)}")
        return torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1)

    def fit_standardizer(self, tokens: torch.Tensor | np.ndarray) -> "SoftRockstarProxy":
        """Per-CHANNEL mean/std over all training rows and tokens.

        Per channel and not per token: a token is a position in a permutation-
        invariant set, so a per-token statistic would build the token index
        into the model through the back door.
        """
        x = torch.as_tensor(np.asarray(tokens, dtype=np.float64))
        flat = self._flat_channels(x).reshape(-1, self.n_features)
        # Keep registered buffers: assignment would leave them on CPU after
        # ``model.to(cuda)`` and break the first training step.
        mean = flat.mean(dim=0).to(device=self.feat_mean.device,
                                   dtype=self.feat_mean.dtype)
        std = flat.std(dim=0, unbiased=False).to(device=self.feat_std.device,
                                                  dtype=self.feat_std.dtype)
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(torch.where(std < 1e-12, torch.ones_like(std), std))
        return self

    # -- forward -------------------------------------------------------------- #
    def forward(self, tokens: torch.Tensor,
                frozen: Optional["TorchSummary"] = None) -> Dict[str, torch.Tensor]:
        """``{'n_sub': (B,J), 'n_host': (B,I), 'occ_numerator': (B,I)}``.

        ``frozen`` is the measured frozen tile summary the residual head
        reconstructs against; ignored by an absolute-count checkpoint.
        """
        x = self._flat_channels(tokens.to(device=self.feat_mean.device,
                                          dtype=self.feat_mean.dtype))
        x = ((x - self.feat_mean.to(device=x.device, dtype=x.dtype))
             / self.feat_std.to(device=x.device, dtype=x.dtype))
        enc_dtype = next(self.encoder.parameters()).dtype
        h = self.encoder(x.to(enc_dtype))                       # (B, T, E)
        n_tok = float(h.shape[1])
        pooled = []
        for p in self.cfg.pools:
            if p == "mean":
                pooled.append(h.mean(dim=1))
            elif p == "max":
                pooled.append(h.max(dim=1).values)
            else:  # lse -- shifted by log T so it is a soft max, not a soft sum,
                   # and does not grow with token count.
                pooled.append(torch.logsumexp(h, dim=1)
                              - torch.log(torch.tensor(n_tok, dtype=h.dtype,
                                                       device=h.device)))
        raw = self.head(torch.cat(pooled, dim=-1))
        return self._counts_from_raw(raw, frozen)
