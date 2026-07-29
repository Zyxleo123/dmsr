"""Test-time scaling for the pretrained SR2/SRS generator.

Spend more inference compute on one LR box -- draw several SR realisations,
score them, keep the good one -- instead of retraining anything. The generator
and its weights are frozen throughout; the only thing that varies is the noise.

Module map (each stage builds on the one above)::

    srs_noise   ControlledG: SR2 with explicit / recordable noise at all six
                injection sites, state-dict compatible with SRmodel/G_z0.pt
    sampling    reproducible multi-sample inference; a root seed defines one
                full-box realisation, tile seeds derive from (seed, coordinate)
    metrics     per-candidate statistics on complete periodic boxes: r(k), T(k),
                density power/PDF, bispectra, velocity, LR consistency, seams
    scores      the phase and statistical oracles, z-normalised per component
    bootstrap   best-of-K curves with box-level paired bootstrap intervals
    features    candidate features available without the test HR box
    verifier    pairwise-ranking selectors over those features (and a patch CNN)
    refine      noise optimisation, coarse to fine, with a distribution prior
    tiling      coordinate-indexed global noise and joint tile selection

Entry points live in ``scripts/``: ``eval_srs_tts.py`` (oracle audit),
``train_srs_verifier.py`` (selector), ``tts_final_table.py`` (comparison table).

Before reading any result: the pretrained ``G_z0`` has ``|std|`` of order 1e-3 at
five of its six noise sites and 5e-2 at the sixth (the finest). Its stochasticity
is therefore concentrated at the smallest scales, and a measured probe on real
data puts the across-seed spread at ~0.4% of rms for displacement and ~50% for
velocity. Selection leverage, if any, should be expected to look like that.
"""
from __future__ import annotations

from .srs_noise import (  # noqa: F401
    NOISE_SITES,
    STAGE_SITES,
    ControlledG,
    load_controlled_generator,
    noise_site_layout,
    site_shapes,
)
from .sampling import (  # noqa: F401
    Candidate,
    GlobalNoiseField,
    derive_tile_seed,
    generate_srs_candidates,
    iter_srs_candidates,
    super_resolve_srs_seeded,
)

__all__ = [
    "Candidate",
    "ControlledG",
    "GlobalNoiseField",
    "NOISE_SITES",
    "STAGE_SITES",
    "derive_tile_seed",
    "generate_srs_candidates",
    "iter_srs_candidates",
    "load_controlled_generator",
    "noise_site_layout",
    "site_shapes",
    "super_resolve_srs_seeded",
]
