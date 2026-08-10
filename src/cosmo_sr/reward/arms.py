"""The proxy arms and the feature-schema version, with **no heavy imports**.

Why this is its own module
--------------------------
``ARMS`` used to live in :mod:`cosmo_sr.reward.phase_space`, which imports torch
through :mod:`cosmo_sr.eval.density`. That was fine while only torch-side code
needed to know the arm list. It stopped being fine once the *submitter* had to
decide, on a login node, whether a candidate directory already carries every
arm's features: importing torch to read a list of two strings is both slow and,
on a node without a GPU stack, occasionally fatal.

So the arm list and the schema version live here, in a module whose only imports
are from the standard library. :mod:`cosmo_sr.reward.phase_space` re-exports
``ARMS`` so existing callers are unaffected, and
``scripts/reward/_proxy_status.py`` imports this module directly.
(:mod:`cosmo_sr.reward.__init__` is itself torch-free -- it is a docstring and an
``__all__`` -- so ``from cosmo_sr.reward.arms import ARMS`` pulls in nothing.)

The feature-schema version
--------------------------
``FEATURE_SCHEMA_VERSION`` is stamped into every candidate manifest and into the
proxy table. It answers one question the input hashes cannot: *does the
``features.npz`` on disk contain the feature blocks the current code expects?*

The input hashes (LR, HR, weights, mask, Rockstar build) describe what went
**into** a candidate. Adding a new feature block changes none of them -- the
field is byte-identical -- so a pure input comparison would happily skip a
candidate whose ``features.npz`` predates arm C and leave a permanent hole in
the table. Bumping this version instead triggers a *features-only* recompute
from the saved ``field.npy``: no generator forward pass, no Rockstar re-run, and
crucially no change to ``field_sha``, so the existing label stays valid. That is
the whole point -- a new feature block must never cost a halo-finder run.

History
-------
``1``
    Arms ``a`` (13 density summaries) and ``b`` (``a`` + 9 phase-space
    summaries), both stored inline in the table as flat ``(2F,)`` vectors.
``2``
    Adds arm ``c``, the soft-Rockstar/DeepSets region model. Its per-tile
    features are a **token grid**, not a flat vector, and are stored in a
    sidecar array rather than inline (see ``ARM_STORAGE``).
"""
from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "ARMS",
    "ARM_STORAGE",
    "FEATURE_SCHEMA_VERSION",
    "MIN_FEATURE_SCHEMA_VERSION",
    "arm_storage",
    "check_arm",
    "features_key",
    "sidecar_arms",
    "tokens_key",
]

#: The three proxy arms being compared, in reporting order. ``a`` is the
#: density-only incumbent, ``b`` adds phase space, ``c`` is the soft-Rockstar
#: region model. Ordered, because every report iterates them and a set would
#: make two runs' tables disagree on column order.
ARMS: Tuple[str, ...] = ("a", "b", "c")

#: Bumped whenever the *content* of ``features.npz`` changes shape or meaning.
#: See the module docstring for why this is separate from the input hashes.
FEATURE_SCHEMA_VERSION: int = 2

#: The oldest schema whose labels (not features) are still usable. Labels are a
#: property of the field and the halo finder, neither of which any feature-schema
#: bump has touched, so every historical schema qualifies -- this exists so that
#: a *future* bump that does invalidate labels has an honest place to say so.
MIN_FEATURE_SCHEMA_VERSION: int = 1

#: How each arm's per-tile features are stored.
#:
#: ``inline``
#:     A flat ``(2F,)`` float vector per row, written straight into
#:     ``rows.jsonl``. Fine at ``F <= 22``: a row is a few hundred numbers.
#: ``sidecar``
#:     A ``(2, T, F_tok)`` token grid per row, written to a single memory-mappable
#:     ``.npy`` beside the table and referenced by row index. Arm C's tokens are
#:     ~1000 numbers per row and ~60,000 rows; inlining them would make
#:     ``rows.jsonl`` a multi-gigabyte text file that has to be JSON-parsed in
#:     full before a single epoch can start.
ARM_STORAGE: Dict[str, str] = {"a": "inline", "b": "inline", "c": "sidecar"}


def check_arm(arm: str) -> str:
    """Normalise and validate an arm name, or raise with the valid list."""
    a = str(arm).lower()
    if a not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}, got {arm!r}")
    return a


def arm_storage(arm: str) -> str:
    return ARM_STORAGE[check_arm(arm)]


def sidecar_arms() -> Tuple[str, ...]:
    return tuple(a for a in ARMS if ARM_STORAGE[a] == "sidecar")


def features_key(arm: str) -> str:
    """The ``features.npz`` / table key holding one arm's per-tile features."""
    return f"features_{check_arm(arm)}"


def tokens_key(arm: str) -> str:
    """The sidecar file name for a token-stored arm."""
    return f"tokens_{check_arm(arm)}.npy"
