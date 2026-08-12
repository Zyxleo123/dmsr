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

The five arms
-------------
``a``  density-only summaries (incumbent);
``b``  ``a`` + phase-space summaries;
``c``  soft-Rockstar / DeepSets region model on an 8^3 token grid;
``d``  a 3-D CNN on the SAME 8^3 token grid as ``c`` -- it keeps the ordered
       token positions and models adjacency with local 3^3 convolutions before
       pooling, where ``c`` pools permutation-invariantly. ``d`` reuses ``c``'s
       ``tokens_c.npy`` sidecar exactly, so the two share their feature cache and
       differ only in the model that reads it;
``e``  a small strided 3-D CNN on the FULL 32^3 Eulerian phase-space grid --
       five channels (log density, three bulk-subtracted mean velocities, one
       dispersion), no hand-designed summaries. Its per-tile features are a
       ``(2, 5, 32, 32, 32)`` grid (candidate block and candidate-minus-frozen
       block), stored in its own ``grids_e.npy`` sidecar as float16;
``f``  the original SR2 discriminator: the exact 20-channel ``critic_input``
       (6 upsampled LR + 6 candidate displacement/velocity + 8 inverse-pixel-
       shuffled fine density) through the ``SR2Critic`` convolutional body, with
       the scalar critic head replaced by the common 16-output catalog head. It
       has **no cached feature block at all** -- a 20x64^3 tensor per tile is far
       too large to store -- so its input is streamed from the saved candidate
       ``field.npy`` and the LR box and built on the GPU at train time. Its
       storage is therefore ``"field"``, not ``"sidecar"``.

The feature-schema version
--------------------------
``FEATURE_SCHEMA_VERSION`` is stamped into every candidate manifest and into the
proxy table. It answers one question the input hashes cannot: *does the
``features.npz`` on disk contain the feature blocks the current code expects?*

The input hashes (LR, HR, weights, mask, Rockstar build) describe what went
**into** a candidate. Adding a new feature block changes none of them -- the
field is byte-identical -- so a pure input comparison would happily skip a
candidate whose ``features.npz`` predates a new arm and leave a permanent hole in
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
``3``
    Adds arm ``e``'s full-grid phase-space features (a ``(2, 5, 32, 32, 32)``
    grid in the ``grids_e.npy`` sidecar) and an arm-neutral ``field_changed``
    flag per tile, computed from the raw six-channel candidate-versus-frozen
    field so it also detects velocity-only changes. Arm ``d`` reuses arm ``c``'s
    tokens and arm ``f`` streams raw fields at train time; neither adds a stored
    feature block.
"""
from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "ARMS",
    "ARM_SIDECAR_FILE",
    "ARM_STORAGE",
    "FEATURE_SCHEMA_VERSION",
    "FIELD_CHANGED_KEY",
    "MIN_FEATURE_SCHEMA_VERSION",
    "arm_storage",
    "check_arm",
    "features_key",
    "field_arms",
    "owned_sidecar_arms",
    "sidecar_arms",
    "sidecar_file",
    "tokens_key",
]

#: The six proxy arms being compared, in reporting order. ``a`` is the
#: density-only incumbent, ``b`` adds phase space, ``c`` is the soft-Rockstar
#: DeepSets region model, ``d`` is a 3-D CNN on the same token grid, ``e`` is a
#: CNN on the full Eulerian phase-space grid, ``f`` is the SR2 discriminator
#: architecture. Ordered, because every report iterates them and a set would make
#: two runs' tables disagree on column order.
ARMS: Tuple[str, ...] = ("a", "b", "c", "d", "e", "f")

#: Bumped whenever the *content* of ``features.npz`` (or a sidecar) changes shape
#: or meaning. See the module docstring for why this is separate from the input
#: hashes.
FEATURE_SCHEMA_VERSION: int = 3

#: The oldest schema whose labels (not features) are still usable. Labels are a
#: property of the field and the halo finder, neither of which any feature-schema
#: bump has touched, so every historical schema qualifies -- this exists so that
#: a *future* bump that does invalidate labels has an honest place to say so.
MIN_FEATURE_SCHEMA_VERSION: int = 1

#: The per-tile column carrying the arm-neutral changed-versus-frozen flag. One
#: flag for the whole row (not per arm): it is read off the RAW six-channel
#: candidate-versus-frozen field, so it detects a velocity-only change a
#: density-only feature difference would miss, and every source then weights the
#: same tiles up (see ``scripts/reward/_proxy_data.row_weights``).
FIELD_CHANGED_KEY: str = "field_changed"

#: How each arm's per-tile features are stored.
#:
#: ``inline``
#:     A flat ``(2F,)`` float vector per row, written straight into
#:     ``rows.jsonl``. Fine at ``F <= 22``: a row is a few hundred numbers.
#: ``sidecar``
#:     A grid per row, written to a single memory-mappable ``.npy`` beside the
#:     table and referenced by row index. Arm C/D's tokens (~1000 numbers/row)
#:     and arm E's full grid (~10000 numbers/row) would make ``rows.jsonl`` a
#:     multi-gigabyte text file that has to be JSON-parsed in full before a
#:     single epoch can start.
#: ``field``
#:     No cached feature block: the arm's input is built at train time from the
#:     saved candidate ``field.npy`` and the LR box. Arm F's 20x64^3 discriminator
#:     input is ~5M numbers per tile -- ~800 GB across the table -- so it is
#:     streamed and constructed on the GPU rather than stored.
ARM_STORAGE: Dict[str, str] = {
    "a": "inline", "b": "inline", "c": "sidecar", "d": "sidecar", "e": "sidecar",
    "f": "field",
}

#: The sidecar file each sidecar-stored arm READS. Arm ``d`` shares arm ``c``'s
#: token cache exactly -- "do not duplicate the cache" -- so both name
#: ``tokens_c.npy``; arm ``e`` owns its own full-grid file. The distinction
#: between "reads a file" (every sidecar arm) and "owns a file the indexer must
#: write" (:func:`owned_sidecar_arms`) is what keeps arm ``d`` from asking the
#: indexer for a block nobody computes.
ARM_SIDECAR_FILE: Dict[str, str] = {
    "c": "tokens_c.npy", "d": "tokens_c.npy", "e": "grids_e.npy",
}


def check_arm(arm: str) -> str:
    """Normalise and validate an arm name, or raise with the valid list."""
    a = str(arm).lower()
    if a not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}, got {arm!r}")
    return a


def arm_storage(arm: str) -> str:
    return ARM_STORAGE[check_arm(arm)]


def sidecar_arms() -> Tuple[str, ...]:
    """Every arm whose features are read from a sidecar (``c``, ``d``, ``e``)."""
    return tuple(a for a in ARMS if ARM_STORAGE[a] == "sidecar")


def field_arms() -> Tuple[str, ...]:
    """Every arm built by streaming raw fields at train time (``f``).

    These have no cached feature block, so the indexer writes nothing for them
    and the trainer/gate build their input from the candidate ``field.npy`` and
    the LR box on demand.
    """
    return tuple(a for a in ARMS if ARM_STORAGE[a] == "field")


def owned_sidecar_arms() -> Tuple[str, ...]:
    """The arms whose sidecar FILE the indexer must write, one per distinct file.

    Arm ``d`` reuses arm ``c``'s ``tokens_c.npy`` and owns nothing, so it is not
    here: the feature extractor computes tokens once (for ``c``/``d``) and the
    full grid once (for ``e``), and the indexer streams exactly those two files.
    """
    seen: set = set()
    out = []
    for a in ARMS:
        if ARM_STORAGE[a] != "sidecar":
            continue
        f = ARM_SIDECAR_FILE[a]
        if f not in seen:
            seen.add(f)
            out.append(a)
    return tuple(out)


def features_key(arm: str) -> str:
    """The ``features.npz`` / table key holding one arm's per-tile features."""
    return f"features_{check_arm(arm)}"


def sidecar_file(arm: str) -> str:
    """The sidecar file name a sidecar-stored arm reads its features from."""
    a = check_arm(arm)
    if a not in ARM_SIDECAR_FILE:
        raise ValueError(f"arm {a!r} is not sidecar-stored; it has no sidecar file")
    return ARM_SIDECAR_FILE[a]


def tokens_key(arm: str) -> str:
    """Historical alias for :func:`sidecar_file` (the file need not hold tokens)."""
    return sidecar_file(arm)
