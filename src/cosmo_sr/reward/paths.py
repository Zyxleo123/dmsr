"""Where the bulky artifacts live.

Home has a tight quota (Rockstar ``*.particles`` filled it once already), so
every generated field, catalog, checkpoint and replay shard goes under ``$ZFS``.
Only code, configs and small JSON/markdown summaries stay in the repository.

Override the root with ``DMSR_REWARD_ROOT`` (tests do this with ``tmp_path``).
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "AUDITS",
    "CATALOG_CACHE",
    "CHECKPOINTS",
    "EVAL",
    "ORACLE",
    "REPLAY",
    "RESIDUAL_CACHE",
    "SR2_BASE_CACHE",
    "reward_root",
    "subdir",
    "zfs_root",
]

_DEFAULT_ZFS = "/zfsauton/scratch/yixiz"


def zfs_root() -> Path:
    """Scratch root, from ``$ZFS`` when set."""
    return Path(os.environ.get("ZFS", _DEFAULT_ZFS))


def reward_root() -> Path:
    """Root of every artifact this package writes."""
    env = os.environ.get("DMSR_REWARD_ROOT")
    if env:
        return Path(env)
    return zfs_root() / "DMSR" / "dmsr_reward"


def subdir(*parts: str, create: bool = False) -> Path:
    """A path under :func:`reward_root`, optionally created."""
    p = reward_root().joinpath(*parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


# Names are resolved lazily via these helpers so that DMSR_REWARD_ROOT can be
# set after import (pytest fixtures, per-job sbatch env).
def SR2_BASE_CACHE(create: bool = False) -> Path:
    return subdir("cache", "sr2_base", create=create)


def RESIDUAL_CACHE(create: bool = False) -> Path:
    return subdir("cache", "residual_targets", create=create)


def AUDITS(name: str = "", create: bool = False) -> Path:
    return subdir("audits", name, create=create) if name else subdir("audits", create=create)


def CHECKPOINTS(name: str = "", create: bool = False) -> Path:
    return subdir("checkpoints", name, create=create) if name else subdir("checkpoints", create=create)


def CATALOG_CACHE(create: bool = False) -> Path:
    return subdir("catalog_cache", create=create)


def ORACLE(run_name: str = "", create: bool = False) -> Path:
    return subdir("oracle", run_name, create=create) if run_name else subdir("oracle", create=create)


def REPLAY(round_name: str = "", create: bool = False) -> Path:
    return subdir("replay", round_name, create=create) if round_name else subdir("replay", create=create)


def EVAL(run_name: str = "", create: bool = False) -> Path:
    return subdir("eval", run_name, create=create) if run_name else subdir("eval", create=create)
