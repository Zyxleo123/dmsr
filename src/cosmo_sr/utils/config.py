"""Config loading and run-metadata capture.

Every saved run must record: the config YAML, this project's git commit, the
external repo commit hashes, environment info, checkpoint and metrics. This
module handles everything except the checkpoint/metrics.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins). Returns a new dict."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Load a YAML config into a plain dict.

    Supports single inheritance via a top-level ``base:`` key holding a path
    relative to the config's own directory. The parent is loaded first and the
    child is deep-merged over it, so a stage config need only state what differs
    from ``_base.yaml`` -- which is what makes "Stage C and D differ in exactly
    one field" checkable by reading the files.
    """
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular config inheritance at {path}")
    _seen.add(path)

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping, got {type(cfg)}")

    base_ref = cfg.pop("base", None)
    if base_ref:
        parent = load_config(path.parent / str(base_ref), _seen)
        cfg = _deep_merge(parent, cfg)
    return cfg


def apply_overrides(cfg: Dict[str, Any], pairs: Optional[list]) -> Dict[str, Any]:
    """Apply ``dotted.key=value`` overrides in-place (values parsed as YAML).

    Example: ``apply_overrides(cfg, ["train.steps=30", "wandb.mode=offline"])``.
    Missing intermediate dicts are created. Useful for smoke/debug launches.
    """
    if not pairs:
        return cfg
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = value
    return cfg


def save_config(cfg: Dict[str, Any], path: str | os.PathLike) -> None:
    """Dump a config dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _git_hash(repo_dir: str | os.PathLike) -> Optional[str]:
    repo_dir = Path(repo_dir)
    if not repo_dir.exists():
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _project_root() -> Path:
    # src/cosmo_sr/utils/config.py -> project root is three parents up from src
    return Path(__file__).resolve().parents[3]


def collect_run_metadata(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect git hashes, external repo commits and environment info."""
    root = _project_root()
    externals = {
        "map2map": _git_hash(root / "external" / "map2map"),
        "SRS-map2map": _git_hash(root / "external" / "SRS-map2map"),
    }

    torch_version = None
    cuda_available = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        pass

    import numpy as np

    meta: Dict[str, Any] = {
        "project_git_commit": _git_hash(root),
        "external_commits": externals,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "argv": sys.argv,
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_metadata(run_dir: str | os.PathLike, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write ``env.json`` into ``run_dir`` and return the metadata dict."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_run_metadata(extra)
    with open(run_dir / "env.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta
