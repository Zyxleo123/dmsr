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


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config into a plain dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping, got {type(cfg)}")
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
