"""Shared plumbing for the DIRECT SR2 fine-tuning scripts.

Kept separate from ``_common.py`` (which serves the residual line) so that the
two lines cannot drift into sharing a default by accident. Everything here reads
``configs/reward/sr2_direct_finetune.yaml`` and, through it, the residual line's
``configs/reward/reward.yaml`` for the bins, the splits and the fitted reward --
those are shared *deliberately*, because a reward that differed between the two
lines would make their results incomparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts" / "reward") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reward"))

from _common import (  # noqa: E402
    append_jsonl, banner, bins_of, read_jsonl, write_json,
)

from cosmo_sr.reward import paths  # noqa: E402
from cosmo_sr.reward.direct_gates import RelativeDensityGate, load_direct_gate  # noqa: E402
from cosmo_sr.reward.reward import RewardModel  # noqa: E402
from cosmo_sr.reward.soft_structure import SoftStructureConfig  # noqa: E402
from cosmo_sr.reward.tiles import TileGrid  # noqa: E402
from cosmo_sr.reward.torch_reward import TorchRewardModel  # noqa: E402
from cosmo_sr.train.sr2_finetune_data import SR2TileGeometry  # noqa: E402
from cosmo_sr.train.train_sr2_direct import DirectFinetuneConfig  # noqa: E402
from cosmo_sr.utils.config import load_config  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reward" / "sr2_direct_finetune.yaml"

__all__ = [
    "DEFAULT_CONFIG", "PROJECT_ROOT", "actor_config_of", "add_direct_args",
    "append_jsonl", "banner", "bins_of", "boxes_of", "code_commit",
    "direct_root", "file_sha", "geometry_of", "gate_of", "load_direct_config",
    "load_lr", "load_hr", "load_reward_models", "manifest_row", "read_jsonl",
    "soft_config_of", "tile_grid_of", "write_json",
]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def add_direct_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="configs/reward/sr2_direct_finetune.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="dotted config override, repeatable")
    ap.add_argument("--run-name", default="direct_a",
                    help="names the artifact directory under the direct root")
    return ap


def load_direct_config(args) -> Dict:
    from cosmo_sr.utils.config import apply_overrides

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, getattr(args, "overrides", None))
    ref = cfg.get("reward_config", "configs/reward/reward.yaml")
    p = Path(ref)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    cfg["_reward"] = yaml.safe_load(p.read_text())
    cfg["_reward_config_path"] = str(p)
    return cfg


def boxes_of(cfg: Mapping, which: str) -> List[str]:
    """One of the direct line's box lists, by name.

    Deliberately its own list rather than a view onto ``reward.yaml``'s splits:
    the direct line uses set0-7 to *fit the proxy* and set8-11 to *gate* it,
    which is a different role assignment from the residual line's train/val even
    though the boxes coincide. Naming them by role makes a leak legible.
    """
    s = dict(cfg.get("split", {}))
    key = {
        "proxy_fit": "proxy_fit_boxes",
        "proxy_gate": "proxy_gate_boxes",
        "actor_train": "actor_train_boxes",
        "actor_eval": "actor_eval_boxes",
        "preflight": "preflight_boxes",
        "sealed": "sealed_boxes",
    }[which]
    return [str(b) for b in s.get(key, [])]


def assert_not_sealed(cfg: Mapping, boxes: Sequence[str]) -> None:
    """set13-15 are closed. set12 is a preflight box and is flagged, not blocked."""
    sealed = set(boxes_of(cfg, "sealed"))
    bad = [b for b in boxes if b in sealed]
    if bad:
        raise SystemExit(
            f"refusing to touch sealed boxes {bad}: they are reserved for the "
            "final comparison and opening them ends their value as evidence"
        )
    pre = set(boxes_of(cfg, "preflight"))
    hit = [b for b in boxes if b in pre]
    if hit:
        print(f"NOTE: {hit} is development-contaminated (the SR2 subhalo study "
              "and the reward sanity check both ran on it). Usable as a late "
              "preflight, never as final evidence.", flush=True)


def geometry_of(cfg: Mapping) -> SR2TileGeometry:
    g = dict(cfg.get("geometry", {}))
    m = dict(cfg.get("model", {}))
    return SR2TileGeometry(
        ng_lr=int(g.get("ng_lr", 64)),
        nsplit=int(g.get("nsplit", 8)),
        pad=int(g.get("pad", 3)),
        scale_factor=int(m.get("scale_factor", g.get("scale_factor", 8))),
    )


def tile_grid_of(cfg: Mapping) -> TileGrid:
    g = dict(cfg.get("geometry", {}))
    return TileGrid(ng_hr=int(g.get("ng_hr", 512)),
                    tile_hr=int(g.get("tile_hr", 64)),
                    boxsize_mpc_h=float(g.get("boxsize_mpc_h", 100.0)))


def soft_config_of(cfg: Mapping) -> SoftStructureConfig:
    s = dict(cfg.get("soft_structure", {}))
    g = dict(cfg.get("geometry", {}))
    return SoftStructureConfig(
        compact_delta=float(s.get("compact_delta", 100.0)),
        envelope_lo=float(s.get("envelope_lo", 10.0)),
        envelope_hi=float(s.get("envelope_hi", 100.0)),
        tau_log=float(s.get("tau_log", 0.35)),
        peak_scales=tuple(int(x) for x in s.get("peak_scales", (1, 2, 4))),
        peak_deltas=tuple(float(x) for x in s.get("peak_deltas", (100.0, 30.0, 10.0))),
        contrast_margin=float(s.get("contrast_margin", 1.5)),
        n_power_bands=int(s.get("n_power_bands", 4)),
        region_fraction=float(s.get("region_fraction", 0.5)),
        grid_mult=int(s.get("grid_mult", 1)),
        boxsize_mpc_h=float(g.get("boxsize_mpc_h", 100.0)),
        ng_hr=int(g.get("ng_hr", 512)),
        dis_norm_kpc_h=float(g.get("dis_norm_kpc_h", 6000.0)),
    )


def actor_config_of(cfg: Mapping) -> DirectFinetuneConfig:
    a = dict(cfg.get("actor", {}))
    # batch_size / noise_draws belong to the loader, not to the step.
    a.pop("batch_size", None)
    a.pop("noise_draws", None)
    return DirectFinetuneConfig.from_dict(a)


def gate_of(cfg: Mapping) -> RelativeDensityGate:
    return load_direct_gate(cfg.get("gates", {}))


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def direct_root(*parts: str, create: bool = False) -> Path:
    """Root of every artifact this line writes. Never the repository, never home."""
    env = os.environ.get("DMSR_SR2_DIRECT_ROOT")
    root = Path(env) if env else paths.subdir("sr2_direct")
    p = root.joinpath(*parts)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def run_dir(run_name: str, *parts: str, create: bool = False) -> Path:
    return direct_root("runs", str(run_name), *parts, create=create)


def data_root(cfg: Mapping) -> Path:
    return Path(cfg["_reward"]["data"]["root"])


def load_lr(cfg: Mapping, box: str) -> np.ndarray:
    return np.load(data_root(cfg) / "lr" / f"{box}.npy", mmap_mode="r")


def load_hr(cfg: Mapping, box: str) -> np.ndarray:
    return np.load(data_root(cfg) / "hr" / f"{box}.npy", mmap_mode="r")


def model_path_of(cfg: Mapping) -> Path:
    p = Path(cfg.get("model", {}).get("path", "external/SRS-map2map/SRmodel/G_z0.pt"))
    return p if p.is_absolute() else PROJECT_ROOT / p


# --------------------------------------------------------------------------- #
# The reward
# --------------------------------------------------------------------------- #
def load_reward_models(cfg: Mapping, path: Optional[str | Path] = None):
    """``(numpy RewardModel, TorchRewardModel)`` from the fitted JSON.

    Both are returned because both are needed and they must be the same object:
    the gates score with the NumPy one and the actor differentiates the torch
    one, and a run where those disagreed would be unreadable.
    """
    p = Path(path) if path else paths.subdir("reward_model") / "reward_model.json"
    if not p.is_file():
        raise SystemExit(
            f"no fitted reward model at {p}; run scripts/reward/fit_reward_model.py "
            "(scripts/slurm/fit_reward_model_cpu.sbatch) first"
        )
    model = RewardModel.from_dict(json.loads(p.read_text()))
    return model, TorchRewardModel.from_numpy(model)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def file_sha(path: str | os.PathLike, digest: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:digest]


def array_sha(a: np.ndarray, digest: int = 8) -> str:
    x = np.ascontiguousarray(np.asarray(a), dtype=np.float32)
    return hashlib.blake2b(x.tobytes(), digest_size=digest).hexdigest()


def code_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=20)
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=False, timeout=20).stdout.strip()
        return f"{rev}{'-dirty' if dirty else ''}" if rev else "unknown"
    except Exception:                       # pragma: no cover - git absent
        return "unknown"


def manifest_row(**kw) -> Dict:
    """Every provenance field the plan requires, with none silently omitted.

    A candidate whose field is on disk but whose provenance is not is unusable:
    six months from now "which weights produced this" is unanswerable from the
    array alone, and a proxy fitted across a mixture of provenances is a proxy
    for nothing in particular.
    """
    required = ("box", "source", "field_path", "model_sha", "lr_sha", "seed")
    missing = [k for k in required if kw.get(k) in (None, "")]
    if missing:
        raise ValueError(f"manifest row is missing required provenance: {missing}")
    row = {"code_commit": code_commit(), "catalog_path": "", "tile_summaries_path": ""}
    row.update(kw)
    return row
