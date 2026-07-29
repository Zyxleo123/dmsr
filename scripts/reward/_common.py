"""Shared plumbing for the reward-guided residual diffusion scripts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosmo_sr.reward import paths  # noqa: E402
from cosmo_sr.reward.catalog import CatalogBins, load_bins  # noqa: E402
from cosmo_sr.reward.constraints import ConstraintSet, load_constraints  # noqa: E402
from cosmo_sr.reward.geometry import ChunkGrid  # noqa: E402
from cosmo_sr.utils.config import load_config  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reward" / "reward.yaml"


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="reward config YAML (configs/reward/reward.yaml)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="dotted config override, repeatable")
    return ap


def load_reward_config(args) -> Dict:
    from cosmo_sr.utils.config import apply_overrides

    cfg = load_config(args.config)
    return apply_overrides(cfg, getattr(args, "overrides", None))


def load_freeze(cfg: Dict) -> Dict:
    p = Path(cfg.get("freeze", "configs/sr2_baseline/freeze.yaml"))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return yaml.safe_load(p.read_text())


def chunk_grid(cfg: Dict) -> ChunkGrid:
    d = cfg.get("data", {})
    g = cfg.get("geometry", {})
    return ChunkGrid(
        ng_hr=int(d.get("ng_hr", 512)),
        chunk_hr=int(g.get("chunk_hr", 128)),
        boxsize_mpc_h=float(d.get("boxsize_mpc_h", 100.0)),
    )


def bins_of(cfg: Dict) -> CatalogBins:
    b = load_bins(cfg.get("catalog", {}))
    g = cfg.get("geometry", {})
    # geometry owns the mask parameters; catalog carries them for provenance
    return CatalogBins(
        sub_mass_edges=b.sub_mass_edges,
        host_mass_edges=b.host_mass_edges,
        min_sub_particles=b.min_sub_particles,
        min_host_particles=b.min_host_particles,
        min_purity=float(g.get("min_purity", b.min_purity)),
        radius_mult=float(g.get("radius_mult", b.radius_mult)),
        abundance_transform=b.abundance_transform,
        occupation_transform=b.occupation_transform,
        abundance_floor_halos=b.abundance_floor_halos,
        occupation_floor=b.occupation_floor,
    )


def constraints_of(cfg: Dict) -> ConstraintSet:
    return load_constraints(cfg.get("constraints", {}))


def split_boxes(cfg: Dict, which: str) -> List[str]:
    """Boxes of one split.

    ``dev`` and ``final`` subdivide the frozen SR2 ``test_boxes``. ``set12`` has
    been inspected repeatedly (the whole subhalo study and the reward sanity
    check ran on it), so it is development data now, not a held-out box; the
    boxes nothing has ever looked at are reserved for the final numbers.
    """
    s = cfg.get("split", {})
    key = {
        "train": "train_boxes", "val": "val_boxes", "test": "test_boxes",
        "dev": "dev_boxes", "final": "final_eval_boxes",
    }[which]
    return list(s.get(key, []))


def assert_no_leak(cfg: Dict, boxes: Sequence[str], allowed: Sequence[str]) -> None:
    """Raise if any requested box belongs to a split other than ``allowed``."""
    allowed_boxes = set()
    for w in allowed:
        allowed_boxes.update(split_boxes(cfg, w))
    bad = [b for b in boxes if b not in allowed_boxes]
    if bad:
        raise SystemExit(
            f"split leak: {bad} are not in {list(allowed)} "
            f"({sorted(allowed_boxes)}); refusing to continue"
        )


def lr_path(cfg: Dict, box: str) -> Path:
    return Path(cfg["data"]["root"]) / "lr" / f"{box}.npy"


def hr_path(cfg: Dict, box: str) -> Path:
    return Path(cfg["data"]["root"]) / "hr" / f"{box}.npy"


def parse_boxes(arg: Optional[str], cfg: Dict, default_split: str) -> List[str]:
    if arg:
        return [b.strip() for b in str(arg).split(",") if b.strip()]
    return split_boxes(cfg, default_split)


def write_json(path: str | Path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))
    return p


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def append_jsonl(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(obj, sort_keys=True, default=_json_default) + "\n")


def read_jsonl(path: str | Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def select_device(name: Optional[str] = None):
    import torch

    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def banner(msg: str) -> None:
    print(f"=== {msg} ===", flush=True)
