"""Shared plumbing for the host-conditioned local-editor scripts.

Deliberately thin, and deliberately separate from ``_common.py``: the two lines
share the frozen SR2 cache and the halo-finding plumbing, and share neither a
config file nor an artifact root. Anything that reaches across (the SR2 base
cache, the catalog bins, the reward model) is an explicit import here rather
than an implicit one at each call site, so the coupling is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward import paths  # noqa: E402
from cosmo_sr.reward.local_editor import search_codec  # noqa: E402
from cosmo_sr.reward.local_reward import load_local_reward_config  # noqa: E402
from cosmo_sr.reward.pipeline import existing_catalog  # noqa: E402
from cosmo_sr.utils.config import apply_overrides, load_config  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reward" / "local_editor.yaml"

# Written into every artifact so a directory listing says which line produced it.
PIPELINE = "local_editor"


def add_local_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="configs/reward/local_editor.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="dotted config override, repeatable")
    return ap


def load_local_config(args) -> Dict:
    return apply_overrides(load_config(args.config), getattr(args, "overrides", None))


def banner(msg: str) -> None:
    print(f"=== {msg} ===", flush=True)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def require_calibrated_constraints(cfg: Dict, *, script: str) -> None:
    """Refuse to emit a reward while the feasibility thresholds are placeholders.

    An uncalibrated filter is worse than no filter: it looks like a measurement
    and is a guess, and every number downstream inherits the guess without
    saying so. The only exempt stage is the audit itself, which measures the
    populations and emits no reward -- it passes ``--i-am-the-audit``.
    """
    if bool(dict(cfg.get("constraints", {})).get("calibrated", False)):
        return
    raise SystemExit(
        f"{script}: configs/reward/local_editor.yaml has "
        "constraints.calibrated: false -- the feasibility thresholds are "
        "placeholders and no reward may be emitted against them.\n"
        "  1. sbatch scripts/slurm/local_editor_audit_cpu.sbatch\n"
        "  2. paste the proposed block from "
        "$DMSR_LOCAL_EDITOR_ROOT/audit/constraints_proposal.json\n"
        "  3. set constraints.calibrated: true\n"
        "Do NOT copy the thresholds from configs/reward/reward.yaml: they were "
        "derived from a whole-field residual and reject localised edits for the "
        "wrong reason."
    )


def assert_no_final_boxes(cfg: Dict, boxes: Sequence[str], *, script: str) -> None:
    """set13/14/15 stay closed until the final comparison."""
    final = set(cfg.get("split", {}).get("final_eval_boxes", []))
    bad = sorted(set(boxes) & final)
    if bad:
        raise SystemExit(
            f"{script}: {bad} are final-eval boxes and must stay untouched "
            "until evaluate_local_editor.py --final. Refusing to continue."
        )


def assert_training_boxes(cfg: Dict, boxes: Sequence[str], *, script: str) -> None:
    train = set(cfg.get("split", {}).get("train_boxes", []))
    bad = sorted(set(boxes) - train)
    if bad:
        raise SystemExit(
            f"{script}: the token library may only be built from training boxes; "
            f"{bad} are not in {sorted(train)}."
        )


# ---------------------------------------------------------------------------
# Artifact layout
# ---------------------------------------------------------------------------


def run_dir(run_name: str, *parts: str, create: bool = False) -> Path:
    return paths.LOCAL_EDITOR("runs", run_name, *parts, create=create)


def hosts_path(run_name: str, box: str) -> Path:
    return run_dir(run_name, "hosts") / f"hosts_{box}.json"


def pool_path(run_name: str, box: str) -> Path:
    return run_dir(run_name, "pools") / f"pools_{box}.npz"


def rows_path(run_name: str, box: str) -> Path:
    return run_dir(run_name, "candidates") / f"rows_{box}.jsonl"


def cem_dir(run_name: str, create: bool = False) -> Path:
    return run_dir(run_name, "cem", create=create)


# ---------------------------------------------------------------------------
# Inputs shared with the residual-diffusion line
# ---------------------------------------------------------------------------


def base_field_path(box: str, base_seed: int = 0) -> Path:
    """The frozen SR2 field for one box, from the shared reward-root cache."""
    cache = paths.SR2_BASE_CACHE()
    hits = sorted(Path(cache).glob(f"{box}_seed{int(base_seed)}_*.npy"))
    if not hits:
        raise SystemExit(
            f"no frozen SR2 cache for {box} seed {base_seed} under {cache}; "
            "produced by scripts/slurm/cache_sr2_base.sbatch (reward line)."
        )
    return hits[0]


def load_base_field(box: str, base_seed: int = 0, *, mmap: bool = True):
    return np.load(base_field_path(box, base_seed),
                   mmap_mode="r" if mmap else None)


def base_catalog(box: str):
    """The frozen SR2 halo catalog, from whichever reward-root job produced it."""
    for root in ("halos", "halos_particles"):
        p = existing_catalog(paths.subdir(root, f"{box}__base__base"), "base")
        if p is not None:
            return load_rockstar_ascii(p)
    raise SystemExit(
        f"no frozen SR2 catalog for {box}. Produced by "
        "scripts/slurm/hr_catalog_summaries_cpu.sbatch with SOURCES=base."
    )


def reward_bins():
    """Catalog bins for the *reported* full-box statistics.

    Taken from the reward line's own YAML rather than duplicated here: the
    full-box numbers this pipeline reports are only comparable to the existing
    ones if the bin edges are literally the same object.
    """
    from cosmo_sr.reward.catalog import load_bins
    ref = load_config(str(PROJECT_ROOT / "configs" / "reward" / "reward.yaml"))
    return load_bins(ref.get("catalog", {}))


def reward_model():
    """The fitted catalog reward model, if the reward line has produced one."""
    from cosmo_sr.reward.reward import RewardModel
    p = paths.AUDITS("tile_decomposition") / "tile_reward_model.json"
    if not p.is_file():
        return None
    return RewardModel.from_dict(json.loads(p.read_text()))


# ---------------------------------------------------------------------------
# Config -> objects
# ---------------------------------------------------------------------------


def codec_for(cfg: Dict, mode: str, *, action_only: bool = False):
    ed = cfg.get("editor", {})
    return search_codec(mode, bounds=dict(ed.get("bounds", {})),
                        action_only=action_only,
                        both_cooling_cap=float(ed.get("both_mode_cooling_cap", 0.92)))


def local_reward_config(cfg: Dict):
    return load_local_reward_config(cfg.get("reward", {}))


def mode_plan(cfg: Dict, n: int) -> List[str]:
    """Deterministic per-candidate mode assignment from the configured weights.

    Largest-remainder allocation, then a fixed interleave, so a round always
    contains the intended proportions exactly -- sampling modes at random would
    make a 28-candidate round's mode balance itself a source of variance.
    """
    w = dict(cfg.get("editor", {}).get("mode_weights", {"disp": 0.4, "both": 0.5, "vel": 0.1}))
    names = [k for k in ("disp", "both", "vel") if float(w.get(k, 0.0)) > 0]
    tot = sum(float(w[k]) for k in names)
    exact = {k: float(w[k]) / tot * int(n) for k in names}
    counts = {k: int(np.floor(exact[k])) for k in names}
    rem = int(n) - sum(counts.values())
    for k in sorted(names, key=lambda k: -(exact[k] - counts[k]))[:rem]:
        counts[k] += 1
    out: List[str] = []
    for k in names:
        out.extend([k] * counts[k])
    return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))
    return p


def append_jsonl(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(obj, sort_keys=True, default=_json_default) + "\n")


def read_jsonl(path) -> List[Dict]:
    out: List[Dict] = []
    p = Path(path)
    if not p.is_file():
        return out
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
