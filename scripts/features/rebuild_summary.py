"""Reconstruct a killed run's ``summary.json`` from its wandb launch argv.

The trainer writes ``summary.json`` only when it finishes, and the gate's export
stage replays that file's ``config`` through the trainer's parser
(``export_gather_tiles.replay_args``). So a run that was cancelled, preempted or
hit its wall clock cannot be gated at all -- its checkpoints are fine, but the
config that produced them is gone.

It is not gone: wandb records the exact argv in
``wandb/latest-run/files/wandb-metadata.json``. Parsing that argv through the
SAME ``build_parser`` the trainer used reproduces the namespace exactly, which is
what ``config`` holds. Nothing is guessed -- an argv token the current parser no
longer accepts makes this fail loudly rather than quietly defaulting.

The written file is marked ``reconstructed`` and carries the checkpoint's own
``step``/``partial`` flags, so a partial run can never be mistaken for a
completed one when its gate is read.

    python scripts/features/rebuild_summary.py --run-dir <arm dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from finetune_member_gather import build_parser  # noqa: E402

META = "wandb/latest-run/files/wandb-metadata.json"


def load_argv(run_dir: Path, meta: Path | None = None) -> tuple[list, dict]:
    p = meta or (run_dir / META)
    if not p.is_file():
        raise SystemExit(f"no wandb metadata at {p}; the argv cannot be recovered")
    d = json.loads(p.read_text())
    argv = d.get("args")
    if not argv:
        raise SystemExit(f"{p} carries no 'args': nothing to replay")
    prov = {"source": str(p), "started_at": d.get("startedAt"),
            "host": d.get("host"), "program": d.get("program")}
    return list(argv), prov


def checkpoint_state(run_dir: Path, name: str = "tuned.pt") -> dict:
    p = run_dir / name
    if not p.is_file():
        return {"checkpoint": None}
    ck = torch.load(p, map_location="cpu", weights_only=False)
    keep = {k: ck[k] for k in ("step", "partial", "rung", "cond") if k in ck}
    return {"checkpoint": str(p), **keep}


def rebuild(run_dir: Path, meta: Path | None = None,
            checkpoint: str = "tuned.pt") -> dict:
    argv, prov = load_argv(run_dir, meta)
    # parse_args exits non-zero on an unknown option: a renamed flag must stop
    # this, not silently fall back to a default that changes the member sets.
    args = build_parser().parse_args(argv)
    cfg = vars(args)
    ck = checkpoint_state(run_dir, checkpoint)
    step, partial = ck.get("step"), ck.get("partial")
    return {
        "ok": False,
        "reconstructed": True,
        "provenance": prov,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
        "rung": cfg.get("rung"),
        **ck,
        "verdict": (f"RECONSTRUCTED from the launch argv; the run did NOT finish "
                    f"(checkpoint step {step}, partial={partial}, configured steps "
                    f"{cfg.get('steps')}). Gate results from it are NOT comparable "
                    f"with a completed arm's -- compare at a shared step instead."),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=None,
                    help=f"wandb metadata json (default: <run-dir>/{META})")
    ap.add_argument("--checkpoint", default="tuned.pt")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <run-dir>/summary.json")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing summary.json")
    a = ap.parse_args(argv)
    out = a.out or (a.run_dir / "summary.json")
    if out.exists() and not a.force:
        print(f"{out} already exists -- refusing to overwrite (use --force). "
              f"A real summary.json is always better than a reconstructed one.")
        return 0
    s = rebuild(a.run_dir, a.meta, a.checkpoint)
    write_json(out, s)
    print(f"wrote {out}")
    print(f"  step {s.get('step')}  partial {s.get('partial')}  "
          f"configured steps {s['config'].get('steps')}")
    print(f"  {s['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
