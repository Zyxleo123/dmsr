#!/usr/bin/env python
"""Freeze the reproduced SR2 baseline into runs/sr2_baseline/ (Stage 0).

Writes a self-contained manifest that later experiments must not silently alter:
checkpoint hashes, freeze.yaml copy, inference settings, Rockstar config, split,
and a pointer to the field-level table command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", default=str(ROOT / "configs/sr2_baseline/freeze.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs/sr2_baseline"))
    args = ap.parse_args()

    freeze_path = Path(args.freeze)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(freeze_path.read_text())

    model = ROOT / cfg["model"]["path"]
    rockstar_bin = ROOT / "external/rockstar/rockstar"
    rockstar_cfg = ROOT / cfg["rockstar"]["config"]

    # Copy immutable configs into the freeze dir.
    shutil.copy2(freeze_path, out / "freeze.yaml")
    shutil.copy2(rockstar_cfg, out / "rockstar.cfg")

    env = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "freeze_version": cfg["freeze_version"],
        "redshift": cfg["redshift"],
        "model_path": str(model.resolve()),
        "model_sha256": _sha256(model),
        "model_z2_unused": str((ROOT / cfg["model"]["path_z2_unused"]).resolve()),
        "model_z2_sha256": _sha256(ROOT / cfg["model"]["path_z2_unused"]),
        "externals": cfg["externals"],
        "rockstar_binary": str(rockstar_bin.resolve()),
        "rockstar_binary_exists": rockstar_bin.is_file(),
        "rockstar_cfg": str((out / "rockstar.cfg").resolve()),
        "data": cfg["data"],
        "cosmology_sim": cfg["cosmology_sim"],
        "split": cfg["split"],
        "inference": cfg["inference"],
        "rockstar": cfg["rockstar"],
        "field_table_command": (
            f"python scripts/sr2/reproduce_field_table.py "
            f"--freeze {out / 'freeze.yaml'} --out {out / 'field_table'}"
        ),
        "git": {},
    }
    try:
        env["git"]["cosmo_sr_project"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        env["git"]["cosmo_sr_project"] = None
    for name, rel in (("rockstar", "external/rockstar"),
                      ("SRS-map2map", "external/SRS-map2map"),
                      ("map2map", "external/map2map")):
        try:
            env["git"][name] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT / rel, text=True
            ).strip()
        except Exception:
            env["git"][name] = None

    with open(out / "manifest.json", "w") as fh:
        json.dump(env, fh, indent=2)

    # One-command wrapper
    cmd = out / "reproduce_field_table.sh"
    cmd.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'cd "{ROOT}"\n'
        "source /zfsauton/scratch/yixiz/miniconda3/etc/profile.d/conda.sh\n"
        "conda activate pjm\n"
        f'python scripts/sr2/reproduce_field_table.py '
        f'--freeze "{out / "freeze.yaml"}" --out "{out / "field_table"}" "$@"\n'
    )
    cmd.chmod(0o755)

    print(f"Froze SR2 baseline -> {out}")
    print(f"  model sha256: {env['model_sha256'][:16]}...")
    print(f"  rockstar:     {env['rockstar_binary_exists']}")
    print(f"  reproduce:    {cmd}")


if __name__ == "__main__":
    main()
