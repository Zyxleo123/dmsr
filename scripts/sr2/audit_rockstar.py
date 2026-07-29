#!/usr/bin/env python
"""Audit Rockstar settings for HR vs SR catalogs (identical finding)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

KEYS = [
    "PARTICLE_MASS", "FORCE_RES", "FORCE_RES_PHYS_MAX", "BOX_SIZE", "PERIODIC",
    "MIN_HALO_OUTPUT_SIZE", "MIN_HALO_PARTICLES", "UNBOUND_THRESHOLD",
    "Om", "Ol", "h0", "SCALE_NOW", "FULL_PARTICLE_CHUNKS",
    "RESCALE_PARTICLE_MASS", "PARALLEL_IO", "NUM_WRITERS",
]


def _parse_cfg(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def _parse_ascii_header(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "Particle mass" in line:
                m = re.search(r"([0-9.eE+-]+)\s*Msun", line)
                if m:
                    out["header_particle_mass"] = m.group(1)
            if "Box size" in line:
                m = re.search(r"([0-9.]+)\s*Mpc", line)
                if m:
                    out["header_box_size"] = m.group(1)
            if line.startswith("#Om"):
                out["header_cosmo"] = line.strip()
            if "Total particles" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    out["header_npart"] = m.group(1)
            if "Force resolution" in line:
                m = re.search(r"([0-9.eE+-]+)", line)
                if m:
                    out["header_force_res"] = m.group(1)
    return out


def _find_ascii(d: Path) -> Optional[Path]:
    for pat in ("halos*.ascii", "halos*.list"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    return None


def audit_one(tag: str, rockstar_dir: Path) -> Dict:
    cfg = _parse_cfg(rockstar_dir / "rockstar.cfg")
    ascii_p = _find_ascii(rockstar_dir)
    hdr = _parse_ascii_header(ascii_p) if ascii_p else {}
    return {
        "tag": tag,
        "dir": str(rockstar_dir),
        "ascii": str(ascii_p) if ascii_p else None,
        "cfg": {k: cfg.get(k) for k in KEYS},
        "header": hdr,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-halos", required=True,
                    help=".../stage1/halos/<box> directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.stage1_halos)
    reports: List[Dict] = []
    hr = root / "hr" / "hr_rockstar"
    if hr.is_dir():
        reports.append(audit_one("hr", hr))
    for d in sorted(root.glob("sr_seed*/")):
        seed = d.name.replace("sr_seed", "")
        cand = list(d.glob("sr*_rockstar"))
        if cand:
            reports.append(audit_one(f"sr_seed{seed}", cand[0]))

    # Compare each SR to HR
    hr_rep = next((r for r in reports if r["tag"] == "hr"), None)
    diffs = []
    if hr_rep:
        for r in reports:
            if r["tag"] == "hr":
                continue
            for k in KEYS:
                a, b = hr_rep["cfg"].get(k), r["cfg"].get(k)
                if a != b:
                    diffs.append({"sr": r["tag"], "key": k, "hr": a, "sr_val": b})
            for k in ("header_particle_mass", "header_box_size", "header_npart"):
                a, b = hr_rep["header"].get(k), r["header"].get(k)
                if a != b:
                    diffs.append({"sr": r["tag"], "key": k, "hr": a, "sr_val": b})

    # Freeze reference
    freeze_cfg = _parse_cfg(ROOT / "configs/sr2_baseline/rockstar.cfg")
    freeze_diffs = []
    if hr_rep:
        for k in ("FORCE_RES", "BOX_SIZE", "MIN_HALO_OUTPUT_SIZE", "Om", "Ol", "h0",
                  "UNBOUND_THRESHOLD", "PERIODIC"):
            # PERIODIC in dumped cfg may reflect runtime; compare after patch expect 1
            fv = freeze_cfg.get(k)
            hv = hr_rep["cfg"].get(k)
            if fv is not None and hv is not None and float(fv) != float(hv) and k != "PERIODIC":
                freeze_diffs.append({"key": k, "freeze": fv, "hr_runtime": hv})
            if k == "PERIODIC" and hv != "1":
                freeze_diffs.append({
                    "key": "PERIODIC", "freeze": "1", "hr_runtime": hv,
                    "note": "upstream serial mode forced PERIODIC=0; patch required",
                })

    out = {
        "box_dir": str(root),
        "n_catalogs": len(reports),
        "hr_vs_sr_diffs": diffs,
        "freeze_vs_hr": freeze_diffs,
        "identical_hr_sr": len(diffs) == 0,
        "reports": reports,
    }
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    if diffs:
        raise SystemExit(f"FAIL: {len(diffs)} HR↔SR Rockstar setting mismatches")
    print("PASS: HR and SR Rockstar settings identical")


if __name__ == "__main__":
    main()
