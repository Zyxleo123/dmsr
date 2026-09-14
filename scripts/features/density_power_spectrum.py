#!/usr/bin/env python
"""Density power spectrum P(k)/P_HR(k) for the member-gather arms.

The trainer's ``highk_ratio`` is *displacement*-component power (what
``field_guards._highk_power`` charges). The physically meaningful quantity for a
talk is the **density** power spectrum: displace the particles, CIC-deposit to an
overdensity grid, and take P(k). Subhalos are sub-grid at the 512^3 particle
mesh (cell ~195 kpc/h), so we deposit on a ``grid_mult``x-finer mesh (default 2,
Nyquist k ~= 32 h/Mpc) to resolve the small-scale density the fine-tune restores.

Reads the saved candidate/HR fields (no model, no rockstar), computes one P(k)
per field, and writes a JSON the plotter turns into a band-by-band ratio. GPU if
available (the 1024^3 CIC + FFT is heavy); one field at a time to bound memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cosmo_sr.eval.density import cic_density  # noqa: E402
from cosmo_sr.eval.spectra import power_spectrum  # noqa: E402

BOX_MPC = 100.0
BOX_KPC = 100000.0
NG = 512
CELL_KPC = BOX_KPC / NG        # Lagrangian cell size
DIS_NORM = 6000.0              # z=0, D(z)=1: disnorm undo is exactly x6000


def density_pk(field_path: str, grid_mult: int, device: str):
    """Load a (6,Ng^3) field, CIC-deposit displacements, return (k, Pk)."""
    x = np.load(field_path, mmap_mode="r")
    disp = np.ascontiguousarray(x[0:3], dtype=np.float32)  # (3,Ng,Ng,Ng)
    t = torch.from_numpy(disp).unsqueeze(0).to(device)     # (1,3,Ng,Ng,Ng)
    with torch.no_grad():
        delta = cic_density(t, cellsize=CELL_KPC, dis_norm=DIS_NORM,
                            grid_mult=grid_mult)            # (1,1,gN,gN,gN)
    d = delta.squeeze().float().cpu().numpy()
    del t, delta
    k, pk = power_spectrum(d, boxsize=BOX_MPC)
    return k.tolist(), pk.tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", nargs="+", required=True,
                    help="label=path pairs; HR must be one of them")
    ap.add_argument("--grid-mult", type=int, default=2,
                    help="finer density mesh (2 => 1024^3, Nyquist ~32 h/Mpc)")
    ap.add_argument("--out", required=True, help="output json")
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}, grid_mult {a.grid_mult}", flush=True)

    out = {"box_mpc": BOX_MPC, "grid_mult": a.grid_mult, "arms": {}}
    for item in a.fields:
        label, _, path = item.partition("=")
        if not Path(path).exists():
            print(f"  MISSING {label}: {path}", flush=True)
            continue
        print(f"  {label}: {path}", flush=True)
        k, pk = density_pk(path, a.grid_mult, device)
        out["arms"][label] = {"k": k, "Pk": pk, "path": path}
        print(f"    {len(k)} k-bins, k in [{k[0]:.3f}, {k[-1]:.2f}]", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
