#!/usr/bin/env python
"""Does the SR/HR SUBFIND ratio depend on SUBFIND's own resolution settings?

SUBFIND smooths density over DesNumNgb neighbours (64) and judges where a
subhalo ends from DesLinkNgb neighbours (20, also its minimum size). The bump
census finds peaks with 32 neighbours, and SR's bumps sit closer together than
HR's (median nearest-neighbour 62-67 vs 93 kpc/h at 0.5-1 R_vir), so a 64-neighbour
density may merge an SR lump's sub-peaks into one candidate while HR's separate
knots stay separate. Rerunning SUBFIND with finer settings on HR and the SR
fields asks whether that is what holds SR at ~0.2-0.5x HR in the outskirts.

Only ratios at the SAME setting mean anything: the finder moves HR's count too.

Per catalog: SUBFIND subhalos (all, and satellites) around the same 60 massive
region hosts as the placement forensics, per r/R_vir band and per size class,
with the DesNumNgb / DesLinkNgb GADGET-4 actually used (param.txt-usedvalues).

    python scripts/features/subfind_setting_compare.py \\
        --runs "default:hr=<npz>,coarse_soft=<npz>;fine:hr=<npz>,coarse_soft=<npz>" \\
        --hr-catalog <halos.ascii> --region-export <export.json> --out <json>

Catalogs only; seconds.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from host_phase_profile import select_region_hosts  # noqa: E402
from peak_supply import nearest_host_x  # noqa: E402
from placement_forensics import read_rockstar  # noqa: E402

X_BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 3.0))
SIZES = ((10, 20), (20, 50), (50, 200), (200, 10 ** 9))


def used_setting(npz: Path) -> dict:
    """DesNumNgb / DesLinkNgb as GADGET-4 parsed them for this catalog."""
    out = {}
    pu = npz.parent / "param.txt-usedvalues"
    if pu.is_file():
        for key in ("DesNumNgb", "DesLinkNgb"):
            m = re.search(rf"^{key}\s+(\S+)", pu.read_text(), re.M)
            if m:
                out[key] = int(float(m.group(1)))
    return out


def census(npz: Path, centres: np.ndarray, rvir: np.ndarray) -> dict:
    z = np.load(npz)
    ln = z["SubhaloLen"]
    pos = z["SubhaloPos"].astype(np.float64)
    sat = z["is_satellite"]
    keep = ln >= SIZES[0][0]
    x = np.full(ln.size, np.inf)
    x[keep] = nearest_host_x(pos[keep], centres, rvir)
    rec = {"setting": used_setting(npz)}
    for lo_p in (10, 20, 50):
        m = ln >= lo_p
        rec[f"all_ge{lo_p}"] = [int(np.sum(m & (x >= a) & (x < b))) for a, b in X_BANDS]
        rec[f"sat_ge{lo_p}"] = [int(np.sum(m & sat & (x >= a) & (x < b))) for a, b in X_BANDS]
    out = (x >= 0.5) & (x < 3.0)
    rec["outskirt_sizes"] = [int(np.sum(out & (ln >= a) & (ln < b))) for a, b in SIZES]
    rec["outskirt_ge20"] = int(np.sum(out & (ln >= 20)))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True,
                    help="'setting:name=npz,name=npz;setting:...' ; 'hr' must be in each")
    ap.add_argument("--hr-catalog", required=True)
    ap.add_argument("--region-export", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    hr = read_rockstar(a.hr_catalog)
    sel = select_region_hosts(hr, a.region_export, 2.0, 10000, 60)
    centres, rvir = hr["pos"][sel], hr["rvir"][sel] / 1000.0
    res = {"x_bands": [list(b) for b in X_BANDS], "sizes": [list(s) for s in SIZES],
           "n_hosts": int(sel.size), "settings": {}}
    for block in a.runs.split(";"):
        setting, items = block.split(":", 1)
        recs = {}
        for item in items.split(","):
            name, path = item.split("=", 1)
            p = Path(path)
            if not p.is_file():
                print(f"[{setting}] {name}: no catalog at {p} -- skipped", flush=True)
                continue
            recs[name] = census(p, centres, rvir)
        if "hr" in recs:
            h = recs["hr"]
            for name, r in recs.items():
                r["outskirt_ge20_over_hr"] = r["outskirt_ge20"] / max(h["outskirt_ge20"], 1)
                r["all_ge20_over_hr"] = [v / hv if hv else None
                                         for v, hv in zip(r["all_ge20"], h["all_ge20"])]
        res["settings"][setting] = recs
        print(f"=== {setting}", flush=True)
        for name, r in recs.items():
            ratio = r.get("outskirt_ge20_over_hr")
            print(f"  {name:16s} {r['setting']}  all>=20p by x {r['all_ge20']}  "
                  f"outskirts 0.5-3 {r['outskirt_ge20']}"
                  + (f" ({ratio:.2f}x HR)" if ratio is not None else "")
                  + f"  sizes {r['outskirt_sizes']}", flush=True)
    write_json(Path(a.out), res)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
