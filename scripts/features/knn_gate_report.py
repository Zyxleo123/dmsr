#!/usr/bin/env python
"""kNN damage report for ONE gated arm, read next to its Rockstar counts.

Reads the JSON ``knn_cdf_split_diagnostic.py --region-json`` wrote for ``hr``,
``frozen`` and the arm (labels fixed by ``gate_knn_cpu.sbatch``), and writes the
three numbers plus advisory flags. It never fails a gate: the Rockstar counts
remain the verdict and this job always exits 0.

The numbers, and what calibrated them
-------------------------------------
Calibrated on set9's 32 gated arms (``knn_gate_crosscheck_set9.json``, jobs
51576/51577, 2026-09-13), same settings as the defaults below. Across those
arms the kNN terms split into two families and rank nothing inside the working
one, so these are thresholds for DAMAGE, not a score to optimise:

``sub50_x_frozen``  class-conditioned kNN loss on HR's 50-200p subhalo particles,
                    as a multiple of frozen's. Working arms (4768-7936 region
                    subhalos) <= 1.13; damaging arms (<= 4019 subhalos, 513-742
                    of frozen's >=200p halos lost) >= 1.45.
``ge200_x_frozen``  the same on HR's >=200p subhalo particles. Working <= 1.34;
                    damaging >= 2.16.
``host_r128``       paired median r_128 on HR's HOST particles vs HR (frozen
                    1.048). Damaging arms > 1.14 (hosts diffused). Working arms
                    0.63-1.07; below ~0.85 is the central over-concentration the
                    host forensics measured (selfhostguard 0.835, abl_nod6 0.63).
``unbound_x_frozen`` class-conditioned loss on unbound particles vs frozen's.
                    Only abl_nod6 exceeds 2 (2.55-2.67).

``DAMAGE`` needs BOTH subhalo terms past their gaps. On set9 that flags all 12
damaging arms and none of the 20 working ones. Other boxes are uncalibrated and
the report says so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUB50_DAMAGE = 1.30
GE200_DAMAGE = 1.75
HOST_DIFFUSED = 1.10
HOST_OVERCONC = 0.85
UNBOUND_CROWD = 2.0
CALIBRATED_BOXES = ("set9",)


def report(knn: dict, arm: str, frozen: str = "frozen", box: str = "") -> dict:
    lf = knn["loss_forms"]
    mem = knn["membership"]
    if arm not in lf or frozen not in lf:
        return {"ok": False, "reason": f"missing {arm!r} or {frozen!r} in the kNN JSON"}

    def ratio(cls):
        a = float(np.mean(lf[arm][f"class_cond[{cls}]"]))
        b = float(np.mean(lf[frozen][f"class_cond[{cls}]"]))
        return a / b if b > 0 else float("nan")

    out = {
        "ok": True, "arm": arm, "box": box,
        "sub50_x_frozen": ratio("sub_50_200"),
        "ge200_x_frozen": ratio("sub_ge200"),
        "unbound_x_frozen": ratio("unbound"),
        "host_r128": float(mem[arm]["classes"]["host"]["paired_ratio_med"][-1]),
        "host_r128_frozen": float(mem[frozen]["classes"]["host"]["paired_ratio_med"][-1]),
        "calibrated": box in CALIBRATED_BOXES,
        "thresholds": {"sub50_damage": SUB50_DAMAGE, "ge200_damage": GE200_DAMAGE,
                       "host_diffused": HOST_DIFFUSED, "host_overconc": HOST_OVERCONC,
                       "unbound_crowd": UNBOUND_CROWD},
    }
    flags = []
    if out["sub50_x_frozen"] > SUB50_DAMAGE and out["ge200_x_frozen"] > GE200_DAMAGE:
        flags.append("DAMAGE")
    if out["host_r128"] > HOST_DIFFUSED:
        flags.append("hosts_diffused")
    if out["host_r128"] < HOST_OVERCONC:
        flags.append("hosts_overconcentrated")
    if out["unbound_x_frozen"] > UNBOUND_CROWD:
        flags.append("unbound_crowded")
    out["flags"] = flags
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knn-json", required=True)
    ap.add_argument("--arm", default="arm")
    ap.add_argument("--box", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = report(json.loads(Path(a.knn_json).read_text()), a.arm, box=a.box)
    Path(a.out).write_text(json.dumps(r, indent=1))
    if not r["ok"]:
        print(f"kNN report skipped: {r['reason']}")
        return 0
    print(f"kNN damage report ({a.box}{'' if r['calibrated'] else ', UNCALIBRATED box'}):")
    print(f"  50-200p subhalo term  {r['sub50_x_frozen']:.2f}x frozen   "
          f"(damage > {SUB50_DAMAGE}; set9 working arms <= 1.13)")
    print(f"  >=200p subhalo term   {r['ge200_x_frozen']:.2f}x frozen   "
          f"(damage > {GE200_DAMAGE}; set9 working arms <= 1.34)")
    print(f"  host r_128 vs HR      {r['host_r128']:.3f}  (frozen {r['host_r128_frozen']:.3f}; "
          f"> {HOST_DIFFUSED} diffused, < {HOST_OVERCONC} over-concentrated)")
    print(f"  unbound term          {r['unbound_x_frozen']:.2f}x frozen   (> {UNBOUND_CROWD} crowded)")
    print(f"  flags: {', '.join(r['flags']) or 'none'}")
    print(f"=== wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
