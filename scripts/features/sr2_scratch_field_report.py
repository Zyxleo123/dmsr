#!/usr/bin/env python
"""Score a from-scratch SR2 checkpoint on the yardstick SR2 itself was scored on.

Why this exists
---------------
``train_sr2_scratch.py`` logs ``holdout_base.mse`` and told the reader to watch
it fall. That instruction was wrong, and this job is the correction.

*Measured*, ``runs/sr2_baseline/field_table`` -- the reproduction of the real
SR2 (set14, seed 0):

    field          P/P_HR (hi-k)   T(k)    r(k)
    displacement       0.396       0.624   0.000
    velocity           0.989       0.994  -0.000
    density            0.977       0.988   0.092

**SR2's high-k ``r(k)`` is ~0 on every channel.** It does not reproduce HR's
*realisation* at small scales -- it generates a different one with the right
statistics, which is what the adversary is for. So a pointwise MSE against the
HR realisation is dominated by a component that is unpredictable *by design*,
and a model scoring BETTER than SR2 on MSE is most likely suppressing
small-scale power rather than learning the map better. MSE cannot tell those two
apart. The power ratios can.

What it reports
---------------
For each box, the same three-row table, computed with the **same estimator**
``reproduce_field_table.py`` used (imported, not reimplemented, so the numbers
are comparable to the ones above), for:

* the checkpoint under test, and
* the **frozen SR2 field cached for that same box and seed**.

The frozen row is recomputed here rather than quoted from the set14 table,
because the honest comparison is same boxes, same seed, same estimator. The
set14 numbers stay in the output as ``reference_set14`` for continuity.

Plus a whole-box version of the trainer's own holdout block -- mse, the LR-scale
anchor and ``vel_rms_ratio`` over **every** tile rather than the ~16 the
training loop samples, because at 16 tiles those numbers swing by a factor of
two between evals and cannot be read to three decimals.

It touches nothing the training jobs own: it reads a checkpoint and the cached
frozen box, and writes one JSON and one markdown table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features", PROJECT_ROOT / "scripts" / "sr2"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, write_json  # noqa: E402
from _sr2_direct import geometry_of, load_direct_config, load_hr, load_lr  # noqa: E402
from overfit_host_mse import BOXSIZE, NG_HR, TILE  # noqa: E402

# The reference table's own estimator, imported so the numbers are comparable.
# A second definition of "high-k mean" or "transfer" would make this job's
# output look like the table while measuring something else.
from reproduce_field_table import _highk, _spectra  # noqa: E402

from cosmo_sr.eval.density import cic_density  # noqa: E402
from cosmo_sr.reward.base import find_base_field  # noqa: E402
from cosmo_sr.train.sr2_scratch import (  # noqa: E402
    base_mse, build_scratch_generator, lowk_anchor, tile_hr_target,
)
from cosmo_sr.tts.sampling import super_resolve_srs_seeded  # noqa: E402

#: The published reproduction of real SR2 (set14, seed 0). Kept for continuity;
#: the per-box frozen rows this job computes are the comparison that matters.
REFERENCE_SET14 = {
    "displacement": {"ratio_highk": 0.396, "transfer_highk": 0.624, "rk_highk": 0.000},
    "velocity": {"ratio_highk": 0.989, "transfer_highk": 0.994, "rk_highk": -0.000},
    "density": {"ratio_highk": 0.977, "transfer_highk": 0.988, "rk_highk": 0.092},
}

FIELDS = (("displacement", 0), ("velocity", 3))


def field_table(sr: np.ndarray, hr: np.ndarray, *, cell_kpc_h: float,
                dis_norm_kpc_h: float, device) -> Dict[str, Dict[str, float]]:
    """The three-row table for one full box, in ``reproduce_field_table``'s form."""
    out: Dict[str, Dict[str, float]] = {}
    for name, ch in FIELDS:
        sp = _spectra(sr[ch], hr[ch])
        out[name] = {
            "ratio_highk": _highk(sp["ratio"]),
            "transfer_highk": _highk(sp["transfer"]),
            "rk_highk": _highk(sp["rk"]),
        }
    d_sr = cic_density(torch.as_tensor(np.ascontiguousarray(sr[0:3]))[None].to(device),
                       cell_kpc_h, dis_norm_kpc_h)[0, 0].cpu().numpy()
    d_hr = cic_density(torch.as_tensor(np.ascontiguousarray(hr[0:3]))[None].to(device),
                       cell_kpc_h, dis_norm_kpc_h)[0, 0].cpu().numpy()
    sp = _spectra(d_sr, d_hr)
    out["density"] = {
        "ratio_highk": _highk(sp["ratio"]),
        "transfer_highk": _highk(sp["transfer"]),
        "rk_highk": _highk(sp["rk"]),
        "sigma_ratio": float(d_sr.std() / max(d_hr.std(), 1e-12)),
    }
    return out


@torch.no_grad()
def whole_box_holdout(sr: np.ndarray, hr: np.ndarray, *, scale_factor: int,
                      device, chunk: int = 8, ng_hr: int = NG_HR,
                      tile: int = TILE) -> Dict[str, float]:
    """The trainer's holdout block, over EVERY tile instead of ~16.

    The training loop averages 4 batches of 4. At that sample size the numbers
    swing by a factor of two between evals, which is how a plateau got reported
    that was not there.
    """
    n = (int(ng_hr) // int(tile)) ** 3
    kw = dict(ng_hr=int(ng_hr), tile=int(tile))
    mses, lows, ldis, lvel, vels = [], [], [], [], []
    for i in range(0, n, chunk):
        ids = range(i, min(i + chunk, n))
        s = torch.from_numpy(np.stack([tile_hr_target(sr, t, **kw) for t in ids])).to(device)
        h = torch.from_numpy(np.stack([tile_hr_target(hr, t, **kw) for t in ids])).to(device)
        mses.append(float(base_mse(s, h)))
        g = lowk_anchor(s, h, int(scale_factor))
        lows.append(float(g["low"])); ldis.append(float(g["low_dis"]))
        lvel.append(float(g["low_vel"])); vels.append(float(g["vel_rms_ratio"]))
    return {"mse": float(np.mean(mses)), "low_k": float(np.mean(lows)),
            "low_dis": float(np.mean(ldis)), "low_vel": float(np.mean(lvel)),
            "vel_rms_ratio": float(np.mean(vels)), "n_tiles": int(n)}


def load_scratch_generator(ckpt: Path, *, scale_factor: int, device):
    """The generator from a `train_sr2_scratch.py` checkpoint.

    Refuses anything else: scoring a fine-tune checkpoint here and filing the
    result under a from-scratch arm is the failure this whole line keeps paying
    for.
    """
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if not blob.get("scratch"):
        raise SystemExit(
            f"{ckpt} has no 'scratch' flag, so it was not written by "
            "train_sr2_scratch.py. Scoring it here would file a fine-tune "
            "result under a from-scratch arm's name.")
    g = build_scratch_generator(scale_factor=int(scale_factor), device=device)
    g.load_state_dict(blob["model"])
    g.to(device).eval()
    return g, {"step": int(blob.get("step", 0)),
               "density_transform": blob.get("density_transform"),
               "has_critic_norm": blob.get("critic_norm") is not None}


def markdown(rows: Dict[str, Dict], boxes: List[str]) -> str:
    out = ["# From-scratch SR2 vs frozen SR2, same boxes, same estimator", ""]
    for box in boxes:
        r = rows.get(box)
        if not r:
            continue
        out += [f"## {box}", "",
                "| field | P/P_HR (hi-k) | T(k) | r(k) | | frozen P/P_HR | frozen T(k) | frozen r(k) |",
                "|---|---:|---:|---:|---|---:|---:|---:|"]
        for f in ("displacement", "velocity", "density"):
            c, z = r["candidate"][f], r["frozen"][f]
            out.append(
                f"| {f} | {c['ratio_highk']:.3f} | {c['transfer_highk']:.3f} | "
                f"{c['rk_highk']:.3f} | | {z['ratio_highk']:.3f} | "
                f"{z['transfer_highk']:.3f} | {z['rk_highk']:.3f} |")
        ch, zh = r["candidate_holdout"], r["frozen_holdout"]
        out += ["",
                "| whole-box holdout | candidate | frozen SR2 |", "|---|---:|---:|",
                f"| mse | {ch['mse']:.4f} | {zh['mse']:.4f} |",
                f"| low_k | {ch['low_k']:.4f} | {zh['low_k']:.4f} |",
                f"| vel_rms_ratio | {ch['vel_rms_ratio']:.3f} | {zh['vel_rms_ratio']:.3f} |",
                ""]
    out += ["## How to read this", "",
            "`r(k)` at high k is ~0 for real SR2 on every channel: it generates a",
            "different realisation with the right statistics, not HR's own. So a",
            "**higher** r(k) or a **lower** mse than the frozen column is not a win --",
            "it is the signature of suppressed small-scale power (mean collapse).",
            "The number to match is `P/P_HR`, and the one with no loss term behind",
            "it on this line is velocity.", ""]
    return "\n".join(out)


def run(args) -> Dict:
    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint at {ckpt}")

    g, prov = load_scratch_generator(ckpt, scale_factor=geom.scale_factor,
                                     device=device)
    banner(f"field report: {ckpt.name} (step {prov['step']})")
    print(f"  provenance: {prov}", flush=True)

    boxes = [b for b in str(args.boxes).replace(",", " ").split() if b]
    cell = BOXSIZE * 1e3 / float(NG_HR)
    dis_norm = float(cfg.get("geometry", {}).get("dis_norm_kpc_h", 6000.0))
    rows: Dict[str, Dict] = {}

    for box in boxes:
        t0 = time.time()
        frozen_path = find_base_field(box, seed=int(args.seed))
        if frozen_path is None:
            print(f"  {box}: SKIPPED -- no cached frozen SR2 box for seed "
                  f"{args.seed}. Build it with scripts/reward/cache_sr2_base.py.",
                  flush=True)
            continue
        hr = np.ascontiguousarray(load_hr(cfg, box), dtype=np.float32)
        lr = np.asarray(load_lr(cfg, box), dtype=np.float32)
        sr = super_resolve_srs_seeded(
            g, lr, int(args.seed), scale_factor=geom.scale_factor,
            nsplit=geom.nsplit, pad=geom.pad, device=device)
        frozen = np.asarray(np.load(str(frozen_path), mmap_mode="r"), dtype=np.float32)
        rows[box] = {
            "candidate": field_table(sr, hr, cell_kpc_h=cell,
                                     dis_norm_kpc_h=dis_norm, device=device),
            "frozen": field_table(frozen, hr, cell_kpc_h=cell,
                                  dis_norm_kpc_h=dis_norm, device=device),
            "candidate_holdout": whole_box_holdout(
                sr, hr, scale_factor=geom.scale_factor, device=device),
            "frozen_holdout": whole_box_holdout(
                frozen, hr, scale_factor=geom.scale_factor, device=device),
            "frozen_field": str(frozen_path),
        }
        c, z = rows[box]["candidate"], rows[box]["frozen"]
        print(f"  {box} [{time.time() - t0:.0f}s]  "
              f"dens P/P {c['density']['ratio_highk']:.3f} (frozen "
              f"{z['density']['ratio_highk']:.3f})  vel P/P "
              f"{c['velocity']['ratio_highk']:.3f} (frozen "
              f"{z['velocity']['ratio_highk']:.3f})  r(k) dens "
              f"{c['density']['rk_highk']:.3f} (frozen "
              f"{z['density']['rk_highk']:.3f})", flush=True)
        del sr, frozen, hr, lr

    if not rows:
        print(">>> no box produced a row; nothing written.", flush=True)
        return {"ok": True, "rows": 0}

    out_dir = Path(args.out) if args.out else ckpt.parent / "field_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ckpt.stem
    summary = {"checkpoint": str(ckpt), "step": prov["step"],
               "provenance": prov, "seed": int(args.seed),
               "boxes": list(rows), "per_box": rows,
               "reference_set14": REFERENCE_SET14}
    write_json(out_dir / f"field_report_{tag}.json", summary)
    (out_dir / f"field_report_{tag}.md").write_text(markdown(rows, list(rows)))
    print(f"\nwrote {out_dir}/field_report_{tag}.{{json,md}}", flush=True)
    print(markdown(rows, list(rows)), flush=True)
    return {"ok": True, **summary}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/reward/sr2_direct_finetune.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--ckpt", required=True,
                    help="a train_sr2_scratch.py checkpoint (latest.pt or "
                         "tuned_stepN.pt). A snapshot is preferred: latest.pt "
                         "is rewritten while the job runs.")
    ap.add_argument("--boxes", default="set9 set10",
                    help="held-out boxes. Each needs a cached frozen SR2 field "
                         "at --seed, which is what the comparison is against.")
    ap.add_argument("--seed", type=int, default=0,
                    help="noise realisation; 0 is the one the frozen boxes are "
                         "cached at, so it is the only seed with a reference.")
    ap.add_argument("--out", default="",
                    help="output directory (default: <ckpt dir>/field_report)")
    ap.add_argument("--device", default="")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return 0 if run(args).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
