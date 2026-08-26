#!/usr/bin/env python
"""Decompose the member-gather high-k guard: which ``k``, and is it structure?

``all_blocks_self`` finished with held-out displacement power **1.70x HR**
(worst host 3.87x) while its train pool sat at 0.73x -- and it is the arm that
won the held-out Rockstar gate (``gather-holdout-rockstar-gate``: 20 -> 366
subhalos against HR's 369). So the excess cannot simply be regularised away
without asking what it is, because whatever built those 366 subhalos also raised
this number.

The guard reports ONE scalar,

    P_cand(|k| >= k_split) / P_HR(|k| >= k_split),   k_split = 4 h/Mpc

as ``sel.mean()`` over every mode passing the mask. On a 64^3 tile at
``dx = 0.1953`` Mpc/h the fundamental is 0.503 and Nyquist 16.08 h/Mpc, so
``k >= 4`` admits **99.2% of all modes** and the unweighted mean over them is
dominated by the outermost shells -- the mode count grows as ``k^2`` and the
cube's corners reach ``sqrt(3) * k_Ny = 27.9``. A scalar built that way cannot
distinguish "built subhalos at k ~ 6-15" from "rang the grid at Nyquist", and it
puts essentially all of its gradient on the latter.

This script separates them, on the tiles the gate already wrote to disk:

1. **Reproduces the guard scalar** per host in numpy, as a self-check against
   the trainer's ``highk_ratio`` -- a decomposition that does not add back up to
   the number being explained is explaining something else.
2. **Binned** ``P_cand(k) / P_HR(k)`` and ``P_frozen(k) / P_HR(k)``, so the
   excess gets a *scale*.
3. **Cross-correlation** ``r(k)`` against HR. Correlated excess is structure in
   the right places with the wrong amplitude; uncorrelated excess is noise. The
   guard charges both identically.
4. **Per-bin share of the guard scalar**, i.e. which bins actually move the
   number the loss sees.
5. **Where the excess lives in space**: a high-pass map ``|Psi_>k|^2`` per voxel,
   its concentration (what volume fraction holds half the power) and its
   correlation with HR's. Grid-scale ringing is spatially flat and uncorrelated;
   collapsed substructure is concentrated and lands where HR's does.
6. ``k_split`` sweep, because if the excess is above the split the guard is
   measuring mostly what we do not care about.

Reads only ``holdout_<box>/tiles.npz`` + ``export.json`` -- no model, no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from cosmo_sr.features.cond_spread import hann_window, wavenumbers  # noqa: E402

BOXSIZE = 100.0
NG_HR = 512
DX = BOXSIZE / NG_HR
K_SPLIT = 4.0

DIS = slice(0, 3)
VEL = slice(3, 6)


def _fft(field: np.ndarray, win: np.ndarray) -> np.ndarray:
    """``(T, n, n, n)`` power summed over components, Hann-windowed.

    Matches :func:`cosmo_sr.features.field_guards._highk_power` exactly: the
    same window, the same component sum, and the same omission of the
    ``dx^3 / n^3`` normalisation, which cancels in every ratio taken here.
    """
    f = np.fft.fftn((field * win).astype(np.float32), axes=(-3, -2, -1))
    return (f.real ** 2 + f.imag ** 2).sum(axis=1)


def _cross(a: np.ndarray, b: np.ndarray, win: np.ndarray) -> np.ndarray:
    fa = np.fft.fftn((a * win).astype(np.float32), axes=(-3, -2, -1))
    fb = np.fft.fftn((b * win).astype(np.float32), axes=(-3, -2, -1))
    return np.real(fa * np.conj(fb)).sum(axis=1)


def guard_ratio(pc: np.ndarray, ph: np.ndarray, mask: np.ndarray) -> float:
    """The trainer's scalar, verbatim: a ratio of means over the masked modes."""
    return float(pc[:, mask].mean() / max(ph[:, mask].mean(), 1e-30))


def binned(power: np.ndarray, which: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean power per radial bin, pooled over the tiles of one host."""
    flat = power.reshape(power.shape[0], -1).mean(axis=0)
    out = np.zeros(n_bins, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.int64)
    ok = (which >= 0) & (which < n_bins)
    np.add.at(out, which[ok], flat[ok])
    np.add.at(cnt, which[ok], 1)
    return np.where(cnt > 0, out / np.maximum(cnt, 1), np.nan)


def highpass_map(field: np.ndarray, kmag: np.ndarray, k_split: float
                 ) -> np.ndarray:
    """``|Psi_{k >= k_split}(x)|^2`` per voxel, summed over components.

    No window: this is a real-space localisation question, and windowing would
    taper the answer toward the tile centre -- which is exactly where the host
    sits, so it would manufacture the concentration it is meant to measure.
    """
    f = np.fft.fftn(field.astype(np.float32), axes=(-3, -2, -1))
    f[:, :, kmag < k_split] = 0.0
    return (np.abs(np.fft.ifftn(f, axes=(-3, -2, -1))) ** 2).sum(axis=1)


def half_power_volume(m: np.ndarray) -> float:
    """Volume fraction holding half the high-pass power. Flat noise -> 0.5."""
    v = np.sort(m.reshape(-1))[::-1]
    c = np.cumsum(v, dtype=np.float64)
    if c[-1] <= 0:
        return float("nan")
    return float((np.searchsorted(c, 0.5 * c[-1]) + 1) / v.size)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def analyse_arm(arm_dir: Path, n_bins: int, k_sweep: List[float]) -> Dict:
    # Each npz member re-inflates on EVERY access, and `out` alone is 201 MB;
    # reading them inside the host loop would decompress ~10 GB to read 600 MB.
    z = np.load(arm_dir / "tiles.npz")
    tiles = z["tiles"]
    cand_all, froz_all, hr_all = z["out"], z["frozen"], z["hr"]
    z.close()
    export = json.loads((arm_dir / "export.json").read_text())
    per_host = export["per_host"]
    n = int(cand_all.shape[-1])
    kmag = wavenumbers(n, DX)
    win = hann_window(n).astype(np.float32)
    mask = kmag >= K_SPLIT

    k_f = 2.0 * np.pi / (n * DX)
    k_ny = np.pi / DX
    edges = np.logspace(np.log10(k_f * 0.999), np.log10(k_ny * 1.001), n_bins + 1)
    which = np.digitize(kmag.reshape(-1), edges) - 1
    kcen = np.sqrt(edges[:-1] * edges[1:])
    # Modes per bin that clear the guard's mask. A property of the grid, so it
    # is computed once; `sel.mean()` is an UNWEIGHTED mean over modes, which is
    # exactly why the per-bin share below has to carry this weight.
    passes = kmag.reshape(-1) >= K_SPLIT
    mode_count = np.array([int(((which == i) & passes).sum())
                           for i in range(n_bins)], dtype=np.float64)

    pos = {int(t): i for i, t in enumerate(tiles)}
    hosts: List[Dict] = []
    for h in per_host:
        idx = [pos[int(t)] for t in h["tiles"] if int(t) in pos]
        if len(idx) != len(h["tiles"]):
            continue
        row: Dict[str, object] = {"key": h["key"], "halo_id": h["halo_id"],
                                  "log_mvir": h["log_mvir"]}
        for tag, sl in (("dis", DIS), ("vel", VEL)):
            c = cand_all[idx, sl]
            f = froz_all[idx, sl]
            r = hr_all[idx, sl]
            pc, pf, pr = _fft(c, win), _fft(f, win), _fft(r, win)
            row[f"guard_{tag}"] = guard_ratio(pc, pr, mask)
            row[f"guard_{tag}_frozen"] = guard_ratio(pf, pr, mask)
            row[f"P_cand_{tag}"] = binned(pc, which, n_bins).tolist()
            row[f"P_frozen_{tag}"] = binned(pf, which, n_bins).tolist()
            row[f"P_hr_{tag}"] = binned(pr, which, n_bins).tolist()
            xc = binned(_cross(c, r, win), which, n_bins)
            xf = binned(_cross(f, r, win), which, n_bins)
            bc, bf, br = (binned(p, which, n_bins) for p in (pc, pf, pr))
            with np.errstate(invalid="ignore", divide="ignore"):
                row[f"r_cand_{tag}"] = (xc / np.sqrt(bc * br)).tolist()
                row[f"r_frozen_{tag}"] = (xf / np.sqrt(bf * br)).tolist()
            # Share of the guard's own mean that each bin supplies. Weighted by
            # the bin's MODE COUNT, because `sel.mean()` is an unweighted mean
            # over modes and that is the whole point of this row.
            cnt = mode_count
            row[f"mode_count_{tag}"] = cnt.tolist()
            num = np.nan_to_num(bc) * cnt
            den = np.nan_to_num(br) * cnt
            row[f"share_cand_{tag}"] = (num / max(num.sum(), 1e-30)).tolist()
            row[f"share_hr_{tag}"] = (den / max(den.sum(), 1e-30)).tolist()
            row[f"ksweep_{tag}"] = {
                f"{ks:g}": guard_ratio(pc, pr, kmag >= ks) for ks in k_sweep}
            row[f"ksweep_{tag}_frozen"] = {
                f"{ks:g}": guard_ratio(pf, pr, kmag >= ks) for ks in k_sweep}
            del pc, pf, pr
            if tag == "dis":
                mc = highpass_map(c, kmag, K_SPLIT)
                mf = highpass_map(f, kmag, K_SPLIT)
                mr = highpass_map(r, kmag, K_SPLIT)
                row["hp_halfvol_cand"] = half_power_volume(mc)
                row["hp_halfvol_frozen"] = half_power_volume(mf)
                row["hp_halfvol_hr"] = half_power_volume(mr)
                row["hp_corr_cand_hr"] = corr(mc, mr)
                row["hp_corr_frozen_hr"] = corr(mf, mr)
                row["hp_corr_cand_frozen"] = corr(mc, mf)
                del mc, mf, mr
            del c, f, r
        hosts.append(row)
        print(f"    {h['key']:<18} guard dis "
              f"{row['guard_dis']:.2f}x (frozen {row['guard_dis_frozen']:.2f}) "
              f"vel {row['guard_vel']:.2f}x  "
              f"halfvol {row['hp_halfvol_cand']:.3f} vs HR "
              f"{row['hp_halfvol_hr']:.3f}  "
              f"corr(cand,HR) {row['hp_corr_cand_hr']:.3f}", flush=True)
    del cand_all, froz_all, hr_all
    return {"arm": arm_dir.parent.name, "n_hosts": len(hosts),
            "k_centres": kcen.tolist(), "k_edges": edges.tolist(),
            "k_split": K_SPLIT, "k_nyquist": k_ny, "k_fundamental": k_f,
            "hosts": hosts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward-root", default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    ap.add_argument("--arms", nargs="+",
                    default=["all_blocks_self", "all_blocks_nocentre",
                             "all_blocks_full", "all_blocks_radial"])
    ap.add_argument("--box", default="set9")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--k-sweep", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    ap.add_argument("--out", required=True, help="output stem (.json written)")
    args = ap.parse_args()

    root = Path(args.reward_root) / "member_gather"
    arms = []
    for a in args.arms:
        d = root / a / f"holdout_{args.box}"
        if not (d / "tiles.npz").exists():
            print(f"  skip {a}: no tiles.npz under {d}", flush=True)
            continue
        print(f"=== {a}", flush=True)
        arms.append(analyse_arm(d, int(args.bins), list(args.k_sweep)))

    if not arms:
        print("no arm had an exported tiles.npz; nothing to do", flush=True)
        return 1

    out = Path(args.out).with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"ok": True, "box": args.box, "arms": arms,
                     "config": vars(args)})
    print(f"\n=== wrote {out}", flush=True)

    print("\n=== median over held-out hosts (displacement channels)")
    print(f"{'arm':<20} {'guard':>7} {'frozen':>7} {'halfvol':>8} {'HRvol':>7} "
          f"{'corrHR':>7} {'frzcorr':>8}")
    for a in arms:
        h = a["hosts"]
        med = lambda k: float(np.median([r[k] for r in h]))  # noqa: E731
        print(f"{a['arm']:<20} {med('guard_dis'):>7.2f} "
              f"{med('guard_dis_frozen'):>7.2f} "
              f"{med('hp_halfvol_cand'):>8.3f} {med('hp_halfvol_hr'):>7.3f} "
              f"{med('hp_corr_cand_hr'):>7.3f} {med('hp_corr_frozen_hr'):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
