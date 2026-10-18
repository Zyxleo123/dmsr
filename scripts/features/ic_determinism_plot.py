#!/usr/bin/env python
"""Figure for the IC-determinism test, drawn only from its JSON outputs (CPU, seconds).

  (a) r(k) of the Lagrangian displacement against HR: does the 256^3 run share
      HR's ripples between the LR and 256^3 grid limits? (frozen SR2 = control)
  (b) fraction of HR objects found at the same host-relative place, by r/R_vir,
      with each catalog's random-direction null (dashed)
  (c) fraction found at the same place AND built from the same IC patch

Either JSON may be missing; the panels it feeds are then left out.

  python scripts/features/ic_determinism_plot.py --ripples R.json --match M.json --out F.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BAND = "#efeee9"
REF = "#8c8b86"
# reference palette slots 1-2 (blue, orange): validated as a pair, both modes
BLUE = "#2a78d6"
ORANGE = "#eb6834"


def _label(name: str) -> str:
    """dmo256 -> 256³ run, sr2 -> frozen SR2."""
    if name.startswith("dmo") and name[3:].isdigit():
        return f"{name[3:]}³ run"
    return {"sr2": "frozen SR2"}.get(name, name)


def _load(path: Optional[str]) -> Optional[Dict]:
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text())
    return None


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _legend(ax, loc="best") -> None:
    leg = ax.legend(loc=loc, frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)


def panel_ripples(ax, rip: Dict) -> None:
    k = np.asarray(rip["k_centre"], dtype=float)
    rd = rip["r"]["disp"]
    g = rip["gate"]
    lo, hi = rip["band"]
    ax.axvspan(lo, hi, color=BAND, lw=0, label="gate band")
    ng = int(rip.get("ng", 256))
    for x, lab in ((rip["lr_nyquist"], "LR grid limit"),
                   (rip["dmo_nyquist"], f"{ng}³ grid limit")):
        ax.axvline(x, color=REF, lw=1, ls=":")
        ax.text(x, 1.06, lab, color=INK2, fontsize=8, ha="center", va="bottom")
    ax.plot(k, rd[f"dmo_{g['variant']}"], color=BLUE, lw=1.8,
            solid_capstyle="round", label=f"{ng}³ run vs HR")
    ax.plot(k, rd["sr2"], color=ORANGE, lw=1.8, solid_capstyle="round",
            label="frozen SR2 vs HR")
    ax.axhline(0, color=REF, lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-0.15, 1.12)
    ax.set_xlabel("k [h/Mpc] of the IC-position (Lagrangian) displacement", color=INK2)
    ax.set_ylabel("correlation with HR, r(k)", color=INK2)
    ax.set_title("(a) Do the runs share the fine ripples?", loc="left",
                 color=INK, fontsize=10.5)
    _legend(ax, "lower left")


def panel_match(ax, match: Dict, key: str, title: str, ylabel: str) -> None:
    cands = match["candidates"]
    prim = match["primary"]
    names = [prim] + [n for n in cands if n != prim][:1]
    colors = dict(zip(names, (BLUE, ORANGE)))
    labels = {n: _label(n) for n in names}
    bins = cands[prim]["radial"]
    xs = np.arange(len(bins))
    ends = []
    for n in names:
        rows = cands[n]["radial"]
        y = np.array([np.nan if r[key] is None else r[key] for r in rows], float)
        ax.plot(xs, y, color=colors[n], lw=1.8, marker="o", ms=6.5, mec=SURFACE,
                mew=1.5, label=labels[n])
        if key == "match":
            yn = np.array([np.nan if r["null"] is None else r["null"] for r in rows],
                          float)
            ax.plot(xs, yn, color=colors[n], lw=1.3, ls=(0, (4, 3)),
                    label=f"{labels[n]}: random-direction null")
        ok = np.flatnonzero(np.isfinite(y))
        if ok.size:
            ends.append([n, xs[ok[-1]], y[ok[-1]]])
    if len(ends) == 2 and abs(ends[0][2] - ends[1][2]) < 0.06:   # keep end labels apart
        hi_, lo_ = (ends[0], ends[1]) if ends[0][2] >= ends[1][2] else (ends[1], ends[0])
        hi_[2], lo_[2] = hi_[2] + 0.03, lo_[2] - 0.03
    for n, x, y in ends:
        ax.text(x + 0.12, y, labels[n], color=INK, fontsize=8, va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b['r_lo']:g}–{b['r_hi']:g}\nn={b['n']}" for b in bins])
    ax.set_xlim(-0.3, len(bins) - 0.3 + 0.9)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("distance from host, r / R_vir", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    _legend(ax, "upper left")


def _bins_axis(ax, rows, extra=None) -> np.ndarray:
    xs = np.arange(len(rows))
    lab = [f"{b['r_lo']:g}–{b['r_hi']:g}\n" + (extra(b) if extra else f"n={b['n']}")
           for b in rows]
    ax.set_xticks(xs)
    ax.set_xticklabels(lab)
    ax.set_xlim(-0.3, len(rows) - 0.3 + 0.9)
    ax.set_xlabel("distance from host, r / R_vir", color=INK2)
    return xs


def _col(rows, key) -> np.ndarray:
    return np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)


def _end_label(ax, xs, y, text) -> None:
    ok = np.flatnonzero(np.isfinite(y))
    if ok.size:
        ax.text(xs[ok[-1]] + 0.12, y[ok[-1]], text, color=INK, fontsize=8, va="center")


def draw_track(track: Dict, out: Path, plt) -> None:
    """(a) where HR's object matter ends up, (b) how compact it is, (c) the
    same-IC-patch object: same radius vs same direction."""
    cands = track["candidates"]
    prim = track["primary"]
    names = [prim] + [n for n in cands if n != prim][:1]
    colors = dict(zip(names, (BLUE, ORANGE)))
    labels = {n: _label(n) for n in names}
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6), facecolor=SURFACE)
    for ax in axes:
        _style(ax)
    rows0 = cands[prim]["radial"]

    ax = axes[0]
    xs = _bins_axis(ax, rows0)
    for n in names:
        y = _col(cands[n]["radial"], "matter_offset_mpc")
        ax.plot(xs, y, color=colors[n], lw=1.8, marker="o", ms=6.5, mec=SURFACE,
                mew=1.5, label=labels[n])
        _end_label(ax, xs, y, labels[n])
    ax.plot(xs, _col(rows0, "rvir_obj_mpc"), color=REF, lw=1.2, ls=":",
            label="the HR object's own r_vir")
    ax.set_yscale("log")
    ax.set_ylabel("offset of its matter from HR's object [Mpc/h]", color=INK2)
    ax.set_title("(a) Where HR's object's own particles end up", loc="left",
                 color=INK, fontsize=10.5)
    _legend(ax, "upper left")

    ax = axes[1]
    for n in names:
        y = _col(cands[n]["radial"], "matter_compact")
        ax.plot(xs, y, color=colors[n], lw=1.8, marker="o", ms=6.5, mec=SURFACE,
                mew=1.5, label=labels[n])
        _end_label(ax, xs, y, labels[n])
    ax.plot(xs, _col(rows0, "hr_compact"), color=REF, lw=1.4, ls=(0, (4, 3)),
            label="HR itself")
    _bins_axis(ax, rows0)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction of those particles within r_vir", color=INK2)
    ax.set_title("(b) ... and how tightly they are packed", loc="left", color=INK,
                 fontsize=10.5)
    _legend(ax, "lower left")

    ax = axes[2]
    xs = _bins_axis(ax, rows0, lambda b: (
        "–" if b.get("patch_found") is None else f"{100 * b['patch_found']:.0f}%"))
    ax.set_xlabel("distance from host, r / R_vir   (2nd line: share found)", color=INK2)
    sr = _col(rows0, "patch_same_r")
    sd = _col(rows0, "patch_same_dir")
    ax.plot(xs, sr, color=BLUE, lw=1.8, marker="o", ms=6.5, mec=SURFACE, mew=1.5,
            label=f"same radius (within ×{track['params']['radius_tol']:g})")
    ax.plot(xs, sd, color=BLUE, lw=1.8, ls=(0, (4, 3)), marker="s", ms=6, mec=SURFACE,
            mew=1.5, label=f"same direction (cos ≥ {track['params']['dir_cos']:g})")
    ax.plot(xs, _col(rows0, "patch_same_r_null"), color=REF, lw=1.2, ls=":",
            label="chance: radius re-paired in the bin")
    ax.plot(xs, _col(rows0, "patch_same_dir_null"), color=REF, lw=1.2,
            ls=(0, (1, 3)), label="chance: random direction")
    _end_label(ax, xs, sr, "radius")
    _end_label(ax, xs, sd, "direction")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction of the found same-patch objects", color=INK2)
    ax.set_title(f"(c) {labels[prim].split()[0]} object from the same IC patch: "
                 "radius vs direction",
                 loc="left", color=INK, fontsize=10.5)
    _legend(ax, "center right")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


AQUA = "#1baf7a"   # reference palette slot 3 (first three validate all-pairs)


def draw_scaling(sc: Dict, out: Path, plt, min_n: int = 15) -> None:
    """Recovery vs object size in the input's own particles, one line per input."""
    order = [("dmo256", BLUE), ("dmo128", AQUA), ("sr2", ORANGE)]
    runs = sc["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), facecolor=SURFACE)
    titles = {"outskirts_1_3": "(a) 1–3 R_vir", "mid_0.25_1": "(b) 0.25–1 R_vir"}
    for ax, (key, title) in zip(axes, titles.items()):
        _style(ax)
        for lab, col in order:
            if lab not in runs:
                continue
            r = runs[lab]
            rows = [b for b in r[key] if b["n"] >= min_n and b["match"] is not None]
            if not rows:
                continue
            xc = np.array([np.sqrt(b["n_lo"] * b["n_hi"]) for b in rows])
            name = ("SR2 (64³ LR input)" if lab == "sr2" else _label(lab))
            name += f", ripples ≥ {r['finest_ripple_mpc']:.2f} Mpc/h"
            ax.plot(xc, [b["match"] for b in rows], color=col, lw=1.8, marker="o",
                    ms=6.5, mec=SURFACE, mew=1.5, label=name)
            ax.plot(xc, [b["null"] for b in rows], color=col, lw=1.2, ls=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("object size in particles of the INPUT run", color=INK2)
        ax.set_ylabel("fraction of HR objects found at the same place", color=INK2)
        ax.set_title(f"{title}: found at the same place (dashed: chance)",
                     loc="left", color=INK, fontsize=10)
        _legend(ax, "upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ripples", default=None)
    p.add_argument("--match", default=None)
    p.add_argument("--out", default=None, help="the ripples + match figure")
    p.add_argument("--track", default=None, help="track JSON (optional)")
    p.add_argument("--track-out", default=None, help="the track figure")
    p.add_argument("--scaling", default=None, help="scaling JSON (optional)")
    p.add_argument("--scaling-out", default=None, help="the scaling figure")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = _load(a.scaling)
    if sc and a.scaling_out:
        draw_scaling(sc, Path(a.scaling_out), plt)
    track = _load(a.track)
    if track and a.track_out:
        draw_track(track, Path(a.track_out), plt)
    elif a.track:
        print(f"no track figure: JSON {a.track} missing or --track-out not given")
    if not a.out:
        return 0
    rip, match = _load(a.ripples), _load(a.match)
    has_match = bool(match and match.get("candidates"))
    panels = (["rip"] if rip else []) + (["match", "patch"] if has_match else [])
    if not panels:
        print(f"nothing to draw: no ripples JSON at {a.ripples} and no match JSON at {a.match}")
        return 0
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4),
                             facecolor=SURFACE, squeeze=False)
    for ax, kind in zip(axes[0], panels):
        _style(ax)
        if kind == "rip":
            panel_ripples(ax, rip)
        elif kind == "match":
            panel_match(ax, match, "match", "(b) HR object found at the same place today",
                        "fraction of HR objects matched")
        else:
            panel_match(ax, match, "same_patch",
                        "(c) ... and made of the same IC patch",
                        "fraction of HR objects")
    head = []
    if has_match:
        head.append(f"Verdict: {match['verdict']['call']}")
    if rip:
        head.append(f"Ripple gate: {rip['gate']['reason']}")
    fig.text(0.01, 0.995, "\n".join(head), color=INK, fontsize=9.5, va="top", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
