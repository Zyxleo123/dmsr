#!/usr/bin/env python
"""Plot subhalo mass vs distance from host centre. CPU, seconds.

Reads only the ``<kind>.npz`` + ``summary.json`` that ``subhalo_radial.py``
wrote, so a colour/label/layout change re-renders without re-parsing the
catalogs -- the redrawable-figure rule.

Two panels per figure (one row per kind, `hr` and `self` by default):
  * scatter of subhalo mass (num_p, log y) vs R/R_vir, points sized by mass,
    with the binned median-mass profile overlaid;
  * the same in Mpc/h so the physical scale is visible.
Plus a shared overlay of the median-mass radial profiles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"hr": "#111111", "self": "#d62728", "base": "#888888"}
LABEL = {"hr": "HR (truth)", "self": "self (tuned)", "base": "base (LR->SR)"}


def _load(out_dir: Path, kind: str):
    d = np.load(out_dir / f"{kind}.npz")
    return {k: d[k] for k in d.files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True,
                    help="dir holding <kind>.npz + summary.json")
    ap.add_argument("--kinds", default="hr,self")
    ap.add_argument("--fig", default="")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    data = {k: _load(out_dir, k) for k in kinds}

    fig, axes = plt.subplots(len(kinds), 2, figsize=(11, 4.2 * len(kinds)),
                             squeeze=False)
    for i, kind in enumerate(kinds):
        arr = data[kind]
        col = C.get(kind, "#333333")
        prof = summary["per_kind"][kind]["profile"]
        edges = np.asarray(prof["rvir_bin_edges"])
        centres = 0.5 * (edges[:-1] + edges[1:])
        med = np.array([b.get("num_p_median", np.nan) for b in prof["bins"]])
        p25 = np.array([b.get("num_p_p25", np.nan) for b in prof["bins"]])
        p75 = np.array([b.get("num_p_p75", np.nan) for b in prof["bins"]])

        # --- panel A: R/R_vir ------------------------------------------------
        ax = axes[i][0]
        x = arr["r_over_rvir"]
        y = np.maximum(arr["num_p"], 1)
        good = np.isfinite(x)
        ax.scatter(x[good], y[good], s=5, c=col, alpha=0.18, linewidths=0,
                   rasterized=True)
        ax.plot(centres, med, "-o", color=col, lw=2, ms=4, mec="white",
                label="median")
        ax.fill_between(centres, p25, p75, color=col, alpha=0.18, lw=0)
        ax.set_yscale("log")
        ax.set_xlim(0, 5)
        ax.set_xlabel(r"$R / R_{\rm vir}^{\rm host}$")
        ax.set_ylabel("subhalo particle count")
        n_sub = summary["per_kind"][kind]["n_sub"]
        ax.set_title(f"{LABEL.get(kind, kind)}   ({n_sub} subhalos)")
        ax.axhline(50, color="gray", ls=":", lw=1)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

        # --- panel B: Mpc/h --------------------------------------------------
        ax = axes[i][1]
        r = arr["r_mpc"]
        ax.scatter(r, y, s=5, c=col, alpha=0.18, linewidths=0, rasterized=True)
        ax.set_yscale("log")
        ax.set_xlabel(r"distance from host centre  [$\mathrm{Mpc}/h$]")
        ax.set_ylabel("subhalo particle count")
        ax.set_title(f"{LABEL.get(kind, kind)} (physical scale)")
        ax.axhline(50, color="gray", ls=":", lw=1)

    box = summary["box"]
    reg = " (edited region)" if summary.get("region_restricted") else ""
    fig.suptitle(f"Subhalo mass vs. distance from host centre -- {box}{reg}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig_path = Path(a.fig) if a.fig else out_dir / "subhalo_radial.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"wrote {fig_path}")

    # --- shared median-profile overlay --------------------------------------
    fig2, ax = plt.subplots(figsize=(6.5, 4.5))
    for kind in kinds:
        prof = summary["per_kind"][kind]["profile"]
        edges = np.asarray(prof["rvir_bin_edges"])
        centres = 0.5 * (edges[:-1] + edges[1:])
        med = np.array([b.get("num_p_median", np.nan) for b in prof["bins"]])
        ax.plot(centres, med, "-o", color=C.get(kind, "#333"), lw=2, ms=5,
                label=LABEL.get(kind, kind))
    ax.set_yscale("log")
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$R / R_{\rm vir}^{\rm host}$")
    ax.set_ylabel("median subhalo particle count")
    ax.set_title(f"Median subhalo mass profile -- {box}{reg}")
    ax.axhline(50, color="gray", ls=":", lw=1)
    ax.legend(frameon=False)
    fig2.tight_layout()
    prof_path = out_dir / "subhalo_radial_profile.png"
    fig2.savefig(prof_path, dpi=140, bbox_inches="tight")
    print(f"wrote {prof_path}")

    # --- host mass vs. distance, one panel per kind -------------------------
    # y = the subhalo's HOST particle count; x = the subhalo's distance from
    # that host centre. One point per subhalo, so massive hosts (many subs)
    # show as vertical stripes. Left column Mpc/h, right column R/R_vir.
    fig3, axes3 = plt.subplots(len(kinds), 2, figsize=(11, 4.2 * len(kinds)),
                               squeeze=False)
    for i, kind in enumerate(kinds):
        arr = data[kind]
        col = C.get(kind, "#333333")
        hm = np.maximum(arr["host_num_p"], 1)
        for j, (xkey, xlab, xlim) in enumerate((
                ("r_mpc", r"distance from host centre  [$\mathrm{Mpc}/h$]", None),
                ("r_over_rvir", r"$R / R_{\rm vir}^{\rm host}$", (0, 5)))):
            ax = axes3[i][j]
            x = arr[xkey]
            good = np.isfinite(x)
            ax.scatter(x[good], hm[good], s=5, c=col, alpha=0.18,
                       linewidths=0, rasterized=True)
            ax.set_yscale("log")
            if xlim:
                ax.set_xlim(*xlim)
            ax.set_xlabel(xlab)
            ax.set_ylabel("host particle count")
            n_sub = summary["per_kind"][kind]["n_sub"]
            ax.set_title(f"{LABEL.get(kind, kind)}   ({n_sub} subhalos)")
            ax.axhline(200, color="gray", ls=":", lw=1)  # host floor
    fig3.suptitle(f"Host mass vs. subhalo distance from host centre -- "
                  f"{box}{reg}", fontsize=13)
    fig3.tight_layout(rect=(0, 0, 1, 0.98))
    host_path = out_dir / "host_mass_vs_distance.png"
    fig3.savefig(host_path, dpi=140, bbox_inches="tight")
    print(f"wrote {host_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
