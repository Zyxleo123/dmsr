"""Render the moment-target verification figure from the build job's artifacts.

Pure redraw: reads ``<box>_moment_target_diag.npz`` and
``<box>_moment_target_summary.json`` (written by
``scripts/features/build_moment_target.py``) and draws one PNG. No field, no
generator, no recomputation -- so it is a cheap CPU job and can be re-run to
restyle the figure.

The figure answers *is the target correct?* in three views, top to bottom:

* **Slices** of ``|displacement|`` through the most massive host: HR, SR2, the raw
  residual ``Psi_HR - Psi_SR2``, and the projected target. The eye-check is that
  the target is the residual with the smooth host bulk removed and the small-scale
  structure kept.
* **Windowed radial spectrum** of the residual vs the target on a crop around that
  host: the projection should strip the low-k (bulk) power and leave the high-k
  (substructure) power intact.
* **Per-host audit**: the affine-moment norm before and after projection (after
  must sit on the floor), and the fraction of each host's displacement variance
  that was affine and was removed, against host mass.

    python scripts/features/render_moment_target.py --box set8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def reward_root() -> Path:
    return Path(os.environ.get(
        "DMSR_REWARD_ROOT", "/zfsauton/scratch/yixiz/DMSR/dmsr_reward"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--box", default=os.environ.get("MT_BOX", "set8"))
    args = ap.parse_args()
    box = args.box
    d = reward_root() / "moment_target" / box

    diag = np.load(d / f"{box}_moment_target_diag.npz")
    summary = json.loads((d / f"{box}_moment_target_summary.json").read_text())

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.0, 1.0], hspace=0.35, wspace=0.3)

    # --- row 1: slices ------------------------------------------------------
    names = ["hr", "sr2", "residual", "target"]
    titles = ["HR  |disp|", "SR2  |disp|",
              "residual  Psi_HR - Psi_SR2", "target  Pi(residual)"]
    panels = {n: diag[f"panel_{n}"] for n in names}
    # HR and SR2 share a scale; residual and target share their own (smaller) one.
    vmax_field = float(np.percentile(np.concatenate(
        [panels["hr"].ravel(), panels["sr2"].ravel()]), 99))
    vmax_res = float(np.percentile(np.concatenate(
        [panels["residual"].ravel(), panels["target"].ravel()]), 99))
    for j, (n, t) in enumerate(zip(names, titles)):
        ax = fig.add_subplot(gs[0, j])
        vmax = vmax_field if n in ("hr", "sr2") else vmax_res
        im = ax.imshow(panels[n], origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    th = summary["top_host"]
    fig.text(0.5, 0.955,
             f"{box}: most massive host id {th['halo_id']}  "
             f"log Mvir = {th['log_mvir']:.2f}   (slice |disp| in Mpc/h)",
             ha="center", fontsize=11)

    # --- row 2 left: spectrum ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0:2])
    k = diag["k_hmpc"]
    m = k > 0
    ax.loglog(k[m], diag["power_residual"][m], label="residual", lw=2)
    ax.loglog(k[m], diag["power_target"][m], label="target  Pi(residual)", lw=2)
    ax.axvspan(k[m].min(), 1.0, color="0.85", zorder=0)
    ax.text(k[m].min() * 1.1, ax.get_ylim()[0], "  bulk (low-k)\n  removed",
            va="bottom", fontsize=8, color="0.35")
    ax.set_xlabel("k  [h/Mpc]"); ax.set_ylabel("windowed |disp|^2(k)")
    ax.set_title("host-crop spectrum: bulk stripped, substructure kept", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2)

    # --- row 2 right: moment norm before/after -----------------------------
    ax = fig.add_subplot(gs[1, 2:4])
    ph = summary["per_host"]
    before = np.array([h["moment_norm_before"] for h in ph])
    after = np.array([h["moment_norm_after"] for h in ph])
    order = np.argsort(-before)
    x = np.arange(len(ph))
    ax.semilogy(x, before[order], "o-", ms=4, label="before (residual)")
    ax.semilogy(x, np.maximum(after[order], 1e-16), "s-", ms=4,
                label="after (target)")
    ax.set_xlabel("host, ranked by moment before")
    ax.set_ylabel("||Phi^T disp||  (affine-moment norm)")
    ax.set_title("affine moment removed per host", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.2)

    # --- row 3 left: affine fraction vs mass -------------------------------
    ax = fig.add_subplot(gs[2, 0:2])
    logm = np.array([h["log_mvir"] for h in ph])
    frac = np.array([h["affine_var_frac"] for h in ph])
    ax.scatter(logm, frac, s=30, alpha=0.8)
    ax.set_xlabel("log10 Mvir"); ax.set_ylabel("affine fraction of disp variance")
    ax.set_ylim(0, 1)
    ax.set_title("fraction of each host's displacement that was bulk (removed)",
                 fontsize=10)
    ax.grid(True, alpha=0.2)

    # --- row 3 right: verdict text -----------------------------------------
    ax = fig.add_subplot(gs[2, 2:4]); ax.axis("off")
    g = summary["global"]
    lines = [
        f"mode: {summary['mode']}    hosts: {summary['n_hosts_total']}",
        "",
        f"off-footprint max |target - residual| = "
        f"{summary['offfootprint_max_abs_diff']:.2e}",
        f"  -> {'OK (untouched)' if summary['offfootprint_ok'] else 'LEAK - investigate'}",
        "",
        f"rms residual disp = {g['rms_residual_disp']:.4f}",
        f"rms target   disp = {g['rms_target_disp']:.4f}",
        f"sites in a footprint = {g['frac_sites_in_footprint']:.1%}",
        "",
        "VERDICT:",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", fontsize=10, family="monospace")
    ax.text(0.0, 0.30, summary["verdict"], va="top", fontsize=9,
            family="monospace", wrap=True,
            color="green" if summary["verdict"].startswith("PASS") else "darkorange")

    out = d / f"{box}_moment_target.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
