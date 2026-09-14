#!/usr/bin/env python
"""Figures for `lr_host_fidelity.py`, redrawn from the saved npz alone.

No catalog parse and no halo finder here: everything comes from
``<out-dir>/<box>.npz`` and ``summary.json``, so a re-plot is seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lr_host_fidelity import HOST_BINS, HOST_LABELS  # noqa: E402


def _bin_index(num_p: np.ndarray) -> np.ndarray:
    idx = np.full(num_p.shape, -1, dtype=np.int64)
    for k, (lo, hi) in enumerate(HOST_BINS):
        idx[(num_p >= lo) & (num_p < hi)] = k
    return idx


def plot_box(out_dir: Path, box: str, summary: dict) -> None:
    d = np.load(out_dir / f"{box}.npz")
    num_p = d["num_p"]
    bi = _bin_index(num_p)
    bins = {b["bin"]: b for b in summary.get("bins", []) if b.get("n_hosts")}

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))

    # (a) THE DECISION PANEL: centre error against HR's own subhalo radius.
    a = ax[0, 0]
    xs = np.arange(len(HOST_LABELS))
    lr_e = [bins.get(l, {}).get("lr_centre_err_mpc_median", np.nan) for l in HOST_LABELS]
    po_e = [bins.get(l, {}).get("pot_centre_err_mpc_median", np.nan) for l in HOST_LABELS]
    sub_r = [bins.get(l, {}).get("hr_median_sub_r_mpc", np.nan) for l in HOST_LABELS]
    a.plot(xs, lr_e, "o-", label="LR catalog centre error")
    a.plot(xs, po_e, "s-", label="LR potential centre error")
    a.plot(xs, sub_r, "k^--", label="HR median subhalo radius (yardstick)")
    a.set_xticks(xs); a.set_xticklabels(HOST_LABELS)
    a.set_yscale("log"); a.set_xlabel("HR host num_p"); a.set_ylabel("Mpc/h")
    a.set_title("(a) can the channel resolve a subhalo's radius?\n"
                "below the dashed line = usable coordinate")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # (b) completeness -- where the catalog channel falls off its cliff
    a = ax[0, 1]
    comp = [bins.get(l, {}).get("completeness_within_rvir", np.nan) for l in HOST_LABELS]
    shar = [bins.get(l, {}).get("shared_lr_host_frac", np.nan) for l in HOST_LABELS]
    a.bar(xs - 0.19, comp, width=0.38, label="LR host within R_vir")
    a.bar(xs + 0.19, shar, width=0.38, label="shares its LR host")
    a.set_xticks(xs); a.set_xticklabels(HOST_LABELS)
    a.set_ylim(0, 1.05); a.set_xlabel("HR host num_p"); a.set_ylabel("fraction")
    a.set_title("(b) LR catalog completeness and degeneracy")
    a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")

    # (c) scale recovery: does either channel carry host mass?
    a = ax[1, 0]
    r2c = [bins.get(l, {}).get("r2_logM_from_lr_catalog", np.nan) for l in HOST_LABELS]
    r2p = [bins.get(l, {}).get("r2_logM_from_potential", np.nan) for l in HOST_LABELS]
    r2r = [bins.get(l, {}).get("r2_log_r_over_rvir_from_potential", np.nan)
           for l in HOST_LABELS]
    a.bar(xs - 0.26, r2c, width=0.25, label="$M_{host}$ from LR catalog")
    a.bar(xs, r2p, width=0.25, label="$M_{host}$ from LR potential")
    a.bar(xs + 0.26, r2r, width=0.25,
          label="$r/R_{vir}$ at the subhalo, from potential")
    a.axhline(0.0, color="k", lw=0.8)
    a.set_xticks(xs); a.set_xticklabels(HOST_LABELS)
    a.set_xlabel("HR host num_p"); a.set_ylabel("held-out $R^2$")
    a.set_title("(c) what each channel actually carries (out of sample)")
    a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")

    # (d) the raw centre-error distribution, in units of HR R_vir
    a = ax[1, 1]
    for k, lab in enumerate(HOST_LABELS):
        m = bi == k
        if m.sum() < 5:
            continue
        v = d["lr_dr_over_rvir"][m]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            a.hist(np.log10(v), bins=30, histtype="step", label=f"{lab} (n={m.sum()})")
    a.axvline(0.0, color="k", ls="--", lw=1, label="one $R_{vir}$")
    a.set_xlabel("$\\log_{10}$ (LR centre error / HR $R_{vir}$)")
    a.set_ylabel("hosts"); a.set_title("(d) LR centre error distribution")
    a.legend(fontsize=7); a.grid(alpha=0.3)

    fig.suptitle(f"LR host channel fidelity -- {box}  "
                 f"(LR 64$^3$, $m_p$={summary.get('lr_particle_mass_msun_h', 0):.2e} "
                 f"M$_\\odot$/h)")
    fig.tight_layout()
    p = out_dir / f"lr_host_fidelity_{box}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    for s in summary.get("per_box", []):
        plot_box(out_dir, s["box"], s)
    print(f"=== figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
