#!/usr/bin/env python
"""Talk figures for the all-hosts abundance result and the host forensic.

Reads only the SAVED json the two diagnostics wrote -- no catalog, no field --
so a colour/label/layout change re-renders in seconds (the redrawable rule):

  report.json                    (gather_all_hosts_report.py, pooled)
  <arm>/holdout_<box>/forensics/host_forensics_<box>.json  (host_forensics.py)

Figures:
  sub_mass_function  subhalo count per num_p bin, HR / base / tuned, pooled
                     over every HR host -- the mass function behind the single
                     "92% of HR" number (linear + log).
  fig1_abundance     subs/host as % of HR per arm (pooled), with the host-
                     survival cost beneath -- the headline "91% recovered" and
                     the "but 6% of hosts" caveat in one panel.
  fig2_occupation    mean subs/host vs host mass, HR/base/self/smallhalo -- the
                     cluster-scale recovery and the top-bin overshoot.
  fig3_tradeoff      abundance (% of HR) vs host survival, one point per arm --
                     the Pareto view: self buys abundance with hosts, smallhalo
                     is the gentler trade.
  fig4_forensic      lost-host fates (SHRUNK/DEMOTED/VANISHED) split genuine vs
                     bookkeeping, self vs selfhostguard -- the -320 is mostly
                     bookkeeping and mostly re-parenting.
  fig5_fate_by_mass  self's lost hosts by mass bin x fate -- DEMOTED is mass-
                     independent (massive hosts re-parented), SHRUNK/VANISHED
                     are the small-host floor channel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Arm colours, extending gather_holdout_figures_plot's map so the talk is
# internally consistent across figures.
C = {"hr": "#111111", "base": "#888888", "frozen": "#1f77b4",
     "self": "#d62728", "selfhostguard": "#e377c2", "smallhalo": "#17becf",
     "nocentre": "#2ca02c", "full": "#9467bd", "radial": "#ff7f0e",
     "selfvel": "#bcbd22", "selfshell": "#8c564b", "hostguard": "#7f7f7f"}
FATE_C = {"SHRUNK": "#4c72b0", "DEMOTED": "#dd8452", "VANISHED": "#c44e52",
          "SURVIVED": "#cccccc"}


def short(arm: str) -> str:
    return arm.replace("all_blocks_", "")


def col(arm: str) -> str:
    return C.get(short(arm), "#555555")


def _save(fig, out: Path, name: str):
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)


# --------------------------------------------------------------------------
def fig_abundance(rep: Dict, out: Path):
    """% of HR subs/host per arm (bars) + host survival beneath."""
    arms = rep["arms"]
    hr_ph = arms["hr"]["subs_per_host"]
    base_tot = arms["base"]["subs_total"]
    hr_tot = arms["hr"]["subs_total"]
    order = [a for a in ("base", "all_blocks_full", "all_blocks_radial",
                         "all_blocks_selfshell", "all_blocks_selfvel",
                         "all_blocks_nocentre", "all_blocks_hostguard",
                         "all_blocks_selfhostguard", "all_blocks_smallhalo",
                         "all_blocks_self") if a in arms]
    pct = [100 * arms[a]["subs_per_host"] / hr_ph for a in order]
    recov = [100 * (arms[a]["subs_total"] - base_tot) / max(hr_tot - base_tot, 1)
             for a in order]
    alive = [100 * arms[a]["hosts_alive_frac"] for a in order]
    labels = [short(a) if a != "base" else "base" for a in order]
    x = np.arange(len(order))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.5), height_ratios=[2.4, 1], sharex=True)
    bars = ax.bar(x, pct, color=[col(a) for a in order], edgecolor="black",
                  linewidth=0.5)
    ax.axhline(100, color="#111111", lw=1.2, ls="--")
    ax.text(len(order) - 0.4, 101.5, "HR", ha="right", va="bottom", fontsize=9)
    for xi, p, r in zip(x, pct, recov):
        ax.text(xi, p + 1.5, f"{p:.0f}%\n({r:+.0f})", ha="center", va="bottom",
                fontsize=7.5)
    ax.set_ylabel("subhalos per host\n(% of HR)")
    ax.set_ylim(0, max(pct) * 1.18)
    n = rep["n_hosts"]
    ax.set_title(f"Subhalo abundance recovered, {n} HR hosts over "
                 f"{'+'.join(rep['boxes'])}\n(bar = % of HR; parenthetical = "
                 f"% of the base deficit recovered)")

    ax2.bar(x, alive, color=[col(a) for a in order], edgecolor="black",
            linewidth=0.5, alpha=0.85)
    ax2.axhline(100 * arms["base"]["hosts_alive_frac"], color="#888888",
                lw=1.0, ls=":")
    ax2.text(0, 100 * arms["base"]["hosts_alive_frac"] + 0.2, "base",
             fontsize=7, color="#555555")
    ax2.set_ylabel("hosts resolved\n(% of HR hosts)")
    ax2.set_ylim(0, max(max(alive) * 1.25, 8))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.text(0.99, 0.9, "the cost: host resolution stays ~6%",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8,
             color="#555555")
    fig.tight_layout()
    _save(fig, out, "fig1_abundance.png")


def fig_occupation(rep: Dict, out: Path):
    """Mean subs/host vs host mass bin, key arms."""
    edges = rep["host_mass_edges_log10"]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    arms = rep["arms"]
    show = [a for a in ("hr", "base", "all_blocks_self", "all_blocks_smallhalo",
                        "all_blocks_selfhostguard") if a in arms]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for a in show:
        y = [b["subs_per_host"] for b in arms[a]["by_host_mass"]]
        ax.plot(centres, y, "o-", color=col(a) if a != "hr" else "#111111",
                lw=2 if a in ("hr", "all_blocks_self") else 1.4,
                ms=5, label=short(a) if a != "hr" else "HR",
                zorder=3 if a == "all_blocks_self" else 2)
    ax.set_yscale("log")
    ax.set_xlabel(r"host mass  $\log_{10}(M_\mathrm{vir}/h^{-1}M_\odot)$")
    ax.set_ylabel("mean subhalos per host (within $R_{vir}$, $\\geq$50p)")
    ax.set_title("Recovery is cluster-scale; self overshoots HR at the top bin")
    # annotate the overshoot
    self_top = arms["all_blocks_self"]["by_host_mass"][-1]["subs_per_host"]
    hr_top = arms["hr"]["by_host_mass"][-1]["subs_per_host"]
    ax.annotate(f"self {self_top:.0f} vs HR {hr_top:.0f}",
                xy=(centres[-1], self_top), xytext=(centres[-1] - 1.1, self_top * 1.5),
                fontsize=8, color=C["self"],
                arrowprops=dict(arrowstyle="->", color=C["self"], lw=1))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, out, "fig2_occupation.png")


def fig_tradeoff(rep: Dict, out: Path):
    """Abundance (% of HR) vs host survival (% of HR hosts), per arm."""
    arms = rep["arms"]
    hr_ph = arms["hr"]["subs_per_host"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for a, e in arms.items():
        if a in ("hr",):
            continue
        x = 100 * e["hosts_alive_frac"]
        y = 100 * e["subs_per_host"] / hr_ph
        ax.scatter(x, y, s=70, color=col(a), edgecolor="black", linewidth=0.6,
                   zorder=3)
        ax.annotate(short(a) if a != "base" else "base", (x, y),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(100, color="#111111", ls="--", lw=1)
    ax.text(ax.get_xlim()[0], 100.5, "HR abundance", fontsize=8, va="bottom")
    ax.set_xlabel("hosts resolved (% of HR hosts)  --  the cost")
    ax.set_ylabel("subhalos per host (% of HR)  --  the win")
    ax.set_title("The trade: self maxes abundance; smallhalo keeps more hosts")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, out, "fig3_tradeoff.png")


def fig_forensic(fors: Dict[str, Dict], out: Path):
    """Lost-host fates split genuine vs bookkeeping, one group per arm."""
    fates = ("SHRUNK", "DEMOTED", "VANISHED")
    arms = [a for a in ("self", "selfhostguard") if a in fors]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    width = 0.8 / len(fates)
    x = np.arange(len(arms))
    for j, f in enumerate(fates):
        genuine = [fors[a]["by_fate_genuine_damage"][f] for a in arms]
        total = [fors[a]["by_fate"][f] for a in arms]
        book = [t - g for t, g in zip(total, genuine)]
        xf = x + (j - (len(fates) - 1) / 2) * width
        ax.bar(xf, genuine, width, color=FATE_C[f], edgecolor="black",
               linewidth=0.5, label=f"{f} (genuine)")
        ax.bar(xf, book, width, bottom=genuine, color=FATE_C[f], alpha=0.35,
               edgecolor="black", linewidth=0.5,
               label=f"{f} (bookkeeping)")
        for xi, t, g in zip(xf, total, genuine):
            ax.text(xi, t + 2, f"{t}\n({g})", ha="center", va="bottom",
                    fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{a}\nnet {fors[a]['reconciliation']['net']:+d}  "
         f"({fors[a]['reconciliation']['arm_hosts_actual']} hosts)"
         for a in arms], fontsize=9)
    ax.set_ylabel("base hosts lost in the edited region")
    ax.set_ylim(0, max(fors[a]["by_fate"][f] for a in arms for f in fates) * 1.18)
    ax.set_title("The lost hosts: solid = genuine (HR keeps it), pale = "
                 "bookkeeping\nDEMOTED (re-parenting) dominates the genuine "
                 "damage")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    _save(fig, out, "fig4_forensic.png")


def fig_fate_by_mass(fors: Dict[str, Dict], out: Path, arm: str = "self"):
    """One arm's lost hosts, stacked fate per mass bin."""
    if arm not in fors:
        return
    bm = fors[arm]["by_mass_bin"]
    order = ["200-500", "500-2000", "2000-inf"]  # sub-floor bins are empty
    bins = [b for b in order if b in bm]
    fates = ("SHRUNK", "VANISHED", "DEMOTED")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(len(bins))
    bottom = np.zeros(len(bins))
    for f in fates:
        vals = np.array([bm[b][f] for b in bins])
        ax.bar(x, vals, bottom=bottom, color=FATE_C[f], edgecolor="black",
               linewidth=0.5, label=f)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}p" for b in bins])
    ax.set_xlabel("host mass (Rockstar num_p)")
    ax.set_ylabel(f"{arm} hosts lost")
    ax.set_title("SHRUNK/VANISHED are the small-host floor channel;\n"
                 "DEMOTED (re-parenting) is mass-independent")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig5_fate_by_mass.png")


# The num_p bins the abundance job histograms into, in physical order (the json
# is a dict, so alphabetical order would put "2000+p" second).
MASS_BIN_ORDER = ["50-100p", "100-200p", "200-500p", "500-2000p", "2000+p"]


def fig_sub_mass_function(rep: Dict, out: Path, tuned: str = "all_blocks_self",
                          yscale: str = "linear"):
    """Subhalo mass function: count per num_p bin, HR / base / tuned.

    The headline of [[all-hosts-abundance-report]] as a mass function rather
    than a single ratio -- pooled over every HR host of every gated box, so the
    bars answer "does the tuned field make subhalos of the RIGHT masses, not
    just the right number". Rendered linear (the slide: the 50-100p bin holds
    46% of HR's subhalos and dominates on purpose) and log (every bin readable,
    including the 2000+p bin where base is already close to HR).
    """
    arms = rep["arms"]
    if tuned not in arms:
        print(f"  (tuned arm {tuned} not in report, skipping)", flush=True)
        return
    bins = [b for b in MASS_BIN_ORDER if b in arms["hr"]["mass_function"]]
    # "ours" on the figure, not the internal arm name -- the arm is provenance
    # and goes to the log, matching fig_occupation_selfonly's labelling.
    print(f"  sub_mass_function: tuned arm = {tuned}", flush=True)
    series = [("hr", "#111111", "HR (truth)"),
              ("base", "#888888", "SR2 (baseline)"),
              (tuned, C["self"], "Tuned (ours)")]
    x = np.arange(len(bins))
    width = 0.27
    lin = yscale == "linear"

    hr_mf = arms["hr"]["mass_function"]
    top = max(arms[a]["mass_function"].get(b, 0)
              for a, _, _ in series for b in bins)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for j, (a, cc, lab) in enumerate(series):
        mf = arms[a]["mass_function"]
        y = [mf.get(b, 0) for b in bins]
        xs = x + (j - 1) * width
        # width*0.93 leaves a surface gap between adjacent bars; counts are
        # direct-labelled because there are only 15 of them and the count IS
        # the message.
        ax.bar(xs, y, width * 0.93, color=cc, edgecolor="white", linewidth=0.8,
               label=lab, zorder=2)
        for xi, v in zip(xs, y):
            ax.text(xi, v * 1.03 if not lin else v + top * 0.012, f"{v}",
                    ha="center", va="bottom", fontsize=8, color="#333333",
                    zorder=3)

    # Tuned-as-%-of-HR beneath each bin label: the per-bin version of the
    # pooled 92% headline, in muted ink rather than on the bars themselves.
    tf = arms[tuned]["mass_function"]
    pct = [f"{100 * tf.get(b, 0) / hr_mf[b]:.0f}%" for b in bins]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n({p} of HR)" for b, p in zip(bins, pct)],
                       fontsize=9.5)
    ax.set_yscale(yscale)
    if lin:
        ax.set_ylim(0, top * 1.12)
    ax.set_xlabel("subhalo mass (Rockstar bound particles, num_p)")
    ax.set_ylabel(r"subhalos within $R_{vir}$ of an HR host")
    ax.set_title(
        f"Subhalo mass function over all {rep['n_hosts']} HR hosts "
        f"({'+'.join(rep['boxes'])})")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "sub_mass_function"
          + ("_log" if not lin else "") + ".png")


def fig_occupation_selfonly(rep: Dict, out: Path, yscale: str = "log",
                            mass_min: Optional[float] = None):
    """Occupation vs host mass, HR / base / self only -- for the talk.

    Rendered twice: log y (every mass bin readable, four decades of occupation)
    and linear y (the top bin dominates -- HR 217 vs tuned 283 -- and the four
    bins below 10^13.5 collapse onto zero). The log panel is the slide; the
    linear one is kept because it is the honest picture of where the mass of
    the signal lives.
    """
    edges = rep["host_mass_edges_log10"]
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    arms = rep["arms"]
    lin = yscale == "linear"
    # Optional low-mass crop: the bins below ~10^12.5 hold almost no resolved
    # (>=50p) substructure even in HR (0.13 subs/host), so they are noise that
    # costs the readable range. keep[] indexes both centres and by_host_mass.
    keep = [i for i, c in enumerate(centres)
            if mass_min is None or c >= mass_min]
    xs = [centres[i] for i in keep]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for a, cc, lab, lw in (("hr", "#111111", "HR (truth)", 2.2),
                           ("base", "#888888", "SR2 (baseline)", 1.6),
                           ("all_blocks_self", C["self"], "Tuned (ours)", 2.4)):
        y = [arms[a]["by_host_mass"][i]["subs_per_host"] for i in keep]
        ax.plot(xs, y, "o-", color=cc, lw=lw, ms=6, label=lab)
        if lin:
            # On a linear axis only the top bin is legible, so print its value.
            ax.annotate(f"{y[-1]:.0f}", (xs[-1], y[-1]),
                        textcoords="offset points", xytext=(6, 0),
                        va="center", fontsize=9, color=cc)
    ax.set_yscale(yscale)
    if lin:
        ax.set_ylim(bottom=0)
        ax.set_xlim(right=xs[-1] + 0.55)
    ax.set_xlabel(r"host mass  $\log_{10}(M_\mathrm{vir}/h^{-1}M_\odot)$")
    ax.set_ylabel(r"mean subhalos per host ($\geq$50p, within $R_{vir}$)")
    ax.set_title("Occupation recovered across host mass "
                 f"({rep['n_hosts']} hosts, {'+'.join(rep['boxes'])})"
                 + (" -- linear $y$" if lin else "")
                 + ("" if mass_min is None
                    else f", $\\log M > {edges[keep[0]]:g}$"))
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, which="major" if lin else "both", axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    name = "occupation_selfonly_linear" if lin else "occupation_selfonly"
    if mass_min is not None:
        name += f"_mmin{mass_min:g}"
    _save(fig, out, name + ".png")


def fig_radial(radial_dir: Path, out: Path):
    """Normalised radial distribution of subhalos, HR vs self -- the one caveat.

    self's subhalos pile up near the centre; HR's fill and extend past R_vir.
    """
    import numpy as _np
    hr = _np.load(radial_dir / "hr.npz")["r_over_rvir"]
    sf = _np.load(radial_dir / "self.npz")["r_over_rvir"]
    hr = hr[_np.isfinite(hr)]
    sf = sf[_np.isfinite(sf)]
    bins = _np.linspace(0, 3, 31)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(hr, bins=bins, density=True, histtype="stepfilled", color="#111111",
            alpha=0.25, label=f"HR (truth), median {_np.median(hr):.2f}")
    ax.hist(hr, bins=bins, density=True, histtype="step", color="#111111", lw=1.8)
    ax.hist(sf, bins=bins, density=True, histtype="stepfilled", color=C["self"],
            alpha=0.30, label=f"Tuned (ours), median {_np.median(sf):.2f}")
    ax.hist(sf, bins=bins, density=True, histtype="step", color=C["self"], lw=1.8)
    ax.axvline(1.0, color="#555555", ls="--", lw=1)
    ax.text(1.02, ax.get_ylim()[1] * 0.92, r"$R_{vir}$", fontsize=9,
            color="#555555")
    ax.set_xlabel(r"distance from host centre  $R / R_{vir}^{\mathrm{host}}$")
    ax.set_ylabel("fraction of subhalos")
    ax.set_title("The one gap: our subhalos sit too close to the host centre")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    _save(fig, out, "subhalo_radial_dist.png")


def fig_density_power(dp_path: Path, out: Path):
    """Density P(k) against HR's -- the physical power spectrum.

    Plotted as P_arm(k) vs P_HR(k) rather than a ratio: a perfect match is the
    y=x diagonal, so the full dynamic range of the spectrum is the reference
    and the small-scale excess reads as the mild departure it is. Note the
    x-axis runs from LOW P_HR (small scales, high k) to HIGH P_HR (large
    scales, low k); that direction is written along the diagonal and k
    markers sit on the tuned curve. Band ratios are printed for the slide.
    """
    import numpy as _np
    d = json.loads(Path(dp_path).read_text())
    arms = d["arms"]
    if "hr" not in arms:
        print("  (density_power json has no 'hr' reference, skipping)", flush=True)
        return
    khr = _np.array(arms["hr"]["k"]); phr = _np.array(arms["hr"]["Pk"])
    # One tuned arm only -- the talk shows "tuned", not the arm zoo. Prefer
    # smallhalo, falling back to whichever tuned arm is present.
    tuned = next((a for a in ("smallhalo", "selfhostguard", "self", "selfbound")
                  if a in arms), None)
    show = [("frozen", "#888888", "SR2 baseline", 2.0),
            (tuned, C["self"], "tuned", 2.4)]

    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    lo, hi = phr.min() * 0.6, phr.max() * 1.8
    # The diagonal is labelled in the legend, not inline: an inline rotated
    # label collides with the k markers at this aspect ratio.
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="#111111", zorder=1,
            label="HR (perfect match)")
    for a, cc, lab, lw in show:
        if a is None or a not in arms:
            continue
        k = _np.array(arms[a]["k"]); pk = _np.array(arms[a]["Pk"])
        if not _np.array_equal(k, khr):
            pk = _np.interp(khr, k, pk)
        ax.plot(phr, pk, "-", color=cc, lw=lw, label=lab, zorder=2)
        r = pk / phr
        print(f"  {a}: max P/P_HR {r.max():.3f} at k={khr[r.argmax()]:.1f}",
              flush=True)

    # k markers on the tuned curve, so "where are the small scales" is legible.
    if tuned is not None:
        ktun = _np.array(arms[tuned]["k"]); ptun = _np.array(arms[tuned]["Pk"])
        if not _np.array_equal(ktun, khr):
            ptun = _np.interp(khr, ktun, ptun)
        for kmark in (0.5, 2.0, 8.0, 24.0):
            i = int(_np.argmin(_np.abs(khr - kmark)))
            ax.plot(phr[i], ptun[i], "o", ms=5.5, color=C["self"],
                    mec="white", mew=1.0, zorder=3)
            ax.annotate(f"$k={kmark:g}$", (phr[i], ptun[i]),
                        textcoords="offset points", xytext=(-7, 9),
                        ha="right", fontsize=8.5, color="#333333")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    # Which way is high-k, written along the diagonal. Points sit a fixed
    # perpendicular offset into the empty wedge BELOW the line (both curves lie
    # on or above it): in log space that is (v*d, v/d) for a point (v,v).
    def _diag(frac, off):
        v = lo * (hi / lo) ** frac
        return v * off, v / off
    for frac, txt, ha in ((0.07, r"$\longleftarrow$  high $k$ (small scales)", "left"),
                          (0.93, r"low $k$ (large scales)  $\longrightarrow$", "right")):
        x, y = _diag(frac, 1.75)
        ax.text(x, y, txt, rotation=45, rotation_mode="anchor",
                ha=ha, va="center", fontsize=9.5, color="#555555")

    ax.set_xlabel(r"$P_\delta^{\mathrm{HR}}(k)$   (HR truth)")
    ax.set_ylabel(r"$P_\delta(k)$  (model)")
    ax.set_title("Density power spectrum vs HR (held-out set9)")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    fig.tight_layout()
    _save(fig, out, "density_power.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, help="pooled report.json")
    ap.add_argument("--forensics", nargs="*", default=[],
                    help="arm=path/to/host_forensics_<box>.json pairs")
    ap.add_argument("--radial-dir", default="",
                    help="subhalo_radial/<box> dir with hr.npz and self.npz")
    ap.add_argument("--density-power", default="",
                    help="density_power_<box>.json from density_power_spectrum.py")
    ap.add_argument("--tuned-arm", default="all_blocks_self",
                    help="arm plotted as 'tuned' in the mass-function bars")
    ap.add_argument("--out", required=True, help="figure output dir")
    a = ap.parse_args(argv)

    rep = json.loads(Path(a.report).read_text())
    fors: Dict[str, Dict] = {}
    for item in a.forensics:
        name, _, path = item.partition("=")
        p = Path(path)
        if p.exists():
            fors[name] = json.loads(p.read_text())
        else:
            print(f"  (forensic missing, skipping: {path})", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"rendering into {out}", flush=True)
    fig_abundance(rep, out)
    fig_sub_mass_function(rep, out, tuned=a.tuned_arm)
    fig_sub_mass_function(rep, out, tuned=a.tuned_arm, yscale="log")
    fig_occupation(rep, out)
    fig_occupation_selfonly(rep, out)
    fig_occupation_selfonly(rep, out, yscale="linear")
    # Same panel cropped to the mass range that actually holds resolved
    # substructure, in both scales -- the low-mass bins are the reason the
    # full-range linear version is unreadable.
    fig_occupation_selfonly(rep, out, mass_min=12.75)
    fig_occupation_selfonly(rep, out, yscale="linear", mass_min=12.75)
    fig_tradeoff(rep, out)
    if a.radial_dir:
        rd = Path(a.radial_dir)
        if (rd / "hr.npz").exists() and (rd / "self.npz").exists():
            fig_radial(rd, out)
        else:
            print(f"  (radial npz missing in {rd}, skipping)", flush=True)
    if a.density_power and Path(a.density_power).exists():
        fig_density_power(Path(a.density_power), out)
    if fors:
        fig_forensic(fors, out)
        fig_fate_by_mass(fors, out, arm="self")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
