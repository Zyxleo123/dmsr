#!/usr/bin/env python
"""Render the soft-membership solve from a dumped npz (see
``finetune_member_gather.py --dump-membership-npz``).

Two figures per host, both redrawn purely from the saved snapshots -- no field,
no model, no GPU:

* ``*_evolution.png`` -- the PROCESS. For the most massive candidate subhaloes,
  an x-y scatter of the pool particles coloured by membership ``P = w / mult``
  in [0, 1], one column per EM sweep (init, sweep1, ...). HR's own members are
  ringed; the window centre and its ``sigma_r`` circle are drawn. The title of
  each panel carries the running mass_ratio and hr_fraction, so the mass dual
  filling the window and the membership walking off HR's material are both
  visible as the sweeps advance.

* ``*_matrix.png`` -- the per-particle BUDGET. A particle x candidate heatmap of
  the final-sweep ``P`` for the candidates that share the most particles, plus a
  background column. Rows sum to <= 1: that is the budget two overlapping
  candidates compete under.

Everything is a pure function of the npz, so this runs on the CPU partition.
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import matplotlib.colors  # noqa: E402

# The simulation box, for the periodic image in the radial figure. A pool that
# straddles the box edge would otherwise read as ~L/2 from its host.
BOXSIZE_MPC_H = 100.0

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_SRC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "src")
if _os.path.isdir(_SRC) and _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)


def _load(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _set_mask(d: dict, s: int) -> np.ndarray:
    return d["setid"] == s


def _prob(d: dict, sweep: int) -> np.ndarray:
    """Membership probability P = w / mult per pool entry, clamped to [0, 1]."""
    mult = np.clip(d["mult"], 1e-12, None)
    return np.clip(d["sweep_w"][sweep] / mult, 0.0, 1.0)


def _final_hr_frac(d: dict) -> np.ndarray:
    """Per-set HR fraction at the final sweep: kept-on-members / total mass."""
    mass = d["sweep_mass"][-1]
    kept = d["sweep_hr_kept"][-1]
    return kept / np.clip(mass, 1e-9, None)


def _pick_candidates(d: dict, n: int, mode: str = "spread") -> List[int]:
    """Which candidate sets to draw.

    ``mass``         the n most massive -- but the biggest subhaloes keep most
                     of their HR material, so they hide the substitution.
    ``substitution`` the n with the LOWEST final hr_fraction among sets big
                     enough to read (>= 300p) -- the objects most rebuilt from
                     non-member material, which is the whole point.
    ``spread``       (default) n sets evenly spaced across the size distribution,
                     so a big, a medium and a small one appear together.
    """
    order = np.argsort(-d["target_mass"])           # big -> small
    if mode == "mass":
        return [int(s) for s in order[:n]]
    if mode == "substitution":
        big = [s for s in order if float(d["target_mass"][s]) >= 300.0]
        pool = big or list(order)
        hrf = _final_hr_frac(d)
        return [int(s) for s in sorted(pool, key=lambda s: hrf[s])[:n]]
    # spread: evenly spaced ranks over the sorted-by-mass list
    idx = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [int(order[i]) for i in idx]


def figure_evolution(d: dict, cands: List[int], out: str) -> None:
    tags = [str(t) for t in d["sweep_tags"]]
    n_sw = len(tags)
    n_c = len(cands)
    fig, axes = plt.subplots(n_c, n_sw, figsize=(3.0 * n_sw, 3.0 * n_c),
                             squeeze=False)
    cmap = plt.get_cmap("inferno")
    rng = np.random.default_rng(0)
    cap = 6000                      # subsample dense sets so points don't merge
    for ri, s in enumerate(cands):
        m = _set_mask(d, s)
        cref = d["centre_ref"][s]
        xy_all = d["x"][m][:, :2] - cref[:2]
        mem_all = d["is_member"][m]
        # One shared subsample across sweeps, so a particle is in the same place
        # in every column and only its colour (P) changes.
        n_pts = xy_all.shape[0]
        sel = (rng.choice(n_pts, size=cap, replace=False) if n_pts > cap
               else np.arange(n_pts))
        xy, mem = xy_all[sel], mem_all[sel]
        tgt = float(d["target_mass"][s])
        for ci in range(n_sw):
            ax = axes[ri][ci]
            p = _prob(d, ci)[m][sel]
            # Members drawn first as a faint grey underlay -- they mark HR's own
            # material without masking the P colours, which sit ON TOP. Recruited
            # (non-member) mass is then whatever is bright but has no grey ring.
            if mem.any():
                ax.scatter(xy[mem, 0], xy[mem, 1], s=16, facecolors="none",
                           edgecolors="#8a8a8a", linewidths=0.25, alpha=0.35)
            # Faint-to-bright by P, so the brightest (most-claimed) draw on top.
            o = np.argsort(p)
            ax.scatter(xy[o, 0], xy[o, 1], c=p[o], cmap=cmap, vmin=0.0, vmax=1.0,
                       s=5, linewidths=0.0)
            # The window this sweep: centre (relative to centre_ref) and sigma_r.
            c_rel = d["sweep_c"][ci][s][:2] - cref[:2]
            ax.plot(c_rel[0], c_rel[1], "x", color="cyan", ms=7, mew=1.5)
            ax.add_patch(Circle((c_rel[0], c_rel[1]), float(d["sigma_r"][s]),
                                fill=False, ec="cyan", lw=0.7, ls="--", alpha=0.7))
            mr = float(d["sweep_mass"][ci][s]) / max(tgt, 1e-9)
            hrf = float(d["sweep_hr_kept"][ci][s]) / max(float(d["sweep_mass"][ci][s]), 1e-9)
            ax.set_title(f"{tags[ci]}\nmass/tgt={mr:.2f}  hr_frac={hrf:.2f}",
                         fontsize=8)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(f"halo {int(d['halo_id'][s])}\ntgt={tgt:.0f}p",
                              fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="membership P = w / mult",
                 location="right", pad=0.01)
    fig.suptitle(f"membership solve on {str(d['host_key'])}  "
                 f"(prior lambda={float(d['lam']):.3g}) -- rows: candidates, "
                 f"cols: EM sweeps", fontsize=11)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_matrix(d: dict, n_cands: int, max_rows: int, out: str) -> None:
    # The candidates that actually compete: rank sets by how many of their pool
    # rows are shared with another set. A budget plot of disjoint sets is empty.
    setid, rows = d["setid"], d["rows"]
    from collections import defaultdict
    row_sets = defaultdict(set)
    for r, s in zip(rows.tolist(), setid.tolist()):
        row_sets[r].add(s)
    shared_count = defaultdict(int)
    for r, ss in row_sets.items():
        if len(ss) > 1:
            for s in ss:
                shared_count[s] += 1
    if shared_count:
        cands = [s for s, _ in sorted(shared_count.items(),
                                      key=lambda kv: -kv[1])[:n_cands]]
    else:
        cands = _pick_candidates(d, n_cands)
        print("  (no candidates share particles; showing the top-mass sets)")
    cands = sorted(cands)
    cand_col = {s: i for i, s in enumerate(cands)}

    # Union of the chosen candidates' pool particles.
    keep = np.isin(setid, cands)
    urows = np.unique(rows[keep])
    row_idx = {int(r): i for i, r in enumerate(urows.tolist())}
    P = _prob(d, len(d["sweep_tags"]) - 1)   # final sweep
    M = np.zeros((len(urows), len(cands) + 1), dtype=np.float32)
    for r, s, p in zip(rows[keep].tolist(), setid[keep].tolist(), P[keep].tolist()):
        M[row_idx[r], cand_col[s]] = p
    M[:, -1] = np.clip(1.0 - M[:, :-1].sum(1), 0.0, 1.0)   # background column

    # Sort rows: by owning candidate (argmax over candidates), then by that P.
    owner = M[:, :-1].argmax(1)
    strength = M[np.arange(len(M)), owner]
    order = np.lexsort((-strength, owner))
    M = M[order]
    if len(M) > max_rows:   # subsample rows for a readable image, keep the sort
        sel = np.linspace(0, len(M) - 1, max_rows).astype(int)
        M = M[sel]

    fig, ax = plt.subplots(figsize=(1.1 * (len(cands) + 1) + 2, 8))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(range(len(cands) + 1))
    ax.set_xticklabels([f"halo\n{int(d['halo_id'][s])}" for s in cands] + ["bg"],
                       fontsize=7)
    ax.set_xlabel("candidate")
    ax.set_ylabel(f"pool particle (sorted; {M.shape[0]} of {len(urows)} shown)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="P (final sweep)")
    ax.set_title(f"per-particle budget on {str(d['host_key'])}\n"
                 f"rows sum <= 1 across candidates + background", fontsize=10)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def figure_heatmap(d: dict, out: str, half_width: float = 10.0) -> None:
    """Region-scale heatmap of the solve, with the GENERATED subhalo centres
    overlaid, in two projections (x-y, x-z).

    Columns: (1) pool particle count, (2) assigned membership weight ``sum P``,
    (3) mean membership ``<P>`` per bin -- all cropped to a cube of side
    ``2*half_width`` Mpc/h centred on the host. On every panel the FINAL-sweep
    window centres ``sweep_c[-1]`` are drawn as cyan dots sized by log mass, and
    the host centre as a magenta x.

    The question this answers: do the centres the solve places sit on the bright
    (high assigned-weight) knots? They do -- but that is a REGION-scale, per-knot
    statement (each centre lands on a density peak of the cosmic web), and says
    nothing about where a subhalo sits inside its OWN host in Rvir units, which
    is the radial-placement metric. The two centres differ: this crop is one
    reference point ~10 Mpc across; radial placement is a sub-Mpc, per-host
    quantity invisible here.
    """
    x = d["x"]
    P = _prob(d, len(d["sweep_tags"]) - 1)          # final sweep P per pool entry
    hc = np.asarray(d["centre_host"])[0]            # single host reference point
    sc = d["sweep_c"][-1]                            # final subhalo centres
    mass = d["sweep_mass"][-1]
    nb = 260
    # marker size ~ log mass, so a 5e4p host reads as bigger than a 200p sub.
    msz = 6.0 + 22.0 * (np.log10(np.clip(mass, 1.0, None))
                        - 2.0) / max(np.log10(mass.max()) - 2.0, 1e-6)

    projs = [(0, 1, "x", "y"), (0, 2, "x", "z")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), squeeze=False)
    for ri, (a, b, la, lb) in enumerate(projs):
        rng = [[hc[a] - half_width, hc[a] + half_width],
               [hc[b] - half_width, hc[b] + half_width]]
        cnt, xe, ye = np.histogram2d(x[:, a], x[:, b], bins=nb, range=rng)
        wsum, _, _ = np.histogram2d(x[:, a], x[:, b], bins=nb, range=rng,
                                    weights=P)
        mean = np.divide(wsum, cnt, out=np.full_like(wsum, np.nan), where=cnt > 0)
        ext = [xe[0], xe[-1], ye[0], ye[-1]]
        panels = [
            (np.log10(cnt.T + 1.0), "viridis", "pool particle count (log)", None, None),
            (np.log10(wsum.T + 1e-3), "inferno", r"membership weight  $\Sigma P$ (log)", None, None),
            (mean.T, "magma", r"mean membership  $\langle P\rangle$ per bin", 0.0, 1.0),
        ]
        for ci, (img, cmap, title, vmin, vmax) in enumerate(panels):
            ax = axes[ri][ci]
            im = ax.imshow(img, origin="lower", extent=ext, aspect="equal",
                           cmap=cmap, vmin=vmin, vmax=vmax)
            # generated subhalo centres, on top of the field
            ax.scatter(sc[:, a], sc[:, b], s=msz, facecolors="none",
                       edgecolors="cyan", linewidths=0.6, alpha=0.9)
            ax.plot(hc[a], hc[b], "x", color="magenta", ms=12, mew=2.5)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xlabel(f"{la} [Mpc/h]"); ax.set_ylabel(f"{lb} [Mpc/h]")
            if ri == 0:
                ax.set_title(title, fontsize=11)
            fig.colorbar(im, ax=ax, shrink=0.8)
    n_in = int(((np.abs(sc[:, 0] - hc[0]) < half_width) &
                (np.abs(sc[:, 1] - hc[1]) < half_width) &
                (np.abs(sc[:, 2] - hc[2]) < half_width)).sum())
    fig.suptitle(
        f"Membership-weight heatmap -- {str(d['host_key'])} coarse_soft "
        f"(final sweep, {int(d['n_sets'])} subhalos, {x.shape[0]:,} pool entries)\n"
        f"cyan = generated subhalo centres ({n_in}/{int(d['n_sets'])} in this "
        f"+/-{half_width:g} Mpc/h crop); magenta x = host centre",
        fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def host_rvir_mpc_h(d: dict, explicit: float = 0.0, ascii_path: str = "") -> float:
    """The parent host's R_vir in Mpc/h, or 0.0 if it cannot be established.

    Three sources, in order: an explicit ``--host-rvir``; the npz's own
    ``host_rvir`` (nonzero only when the dump ran with ``--soft-host-profile``);
    a scan of the HR Rockstar ascii for the id in ``host_key`` (``set9:h165513``),
    whose column 5 is R_vir in **kpc/h**. Zero means the caller must fall back to
    an absolute Mpc/h axis rather than invent a scale.
    """
    if explicit > 0.0:
        return float(explicit)
    hr = np.asarray(d.get("host_rvir", np.zeros(1)), dtype=np.float64)
    if hr.size and float(np.max(hr)) > 0.0:
        return float(np.max(hr))
    if ascii_path and os.path.isfile(ascii_path):
        key = str(d["host_key"])
        hid = key.split(":h")[-1] if ":h" in key else ""
        if hid.isdigit():
            with open(ascii_path) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    f = line.split(None, 5)
                    if f and f[0] == hid:
                        return float(f[4]) / 1000.0     # kpc/h -> Mpc/h
    return 0.0


def _max_membership_per_particle(d: dict):
    """Per distinct particle: its largest membership over all non-background
    candidates, the winning candidate, and its radius from that candidate's own
    parent host.

    ``rows`` repeats a particle once per candidate pool it lands in, so the
    per-particle quantity is a segmented max over ``rows``. ``P = 1`` means the
    particle is wholly some subhalo's; ``P = 0`` that the background (the host)
    keeps all of it. The radius is measured against ``centre_host`` of the
    **winning** candidate -- sets are tile-selected and need not share a parent
    -- with the periodic image taken, because a pool straddling the box edge
    would otherwise read as ~L/2 away.
    """
    rows = np.asarray(d["rows"])
    P = _prob(d, len(d["sweep_tags"]) - 1)
    order = np.lexsort((-P, rows))                   # each row's best first
    rs, Ps, ss = rows[order], P[order], np.asarray(d["setid"])[order]
    first = np.ones(rs.shape[0], dtype=bool)
    first[1:] = rs[1:] != rs[:-1]
    keep = np.flatnonzero(first)
    ent = order[keep]                                # pool entry of each winner
    x = np.asarray(d["x"], dtype=np.float64)[ent]
    hc = np.asarray(d["centre_host"], dtype=np.float64)[ss[keep]]
    dx = x - hc
    dx -= BOXSIZE_MPC_H * np.round(dx / BOXSIZE_MPC_H)
    return Ps[keep], np.linalg.norm(dx, axis=1), ent


def figure_radial(d: dict, out: str, rvir: float = 0.0,
                  r_max_rvir: float = 4.0) -> None:
    """How subhalo-like a particle is, against how far out it sits.

    Left: the joint distribution of (radius, max non-background membership) over
    every distinct pool particle, with the median membership per radial bin.
    Right: where the *assigned* mass comes from -- membership-weighted counts per
    radial bin, split into HR's own named members and everything else. The gap
    between the two curves is the material the solve recruited that HR never
    named, and its radial extent is the point: if the right panel's ``other``
    histogram reaches past ~1 R_vir, the relaxation is drawing on outskirt
    particles rather than re-shuffling the core.
    """
    P, r, ent = _max_membership_per_particle(d)
    member = np.asarray(d["is_member"], dtype=bool)[ent]
    unit = "$r / R_{\\rm vir}$" if rvir > 0 else "$r$ [Mpc/h]"
    rr = r / rvir if rvir > 0 else r
    hi = r_max_rvir if rvir > 0 else r_max_rvir * 1.5
    sel = rr <= hi
    bins = np.linspace(0.0, hi, 41)
    ctr = 0.5 * (bins[1:] + bins[:-1])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    h = ax.hist2d(rr[sel], P[sel], bins=[bins, np.linspace(0, 1, 41)],
                  cmap="magma", norm=matplotlib.colors.LogNorm())
    idx = np.clip(np.digitize(rr, bins) - 1, 0, len(ctr) - 1)
    med = np.array([np.median(P[idx == k]) if np.any(idx == k) else np.nan
                    for k in range(len(ctr))])
    ax.plot(ctr, med, color="cyan", lw=2.0, label="median")
    if rvir > 0:
        ax.axvline(1.0, color="w", ls="--", lw=1.2, label="$R_{\\rm vir}$")
    ax.set_xlabel(unit)
    ax.set_ylabel(r"$\max_j P_{ij}$  (non-background)")
    ax.set_title("how subhalo-like a particle is, vs radius", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(h[3], ax=ax, shrink=0.85, label="particles")

    ax = axes[1]
    wm, _ = np.histogram(rr[member], bins=bins, weights=P[member])
    wo, _ = np.histogram(rr[~member], bins=bins, weights=P[~member])
    ax.step(ctr, wm, where="mid", color="tab:blue", lw=1.8,
            label="HR's own members")
    ax.step(ctr, wo, where="mid", color="tab:red", lw=1.8,
            label="recruited (not HR's)")
    ax.fill_between(ctr, 0, wo, step="mid", color="tab:red", alpha=0.18)
    if rvir > 0:
        ax.axvline(1.0, color="k", ls="--", lw=1.2)
    ax.set_xlabel(unit)
    ax.set_ylabel(r"assigned mass  $\Sigma \max_j P_{ij}$")
    ax.set_title("where the chosen material comes from", fontsize=11)
    ax.legend(fontsize=8)

    tot = float(P.sum())
    out_frac = float(P[rr > 1.0].sum()) / max(tot, 1e-9) if rvir > 0 else float("nan")
    hrf = float(P[member].sum()) / max(tot, 1e-9)
    fig.suptitle(
        f"Membership vs radius -- {str(d['host_key'])} "
        f"(final sweep, {int(d['n_sets'])} subhalos, {P.size:,} particles; "
        f"HR share {hrf:.2f}"
        + (f", {out_frac:.2f} of assigned mass beyond 1 $R_{{\\rm vir}}$"
           if rvir > 0 else "") + ")",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  (median P {np.nanmedian(med):.3f}, HR share {hrf:.3f})")


def figure_heatmap_panel(d: dict, out: str, half_width: float = 10.0) -> None:
    """The mean-membership x-z panel of ``figure_heatmap`` alone, at slide size.

    Same bins, same crop, same overlays as the bottom-right panel of the 2x3
    figure -- this renders it from the npz rather than cropping the PNG, so it
    keeps full resolution and gets its own colourbar and title.
    """
    x = d["x"]
    P = _prob(d, len(d["sweep_tags"]) - 1)
    hc = np.asarray(d["centre_host"])[0]
    sc = d["sweep_c"][-1]
    mass = d["sweep_mass"][-1]
    nb = 260
    msz = 6.0 + 22.0 * (np.log10(np.clip(mass, 1.0, None))
                        - 2.0) / max(np.log10(mass.max()) - 2.0, 1e-6)
    a, b, la, lb = 0, 2, "x", "z"
    rng = [[hc[a] - half_width, hc[a] + half_width],
           [hc[b] - half_width, hc[b] + half_width]]
    cnt, xe, ye = np.histogram2d(x[:, a], x[:, b], bins=nb, range=rng)
    wsum, _, _ = np.histogram2d(x[:, a], x[:, b], bins=nb, range=rng, weights=P)
    mean = np.divide(wsum, cnt, out=np.full_like(wsum, np.nan), where=cnt > 0)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(mean.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   aspect="equal", cmap="magma", vmin=0.0, vmax=1.0)
    ax.scatter(sc[:, a], sc[:, b], s=msz, facecolors="none", edgecolors="cyan",
               linewidths=0.6, alpha=0.9)
    ax.plot(hc[a], hc[b], "x", color="magenta", ms=12, mew=2.5)
    ax.set_xlabel(f"{la} [Mpc/h]")
    ax.set_ylabel(f"{lb} [Mpc/h]")
    ax.set_title(r"mean membership $\langle P\rangle$ per bin"
                 f"  --  {str(d['host_key'])}", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

def _hr_parent_radii(d: dict, ascii_path: str):
    """Per candidate: the radius of its HR reference centre and its optimized
    centre ``c_j``, both against the candidate's OWN HR parent host in R_vir.

    The pooled ``centre_host`` is a single reference point for a whole tile
    region (~20 Mpc/h of many hosts), so a radius against it is dominated by
    inter-host spacing, not satellite placement. The faithful parent is the one
    Rockstar assigned: match the npz's ``halo_id`` to its row in the HR ascii,
    follow ``parent_ids`` to the host, and normalise by that host's R_vir --
    exactly the rule ``subhalo_radial.collect`` uses for the gate, so the two
    are comparable. Candidates that are hosts themselves (no parent) are
    dropped. Returns ``(r_ref, r_opt, n_matched, n_total)``.
    """
    from cosmo_sr.eval.rockstar import load_rockstar_ascii
    cat = load_rockstar_ascii(ascii_path)
    id2row = {int(h): i for i, h in enumerate(cat.ids)}
    ref = np.asarray(d["centre_ref"], dtype=np.float64)
    copt = np.asarray(d["sweep_c"], dtype=np.float64)[-1]
    hid = np.asarray(d["halo_id"])

    def mind(a, b):
        e = a - b
        e -= BOXSIZE_MPC_H * np.round(e / BOXSIZE_MPC_H)
        return np.linalg.norm(e)

    r_ref, r_opt = [], []
    for j, h in enumerate(hid):
        row = id2row.get(int(h))
        if row is None:
            continue
        pid = int(cat.parent_ids[row])
        prow = id2row.get(pid) if pid >= 0 else None
        if prow is None:
            continue
        rvir = float(cat.rvir[prow]) / 1000.0
        if rvir <= 0:
            continue
        hostpos = cat.pos[prow]
        r_ref.append(mind(ref[j], hostpos) / rvir)
        r_opt.append(mind(copt[j], hostpos) / rvir)
    return (np.asarray(r_ref), np.asarray(r_opt), len(r_ref), int(hid.size))


def figure_centre_radial(d: dict, out: str, ascii_path: str,
                         sr_median: float = float("nan"),
                         r_max_rvir: float = 4.0) -> None:
    """Where the solve AIMS, radially: the distribution of optimized candidate
    centres ``c_j`` against each candidate's own HR parent, overlaid on HR's own
    subhalo placement.

    The hypothesis this tests: the outer targets exist -- the solve places
    ``c_j`` at HR-like radii -- but the generator does not condense detectable
    subhalos there, so the Rockstar-found SR subhalos sit far inside. If the
    ``c_j`` histogram tracks the HR one while ``sr_median`` (the gate's measured
    SR subhalo median, passed in) sits well inside both, the placement defect is
    downstream of the membership target, not in it.
    """
    r_ref, r_opt, n, ntot = _hr_parent_radii(d, ascii_path)
    if n == 0:
        raise SystemExit(f"no candidate matched an HR parent in {ascii_path}")
    bins = np.linspace(0.0, r_max_rvir, 33)
    # medians are reported on the FULL sample; only the histogram is windowed,
    # and beyond-range candidates are dropped (not clipped into an edge spike).
    n_out = int((r_opt > r_max_rvir).sum())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(r_ref[r_ref <= r_max_rvir], bins=bins, histtype="step",
            color="tab:blue", lw=2.0, density=True,
            label=f"HR subhalos  (median {np.median(r_ref):.2f})")
    ax.hist(r_opt[r_opt <= r_max_rvir], bins=bins, histtype="stepfilled",
            color="tab:red", lw=2.0, density=True, alpha=0.30,
            label=(f"solve targets $c_j$  (median {np.median(r_opt):.2f}"
                   + (f", {n_out} past {r_max_rvir:g})" if n_out else ")")))
    ax.axvline(np.median(r_ref), color="tab:blue", ls="--", lw=1.3)
    ax.axvline(np.median(r_opt), color="tab:red", ls="--", lw=1.3)
    if np.isfinite(sr_median):
        ax.axvline(sr_median, color="k", ls=":", lw=1.8,
                   label=f"SR subhalos found (median {sr_median:.2f})")
    ax.set_xlabel(r"$r\ /\ R_{\rm vir}$  (from the candidate's own HR parent)")
    ax.set_ylabel("density")
    ax.set_title(f"where the solve aims vs where subhalos land -- "
                 f"{str(d['host_key'])} ({n}/{ntot} matched)", fontsize=11)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  (c_j median {np.median(r_opt):.3f} Rvir, "
          f"HR {np.median(r_ref):.3f}, matched {n}/{ntot})")

# radial bin edges in R_vir, matching subhalo_radial.RVIR_BINS plus an overflow
CENTRE_RADIAL_BINS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])


def figure_centre_radial_bars(d: dict, out: str, ascii_path: str,
                              sr_median: float = float("nan")) -> None:
    """Grouped bar chart: fraction of candidates per radial range, HR reference
    centres vs optimized ``c_j``. Same parent-matched radii as
    ``figure_centre_radial``, binned instead of a histogram outline."""
    r_ref, r_opt, n, ntot = _hr_parent_radii(d, ascii_path)
    if n == 0:
        raise SystemExit(f"no candidate matched an HR parent in {ascii_path}")
    edges = CENTRE_RADIAL_BINS
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f">{edges[-1]:g}")
    ext = np.append(edges, np.inf)

    def frac(r):
        c, _ = np.histogram(r, bins=ext)
        return c / max(c.sum(), 1)

    fr_ref, fr_opt = frac(r_ref), frac(r_opt)
    xpos = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(xpos - w / 2, fr_ref, w, color="tab:blue",
           label=f"HR subhalos $c_{{\\rm ref}}$  (median {np.median(r_ref):.2f})")
    ax.bar(xpos + w / 2, fr_opt, w, color="tab:red", alpha=0.75,
           label=f"solve targets $c_j$  (median {np.median(r_opt):.2f})")
    if np.isfinite(sr_median):
        # place the line on the categorical axis: bin index + fractional
        # position within that bin, with a bar centred at integer k spanning
        # [k-0.5, k+0.5].
        k = int(np.searchsorted(edges, sr_median, side="right") - 1)
        k = max(0, min(k, len(labels) - 1))
        if k < len(edges) - 1:
            fr = (sr_median - edges[k]) / (edges[k + 1] - edges[k])
        else:
            fr = 0.5
        xs = k - 0.5 + fr
        ax.axvline(xs, color="k", ls=":", lw=1.8,
                   label=f"SR subhalos found (median {sr_median:.2f})")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$r\ /\ R_{\rm vir}$  (from the candidate's own HR parent)")
    ax.set_ylabel("fraction of candidates")
    ax.set_title(f"radial placement by range -- {str(d['host_key'])} "
                 f"({n}/{ntot} matched)", fontsize=11)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  (c_j median {np.median(r_opt):.3f} Rvir, "
          f"HR {np.median(r_ref):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", help="dumped membership npz file(s)")
    ap.add_argument("--out-dir", default="", help="where the PNGs go "
                    "(default: alongside each npz)")
    ap.add_argument("--n-evolution", type=int, default=3,
                    help="how many candidate subhaloes (rows) in the evolution figure")
    ap.add_argument("--n-matrix", type=int, default=6,
                    help="how many competing candidates (columns) in the budget matrix")
    ap.add_argument("--max-matrix-rows", type=int, default=400)
    ap.add_argument("--heatmap", action="store_true",
                    help="also render the region-scale weight heatmap with the "
                    "generated subhalo centres overlaid")
    ap.add_argument("--heatmap-only", action="store_true",
                    help="render ONLY the heatmap (skip evolution/matrix)")
    ap.add_argument("--heatmap-halfwidth", type=float, default=10.0,
                    help="crop half-width in Mpc/h around the host centre")
    ap.add_argument("--radial", action="store_true",
                    help="also render membership vs host-centric radius")
    ap.add_argument("--radial-only", action="store_true",
                    help="render ONLY the radial figure")
    ap.add_argument("--host-rvir", type=float, default=0.0,
                    help="parent host R_vir in Mpc/h for the radial axis; "
                    "0 = take it from the npz, then from --rockstar-ascii, "
                    "then fall back to an absolute Mpc/h axis")
    ap.add_argument("--rockstar-ascii", default="",
                    help="HR Rockstar halos_0.0.ascii to look R_vir up in "
                    "(column 5, kpc/h) by the id in the npz's host_key")
    ap.add_argument("--r-max-rvir", type=float, default=4.0,
                    help="outer edge of the radial axis, in R_vir")
    ap.add_argument("--centre-radial", action="store_true",
                    help="also render the radial distribution of optimized "
                    "candidate centres c_j vs HR, against each candidate's own "
                    "HR parent (needs --rockstar-ascii)")
    ap.add_argument("--centre-radial-only", action="store_true",
                    help="render ONLY that figure")
    ap.add_argument("--centre-radial-bars", action="store_true",
                    help="grouped bar chart of c_j vs HR per radial range")
    ap.add_argument("--centre-radial-bars-only", action="store_true",
                    help="render ONLY the bar chart")
    ap.add_argument("--sr-subhalo-median", type=float, default=float("nan"),
                    help="the gate's measured SR subhalo median r/Rvir, drawn "
                    "as a reference line on the centre-radial figure")
    ap.add_argument("--panel", action="store_true",
                    help="also render the mean-membership x-z panel alone "
                    "(the bottom-right panel of --heatmap, at slide size)")
    ap.add_argument("--panel-only", action="store_true",
                    help="render ONLY that single panel")
    ap.add_argument("--select", default="spread",
                    choices=("spread", "mass", "substitution"),
                    help="which candidate subhaloes the evolution figure shows: "
                    "spread across sizes (default), the most massive, or the "
                    "most rebuilt from non-member material (lowest hr_fraction)")
    args = ap.parse_args()

    for path in args.npz:
        d = _load(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        odir = args.out_dir or os.path.dirname(os.path.abspath(path))
        os.makedirs(odir, exist_ok=True)
        print(f"{path}: {int(d['n_sets'])} sets, {d['x'].shape[0]} pool entries, "
              f"{len(d['sweep_tags'])} snapshots")
        if args.radial or args.radial_only:
            rv = host_rvir_mpc_h(d, args.host_rvir, args.rockstar_ascii)
            if rv <= 0.0:
                print("  (no host R_vir: plotting an absolute Mpc/h axis)")
            figure_radial(d, os.path.join(odir, f"{stem}_radial.png"),
                          rvir=rv, r_max_rvir=args.r_max_rvir)
        if args.centre_radial or args.centre_radial_only:
            asc = args.rockstar_ascii
            if not asc:
                raise SystemExit("--centre-radial needs --rockstar-ascii "
                                 "(the HR halos_0.0.ascii for parent lookup)")
            figure_centre_radial(
                d, os.path.join(odir, f"{stem}_centre_radial.png"), asc,
                sr_median=args.sr_subhalo_median, r_max_rvir=args.r_max_rvir)
        if args.centre_radial_bars or args.centre_radial_bars_only:
            asc = args.rockstar_ascii
            if not asc:
                raise SystemExit("--centre-radial-bars needs --rockstar-ascii")
            figure_centre_radial_bars(
                d, os.path.join(odir, f"{stem}_centre_radial_bars.png"), asc,
                sr_median=args.sr_subhalo_median)
        if args.panel or args.panel_only:
            figure_heatmap_panel(
                d, os.path.join(odir, f"{stem}_meanP_xz.png"),
                half_width=args.heatmap_halfwidth)
        if (args.radial_only or args.panel_only or args.centre_radial_only
                or args.centre_radial_bars_only):
            continue
        if args.heatmap or args.heatmap_only:
            figure_heatmap(d, os.path.join(odir, f"{stem}_heatmap_centres.png"),
                           half_width=args.heatmap_halfwidth)
        if args.heatmap_only:
            continue
        cands = _pick_candidates(d, min(args.n_evolution, int(d["n_sets"])),
                                 mode=args.select)
        figure_evolution(d, cands, os.path.join(odir, f"{stem}_evolution.png"))
        figure_matrix(d, min(args.n_matrix, int(d["n_sets"])),
                      args.max_matrix_rows,
                      os.path.join(odir, f"{stem}_matrix.png"))


if __name__ == "__main__":
    main()
