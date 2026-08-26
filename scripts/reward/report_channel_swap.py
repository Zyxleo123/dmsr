#!/usr/bin/env python
"""Read the four channel-swap catalogs and say which half of the field is at fault.

Pure aggregation: it opens catalogs that already exist and writes a JSON plus a
markdown table. No halo finding, no fields, no GPU -- so the verdict can be
re-rendered after a definition change without paying for Rockstar again.

Arms (see `scripts/reward/channel_swap_rockstar.py` for how they are built):

    hr           HR displacement  + HR velocity   (frozen control)
    base         SR2 displacement + SR2 velocity  (frozen control)
    srpos_hrvel  SR2 displacement + HR velocity
    hrpos_srvel  HR displacement  + SR2 velocity

The scalar that decides it is the **recovery fraction** of an arm's subhalo
count, `(N_arm - N_base) / (N_hr - N_base)`: 0 means the arm sits on SR2, 1 means
it sits on HR. Read `srpos_hrvel` first -- it is the arm that adds the correct
velocities to SR2's own positions, so its recovery fraction is the share of the
deficit the velocity channels are responsible for. `hrpos_srvel` is the mirror
test and should land near `1 - recovery(srpos_hrvel)` if the two channel groups
carry independent shares; if both arms recover a lot, the two halves interact and
neither alone is the story.

Two guards against reading the wrong thing into a moved count:

* **Occupancy** (bound particles / 512^3) is printed beside every count. A swap
  changes which particles Rockstar unbinds -- cold velocities under-unbind, hot
  ones over-unbind -- so an arm that gained subhalos while shedding bound mass
  did not necessarily gain substructure.
* **Counts by `num_p`** are reported alongside the total, because the deficit is
  strongly size-dependent (SR2/HR is 0.31 at 20-50 particles and 0.86 above 500).
  An intervention that only moves the 20-50 bin is doing something different from
  one that lifts every bin.

    python scripts/reward/report_channel_swap.py --box set8
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (  # noqa: E402
    add_common_args, banner, load_reward_config, paths, write_json,
)

ARM_ORDER = ("hr", "base", "srpos_hrvel", "hrpos_srvel")
ARM_LABEL = {
    "hr": "HR (control)",
    "base": "SR2 (control)",
    "srpos_hrvel": "SR2 pos + HR vel",
    "hrpos_srvel": "HR pos + SR2 vel",
}
NP_BINS = ((20, 50), (50, 100), (100, 200), (200, 500), (500, 1 << 62))
MASS_BINS = tuple(np.arange(11.0, 15.0, 0.5))
G_KPC = 4.30091e-6      # kpc (km/s)^2 / Msun -- Rockstar's rvir is kpc/h


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/reward/test_channel_swap.py)
# --------------------------------------------------------------------------

def recovery_fraction(n_arm: float, n_base: float, n_hr: float):
    """Where an arm's count sits on the SR2 -> HR line. ``None`` if they tie.

    0.0 = indistinguishable from SR2, 1.0 = indistinguishable from HR. Values
    outside [0, 1] are meaningful and deliberately not clipped: an arm that
    overshoots HR or falls below SR2 is a real result about the intervention,
    not a number to tidy away.
    """
    span = float(n_hr) - float(n_base)
    if span == 0.0:
        return None
    return (float(n_arm) - float(n_base)) / span


def counts_by_num_p(num_p, is_sub, bins=NP_BINS) -> dict:
    """``{"20-50": {"hosts": h, "subs": s}, ...}`` for one catalog."""
    num_p = np.asarray(num_p, dtype=np.int64)
    is_sub = np.asarray(is_sub, dtype=bool)
    out = {}
    for lo, hi in bins:
        m = (num_p >= lo) & (num_p < hi)
        label = f"{lo}-{hi}" if hi < (1 << 61) else f"{lo}+"
        out[label] = {"hosts": int((m & ~is_sub).sum()),
                      "subs": int((m & is_sub).sum())}
    return out


def parent_ids_from_columns(ids, idx, i_so) -> np.ndarray:
    """Rockstar's internal ``i_so`` remapped to printed halo ids, vectorised.

    This is `cosmo_sr.eval.rockstar.load_rockstar_ascii`'s remap, reproduced here
    so the whole catalog can be parsed once with pandas (which
    ``load_rockstar_ascii`` cannot do -- it drops the ``vrms``/``Rs``/``Xoff``/
    ``T/|U|`` columns this report lives on) without the two disagreeing about
    what a subhalo is.

    The part that matters, and the reason this is tested rather than inlined:
    ``i_so >= 0`` is **not** the same test as "is a subhalo". An object can name
    a parent index that the catalog never prints, and the project's convention
    counts that orphan as a host. On set8 the two definitions differ by 332
    objects in HR and 993 in SR2 -- around 2% of the SR2 subhalo count, which is
    the same order as the effects this experiment is trying to measure.
    """
    ids = np.asarray(ids, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    i_so = np.asarray(i_so, dtype=np.int64)
    if not (ids.shape == idx.shape == i_so.shape):
        raise ValueError("ids, idx and i_so must have the same length")
    if ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    valid_idx = idx[idx >= 0]
    lut = np.full(int(valid_idx.max()) + 1 if valid_idx.size else 1, -1, np.int64)
    lut[idx[idx >= 0]] = ids[idx >= 0]
    out = np.full(ids.size, -1, dtype=np.int64)
    known = (i_so >= 0) & (i_so < lut.size)
    out[known] = lut[i_so[known]]      # an unprinted parent stays -1 via the lut
    return out


def structure_by_mass(df, mass_bins=MASS_BINS, min_objects: int = 20) -> dict:
    """Median internal structure of the *hosts* in each ``log Mvir`` bin.

    The four columns are the ones that separate "the density profile is wrong"
    from "the dynamics are wrong":

    ``c``      ``rvir / Rs``, NFW concentration -- the density-side control.
    ``vmax_over_vvir``  peak circular speed over the virial speed.
    ``vrms``   member velocity dispersion, km/s -- compared at *fixed* Mvir, and
               rvir is fixed by the SO definition once Mvir is, so this is a
               structural statement and not a mass-definition one.
    ``xoff``   density-peak-to-centre-of-mass offset over rvir; the standard
               relaxation indicator.
    ``t_over_u``  kinetic over potential energy. The virial theorem puts an
               equilibrium system at 0.5, so this one has an absolute meaning
               rather than only a relative one.

    Bins with fewer than ``min_objects`` hosts are dropped: Rockstar's NFW fit is
    noisy at low particle counts and a median over a handful of them would invite
    exactly the over-reading this experiment exists to avoid.
    """
    hosts = df[~df["is_sub"]]
    logm = np.log10(np.clip(hosts["mvir"].to_numpy(), 1.0, None))
    out = {}
    for lo in mass_bins:
        m = (logm >= lo) & (logm < lo + 0.5)
        n = int(m.sum())
        if n < min_objects:
            continue
        h = hosts[m]
        rvir = h["rvir"].to_numpy()
        mvir = h["mvir"].to_numpy()
        vvir = np.sqrt(G_KPC * mvir / np.clip(rvir, 1e-6, None))
        out[f"{lo:.1f}-{lo + 0.5:.1f}"] = {
            "n": n,
            "c": float(np.median(rvir / np.clip(h["Rs"].to_numpy(), 1e-6, None))),
            "vmax_over_vvir": float(np.median(h["vmax"].to_numpy()
                                              / np.clip(vvir, 1e-6, None))),
            "vrms": float(np.median(h["vrms"].to_numpy())),
            "xoff": float(np.median(h["Xoff"].to_numpy()
                                    / np.clip(rvir, 1e-6, None))),
            "t_over_u": float(np.median(h["T/|U|"].to_numpy())),
        }
    return out


def verdict(rows: dict) -> str:
    """One sentence naming what the arms imply, or why they cannot say."""
    need = ("hr", "base", "srpos_hrvel")
    if any(rows.get(a) is None for a in need):
        have = sorted(a for a in ARM_ORDER if rows.get(a) is not None)
        return (f"INCONCLUSIVE: arms present = {have}; "
                f"{sorted(set(need) - set(have))} missing, so no comparison.")
    r_v = recovery_fraction(rows["srpos_hrvel"]["n_subhalos"],
                            rows["base"]["n_subhalos"], rows["hr"]["n_subhalos"])
    mirror = rows.get("hrpos_srvel")
    r_d = (recovery_fraction(mirror["n_subhalos"], rows["base"]["n_subhalos"],
                             rows["hr"]["n_subhalos"])
           if mirror is not None else None)
    parts = [f"SR2 pos + HR vel recovers {r_v:+.0%} of the SR2->HR subhalo gap"]
    if r_d is not None:
        parts.append(f"HR pos + SR2 vel sits at {r_d:+.0%} of it")
    head = "; ".join(parts) + ". "
    if r_v >= 0.6:
        return head + ("The velocity channels carry most of the deficit: the "
                       "displacement field already contains the substructure "
                       "and the halo finder could not see it without the "
                       "correct velocities. Fix the velocity head.")
    if r_v <= 0.2:
        return head + ("The velocity channels are not the constraint. The "
                       "substructure is absent from the displacement field "
                       "itself, and the cold halo interiors are a symptom -- "
                       "which points at the seeded-substructure route in "
                       "docs/sr2_subhalo_deficit.md, not at a velocity fix.")
    return head + ("Neither half accounts for the deficit on its own; the two "
                   "channel groups interact and a single-head fix will not "
                   "close the gap.")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def catalog_path(box: str, arm: str) -> Path | None:
    """Where this arm's catalog lives, frozen control or swap alike."""
    if arm in ("hr", "base"):
        pat = paths.subdir("halos", f"{box}__{arm}__{arm}", arm + "_rockstar")
    else:
        pat = paths.subdir("channel_swap", f"{box}__{arm}", arm + "_rockstar")
    hits = sorted(glob.glob(str(Path(pat) / "halos*.ascii"))
                  + glob.glob(str(Path(pat) / "halos*.list")))
    return Path(hits[0]) if hits else None


def load_catalog_df(path: Path):
    """The full Rockstar ASCII as a DataFrame, with an ``is_sub`` column.

    Read with pandas rather than ``load_rockstar_ascii`` because the structural
    columns this report lives on -- ``vrms``, ``Rs``, ``Xoff``, ``T/|U|`` -- are
    not part of ``HaloCatalog``.
    """
    import pandas as pd

    with open(path) as fh:
        header = fh.readline()
    if not header.startswith("#"):
        raise SystemExit(f"{path} does not start with a Rockstar header line")
    names = header.lstrip("#").split()
    df = pd.read_csv(path, comment="#", sep=r"\s+", header=None, names=names)
    parent = parent_ids_from_columns(df["id"], df["idx"], df["i_so"])
    df["is_sub"] = parent >= 0
    return df


def summarise_arm(box: str, arm: str, occupancy_norm: int) -> dict | None:
    p = catalog_path(box, arm)
    if p is None:
        print(f"    GATE: no catalog for arm {arm!r}; it did not run yet")
        return None
    df = load_catalog_df(p)
    is_sub = df["is_sub"].to_numpy()
    num_p = df["num_p"].to_numpy(dtype=np.int64)
    row = {
        "arm": arm, "label": ARM_LABEL.get(arm, arm), "catalog": str(p),
        "n_objects": int(len(df)),
        "n_hosts": int((~is_sub).sum()),
        "n_subhalos": int(is_sub.sum()),
        "n_bound_particles": int(num_p.sum()),
        "occupancy": float(num_p.sum() / float(occupancy_norm)),
        "by_num_p": counts_by_num_p(num_p, is_sub),
        "structure": structure_by_mass(df),
    }
    print(f"    {arm:12s} {row['n_objects']:>7} objects  "
          f"{row['n_hosts']:>7} hosts  {row['n_subhalos']:>7} subhalos  "
          f"occupancy {row['occupancy']:.4f}", flush=True)
    return row


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def markdown(box: str, rows: dict, text: str) -> str:
    present = [a for a in ARM_ORDER if rows.get(a) is not None]
    hr, base = rows.get("hr"), rows.get("base")
    L = [f"# Channel-swap intervention on {box}", "",
         "Which half of the SR2 field costs it the subhalos: the displacement",
         "channels (0:3) or the velocity channels (3:6). Built by",
         "`scripts/reward/channel_swap_rockstar.py`, aggregated here.", "",
         "## Counts", "",
         "| arm | displacement | velocity | hosts | subhalos | SR2->HR recovery | occupancy |",
         "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    src = {"hr": "HR", "base": "SR2", "srpos_hrvel": ("SR2", "HR"),
           "hrpos_srvel": ("HR", "SR2")}
    for a in present:
        r = rows[a]
        d, v = (src[a] if isinstance(src[a], tuple) else (src[a], src[a]))
        rec = (recovery_fraction(r["n_subhalos"], base["n_subhalos"],
                                 hr["n_subhalos"])
               if hr is not None and base is not None else None)
        L.append(f"| {r['label']} | {d} | {v} | {r['n_hosts']} | "
                 f"{r['n_subhalos']} | "
                 + ("n/a" if rec is None else f"{rec:+.0%}") + " | "
                 + f"{r['occupancy']:.4f} |")
    L += ["", f"**Verdict.** {text}", "", "## Subhalos by member count", "",
          "| num_p | " + " | ".join(rows[a]["label"] for a in present) + " |",
          "| --- | " + " | ".join("---:" for _ in present) + " |"]
    for label in counts_by_num_p([], []).keys():
        L.append(f"| {label} | "
                 + " | ".join(str(rows[a]["by_num_p"][label]["subs"])
                              for a in present) + " |")
    L += ["", "## Host internal structure (medians per log Mvir bin)", "",
          "`T/|U|` = 0.5 is virial equilibrium; `c` is the density-side control.", ""]
    for field, title in (("vrms", "vrms [km/s]"), ("t_over_u", "T/|U|"),
                         ("xoff", "Xoff / rvir"), ("c", "c = rvir/Rs")):
        L += [f"### {title}", "",
              "| log Mvir | " + " | ".join(rows[a]["label"] for a in present) + " |",
              "| --- | " + " | ".join("---:" for _ in present) + " |"]
        keys = sorted({k for a in present for k in rows[a]["structure"]})
        for k in keys:
            cells = []
            for a in present:
                s = rows[a]["structure"].get(k)
                cells.append("-" if s is None else f"{s[field]:.3g}")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--box", default="set8")
    ap.add_argument("--occupancy-norm", type=int, default=512 ** 3)
    args = ap.parse_args(argv)
    load_reward_config(args)
    box = str(args.box)

    banner(f"channel-swap report for {box}")
    rows = {a: summarise_arm(box, a, int(args.occupancy_norm)) for a in ARM_ORDER}
    text = verdict(rows)

    out = paths.subdir("channel_swap", create=True)
    write_json(out / f"{box}_channel_swap.json",
               {"box": box, "verdict": text,
                "arms": {a: r for a, r in rows.items() if r is not None}})
    md = out / f"{box}_channel_swap.md"
    md.write_text(markdown(box, rows, text))
    banner("verdict")
    print(text)
    print(f"\n    wrote {md}")
    print(f"    wrote {out / (box + '_channel_swap.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
