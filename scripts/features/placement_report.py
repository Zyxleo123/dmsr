#!/usr/bin/env python
"""Decision tables from the placement forensics (reads saved JSON only).

Inputs, all written by ``submit_placement_forensics.sh``:

* ``<box>_forensics.json``       every catalog + SUBFIND + cross-arm table
* ``<box>_forensics_swap.json``  the two velocity-swap catalogs
* ``host_profiles/<box>__*.json`` stacked host phase-space profiles

Prints, and writes as markdown next to the inputs:

1. what the core (< 0.25 R_vir) is made of, per catalog: mass bins, linked
   structures, fragments per structure, co-moving fraction, T/|U|;
2. SUBFIND's self-bound satellites on the same radial bins;
3. the velocity swap: does frozen SR2's velocity field remove the core pile,
   does the arm's velocity field create it in frozen's density?;
4. the outskirt mass shift: the num_p factor that maps the arm's outer
   cumulative count onto HR's;
5. field profiles, ratio to HR per radial bin, and HR's host-to-host scatter;
6. the strongest cross-arm correlations.

No catalog or field is read, so it reruns in seconds.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

KEY = ["hr", "base", "frozen_gate", "all_blocks_self", "all_blocks_sm_control",
       "all_blocks_sm_coarse_soft", "all_blocks_knn6d_w5",
       "all_blocks_abl_min_v_l", "all_blocks_abl_nod6", "all_blocks_selfvel",
       "all_blocks_abl_novirial", "all_blocks_nocentre", "all_blocks_radparent",
       "all_blocks_abl_shp_iso1@2000", "all_blocks_nearclean2",
       "all_blocks_reloc_B", "m2m_base_e77", "m2m_knn_e77", "m2m_mg_e77"]


def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if not np.isfinite(v):
        return "-"
    return f"{v:.{nd}f}"


def table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" if i == 0 else "--:" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def core_section(rk: Dict[str, Dict], names: List[str]) -> str:
    rows = []
    for n in names:
        if n not in rk:
            continue
        r = rk[n]
        c = np.asarray(r["radial_by_mass"]["counts"])       # (9 r, 5 m)
        tot = np.asarray(r["radial_by_mass"]["total_by_r"])
        s20 = r["structures"]["ge20"]
        s50 = r["structures"]["ge50"]
        kin = r["kinematics"]
        # T/|U| and |dv|/vrms_host for 50-200p (mass bins 1,2) in the core and at 1-1.5
        def kmed(field, rbin):
            vals = [kin[field][rbin][j] for j in (1, 2)]
            ns = [kin["n"][rbin][j] for j in (1, 2)]
            vals = [v for v, k in zip(vals, ns) if k > 0 and np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")
        rows.append([
            n, fmt(int(r["n_region_subs"])), fmt(int(tot[0])),
            fmt(float(tot[0]) / max(r["n_region_subs"], 1)),
            fmt(int(c[0, 0])), fmt(int(c[0, 1])), fmt(int(c[0, 2:].sum())),
            fmt(int(s20["structures_by_r"][0])),
            fmt(s20["frag_per_structure_by_r"][0], 1),
            fmt(int(s50["structures_by_r"][0])),
            fmt(r["comoving_frac_by_r"][0]),
            fmt(kmed("t_over_u", 0)), fmt(kmed("t_over_u", 4)),
            fmt(kmed("dv_over_host_vrms", 0)),
            fmt(int(tot[4])), fmt(int(c[4, 1:].sum())),
            fmt(r["radial_by_mass"]["median_x"]),
            fmt(s20["median_x_structures"]),
        ])
    h = ["catalog", "subs", "core", "core frac", "core 20-50p", "core 50-100p",
         "core >=100p", "core structs", "frag/struct", "core structs>=50p",
         "co-moving", "T/U core", "T/U 1-1.5", "dv/vrms core", "1-1.5 all",
         "1-1.5 >=50p", "median x", "median x structs"]
    return table(h, rows)


def subfind_section(sf: Dict[str, Dict]) -> str:
    rows = []
    for n, r in sf.items():
        c = np.asarray(r["radial_by_mass"]["counts"])
        tot = np.asarray(r["radial_by_mass"]["total_by_r"])
        ge50 = c[:, 1:].sum(axis=1)
        rows.append([n, fmt(int(r["n_region_sats"])), fmt(int(r["n_region_sats_ge50"])),
                     fmt(int(tot[0])), fmt(float(tot[0]) / max(tot.sum(), 1)),
                     fmt(int(ge50[0])), fmt(float(ge50[0]) / max(ge50.sum(), 1)),
                     fmt(int(tot[4])), fmt(int(ge50[4])),
                     fmt(r["radial_by_mass"]["median_x"]),
                     fmt(r["radial_by_mass"]["median_x_ge50"])])
    h = ["SUBFIND", "sats", "sats>=50p", "core", "core frac", "core>=50p",
         "core frac>=50p", "1-1.5", "1-1.5>=50p", "median x", "median x>=50p"]
    return table(h, rows)


def mass_shift(cum_arm: List[int], cum_hr: List[int], thr: List[int],
               t: int) -> float:
    """f such that N_arm(>= f t) = N_HR(>= t), log-interpolated; nan if out of range."""
    target = cum_hr[thr.index(t)]
    x = np.log(np.asarray(thr, float))
    y = np.asarray(cum_arm, float)
    if target <= 0 or not (y.min() <= target <= y.max()):
        return float("nan")
    # cumulative counts fall with threshold; interpolate log(threshold) at target
    order = np.argsort(y)
    xi = np.interp(target, y[order], x[order])
    return float(np.exp(xi) / t)


def outskirt_section(rk: Dict[str, Dict], names: List[str]) -> str:
    hr = rk["hr"]["outskirt_cumulative"]
    thr = hr["thresholds"]
    rows = []
    for n in names:
        if n not in rk:
            continue
        o = rk[n]["outskirt_cumulative"]
        row = [n]
        for band in ("0.75-2", "1-1.5"):
            row += [fmt(int(o[band][thr.index(50)])), fmt(int(hr[band][thr.index(50)])),
                    fmt(mass_shift(o[band], hr[band], thr, 50)),
                    fmt(mass_shift(o[band], hr[band], thr, 100))]
        rows.append(row)
    h = ["catalog", "N>=50 .75-2", "HR", "f@50", "f@100",
         "N>=50 1-1.5", "HR", "f@50", "f@100"]
    return ("f = the num_p factor that maps the catalog's cumulative count onto "
            "HR's (N_cat(>= f t) = N_HR(>= t)); f < 1 means the outer objects "
            "exist but are under-massive, '-' means the catalog never reaches "
            "HR's count at any threshold (objects missing, not light).\n\n"
            + table(h, rows))


def swap_section(rs: Dict[str, Dict]) -> str:
    order = ["base", "frozen_gate", "coarse_soft", "cs_disp_frzvel", "frz_disp_csvel", "hr"]
    rows = []
    for n in order:
        if n not in rs:
            continue
        r = rs[n]
        c = np.asarray(r["radial_by_mass"]["counts"])
        tot = np.asarray(r["radial_by_mass"]["total_by_r"])
        s20 = r["structures"]["ge20"]
        rows.append([n, fmt(int(r["n_region_subs"])), fmt(int(r["n_region_subs_ge50"])),
                     fmt(int(tot[0])), fmt(int(c[0, 1:].sum())),
                     fmt(int(s20["structures_by_r"][0])),
                     fmt(s20["frag_per_structure_by_r"][0], 1),
                     fmt(int(tot[4])), fmt(int(c[4, 1:].sum())),
                     fmt(r["radial_by_mass"]["median_x"]),
                     fmt(r["comoving_frac_by_r"][0])])
    h = ["catalog (disp+vel)", "subs", "subs>=50p", "core", "core>=50p",
         "core structs", "frag/struct", "1-1.5", "1-1.5>=50p", "median x", "co-moving"]
    return table(h, rows)


def profile_section(prof: Dict[str, Dict], which: str = "stack_all") -> str:
    if "hr" not in prof:
        return "(no HR profile)"
    edges = prof["hr"]["edges_x"]
    labels = [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
    out = []
    for q in ("rho", "sigma3d", "sigma_loc", "coldness", "f_clump10", "f_coldclump", "vr"):
        hr = np.asarray(prof["hr"][which][q]["median"], float)
        rows = [["HR (value)"] + [fmt(v, 3) for v in hr]]
        p16 = np.asarray(prof["hr"][which][q]["p16"], float)
        p84 = np.asarray(prof["hr"][which][q]["p84"], float)
        with np.errstate(all="ignore"):
            rows.append(["HR scatter (p84-p16)/2/med"]
                        + [fmt(v, 2) for v in (p84 - p16) / 2.0 / np.abs(hr)])
        for n, p in prof.items():
            if n == "hr":
                continue
            v = np.asarray(p[which][q]["median"], float)
            if q == "vr":
                rows.append([n + " (value)"] + [fmt(x, 3) for x in v])
            else:
                with np.errstate(all="ignore"):
                    rows.append([n + " / HR"] + [fmt(x, 2) for x in v / hr])
        out.append(f"**{q}** ({which}, median over hosts)\n\n" + table(["r/R_vir"] + labels, rows))
    sc = ["centre_offset_over_rvir", "n_within_rvir_over_hr", "conc_n025_over_n1"]
    rows = [[n] + [fmt(p["scalars_median"].get(k), 3) for k in sc] for n, p in prof.items()]
    out.append("**per-host scalars (median)**\n\n" + table(["field"] + sc, rows))
    return "\n\n".join(out)


def corr_section(fx: Dict) -> str:
    corr = fx["correlations"]
    rows = []
    for t, d in corr.items():
        for p, v in d.items():
            if v["n"] >= 8 and np.isfinite(v["rho"]):
                rows.append((abs(v["rho"]), t, p, v))
    rows.sort(reverse=True)
    body = [[t, p, fmt(v["rho"]), str(v["n"]), fmt(v["p"], 3)] for _, t, p, v in rows[:24]]
    return table(["catalog quantity", "held-out diagnostic", "Spearman rho", "n", "perm p"], body)


def knn_host_section(fx: Dict, knn_json: Path) -> str:
    """Host phase-space collapse (6D kNN, saved by the kNN round 5b) vs the core.

    ``membership[field].classes.host.median_r`` is the median k-NN distance of
    particles HR files as HOST material; field/HR < 1 means host phase space
    collapsed. If that collapse is what Rockstar fragments, it tracks the core
    excess and the core structure count across arms.
    """
    if not knn_json.is_file():
        return f"(no {knn_json})"
    from scipy.stats import spearmanr
    k = json.loads(knn_json.read_text())["membership"]
    hr = np.asarray(k["hr"]["classes"]["host"]["median_r"], float)
    rk = fx["rockstar"]
    rows, xs, ys, zs, ws = [], [], [], [], []
    for name, m in k.items():
        if name == "hr":
            continue
        key = "base" if name == "frozen" else name.replace("_step2000", "@2000")
        if key not in rk:
            continue
        ratio = float(np.mean(np.asarray(m["classes"]["host"]["median_r"], float) / hr))
        sub = np.asarray(m["classes"]["sub_50_200"]["median_r"], float)
        sub_hr = np.asarray(k["hr"]["classes"]["sub_50_200"]["median_r"], float)
        r = rk[key]
        core = int(r["radial_by_mass"]["total_by_r"][0])
        cs = int(r["structures"]["ge20"]["structures_by_r"][0])
        frag = float(r["structures"]["ge20"]["frag_per_structure_by_r"][0])
        rows.append([key, fmt(ratio, 3), fmt(float(np.mean(sub / sub_hr)), 3),
                     str(core), str(cs), fmt(frag, 1)])
        xs.append(ratio); ys.append(core); zs.append(frag); ws.append(cs)
    out = table(["catalog", "host r_k / HR", "sub_50_200 r_k / HR", "core subs",
                 "core structs", "frag/struct"], rows)
    if len(xs) >= 5:
        for lab, v in (("core subs", ys), ("core frag/struct", zs), ("core structs", ws)):
            rho, p = spearmanr(xs, v)
            out += f"\n\nSpearman(host r_k/HR, {lab}) = {rho:.2f} (p {p:.2g}, n {len(xs)})"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward/placement_forensics")
    ap.add_argument("--box", default="set9")
    ap.add_argument("--knn-json", default="/zfsauton/scratch/yixiz/DMSR/dmsr_reward/member_gather/"
                    "knn_cdf/knn_phase_l4000_k8-16-32-64_region_allarms_set9.json")
    a = ap.parse_args(argv)
    D = Path(a.dir)
    parts = []
    fx_p = D / f"{a.box}_forensics.json"
    fx = json.loads(fx_p.read_text()) if fx_p.is_file() else None
    if fx:
        rk = fx["rockstar"]
        names = [n for n in KEY if n in rk]
        parts.append("## 1. What the core (< 0.25 R_vir) is made of (Rockstar, edited region)\n\n"
                     "T/U and dv/vrms are for 50-200p subhalos; co-moving = |v_sub - v_host| "
                     "< 0.3 vrms_host; structures link subhalos whose centre lies inside "
                     "another's r_vir.\n\n" + core_section(rk, names))
        parts.append("## 2. SUBFIND self-bound satellites, same radial bins "
                     "(R_vir from FOF mass)\n\n" + subfind_section(fx["subfind"]))
        parts.append("## 4. Outskirt mass shift\n\n" + outskirt_section(rk, names))
        parts.append("## 6. Strongest cross-arm correlations\n\n" + corr_section(fx))
        knn = Path(a.knn_json)
        parts.append("## 6b. Host phase-space collapse (6D kNN, lambda 4000) vs the core\n\n"
                     + knn_host_section(fx, knn))
        hm = []
        for n in names:
            b = rk[n]["by_host_mass"]
            hm.append([n] + [f"{fmt(x['median_r'], 3)} / {fmt(x['median_r_outside_core'], 3)}"
                             for x in b])
        parts.append("## 4b. Median physical r (Mpc/h) by top-level host mass, all / outside "
                     "0.25 R_vir\n\n" + table(["catalog"] + fx["host_mass_bins"], hm))
        hmatch = [[n] + [fmt(rk[n].get("host_match", {}).get(k), 3) for k in
                         ("median_offset_over_rvir_hr", "median_rvir_ratio",
                          "median_num_p_ratio", "frac_matched_within_0.3rvir")]
                  for n in names if "host_match" in rk[n]]
        parts.append("## 4c. Host match at massive HR hosts\n\n"
                     + table(["catalog", "offset/Rvir_HR", "Rvir ratio", "num_p ratio",
                              "matched<0.3Rvir"], hmatch))
    sw_p = D / f"{a.box}_forensics_swap.json"
    if sw_p.is_file():
        sw = json.loads(sw_p.read_text())["rockstar"]
        parts.append("## 3. Velocity swap (coarse_soft vs frozen)\n\n" + swap_section(sw))
    prof = {}
    for p in sorted(glob.glob(str(D / "host_profiles" / f"{a.box}__*.json"))):
        d = json.loads(Path(p).read_text())
        prof[d["name"]] = d
    if prof:
        parts.append("## 5. Stacked host phase-space profiles\n\n" + profile_section(prof))
        parts.append("## 5b. Hosts >= 100k particles only\n\n"
                     + profile_section(prof, "stack_ge100k"))
    md = "\n\n".join(parts)
    print(md)
    out = D / f"{a.box}_placement_report.md"
    out.write_text(md + "\n")
    print(f"\n=== wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
