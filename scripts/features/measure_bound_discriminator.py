#!/usr/bin/env python
"""Name the term the gather objective is missing, without running a halo finder.

``docs/sr2_gather_finetune.md`` sections 5-6 left one question open: the tuned
field matched every statistic the loss constrained and Rockstar found 0 of 43
supervised subhalos, while the true HR tiles through the identical harness gave
42 of 43. Both fields are on disk, in the same splice geometry, evaluated on the
same HR member sets. So the statistics that separate them can be read off
directly -- and whichever separates them *is* the term the loss is missing.

The reference is the HR-tile field (the verified positive). Every other field is
scored against it, per statistic, by a paired median and a Mann-Whitney AUC over
the supervised sets. ``r_rms`` and ``sigma_v`` are controls: the loss did
constrain them, so they should NOT separate, and a run where they do has
measured something generic instead of the thing it set out to.

    python scripts/features/measure_bound_discriminator.py --box set8 \
        --run-dir <.../host_gather/set8_h271800_fine_anchored> --host-id 271800

Writes ``<reward_root>/bound_discriminator/<box>__<run>.json`` with every set's
statistics on every field, the discrimination table, and a ``verdict`` naming
the separating statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cosmo_sr.eval.particles import particle_mass_msun_h            # noqa: E402
from cosmo_sr.features.bound_discriminator import (                 # noqa: E402
    mann_whitney_auc, member_ids, particles_at, set_statistics,
)

# The statistics, and whether each is a control (constrained by the gather loss)
# or a candidate for the missing term. Order is the report order.
STATS = [
    ("bound_frac", "candidate", "Rockstar's unbinding test on the set"),
    ("virial_ratio", "candidate", "2T/|W| -- section 7.4's candidate"),
    ("d6", "candidate", "6-D compactness in Rockstar's linking metric"),
    ("vr_corr", "candidate", "radial position vs radial velocity (gradient)"),
    ("vr_mean", "candidate", "net radial drift in units of sigma_v"),
    ("coldness", "candidate", "sigma_v / v_circ"),
    ("r_rms", "control", "size -- the loss constrained compact mass"),
    ("sigma_v", "control", "dispersion -- the loss constrained it directly"),
]


def _field_path(root: str, box: str, tag: str, seed: int) -> str:
    return os.path.join(root, "flow_rockstar", "fields",
                        f"{box}__{tag}__seed{seed}.npy")


def _median(vals: List[float]) -> float:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--box", default="set8")
    ap.add_argument("--run-dir", required=True,
                    help="a finished host_gather run (for subhalos.json)")
    ap.add_argument("--host-id", type=int, default=271800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tags", default="",
                    help="comma list of field tags; default derives from run-dir")
    ap.add_argument("--reference-tag", default="",
                    help="the verified positive; default <derived>_hr")
    ap.add_argument("--softening-mpc-h", type=float, default=0.01)
    ap.add_argument("--boxsize-mpc-h", type=float, default=100.0)
    ap.add_argument("--omega-m", type=float, default=0.2814)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    root = os.environ.get("DMSR_REWARD_ROOT",
                          "/zfsauton/scratch/yixiz/DMSR/dmsr_reward")
    run = os.path.basename(os.path.normpath(a.run_dir))
    base_tag = f"gather_{run}"
    ref_tag = a.reference_tag or f"{base_tag}_hr"
    tags = ([t for t in a.tags.split(",") if t] or
            [ref_tag, base_tag, f"{base_tag}_frozen"])
    if ref_tag not in tags:
        tags = [ref_tag] + tags

    # --- gates: a missing input is "nothing to do", never a crash ----------
    subs_p = os.path.join(a.run_dir, "subhalos.json")
    owner_p = os.path.join(root, "halos_particles", f"{a.box}__hr__hr",
                           f"{a.box}_hr_owner.npy")
    for p, what in ((subs_p, "the run's supervised targets"),
                    (owner_p, "the HR owner array")):
        if not os.path.isfile(p):
            print(f"GATE: missing {what} at {p}")
            print("GATE: nothing to measure; exiting 0 so dependents report the same.")
            return 0

    present = [t for t in tags if os.path.isfile(_field_path(root, a.box, t, a.seed))]
    for t in tags:
        if t not in present:
            print(f"NOTE: no field for tag {t} "
                  f"({_field_path(root, a.box, t, a.seed)}); skipping it.")
    if ref_tag not in present:
        print(f"GATE: the reference field {ref_tag} is absent -- without the "
              f"verified positive there is nothing to discriminate against.")
        return 0
    if len(present) < 2:
        print("GATE: need the reference plus at least one other field.")
        return 0

    rows = json.load(open(subs_p))["rows"]
    halo_ids = [int(r["halo_id"]) for r in rows]
    num_p = {int(r["halo_id"]): int(r["num_p"]) for r in rows}
    print(f"=== bound discriminator: {a.box}, run {run}, "
          f"{len(halo_ids)} supervised targets, host {a.host_id}")
    print(f"    fields: {', '.join(present)}   (reference: {ref_tag})")
    print(f"    softening {a.softening_mpc_h} Mpc/h")

    # Grid size comes from the reference field, not a constant: the particle
    # mass scales as Ng^-3 and a silently wrong one would rescale every energy.
    ng = int(np.load(_field_path(root, a.box, ref_tag, a.seed),
                     mmap_mode="r").shape[1])
    m_p = particle_mass_msun_h(a.omega_m, a.boxsize_mpc_h, ng ** 3)
    print(f"    grid {ng}^3, particle mass {m_p:.5e} Msun/h")

    print("--- reading the owner array (one pass) ---")
    sets = member_ids(owner_p, halo_ids + [a.host_id])
    host_ids = sets.pop(a.host_id, np.empty(0, dtype=np.int64))
    print(f"    host {a.host_id}: {host_ids.size} particles")

    # Self-check: the owner array's set size must equal the catalog's num_p.
    bad = [(h, int(sets[h].size), num_p[h])
           for h in halo_ids if int(sets[h].size) != num_p[h]]
    if bad:
        print(f"    WARNING: {len(bad)} sets disagree with catalog num_p, "
              f"e.g. {bad[:3]} -- ids are not the objects' particle lists.")
    else:
        print(f"    owner/num_p invariant holds for all {len(halo_ids)} sets")

    out: Dict = {
        "box": a.box, "run": run, "host_id": a.host_id,
        "reference_tag": ref_tag, "tags": present,
        "softening_mpc_h": a.softening_mpc_h,
        "particle_mass_msun_h": m_p,
        "n_targets": len(halo_ids),
        "owner_num_p_consistent": not bad,
        "fields": {},
    }

    for tag in present:
        path = _field_path(root, a.box, tag, a.seed)
        fld = np.load(path, mmap_mode="r")
        hx, hv = particles_at(fld, host_ids)
        from cosmo_sr.features.bound_discriminator import unwrap_periodic
        hx = unwrap_periodic(hx, a.boxsize_mpc_h)
        h_sig_x = float(np.sqrt(((hx - hx.mean(0)) ** 2).sum(1).mean()))
        h_sig_v = float(np.sqrt(((hv - hv.mean(0)) ** 2).sum(1).mean()))
        print(f"--- {tag}: host sigma_x {h_sig_x:.3f} Mpc/h, "
              f"sigma_v {h_sig_v:.1f} km/s")

        per: List[Dict] = []
        for h in halo_ids:
            ids = sets[h]
            px, pv = particles_at(fld, ids)
            st = set_statistics(px, pv, particle_mass_msun_h=m_p,
                                boxsize_mpc_h=a.boxsize_mpc_h,
                                softening_mpc_h=a.softening_mpc_h,
                                host_sigma_x=h_sig_x, host_sigma_v=h_sig_v)
            d = st.to_dict()
            d["halo_id"] = h
            per.append(d)
        med = {k: _median([p[k] for p in per]) for k, _, _ in STATS}
        out["fields"][tag] = {
            "host_sigma_x_mpc_h": h_sig_x, "host_sigma_v_kms": h_sig_v,
            "median": med, "sets": per,
        }
        print("    " + "  ".join(f"{k}={med[k]:.3g}" for k, _, _ in STATS))
        del fld

    # --- discrimination against the verified positive ----------------------
    ref = out["fields"][ref_tag]["sets"]
    disc: Dict = {}
    for tag in present:
        if tag == ref_tag:
            continue
        cur = out["fields"][tag]["sets"]
        table = {}
        for k, kind, _ in STATS:
            rv = [p[k] for p in ref]
            cv = [p[k] for p in cur]
            pair = [(x, y) for x, y in zip(rv, cv)
                    if np.isfinite(x) and np.isfinite(y)]
            table[k] = {
                "kind": kind,
                "median_reference": _median(rv),
                "median_candidate": _median(cv),
                "auc": mann_whitney_auc(rv, cv),
                "paired_frac_reference_greater":
                    (float(np.mean([x > y for x, y in pair])) if pair
                     else float("nan")),
                "n_paired": len(pair),
            }
        disc[tag] = table
    out["discrimination"] = disc

    print()
    for tag, table in disc.items():
        print(f"=== {ref_tag}  vs  {tag} ===")
        print(f"  {'statistic':<14}{'kind':<11}{'reference':>12}"
              f"{'candidate':>12}{'AUC':>8}{'|sep|':>8}")
        for k, _, why in STATS:
            t = table[k]
            sep = abs(t["auc"] - 0.5) if np.isfinite(t["auc"]) else float("nan")
            print(f"  {k:<14}{t['kind']:<11}{t['median_reference']:>12.4g}"
                  f"{t['median_candidate']:>12.4g}{t['auc']:>8.3f}{sep:>8.3f}")

    # --- verdict ------------------------------------------------------------
    lines = []
    for tag, table in disc.items():
        cands = sorted(
            ((abs(table[k]["auc"] - 0.5), k) for k, kind, _ in STATS
             if kind == "candidate" and np.isfinite(table[k]["auc"])),
            reverse=True)
        ctrls = [(abs(table[k]["auc"] - 0.5), k) for k, kind, _ in STATS
                 if kind == "control" and np.isfinite(table[k]["auc"])]
        worst_ctrl = max(ctrls)[0] if ctrls else float("nan")
        if not cands:
            lines.append(f"{tag}: no candidate statistic was computable.")
            continue
        top_sep, top = cands[0]
        strong = [k for s, k in cands if s >= 0.25]
        if top_sep < 0.15:
            lines.append(
                f"{tag}: NO candidate separates it from the reference "
                f"(best {top} at |AUC-0.5| {top_sep:.3f}). The missing term is "
                f"not among the statistics tested here.")
        elif np.isfinite(worst_ctrl) and top_sep <= worst_ctrl:
            lines.append(
                f"{tag}: {top} separates ({top_sep:.3f}) but no better than a "
                f"CONTROL ({worst_ctrl:.3f}), so the separation is generic -- "
                f"the two fields differ overall, not in this statistic "
                f"specifically. Not attributable.")
        else:
            names = ", ".join(strong) if strong else top
            lines.append(
                f"{tag}: SEPARATES on {names} (best {top}, |AUC-0.5| "
                f"{top_sep:.3f}) while the controls stay at {worst_ctrl:.3f}. "
                f"That is the term the gather loss is missing.")
    out["verdict"] = " ".join(lines)
    print()
    print("VERDICT: " + out["verdict"])

    dest = a.out or os.path.join(root, "bound_discriminator",
                                 f"{a.box}__{run}.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"=== wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
