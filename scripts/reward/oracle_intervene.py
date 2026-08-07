#!/usr/bin/env python
"""Experiment 1, step 3: build one intervention box, halo-find it, score it.

One array task = one cell of the grid ``(kind, mode, alpha)``:

    x_alpha = x_SR2 + alpha * w * (x_HR - x_SR2)

All of a box's targets are edited in the *same* field, which is what makes the
experiment affordable -- ``oracle_select_targets.py`` already forced them apart
so their masks cannot overlap. The complete periodic box then goes through
Rockstar exactly as the frozen pipeline does.

``kind`` selects what ``w`` is built from:

``targeted``
    A smooth mask over the Lagrangian sites of the missing HR subhalo's own
    particles. The experiment.
``exact_particles``
    A hard indicator on exactly those sites, no dilation, no smoothing -- "replace
    only the HR particles belonging to the missing subhalo".
``control``
    The same *number* of Lagrangian sites, drawn from the same host, excluding
    the target's. If this recovers subhalos too, the targeted result was just
    added fluctuation power and means nothing.

Scoring per box: recovery of every target, whole-box occupation in every host
bin, subhalo abundance, the catalog reward, and the five field constraints. The
occupation numbers are whole-box, which by Experiment 0's identity is exactly
what the tile decomposition sums to -- so no member-particle output is needed
here and these boxes cost one plain Rockstar run each.

    python scripts/reward/oracle_intervene.py --box set8 --kind targeted \\
        --mode disp --alpha 0.5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import (  # noqa: E402
    add_common_args, append_jsonl, banner, bins_of, constraints_of, hr_path,
    load_reward_config, lr_path, paths, write_json,
)

from cosmo_sr.eval.rockstar import run_rockstar_on_field  # noqa: E402
from cosmo_sr.reward.catalog import EnsembleSummary  # noqa: E402
from cosmo_sr.reward.constraints import check_constraints, constraint_values  # noqa: E402
from cosmo_sr.reward.oracle_hr import (  # noqa: E402
    InterventionSpec, Target, apply_intervention, ids_to_lattice,
    lagrangian_mask, random_control_sites, recovery_report,
)
from cosmo_sr.reward.reward import RewardModel  # noqa: E402
from cosmo_sr.reward.tiles import direct_full_box_stats  # noqa: E402

from rockstar_particles import load_field  # noqa: E402
from oracle_select_targets import catalog  # noqa: E402


def load_members(box: str) -> dict:
    """``halo id -> member particle ids``, written by ``--extract-members``."""
    hits = sorted(paths.subdir("halos_particles", f"{box}__hr__hr")
                  .glob(f"{box}_hr_members.npz"))
    if not hits:
        raise SystemExit(
            f"no member ids for {box}. Run rockstar_particles.py --box {box} "
            f"--source hr --extract-members .../halo_ids_{box}.json first; the "
            "masks cannot be built in Lagrangian space without them."
        )
    z = np.load(hits[0])
    ids, off, pid = z["halo_id"], z["offset"], z["particle_id"]
    return {int(h): pid[off[k]:off[k + 1]] for k, h in enumerate(ids)}


def build_mask(targets, members, spec: InterventionSpec, ng: int):
    """``(mask, n_sites)``: one Lagrangian mask covering every target in this box.

    The site sets of all targets are unioned **before** dilating and smoothing,
    not after. Per-target masks would mean ~33 periodic rolls of a 512^3 volume
    each, times 24 targets -- around ten minutes of pure array shuffling per
    cell, repeated over 32 cells. Doing it once is exact here, not merely close:
    ``oracle_select_targets`` forces targets >= 6 Mpc/h apart (~30 HR cells),
    far beyond the 1.5-cell smoothing, so the blobs are disjoint and
    ``smooth(union) == max(smooth(each))``.
    """
    all_sites = []
    for t in targets:
        sub = members.get(int(t.hr_sub_id))
        if sub is None or sub.size == 0:
            print(f"    !! target {t.hr_sub_id} has no member ids; skipped",
                  flush=True)
            continue
        sites = ids_to_lattice(sub, ng)
        if spec.kind == "control":
            host = members.get(int(t.hr_host_id))
            if host is None or host.size == 0:
                print(f"    !! host {t.hr_host_id} has no member ids; skipped",
                      flush=True)
                continue
            sites = random_control_sites(
                ids_to_lattice(host, ng), sites,
                seed=int(spec.seed) + int(t.hr_sub_id),
            )
        all_sites.append(sites)

    if not all_sites:
        return np.zeros((ng, ng, ng), dtype=np.float32), 0
    sites = np.concatenate(all_sites, axis=0)
    if spec.kind == "exact_particles":
        mask = lagrangian_mask(sites, ng, dilate=0, smooth=0.0)
    else:
        mask = lagrangian_mask(sites, ng, dilate=spec.dilate, smooth=spec.smooth)
    return mask, int(sites.shape[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--box", required=True)
    ap.add_argument("--kind", default="targeted",
                    choices=("targeted", "control", "exact_particles"))
    ap.add_argument("--mode", default="both", choices=("disp", "vel", "both"))
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--smooth", type=float, default=1.5)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-fields", action="store_true",
                    help="skip the field constraints (they are the memory peak)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    cons = constraints_of(cfg)
    d = cfg["data"]
    ng, box_l = int(d["ng_hr"]), float(d["boxsize_mpc_h"])
    box = str(args.box)

    spec = InterventionSpec(box=box, alpha=float(args.alpha), mode=args.mode,
                            kind=args.kind, dilate=args.dilate,
                            smooth=args.smooth, seed=args.seed)
    out_dir = paths.subdir("oracle_hr", box, create=True)
    row_path = out_dir / "interventions.jsonl"

    tj = paths.subdir("oracle_hr") / f"targets_{box}.json"
    if not tj.is_file():
        raise SystemExit(f"no targets for {box}: run oracle_select_targets.py")
    tdoc = json.loads(tj.read_text())
    targets = [Target.from_dict(x) for x in tdoc["targets"]]
    if not targets:
        raise SystemExit(f"{box} has no targets; nothing to intervene on")

    banner(f"{box} / {spec.tag}: {len(targets)} targets")
    t0 = time.time()
    base = np.asarray(load_field(cfg, box, "base", args.base_seed))
    hr = np.load(hr_path(cfg, box), mmap_mode="r")

    if spec.alpha == 0.0:
        field = base
        mask_frac, n_sites = 0.0, 0
        print("    alpha = 0: the frozen SR2 box, used as the anchor", flush=True)
    else:
        members = load_members(box)
        mask, n_sites = build_mask(targets, members, spec, ng)
        mask_frac = float((mask > 0.01).mean())
        print(f"    mask: {n_sites} sites, {mask_frac:.3e} of the volume above 0.01",
              flush=True)
        field = apply_intervention(base, hr, mask, spec.alpha, spec.mode)
        del mask

    work = paths.subdir("oracle_hr", box, spec.tag, create=True)
    cat = run_rockstar_on_field(
        field, work, tag=spec.tag,
        boxsize_kpc_h=box_l * 1000.0, redshift=float(d.get("redshift", 0.0)),
        overwrite=bool(args.overwrite),
    )
    for g in Path(work).glob("*.gadget2"):
        g.unlink(missing_ok=True)
    print(f"    Rockstar: {cat.n} objects in {(time.time() - t0) / 60:.1f} min",
          flush=True)

    # --- catalog -----------------------------------------------------------
    stats = direct_full_box_stats(cat, bins)
    ens = EnsembleSummary(stats["n_sub"], stats["n_host"], stats["occ_numerator"],
                          box_l ** 3, (f"{box}/{spec.tag}",))
    rec = recovery_report(targets, catalog(box, "hr"), cat, boxsize_mpc_h=box_l)
    n_rec = sum(1 for r in rec if r.get("recovered"))

    row = {
        "box": box, "kind": spec.kind, "mode": spec.mode, "alpha": spec.alpha,
        "tag": spec.tag, "n_targets": len(targets), "n_recovered": n_rec,
        "recovery_rate": n_rec / max(len(targets), 1),
        "n_as_host": sum(1 for r in rec if r.get("as_host")),
        "n_sites": n_sites, "mask_volume_fraction": mask_frac,
        "n_objects": int(cat.n),
        "occupation": stats["occupation"].tolist(),
        "n_host": stats["n_host"].tolist(),
        "occ_numerator": stats["occ_numerator"].tolist(),
        "n_sub": stats["n_sub"].tolist(),
        "recovery": rec,
    }

    rm_path = paths.AUDITS("tile_decomposition") / "tile_reward_model.json"
    if rm_path.is_file():
        rm = RewardModel.from_dict(json.loads(rm_path.read_text()))
        occ_bins = cfg.get("reward", {}).get("occupation", {})
        row.update(rm.scores(ens, occ_bins.get("reliable_host_bins")))
        row["occupation_gap"] = rm.occupation_gap(ens).tolist()
    else:
        print("    (no tile reward model yet; R_cat/R_occ omitted)", flush=True)

    # --- field constraints -------------------------------------------------
    if not args.skip_fields and spec.alpha != 0.0:
        vals = constraint_values(
            field, base, np.load(lr_path(cfg, box), mmap_mode="r"),
            hr=np.asarray(hr),
            scale_factor=int(d.get("scale_factor", 8)),
            boxsize_mpc_h=box_l,
            dis_norm_kpc_h=float(d.get("dis_norm_kpc_h", 6000.0)),
            redshift=float(d.get("redshift", 0.0)),
        )
        chk = check_constraints(vals, cons)
        row["constraints"] = vals
        row["feasible"] = bool(chk["feasible"])
        row["violations"] = chk["violations"]
        row["constraint_warnings"] = chk["warnings"]
        row["constraint_critical"] = chk["critical"]
        print(f"    constraints: feasible={chk['feasible']} {chk['violations']} "
              f"{chk['warnings']}", flush=True)

    row["wall_min"] = (time.time() - t0) / 60.0
    append_jsonl(row_path, row)
    write_json(work / "row.json", row)
    banner(f"{box}/{spec.tag}: recovered {n_rec}/{len(targets)} -> {row_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
