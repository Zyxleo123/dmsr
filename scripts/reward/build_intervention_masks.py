#!/usr/bin/env python
"""Lagrangian masks for the HR-directed interventions, on every proxy box.

Two stages, because they need different things and sit on opposite sides of the
HR labelling job:

``targets`` (CPU, seconds)
    Pick the clearly-missing HR subhalos this box's interventions will aim at,
    from the HR and frozen-SR2 catalogs the residual line already has for
    set0-15. Writes ``targets_<box>.json`` and ``halo_ids_<box>.json``. Pure
    arithmetic -- no fields, no halo finding.
``masks`` (CPU, minutes)
    Turn the targets' **Lagrangian member particle ids** into one smooth
    periodic mask per box, saved as ``intervention_masks/<box>.npz``. The member
    ids come from ``hr_members.npz``, which the HR candidate's labelling job
    extracts from the ``.particles`` table it is already streaming.

Why the ordering is what it is
------------------------------
Experiment 1 only ever ran on set8 and set9, and got its member ids from a
dedicated Rockstar run per box with ``FULL_PARTICLE_CHUNKS = 1``. Doing that for
twelve boxes would be twelve extra full-box halo-finder runs purely to build
masks. But the proxy dataset *already* runs exactly that configuration on every
box's HR candidate, so the ids are a second streaming pass over a table that is
about to be deleted rather than a new run. The cost of the interventions on ten
new boxes is therefore one cheap CPU job, at the price of a dependency:

    targets  ->  generate HR  ->  label HR  ->  masks  ->  generate intervention

``targets`` is deliberately *not* folded into ``masks``: the target list is the
scientific choice (which subhalos, which host bins, how far apart) and wants to
be readable and reviewable before eleven hours of Rockstar are spent on it.

    python scripts/reward/build_intervention_masks.py --stage targets
    python scripts/reward/build_intervention_masks.py --stage masks --box set0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from _sr2_direct import (  # noqa: E402
    add_direct_args, assert_not_sealed, banner, bins_of, dataset_of,
    direct_root, load_direct_config, write_json_atomic,
)

from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward import paths  # noqa: E402
from cosmo_sr.reward.oracle_hr import (  # noqa: E402
    Target, ids_to_lattice, lagrangian_mask, select_targets,
)
from cosmo_sr.reward.pipeline import existing_catalog  # noqa: E402


def targets_dir(*parts: str, create: bool = False) -> Path:
    return direct_root("intervention_targets", *parts, create=create)


def catalog(box: str, source: str):
    """The residual line's existing whole-box catalog for ``box``.

    Read-only, and read rather than recomputed: these are the same HR and frozen
    catalogs Experiment 1 selected its targets from, so an intervention built
    here is aimed at the same objects that study measured.
    """
    for root in ("halos", "halos_particles"):
        p = existing_catalog(paths.subdir(root, f"{box}__{source}__{source}"), source)
        if p is not None:
            return load_rockstar_ascii(p)
    raise SystemExit(
        f"no {source} catalog for {box}. HR comes from "
        f"hr_catalog_summaries_cpu.sbatch; base from the same job with SOURCES=base."
    )


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def stage_targets(cfg, args) -> int:
    bins = bins_of(cfg["_reward"])
    d = dataset_of(cfg)
    spec = dict(d.get("intervention", {}))
    box_l = float(cfg["_reward"]["data"]["boxsize_mpc_h"])
    host_bins = tuple(int(b) for b in spec.get("host_bins", (2, 3)))

    sparse = cfg["_reward"].get("reward", {}).get("occupation", {}) \
        .get("sparse_host_bins", [4])
    bad = sorted(set(host_bins) & {int(s) for s in sparse})
    if bad:
        raise SystemExit(
            f"host bins {bad} are configured as sparse/evaluation-only; targets "
            "drawn there cannot support a decision. Drop them from host_bins.")

    boxes = args.boxes or [str(b) for b in d.get("boxes", [])]
    assert_not_sealed(cfg, boxes)
    out_dir = targets_dir(create=True)
    total, empty = 0, []
    for box in boxes:
        t = select_targets(
            catalog(box, "hr"), catalog(box, "base"),
            host_mass_edges=bins.host_mass_edges,
            host_bins=host_bins,
            boxsize_mpc_h=box_l,
            min_sub_particles=int(spec.get("min_sub_particles", 50)),
            max_targets=int(spec.get("max_targets", 24)),
            min_separation_mpc_h=float(spec.get("min_separation_mpc_h", 6.0)),
            seed=int(args.seed))
        banner(f"{box}: {len(t)} targets")
        if not t:
            empty.append(box)
        for x in t:
            print(f"  sub {x.hr_sub_id:>8d} in host {x.hr_host_id:>8d} "
                  f"bin {x.host_bin} M_host {x.host_mvir:9.3e} "
                  f"M_sub {x.sub_mvir:9.3e} n_p {x.sub_num_p:>6d}")
        write_json_atomic(out_dir / f"targets_{box}.json", {
            "box": box, "host_bins": list(host_bins), "boxsize_mpc_h": box_l,
            "min_separation_mpc_h": float(spec.get("min_separation_mpc_h", 6.0)),
            "targets": [x.to_dict() for x in t]})
        # Hosts are needed too: the mask is dilated within the host's own
        # Lagrangian footprint, and any control drawn later comes from it.
        ids = sorted({x.hr_sub_id for x in t} | {x.hr_host_id for x in t})
        write_json_atomic(out_dir / f"halo_ids_{box}.json",
                          {"box": box, "halo_ids": ids})
        total += len(t)

    banner(f"{total} targets over {len(boxes)} boxes -> {out_dir}")
    if empty:
        print(f">>> {empty} produced NO targets. An empty selection is a selection "
              ">>> failure, not a negative result: those boxes will have no "
              ">>> intervention candidates and will contribute only hr/frozen "
              ">>> rows, which is the dataset failure mode this line exists to "
              ">>> avoid. Loosen min_sub_particles or min_separation_mpc_h.")
    return 0


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #
def members_from_npz(path: Path) -> Dict[int, np.ndarray]:
    z = np.load(path)
    ids, off, pid = z["halo_id"], z["offset"], z["particle_id"]
    return {int(h): pid[off[k]:off[k + 1]] for k, h in enumerate(ids)}


def stage_masks(cfg, args) -> int:
    from collect_catalog_proxy_data import candidate_dir

    d = dataset_of(cfg)
    spec = dict(d.get("intervention", {}))
    ng = int(cfg["geometry"]["ng_hr"])
    box = args.box
    assert_not_sealed(cfg, [box])

    tj = targets_dir() / f"targets_{box}.json"
    if not tj.is_file():
        print(f">>> MISSING INPUT: {tj}")
        print(">>> produced by: build_intervention_masks.py --stage targets")
        return 0
    targets = [Target.from_dict(x) for x in json.loads(tj.read_text())["targets"]]
    if not targets:
        print(f">>> GATE FAILED: {box} has no targets, so it can have no HR-directed")
        print(">>> interventions. See the --stage targets log.")
        return 0

    mem_npz = candidate_dir(box, "hr") / "hr_members.npz"
    if not mem_npz.is_file():
        print(f">>> MISSING INPUT: {mem_npz}")
        print(">>> produced by: collect_catalog_proxy_data.py --stage label")
        print(f">>>              --box {box} --source hr  (it extracts the member")
        print(">>>              ids from the .particles table in the same pass)")
        return 0
    members = members_from_npz(mem_npz)

    sites: List[np.ndarray] = []
    skipped: List[int] = []
    for t in targets:
        sub = members.get(int(t.hr_sub_id))
        if sub is None or sub.size == 0:
            skipped.append(int(t.hr_sub_id))
            continue
        sites.append(ids_to_lattice(sub, ng))
    if not sites:
        print(f">>> GATE FAILED: none of {box}'s {len(targets)} targets has member")
        print(">>> ids in hr_members.npz. The target list and the HR catalog that")
        print(">>> was labelled do not describe the same run.")
        return 0

    # Union first, then dilate and smooth: `select_targets` forces targets
    # >= 6 Mpc/h apart (~30 HR cells), far beyond the 1.5-cell smoothing, so the
    # blobs are disjoint and smooth(union) == max(smooth(each)) exactly -- while
    # per-target masking would be ~33 periodic rolls of a 512^3 volume per target.
    all_sites = np.concatenate(sites, axis=0)
    mask = lagrangian_mask(all_sites, ng,
                           dilate=int(spec.get("dilate", 2)),
                           smooth=float(spec.get("smooth", 1.5)))
    frac = float((mask > 0.01).mean())
    out = direct_root("intervention_masks", create=True) / f"{box}.npz"
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, mask=mask.astype(np.float32),
                        n_sites=np.int64(all_sites.shape[0]),
                        n_targets=np.int64(len(targets) - len(skipped)))
    tmp.replace(out)
    write_json_atomic(out.with_suffix(".json"), {
        "box": box, "n_targets": len(targets), "n_used": len(targets) - len(skipped),
        "skipped_targets_without_members": skipped,
        "n_sites": int(all_sites.shape[0]),
        "mask_volume_fraction_above_0.01": frac,
        "mask_max": float(mask.max()),
        "dilate": int(spec.get("dilate", 2)),
        "smooth": float(spec.get("smooth", 1.5)),
        "hr_members_path": str(mem_npz),
        "targets_path": str(tj),
    })
    banner(f"{box}: {all_sites.shape[0]} sites from {len(targets) - len(skipped)} "
           f"targets, {frac:.3e} of the volume above 0.01 -> {out}")
    if skipped:
        print(f">>> {len(skipped)} targets had no member ids and were skipped: "
              f"{skipped[:5]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--stage", required=True, choices=("targets", "masks"))
    ap.add_argument("--box", default="", help="one box, for --stage masks")
    ap.add_argument("--boxes", nargs="*", default=None,
                    help="override the configured box list for --stage targets")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    if args.stage == "targets":
        return stage_targets(cfg, args)
    if not args.box:
        raise SystemExit("--box is required for --stage masks")
    return stage_masks(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
