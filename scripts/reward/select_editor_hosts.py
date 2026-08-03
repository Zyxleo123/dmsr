#!/usr/bin/env python
"""Stage 1: pick the frozen-SR2 hosts the local editor will operate on.

Pure arithmetic on the frozen SR2 catalog -- no halo finding, no fields, seconds
of CPU. Per box it writes:

* ``hosts/hosts_<box>.json``   -- the hosts, with everything later stages need;
* ``hosts/halo_ids_<box>.json`` -- those hosts **and all of their subhalos**,
  which is the list ``extract_editor_members.py`` feeds to Rockstar's member-id
  pass. Both are needed: the host's ids define the editable pool, the subhalos'
  ids define what has to be removed from it.

Hosts are drawn from the upper reliable mass bins (``1e13`` and ``3.16e13``),
where the SR2 occupation deficit is worst and where Gate B is decided, and are
forced apart by ``--min-separation`` so that one Rockstar run can score one
independent proposal per host. That batching is the whole economics of the
search: 8 well-separated hosts turn a ~15-minute halo run into 8 rewards.

``--report-bounds`` additionally summarises the satellite populations of the
**training** boxes -- mass ratios and radii -- so the action bounds in
``configs/reward/local_editor.yaml`` come from the catalogs rather than from the
plan's round numbers. That is aggregate training-catalog information, which
stage 3 explicitly permits; it never reads the search boxes' HR.

    python scripts/reward/select_editor_hosts.py --boxes set8,set9
    python scripts/reward/select_editor_hosts.py --report-bounds
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from _local_common import (  # noqa: E402
    PIPELINE, add_local_args, assert_no_final_boxes, assert_training_boxes,
    banner, base_catalog, hosts_path, load_local_config, reward_bins, run_dir,
    write_json,
)


def _min_image(d: np.ndarray, box: float) -> np.ndarray:
    return d - float(box) * np.round(d / float(box))


def candidate_hosts(cat, cfg: Dict) -> Dict:
    """Every host that *could* be selected, with its subhalos.

    The member-id extraction list, deliberately wider than the selection: see
    the call site for why. Returns host ids and the flat id list to extract.
    """
    h = cfg.get("hosts", {})
    edges = np.asarray(reward_bins().host_mass_edges, dtype=np.float64)
    want = set(int(b) for b in h.get("host_bins", [2, 3]))
    hosts, subs = cat.hosts(), cat.subhalos()
    by_parent: Dict[int, List[int]] = {}
    for k, pid in enumerate(subs.parent_ids):
        by_parent.setdefault(int(pid), []).append(k)
    hb = np.digitize(hosts.mvir, edges) - 1
    ok = np.nonzero(np.isin(hb, sorted(want))
                    & (hosts.num_p >= int(h.get("min_host_particles", 2000))))[0]
    host_ids = [int(hosts.ids[i]) for i in ok]
    ids = set(host_ids)
    for i in ok:
        ids.update(int(subs.ids[k]) for k in by_parent.get(int(hosts.ids[i]), []))
    return {"host_ids": host_ids, "halo_ids": sorted(ids)}


def select_hosts(cat, cfg: Dict, box: str) -> List[Dict]:
    """Well-separated hosts, **stratified across the requested mass bins**.

    Taking the ``n_hosts`` most massive qualifying hosts is the obvious rule and
    it is wrong here. Bin 3 (3.16e13-1e14) holds ~30 hosts per box, comfortably
    more than the quota, so a pure mass ordering fills every slot from bin 3 and
    bin 2 never gets one -- measured on set8/set9, 16 of 16 selected hosts came
    out of bin 3.

    That is fatal to the primary result rather than merely untidy. Edits only in
    bin-3 hosts can only move the bin-3 occupation, so at most one reliable bin
    improves, and the plan's criterion (``min_improved_reliable_bins: 2``) cannot
    be satisfied no matter how well the editor works.

    So the quota is split evenly across the requested bins, most massive first
    *within* each bin, and any shortfall in one bin is redistributed to the
    others rather than left unused.
    """
    h = cfg.get("hosts", {})
    bins = reward_bins()
    edges = np.asarray(bins.host_mass_edges, dtype=np.float64)
    want = set(int(b) for b in h.get("host_bins", [2, 3]))
    box_l = float(cfg["data"]["boxsize_mpc_h"])

    hosts = cat.hosts()
    subs = cat.subhalos()
    by_parent: Dict[int, List[int]] = {}
    for k, pid in enumerate(subs.parent_ids):
        by_parent.setdefault(int(pid), []).append(k)

    hb = np.digitize(hosts.mvir, edges) - 1
    ok = np.nonzero(
        np.isin(hb, sorted(want))
        & (hosts.num_p >= int(h.get("min_host_particles", 2000)))
    )[0]

    n_total = int(h.get("n_hosts", 8))
    sep = float(h.get("min_separation_mpc_h", 6.0))
    if bool(h.get("stratify_by_bin", True)) and len(want) > 1:
        # Round-robin over the bins, each bin's candidates in descending mass.
        # Interleaving rather than filling bin-by-bin means a separation
        # rejection late in one bin costs that bin one slot, not the quota.
        per_bin = {b: list(ok[hb[ok] == b][np.argsort(-hosts.mvir[ok[hb[ok] == b]])])
                   for b in sorted(want)}
        order: List[int] = []
        while any(per_bin.values()):
            for b in sorted(per_bin):
                if per_bin[b]:
                    order.append(int(per_bin[b].pop(0)))
    else:
        order = list(ok[np.argsort(-hosts.mvir[ok])])

    chosen: List[Dict] = []
    for i in order:
        if len(chosen) >= n_total:
            break
        if chosen:
            d = np.linalg.norm(
                _min_image(np.asarray([c["center_mpc"] for c in chosen])
                           - hosts.pos[i], box_l), axis=1)
            if float(d.min()) < sep:
                continue
        kids = by_parent.get(int(hosts.ids[i]), [])
        ki = np.asarray(kids, dtype=np.int64)
        rvir_mpc = float(hosts.rvir[i]) * 1e-3
        chosen.append({
            "box": box,
            "host_id": int(hosts.ids[i]),
            "host_bin": int(hb[i]),
            "center_mpc": [float(x) for x in hosts.pos[i]],
            "vel_kms": [float(x) for x in hosts.vel[i]],
            "rvir_mpc": rvir_mpc,
            "mvir": float(hosts.mvir[i]),
            "vmax": float(hosts.vmax[i]),
            "num_p": int(hosts.num_p[i]),
            "n_sub_current": int(ki.size),
            "sub_ids": [int(x) for x in subs.ids[ki]],
            "sub_mvir": [float(x) for x in subs.mvir[ki]],
            "sub_pos_mpc": [[float(v) for v in p] for p in subs.pos[ki]],
            "sub_rvir_mpc": [float(x) * 1e-3 for x in subs.rvir[ki]],
            # log10(M_sub / M_host) of what SR2 already produced -- the list the
            # stage-6 bootstrap subtracts its desired population against.
            "existing_log_mass_ratio": [
                float(np.log10(max(m, 1e-30) / max(hosts.mvir[i], 1e-30)))
                for m in subs.mvir[ki]
            ],
        })
    return chosen


def report_bounds(cfg: Dict) -> Dict:
    """Percentiles of the training-box satellite population, for the YAML bounds."""
    boxes = list(cfg.get("tokens", {}).get("library_boxes", []))
    assert_training_boxes(cfg, boxes, script="select_editor_hosts.py")
    box_l = float(cfg["data"]["boxsize_mpc_h"])
    ratios, radii, counts = [], [], []
    for b in boxes:
        cat = base_catalog(b)
        hosts, subs = cat.hosts(), cat.subhalos()
        by_parent: Dict[int, List[int]] = {}
        for k, pid in enumerate(subs.parent_ids):
            by_parent.setdefault(int(pid), []).append(k)
        for i in range(hosts.n):
            if int(hosts.num_p[i]) < int(cfg.get("tokens", {}).get("min_host_particles", 2000)):
                continue
            ki = np.asarray(by_parent.get(int(hosts.ids[i]), []), dtype=np.int64)
            counts.append(int(ki.size))
            if ki.size == 0:
                continue
            ratios.append(np.log10(np.maximum(subs.mvir[ki], 1e-30)
                                   / max(hosts.mvir[i], 1e-30)))
            radii.append(np.linalg.norm(
                _min_image(subs.pos[ki] - hosts.pos[i], box_l), axis=1)
                / max(float(hosts.rvir[i]) * 1e-3, 1e-9))
    r = np.concatenate(ratios) if ratios else np.zeros(0)
    d = np.concatenate(radii) if radii else np.zeros(0)
    q = [1, 5, 50, 95, 99]
    return {
        "boxes": boxes, "n_hosts": len(counts), "n_subhalos": int(r.size),
        "K_h_mean": float(np.mean(counts)) if counts else 0.0,
        "log_mass_ratio_percentiles": {
            str(p): (float(np.percentile(r, p)) if r.size else None) for p in q},
        "radius_rvir_percentiles": {
            str(p): (float(np.percentile(d, p)) if d.size else None) for p in q},
        "suggested_bounds": {
            "log_mass_ratio": ([float(np.percentile(r, 1)), float(np.percentile(r, 99))]
                               if r.size else None),
            "radius_rvir": ([float(np.percentile(d, 1)), float(np.percentile(d, 99))]
                            if d.size else None),
        },
        "note": "aggregate TRAINING-box catalogs only; paste into "
                "configs/reward/local_editor.yaml editor.bounds after reading "
                "the percentiles, not blindly.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_local_args(ap)
    ap.add_argument("--run-name", default="le_a")
    ap.add_argument("--boxes", default="", help="default: config search_boxes")
    ap.add_argument("--report-bounds", action="store_true",
                    help="summarise training-box satellite populations and exit")
    args = ap.parse_args(argv)

    cfg = load_local_config(args)
    if args.report_bounds:
        rep = report_bounds(cfg)
        out = write_json(run_dir(args.run_name, create=True) / "training_bounds.json", rep)
        banner(f"training-box bounds report -> {out}")
        for k, v in rep["suggested_bounds"].items():
            print(f"    {k}: {v}", flush=True)
        return 0

    boxes = ([b.strip() for b in args.boxes.split(",") if b.strip()]
             or list(cfg.get("search_boxes", [])))
    assert_no_final_boxes(cfg, boxes, script="select_editor_hosts.py")

    for box in boxes:
        cat = base_catalog(box)
        chosen = select_hosts(cat, cfg, box)
        if not chosen:
            print(f"    !! {box}: no host passed the mass/separation cuts", flush=True)
        doc = {"pipeline": PIPELINE, "box": box, "n_hosts": len(chosen),
               "hosts": chosen, "config": {"hosts": cfg.get("hosts", {})}}
        p = hosts_path(args.run_name, box)
        write_json(p, doc)

        # The extraction list is a SUPERSET: every host that qualifies for the
        # requested mass bins, not just the ones selected now.
        #
        # The expensive step downstream is the Rockstar member-id pass (~1-2 h
        # per box); streaming a longer id list out of the same table costs
        # almost nothing. Extracting only the selected hosts makes any later
        # change of selection -- a different `n_hosts`, a different bin split,
        # a re-run after a separation tweak -- require the whole pass again, and
        # the shared work directory made reusing the old one silently drop the
        # new hosts instead of failing. A superset removes the trap entirely.
        cand = candidate_hosts(cat, cfg)
        write_json(p.parent / f"halo_ids_{box}.json",
                   {"box": box,
                    "note": ("all hosts qualifying for hosts.host_bins plus their "
                             "subhalos, so re-selecting within these bins never "
                             "needs another Rockstar member-id pass"),
                    "n_candidate_hosts": len(cand["host_ids"]),
                    "n_selected_hosts": len(chosen),
                    "halo_ids": cand["halo_ids"]})
        banner(f"{box}: {len(chosen)} hosts "
               f"(bins {sorted({h['host_bin'] for h in chosen})}) -> {p}")
        for h in chosen:
            print(f"    host {h['host_id']:>8}  bin {h['host_bin']}  "
                  f"Mvir {h['mvir']:.3e}  Rvir {h['rvir_mpc']:.3f} Mpc/h  "
                  f"N_p {h['num_p']:>7}  N_sub {h['n_sub_current']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
