#!/usr/bin/env python
"""Stage 2: get the Lagrangian member particle ids of the selected hosts.

The editor's pool is *host members minus subhalo members*, and both sets have to
be exact. They are available exactly -- our GADGET2 ids are the flat Lagrangian
index and Rockstar preserves them -- but only from a run with
``FULL_PARTICLE_CHUNKS = 1``, which writes a ~7 GB ASCII member table for a
512^3 box.

Rather than reimplement that, this script drives the verified machinery in
``scripts/reward/rockstar_particles.py``: it re-runs the frozen SR2 box with the
member-id config, streams the table for exactly the halo ids stage 1 asked for,
and deletes the table inside the same job. The one addition here is the
reduction into a single per-box **pool cache** holding host ids, subhalo ids and
their member sets together, so every later stage opens one file.

The heavy work directory is redirected under ``$DMSR_LOCAL_EDITOR_ROOT`` with
``--out-dir``. A small tile-summary JSONL still lands in the reward line's
``tile_cache/`` -- that is a legitimate by-product of the shared code path (it is
the same frozen base box either line would summarise), and it is kilobytes.

    python scripts/reward/extract_editor_members.py --run-name le_a --box set8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _local_common import (  # noqa: E402
    PIPELINE, add_local_args, assert_no_final_boxes, banner, hosts_path,
    load_local_config, pool_path, write_json,
)

from cosmo_sr.reward import paths  # noqa: E402


def members_from_npz(path: Path) -> dict:
    z = np.load(path)
    ids, off, pid = z["halo_id"], z["offset"], z["particle_id"]
    return {int(h): pid[off[k]:off[k + 1]] for k, h in enumerate(ids)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_local_args(ap)
    ap.add_argument("--run-name", default="le_a")
    ap.add_argument("--box", required=True)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--reward-config", default="configs/reward/reward.yaml",
                    help="the config rockstar_particles.py reads for tile "
                         "geometry and bins; unrelated to the editor's own")
    ap.add_argument("--reuse", action="store_true", default=True)
    ap.add_argument("--overwrite", dest="reuse", action="store_false")
    args = ap.parse_args(argv)

    cfg = load_local_config(args)
    box = str(args.box)
    assert_no_final_boxes(cfg, [box], script="extract_editor_members.py")

    hp = hosts_path(args.run_name, box)
    if not hp.is_file():
        raise SystemExit(f"no host selection for {box}: run select_editor_hosts.py "
                         f"(expected {hp})")
    hosts_doc = json.loads(hp.read_text())
    hosts = hosts_doc["hosts"]
    if not hosts:
        print(f">>> {box}: stage 1 selected no hosts; nothing to extract.", flush=True)
        return 0

    want_hosts = {int(h["host_id"]) for h in hosts}
    out_npz = pool_path(args.run_name, box)
    if args.reuse and out_npz.is_file():
        # "Already present" is not the same as "correct for these hosts". The
        # cache is keyed by run name and box, but a run name can be re-selected
        # (different n_hosts, different bin split), and the previous cache would
        # then be reused for a host set it does not cover. Validate rather than
        # trust the filename.
        z = np.load(out_npz)
        have = {int(h): int(z["offset"][k + 1] - z["offset"][k])
                for k, h in enumerate(z["halo_id"])}
        short = sorted(h for h in want_hosts if have.get(h, 0) == 0)
        if not short:
            banner(f"{box}: pool cache already covers all {len(want_hosts)} "
                   f"selected hosts -> {out_npz}")
            return 0
        banner(f"{box}: existing pool cache is missing {len(short)} of "
               f"{len(want_hosts)} selected hosts ({short[:5]}...); rebuilding")

    work = paths.LOCAL_EDITOR("halos_particles", f"{box}__base__base", create=True)
    ids_json = hp.parent / f"halo_ids_{box}.json"

    # --- does an existing extraction actually cover THIS run's hosts? --------
    # The Rockstar work directory is shared across run names (it depends only on
    # the box), and ``rockstar_particles.py --reuse`` short-circuits on the
    # *tile-weights* cache -- it returns before re-streaming member ids for a
    # different halo-id list. So a second run name reusing a first one's
    # directory silently inherits the FIRST run's member ids.
    #
    # Measured: run `le_b` selected 8 hosts stratified over mass bins 2 and 3,
    # reused `le_a`'s bin-3-only extraction, and every bin-2 host came back with
    # zero member rows. `build_pools` skips such hosts, so the run would have
    # proceeded on half its hosts, all from one bin -- undoing the stratification
    # it existed to introduce, without failing.
    #
    # Coverage is therefore checked against the selected HOST ids before
    # deciding to reuse, and a shortfall forces a full re-run.
    mpath = work / f"{box}_base_members.npz"
    force = not args.reuse
    if args.reuse and mpath.is_file():
        have = members_from_npz(mpath)
        missing = sorted(h for h in want_hosts if have.get(h, np.zeros(0)).size == 0)
        if missing:
            force = True
            banner(f"{box}: the cached extraction is missing {len(missing)} of "
                   f"{len(want_hosts)} selected hosts ({missing[:5]}...); "
                   "re-running the member-id pass rather than proceeding on a "
                   "reduced host set")

    # Report the list actually streamed, not the selection. They differ on
    # purpose -- the extraction is a superset over every host qualifying for the
    # configured mass bins -- and a banner naming only the 8 selected hosts made
    # the log read as if a later re-selection would be covered when it was not.
    n_extract = len(json.loads(ids_json.read_text()).get("halo_ids", []))
    banner(f"{box}: Rockstar member-id pass, extracting {n_extract} objects "
           f"({len(hosts)} hosts selected now, the rest are the superset for "
           f"future re-selection)" + ("  [forced re-run]" if force else ""))
    import rockstar_particles  # noqa: E402  (same directory)
    rc = rockstar_particles.main([
        "--config", args.reward_config,
        "--box", box, "--source", "base",
        "--base-seed", str(int(args.base_seed)),
        "--out-dir", str(work),
        "--extract-members", str(ids_json),
    ] + (["--overwrite"] if force else []))
    if rc != 0:
        print(f">>> rockstar_particles.py returned {rc}; not writing a pool cache.",
              flush=True)
        return rc

    mpath = work / f"{box}_base_members.npz"
    if not mpath.is_file():
        raise SystemExit(f"member extraction produced no {mpath}")
    members = members_from_npz(mpath)

    # --- reduce to one per-box cache ----------------------------------------
    halo_ids, offsets, flat, is_host = [], [0], [], []
    missing = []
    for h in hosts:
        for hid, host_flag in ([(int(h["host_id"]), True)]
                               + [(int(s), False) for s in h["sub_ids"]]):
            m = members.get(hid)
            if m is None or m.size == 0:
                missing.append(hid)
                m = np.zeros(0, dtype=np.int64)
            halo_ids.append(hid)
            is_host.append(host_flag)
            flat.append(np.asarray(m, dtype=np.int64))
            offsets.append(offsets[-1] + int(m.size))

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        halo_id=np.asarray(halo_ids, dtype=np.int64),
        is_host=np.asarray(is_host, dtype=bool),
        offset=np.asarray(offsets, dtype=np.int64),
        particle_id=(np.concatenate(flat) if flat else np.zeros(0, dtype=np.int64)),
    )
    report = {
        "pipeline": PIPELINE, "box": box, "n_hosts": len(hosts),
        "n_objects": len(halo_ids),
        "n_particles": int(offsets[-1]),
        "objects_with_no_members": missing,
        "members_npz": str(mpath), "pool_cache": str(out_npz),
    }
    write_json(out_npz.parent / f"extract_{box}.json", report)
    # A missing SUBHALO is not fatal: it simply is not subtracted from the pool,
    # which makes the pool larger, not wrong. A missing HOST is fatal, because
    # build_pools drops that host and the run silently proceeds on fewer hosts
    # than it was configured for -- and if the dropped ones share a mass bin, on
    # a different scientific question than the one asked.
    missing_hosts = sorted(set(missing) & want_hosts)
    if missing:
        print(f"    !! {len(missing)} objects had no member rows: {missing[:5]}",
              flush=True)
    if missing_hosts:
        out_npz.unlink(missing_ok=True)
        print(f"!!! {len(missing_hosts)} SELECTED HOSTS have no member rows even "
              f"after a full re-run: {missing_hosts}", flush=True)
        print("!!! refusing to write a pool cache that would silently drop them. "
              "Check that these ids are in the frozen base catalog and that "
              f"{ids_json.name} lists them.", flush=True)
        return 4
    banner(f"{box}: pool cache -> {out_npz} ({offsets[-1]} particle ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
