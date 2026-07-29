#!/usr/bin/env python
"""Experiment 1, step 1: pick the missing HR subhalos the oracle will try to restore.

Pure arithmetic on the frozen HR and SR2 catalogs -- no halo finding, no fields,
seconds of CPU. It writes two files:

* ``targets_<box>.json``   -- the targets, with everything the later stages need;
* ``halo_ids_<box>.json``  -- the target subhalos **and their hosts**, which is
  the list ``rockstar_particles.py --extract-members`` needs so that the masks
  can be built from real Lagrangian member ids in the same pass that produces
  the Experiment-0 tile weights.

Targets are drawn from the upper reliable host bins (``1e13`` and ``3.16e13``)
because that is where Gate B has to be decided; the sparse ``1e14`` bin is
excluded for the reason the covariance audit gives. They are also forced apart
by ``--min-separation`` so that many targets can share one intervention box: a
full-box Rockstar run per target would make the experiment cost an order of
magnitude more than it is worth.

    python scripts/reward/oracle_select_targets.py --boxes set8,set9
"""
from __future__ import annotations

import argparse

from _common import (  # noqa: E402
    add_common_args, banner, bins_of, load_reward_config, paths, write_json,
)

from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.reward.oracle_hr import select_targets  # noqa: E402
from cosmo_sr.reward.pipeline import existing_catalog  # noqa: E402


def catalog(box: str, source: str):
    for root in ("halos", "halos_particles"):
        p = existing_catalog(paths.subdir(root, f"{box}__{source}__{source}"), source)
        if p is not None:
            return load_rockstar_ascii(p)
    raise SystemExit(
        f"no {source} catalog for {box}. HR comes from "
        f"hr_catalog_summaries_cpu.sbatch; base from the same job with SOURCES=base."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--max-targets", type=int, default=24)
    ap.add_argument("--min-sub-particles", type=int, default=50,
                    help="resolution floor for a target; 50 particles is ~2.9e10 "
                         "Msun/h, comfortably above Rockstar's 20-particle cut so "
                         "a recovered object is detectable rather than marginal")
    ap.add_argument("--min-separation", type=float, default=6.0,
                    help="Mpc/h between selected hosts, so batched interventions "
                         "in one box cannot interact")
    ap.add_argument("--host-bins", default="2,3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_reward_config(args)
    bins = bins_of(cfg)
    box_l = float(cfg["data"]["boxsize_mpc_h"])
    host_bins = tuple(int(b) for b in args.host_bins.split(",") if b.strip())
    out_dir = paths.subdir("oracle_hr", create=True)

    sparse = cfg.get("reward", {}).get("occupation", {}).get("sparse_host_bins", [4])
    bad = sorted(set(host_bins) & set(int(s) for s in sparse))
    if bad:
        raise SystemExit(
            f"host bins {bad} are configured as sparse/evaluation-only; an oracle "
            "result there cannot feed a Gate B decision. Drop them from --host-bins."
        )

    total = 0
    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        hr, sr = catalog(box, "hr"), catalog(box, "base")
        t = select_targets(
            hr, sr,
            host_mass_edges=bins.host_mass_edges,
            host_bins=host_bins,
            boxsize_mpc_h=box_l,
            min_sub_particles=int(args.min_sub_particles),
            max_targets=int(args.max_targets),
            min_separation_mpc_h=float(args.min_separation),
            seed=int(args.seed),
        )
        banner(f"{box}: {len(t)} targets")
        if not t:
            print("  no clearly-missing subhalos passed the cuts. Loosen "
                  "--min-sub-particles or --min-separation before concluding "
                  "anything -- an empty selection is a selection failure, not a "
                  "negative oracle result.", flush=True)
        head = (f"{'hr_sub':>8s} {'host':>8s} {'bin':>4s} {'M_host':>10s} "
                f"{'M_sub':>10s} {'n_p':>6s} {'r/Rvir':>7s}")
        print(head)
        for x in t:
            print(f"{x.hr_sub_id:>8d} {x.hr_host_id:>8d} {x.host_bin:>4d} "
                  f"{x.host_mvir:10.3e} {x.sub_mvir:10.3e} {x.sub_num_p:>6d} "
                  f"{x.r_rvir:7.2f}")

        write_json(out_dir / f"targets_{box}.json",
                   {"box": box, "host_bins": list(host_bins),
                    "boxsize_mpc_h": box_l,
                    "min_separation_mpc_h": float(args.min_separation),
                    "targets": [x.to_dict() for x in t]})
        # Hosts are needed too: the equal-count random control draws its sites
        # from the host's own Lagrangian footprint.
        ids = sorted({x.hr_sub_id for x in t} | {x.hr_host_id for x in t})
        write_json(out_dir / f"halo_ids_{box}.json", {"box": box, "halo_ids": ids})
        total += len(t)

    banner(f"{total} targets total -> {out_dir}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
