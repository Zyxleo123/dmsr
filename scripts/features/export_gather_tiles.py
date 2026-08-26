#!/usr/bin/env python
"""Write a finished member-gather run's HELD-OUT tiles in the splice layout. GPU, minutes.

Why this exists
---------------
``docs/sr2_member_gather_training.md`` section 11 item 1 says it in capitals:
NOTHING HAS BEEN GATED. Every number the trainer reports is a member-set
statistic scoring itself, and ``tile-overfit-proxy-exploitation`` measured such a
statistic reaching +255 while the halo finder showed no gain at all.

The gate itself is already built and already generic --
``scripts/reward/splice_gather_field.py`` loops over whatever tile list it is
handed, ``flow_rockstar_catalog_cpu.sbatch`` takes a box and a tag, and
``compare_gather_catalog.py --targets-json`` scores an arbitrary list of halo
ids. What was missing is only the hand-off: ``free_field_gather.py`` writes
``tiles.npz`` and ``subhalos.json``, and the trainer never did, so there was
nothing for the chain to splice. This writes exactly those two files from a
``tuned.pt``, in the same ``(tiles, out, frozen, hr)`` layout, and then the
existing chain runs unchanged.

Held out means held out
-----------------------
The boxes here (``set9``, ``set10`` by default) were never trained on and were
never used to develop the line. That matters more than usual for this objective:
at the last step of the 2026-08-24 arms the operator's held-out ``bound_hard``
was less than half its training value (``self`` 0.140 against 0.322) and its
high-k ratio was four times worse out of sample than in it (3.87 against 0.89).
An in-sample gate would have reported neither.

One Rockstar run per box, not per host
--------------------------------------
The eight held-out hosts of a box own 32 tiles between them with **no overlap**
(verified from ``pool.json``), so their tiles splice into a single field and one
full-box Rockstar run scores all of them -- 1,127 supervised targets for
``set9``. The per-host sections of ``compare_gather_catalog.py`` still describe
whichever host ``HG_HOST_ID`` names; the supervised-target rate, which is the
number this line is actually asking about, covers every host in the file.

The pool is rebuilt, never cached
---------------------------------
The run's own ``summary.json`` config is replayed through the trainer's parser so
the member sets are assembled under the same purity, softening, background and
``min_num_p`` settings the run was evaluated under. A wrong reachable reference
is invisible -- every statistic still looks fine and means something else --
which is why ``build_pool`` refuses to cache and why this replays rather than
re-declares.

Usage
-----
    python scripts/features/export_gather_tiles.py \
        --run-dir $DMSR_REWARD_ROOT/member_gather/all_blocks_self --box set9

Writes ``<run-dir>/holdout_<box>/{tiles.npz,subhalos.json,export.json}``, which
is what ``HG_RUN_DIR`` should then point at.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, write_json  # noqa: E402
from _sr2_direct import (  # noqa: E402
    geometry_of, load_direct_config, model_path_of, phase_space_config_of,
    soft_config_of,
)
from free_field_gather import member_config_of  # noqa: E402
from overfit_host_mse import BOXSIZE, NG_HR, TILE  # noqa: E402
# The trainer's own pool builder, forward and parser. Three separate reasons,
# all the same reason: a second definition here would be a different experiment.
from finetune_member_gather import (  # noqa: E402
    build_parser, build_pool, host_forward,
)

from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402


def replay_args(run_dir: Path, box: str, argv_extra: List[str]):
    """The run's own config, back through the trainer's parser.

    ``summary.json`` stores the parsed namespace, so every key is a real option
    and round-trips as ``--dashed-name value``. Store-true flags are emitted only
    when they were set. Anything the current parser no longer accepts is dropped
    loudly rather than silently: a config key that has been renamed since the run
    is exactly the case where a quiet default would change the member sets.
    """
    cfg = json.loads((run_dir / "summary.json").read_text())["config"]
    ap = build_parser()
    known = {a.dest: a for a in ap._actions}
    argv: List[str] = []
    dropped: List[str] = []
    for k, v in cfg.items():
        act = known.get(k)
        if act is None or not act.option_strings:
            dropped.append(k)
            continue
        flag = max(act.option_strings, key=len)
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif v is None:
            continue
        elif isinstance(v, (list, tuple)):
            # A repeatable option (`--set key=value`, an append action). An
            # empty list must emit NOTHING: `str([])` is the literal "[]", and
            # `apply_overrides` rejects it as not key=value -- which is how the
            # first launch died three seconds in (jobs 36243-36256).
            for item in v:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(v)]
    if dropped:
        print(f"  config keys not in the current parser, IGNORED: {dropped}")
    # The pool must hold this box and nothing else: build_pool loads a 537 MB
    # owner array per box and we need exactly one.
    argv += ["--train-boxes", "", "--holdout-boxes", box, "--holdout-hosts", ""]
    argv += list(argv_extra)
    args = ap.parse_args(argv)
    return args


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="a finished member-gather run (holds tuned.pt and "
                         "summary.json)")
    ap.add_argument("--box", required=True,
                    help="ONE held-out box, e.g. set9. Its hosts are all "
                         "held out by construction -- the split is by box.")
    ap.add_argument("--out-dir", default="",
                    help="default <run-dir>/holdout_<box>")
    ap.add_argument("--device", default="")
    ap.add_argument("--max-hosts", type=int, default=0,
                    help="export only the first N hosts of the box. 0 = all. "
                         "A smaller splice is a cheaper shakeout, not a "
                         "different measurement.")
    args, extra = ap.parse_known_args(argv)

    run_dir = Path(args.run_dir)
    ckpt_p = run_dir / "tuned.pt"
    if not ckpt_p.is_file():
        raise SystemExit(f"no tuned.pt under {run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir \
        else run_dir / f"holdout_{args.box}"
    out_dir.mkdir(parents=True, exist_ok=True)

    banner(f"export held-out tiles: {run_dir.name} -> {args.box}")
    targs = replay_args(run_dir, args.box, extra)
    if args.device:
        targs.device = args.device

    cfg = load_direct_config(targs)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    pscfg = phase_space_config_of(cfg)
    mcfg = member_config_of(targs)
    device = torch.device(targs.device if targs.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(targs.seed))

    in_chan = int(cfg.get("model", {}).get("in_chan", 6))
    out_chan = int(cfg.get("model", {}).get("out_chan", 6))
    frozen = load_controlled_generator(
        model_path_of(cfg), in_chan=in_chan, out_chan=out_chan,
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    for p in frozen.parameters():
        p.requires_grad_(False)

    tuned = load_controlled_generator(
        model_path_of(cfg), in_chan=in_chan, out_chan=out_chan,
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    tuned.load_state_dict(ck["model"], strict=True)
    for p in tuned.parameters():
        p.requires_grad_(False)
    tuned.eval()
    print(f"  loaded {ckpt_p.name}: rung={ck.get('rung')} step={ck.get('step')} "
          f"{len(ck.get('trained_names', []))} trained tensors")

    # The checkpoint is a FULL state dict on purpose, so a silently mismatched
    # architecture would load clean and generate a different field. Confirm the
    # two models actually differ, and only where the rung says.
    n_diff = sum(1 for (n, a), (_, b) in
                 zip(tuned.state_dict().items(), frozen.state_dict().items())
                 if not torch.equal(a.cpu(), b.cpu()))
    print(f"  {n_diff} tensors differ from frozen "
          f"(rung claims {len(ck.get('trained_names', []))})")
    if n_diff == 0:
        raise SystemExit(
            "tuned and frozen are bit-identical: the checkpoint did not load, "
            "or the run trained nothing. Refusing to export a control as a "
            "candidate.")

    print(f"\nbuilding the pool for {args.box} "
          f"(one owner-array load)", flush=True)
    t0 = time.time()
    tasks = build_pool(targs, cfg, geom, scfg, pscfg, mcfg, frozen, device,
                       [args.box])
    if not tasks:
        raise SystemExit(
            f"no hosts selected for {args.box}. The owner array is missing "
            f"(scripts/slurm/submit_owner_arrays.sh) or --min-log-mvir is "
            f"above its most massive host.")
    if args.max_hosts > 0:
        tasks = tasks[: int(args.max_hosts)]
    print(f"  {len(tasks)} hosts, {sum(t.sets.n_sets for t in tasks)} "
          f"supervised sets [{time.time() - t0:.0f}s]", flush=True)

    kw = dict(ng_hr=NG_HR, tile_hr=TILE, boxsize_mpc_h=BOXSIZE,
              dis_scale_mpc_h=float(scfg.dis_norm_kpc_h) * 1e-3,
              vel_scale_kms=float(pscfg.vel_norm_km_s))

    tiles: List[int] = []
    out_t: List[np.ndarray] = []
    frz_t: List[np.ndarray] = []
    hr_t: List[np.ndarray] = []
    rows: List[Dict] = []
    halo_id: List[int] = []
    num_p: List[int] = []
    n_live: List[int] = []
    per_host: List[Dict] = []

    for task in tasks:
        with torch.no_grad():
            # The sixth return is the HOST-preservation term (`--w-host-sets`,
            # the hostguard arm). It is a loss, not a field: the export writes
            # tiles, so it is deliberately dropped here. Named rather than `_`
            # so a future seventh return fails loudly instead of vanishing.
            cand, base, hr, gather, diag, host_term = host_forward(
                tuned, task, geom, kw, mcfg, device, frozen_model=frozen)
            del host_term
        overlap = sorted(set(tiles) & set(task.tiles))
        if overlap:
            # Two hosts sharing a tile would splice one host's field over the
            # other's and both would be scored against it. The pool has none;
            # if that ever changes this must become a merge, not a silent last
            # writer wins.
            raise SystemExit(
                f"{task.key} shares tiles {overlap} with an already-exported "
                f"host. Splicing would overwrite; export these hosts to "
                f"separate directories instead.")
        tiles += [int(t) for t in task.tiles]
        out_t.append(cand.detach().cpu().numpy())
        frz_t.append(base.detach().cpu().numpy())
        hr_t.append(hr.detach().cpu().numpy())
        for r in diag["rows"]:
            r = dict(r)
            r["host"] = task.key
            rows.append(r)
        halo_id += [int(x) for x in task.sets.halo_id]
        num_p += [int(x) for x in task.sets.num_p]
        n_live += [int(x) for x in task.sets.n_live]
        per_host.append({
            "key": task.key, "tiles": [int(t) for t in task.tiles],
            "n_sets": int(task.sets.n_sets),
            "log_mvir": float(task.sel.log_mvir),
            "halo_id": int(task.sel.halo_id),
            "gather": float(gather.detach()),
            "bound_hard": diag["median_bound_hard"],
            "virial": diag["median_virial"],
            "centre_offset_radii": diag["median_centre_offset_radii"],
            "r_rms_over_hr": diag["median_r_rms_over_hr"],
            "sigma_v_over_hr": diag["median_sigma_v_over_hr"],
        })
        print(f"    {task.key}: {task.sets.n_sets:4d} sets, "
              f"bound {diag['median_bound_hard']:.3f}, "
              f"dx {diag['median_centre_offset_radii']:.2f}r, "
              f"tiles {task.tiles}", flush=True)

    np.savez_compressed(out_dir / "tiles.npz",
                        tiles=np.array(tiles, dtype=np.int64),
                        out=np.concatenate(out_t, axis=0),
                        frozen=np.concatenate(frz_t, axis=0),
                        hr=np.concatenate(hr_t, axis=0))
    write_json(out_dir / "subhalos.json",
               {"rows": rows, "halo_id": halo_id, "num_p": num_p,
                "n_live": n_live})
    write_json(out_dir / "export.json", {
        "ok": True, "run_dir": str(run_dir), "box": args.box,
        "checkpoint": {"rung": ck.get("rung"), "step": ck.get("step"),
                       "n_trained_tensors": len(ck.get("trained_names", [])),
                       "n_tensors_differing": int(n_diff)},
        "n_hosts": len(tasks), "n_tiles": len(tiles),
        "n_sets": len(rows),
        "spliced_volume_fraction": len(tiles) / float((NG_HR // TILE) ** 3),
        "per_host": per_host,
        "centre_mode": str(getattr(targs, "centre_mode", "")),
    })
    frac = 100.0 * len(tiles) / float((NG_HR // TILE) ** 3)
    print(f"\n  wrote {out_dir}/tiles.npz  "
          f"{len(tiles)} tiles ({frac:.2f}% of the box), "
          f"{len(rows)} supervised targets")
    print(f"  gate it with:\n"
          f"    HG_BOX={args.box} HG_HOST_ID={tasks[0].sel.halo_id} \\\n"
          f"    HG_RUN_DIR={out_dir} bash scripts/slurm/submit_gather_rockstar.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
