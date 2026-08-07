#!/usr/bin/env python
"""Stage 4: propose one CEM round's actions, and aggregate the round that ran.

Two stages, one script, because they are two halves of the same bookkeeping and
splitting them across files is how the two halves come to disagree about what a
candidate index means.

``--stage propose``
    Draw this round's action vectors and write ``round_<r>_actions.json``, which
    ``run_editor_candidates.py`` reads. Round 0 draws from the initial
    distribution; later rounds draw from the state produced by replaying every
    scored round on disk.
``--stage aggregate``
    Read the candidate rows the round produced, record the rewards against the
    round manifest, refit, and report Gate 1.

The population is per *proposal*, not per candidate box
-------------------------------------------------------
Each candidate box carries one independently-sampled action per host, and each
of those gets its own object-level reward. So a round of 28 boxes over 8 hosts
is **224** CEM samples, not 28 -- which is what makes 2-3 rounds a viable budget
at ~15 minutes of Rockstar per box.

One search per mode
-------------------
``disp``, ``both`` and ``vel`` get separate CEM states. They pin different
coordinates of the same 12-dimensional vector, so pooling them would fit a
distribution over ``velocity_cooling`` from samples where it was held at zero,
and would then sample that coordinate in the next round as if it had been
measured. Separate states also make the mode comparison a comparison rather than
an artefact of the mixing ratio.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np

from _local_common import (  # noqa: E402
    PIPELINE, add_local_args, banner, cem_dir, codec_for, load_local_config,
    mode_plan, read_jsonl, rows_path, run_dir, write_json,
)

from cosmo_sr.reward.cem import CEMRun, CEMState  # noqa: E402
from cosmo_sr.reward.local_reward import (  # noqa: E402
    ProposalOutcome, gate1_verdict, is_scientific_success,
)

MODES = ("disp", "both", "vel")


def initial_state(cfg: Dict, mode: str, n_samples: int) -> CEMState:
    c = cfg.get("cem", {})
    return CEMState.initial(
        codec_for(cfg, mode).dim,
        # A per-mode seed offset so the three searches are independent rather
        # than three views of the same normal deviates.
        seed=int(c.get("seed", 0)) * 1000 + MODES.index(mode),
        sigma=float(c.get("init_sigma", 1.5)),
        n_samples=int(n_samples),
        elite_frac=float(c.get("elite_frac", 0.2)),
        min_elites=int(c.get("min_elites", 3)),
        sigma_floor=float(c.get("sigma_floor", 0.25)),
        explore_mix=float(c.get("explore_mix", 0.25)),
    )


def _runs(cfg: Dict, run_name: str, per_mode_n: Dict[str, int]):
    root = cem_dir(run_name, create=True)
    out = {}
    for m in MODES:
        if per_mode_n.get(m, 0) <= 0:
            continue
        run = CEMRun(root=root, name=m)
        state, infos = run.resume(initial_state(cfg, m, per_mode_n[m]))
        out[m] = (run, state, infos)
    return out


def _per_mode_counts(cfg: Dict, n_cand: int, n_hosts: int) -> Dict[str, int]:
    """Samples each mode's CEM state must supply this round."""
    modes = mode_plan(cfg, n_cand)
    return {m: modes.count(m) * int(n_hosts) for m in MODES}


def stage_propose(cfg: Dict, args) -> int:
    n_cand = int(args.candidates or cfg.get("cem", {}).get("candidates_per_round", 28))
    n_hosts = int(args.n_hosts or cfg.get("cem", {}).get("proposals_per_box", 8))
    per_mode = _per_mode_counts(cfg, n_cand, n_hosts)
    runs = _runs(cfg, args.run_name, per_mode)

    modes = mode_plan(cfg, n_cand)
    drawn: Dict[str, np.ndarray] = {}
    for m, (run, state, _) in runs.items():
        if int(state.round_index) != int(args.round):
            raise SystemExit(
                f"mode {m}: replaying the manifests puts the search at round "
                f"{state.round_index}, but --round {args.round} was requested. "
                "Aggregate the missing round first; do not skip one."
            )
        z = state.sample(per_mode[m])
        run.write_round(state, z, extra={"mode": m, "n_hosts": n_hosts,
                                         "n_candidates": modes.count(m)})
        drawn[m] = z

    cursor = {m: 0 for m in runs}
    candidates: List[Dict] = []
    for i, m in enumerate(modes):
        z = drawn[m][cursor[m]:cursor[m] + n_hosts]
        cursor[m] += n_hosts
        candidates.append({"index": int(i), "mode": m, "z": z.tolist()})

    out = run_dir(args.run_name, "cem", create=True) / \
        f"round_{int(args.round):03d}_actions.json"
    write_json(out, {"pipeline": PIPELINE, "run_name": args.run_name,
                     "round": int(args.round), "n_hosts": n_hosts,
                     "mode_counts": {m: modes.count(m) for m in MODES},
                     "candidates": candidates})
    banner(f"round {args.round}: {len(candidates)} candidate manifests -> {out}")
    for m, (_, state, _) in runs.items():
        print(f"    {m:5s} n={per_mode[m]:4d}  |mean|={np.abs(state.mean).mean():.3f}  "
              f"mean std={state.std.mean():.3f}", flush=True)
    return 0


def _refit_modes(cfg: Dict, args, cem_rows: List[Dict]) -> Dict[str, Dict]:
    """Replay one round's CEM rows into the per-mode states and refit.

    Split out from :func:`stage_aggregate` because only this half needs the
    ``cem`` arm. The Gate 1 verdict is computed from every scored row, so a
    round with no CEM rows -- which is exactly what the ``gate1`` stage
    produces, random plus its controls and nothing else -- must still reach it.
    """
    n_hosts = int(cfg.get("cem", {}).get("proposals_per_box", 8))
    n_cand = int(args.candidates or cfg.get("cem", {}).get("candidates_per_round", 28))
    per_mode = _per_mode_counts(cfg, n_cand, n_hosts)
    runs = _runs(cfg, args.run_name, per_mode)

    # --- collect (z, reward) per mode, in manifest order ---------------------
    # Reconstructed from the round manifest rather than from row order: a shard
    # that reran, or arrived out of order, must not permute the population.
    manifest = json.loads(
        (run_dir(args.run_name, "cem") /
         f"round_{int(args.round):03d}_actions.json").read_text())
    by_index = {int(c["index"]): c for c in manifest["candidates"]}
    per_mode_rows: Dict[str, List[Dict]] = {m: [] for m in runs}

    for r in cem_rows:
        idx = int(r["index"])
        m = str(by_index[idx]["mode"])
        z = np.asarray(r["z"], dtype=np.float64)
        outs = [ProposalOutcome.from_dict(o) for o in r.get("outcomes", [])]
        if len(outs) != z.shape[0]:
            raise SystemExit(
                f"candidate {r['candidate_id']}: {z.shape[0]} action rows but "
                f"{len(outs)} outcomes; the row is inconsistent")
        for k, o in enumerate(outs):
            per_mode_rows[m].append({
                "box": r["box"], "index": idx, "host_slot": k,
                "z": z[k], "reward": float(o.reward),
                "feasible": bool(o.feasible_field),
                "detected": bool(o.detected),
                "success": bool(is_scientific_success(o)),
                "proxy": float(o.compactness_proxy),
                "host_id": int(o.base_host_id),
                # An edit whose window contained none of its claimed particles
                # is a no-op that still cost a Rockstar run. It scores zero for
                # a reason unrelated to whether the action space works, so it is
                # counted separately rather than pooled into "failed".
                "inert": int(o.n_active_particles) == 0,
            })

    modes: Dict[str, Dict] = {}
    for m, (run, state, _) in runs.items():
        recs = per_mode_rows.get(m, [])
        if not recs:
            modes[m] = {"n": 0, "note": "no scored rows this round"}
            continue
        z = np.stack([r["z"] for r in recs])
        rew = np.asarray([r["reward"] for r in recs], dtype=np.float64)
        feas = np.asarray([r["feasible"] for r in recs], dtype=bool)
        n_succ = int(sum(1 for r in recs if r["success"]))

        # Tie-break with the proxy ONLY when every candidate scored identically:
        # that is the situation the proxy exists for, and applying it anywhere
        # else would let a proxy score outrank a real detection.
        used_proxy = False
        if n_succ == 0 and float(np.max(rew) - np.min(rew)) <= 1e-12:
            px = np.asarray([r["proxy"] for r in recs], dtype=np.float64)
            if np.isfinite(px).any() and float(np.nanmax(px) - np.nanmin(px)) > 0:
                rew = np.nan_to_num(px, nan=float(np.nanmin(px)))
                used_proxy = True

        run.record_rewards(int(args.round), rew.tolist(), feas.tolist(),
                           extra={"n_success": n_succ, "proxy_tiebreak": used_proxy,
                                  "hosts": [r["host_id"] for r in recs],
                                  "boxes": [r["box"] for r in recs]})
        nxt, info = state.update(z, rew, feas)
        n_inert = int(sum(1 for r in recs if r["inert"]))
        modes[m] = {
            "n": len(recs), "n_success": n_succ,
            "n_detected": int(sum(1 for r in recs if r["detected"])),
            "n_feasible": int(np.count_nonzero(feas)),
            "n_inert": n_inert, "inert_fraction": n_inert / max(len(recs), 1),
            "reward_max": float(rew.max()), "reward_mean": float(rew.mean()),
            "proxy_tiebreak": used_proxy,
            "update": info,
            "next_round": int(nxt.round_index),
        }
        print(f"    {m:5s}: n={len(recs):4d}  success={n_succ:3d}  "
              f"inert={n_inert:3d}  r_max={rew.max():+.3f}  "
              f"r_mean={rew.mean():+.3f}  update={info['reason']}"
              + ("  [proxy tie-break]" if used_proxy else ""), flush=True)
        if n_inert > 0.25 * len(recs):
            print(f"           !! {100 * n_inert / len(recs):.0f}% of this mode's "
                  "proposals moved no particles: the window contained none of "
                  "the claimed set. Raise editor.bounds.source_radius_rvir's "
                  "lower bound -- those Rockstar runs bought nothing.",
                  flush=True)
    return modes


def stage_aggregate(cfg: Dict, args) -> int:
    boxes = ([b.strip() for b in args.boxes.split(",") if b.strip()]
             or list(cfg.get("search_boxes", [])))
    rows: List[Dict] = []
    for box in boxes:
        rows.extend(read_jsonl(rows_path(args.run_name, box)))
    scored = [r for r in rows
              if r.get("scored") and int(r.get("round", -1)) == int(args.round)]
    if not scored:
        print(f">>> GATE: no scored rows at all for round {args.round} in {boxes}.")
        print(">>> produced by: local_editor_candidates_cpu.sbatch")
        print(">>> exiting 0 so dependents report the same rather than stranding.")
        return 0

    cem_rows = [r for r in scored if r.get("arm") == "cem"
                and r.get("control", "none") == "none"]
    summary: Dict = {"pipeline": PIPELINE, "run_name": args.run_name,
                     "round": int(args.round), "boxes": boxes, "modes": {}}
    if cem_rows:
        summary["modes"] = _refit_modes(cfg, args, cem_rows)
    else:
        # The `gate1` stage scores the random arm and its three controls and
        # never the `cem` arm, so there is no population to refit -- but the
        # verdict that stage exists to produce is computed below from every
        # scored row. Returning here (as this script used to) made the Gate 1
        # job exit 0 having written no gate1.json, which reads like a clean run.
        summary["modes"] = {}
        summary["note"] = "no cem-arm rows this round; CEM refit skipped"
        print(f">>> no scored CEM rows for round {args.round}: reporting Gate 1 "
              f"from the {len(scored)} non-CEM rows and skipping the refit.",
              flush=True)

    gate = gate1_verdict(
        scored,
        forbidden_boxes=cfg.get("split", {}).get("final_eval_boxes", []),
    )
    summary["gate1"] = gate
    out = write_json(run_dir(args.run_name) / f"round_{int(args.round):03d}_summary.json",
                     summary)
    write_json(run_dir(args.run_name) / "gate1.json", gate)
    banner(f"round {args.round}: Gate 1 = {'PASS' if gate['pass'] else 'FAIL'} "
           f"({gate['n_successes']} successes, {len(gate['hosts'])} hosts, "
           f"{len(gate['boxes'])} boxes) -> {out}")
    for k, v in gate["checks"].items():
        print(f"    {'ok ' if v else 'NO '} {k}", flush=True)
    if not gate["pass"]:
        print("    Gate 1 failed. Per the plan: expand the editor representation "
              "(compensating shell, small learned local residual basis) before "
              "implementing the flow. Do NOT proceed to stage 5.", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_local_args(ap)
    ap.add_argument("--run-name", default="le_a")
    ap.add_argument("--stage", default="aggregate", choices=("propose", "aggregate"))
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=0)
    ap.add_argument("--n-hosts", type=int, default=0)
    ap.add_argument("--boxes", default="")
    args = ap.parse_args(argv)

    cfg = load_local_config(args)
    return (stage_propose(cfg, args) if args.stage == "propose"
            else stage_aggregate(cfg, args))


if __name__ == "__main__":
    raise SystemExit(main())
