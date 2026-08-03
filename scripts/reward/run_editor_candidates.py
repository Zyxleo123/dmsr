#!/usr/bin/env python
"""Stage 3/4: compose one candidate box with the local editor, halo-find, score.

One array task = one **candidate box** = one Rockstar run. Every selected host in
the box receives one proposal, and because stage 1 forced the hosts >= 6 Mpc/h
apart, that single halo run returns one independent proposal-level reward per
host. This is the entire economics of the search: 8 rewards per ~15-minute run
rather than 1.

Arms
----
``frozen``
    The no-op editor. The anchor every other arm is read against, and a hard
    check: it must reproduce the frozen SR2 catalog exactly.
``random``
    Actions drawn from the codec's own prior. The "did search do anything"
    control for CEM.
``cem`` / ``flow`` / ``gmm``
    Actions read from a manifest written by ``aggregate_cem_round.py`` or
    ``train_action_flow.py``.

Controls (``--control``), all of which keep the action and change one thing:

``random_particles``
    Equal count of particles drawn uniformly from the same host's smooth pool,
    contracted toward *their own* centroid at the same strength and radius. If
    this creates subhalos as often as the targeted edit, the targeted result was
    "compressing any few hundred particles works" and means nothing.
``shuffled_host``
    Proposal ``i``'s action applied to host ``i+1``. Separates "this action is
    right for this host" from "this action is right".
``near_subhalo``
    Centred on an existing SR2 subhalo instead of on empty smooth material. The
    reward **must** score zero here -- an object that was already in the frozen
    catalog can never be new -- so this is a test of the reward, not of the
    editor, and a nonzero score is a bug report.

Nothing here loads the HR field, HR residuals, HR subhalo positions or HR member
ids. Feasibility is therefore measured only against quantities available at
deployment (the frozen base field and the LR input), which is why the
HR-referenced constraints are ``null`` in the editor's YAML.

    python scripts/reward/run_editor_candidates.py --run-name le_a --box set8 \\
        --arm cem --actions-json .../round_000_actions.json --shard 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from _local_common import (  # noqa: E402
    PIPELINE, add_local_args, assert_no_final_boxes, append_jsonl, banner,
    base_catalog, codec_for, hosts_path, load_base_field, load_local_config,
    local_reward_config, mode_plan, pool_path, require_calibrated_constraints,
    reward_bins, reward_model, rows_path, run_dir, write_json,
)

from cosmo_sr.eval.particles import particle_mass_msun_h  # noqa: E402
from cosmo_sr.eval.rockstar import run_rockstar_on_field  # noqa: E402
from cosmo_sr.reward import paths  # noqa: E402
from cosmo_sr.reward.constraints import (  # noqa: E402
    check_feasible, constraint_values, load_constraints,
)
from cosmo_sr.reward.local_editor import (  # noqa: E402
    HostPool, action_from_values, apply_edits, build_host_pool,
    min_image, n_particles_for_token, plan_edits, proposal_center_mpc,
    token_from_values,
)
from cosmo_sr.reward.local_reward import (  # noqa: E402
    compactness_proxy, evaluate_candidate, full_box_scores,
)

ARMS = ("frozen", "random", "cem", "flow", "gmm")
CONTROLS = ("none", "random_particles", "shuffled_host", "near_subhalo")


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------


def load_pool_cache(run_name: str, box: str) -> Dict[int, np.ndarray]:
    p = pool_path(run_name, box)
    if not p.is_file():
        raise SystemExit(f"no pool cache for {box}: run extract_editor_members.py "
                         f"(expected {p})")
    z = np.load(p)
    hid, off, pid = z["halo_id"], z["offset"], z["particle_id"]
    return {int(h): pid[off[k]:off[k + 1]] for k, h in enumerate(hid)}


def build_pools(cfg: Dict, base_field, hosts: Sequence[Dict],
                members: Dict[int, np.ndarray]) -> Dict[int, HostPool]:
    """One :class:`HostPool` per selected host, from the *frozen* field.

    Built once per job and reused by every proposal in it: the pool depends only
    on the frozen box, so rebuilding it per candidate would repeat a few seconds
    of gathering for no reason -- and, worse, would make it possible for two
    candidates to disagree about what the pool is.
    """
    d = cfg["data"]
    h = cfg.get("hosts", {})

    # A host with no member ids used to be skipped with a warning. That is the
    # wrong default here: the hosts are chosen to span mass bins, so dropping a
    # subset changes which scientific question the run answers, and it does it
    # without failing. Measured on run `le_b`: a stale extraction left all four
    # bin-2 hosts memberless, and the run would have proceeded on bin-3 hosts
    # alone -- exactly the imbalance that host set was created to remove.
    missing = [int(r["host_id"]) for r in hosts
               if members.get(int(r["host_id"]), np.zeros(0)).size == 0]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(hosts)} selected hosts have no member ids "
            f"in the pool cache: {missing}\n"
            "The cache was built for a different host selection. Re-run\n"
            "  scripts/slurm/local_editor_members_cpu.sbatch\n"
            "for this run name; extract_editor_members.py now detects the "
            "shortfall and forces a full re-extraction."
        )

    out: Dict[int, HostPool] = {}
    for rec in hosts:
        hid = int(rec["host_id"])
        host_ids = members.get(hid)
        if host_ids is None or host_ids.size == 0:
            continue
        sub_ids = [members.get(int(s), np.zeros(0, np.int64)) for s in rec["sub_ids"]]
        out[hid] = build_host_pool(
            base_field, host_id=hid, host_member_ids=host_ids,
            subhalo_member_ids=sub_ids,
            subhalo_centers_mpc=np.asarray(rec["sub_pos_mpc"], dtype=np.float64).reshape(-1, 3),
            subhalo_radii_mpc=np.asarray(rec["sub_rvir_mpc"], dtype=np.float64).reshape(-1),
            center_mpc=rec["center_mpc"], rvir_mpc=float(rec["rvir_mpc"]),
            mvir=float(rec["mvir"]), vmax=float(rec["vmax"]),
            boxsize_mpc_h=float(d["boxsize_mpc_h"]),
            redshift=float(d.get("redshift", 0.0)),
            exclusion_mult=float(h.get("subhalo_exclusion_mult", 1.0)),
        )
    return out


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def actions_for_candidate(
    cfg: Dict, arm: str, index: int, n_cand: int, seed: int, n_hosts: int,
    manifest: Optional[Dict],
) -> Tuple[str, np.ndarray]:
    """``(mode, z)`` for one candidate: ``z`` is ``(n_hosts, 12)``.

    ``random`` draws from the *initial* CEM distribution rather than uniformly in
    the boxed parameters. Those are different priors after the squashing map, and
    the first is the one CEM actually starts from -- so "random" is the honest
    control for "did the search move", not a differently-shaped baseline.

    ``frozen`` returns zeros, which the codec maps to the centre of every box --
    including ``contraction`` and ``velocity_cooling``. Those two are what make
    an action a no-op, so the frozen arm overrides them explicitly rather than
    relying on where the centre of the box happens to be.
    """
    if manifest is not None:
        cand = {int(c["index"]): c for c in manifest["candidates"]}
        if index not in cand:
            raise SystemExit(f"candidate {index} not in the actions manifest")
        c = cand[index]
        return str(c["mode"]), np.asarray(c["z"], dtype=np.float64).reshape(n_hosts, -1)
    if arm == "frozen":
        return "both", np.zeros((n_hosts, codec_for(cfg, "both").dim), dtype=np.float64)
    modes = mode_plan(cfg, max(int(n_cand), 1))
    mode = modes[int(index) % len(modes)] if modes else "both"
    rng = np.random.default_rng([int(seed), int(index)])
    sigma = float(cfg.get("cem", {}).get("init_sigma", 1.5))
    return mode, rng.standard_normal((n_hosts, codec_for(cfg, mode).dim)) * sigma


def apply_control(
    control: str, tokens, actions, pools, hosts: Sequence[Dict], rng, boxsize: float,
):
    """Rewrite the proposal list to realise one control. Returns (tokens, actions)."""
    if control in ("none", ""):
        return tokens, actions, {}
    info: Dict = {"control": control}
    if control == "shuffled_host":
        hids = [int(t.host_id) for t in tokens]
        rolled = hids[1:] + hids[:1]
        for t, h in zip(tokens, rolled):
            t.host_id = int(h)
        info["host_map"] = dict(zip(map(str, hids), map(int, rolled)))
        return tokens, actions, info
    if control == "near_subhalo":
        # Re-aim each proposal at an existing SR2 subhalo of its host, expressed
        # in the token's own (radius, direction) parameterisation so that
        # *nothing else about the edit changes*.
        by_id = {int(h["host_id"]): h for h in hosts}
        n_hit = 0
        for t in tokens:
            rec = by_id.get(int(t.host_id))
            if not rec or not rec["sub_pos_mpc"]:
                continue
            pool = pools[int(t.host_id)]
            k = int(rng.integers(len(rec["sub_pos_mpc"])))
            d = min_image(np.asarray(rec["sub_pos_mpc"][k], dtype=np.float64)
                          - pool.center_mpc, boxsize) / max(pool.rvir_mpc, 1e-9)
            r = float(np.linalg.norm(d))
            t.radius_rvir = r
            t.direction = tuple(d / r) if r > 0 else (0.0, 0.0, 1.0)
            n_hit += 1
        info["n_reaimed"] = n_hit
        return tokens, actions, info
    if control == "random_particles":
        info["note"] = ("claims resolved by uniform draw from the host pool; "
                        "contraction is toward the drawn set's own centroid")
        return tokens, actions, info
    raise SystemExit(f"unknown control {control!r}")


def random_particle_plans(pools, tokens, actions, cfg, rng):
    """Equal-count random claims, contracted toward their own centroid.

    Contracting a uniformly-drawn set toward the *token's* centre would drag
    particles across half a virial radius and test nothing anyone would deploy.
    Toward their own centroid it is the same operator at the same strength on
    the same number of particles, differing only in where in the host they came
    from -- which is exactly the comparison the plan asks for.
    """
    from cosmo_sr.reward.local_editor import EditPlan, edge_window

    ed = cfg.get("editor", {})
    plans: List[EditPlan] = []
    claimed: List[np.ndarray] = []
    for token, action in zip(tokens, actions):
        pool = pools[int(token.host_id)]
        ids, pos, vel = pool.ids, pool.pos_mpc, pool.vel_kms
        if claimed:
            free = ~np.isin(ids, np.concatenate(claimed))
            ids, pos, vel = ids[free], pos[free], vel[free]
        n = int(min(n_particles_for_token(token, pool,
                                          n_min=int(ed.get("n_particles_min", 40)),
                                          n_max=int(ed.get("n_particles_max", 400))),
                    ids.size))
        if n == 0:
            plans.append(EditPlan(pool.host_id, pool.center_mpc.copy(), 0.0,
                                  np.zeros(0, np.int64), np.zeros(0),
                                  np.zeros((0, 3)), np.zeros((0, 3))))
            continue
        pick = np.sort(rng.choice(ids.size, size=n, replace=False))
        p = pos[pick]
        centroid = (p[0] + min_image(p - p[0], pool.boxsize_mpc_h).mean(axis=0)
                    ) % pool.boxsize_mpc_h
        r = np.linalg.norm(min_image(p - centroid, pool.boxsize_mpc_h), axis=1)
        r_src = float(action.source_radius_rvir) * pool.rvir_mpc
        plans.append(EditPlan(
            host_id=pool.host_id, center_mpc=centroid, source_radius_mpc=r_src,
            ids=ids[pick], weights=edge_window(r / max(r_src, 1e-12),
                                               action.edge_softness),
            pos_mpc=p, vel_kms=vel[pick], n_requested=n, n_short=0))
        claimed.append(ids[pick])
    return plans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_local_args(ap)
    ap.add_argument("--run-name", default="le_a")
    ap.add_argument("--box", required=True)
    ap.add_argument("--arm", default="random", choices=ARMS)
    ap.add_argument("--control", default="none", choices=CONTROLS)
    ap.add_argument("--actions-json", default="",
                    help="manifest from aggregate_cem_round.py / train_action_flow.py")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--candidates", type=int, default=0,
                    help="0 = config cem.candidates_per_round (ignored with a manifest)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure-only", action="store_true",
                    help="constraints and field diagnostics only, no reward. The "
                         "audit stage uses this before the thresholds exist")
    ap.add_argument("--with-density", action="store_true",
                    help="also compute the full-box CIC density diagnostics "
                         "(~1-2 min/candidate); off during search")
    ap.add_argument("--no-rockstar", action="store_true",
                    help="compose and diagnose the field but skip halo finding "
                         "(smoke tests only; emits no reward)")
    ap.add_argument("--save-field", action="store_true",
                    help="debugging only: write the 3.2 GB candidate field")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_local_config(args)
    box = str(args.box)
    assert_no_final_boxes(cfg, [box], script="run_editor_candidates.py")
    emits_reward = not (args.measure_only or args.no_rockstar)
    if emits_reward:
        require_calibrated_constraints(cfg, script="run_editor_candidates.py")

    d = cfg["data"]
    box_l = float(d["boxsize_mpc_h"])
    z_red = float(d.get("redshift", 0.0))
    ed = cfg.get("editor", {})
    n_min = int(ed.get("n_particles_min", 40))
    n_max = int(ed.get("n_particles_max", 400))

    hp = hosts_path(args.run_name, box)
    if not hp.is_file():
        raise SystemExit(f"no host selection for {box}: run select_editor_hosts.py")
    hosts = json.loads(hp.read_text())["hosts"]
    if not hosts:
        print(f">>> {box}: no hosts selected; nothing to do.", flush=True)
        return 0

    manifest = (json.loads(Path(args.actions_json).read_text())
                if args.actions_json else None)
    n_cand = (len(manifest["candidates"]) if manifest is not None
              else int(args.candidates or cfg.get("cem", {}).get("candidates_per_round", 28)))
    if args.arm == "frozen":
        n_cand = 1

    banner(f"{box} / run {args.run_name} / arm {args.arm} / control {args.control}: "
           f"{n_cand} candidates, {len(hosts)} hosts, shard "
           f"{args.shard}/{args.num_shards}")

    base_field = load_base_field(box, args.base_seed, mmap=True)
    particle_mass = particle_mass_msun_h(
        float(d.get("omega_m", 0.2814)), box_l, int(base_field.shape[1]) ** 3)
    members = load_pool_cache(args.run_name, box)
    pools = build_pools(cfg, base_field, hosts, members)
    if not pools:
        print(">>> no usable host pool; exiting 0 so dependents report the same.",
              flush=True)
        return 0
    host_order = [h for h in hosts if int(h["host_id"]) in pools]
    write_json(run_dir(args.run_name, "pools") / f"pool_summary_{box}.json",
               {"box": box, "pools": [pools[int(h["host_id"])].summary()
                                      for h in host_order]})

    base_cat = base_catalog(box)
    bins = reward_bins()
    rmodel = reward_model()
    rcfg = local_reward_config(cfg)
    cons = load_constraints(cfg.get("constraints", {}))
    lr = np.load(Path(d["root"]) / "lr" / f"{box}.npy", mmap_mode="r")
    rows_out = rows_path(args.run_name, box)

    for index in range(n_cand):
        if index % max(int(args.num_shards), 1) != int(args.shard):
            continue
        cid = (f"{args.run_name}_{box}_{args.arm}_{args.control}"
               f"_r{int(args.round)}_c{index:03d}")
        work = paths.LOCAL_EDITOR("candidates", args.run_name, box, cid, create=True)
        if (work / "row.json").is_file() and not args.overwrite:
            print(f"    {cid}: already scored; skipping", flush=True)
            continue

        t0 = time.time()
        mode, z = actions_for_candidate(cfg, args.arm, index, n_cand,
                                        int(args.seed), len(host_order), manifest)
        codec = codec_for(cfg, mode)
        rng = np.random.default_rng([int(args.seed), int(args.round), int(index)])

        tokens, actions = [], []
        for k, rec in enumerate(host_order):
            vals = codec.decode(z[k])
            if args.arm == "frozen":
                vals["contraction"] = 0.0
                vals["velocity_cooling"] = 0.0
            tokens.append(token_from_values(int(rec["host_id"]), vals))
            actions.append(action_from_values(vals))
        tokens, actions, ctl_info = apply_control(
            args.control, tokens, actions, pools, host_order, rng, box_l)

        if args.control == "random_particles":
            plans = random_particle_plans(pools, tokens, actions, cfg, rng)
        else:
            plans = plan_edits(pools, list(zip(tokens, actions)),
                               n_min=n_min, n_max=n_max)

        field, plans, edit_stats = apply_edits(
            base_field, pools, list(zip(tokens, actions)),
            boxsize_mpc_h=box_l, redshift=z_red, n_min=n_min, n_max=n_max,
            plans=plans)

        # Disjointness is an invariant, not a hope: assert it on every candidate.
        claimed = np.concatenate([p.ids for p in plans]) if plans else np.zeros(0, np.int64)
        if claimed.size != np.unique(claimed).size:
            raise SystemExit(f"{cid}: proposals claimed overlapping particles")

        row: Dict = {
            "pipeline": PIPELINE, "run_name": args.run_name, "box": box,
            "candidate_id": cid, "index": int(index), "round": int(args.round),
            "arm": args.arm, "control": args.control, "mode": mode,
            "control_info": ctl_info,
            "n_claimed_total": int(claimed.size),
            "z": z.tolist(),
            "codec": codec.to_dict(),
            "tokens": [t.to_dict() for t in tokens],
            "actions": [a.to_dict() for a in actions],
            "plans": [p.summary() for p in plans],
            "edit_stats": edit_stats,
        }

        # --- field feasibility (frozen base + LR only; never HR) -------------
        vals = constraint_values(
            field, np.asarray(base_field), np.asarray(lr), hr=None,
            scale_factor=int(d.get("scale_factor", 8)), boxsize_mpc_h=box_l,
            dis_norm_kpc_h=float(d.get("dis_norm_kpc_h", 6000.0)), redshift=z_red,
            compute_density=bool(args.with_density),
        )
        ok, viol = check_feasible(vals, cons)
        row["constraints"] = vals
        row["feasible_field"] = bool(ok)
        row["violations"] = viol

        proxies = []
        for plan, action in zip(plans, actions):
            act = plan.active
            if not np.any(act):
                proxies.append(float("nan"))
                continue
            from cosmo_sr.reward.local_editor import (
                particle_positions_mpc, particle_velocities_kms)
            ids = plan.ids[act]
            proxies.append(compactness_proxy(
                plan.pos_mpc[act], plan.vel_kms[act],
                particle_positions_mpc(field, ids, boxsize_mpc_h=box_l, redshift=z_red),
                particle_velocities_kms(field, ids, redshift=z_red),
                boxsize_mpc_h=box_l))
        row["compactness_proxy"] = proxies

        if args.save_field:
            np.save(work / "field.npy", field)

        if args.no_rockstar:
            row["wall_min"] = (time.time() - t0) / 60.0
            row["scored"] = False
            write_json(work / "row.json", row)
            append_jsonl(rows_out, row)
            print(f"    {cid}: field composed, halo finding skipped", flush=True)
            continue

        cat = run_rockstar_on_field(
            field, work, tag=cid, boxsize_kpc_h=box_l * 1000.0, redshift=z_red,
            overwrite=bool(args.overwrite))
        del field
        for g in Path(work).glob("*.gadget2"):
            g.unlink(missing_ok=True)
        print(f"    {cid}: Rockstar {cat.n} objects in "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

        # The mass the reward compares a detected object against is the mass the
        # editor could actually build: the particles it moved, times the particle
        # mass. NOT the token's nominal ratio.
        #
        # Those two diverge whenever the count clamp binds, and they diverge in
        # the direction that silently destroys the result. On the selected
        # ~1.4e5-member hosts a token at log_mass_ratio = -1.3 asks for 4.5e12
        # Msun/h while the clamp allows 400 particles = 2.3e11 -- a 1.29 dex gap
        # against a 0.8 dex tolerance, so a *genuine* new subhalo would be
        # rejected on mass grounds and scored as a failure. Judging the edit by
        # what it moved makes the comparison one the editor can pass when it
        # works, and keeps failing it when the object is the wrong size.
        proposals = []
        for t, a, p in zip(tokens, actions, plans):
            pool = pools[int(t.host_id)]
            n_act = int(np.count_nonzero(p.active))
            proposals.append({
                "base_host_id": int(t.host_id),
                "center_mpc": (p.center_mpc.tolist() if p.ids.size
                               else proposal_center_mpc(pool, t, a).tolist()),
                "host_rvir_mpc": float(pool.rvir_mpc),
                "requested_mvir": float(n_act * particle_mass),
                # Provenance: what the token asked for before the clamp, so a
                # round where the clamp bound everywhere is visible in the rows.
                "nominal_mvir": float(10.0 ** t.log_mass_ratio * pool.mvir),
                "n_active_particles": n_act,
                "n_nominal_particles": int(round(10.0 ** t.log_mass_ratio
                                                 * max(pool.n_members, 1))),
            })
        row["proposals"] = proposals
        row["n_count_clamped"] = int(sum(
            1 for q in proposals
            if q["n_nominal_particles"] > n_max or q["n_nominal_particles"] < n_min))

        if args.measure_only:
            row["scored"] = False
            row["note"] = "measure-only: constraints and catalog reported, no reward"
            row.update(full_box_scores(cat, bins, boxsize_mpc_h=box_l, box=box, tag=cid))
        else:
            outcomes = evaluate_candidate(
                base_cat, cat, proposals, rcfg, boxsize_mpc_h=box_l,
                feasible_field=bool(ok), violations=viol)
            for o, pr, plan in zip(outcomes, proxies, plans):
                o.compactness_proxy = float(pr)
                o.n_active_particles = int(np.count_nonzero(plan.active))
            row["outcomes"] = [o.to_dict() for o in outcomes]
            row["reward_mean"] = float(np.mean([o.reward for o in outcomes]))
            row["n_detected"] = int(sum(1 for o in outcomes if o.detected))
            row["n_inert"] = int(sum(1 for o in outcomes if o.n_active_particles == 0))
            row["scored"] = True
            row.update(full_box_scores(
                cat, bins, boxsize_mpc_h=box_l, reward_model=rmodel,
                reliable_host_bins=cfg.get("reward", {}).get("reliable_host_bins"),
                box=box, tag=cid))
            print(f"    {cid}: {row['n_detected']}/{len(outcomes)} detected, "
                  f"{row['n_inert']} inert, mean r = {row['reward_mean']:+.3f}, "
                  f"feasible={ok}", flush=True)

        row["wall_min"] = (time.time() - t0) / 60.0
        write_json(work / "row.json", row)
        append_jsonl(rows_out, row)

    banner(f"{box}: rows -> {rows_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
