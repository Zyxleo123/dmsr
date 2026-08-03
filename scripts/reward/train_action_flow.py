#!/usr/bin/env python
"""Stage 5: distil the CEM successes into ``q_theta(a | h, c)``.

Only run after Gate 1 passes. The script checks ``gate1.json`` and refuses
otherwise, because a flow trained on an action space with no successes in it is
a flow trained on noise that will look like it converged.

What is learned
---------------
The **action** block only -- the eight ``EditorAction`` coordinates -- conditioned
on the frozen host's features ``h``, the token ``c`` the proposal asked for, and
a one-hot of the edit mode. The token block of the search vector is
*conditioning*, not a target: at deployment the token comes from the catalog
generator (stage 6), and the policy's job is to say how to realise it.

Coordinates pinned by the mode (``velocity_cooling`` under ``disp``, and so on)
are zeroed in the targets. They carry no information -- the codec ignores them --
so leaving the sampled noise in would train the network to reproduce it.

Two models, and the baseline decides
------------------------------------
A conditional flow and a Gaussian mixture are trained on identical data with
identical weights. ``comparison.json`` reports both on held-out reward-weighted
log-likelihood proxies and on action diversity. If the flow does not beat the
mixture, the honest conclusion is that an 8-dimensional action space did not
need a flow, and ``evaluate_local_editor.py`` will report it that way.

    python scripts/reward/train_action_flow.py --run-name le_a
"""
from __future__ import annotations

import argparse
import copy
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np

from _local_common import (  # noqa: E402
    PIPELINE, add_local_args, banner, codec_for, hosts_path, load_local_config,
    mode_plan, read_jsonl, rows_path, run_dir, write_json,
)

from cosmo_sr.reward.local_editor import ACTION_PARAMS, TOKEN_PARAMS  # noqa: E402
from cosmo_sr.reward.local_reward import (  # noqa: E402
    ProposalOutcome, is_scientific_success,
)

MODES = ("disp", "both", "vel")
# Kept in step with aggregate_cem_round.MODES; the per-mode seed offset there
# depends on the ordering, so a divergence would silently resume a different
# search.
N_TOKEN = len(TOKEN_PARAMS)
N_ACTION = len(ACTION_PARAMS)


def build_replay(cfg: Dict, run_name: str, boxes: Sequence[str]) -> Dict:
    """The successful-action set, with everything needed to condition on it."""
    from cosmo_sr.reward.action_flow import host_features, token_features

    host_meta: Dict[Tuple[str, int], Dict] = {}
    for box in boxes:
        hp = hosts_path(run_name, box)
        if not hp.is_file():
            continue
        summ = {}
        sp = run_dir(run_name, "pools") / f"pool_summary_{box}.json"
        if sp.is_file():
            summ = {int(p["host_id"]): p for p in json.loads(sp.read_text())["pools"]}
        for h in json.loads(hp.read_text())["hosts"]:
            hid = int(h["host_id"])
            s = summ.get(hid, {})
            host_meta[(box, hid)] = {
                "mvir": float(h["mvir"]), "rvir_mpc": float(h["rvir_mpc"]),
                "vmax": float(h["vmax"]), "n_sub_current": float(h["n_sub_current"]),
                "n_members": float(s.get("n_members", h["num_p"])),
                "smooth_fraction": float(s.get("smooth_fraction", 1.0)),
            }

    A, C, R, H, M, meta = [], [], [], [], [], []
    for box in boxes:
        for r in read_jsonl(rows_path(run_name, box)):
            if not r.get("scored") or r.get("control", "none") != "none":
                continue
            mode = str(r.get("mode", "both"))
            z = np.asarray(r["z"], dtype=np.float64)
            pinned = [k for k, p in enumerate(codec_for(cfg, mode).params) if p.pinned]
            for k, (o_raw, tok) in enumerate(zip(r.get("outcomes", []), r["tokens"])):
                o = ProposalOutcome.from_dict(o_raw)
                if not is_scientific_success(o):
                    continue
                hm = host_meta.get((box, int(o.base_host_id)))
                if hm is None:
                    continue
                a = z[k, N_TOKEN:].copy()
                for j in pinned:
                    if j >= N_TOKEN:
                        a[j - N_TOKEN] = 0.0
                onehot = np.zeros(len(MODES))
                onehot[MODES.index(mode)] = 1.0
                A.append(a)
                C.append(np.concatenate([host_features(hm), token_features(tok), onehot]))
                R.append(float(o.reward))
                H.append(int(o.base_host_id))
                M.append(mode)
                meta.append({"box": box, "candidate_id": r["candidate_id"],
                             "host_id": int(o.base_host_id), "mode": mode})
    return {"a": np.asarray(A, dtype=np.float64).reshape(-1, N_ACTION),
            "cond": np.asarray(C, dtype=np.float64).reshape(len(A), -1) if A else np.zeros((0, 0)),
            "reward": np.asarray(R, dtype=np.float64),
            "host_id": np.asarray(H, dtype=np.int64),
            "mode": M, "meta": meta}


def _fit_flow(net, cond, a, w, *, steps, batch, lr, device, gen,
              ref=None, ref_lambda=0.0, warmup=0, log_every=500):
    import torch
    from cosmo_sr.reward.action_flow import (
        flow_matching_loss, reference_velocity_penalty)

    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    n = a.shape[0]
    hist = []
    for step in range(int(steps)):
        idx = torch.randint(0, n, (min(int(batch), n),), device=device, generator=gen)
        loss, info = flow_matching_loss(net, a[idx], cond[idx], w[idx], generator=gen)
        if ref is not None and float(ref_lambda) > 0 and step >= int(warmup):
            loss = loss + float(ref_lambda) * reference_velocity_penalty(
                net, ref, a[idx], cond[idx], generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % int(log_every) == 0 or step == int(steps) - 1:
            hist.append({"step": step, "loss": float(loss.detach().cpu()), **info})
            print(f"      step {step:6d}  loss {float(loss.detach().cpu()):.4f}",
                  flush=True)
    return hist


def token_library(cfg: Dict, run_name: str):
    """Stage 6's empirical token library, built (once) from TRAINING boxes."""
    from _local_common import assert_training_boxes, base_catalog
    from cosmo_sr.reward.token_bootstrap import HostTokenLibrary

    p = run_dir(run_name, "tokens", create=True) / "token_library.npz"
    if p.is_file():
        return HostTokenLibrary.from_npz(p)
    tk = cfg.get("tokens", {})
    boxes = list(tk.get("library_boxes", []))
    assert_training_boxes(cfg, boxes, script="train_action_flow.py --stage sample")
    lib = HostTokenLibrary.from_catalogs(
        {b: base_catalog(b) for b in boxes},
        boxsize_mpc_h=float(cfg["data"]["boxsize_mpc_h"]),
        min_host_particles=int(tk.get("min_host_particles", 2000)))
    lib.to_npz(p)
    print(f"    token library: {lib.n_hosts} donor hosts, "
          f"{lib.log_mass_ratio.size} satellites -> {p}", flush=True)
    return lib


def _token_values(token) -> Dict[str, float]:
    d = np.asarray(token.direction, dtype=np.float64)
    return {
        "log_mass_ratio": float(token.log_mass_ratio),
        "radius_rvir": float(token.radius_rvir),
        "dir_cos_theta": float(d[2]),
        "dir_phi": float(np.arctan2(d[1], d[0]) % (2.0 * np.pi)),
    }


def stage_sample(cfg: Dict, args) -> int:
    """Write an actions manifest from a trained policy (or the final CEM state).

    Tokens come from the stage-6 bootstrap so that the flow and GMM arms are
    asked to realise a *plausible desired population*, not to reproduce the
    search's own token distribution -- which is what the final comparison is
    supposed to test.
    """
    import torch
    from cosmo_sr.reward.action_flow import ActionFlow, GaussianMixturePolicy, \
        host_features, token_features
    from cosmo_sr.reward.cem import CEMRun
    # The CEM arm has to resume the *same* search aggregate_cem_round.py wrote,
    # so it reuses that module's state constructor rather than rebuilding one.
    from aggregate_cem_round import initial_state

    boxes = ([b.strip() for b in args.boxes.split(",") if b.strip()]
             or list(cfg.get("search_boxes", [])))
    n_cand = int(args.n_candidates or cfg.get("cem", {}).get("candidates_per_round", 28))
    modes = mode_plan(cfg, n_cand)
    rng = np.random.default_rng([int(cfg.get("flow", {}).get("seed", 0)),
                                 int(args.sample_seed)])

    hp = hosts_path(args.run_name, boxes[0])
    hosts = json.loads(hp.read_text())["hosts"]
    sp = run_dir(args.run_name, "pools") / f"pool_summary_{boxes[0]}.json"
    summ = ({int(p["host_id"]): p for p in json.loads(sp.read_text())["pools"]}
            if sp.is_file() else {})
    lib = token_library(cfg, args.run_name)

    policy = str(args.policy)
    net = gmm = None
    if policy in ("flow", "gmm"):
        ck = run_dir(args.run_name, "flow") / "action_policy.pt"
        if not ck.is_file():
            print(f">>> GATE: no policy checkpoint at {ck}; exiting 0.")
            return 0
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        if policy == "flow":
            net = ActionFlow(blob["action_dim"], blob["cond_dim"],
                             width=int(blob["config"].get("width", 128)),
                             depth=int(blob["config"].get("depth", 3)))
            net.load_state_dict(blob["flow"])
            net.eval()
        else:
            gmm = GaussianMixturePolicy(
                blob["action_dim"], blob["cond_dim"],
                n_components=int(blob["config"].get("baseline_components", 4)),
                width=int(blob["config"].get("width", 128)))
            gmm.load_state_dict(blob["gmm"])
            gmm.eval()
        cmu = np.asarray(blob["cond_mu"]); csd = np.asarray(blob["cond_sd"])
    elif policy == "cem":
        runs = {m: CEMRun(root=run_dir(args.run_name, "cem"), name=m) for m in MODES}
    else:
        raise SystemExit(f"unknown --policy {policy!r}")

    candidates: List[Dict] = []
    for i in range(n_cand):
        mode = modes[i % len(modes)]
        codec = codec_for(cfg, mode)
        onehot = np.zeros(len(MODES)); onehot[MODES.index(mode)] = 1.0
        rows = []
        for h in hosts:
            hid = int(h["host_id"])
            s = summ.get(hid, {})
            hm = {"mvir": float(h["mvir"]), "rvir_mpc": float(h["rvir_mpc"]),
                  "vmax": float(h["vmax"]),
                  "n_sub_current": float(h["n_sub_current"]),
                  "n_members": float(s.get("n_members", h["num_p"])),
                  "smooth_fraction": float(s.get("smooth_fraction", 1.0))}
            toks = lib.sample_tokens(
                host_id=hid, host_mvir=float(h["mvir"]), rng=rng,
                existing_log_mass_ratio=h.get("existing_log_mass_ratio", []),
                donor_dex=float(cfg.get("tokens", {}).get("donor_dex", 0.15)),
                max_tokens=1)
            if not toks:
                rows.append(np.zeros(codec.dim))
                continue
            tv = codec.clip_to_bounds(_token_values(toks[0]))
            z = codec.encode({**tv, **{p.name: 0.0 for p in codec.params
                                       if p.name not in tv}})
            if policy == "cem":
                # The state that *would have produced* the next round: the last
                # scored manifest's stored state refit on its own rewards. Using
                # the mean alone would compare a point estimate against two
                # stochastic policies.
                st = initial_state(cfg, mode, 1)
                st, _ = runs[mode].resume(st)
                z[N_TOKEN:] = (st.mean + st.std
                               * rng.standard_normal(codec.dim))[N_TOKEN:]
            else:
                cond = np.concatenate([host_features(hm),
                                       token_features(toks[0].to_dict()), onehot])
                ct = torch.as_tensor(((cond - cmu) / csd)[None, :], dtype=torch.float32)
                with torch.no_grad():
                    a = (net.sample(ct) if net is not None
                         else gmm.sample(ct)).numpy().reshape(-1)
                z[N_TOKEN:] = a
            rows.append(z)
        candidates.append({"index": int(i), "mode": mode,
                           "z": np.asarray(rows).tolist()})

    out = run_dir(args.run_name, "actions", create=True) / f"{policy}_actions.json"
    write_json(out, {"pipeline": PIPELINE, "run_name": args.run_name,
                     "policy": policy, "boxes": boxes, "n_hosts": len(hosts),
                     "token_source": "bootstrap_training_catalogs",
                     "candidates": candidates})
    banner(f"{policy}: {len(candidates)} candidate manifests -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_local_args(ap)
    ap.add_argument("--run-name", default="le_a")
    ap.add_argument("--stage", default="train", choices=("train", "sample"))
    ap.add_argument("--policy", default="flow", choices=("flow", "gmm", "cem"))
    ap.add_argument("--n-candidates", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--boxes", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--min-samples", type=int, default=24,
                    help="below this the replay set cannot support a conditional "
                         "model and the script exits 0 with an explanation")
    ap.add_argument("--ignore-gate", action="store_true",
                    help="train anyway (diagnostics only; never for a result)")
    args = ap.parse_args(argv)

    if args.stage == "sample":
        return stage_sample(load_local_config(args), args)

    import torch
    from cosmo_sr.reward.action_flow import (
        ActionFlow, GaussianMixturePolicy, action_diversity, reward_weights)

    cfg = load_local_config(args)
    fl = cfg.get("flow", {})
    boxes = ([b.strip() for b in args.boxes.split(",") if b.strip()]
             or list(cfg.get("search_boxes", [])))

    gate_path = run_dir(args.run_name) / "gate1.json"
    if not args.ignore_gate:
        if not gate_path.is_file():
            print(">>> GATE: no gate1.json; run aggregate_cem_round.py --stage aggregate")
            print(">>> exiting 0 so dependents report the same rather than stranding.")
            return 0
        gate = json.loads(gate_path.read_text())
        if not gate.get("pass"):
            print(">>> GATE 1 FAILED -- not training the flow. Failed checks: "
                  f"{[k for k, v in gate['checks'].items() if not v]}")
            print(">>> Per the plan, expand the editor representation first.")
            print(">>> exiting 0 so dependents report the same.")
            return 0

    banner(f"building the successful-action replay set from {boxes}")
    data = build_replay(cfg, args.run_name, boxes)
    n = data["a"].shape[0]
    print(f"    {n} successful actions over "
          f"{len(set(data['host_id'].tolist()))} hosts, modes "
          f"{ {m: data['mode'].count(m) for m in MODES} }", flush=True)
    if n < int(args.min_samples):
        print(f">>> only {n} successes (< {args.min_samples}); the replay set "
              "cannot support a conditional model. exiting 0.")
        write_json(run_dir(args.run_name) / "flow_report.json",
                   {"pipeline": PIPELINE, "trained": False, "n_successes": n})
        return 0

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gen = torch.Generator(device=dev).manual_seed(int(fl.get("seed", 0)))
    torch.manual_seed(int(fl.get("seed", 0)))

    # Standardise the conditioning: host mass and Vmax are decades apart in raw
    # units and an unnormalised MLP input makes the first layer's job the
    # scaling rather than the physics.
    cond_np = data["cond"]
    mu, sd = cond_np.mean(0), cond_np.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    cond = torch.as_tensor((cond_np - mu) / sd, dtype=torch.float32, device=dev)
    a = torch.as_tensor(data["a"], dtype=torch.float32, device=dev)
    w_np = reward_weights(data["reward"], data["host_id"],
                          tau=float(fl.get("weight_tau", 0.5)),
                          w_max=float(fl.get("weight_max", 10.0)))
    w = torch.as_tensor(w_np, dtype=torch.float32, device=dev)
    print(f"    weights: mean {w_np.mean():.3f} max {w_np.max():.3f} "
          f"min {w_np.min():.3f}", flush=True)

    kw = dict(steps=int(fl.get("steps", 4000)), batch=int(fl.get("batch_size", 128)),
              lr=float(fl.get("lr", 1e-3)), device=dev, gen=gen)

    # --- reference: the same architecture, unweighted --------------------------
    banner("fitting the unweighted reference flow")
    ref = ActionFlow(N_ACTION, cond.shape[1], width=int(fl.get("width", 128)),
                     depth=int(fl.get("depth", 3))).to(dev)
    ref_hist = _fit_flow(ref, cond, a, torch.ones_like(w), **kw)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    banner("fitting the reward-weighted flow with the reference penalty")
    net = ActionFlow(N_ACTION, cond.shape[1], width=int(fl.get("width", 128)),
                     depth=int(fl.get("depth", 3))).to(dev)
    net.load_state_dict(copy.deepcopy(ref.state_dict()))
    for p in net.parameters():
        p.requires_grad_(True)
    hist = _fit_flow(net, cond, a, w, ref=ref,
                     ref_lambda=float(fl.get("reference_penalty", 0.1)),
                     warmup=int(fl.get("reference_warmup_steps", 800)), **kw)

    # --- mandatory baseline ---------------------------------------------------
    banner("fitting the Gaussian-mixture baseline")
    gmm = GaussianMixturePolicy(N_ACTION, cond.shape[1],
                                n_components=int(fl.get("baseline_components", 4)),
                                width=int(fl.get("width", 128))).to(dev)
    opt = torch.optim.Adam(gmm.parameters(), lr=float(fl.get("lr", 1e-3)))
    gmm_hist = []
    for step in range(int(fl.get("steps", 4000))):
        idx = torch.randint(0, n, (min(int(fl.get("batch_size", 128)), n),),
                            device=dev, generator=gen)
        loss = gmm.loss(a[idx], cond[idx], w[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == int(fl.get("steps", 4000)) - 1:
            gmm_hist.append({"step": step, "nll": float(loss.detach().cpu())})
            print(f"      step {step:6d}  nll {float(loss.detach().cpu()):.4f}",
                  flush=True)

    # --- comparison -----------------------------------------------------------
    n_eval = int(fl.get("eval_samples", 256))
    pick = torch.randint(0, n, (min(n_eval, n),), device=dev, generator=gen)
    with torch.no_grad():
        s_flow = net.sample(cond[pick], generator=gen).cpu().numpy()
        s_gmm = gmm.sample(cond[pick], generator=gen).cpu().numpy()
        ll_gmm = float(gmm.log_prob(a, cond).mean().cpu())
    comparison = {
        "n_successes": n,
        "flow_diversity": action_diversity(s_flow),
        "gmm_diversity": action_diversity(s_gmm),
        "data_diversity": action_diversity(data["a"]),
        "gmm_mean_log_prob": ll_gmm,
        "flow_final_loss": hist[-1] if hist else None,
        "reference_final_loss": ref_hist[-1] if ref_hist else None,
        "note": ("The flow is justified only if evaluate_local_editor.py shows it "
                 "ahead of the GMM on realised reward or on covering distinct "
                 "action modes at equal Rockstar budget. Diversity here is a "
                 "necessary condition, not that comparison."),
    }

    ck = run_dir(args.run_name, "flow", create=True)
    torch.save({"flow": net.state_dict(), "reference": ref.state_dict(),
                "gmm": gmm.state_dict(),
                "cond_mu": mu, "cond_sd": sd,
                "cond_dim": int(cond.shape[1]), "action_dim": N_ACTION,
                "config": dict(fl), "modes": list(MODES)},
               ck / "action_policy.pt")
    write_json(ck / "comparison.json", comparison)
    write_json(run_dir(args.run_name) / "flow_report.json",
               {"pipeline": PIPELINE, "trained": True, "n_successes": n,
                "checkpoint": str(ck / "action_policy.pt"),
                "comparison": comparison,
                "flow_history": hist, "gmm_history": gmm_hist})
    banner(f"policies -> {ck / 'action_policy.pt'}")
    print(f"    flow diversity  {comparison['flow_diversity']}", flush=True)
    print(f"    gmm  diversity  {comparison['gmm_diversity']}", flush=True)
    print(f"    data diversity  {comparison['data_diversity']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
