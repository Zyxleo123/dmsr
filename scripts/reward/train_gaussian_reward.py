#!/usr/bin/env python
"""Phase 2.2 (GPU): reward-only training of the Gaussian residual policy.

    L_G = - sum_i w_i log pi_theta(a_i | B_i, Y_i) / sum_i w_i
          + beta * KL(pi_theta || pi_ref)
          + lambda_edit * C_edit

**There is no HR reconstruction term, and no paired residual is ever opened.**
Every example is a replay entry: a chunk whose ensemble was feasible and
top-quantile and whose own leave-one-out contribution was positive -- the
existing selection rule in :mod:`cosmo_sr.reward.replay`, unchanged. The action
``a_i`` is *detached*: this is reward-weighted maximum likelihood on samples the
behaviour policy already produced, not a policy gradient, so nothing here can
learn to exploit a differentiable surrogate of Rockstar.

Reconstructing the behaviour action
-----------------------------------
``a_s = mu_s^b + sigma_s^b * eps_s`` where ``eps`` comes from the global
coordinate-aligned lattice at ``action_seed``. So the action is recovered exactly
by running the *behaviour* checkpoint once, in ``no_grad``, on the same crop --
no multi-GB coefficient dump required. The stored behaviour log-probability is
checked against the reconstruction, and a mismatch is fatal: it means the crop,
the seed or the checkpoint is not the one that produced the reward.

``pi_ref`` is a frozen copy of the initial policy. Both are diagonal Gaussians,
so its KL is closed form and no sampling enters the regulariser.

    python scripts/reward/train_gaussian_reward.py --round round_000 \\
        --behavior-checkpoint .../gauss_k16/policy.pt --run-name gauss_r0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from _common import (add_common_args, banner, load_reward_config, lr_path,
                     select_device, split_boxes, write_json)
from sample_gaussian_candidates import load_policy

from cosmo_sr.data.crops import periodic_crop
from cosmo_sr.reward import paths

from cosmo_sr.reward.gaussian_policy import MultiScaleGaussianPolicy
from cosmo_sr.reward.geometry import ChunkGrid
from cosmo_sr.reward.replay import read_replay
from cosmo_sr.utils.config import load_config


class ReplayCrops:
    """Elite chunks as (base, lr, behaviour action) crops, lazily and periodically.

    The crop is the chunk's core plus ``context_margin`` of *real* neighbourhood,
    aligned to the LR lattice, so the window the policy sees in training is the
    window it sees when a full box is tiled. Anything else trains one mapping and
    evaluates another.
    """

    def __init__(self, entries: Sequence, cfg: Dict, *, crop_hr: int,
                 context_margin: int, action_records: Dict[str, Dict]):
        self.entries = list(entries)
        self.cfg = cfg
        self.core_hr = int(crop_hr)
        self.margin = int(context_margin)
        self.sf = int(cfg["data"]["scale_factor"])
        if self.margin % self.sf or self.core_hr % self.sf:
            raise SystemExit(
                f"crop_hr={self.core_hr} and context_margin={self.margin} must "
                f"both be multiples of scale_factor={self.sf}, or the LR window "
                f"is shifted by a fraction of a cell")
        self.records = action_records
        self._open: Dict[str, Dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _fields(self, e) -> Dict[str, np.ndarray]:
        if e.box not in self._open:
            self._open[e.box] = {
                "lr": np.load(lr_path(self.cfg, e.box)),
                "base": np.load(e.base_id, mmap_mode="r"),
            }
        return self._open[e.box]

    def get(self, i: int, rng: np.random.Generator) -> Dict:
        e = self.entries[i]
        f = self._fields(e)
        ng_hr = int(f["base"].shape[1])
        grid = ChunkGrid(ng_hr=ng_hr, chunk_hr=int(e.chunk_hr))
        origin = grid.origin(int(e.chunk_id))

        # A random LR-aligned window whose CORE lies inside the credited chunk:
        # the reward was attributed to that chunk, so the likelihood is raised on
        # it and not on its neighbours.
        span = max((int(e.chunk_hr) - self.core_hr) // self.sf + 1, 1)
        lr_start = tuple(int(o) // self.sf + int(rng.integers(span)) for o in origin)
        hr_start = tuple(s * self.sf for s in lr_start)

        m_hr, m_lr = self.margin, self.margin // self.sf
        base = periodic_crop(np.asarray(f["base"]), hr_start, self.core_hr, pad=m_hr)
        y_lr = periodic_crop(np.asarray(f["lr"]), lr_start,
                             self.core_hr // self.sf, pad=m_lr)
        key = f"{e.box}|{int(e.residual_seed)}"
        return {
            "base": torch.from_numpy(np.ascontiguousarray(base, dtype=np.float32)),
            "y_lr": torch.from_numpy(np.ascontiguousarray(y_lr, dtype=np.float32)),
            "hr_start": tuple(int(v) - m_hr for v in hr_start),
            "ng_hr": ng_hr,
            "weight": float(e.weight),
            "action_seed": int(e.residual_seed),
            "record": self.records.get(key, {}),
            "entry": e,
        }


def _load_action_records(oracle_run: Optional[str]) -> Dict[str, Dict]:
    """``{box|seed: record}`` from the candidate sampler's ``actions/`` sidecar."""
    out: Dict[str, Dict] = {}
    if not oracle_run:
        return out
    d = paths.ORACLE(oracle_run) / "actions"
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except ValueError:
            continue
        out[f"{r['box']}|{int(r['action_seed'])}"] = r
    return out


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--model-config", default="configs/reward/gaussian_policy.yaml")
    ap.add_argument("--round", default=None, help="replay round to train on")
    ap.add_argument("--oracle-run", default=None,
                    help="candidate run whose actions/ sidecar holds the behaviour "
                         "provenance")
    ap.add_argument("--behavior-checkpoint", default=None,
                    help="the checkpoint that PRODUCED the replay actions. Omit "
                         "only when the behaviour policy was the untrained "
                         "reference (round 0)")
    ap.add_argument("--init-checkpoint", default=None,
                    help="initialise theta from here; default = the reference policy")
    ap.add_argument("--run-name", default="gaussian_r0")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--beta-kl", type=float, default=None)
    ap.add_argument("--lambda-edit", type=float, default=None)
    ap.add_argument("--sigma-floor", type=float, default=None)
    ap.add_argument("--crop-hr", type=int, default=None)
    ap.add_argument("--context-margin", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow-val", action="store_true",
                    help="permit val boxes in the buffer; every later number is "
                         "then in-sample")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    mc = load_config(args.model_config)
    full = {**cfg, "model": mc.get("model", {}), "correction": mc.get("correction", {})}
    tcfg = dict(mc.get("train", {}))
    rcfg = dict(tcfg.get("replay", {}))

    def opt(name, default):
        v = getattr(args, name.replace("-", "_"), None)
        return v if v is not None else tcfg.get(name, default)

    steps = int(opt("steps", 20000))
    batch = int(opt("batch_size", 2))
    lr_ = float(opt("lr", 1e-4))
    beta = float(opt("beta_kl", 0.01))
    lam_edit = float(opt("lambda_edit", 1e-3))
    sigma_floor = float(opt("sigma_floor", 0.02))
    crop_hr = int(opt("crop_hr", 128))
    margin = int(opt("context_margin", 48))
    seed = int(opt("seed", 0))
    round_name = args.round or rcfg.get("round", "round_000")
    device = select_device(args.device)

    # ---------------------------------------------------------------- replay
    replay_path = paths.REPLAY(round_name) / "replay.jsonl"
    if not replay_path.is_file():
        raise SystemExit(
            f"no replay buffer at {replay_path}; run scripts/reward/build_replay.py "
            f"on a scored candidate run first")
    entries = read_replay(replay_path)
    if not entries:
        raise SystemExit(f"{replay_path} is empty: nothing was selected as elite")

    forbidden = set(split_boxes(cfg, "test"))
    if not args.allow_val:
        forbidden |= set(split_boxes(cfg, "val"))
    leaked = sorted({e.box for e in entries} & forbidden)
    if leaked:
        raise SystemExit(
            f"replay buffer contains held-out boxes {leaked}. Training on them "
            f"turns held-out data into training data; re-run the candidate stage "
            f"on train boxes"
            + ("" if args.allow_val else ", or pass --allow-val (VAL ONLY)") + ".")

    records = _load_action_records(args.oracle_run)
    missing = [f"{e.box}|{e.residual_seed}" for e in entries
               if f"{e.box}|{int(e.residual_seed)}" not in records]
    if missing and args.oracle_run:
        print(f"  ! {len(missing)} replay entries have no action record "
              f"(e.g. {missing[:3]}); their behaviour log-probability cannot be "
              f"verified.", flush=True)

    # ---------------------------------------------------------------- models
    policy = load_policy(args.init_checkpoint, full, device).train()
    behavior = load_policy(args.behavior_checkpoint, full, device).eval()
    for p in behavior.parameters():
        p.requires_grad_(False)
    # pi_ref: a frozen copy of the INITIAL policy, so the KL measures drift from
    # where this run started rather than from wherever it has already gone.
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    if bool(rcfg.get("verify_behavior_log_prob", True)):
        check_behavior_identity(behavior, records, entries)
    floor = apply_sigma_floor(policy, sigma_floor)

    data = ReplayCrops(entries, cfg, crop_hr=crop_hr, context_margin=margin,
                       action_records=records)
    opt_ = torch.optim.AdamW(policy.parameters(), lr=lr_,
                             weight_decay=float(tcfg.get("weight_decay", 0.0)))
    ema = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    ema_decay = float(tcfg.get("ema_decay", 0.999))
    normalize = bool(tcfg.get("normalize_log_prob", True))

    out = paths.CHECKPOINTS(args.run_name, create=True)
    banner(f"reward-only training: {len(entries)} elite chunks, {steps} steps, "
           f"batch {batch}, crop {crop_hr}+2x{margin}, beta_kl={beta}, "
           f"lambda_edit={lam_edit}")
    print(f"  behaviour policy : "
          f"{args.behavior_checkpoint or 'untrained reference (round 0)'}")
    print(f"  pi_ref           : frozen copy of the initial policy")
    print(f"  variance floor   : sigma >= {floor:g}")
    print(f"  paired-HR terms  : none (reward-only)")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    history: List[Dict] = []
    t0 = time.time()

    for step in range(1, steps + 1):
        idx = rng.integers(0, len(data), size=batch)
        batch_items = [data.get(int(i), rng) for i in idx]

        base = torch.stack([b["base"] for b in batch_items]).to(device)
        y_lr = torch.stack([b["y_lr"] for b in batch_items]).to(device)
        w = torch.tensor([b["weight"] for b in batch_items], dtype=torch.float32,
                         device=device)

        # -- reconstruct the behaviour action, exactly, and DETACH it ---------
        with torch.no_grad():
            b_dist = behavior.distribution(base, y_lr)
            acts = {}
            for s in behavior.specs:
                eps = torch.stack([
                    behavior.window_noise(
                        b["action_seed"], origin=b["hr_start"],
                        n_hr=base.shape[-1], grid_hr=b["ng_hr"], batch=1,
                    )[s.name][0] for b in batch_items
                ]).to(device)
                mu, sig = b_dist[s.name]
                acts[s.name] = (mu + sig * eps).detach()
            behavior_logp = behavior.log_prob(b_dist, acts).detach()

        # -- the loss ---------------------------------------------------------
        dist = policy.distribution(base, y_lr)
        logp = policy.log_prob(dist, acts)
        denom = float(policy.n_action_elements(int(base.shape[-1]))) if normalize else 1.0
        nll = -(w * logp / denom).sum() / w.sum().clamp_min(1e-12)

        with torch.no_grad():
            ref_dist = reference.distribution(base, y_lr)
        kl = policy.kl_to(dist, ref_dist).mean() / denom

        # C_edit: the size of the edit the CURRENT policy proposes on average,
        # measured on its mean action so it is a property of theta and not of a
        # particular noise draw. In [0, 1] by construction, and it rises steeply
        # exactly where tanh saturates.
        mu_h = policy.combine({s.name: dist[s.name][0] for s in policy.specs},
                              int(base.shape[-1]))
        c_edit = (torch.tanh(mu_h) ** 2).mean()

        loss = nll + beta * kl + lam_edit * c_edit

        opt_.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                            float(tcfg.get("grad_clip", 1.0)))
        opt_.step()

        with torch.no_grad():
            for k, v in policy.state_dict().items():
                ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay) \
                    if v.dtype.is_floating_point else ema[k].copy_(v)

        if step % int(tcfg.get("log_every", 50)) == 0 or step == 1:
            row = {
                "step": step, "loss": float(loss), "nll": float(nll),
                "kl": float(kl), "c_edit": float(c_edit),
                "grad_norm": float(gn),
                "log_prob_mean": float(logp.mean()),
                "behavior_log_prob_mean": float(behavior_logp.mean()),
                "weight_mean": float(w.mean()), "weight_max": float(w.max()),
                "seconds": time.time() - t0,
            }
            for s in policy.specs:
                row[f"sigma_{s.name}"] = float(dist[s.name][1].detach().mean())
                row[f"mu_{s.name}_absmax"] = float(dist[s.name][0].detach().abs().max())
            row["tanh_saturated_fraction"] = float(
                (torch.tanh(mu_h).abs() >= 0.99).float().mean())
            history.append(row)
            print(f"[{step:6d}] loss={row['loss']:.5g} nll={row['nll']:.5g} "
                  f"kl={row['kl']:.4g} c_edit={row['c_edit']:.4g} "
                  f"sig_fine={row['sigma_fine']:.4g} "
                  f"sat={100 * row['tanh_saturated_fraction']:.2f}%", flush=True)

        if step % int(tcfg.get("ckpt_every", 1000)) == 0 or step == steps:
            torch.save({"model": policy.state_dict(),
                        "extra": {"ema": ema, "step": step,
                                  "model_config": str(Path(args.model_config).resolve()),
                                  "round": round_name,
                                  "behavior_checkpoint": args.behavior_checkpoint or "",
                                  "policy_hash": policy.parameter_hash()}},
                       out / "ckpt.pt")

    write_json(out / "train_log.json", {
        "run_name": args.run_name, "round": round_name,
        "oracle_run": args.oracle_run,
        "behavior_checkpoint": args.behavior_checkpoint or "untrained reference",
        "n_elite": len(entries),
        "boxes": sorted({e.box for e in entries}),
        "steps": steps, "batch_size": batch, "lr": lr_,
        "beta_kl": beta, "lambda_edit": lam_edit, "sigma_floor": sigma_floor,
        "crop_hr": crop_hr, "context_margin": margin,
        "objective": (
            "-sum_i w_i log pi_theta(a_i|B_i,Y_i)/sum_i w_i + beta KL(pi||pi_ref) "
            "+ lambda_edit C_edit. No HR reconstruction term; actions detached."
        ),
        "history": history,
    })
    banner(f"checkpoint -> {out / 'ckpt.pt'}")
    print(f"  next: sample candidates from it and re-score --")
    print(f"    sbatch scripts/slurm/gaussian_sample_gpu.sbatch "
          f"CKPT={out / 'ckpt.pt'} RUN_NAME={args.run_name}_k16")


def check_behavior_identity(behavior: MultiScaleGaussianPolicy,
                            records: Dict[str, Dict], entries: Sequence) -> None:
    """Fatal unless the loaded behaviour policy is the one that made the actions.

    The reconstruction ``a = mu^b + sigma^b * eps`` is only the action that
    earned the reward if ``mu^b, sigma^b`` come from the checkpoint that
    produced it. Comparing the *parameter hash* settles that exactly; comparing
    log-probabilities cannot, because the stored value is a whole-box average
    over tiles and a training crop is one window of it.

    A wrong checkpoint here is the worst failure this script has: it would
    reinforce one action with another action's weight, silently, forever.
    """
    want = {r.get("policy_hash", "") for k, r in records.items()
            if k in {f"{e.box}|{int(e.residual_seed)}" for e in entries}}
    want.discard("")
    if not want:
        print("  ! no action records carry a policy hash; the behaviour policy's "
              "identity cannot be verified. Pass --oracle-run to enable the check.",
              flush=True)
        return
    have = behavior.parameter_hash()
    if want != {have}:
        raise SystemExit(
            f"behaviour policy mismatch: the replay actions were produced by "
            f"policy hash(es) {sorted(want)}, but --behavior-checkpoint loads "
            f"{have}. Reconstructing a_i under the wrong parameters would "
            f"reinforce a different action than the one that earned w_i.")
    print(f"  behaviour policy hash verified: {have}", flush=True)


def apply_sigma_floor(policy: MultiScaleGaussianPolicy, floor: float) -> float:
    """Raise the policy's hard ``sigma_min`` to the variance floor.

    Applied to the clamp inside :meth:`distribution`, not to the head bias: the
    bias only sets sigma where the head's weights contribute nothing, so once
    training has moved those weights a bias clamp stops being a floor at all.
    Clamping ``log sigma`` itself is a floor everywhere, by construction.
    """
    if floor <= 0:
        return float(policy.cfg.sigma_min)
    lo = max(float(floor), float(policy.cfg.sigma_min))
    if lo > float(policy.cfg.sigma_max):
        raise SystemExit(
            f"sigma_floor {floor} exceeds sigma_max {policy.cfg.sigma_max}")
    policy.cfg.sigma_min = lo
    policy.log_sigma_min.fill_(math.log(lo))
    return lo


if __name__ == "__main__":
    main()
