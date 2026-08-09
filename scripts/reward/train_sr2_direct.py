#!/usr/bin/env python
"""Train the actor: direct reward-guided fine-tuning of SR2, one rung at a time.

Refuses to start unless ``proxy_benchmark.json`` says this arm may advance. That
is the whole architecture of this line: the surrogate is checked against real
catalogs first -- including twelve real spliced-and-re-run Rockstar boxes -- and
only then is it allowed to move a pretrained generator's weights.

The benchmark, not a per-arm gate, is the authority. An arm can clear its own
criteria and still not be the one to advance, and the comparison that decides
that is made in one place.

    python scripts/reward/train_sr2_direct.py --run-name direct_a --rung proj_noise
    python scripts/reward/train_sr2_direct.py --run-name direct_a --overfit
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from _sr2_direct import (  # noqa: E402
    actor_config_of, add_direct_args, append_jsonl, assert_not_sealed, banner,
    boxes_of, direct_root, geometry_of, load_direct_config, load_hr, load_lr,
    load_reward_models, model_path_of, run_dir, soft_config_of, write_json,
)

from cosmo_sr.reward.base import find_base_field  # noqa: E402
from cosmo_sr.reward.catalog_proxy import ProxyEnsemble  # noqa: E402
from cosmo_sr.reward.tiles import TileSummary, read_tile_summaries  # noqa: E402
from cosmo_sr.reward.torch_reward import summary_from_tiles  # noqa: E402
from cosmo_sr.train import sr2_unfreeze  # noqa: E402
from cosmo_sr.train.sr2_finetune_data import SR2TileDataset, collate_tiles  # noqa: E402
from cosmo_sr.train.train_sr2_direct import (  # noqa: E402
    DirectFinetuneTrainer, attach_summaries,
)
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402


def frozen_summaries_for(cfg, box: str) -> Dict[int, TileSummary]:
    """The frozen box's real per-tile summaries, from a labelled candidate."""
    p = direct_root("candidates", f"{box}__frozen_seed0") / "tile_summaries.jsonl"
    if not p.is_file():
        raise SystemExit(
            f"no frozen tile summaries for {box} at {p}; run "
            "collect_catalog_proxy_data.py --stage generate/label --source frozen")
    return {int(s.tile_id): s for s in read_tile_summaries(p)}


def frozen_field_for(cfg, box: str, seed: int) -> Optional[np.ndarray]:
    p = direct_root("candidates", f"{box}__frozen_seed{int(seed)}") / "field.npy"
    if p.is_file():
        return np.load(p, mmap_mode="r")
    cached = find_base_field(box, int(seed))
    return np.load(cached, mmap_mode="r") if cached else None


def host_rich_tiles(summaries: Dict[int, TileSummary], n: int,
                    upper_bins: List[int]) -> List[int]:
    """Tiles carrying the most weight in the upper reliable host bins.

    The small overfit needs tiles where the occupation statistic can move at
    all: a tile with no 1e13+ host contributes nothing to the bins Gate B is
    decided on, so training on it measures the guards and nothing else.
    """
    score = {
        t: float(np.sum([s.n_host[b] for b in upper_bins if b < len(s.n_host)]))
        for t, s in summaries.items()
    }
    return [t for t, _ in sorted(score.items(), key=lambda kv: -kv[1])[:int(n)]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--rung", default="")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume", default="", help="actor checkpoint to continue from")
    ap.add_argument("--overfit", action="store_true",
                    help="section 10: one or two boxes, fixed host-rich tiles, "
                         "proj_noise only, short")
    ap.add_argument("--boxes", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--arm", default="a",
                    help="which proxy arm to fine-tune. It must appear in "
                         "proxy_benchmark.json's advance list; the benchmark, "
                         "not this flag, decides what may be advanced")
    ap.add_argument("--ignore-gate", action="store_true",
                    help="run without a passing proxy gate (debugging only; the "
                         "resulting checkpoint is not evidence about anything)")
    args = ap.parse_args(argv)

    cfg = load_direct_config(args)
    run = run_dir(args.run_name, create=True)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    acfg = actor_config_of(cfg)
    ocfg = dict(cfg.get("overfit", {}))
    loader_cfg = dict(cfg.get("actor", {}))

    # ---- the gate ---------------------------------------------------------
    # The authority is proxy_benchmark.json, not a per-arm verdict: the arms are
    # compared there, and its `decision` is what says whether ANY arm may be
    # fine-tuned and which. Reading a single arm's gate would let an arm that
    # passed on its own be advanced past a decision that said not to.
    bench_file = run / "proxy_benchmark.json"
    if not args.ignore_gate:
        if not bench_file.is_file():
            print(f">>> MISSING INPUT: {bench_file}")
            print(">>> produced by: scripts/slurm/submit_proxy_benchmark.sh all")
            print(">>> exiting 0 so dependents report the same rather than "
                  ">>> stranding on DependencyNeverSatisfied.")
            return 0
        bench = json.loads(bench_file.read_text())
        decision = dict(bench.get("decision", {}))
        advance = [str(a) for a in decision.get("advance", [])]
        if not advance:
            print(">>> DECISION: " + str(decision.get("decision", "unknown")))
            print(">>> " + str(decision.get("rationale", "")))
            for a, fails in (bench.get("failures") or {}).items():
                for f in fails:
                    print(f">>>   arm {a}: {f}")
            print(">>> Not training. Improve the proxy or its data; do NOT "
                  ">>> compensate by unfreezing more of the generator.")
            return 0
        if str(args.arm) not in advance:
            print(f">>> arm {args.arm!r} is not in the benchmark's advance list "
                  f"{advance}.")
            print(">>> " + str(decision.get("rationale", "")))
            print(">>> Not training this arm.")
            return 0
        print(f"=== benchmark decision {decision.get('decision')}: advancing "
              f"arm {args.arm}" +
              (f" (preferred: {decision['preferred']})" if "preferred" in decision
               else ""), flush=True)

    # ---- what to train ----------------------------------------------------
    if args.overfit:
        boxes = [str(b) for b in ocfg.get("boxes", ["set0"])]
        rung = str(ocfg.get("rung", "proj_noise"))
        max_steps = int(args.max_steps or ocfg.get("max_steps", 300))
    else:
        boxes = ([b.strip() for b in args.boxes.split(",") if b.strip()]
                 if args.boxes else boxes_of(cfg, "actor_train"))
        rung = str(args.rung or acfg.rung)
        max_steps = int(args.max_steps or acfg.max_steps)
    assert_not_sealed(cfg, boxes)
    acfg.rung = rung
    acfg.max_steps = max_steps

    # ---- data -------------------------------------------------------------
    base_seed = 0
    lr_fields, hr_fields, frozen_fields, summaries = {}, {}, {}, {}
    for b in boxes:
        lr_fields[b] = np.asarray(load_lr(cfg, b), dtype=np.float32)
        hr_fields[b] = load_hr(cfg, b)
        f = frozen_field_for(cfg, b, base_seed)
        if f is not None:
            frozen_fields[b] = f
        summaries[b] = frozen_summaries_for(cfg, b)

    tile_ids = None
    if args.overfit:
        upper = [int(i) for i in ocfg.get("upper_reliable_host_bins", (2, 3))]
        tile_ids = host_rich_tiles(summaries[boxes[0]],
                                   int(ocfg.get("n_tiles", 4)), upper)
        banner(f"overfit tiles (host-rich, bins {upper}): {tile_ids}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    frozen_gen = load_controlled_generator(
        model_path_of(cfg), scale_factor=geom.scale_factor,
        device=torch.device("cpu"), eval_mode=True)

    ds = SR2TileDataset(
        boxes=boxes, lr_fields=lr_fields, hr_fields=hr_fields, geom=geom,
        frozen_fields=frozen_fields, frozen_summaries=summaries,
        frozen_generator=frozen_gen, tile_ids=tile_ids, base_seed=base_seed,
        noise_draws=int(loader_cfg.get("noise_draws", 2)))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(loader_cfg.get("batch_size", 2)), shuffle=True,
        collate_fn=collate_tiles, num_workers=0, drop_last=True)

    # S_0 and s_{0,j}: measured constants of the frozen box.
    box_summaries = {b: summary_from_tiles(list(summaries[b].values())) for b in boxes}
    frozen_tiles = {
        (b, t): summary_from_tiles([summaries[b][t]]) for b in boxes for t in summaries[b]
    }

    # ---- model ------------------------------------------------------------
    _, reward_t = load_reward_models(cfg)
    proxy_dir = run / f"proxy_{args.arm}"
    if not proxy_dir.is_dir():
        print(f">>> MISSING INPUT: {proxy_dir}")
        print(">>> produced by: scripts/reward/train_catalog_proxy.py")
        return 0
    proxies = ProxyEnsemble.load(proxy_dir)

    # The trainer computes the actor's features itself, and today it computes
    # arm A's -- density only. An arm-B ensemble expects the phase-space block
    # too, and feeding it a 26-vector where it wants 44 is a shape error deep in
    # a training step rather than a statement about what is missing.
    from cosmo_sr.reward.soft_structure import feature_names
    want = int(proxies.members[0].cfg.n_features)
    have = 2 * len(feature_names(scfg))
    if want != have:
        print(f">>> arm {args.arm!r}'s proxy takes {want} features, but the actor "
              f">>> computes {have}.")
        print(">>> Fine-tuning against a phase-space proxy needs the actor's own "
              ">>> feature path extended to arm B (cosmo_sr.reward.phase_space."
              ">>> arm_paired_features). That is deliberately not part of the "
              ">>> benchmark milestone; not training.")
        return 0

    trainer = DirectFinetuneTrainer(
        model_path_of(cfg), proxies, reward_t, cfg=acfg, geom=geom,
        soft_cfg=scfg, device=device)
    described = sr2_unfreeze.print_trainable(trainer.actor, rung, acfg.group_lr)
    stage = run / ("overfit" if args.overfit else f"rung_{rung}")
    stage.mkdir(parents=True, exist_ok=True)
    write_json(stage / "trainable_parameters.json", described)
    if args.resume:
        trainer.load(args.resume)
        banner(f"resumed from {args.resume} at step {trainer.step_index}")

    write_json(stage / "config.json", {
        "run_name": args.run_name, "rung": rung, "boxes": boxes,
        "tile_ids": tile_ids, "max_steps": max_steps,
        "actor": acfg.to_dict(), "n_proxy_members": len(proxies),
        "overfit": bool(args.overfit), "ignore_gate": bool(args.ignore_gate),
        "adversarial_weight": float(cfg.get("adversarial", {}).get("weight", 0.0)),
    })

    # ---- train ------------------------------------------------------------
    log = stage / "metrics.jsonl"
    t0 = time.time()
    step = trainer.step_index
    banner(f"training rung {rung}: {max_steps} steps over {len(ds)} tiles")
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = attach_summaries(batch, box_summaries, frozen_tiles)
            m = trainer.step(batch)
            m["elapsed_s"] = round(time.time() - t0, 1)
            append_jsonl(log, m)
            if step % int(acfg.log_every) == 0:
                print(f"  step {step:5d} loss {m['loss']:+.5f} "
                      f"q_safe {m['q_safe']:+.4f} dR_occ {m['dR_occ']:+.4f} "
                      f"low_k {m['low_k_change']:.2e} "
                      f"div {m.get('div_d_struct', float('nan')):.3f}", flush=True)
            step = trainer.step_index
            if step % int(acfg.checkpoint_every) == 0:
                trainer.save(stage / "last.pt")
                trainer.save_ema_generator(stage / "ema_generator.pt")

    trainer.save(stage / "last.pt")
    trainer.save_ema_generator(
        stage / "ema_generator.pt",
        extra={"run_name": args.run_name, "rung": rung, "boxes": boxes,
               "steps": int(trainer.step_index)})
    banner(f"done in {time.time() - t0:.0f}s -> {stage}")
    print("  the EMA checkpoint is the one to evaluate; it loads through "
          "load_controlled_generator like G_z0.pt.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
