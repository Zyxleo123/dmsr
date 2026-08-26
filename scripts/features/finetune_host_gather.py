#!/usr/bin/env python
"""Fine-tune SR2 with an auxiliary loss that gathers particles into true HR subhalos.

The sequel to ``scripts/features/overfit_host_mse.py``. That script settled
``docs/pilot_steps_2_4.md`` section 2: at 4.4 trainable parameters per target
value the rung could have memorised one cluster outright and did not -- MSE
plateaued at 0.39x frozen while the high-k power fell. **Capacity is not the
limit; the squared loss is.** This script keeps everything else identical -- the
same box, the same cluster, the same rungs, the same field report -- and swaps
the objective for the one in :mod:`cosmo_sr.features.subhalo_gather`:

    L = w_gather * mean_s [1 - C_theta(s)/C_HR(s)]_+^2
        + w_low * ||A(Psi_theta) - A(Psi_0)||^2 / ||A(Psi_0)||^2
        + w_anchor * ||Psi_theta - Psi_0||^2 / ||Psi_0||^2
        + w_mse * ||Psi_theta - Psi_HR||^2          (0 by default)

``C(s)`` is soft-structure's compact-mass coordinate -- a sigmoid on
``log(1 + delta)``, the form whose gradient is well conditioned across the whole
density range -- read in a Gaussian window at the Eulerian centre of a *true HR
subhalo*. It asks for a collapsed object at a known place and says nothing about
which particle goes there, so blurring is the worst move available rather than
the minimiser.

Where the HR information enters, and where it does not
-----------------------------------------------------
The HR catalog, the owner array and the reference statistics ``C_HR`` are built
**once, before the loop**, into a :class:`GatherTargets` tensor. They are data in
the loss, exactly as HR displacements are data in an MSE. The generator's inputs
are unchanged, so a checkpoint from this run is run at inference from ``(Y, z)``
alone -- no catalog, no HR field, no change to the sampling path.

The guards, and why they are here
---------------------------------
An objective that adds mass at chosen points can be satisfied by wrecking
everything else. ``w_low`` holds the block-averaged (LR-scale) field to the
frozen generator's, ``w_anchor`` holds the whole field weakly, and every eval
step re-reports the *unmodified* ``field_report`` from the MSE experiment --
windowed cross-spectra against HR, the density one-point statistics and the
soft-structure ratios. A run that raises ``compact_ratio`` while ``low_k_change``
climbs has not done the thing this experiment is testing.

What this cannot say
--------------------
Whether the clumps are **bound**. Every number here is a field statistic; only
Rockstar on a reassembled box settles it (``sr2_substructure_module.md`` step 6).
And the supervision is in-sample by construction -- one host's tiles, with the
true subhalo positions in the loss -- so this measures whether the objective can
drive the generator to build substructure at all, not whether it generalises.

    python scripts/features/finetune_host_gather.py --box set8 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, paths, write_json  # noqa: E402
from _sr2_direct import (  # noqa: E402
    add_direct_args, geometry_of, load_direct_config, model_path_of,
    phase_space_config_of, soft_config_of,
)
# The MSE experiment's host selection, tile builder and field report, reused
# unchanged: the two runs must be about the same cluster measured the same way,
# and a second copy of `host_tiles` would be a second definition of "the big
# halo we evaluate on".
from overfit_host_mse import (  # noqa: E402
    BOXSIZE, NG_HR, TILE, _catalog, _owner, build_tiles, field_report,
    forward_tiles, host_tiles,
)
from render_gather_slices import render_npz  # noqa: E402

from cosmo_sr.eval.density import valid_center_bulk  # noqa: E402
from cosmo_sr.eval.particle_identity import build_owner_index  # noqa: E402
from cosmo_sr.features.subhalo_gather import (  # noqa: E402
    GatherConfig, attach_hr_reference, deposit_for_gather, gather_loss,
    outside_weight_map, per_subhalo_table, preserve_loss, preserve_statistic,
    stack_tile_subhalos, subhalo_home_tiles, tile_subhalos,
)
from cosmo_sr.train.train_sr2_direct import block_average_torch  # noqa: E402
from cosmo_sr.train.sr2_unfreeze import (  # noqa: E402
    assert_only_trainable_changed, parameter_groups, snapshot_parameters,
    trainable_names,
)
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def build_targets(box, tiles, frozen_field, hr_field, scfg, pscfg, gcfg, device):
    """``(GatherTargets, bulks, per-tile report)`` for the training tiles.

    ``bulks`` is the frozen generator's rounded valid-centre offset per tile, and
    it is used for **all three** deposits (candidate, frozen, HR). Each field
    rounding its own bulk would put the three grids at whole-cell offsets from
    each other, and the window that is a subhalo in one would be its neighbour in
    another.
    """
    cat = _catalog(box, "hr")
    oidx = build_owner_index(_owner(box, "hr"))
    home = subhalo_home_tiles(cat, oidx, ng_hr=NG_HR, tile_hr=TILE,
                              min_num_p=int(gcfg.min_num_p))

    bulks = {}
    per_tile = []
    for i, t in enumerate(tiles):
        b = valid_center_bulk(frozen_field[i: i + 1, 0:3].float(),
                              scfg.cellsize_kpc_h, float(scfg.dis_norm_kpc_h))
        bulks[t] = b[0].detach().cpu().numpy()
        per_tile.append(tile_subhalos(cat, home, t, bulks[t], gcfg, scfg,
                                      tile_hr=TILE))

    targets = stack_tile_subhalos(per_tile, device=device)
    bulk_t = torch.stack([torch.as_tensor(bulks[t], dtype=torch.float32)
                          for t in tiles]).to(device)
    # ONE deposit gives density and kinematics on the same cells, so "HR's mass
    # but not HR's dispersion" can never be a statement about two regions.
    hr_dep = deposit_for_gather(hr_field.float(), scfg, pscfg, bulk=bulk_t)
    targets = attach_hr_reference(targets, hr_dep, gcfg,
                                  grid_mult=int(scfg.grid_mult))
    # The complement of the supervision, and the frozen generator's structure in
    # it. Both are constants of the run: the windows never move, and "do not
    # degrade what SR2 already had here" is measured against the field we started
    # from, not against HR (which SR2 does not match there either).
    grid = int(hr_dep.delta.shape[-1])
    targets.outside_w = outside_weight_map(
        targets.centre, targets.sigma, targets.half_width, grid,
        mask=targets.mask, device=targets.centre.device)
    frozen_dep = deposit_for_gather(frozen_field.float(), scfg, pscfg, bulk=bulk_t)
    with torch.no_grad():
        targets.frozen_preserve = preserve_statistic(
            frozen_dep.delta, targets.outside_w, gcfg)

    live = targets.mask.cpu().numpy()
    report = {
        "n_selected": int(sum(t.n for t in per_tile)),
        "n_targets": int(live.sum()),
        "half_width": int(targets.half_width),
        "per_tile": [
            {"tile": int(t.tile_id), "n_selected": int(t.n),
             "n_targets": int(live[i].sum()),
             "num_p_median": (float(np.median(t.num_p)) if t.n else 0.0),
             "num_p_max": (int(np.max(t.num_p)) if t.n else 0)}
            for i, t in enumerate(per_tile)],
        "hr_compact_mean": float(
            (targets.hr_compact * targets.mask).sum().item()
            / max(float(targets.mask.sum().item()), 1.0)),
        "outside_mass_fraction": float(
            (targets.outside_w * (1.0 + hr_dep.delta)).sum().item()
            / float((1.0 + hr_dep.delta).sum().item())),
        "frozen_preserve": [float(x) for x in targets.frozen_preserve],
        "hr_vdisp_mean_km_s": float(
            (targets.hr_vdisp * targets.mask).sum().item()
            / max(float(targets.mask.sum().item()), 1.0)),
    }
    return targets, bulk_t, report


# --------------------------------------------------------------------------- #
# Loss terms
# --------------------------------------------------------------------------- #
def loss_terms(out, base, hr, dep, targets, gcfg, geom, scfg):
    """The five terms, unweighted, plus the gather diagnostics."""
    g_loss, diag = gather_loss(dep, targets, gcfg, grid_mult=int(scfg.grid_mult))
    p_loss, p_ratio = preserve_loss(dep.delta, targets, gcfg)
    diag["preserve_ratio"] = p_ratio

    a_c = block_average_torch(out, geom.scale_factor)
    a_b = block_average_torch(base, geom.scale_factor).detach()
    # The SQUARED relative change, never its square root: at step zero the actor
    # IS the frozen generator, this term is exactly zero, and d(sqrt)/dx there is
    # infinite -- measured in the direct line, NaN weights within three steps.
    low = (a_c - a_b).pow(2).mean() / a_b.pow(2).mean().clamp_min(1e-30)
    anchor = (out - base.detach()).pow(2).mean() / base.detach().pow(2).mean().clamp_min(1e-30)
    mse = (out - hr).pow(2).mean()

    terms = {"gather": g_loss, "preserve": p_loss, "low": low, "anchor": anchor,
             "mse": mse}
    return terms, diag


def weights_of(args):
    return {"gather": float(args.w_gather), "preserve": float(args.w_preserve),
            "low": float(args.w_low), "anchor": float(args.w_anchor),
            "mse": float(args.w_mse)}


def grad_norms(terms, weights, groups):
    """Weighted gradient norm of each term, w.r.t. the trainable parameters.

    The measurement that answers "is the auxiliary loss overwhelming the guards,
    or the other way round". One extra backward per term, so it runs on eval
    steps only.
    """
    params = [p for g in groups for p in g["params"]]
    out = {}
    for k, t in terms.items():
        if float(weights[k]) == 0.0 or not t.requires_grad:
            out[f"gradnorm_{k}"] = 0.0
            continue
        g = torch.autograd.grad(float(weights[k]) * t, params,
                                retain_graph=True, allow_unused=True)
        out[f"gradnorm_{k}"] = float(
            sum(float(x.detach().pow(2).sum()) for x in g if x is not None) ** 0.5)
    return out


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    pscfg = phase_space_config_of(cfg)
    gcfg = GatherConfig.from_soft(
        scfg,
        contrast_scale=int(args.contrast_scale),
        w_contrast=float(args.w_contrast),
        w_vdisp=float(args.w_vdisp),
        w_vbulk=float(args.w_vbulk),
        w_preserve=float(args.w_preserve),
        sigma_floor_cells=float(args.sigma_floor),
        sigma_rvir_factor=float(args.sigma_rvir),
        radius_factor=float(args.radius_factor),
        min_num_p=int(args.min_num_p),
        min_purity=float(args.min_purity),
        min_hr_compact=float(args.min_hr_compact),
    )
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    banner(f"gather fine-tune: {args.box}, rung={args.rung}")

    meta, train, hold = host_tiles(args.box, args)
    print(f"  host {meta['halo_id']} logM={meta['log_mvir']:.2f} "
          f"num_p={meta['num_p']} sites={meta['n_member_sites']}")
    print(f"  train tiles {train} (sites {meta['train_tile_member_sites']}), "
          f"holdout tile {hold}", flush=True)

    model = load_controlled_generator(
        model_path_of(cfg), in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=False)
    frozen = load_controlled_generator(
        model_path_of(cfg), in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    for p in frozen.parameters():
        p.requires_grad_(False)

    lrs = {g: v * args.lr_scale for g, v in
           {"proj_noise": 1e-5, "fine": 3e-6, "middle": 1e-6, "coarse": 3e-7}.items()}
    groups = parameter_groups(model, args.rung, lrs)
    names = trainable_names(model)
    theta0 = snapshot_parameters(model)
    n_train_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"  rung {args.rung}: {len(names)} tensors, {n_train_params} parameters "
          f"trainable of {sum(p.numel() for p in model.parameters())}", flush=True)

    data = build_tiles(args.box, train + ([hold] if hold >= 0 else []), cfg, geom,
                       device, args)
    hr_train = torch.stack([data[t]["hr"] for t in train], dim=0)
    with torch.no_grad():
        base_train = forward_tiles(frozen, data, train, geom)

    targets, bulk_train, tgt_report = build_targets(
        args.box, train, base_train, hr_train, scfg, pscfg, gcfg, device)
    print(f"  targets: {tgt_report['n_targets']} live HR subhalos over "
          f"{len(train)} tiles (window half-width {tgt_report['half_width']} "
          f"cells, mean HR compact mass {tgt_report['hr_compact_mean']:.1f} "
          f"particles)", flush=True)
    for row in tgt_report["per_tile"]:
        print(f"    tile {row['tile']:4d}: {row['n_targets']:4d} targets "
              f"(median {row['num_p_median']:.0f}p, max {row['num_p_max']}p)")
    if tgt_report["n_targets"] == 0:
        raise SystemExit(
            "no live targets: every HR subhalo was filtered out. Lower "
            "--min-num-p / --min-hr-compact / --min-purity, or widen the scored "
            "cube with --set soft_structure.region_fraction=0.75.")
    if args.targets_only:
        # The shakeout: selection, geometry and the HR reference statistics are
        # everything that can be wrong before a single step is taken, and they
        # cost two forward passes to check. Run this first; a run that reports
        # three targets is not worth three GPU hours.
        out_dir = paths.subdir("host_gather", f"{args.box}_h{meta['halo_id']}"
                               f"_{args.rung}{args.label}", create=True)
        with torch.no_grad():
            base_dep = deposit_for_gather(base_train.float(), scfg, pscfg,
                                          bulk=bulk_train)
            _, frozen_diag = gather_loss(base_dep, targets, gcfg,
                                         grid_mult=int(scfg.grid_mult))
        write_json(out_dir / "targets.json",
                   {"host": meta, "targets": tgt_report,
                    "frozen_gather": frozen_diag,
                    "gather_config": gcfg.to_dict()})
        print(f"  frozen generator: compact_ratio "
              f"{frozen_diag['compact_ratio']:.3f} of HR at these targets")
        print(f"  targets-only: wrote {out_dir / 'targets.json'}")
        return {"ok": True, "targets_only": True, "targets": tgt_report,
                "frozen_gather": frozen_diag, "host": meta,
                "verdict": {"text": "targets-only shakeout; nothing trained"},
                "out_dir": str(out_dir)}

    with torch.no_grad():
        base_dep = deposit_for_gather(base_train.float(), scfg, pscfg,
                                      bulk=bulk_train)
        hr_dep = deposit_for_gather(hr_train.float(), scfg, pscfg, bulk=bulk_train)
        base_delta, hr_delta = base_dep.delta, hr_dep.delta
        _, frozen_diag = gather_loss(base_dep, targets, gcfg,
                                     grid_mult=int(scfg.grid_mult))
    print(f"  frozen generator: compact_ratio {frozen_diag['compact_ratio']:.3f} "
          f"(median {frozen_diag['compact_ratio_median']:.3f}), "
          f"contrast_ratio {frozen_diag['contrast_ratio']:.3f}, "
          f"satisfied {frozen_diag['compact_satisfied']:.3f}")
    print(f"  frozen kinematics: vdisp {frozen_diag.get('vdisp_ratio', float('nan')):.3f} "
          f"of HR's {frozen_diag.get('vdisp_hr_mean', float('nan')):.0f} km/s, "
          f"bulk offset {frozen_diag.get('vbulk_offset', float('nan')):.3f} sigma",
          flush=True)

    out_dir = paths.subdir("host_gather", f"{args.box}_h{meta['halo_id']}"
                           f"_{args.rung}{args.label}", create=True)
    (out_dir / "eval").mkdir(exist_ok=True)
    jsonl = out_dir / "metrics.jsonl"
    jsonl.write_text("")
    weights = weights_of(args)

    opt = torch.optim.Adam(groups)
    rng = np.random.default_rng(args.seed)
    t0, history, last_report = time.time(), [], None
    for step in range(1, args.steps + 1):
        idx = rng.choice(len(train), size=min(args.batch, len(train)), replace=False)
        idx = np.sort(idx)
        pick = [train[i] for i in idx]
        out = forward_tiles(model, data, pick, geom)
        bulk = bulk_train[idx]
        dep = deposit_for_gather(out.float(), scfg, pscfg, bulk=bulk)
        terms, diag = loss_terms(out, base_train[idx], hr_train[idx], dep,
                                 targets.select(idx.tolist()), gcfg, geom, scfg)
        loss = sum(weights[k] * v for k, v in terms.items())

        gn = {}
        will_eval = (step % args.eval_every == 0 or step == args.steps)
        if will_eval or step == 1:
            gn = grad_norms(terms, weights, groups)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], args.clip)
        opt.step()

        if will_eval:
            model.eval()
            with torch.no_grad():
                cur = forward_tiles(model, data, train, geom)
                cur_dep = deposit_for_gather(cur.float(), scfg, pscfg,
                                             bulk=bulk_train)
                cur_delta = cur_dep.delta
                _, edia = gather_loss(cur_dep, targets, gcfg,
                                      grid_mult=int(scfg.grid_mult))
                _, edia["preserve_ratio"] = preserve_loss(cur_dep.delta, targets,
                                                          gcfg)
                rep = field_report(cur, hr_train, base_train, scfg, args)
                # The LR-scale change is a gate, so it is measured over every
                # training tile rather than over the two in this step's batch.
                a_c = block_average_torch(cur, geom.scale_factor)
                a_b = block_average_torch(base_train, geom.scale_factor)
                low_full = float((a_c - a_b).pow(2).mean().item()
                                 / max(float(a_b.pow(2).mean().item()), 1e-30))
                row = {
                    "step": step,
                    "loss": float(loss.item()),
                    **{f"term_{k}": float(v.item()) for k, v in terms.items()},
                    **gn,
                    "compact_ratio": edia["compact_ratio"],
                    "compact_ratio_frozen": frozen_diag["compact_ratio"],
                    # The mean of a ratio is heavy-tailed here -- a handful of
                    # windows sitting in the cluster core dominate it. The median
                    # is the number to read; the mean is kept for continuity.
                    "compact_ratio_median": edia["compact_ratio_median"],
                    "compact_ratio_median_frozen": frozen_diag["compact_ratio_median"],
                    "vdisp_ratio": edia.get("vdisp_ratio"),
                    "vdisp_ratio_frozen": frozen_diag.get("vdisp_ratio"),
                    "vbulk_offset": edia.get("vbulk_offset"),
                    "vbulk_offset_frozen": frozen_diag.get("vbulk_offset"),
                    # Local peak structure OUTSIDE the windows, over the frozen
                    # generator's. This is the collateral-damage number, and it
                    # is in the verdict's field gate -- an earlier run passed on
                    # low_k alone while this sat at 0.57.
                    "preserve_ratio": edia["preserve_ratio"],
                    "contrast_ratio": edia["contrast_ratio"],
                    "contrast_ratio_frozen": frozen_diag["contrast_ratio"],
                    "compact_satisfied": edia["compact_satisfied"],
                    "gather_loss": edia["gather_loss"],
                    "n_targets": edia["n_targets"],
                    # The gate-comparable RMS change, from the squared term.
                    "low_k_change": float(max(low_full, 0.0) ** 0.5),
                    "low_k_change_batch": float(
                        max(float(terms["low"].item()), 0.0) ** 0.5),
                    "mse_vs_hr": float(torch.mean((cur - hr_train) ** 2).item()),
                    "mse_frozen_vs_hr": float(
                        torch.mean((base_train - hr_train) ** 2).item()),
                    "highk_power_ratio": rep["out"]["highk_power_ratio"],
                    "highk_power_ratio_frozen": rep["frozen"]["highk_power_ratio"],
                    "frac_above_delta100": rep["density"]["out"]["frac_above_delta100"],
                    "frac_above_delta100_hr": rep["density"]["hr"]["frac_above_delta100"],
                    "peak_contrast_s1_over_hr":
                        rep["structure"]["out"]["peak_contrast_s1_over_hr"],
                    "peak_contrast_s1_over_hr_frozen":
                        rep["structure"]["frozen"]["peak_contrast_s1_over_hr"],
                    "seconds": round(time.time() - t0, 1),
                }
                if hold >= 0:
                    hout = forward_tiles(model, data, [hold], geom)
                    row["mse_holdout"] = float(
                        torch.mean((hout - data[hold]["hr"].unsqueeze(0)) ** 2).item())

                # The eval artifact: everything a figure needs, so the PNGs are
                # redrawn from disk (render_gather_slices.py) and never require
                # the generator again.
                npz = out_dir / "eval" / f"step{step:06d}.npz"
                np.savez_compressed(
                    npz, step=step, tiles=np.array(train),
                    delta_out=cur_delta[:, 0].cpu().numpy().astype(np.float32),
                    delta_frozen=base_delta[:, 0].cpu().numpy().astype(np.float32),
                    delta_hr=hr_delta[:, 0].cpu().numpy().astype(np.float32),
                    centre=targets.centre.cpu().numpy(),
                    sigma=targets.sigma.cpu().numpy(),
                    mask=targets.mask.cpu().numpy(),
                    num_p=targets.num_p.cpu().numpy(),
                    hr_compact=targets.hr_compact.cpu().numpy(),
                    halo_id=targets.halo_id.cpu().numpy(),
                    host_id=meta["halo_id"], box=args.box,
                    cellsize_mpc_h=float(BOXSIZE) / float(NG_HR)
                    / float(scfg.grid_mult))
                if args.png and (step % args.png_every == 0 or step == args.steps):
                    render_npz(npz, out_dir / "slices", slab=args.slab)
                # Checkpoint every eval, not only at the end: hitting the time
                # limit should cost the remaining steps, not the whole run.
                torch.save({"rung": args.rung, "step": step, "names": names,
                            "state": {n: q.detach().cpu()
                                      for n, q in model.named_parameters()
                                      if n in set(names)}},
                           out_dir / "trainable.pt")
            model.train()
            history.append(row)
            last_report = rep
            with jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  step {step:6d}  gather {edia['gather_loss']:.4f}"
                  f"  compact/HR {row['compact_ratio_median']:.3f} med"
                  f" (frozen {row['compact_ratio_median_frozen']:.3f})"
                  f"  vdisp/HR {row['vdisp_ratio']:.3f}"
                  f"  vbulk {row['vbulk_offset']:.2f}sig"
                  f"  outside {row['preserve_ratio']:.3f}"
                  f"  low_k {row['low_k_change']:.4f}"
                  f"  pk_ctr {row['peak_contrast_s1_over_hr']:.3f}", flush=True)

    with torch.no_grad():
        final = forward_tiles(model, data, train, geom)
        final_dep = deposit_for_gather(final.float(), scfg, pscfg, bulk=bulk_train)
        table = per_subhalo_table(final_dep, targets, gcfg,
                                  grid_mult=int(scfg.grid_mult))
        frozen_table = per_subhalo_table(base_dep, targets, gcfg,
                                         grid_mult=int(scfg.grid_mult))
    for a, b in zip(table, frozen_table):
        a["compact_frozen"] = b["compact"]
        a["contrast_frozen"] = b["contrast"]
        a["vdisp_frozen"] = b.get("vdisp")
        a["vbulk_err_frozen"] = b.get("vbulk_err")
    write_json(out_dir / "subhalos.json", {"rows": table})

    np.savez_compressed(out_dir / "tiles.npz",
                        tiles=np.array(train), out=final.cpu().numpy(),
                        frozen=base_train.cpu().numpy(), hr=hr_train.cpu().numpy())
    torch.save({"rung": args.rung, "names": names,
                "state": {n: p.detach().cpu() for n, p in model.named_parameters()
                          if n in set(names)}}, out_dir / "trainable.pt")

    assert_only_trainable_changed(model, theta0, names)
    drift = {n: float(torch.linalg.vector_norm(
        dict(model.named_parameters())[n].detach().cpu() - theta0[n]).item())
        for n in names}
    summary = {
        "ok": True,
        "box": args.box,
        "host": meta,
        "rung": args.rung,
        "objective": "subhalo_gather",
        "weights": weights,
        "gather_config": gcfg.to_dict(),
        "targets": tgt_report,
        "frozen_gather": frozen_diag,
        "trainable_tensors": names,
        "trainable_parameters": int(n_train_params),
        "steps": int(args.steps),
        "batch": int(args.batch),
        "lr_scale": float(args.lr_scale),
        "history": history,
        "final_report": last_report,
        "param_drift_l2": drift,
        "verdict": verdict(history, frozen_diag, args),
        "device": str(device),
        "seconds": round(time.time() - t0, 1),
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"  wrote {out_dir / 'summary.json'}")
    return summary


def verdict(history, frozen_diag, args) -> dict:
    """Three questions, answered separately, because the answer is the triple.

    A run can build the clumps, get their kinematics wrong, and leave the
    large-scale field intact -- and that is a different result from any of the
    three taken alone. In particular a density-only success is **not** a result:
    the channel swap measured HR losing 65% of its subhalos when given SR2's
    velocities, so a spatially perfect, kinematically wrong clump is one a
    phase-space finder discards.
    """
    if not history:
        return {"text": "no eval steps ran"}
    last = history[-1]
    # The median, not the mean: the mean of this ratio is dominated by the few
    # windows sitting in the cluster core.
    c0 = float(frozen_diag.get("compact_ratio_median",
                               frozen_diag.get("compact_ratio", 0.0)))
    c1 = float(last.get("compact_ratio_median", last.get("compact_ratio", 0.0)))
    gain = (c1 - c0) / max(c0, 1e-6)
    moved = c1 >= c0 + float(args.gain_min)
    low_k = float(last["low_k_change"])
    hk = float(last["highk_power_ratio"])
    # `low_k` alone is not "the field is intact": it constrains the block-averaged
    # LR scale only. An earlier run passed it at 0.0187 while the local peak
    # structure outside the supervised windows sat at 0.57 of the frozen
    # generator's -- the objective sharpening 43 windows and blurring everything
    # else. Both have to hold.
    pres = last.get("preserve_ratio")
    pres_ok = True if pres is None else (
        float(pres) >= 1.0 - float(args.contrast_drop_max))
    low_ok = low_k <= float(args.low_k_max) and pres_ok

    vd, vb = last.get("vdisp_ratio"), last.get("vbulk_offset")
    if vd is None:
        kin_ok, kin = None, "velocity not measured (w_vdisp = w_vbulk = 0)"
    else:
        # bool(), not the numpy scalar: this lands in JSON and in `is False`
        # checks, and np.False_ satisfies neither.
        kin_ok = bool(abs(float(np.log(max(float(vd), 1e-6)))) <= float(args.vdisp_tol)
                      and float(vb) <= float(args.vbulk_tol))
        kin = (f"dispersion {float(vd):.3f} of HR's and bulk offset "
               f"{float(vb):.2f} sigma "
               f"(frozen {float(last.get('vdisp_ratio_frozen', float('nan'))):.3f} / "
               f"{float(last.get('vbulk_offset_frozen', float('nan'))):.2f})")

    parts = [
        f"density {'MOVED' if moved else 'FLAT'} "
        f"({c0:.3f} -> {c1:.3f} of HR, {gain:+.0%} median)",
        f"kinematics {'OK' if kin_ok else ('UNMEASURED' if kin_ok is None else 'WRONG')}: {kin}",
        f"field {'PRESERVED' if low_ok else 'DISTORTED'} "
        f"(low_k {low_k:.4f} vs {args.low_k_max}; unsupervised structure "
        f"{'n/a' if pres is None else format(float(pres), '.3f')} of frozen "
        f"vs {1.0 - float(args.contrast_drop_max):.2f}; high-k P/P_HR {hk:.3f})",
    ]
    if moved and kin_ok and low_ok:
        head = ("ALL THREE HELD -- and this is still a FIELD statement. Whether "
                "the clumps are bound is Rockstar's to say, and the supervision "
                "was the true positions, so a held-out box is what tests "
                "generalisation.")
    elif moved and kin_ok is False:
        head = ("DENSITY WITHOUT KINEMATICS: the clumps are in the right places "
                "with the wrong velocities, which is the configuration the "
                "channel swap showed a phase-space finder discarding 65% of. "
                "Raise --w-vdisp / --w-vbulk before reading the density gain.")
    elif moved and not pres_ok:
        head = ("SHARPENED HERE, BLURRED EVERYWHERE ELSE: the targets improved "
                "while the unsupervised field lost local structure. One "
                "convolutional operator acts at every site, so this is the "
                "objective's cost, not a coincidence -- raise --w-preserve. Note "
                "the L2 --w-anchor does NOT fix it (measured: 100x moved the "
                "outside contrast 0.570 -> 0.517).")
    elif moved and not low_ok:
        head = ("MOVED, BUT PAID FOR IT: the gain is not separable from a "
                "large-scale distortion; raise --w-low, or lower --lr-scale, "
                "and rerun.")
    elif moved:
        head = "DENSITY MOVED; read the other two lines before calling it a result."
    else:
        head = ("NO MOVEMENT in density. Check gradnorm_gather against "
                "gradnorm_low and gradnorm_anchor in the history before "
                "concluding anything about the objective.")

    return {"text": head + "  [" + "; ".join(parts) + "]",
            "compact_ratio_frozen": c0, "compact_ratio_final": c1,
            "relative_gain": gain, "low_k_change": low_k,
            "vdisp_ratio": vd, "vbulk_offset": vb, "preserve_ratio": pres,
            "moved": bool(moved), "kinematics_ok": kin_ok,
            "field_preserved": bool(low_ok)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_direct_args(ap)
    ap.add_argument("--box", default="set8")
    ap.add_argument("--host-id", type=int, default=-1,
                    help="HR halo id to fine-tune on; -1 takes the most massive")
    ap.add_argument("--n-candidate-hosts", type=int, default=8)
    ap.add_argument("--rung", default="fine")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n-tiles", type=int, default=4)
    ap.add_argument("--lr-scale", type=float, default=10.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    # --- the objective -----------------------------------------------------
    ap.add_argument("--w-gather", type=float, default=1.0)
    ap.add_argument("--w-contrast", type=float, default=1.0,
                    help="weight of the peak-contrast term inside the gather loss")
    ap.add_argument("--w-vdisp", type=float, default=1.0,
                    help="velocity-dispersion match. NOT optional: Rockstar links "
                         "in 6-D and the channel swap measured HR losing 65%% of "
                         "its subhalos when given SR2's velocities")
    ap.add_argument("--w-vbulk", type=float, default=1.0,
                    help="bulk-velocity match, in units of the subhalo's own "
                         "HR dispersion")
    ap.add_argument("--w-preserve", type=float, default=1.0,
                    help="hinge against LOSING local peak contrast outside the "
                         "supervised windows. An L2 anchor cannot do this job: "
                         "measured, raising it 100x moved the outside contrast "
                         "ratio 0.570 -> 0.517, the wrong way")
    ap.add_argument("--w-low", type=float, default=1.0,
                    help="LR-scale (block-averaged) anchor to the frozen field")
    ap.add_argument("--w-anchor", type=float, default=0.1,
                    help="weak whole-field anchor to the frozen field")
    ap.add_argument("--w-mse", type=float, default=0.0,
                    help="plain MSE against HR; 0 by default -- it is the "
                         "objective this experiment exists to replace")
    ap.add_argument("--contrast-scale", type=int, default=2)
    ap.add_argument("--sigma-floor", type=float, default=1.0,
                    help="kernel width floor in deposit cells; also the radius "
                         "within which material can be gathered")
    ap.add_argument("--sigma-rvir", type=float, default=1.0)
    ap.add_argument("--radius-factor", type=float, default=2.5)
    ap.add_argument("--min-num-p", type=int, default=50)
    ap.add_argument("--min-purity", type=float, default=0.5)
    ap.add_argument("--min-hr-compact", type=float, default=5.0)
    # --- reporting ---------------------------------------------------------
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--k-split", type=float, default=4.0)
    ap.add_argument("--png", dest="png", action="store_true", default=True)
    ap.add_argument("--no-png", dest="png", action="store_false")
    ap.add_argument("--png-every", type=int, default=500)
    ap.add_argument("--slab", type=int, default=4,
                    help="half-thickness in cells of the max-projected slab")
    ap.add_argument("--gain-min", type=float, default=0.05)
    ap.add_argument("--contrast-drop-max", type=float, default=0.10,
                    help="fractional loss of local peak structure outside the "
                         "supervised windows the verdict tolerates")
    ap.add_argument("--vdisp-tol", type=float, default=0.15,
                    help="|log(sigma_v/sigma_v_HR)| the verdict calls kinematically "
                         "matched; 0.15 is ~15 per cent")
    ap.add_argument("--vbulk-tol", type=float, default=0.5,
                    help="bulk-velocity offset, in units of the subhalo's own HR "
                         "dispersion, the verdict calls matched")
    ap.add_argument("--low-k-max", type=float, default=0.02)
    ap.add_argument("--targets-only", action="store_true",
                    help="build and report the targets, then stop (shakeout)")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="")
    args = ap.parse_args(argv)

    s = run(args)
    print()
    print("VERDICT:", s["verdict"]["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
