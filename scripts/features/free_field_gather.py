#!/usr/bin/env python
"""Is the member-set objective a valid specification of a bound halo? Free-field test.

**Run this before training a network on the loss in
:mod:`cosmo_sr.features.member_gather`.** It answers the one question that
decides whether that loss is worth a GPU line, and it answers it in minutes.

Why this exists
---------------
``docs/sr2_gather_finetune.md`` spent four fine-tuning runs and four Rockstar
gates to learn -- in section 8.2, last -- that the window objective never moved
the supervised material at all. Every one of those runs confounded two
questions: *is the objective right* and *can the generator reach it*. This
script separates them by removing the generator.

The optimised variable is the tile field itself: ``candidate = frozen + delta``
with ``delta`` a free ``(T, 6, 64, 64, 64)`` tensor at zero. There is no
network, no convolution and no shared operator, so nothing constrains the answer
except the loss and the guards.

    if the free field reaches HR-like ``bound_frac`` AND survives Rockstar
        -> the objective is a valid specification. What remains is a question
           about the generator's capacity and regularisation.
    if it does not
        -> the loss is still wrong, and no amount of fine-tuning fixes it.

Either way the answer arrives before the expensive line starts.

The free field is a ceiling, not a proposal
-------------------------------------------
Nothing here is a model. ``delta`` has ~6.3M free parameters for four tiles and
sees the true member sets, so it is a per-realisation cheat by construction and
generalises to nothing. Its only job is to bracket what the objective *permits*.
A run that reaches bound halos says the loss admits them; it does not say a
generator can be driven there.

One structural advantage worth reading in the output: the loss's gradient is
non-zero on exactly the member and background rows and exactly zero everywhere
else (pinned in ``tests/features/test_member_gather.py``), and ``delta`` starts
at zero, so Adam leaves every other particle untouched **bit for bit**. Section
3.2's collateral damage -- whole-tile ``peak_contrast`` falling to 0.570 of HR
because a convolutional operator applies everywhere -- is structurally absent
here. That is the cleanest possible reading of the objective, and it is another
reason a negative result would be conclusive.

The output plugs into the existing gate unchanged
-------------------------------------------------
``tiles.npz`` is written in the same ``(tiles, out, frozen, hr)`` layout
``finetune_host_gather.py`` writes, so
``scripts/slurm/submit_gather_rockstar.sh`` splices and gates this run with no
changes -- against the same calibrated ceiling of 227 subhalos in ``R_vir``,
the same +-9 noise floor and the same 42/43 per-target sensitivity
(``docs/sr2_gather_finetune.md`` section 8.1).

    python scripts/features/free_field_gather.py --box set8 --steps 2000
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
from overfit_host_mse import (  # noqa: E402
    BOXSIZE, NG_HR, TILE, _catalog, _owner, build_tiles, field_report,
    forward_tiles, host_tiles,
)

from cosmo_sr.eval.particle_identity import build_owner_index  # noqa: E402
from cosmo_sr.features.member_gather import (  # noqa: E402
    MemberGatherConfig, build_member_sets, member_gather_loss, tile_particles,
)
from cosmo_sr.reward.base import find_base_field  # noqa: E402
from cosmo_sr.train.train_sr2_direct import block_average_torch  # noqa: E402
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def member_config_of(args) -> MemberGatherConfig:
    """Shared by the free field and the fine-tune, so both run one estimator.

    The knobs added on 2026-08-23 are read with ``getattr`` defaults rather than
    as required attributes: they are off in every run recorded so far, and a
    hard attribute read here would make this function the reason the OTHER
    script fails to start.
    """
    return MemberGatherConfig(
        min_num_p=int(args.min_num_p),
        min_purity=float(args.min_purity),
        min_live_frac=float(args.min_live_frac),
        max_sets=int(args.max_sets),
        softening_mpc_h=float(args.softening),
        softening_kind=args.softening_kind,
        pot_chunk=int(args.pot_chunk),
        bound_tau=float(args.bound_tau),
        bound_temperature=args.bound_temperature,
        bound_penalty=str(getattr(args, "bound_penalty", "hinge")),
        bg_k=int(args.bg_k),
        bg_radius_factor=float(args.bg_radius),
        bg_seed=int(args.seed),
        w_virial=float(args.w_virial),
        w_bound=float(args.w_bound),
        w_d6=float(args.w_d6),
        w_rrms=float(args.w_rrms),
        w_sigmav=float(args.w_sigmav),
        w_centre=float(args.w_centre),
        pot_max_elems=int(getattr(args, "pot_max_elems", 1 << 24)),
        max_set_particles=int(getattr(args, "max_set_particles", 0)),
        centre_dead_zone=float(getattr(args, "centre_dead_zone", 0.0)),
        centre_huber_radii=float(getattr(args, "centre_huber_radii", 0.0)),
        centre_mode=str(getattr(args, "centre_mode", "full")),
    )


def frozen_tiles(model, data, train, geom, chunk: int) -> torch.Tensor:
    """The frozen field on every trained tile, forwarded in batches of ``chunk``.

    ``forward_tiles`` stacks the whole tiling into one batch, which is fine at
    ``--n-tiles 4`` and is not the only tiling worth running: the ``R_vir``
    ceiling of 227 is 0.449 of HR only because four tiles hold 42.4% of the
    host's Lagrangian sites, so raising it means raising the tile count, and a
    24-tile batch through the generator is an allocation failure on a 2080 Ti --
    which on this cluster kills the process with no traceback. The result is
    identical either way; only the peak allocation differs.
    """
    outs = []
    with torch.no_grad():
        for i in range(0, len(train), max(1, int(chunk))):
            part = train[i:i + max(1, int(chunk))]
            outs.append(forward_tiles(model, data, part, geom).float())
    return torch.cat(outs, dim=0)


def guards(cand: torch.Tensor, base: torch.Tensor, geom) -> dict:
    """The LR-scale anchor, on **all six** channels.

    ``finetune_host_gather.py`` block-averages the whole tensor too, but the
    reason matters more here: a boundness objective is cheapest to satisfy by
    cooling velocities, and a guard that watched only the displacement channels
    would report a healthy field while the velocity power spectrum collapsed.
    The velocity half is reported separately so a run cannot hide one inside the
    other's mean.
    """
    out = {}
    for name, sl in (("low", slice(0, 6)), ("low_dis", slice(0, 3)),
                     ("low_vel", slice(3, 6))):
        a = block_average_torch(cand[:, sl], geom.scale_factor)
        b = block_average_torch(base[:, sl], geom.scale_factor).detach()
        # The SQUARED relative change, never its square root: at step zero the
        # candidate IS the frozen field, this term is exactly zero, and
        # d(sqrt)/dx there is infinite -- NaN weights within three steps.
        out[name] = (a - b).pow(2).mean() / b.pow(2).mean().clamp_min(1e-30)
    # Velocity dispersion of the whole tile, not just the supervised sets: the
    # global cooling cheat shows up here first and cheapest.
    out["vel_rms_ratio"] = (cand[:, 3:6].std() / base[:, 3:6].std().detach())
    return out


def touched_mask(sets, n_rows: int, device) -> torch.Tensor:
    """Rows the loss can move: members plus their local background.

    Used to mask the gradient, which is not optional. The member loss alone has
    non-zero gradient on exactly these rows (pinned in
    ``tests/features/test_member_gather.py``), but the **guard** does not: the
    LR-scale anchor block-averages over ``scale_factor^3 = 512`` cells, so one
    moved particle puts a gradient on all 512 cells of its block. Adam then
    rescales any non-zero gradient, however tiny, to ~``lr`` per step. The
    2026-08-21 run measured the consequence: 99.56% of the trained tiles moved,
    a median of 0.54 Mpc/h, at a maximum of 30 Mpc/h -- against the 36 Mpc/h
    ceiling of ``lr * steps``, i.e. the untouched particles drifted at close to
    the fastest rate the optimiser permits. Nothing about the resulting field
    was attributable to the objective.
    """
    m = torch.zeros(n_rows, dtype=torch.bool, device=device)
    for r in sets.live_rows:
        m[r] = True
    for b in sets.bg_rows:
        if b is not None:
            m[b] = True
    return m


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    pscfg = phase_space_config_of(cfg)
    mcfg = member_config_of(args)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    banner(f"free-field member gather: {args.box}")

    meta, train, _ = host_tiles(args.box, args)
    if args.tiles:
        # An explicit tiling. `host_tiles` ranks by the HOST's Lagrangian sites,
        # which is the right supervision for the host and not necessarily the
        # tiling that maximises recovered SUBhalos --
        # scripts/features/gather_coverage_curve.py measures both orderings, and
        # this is how a rung it recommends gets trained and gated.
        train = [int(t) for t in str(args.tiles).split(",") if t.strip()]
        n_all = (NG_HR // TILE) ** 3
        bad = [t for t in train if not 0 <= t < n_all]
        if bad or len(set(train)) != len(train):
            raise SystemExit(f"--tiles must be distinct ids in [0,{n_all}): "
                             f"{args.tiles}")
        meta["train_tiles"] = train
        meta["tiles_explicit"] = True
    print(f"  host {meta['halo_id']} logM={meta['log_mvir']:.2f} "
          f"num_p={meta['num_p']} sites={meta['n_member_sites']}")
    sites = ("explicit" if args.tiles
             else f"sites {meta['train_tile_member_sites']}")
    print(f"  train tiles {train} ({sites})", flush=True)

    frozen = load_controlled_generator(
        model_path_of(cfg), in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    for p in frozen.parameters():
        p.requires_grad_(False)

    data = build_tiles(args.box, train, cfg, geom, device, args)
    hr_train = torch.stack([data[t]["hr"] for t in train], dim=0).float()
    base_train = frozen_tiles(frozen, data, train, geom,
                              int(args.forward_chunk))

    # The cached full-box frozen field supplies the members that live outside
    # the trained tiles. They cannot be moved by this run and will not be moved
    # by the splice either, so they belong in the potential as constants --
    # dropping them would compute a fragment's energy and overstate boundness.
    box_path = find_base_field(args.box, seed=int(args.seed))
    if box_path is None and not args.allow_no_box:
        raise SystemExit(
            f"no cached frozen SR2 box for {args.box} seed {args.seed} under "
            f"{paths.SR2_BASE_CACHE()}. Produce it with "
            "scripts/reward/cache_sr2_base.py, or pass --allow-no-box to drop "
            "the out-of-tile members (which overstates bound_frac).")
    frozen_box = np.load(str(box_path), mmap_mode="r") if box_path else None

    kw = dict(ng_hr=NG_HR, tile_hr=TILE, boxsize_mpc_h=BOXSIZE,
              dis_scale_mpc_h=float(scfg.dis_norm_kpc_h) * 1e-3,
              vel_scale_kms=float(pscfg.vel_norm_km_s))
    report: dict = {}
    t0 = time.time()
    hr_cat = _catalog(args.box, "hr")
    # The host's own position, for `centre_mode="radial"`: the projection axis
    # is clustercentric, so the term is undefined without it and the loss says
    # so rather than silently charging zero.
    host_pos = np.asarray(hr_cat.pos[int(meta["row"])], dtype=np.float64)
    sets = build_member_sets(
        hr_cat, build_owner_index(_owner(args.box, "hr")),
        train, hr_train, base_train, mcfg,
        particle_mass_msun_h=float(pscfg.particle_mass_msun_h),
        frozen_box=frozen_box, host_pos=host_pos, report=report, **kw)
    cap = (f", CAPPED to {report['max_sets']} "
           f"(--max-sets dropped {report['n_dropped_by_cap']})"
           if report.get("cap_binds") else "")
    print(f"  sets: {report['n_resolved']} resolved HR subhalos >= "
          f"{mcfg.min_num_p}p, {report['n_home_tile_trained']} homed in these "
          f"tiles, {report['n_after_purity']} past purity{cap}, "
          f"{report['n_sets']} past live fraction "
          f"(median live {report['median_live_frac']:.3f})   "
          f"[{time.time() - t0:.0f}s]")
    if report.get("cap_binds"):
        print(f"  NOTE: --max-sets binds. The tiling covers "
              f"{report['n_after_purity']} supervisable subhalos and the run "
              f"sees {report['n_sets']}; the cap keeps the largest, which are "
              "also the ones the O(N^2) pair sums cost the most.", flush=True)
    rm = report["reference_median"]
    pm = report["pure_hr_median"]
    print(f"  reachable reference (HR in-tile, frozen outside): "
          f"r_rms {rm['r_rms']:.3f} Mpc/h, sigma_v {rm['sigma_v']:.0f} km/s, "
          f"2T/|W| {rm['virial']:.1f}, bound_frac {rm['bound_hard']:.3f}, "
          f"d6 {rm['d6']:.3f}")
    print(f"  pure HR, for contrast:  r_rms {pm['r_rms']:.3f}, "
          f"sigma_v {pm['sigma_v']:.0f}, 2T/|W| {pm['virial']:.1f}, "
          f"bound_frac {pm['bound_hard']:.3f}", flush=True)

    out_dir = paths.subdir("free_field_gather",
                           f"{args.box}_h{meta['halo_id']}{args.label}",
                           create=True)
    n_rows = len(train) * TILE ** 3
    touched = touched_mask(sets, n_rows, device)
    frac_touched = float(touched.float().mean())
    print(f"  movable: {int(touched.sum())} of {n_rows} particles "
          f"({100 * frac_touched:.2f}% of the trained tiles)", flush=True)

    if args.targets_only:
        write_json(out_dir / "targets.json",
                   {"host": meta, "selection": report,
                    "frac_touched": frac_touched,
                    "config": _cfg_dict(mcfg, args)})
        print(f"  targets-only: wrote {out_dir / 'targets.json'}")
        return {"ok": True, "targets_only": True, "selection": report,
                "host": meta, "out_dir": str(out_dir),
                "verdict": {"text": "targets-only shakeout; nothing optimised"}}

    # --- the free field ----------------------------------------------------
    delta = torch.zeros_like(base_train, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=float(args.lr))
    # (B, 1, T, T, T), broadcast over the six channels. `touched` is indexed
    # b * T^3 + (i * T + j) * T + k, the same ordering `tile_particles` flattens
    # to, so this view lines the mask up with the field's own axes.
    grad_mask = touched.view(len(train), TILE, TILE, TILE).unsqueeze(1).to(
        base_train.dtype)
    hist: list = []

    def evaluate(step: int, cand: torch.Tensor) -> dict:
        with torch.no_grad():
            pos, vel = tile_particles(cand, train, **kw)
            _, d = member_gather_loss(pos, vel, sets, mcfg)
            g = {k: float(v) for k, v in guards(cand, base_train, geom).items()}
            rep = field_report(cand, hr_train, base_train, scfg, args)
            dev = (cand - base_train)
            row = {
                "step": step,
                "bound_hard": d["median_bound_hard"],
                "bound_soft": d["median_bound_soft"],
                "virial": d["median_virial"],
                "virial_over_hr": d["median_virial_over_hr"],
                "centre_offset_mpc_h": d["median_centre_offset_mpc_h"],
                "centre_offset_radii": d["median_centre_offset_radii"],
                # Under --centre-mode radial/self these two diverge, and the
                # divergence IS the result: the first is what the loss drove,
                # the second is what Rockstar will be asked about.
                "centre_penalised_radii": d["median_centre_penalised_radii"],
                "frac_centre_within_1_radius": d["frac_centre_within_1_radius"],
                "r_rms_over_hr": d["median_r_rms_over_hr"],
                "sigma_v_over_hr": d["median_sigma_v_over_hr"],
                "low_k": g["low"], "low_k_dis": g["low_dis"],
                "low_k_vel": g["low_vel"], "vel_rms_ratio": g["vel_rms_ratio"],
                "max_abs_delta": float(dev.abs().max()),
                "untouched_max_abs_delta": float(
                    dev.permute(0, 2, 3, 4, 1).reshape(-1, 6)[~touched].abs().max()),
                "highk_power_ratio": rep["out"]["highk_power_ratio"],
                # Whole-tile local peak structure against HR's. Section 3.2's
                # collateral damage lived here: the window run drove it to 0.570
                # while every supervised statistic reported success.
                "peak_contrast_over_hr": rep["structure"]["out"].get(
                    "peak_contrast_s1_over_hr", float("nan")),
                "peak_contrast_frozen_over_hr": rep["structure"]["frozen"].get(
                    "peak_contrast_s1_over_hr", float("nan")),
            }
            for k in ("term_virial", "term_bound", "term_d6", "term_rrms",
                      "term_sigmav", "term_centre"):
                row[k] = d[k]
        return row, d

    row0, diag0 = evaluate(0, base_train)
    hist.append(row0)
    print(f"  frozen: bound_frac {row0['bound_hard']:.3f} "
          f"(reference {rm['bound_hard']:.3f}), 2T/|W| {row0['virial']:.0f}, "
          f"r_rms {row0['r_rms_over_hr']:.2f}x, "
          f"sigma_v {row0['sigma_v_over_hr']:.2f}x of the reference", flush=True)

    metrics_p = out_dir / "metrics.jsonl"
    metrics_p.write_text(json.dumps(row0) + "\n")
    t0 = time.time()
    for step in range(1, int(args.steps) + 1):
        opt.zero_grad(set_to_none=True)
        cand = base_train + delta
        pos, vel = tile_particles(cand, train, **kw)
        gather, _ = member_gather_loss(pos, vel, sets, mcfg)
        g = guards(cand, base_train, geom)
        loss = gather + float(args.w_low) * g["low"]
        loss.backward()
        if args.mask_grad:
            delta.grad.mul_(grad_mask)
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_([delta], float(args.clip))
        opt.step()

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            with torch.no_grad():
                row, _ = evaluate(step, base_train + delta)
            row["loss"] = float(loss.detach())
            row["wall_s"] = time.time() - t0
            hist.append(row)
            with metrics_p.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  step {step:6d}  loss {row['loss']:.4f}  "
                  f"bound {row['bound_hard']:.3f}  2T/|W| {row['virial']:.1f}  "
                  f"r_rms {row['r_rms_over_hr']:.2f}x  "
                  f"sig_v {row['sigma_v_over_hr']:.2f}x  "
                  f"dx {row['centre_offset_radii']:.2f}r  "
                  f"low_k {row['low_k']:.4f}  "
                  f"[{row['wall_s']:.0f}s]", flush=True)

    final = (base_train + delta).detach()
    last, last_diag = evaluate(int(args.steps), final)

    np.savez_compressed(out_dir / "tiles.npz", tiles=np.array(train),
                        out=final.cpu().numpy(),
                        frozen=base_train.cpu().numpy(),
                        hr=hr_train.cpu().numpy())
    write_json(out_dir / "subhalos.json",
               {"rows": last_diag["rows"],
                "reference": {k: v.detach().cpu().numpy().tolist()
                              for k, v in sets.ref.items()},
                "halo_id": sets.halo_id.tolist(),
                "num_p": sets.num_p.tolist(),
                "n_live": sets.n_live.tolist()})

    v = verdict(row0, last, rm, args)
    summary = {
        "ok": True, "box": args.box, "host": meta, "tiles": train,
        "selection": report, "config": _cfg_dict(mcfg, args),
        "frac_touched": frac_touched, "frozen": row0, "final": last,
        "history": hist, "verdict": v, "out_dir": str(out_dir),
        "frozen_box": str(box_path) if box_path else None,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"\n  wrote {out_dir}")
    return summary


def _cfg_dict(mcfg: MemberGatherConfig, args) -> dict:
    d = {k: getattr(mcfg, k) for k in mcfg.__dataclass_fields__}
    d.update({"lr": float(args.lr), "steps": int(args.steps),
              "w_low": float(args.w_low), "clip": float(args.clip),
              "mask_grad": bool(args.mask_grad)})
    return d


def verdict(first: dict, last: dict, ref: dict, args) -> dict:
    """Did the free field reach bound objects, and at what cost?

    Three readings, and the middle one is the point. ``reached`` is the
    feasibility answer; the guards say whether it was bought by wrecking the
    field, which would make it uninformative. Nothing here is a substitute for
    the Rockstar gate -- ``bound_frac`` is this module's own statistic, and
    ``docs/sr2_gather_finetune.md`` section 6 is the standing reason not to
    believe a differentiable surrogate about its own success.
    """
    target = float(ref["bound_hard"]) * float(args.gain_frac)
    reached = bool(last["bound_hard"] >= target and target > 0)
    if int(args.steps) == 0:
        # `--steps 0` is the ceiling probe: it writes tiles.npz from the frozen
        # forward so the splice can gate the TRUE HR tiles at this tiling
        # (`FF_WHICH=hr`). Nothing was optimised, so "not feasible" would be a
        # statement about a run that never happened.
        return {"reached": False, "guards_held": True, "target": target,
                "text": ("CEILING PROBE: --steps 0, nothing optimised. "
                         f"tiles.npz holds the frozen forward and HR for the "
                         f"{int(args.n_tiles)} trained tiles; gate it with "
                         "FF_WHICH=hr to measure what this tiling could reach. "
                         "No feasibility claim is made here.")}
    guards_held = bool(last["low_k"] <= float(args.low_k_max)
                       and last["untouched_max_abs_delta"] == 0.0
                       and abs(last["vel_rms_ratio"] - 1.0)
                       <= float(args.vel_rms_tol))
    if reached and guards_held:
        text = (f"FEASIBLE: bound_frac {first['bound_hard']:.3f} -> "
                f"{last['bound_hard']:.3f} against a reachable reference of "
                f"{ref['bound_hard']:.3f}, guards held. The objective admits "
                f"bound sets. Gate it on Rockstar before believing it.")
    elif reached:
        text = (f"FEASIBLE BUT UNGUARDED: bound_frac reached "
                f"{last['bound_hard']:.3f} of {ref['bound_hard']:.3f} while a "
                f"guard failed (low_k {last['low_k']:.4f}, vel_rms "
                f"{last['vel_rms_ratio']:.3f}, untouched drift "
                f"{last['untouched_max_abs_delta']:.2e}). Not yet an answer.")
    else:
        text = (f"NOT FEASIBLE at this budget: bound_frac "
                f"{first['bound_hard']:.3f} -> {last['bound_hard']:.3f} against "
                f"{ref['bound_hard']:.3f}, needing {target:.3f}. A free field "
                f"with {int(args.steps)} steps could not satisfy the objective, "
                f"so the objective -- not the generator -- is what to change.")
    return {"reached": reached, "guards_held": guards_held, "target": target,
            "text": text}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_direct_args(ap)
    ap.add_argument("--box", default="set8")
    ap.add_argument("--host-id", type=int, default=-1)
    ap.add_argument("--n-candidate-hosts", type=int, default=8)
    ap.add_argument("--n-tiles", type=int, default=4,
                    help="tiles to train, ranked by the host's Lagrangian "
                         "sites. This is the knob that moves the R_vir "
                         "CEILING: 4 tiles hold 42.4%% of the host's sites and "
                         "the measured ceiling is 227 of HR's 506. See "
                         "scripts/features/gather_coverage_curve.py")
    ap.add_argument("--tiles", default="",
                    help="comma-separated tile ids, overriding the host-site "
                         "ranking --n-tiles takes. The ranking maximises the "
                         "HOST's covered material; gather_coverage_curve.py "
                         "also reports the ordering that maximises R_vir "
                         "SUBhalo material, and this is how to train it")
    ap.add_argument("--forward-chunk", type=int, default=4,
                    help="tiles per generator forward pass when building the "
                         "frozen field; only the peak allocation depends on it")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3,
                    help="Adam step on the NORMALISED field; 1.0 is 6 Mpc/h of "
                         "displacement, so 3e-3 is ~18 kpc/h a step")
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=100)
    # --- the objective -----------------------------------------------------
    ap.add_argument("--w-virial", type=float, default=1.0)
    ap.add_argument("--w-bound", type=float, default=1.0)
    ap.add_argument("--w-d6", type=float, default=1.0)
    ap.add_argument("--w-rrms", type=float, default=0.3)
    ap.add_argument("--w-sigmav", type=float, default=0.3)
    ap.add_argument("--w-centre", type=float, default=1.0,
                    help="pin each set's centre of mass. The 2026-08-21 run had "
                         "no such term and built bound objects a median 0.414 "
                         "Mpc/h from their targets against a 0.150 Mpc/h search "
                         "radius: 8/154 recovered, 96%% of misses with no halo "
                         "of any mass inside the radius")
    # --- what the centre term charges for -----------------------------------
    # All three default to the 72/154 objective. They exist because
    # `centre_offset/pool/offsets.json` measured that the address the `full`
    # term asks for is 62.9% radial by variance but only 11.5% predictable out
    # of sample from the features a generator can condition on.
    ap.add_argument("--bound-penalty", choices=("hinge", "log"), default="hinge",
                    help="hinge is [1-x/ref]_+^2, capped at 1 by construction; "
                         "log is [log(ref/x)]_+^2, unbounded as x -> 0 and "
                         "still exactly zero for meeting or beating HR.")
    ap.add_argument("--centre-mode", choices=("full", "radial", "self"),
                    default="full",
                    help="full: the whole offset to the reachable HR centroid "
                         "(the 72/154 term). radial: only its projection on the "
                         "clustercentric direction, dropping the transverse "
                         "part no input determines. self: re-anchor to the "
                         "set's own frozen centroid -- zero at step 0, and no "
                         "address at all.")
    ap.add_argument("--centre-dead-zone", type=float, default=0.0,
                    help="radii inside which the centre term costs nothing. "
                         "The gate is a threshold at ONE search radius, so "
                         "cost paid at 0.1 radii buys nothing there.")
    ap.add_argument("--centre-huber-radii", type=float, default=0.0,
                    help="radii beyond which the centre term is linear rather "
                         "than quadratic. At the measured median 5.6 radii a "
                         "quadratic charges 31 with gradient 11; linear there "
                         "is a ~10x cut that hands the budget to the five "
                         "concentration terms.")
    ap.add_argument("--w-low", type=float, default=100.0)
    # --- selection ---------------------------------------------------------
    ap.add_argument("--min-num-p", type=int, default=200)
    ap.add_argument("--min-purity", type=float, default=0.5)
    ap.add_argument("--min-live-frac", type=float, default=0.5)
    ap.add_argument("--max-sets", type=int, default=256)
    # --- estimator ---------------------------------------------------------
    ap.add_argument("--softening", type=float, default=0.01)
    ap.add_argument("--softening-kind", default="plummer",
                    choices=("plummer", "clamp"))
    ap.add_argument("--pot-chunk", type=int, default=2048)
    ap.add_argument("--bound-tau", type=float, default=0.5)
    ap.add_argument("--bound-temperature", default="adaptive",
                    choices=("adaptive", "hr"))
    ap.add_argument("--bg-k", type=int, default=4096)
    ap.add_argument("--bg-radius", type=float, default=4.0)
    # --- reporting and verdict ---------------------------------------------
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--k-split", type=float, default=4.0)
    ap.add_argument("--gain-frac", type=float, default=0.5,
                    help="share of the reachable reference bound_frac the "
                         "verdict calls feasible")
    ap.add_argument("--low-k-max", type=float, default=0.02)
    ap.add_argument("--vel-rms-tol", type=float, default=0.10,
                    help="tolerated change in the tile's whole velocity rms; "
                         "the global-cooling cheat shows up here")
    ap.add_argument("--mask-grad", dest="mask_grad", action="store_true",
                    default=True,
                    help="zero the gradient outside members and background, so "
                         "the guard's block-averaged gradient cannot let Adam "
                         "rewrite the whole tile (default on)")
    ap.add_argument("--no-mask-grad", dest="mask_grad", action="store_false")
    ap.add_argument("--allow-no-box", action="store_true")
    ap.add_argument("--targets-only", action="store_true")
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
