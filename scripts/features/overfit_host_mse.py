#!/usr/bin/env python
"""Capacity or incentive? Fine-tune SR2's last block onto ONE cluster under MSE.

``docs/sr2_substructure_module.md`` section 9 step 4. The subhalo deficit has two
candidate explanations that no diagnostic so far separates:

* **capacity** -- the frozen generator cannot represent HR's substructure at
  all, in which case an additive module is the wrong fix and more network is
  the right one;
* **incentive** -- it can, and the training objective never asked it to, which
  is section 6.1's claim: L2 converges to ``E[HR | SR2]``, which is empty in the
  fine modes, so the minimiser blurs.

The experiment separates them by removing the incentive problem in the crudest
possible way: take a *single* 1e14 host's Lagrangian region, unfreeze one rung,
and minimise plain MSE against that region's true HR field until it stops
improving. There is no generalisation claim here and no held-out objective --
overfitting is the *point*. A few tiles and a few thousand steps is far more
supervision per parameter than the real training ever had.

Reading the result, and what it cannot say
------------------------------------------
Three numbers, and the third decides how to read the first two.

``mse_ratio``
    Final MSE over the frozen generator's on the same tiles.
``highk_power_ratio``
    Power of the output displacement above ``--k-split``, over HR's.
``params_per_target``
    Trainable parameters divided by target values. **This is the one that
    licenses the reading.** At the deployed rung and four tiles it is ~0.05:
    the network cannot memorise the region, so it has to fit a shared local
    function across every subhalo-scale patch inside it -- and a squared loss
    over many patches whose fine realisation it cannot predict is minimised by
    averaging. A blurred result at that ratio is therefore consistent with both
    "the rung cannot express substructure" and "L2 declined to commit", which
    are the two things this step exists to separate. Only an over-parameterised
    run separates them, so the ratio gates the verdict's wording.

The capacity-decisive setting is one tile with the whole generator unfrozen
(``--n-tiles 1 --rung all_blocks``, ratio 4.4): a plateau *there* is a capacity
statement, and a fit that recovers high-k power *there* proves the function
class can express this cluster's substructure.

What this run cannot do, at any ratio, is tell you whether a **regressor** would
work -- whether ``E[HR | SR2]`` is full or empty in the fine modes. That is a
property of the conditional distribution, not of one optimisation, and it is
measured in closed form and on held-out data by
``scripts/features/measure_conditional_spread.py``. The two steps are
complementary and neither substitutes for the other: this one asks whether the
network *could*, that one asks whether there is anything to learn.

Rockstar is not run here. This measures the field; whether the resulting clumps
are *bound* is section 9 step 6 and needs the whole box.

    python scripts/features/overfit_host_mse.py --box set8 --steps 3000
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, paths, write_json  # noqa: E402
from _sr2_direct import (  # noqa: E402
    add_direct_args, geometry_of, load_direct_config, load_hr, load_lr,
    model_path_of, soft_config_of,
)

from cosmo_sr.data.preprocess_srs import disnorm  # noqa: E402
from cosmo_sr.eval.particle_identity import (  # noqa: E402
    build_owner_index, child_map,
)
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.features import (  # noqa: E402
    flat_to_sites, hann_window, radial_cross_spectra,
)
from cosmo_sr.reward.soft_structure import (  # noqa: E402
    density_from_disp, feature_names, soft_structure_features,
)
from cosmo_sr.train.sr2_finetune_data import (  # noqa: E402
    tile_lr_crop, tile_noise_stack, trim_to_tile,
)
from cosmo_sr.train.sr2_unfreeze import (  # noqa: E402
    assert_only_trainable_changed, parameter_groups, snapshot_parameters,
    trainable_names,
)
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402

BOXSIZE = 100.0
NG_HR = 512
TILE = 64
DX = BOXSIZE / NG_HR


# --------------------------------------------------------------------------- #
# Which cluster, which tiles
# --------------------------------------------------------------------------- #
def _catalog(box: str, tag: str = "hr"):
    root = paths.reward_root()
    hits = sorted(glob.glob(str(root / "halos" / f"{box}__{tag}__{tag}"
                                 / "*_rockstar" / "halos_*.ascii")))
    if not hits:
        raise SystemExit(f"no {tag} catalog for {box} under {root/'halos'}")
    return load_rockstar_ascii(hits[0])


def _owner(box: str, tag: str = "hr") -> np.ndarray:
    p = (paths.reward_root() / "halos_particles" / f"{box}__{tag}__{tag}"
         / f"{box}_{tag}_owner.npy")
    if not p.is_file():
        raise SystemExit(f"no owner array at {p}; "
                         "run scripts/reward/rockstar_particles.py")
    return np.load(p)


def host_tiles(box: str, args):
    """``(meta, train_tiles, holdout_tile)`` for the chosen cluster.

    Tiles are ranked by how many of the host's Lagrangian sites they hold. A
    1e14 host has ``R_L = 6.74`` Mpc/h against a 12.5 Mpc/h tile, so its
    footprint straddles several -- taking the top few keeps the supervision on
    material that is actually the host's rather than on its neighbours.
    """
    cat = _catalog(box)
    oidx = build_owner_index(_owner(box))
    children = child_map(cat)
    hosts = np.flatnonzero(cat.parent_ids < 0)
    order = hosts[np.argsort(-cat.mvir[hosts])]
    if args.host_id >= 0:
        rows = np.flatnonzero(cat.ids == args.host_id)
        if rows.size == 0:
            raise SystemExit(f"host id {args.host_id} not in the {box} HR catalog")
        order = np.concatenate([rows, order[order != rows[0]]])

    picked = []
    for row in order[: args.n_candidate_hosts]:
        hid = int(cat.ids[row])
        members = oidx.members_with_substructure(cat, hid, children=children)
        if members.size == 0:
            continue
        sites = flat_to_sites(members, NG_HR) // TILE
        n = NG_HR // TILE
        tid = (sites[:, 0] * n + sites[:, 1]) * n + sites[:, 2]
        counts = np.bincount(tid, minlength=n ** 3)
        picked.append({
            "row": int(row), "halo_id": hid,
            "log_mvir": float(np.log10(max(cat.mvir[row], 1.0))),
            "num_p": int(cat.num_p[row]),
            "n_member_sites": int(members.size),
            "counts": counts,
        })
    if not picked:
        raise SystemExit(f"{box}: no host with member sites in the owner array")

    top = picked[0]
    train = np.argsort(-top["counts"])[: args.n_tiles].astype(int).tolist()
    # A held-out tile from a DIFFERENT cluster: it says whether the rung moved
    # a shared function or memorised one region. Not a success criterion --
    # overfitting one host is the experiment -- but a run that improved
    # everywhere would mean something quite different from one that did not.
    hold = -1
    for other in picked[1:]:
        cand = [int(t) for t in np.argsort(-other["counts"])[:4]
                if int(t) not in train and other["counts"][int(t)] > 0]
        if cand:
            hold = cand[0]
            break
    meta = {k: v for k, v in top.items() if k != "counts"}
    meta["train_tiles"] = train
    meta["train_tile_member_sites"] = [int(top["counts"][t]) for t in train]
    meta["holdout_tile"] = int(hold)
    meta["n_hosts_scanned"] = len(picked)
    return meta, train, int(hold)


# --------------------------------------------------------------------------- #
# Tile data
# --------------------------------------------------------------------------- #
def tile_hr_target(hr_field, tile_id: int, geom) -> np.ndarray:
    n = NG_HR // TILE
    ix, iy, iz = tile_id // (n * n), (tile_id // n) % n, tile_id % n
    s = TILE
    return np.asarray(hr_field[:, ix * s:(ix + 1) * s, iy * s:(iy + 1) * s,
                               iz * s:(iz + 1) * s], dtype=np.float32)


def build_tiles(box, tiles, cfg, geom, device, args):
    lr = load_lr(cfg, box)
    hr = load_hr(cfg, box)
    out = {}
    for t in tiles:
        if t < 0:
            continue
        out[t] = {
            "lr": torch.from_numpy(np.ascontiguousarray(
                tile_lr_crop(np.asarray(lr), t, geom))).float().to(device),
            "noise": tile_noise_stack([args.seed], t, geom, device=device),
            "hr": torch.from_numpy(tile_hr_target(hr, t, geom)).to(device),
        }
    return out


def forward_tiles(model, data, tiles, geom):
    lr = torch.stack([data[t]["lr"] for t in tiles], dim=0)
    sites = list(data[tiles[0]]["noise"].keys())
    noise = {s: torch.cat([data[t]["noise"][s] for t in tiles], dim=0)
             for s in sites}
    return trim_to_tile(model(lr, noise=noise), geom)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def to_mpc(x: torch.Tensor) -> np.ndarray:
    """``(B, 3, N, N, N)`` displacement channels, on-disk units -> Mpc/h."""
    d = x[:, 0:3].detach().float().cpu().numpy().astype(np.float64)
    return (disnorm(d, z=0.0, undo=True) * 1e-3).astype(np.float32)


def field_report(out: torch.Tensor, hr: torch.Tensor, frozen: torch.Tensor,
                 scfg, args) -> dict:
    """Everything measurable about one batch of tiles, without a halo finder."""
    o, h, f = to_mpc(out), to_mpc(hr), to_mpc(frozen)
    # A tile is a sub-cube of a periodic box and is NOT itself periodic, so its
    # FFT leaks the shared bulk flow across every k. Both fields carry the same
    # leak, which drags any power RATIO toward 1 and hides exactly the deficit
    # this run is trying to see. The first ladder was run without this window
    # and its high-k numbers are biased toward 1.
    win = hann_window(int(o.shape[-1]))
    rows = {"out": [], "frozen": []}
    for b in range(o.shape[0]):
        rows["out"].append(radial_cross_spectra(h[b], o[b], DX, n_bins=args.n_bins,
                                                window=win))
        rows["frozen"].append(radial_cross_spectra(h[b], f[b], DX, n_bins=args.n_bins,
                                                   window=win))

    def mean_of(key, which):
        return np.nanmean(np.stack([r[key] for r in rows[which]]), axis=0)

    k = rows["out"][0]["k"]
    counts = rows["out"][0]["counts"]
    hi = np.isfinite(k) & (k >= args.k_split) & (counts > 0)
    rep = {"k": k.tolist(), "counts": counts.tolist()}
    for which in ("out", "frozen"):
        p_hr = mean_of("P_a", which)
        p_x = mean_of("P_b", which)
        r = mean_of("P_cross", which) / np.sqrt(np.maximum(p_hr * p_x, 1e-300))
        rep[which] = {
            "P": p_x.tolist(),
            "r": r.tolist(),
            "P_diff": mean_of("P_diff", which).tolist(),
            "highk_power_ratio": float(
                np.sum((p_x * counts)[hi]) / max(np.sum((p_hr * counts)[hi]), 1e-300)),
        }
        if which == "out":
            rep["P_hr"] = p_hr.tolist()

    # Density: the nonlinear functional an MSE on displacement does not control.
    with torch.no_grad():
        dens = {name: density_from_disp(t[:, 0:3].float(), scfg)
                for name, t in (("out", out), ("hr", hr), ("frozen", frozen))}
    rep["density"] = {
        name: {
            "log1p_var": float(torch.log1p(d.clamp_min(0)).var().item()),
            "frac_above_delta100": float((d > 100.0).float().mean().item()),
            "max_delta": float(d.max().item()),
        } for name, d in dens.items()
    }

    # Structure, not power. The first run of this experiment showed frozen SR2
    # already at 0.87 of HR's high-k displacement power and slightly ABOVE HR's
    # fraction of cells over delta=100, while section 6.1's table puts its
    # cluster interiors at ~7% of HR's clump count. A power ratio and a
    # one-point threshold are therefore both nearly blind to the deficit: the
    # missing thing is coherence, not amplitude. `peak_contrast_s*` is the
    # cheapest available statistic that asks the right question -- mass in cells
    # that beat their OWN smoothed neighbourhood by a margin, which a smooth
    # puffy cluster cannot fake by having the right spectrum.
    with torch.no_grad():
        names = feature_names(scfg)
        feats = {name: soft_structure_features(d, scfg).mean(dim=0)
                 for name, d in dens.items()}
    rep["structure"] = {
        name: {n: float(v[j].item()) for j, n in enumerate(names)
               if n.startswith(("peak_", "compact_", "envelope_", "second_"))}
        for name, v in feats.items()
    }
    ref = rep["structure"]["hr"]
    for name in ("out", "frozen"):
        rep["structure"][name] = dict(
            rep["structure"][name],
            **{f"{n}_over_hr": (rep["structure"][name][n] / ref[n]
                                if abs(ref[n]) > 1e-12 else float("nan"))
               for n in ref})
    return rep


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    banner(f"overfit one host under MSE: {args.box}, rung={args.rung}")

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
    n_target = len(train) * 6 * TILE ** 3
    params_per_target = n_train_params / max(n_target, 1)
    print(f"  rung {args.rung}: {len(names)} tensors, {n_train_params} parameters "
          f"trainable of {sum(p.numel() for p in model.parameters())}")
    print(f"  {n_target} target values over {len(train)} tiles -> params/target = "
          f"{params_per_target:.3f}"
          + ("  (over-parameterised: a plateau here IS a capacity statement)"
             if params_per_target >= args.memorise_ratio else
             "  (UNDER-parameterised: the rung must fit a shared local function, "
             "so blurring here is not by itself a capacity result)"), flush=True)

    data = build_tiles(args.box, train + ([hold] if hold >= 0 else []), cfg, geom,
                       device, args)
    hr_train = torch.stack([data[t]["hr"] for t in train], dim=0)
    with torch.no_grad():
        base_train = forward_tiles(frozen, data, train, geom)
        mse_frozen = float(torch.mean((base_train - hr_train) ** 2).item())
    print(f"  frozen MSE on the training tiles: {mse_frozen:.6e}", flush=True)

    out_dir = paths.subdir("host_overfit", f"{args.box}_h{meta['halo_id']}"
                           f"_{args.rung}{args.label}", create=True)
    jsonl = out_dir / "metrics.jsonl"
    jsonl.write_text("")

    opt = torch.optim.Adam(groups)
    rng = np.random.default_rng(args.seed)
    t0, history, last_report = time.time(), [], None
    for step in range(1, args.steps + 1):
        pick = [train[i] for i in rng.choice(len(train),
                                             size=min(args.batch, len(train)),
                                             replace=False)]
        target = torch.stack([data[t]["hr"] for t in pick], dim=0)
        out = forward_tiles(model, data, pick, geom)
        loss = torch.mean((out - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], args.clip)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                cur = forward_tiles(model, data, train, geom)
                mse = float(torch.mean((cur - hr_train) ** 2).item())
                rep = field_report(cur, hr_train, base_train, scfg, args)
                row = {"step": step, "train_loss": float(loss.item()),
                       "mse_train": mse, "mse_ratio": mse / max(mse_frozen, 1e-30),
                       "highk_power_ratio": rep["out"]["highk_power_ratio"],
                       "highk_power_ratio_frozen": rep["frozen"]["highk_power_ratio"],
                       "frac_above_delta100": rep["density"]["out"]["frac_above_delta100"],
                       "frac_above_delta100_hr": rep["density"]["hr"]["frac_above_delta100"],
                       "frac_above_delta100_frozen": rep["density"]["frozen"]["frac_above_delta100"],
                       "peak_contrast_s1_over_hr":
                           rep["structure"]["out"]["peak_contrast_s1_over_hr"],
                       "peak_contrast_s1_over_hr_frozen":
                           rep["structure"]["frozen"]["peak_contrast_s1_over_hr"],
                       "seconds": round(time.time() - t0, 1)}
                if hold >= 0:
                    hout = forward_tiles(model, data, [hold], geom)
                    hbase = forward_tiles(frozen, data, [hold], geom)
                    htgt = data[hold]["hr"].unsqueeze(0)
                    row["mse_holdout"] = float(torch.mean((hout - htgt) ** 2).item())
                    row["mse_holdout_frozen"] = float(
                        torch.mean((hbase - htgt) ** 2).item())
            model.train()
            history.append(row)
            last_report = rep
            with jsonl.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"  step {step:6d}  mse {mse:.4e} ({row['mse_ratio']:.3f} x frozen)"
                  f"  highk P/P_HR {row['highk_power_ratio']:.3f}"
                  f" (frozen {row['highk_power_ratio_frozen']:.3f})"
                  f"  peak_contrast/HR {row['peak_contrast_s1_over_hr']:.3f}"
                  f" (frozen {row['peak_contrast_s1_over_hr_frozen']:.3f})",
                  flush=True)

    with torch.no_grad():
        final = forward_tiles(model, data, train, geom)
    np.savez_compressed(out_dir / "tiles.npz",
                        tiles=np.array(train), out=final.cpu().numpy(),
                        frozen=base_train.cpu().numpy(), hr=hr_train.cpu().numpy())
    torch.save({"rung": args.rung, "names": names,
                "state": {n: p.detach().cpu() for n, p in model.named_parameters()
                          if n in set(names)}}, out_dir / "trainable.pt")

    # The rung is the whole safety story of this experiment, so it is audited
    # rather than assumed: anything outside it that moved would mean the result
    # is about a different model than the one described.
    assert_only_trainable_changed(model, theta0, names)
    drift = {n: float(torch.linalg.vector_norm(
        dict(model.named_parameters())[n].detach().cpu() - theta0[n]).item())
        for n in names}
    summary = {
        "ok": True,
        "box": args.box,
        "host": meta,
        "rung": args.rung,
        "trainable_tensors": names,
        "trainable_parameters": int(n_train_params),
        "steps": int(args.steps),
        "batch": int(args.batch),
        "lr_scale": float(args.lr_scale),
        "k_split": float(args.k_split),
        "mse_frozen": mse_frozen,
        "history": history,
        "final_report": last_report,
        "param_drift_l2": drift,
        "n_target_values": int(n_target),
        "params_per_target": float(params_per_target),
        "verdict": verdict(history, mse_frozen, args,
                           params_per_target=params_per_target),
        "device": str(device),
        "seconds": round(time.time() - t0, 1),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"  wrote {out_dir / 'summary.json'}")
    return summary


def verdict(history, mse_frozen: float, args, *,
            params_per_target: float = float("nan")) -> dict:
    """Capacity and incentive read off separately -- and only when separable.

    The reading depends on one ratio that is easy to skip past: trainable
    parameters over target values. Below ~1 the rung cannot memorise the region
    and has to fit a shared local function across every subhalo-scale patch in
    it, so L2's averaging applies *inside* this experiment and a blurred result
    is consistent with both "cannot express it" and "will not commit to a
    realisation". Only an over-parameterised fit separates them, which is why
    the ratio gates the wording rather than decorating it.

    Nothing here bears on whether a *regressor* would work across examples. That
    is the conditional mean, it is measured in closed form by
    ``scripts/features/measure_conditional_spread.py``, and this run cannot
    substitute for it in either direction.
    """
    if not history:
        return {"text": "no eval steps ran"}
    last = history[-1]
    ratio = float(last["mse_ratio"])
    hk, hk0 = float(last["highk_power_ratio"]), float(last["highk_power_ratio_frozen"])
    fitted = ratio <= args.capacity_max
    recovered = hk >= args.highk_recovered
    memorisable = float(params_per_target) >= args.memorise_ratio

    ppt = (f"params/target = {params_per_target:.3f}"
           if np.isfinite(params_per_target) else "params/target unknown")
    if fitted and recovered:
        text = (f"CAPACITY IS NOT THE LIMIT ({ppt}): MSE fell to {ratio:.3f} of "
                f"frozen and the high-k power reached {hk:.3f} of HR's (frozen "
                f"{hk0:.3f}). Rung '{args.rung}' can represent this cluster's "
                "substructure. This says nothing about whether a regressor "
                "trained across examples would -- that is the conditional mean, "
                "and it is measured by measure_conditional_spread.py.")
    elif fitted and not recovered:
        if memorisable:
            text = (f"BLURRED WITH ROOM TO SPARE ({ppt}): MSE fell to "
                    f"{ratio:.3f} of frozen while the high-k power went "
                    f"{hk0:.3f} -> {hk:.3f} of HR's, and the rung had enough "
                    "parameters to memorise the region outright. Lowering the "
                    "squared error by averaging was the *preferred* solution, "
                    "not the only reachable one. That is section 6.1 as a "
                    "measurement.")
        else:
            text = (f"AMBIGUOUS ({ppt} < {args.memorise_ratio}): MSE fell to "
                    f"{ratio:.3f} of frozen but the high-k power went "
                    f"{hk0:.3f} -> {hk:.3f} of HR's. Under-parameterised, so "
                    "this is equally consistent with 'the rung cannot express "
                    "the substructure' and with 'L2 averaged over a realisation "
                    "it could not predict'. Rerun over-parameterised "
                    "(OH_N_TILES=1 OH_RUNG=all_blocks) before reading it "
                    "either way.")
    else:
        if memorisable:
            text = (f"CAPACITY IS THE LIMIT ({ppt}): {last['step']} steps with "
                    "enough parameters to memorise the region left MSE at "
                    f"{ratio:.3f} of frozen. Check the learning rate first -- an "
                    "under-trained run looks identical -- but if it holds, an "
                    "additive module is the wrong fix and more network is the "
                    "right one.")
        else:
            text = (f"INCONCLUSIVE ({ppt} < {args.memorise_ratio}): MSE only "
                    f"reached {ratio:.3f} of frozen after {last['step']} steps "
                    "on a rung that could not have memorised the region anyway. "
                    "Raise the rung or drop to one tile before concluding "
                    "anything about capacity.")
    return {"fitted": bool(fitted), "highk_recovered": bool(recovered),
            "memorisable": bool(memorisable),
            "params_per_target": float(params_per_target),
            "mse_ratio": ratio, "highk_power_ratio": hk,
            "highk_power_ratio_frozen": hk0, "text": text}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--box", default="set8")
    ap.add_argument("--host-id", type=int, default=-1,
                    help="HR catalog id; -1 takes the most massive host")
    ap.add_argument("--n-candidate-hosts", type=int, default=4)
    ap.add_argument("--n-tiles", type=int, default=4,
                    help="tiles holding the most of the host's Lagrangian sites")
    ap.add_argument("--rung", default="fine",
                    help="proj_noise | fine | middle_fine | all_blocks | full")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr-scale", type=float, default=10.0,
                    help="multiplier on the default per-group rates; the direct "
                         "line's defaults are sized for a reward surrogate, and "
                         "this is a deliberate overfit")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-bins", type=int, default=16)
    ap.add_argument("--k-split", type=float, default=4.0,
                    help="h/Mpc above which power is called substructure")
    ap.add_argument("--capacity-max", type=float, default=0.8)
    ap.add_argument("--memorise-ratio", type=float, default=1.0,
                    help="trainable parameters per target value above which the "
                         "rung could memorise the region outright. Below it, a "
                         "blurred result cannot be attributed to capacity")
    ap.add_argument("--highk-recovered", type=float, default=0.8)
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
