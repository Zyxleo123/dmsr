#!/usr/bin/env python
"""Is the centre term a physical RULE or a per-object ADDRESS?

``docs/sr2_member_gather.md`` section 5.1 measured that the member-gather loss
says nothing about where an object goes unless ``centre`` is in it, and section
6 measured what adding it was worth: 8/154 -> 72/154. So ``centre`` is load
bearing. It is also the one term that is not an invariant statistic -- every
other term is a moment about the set's OWN centroid, and a shared convolutional
operator can learn a moment because a moment is a rule. An address is not a
rule, and the pool asks for 5,396 of them from 335,954 shared parameters.

*Measured*, the step-0 row of the 2026-08-23 pool: frozen SR2's supervised sets
sit a median **5.59 search radii** (train) and **5.90** (held out) from the
centroid they must reach. That is ~1 Mpc/h of position per object. This script
asks how much of it is predictable, and it is the measurement that decides
whether the centre term needs softening or merely needs training.

The decomposition
-----------------
For each supervised set, with ``xbar_frozen`` its centroid under the frozen
generator and ``xbar_ref`` the reachable centroid the loss targets::

    o        = xbar_frozen - xbar_ref                  the offset to be closed
    rhat     = unit(xbar_ref - x_host)                 clustercentric direction
    o_par    = o . rhat                                infall deficit, signed
    o_perp   = |o - o_par rhat|                        the part with no direction

An **infall deficit** -- material that never fell far enough into the host's
potential -- is radial, systematic, and a function of the environment, so it is
learnable and a conv can express it. An isotropic scatter is realisation noise
that LR does not contain, and no architecture fits it. The isotropic null for
``sum o_par^2 / sum |o|^2`` is exactly **1/3**; distance from 1/3 is the signal.

Then three candidate RULES are fitted, each using only quantities available at
inference (clustercentric distance, host mass, set size -- never the answer):

===============  =========================================================
``none``         the frozen field as it stands. The baseline.
``radial``       one scalar over the whole pool: ``o ~ a rhat``.
``regressed``    ``o_par ~ a + b d_host + c log10 num_p + d logM_host``,
                 and the perpendicular part left unmodelled, because it has
                 no direction to predict from.
===============  =========================================================

**Fitted on the training hosts and reported on the held-out ones**, which is the
same split the fine-tune uses, so "a rule that generalises across clusters" means
here what it means there.

What to read
------------
The last column of the rule table: the share of sets that would land inside ONE
search radius -- ``compare_gather_catalog``'s own hit criterion -- if the
generator learned that rule perfectly and nothing else. Against 72/154 (the free
field, which saw every address) and 151/154 (the geometric ceiling):

* a rule that gets most sets inside 1 radius means ``centre`` is learnable and
  the fine-tune's job is to learn it;
* a rule that barely moves the number means ``centre`` is an address, the
  quadratic is charging the generator for information its input does not carry,
  and the softened / self-consistent forms are not optional.

No halo finder and no optimisation: one frozen forward per host, the same pool
the trainer builds, and a least-squares fit.

    python scripts/features/centre_offset_decompose.py --label _v1
    python scripts/features/centre_offset_decompose.py --from-json <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

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
    BOXSIZE, NG_HR, TILE, _catalog, forward_tiles,
)
from free_field_gather import member_config_of  # noqa: E402
from finetune_member_gather import build_pool  # noqa: E402

from cosmo_sr.features.member_gather import (  # noqa: E402
    _gather_one, tile_particles, unwrap_about,
)
from cosmo_sr.features.member_pool import split_pool  # noqa: E402
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402

RULES = ("none", "radial", "regressed")


# --------------------------------------------------------------------------- #
# The offsets
# --------------------------------------------------------------------------- #
def host_offsets(task, model, geom, kw, host_pos: np.ndarray,
                 device) -> List[Dict]:
    """One row per supervised set: the offset, and the features a rule may use.

    The frozen field is re-forwarded from the task's own tiles rather than read
    from the cached box: ``build_host_sets`` builds every reference from the
    tile-wise forward, and the two paths differ slightly, so mixing them would
    put a spurious offset into every row.
    """
    data = {t: {"lr": task.tile_data[t]["lr"].to(device),
                "noise": {s: v.to(device)
                          for s, v in task.tile_data[t]["noise"].items()},
                "hr": task.tile_data[t]["hr"].to(device)}
           for t in task.tiles}
    with torch.no_grad():
        base = forward_tiles(model, data, task.tiles, geom).float()
        pos, vel = tile_particles(base, task.tiles, **kw)
        sets = task.sets.to(device)

        hx = torch.as_tensor(host_pos, dtype=pos.dtype, device=pos.device)
        rows: List[Dict] = []
        for s in range(sets.n_sets):
            p, _, _ = _gather_one(pos, vel, sets, s)
            frozen_c = p.mean(dim=0)
            ref_c = sets.centre_target[s]
            # The host sits on the same periodic branch as the set it owns, or
            # rhat is a box-length artifact for any cluster near a face.
            host_c = unwrap_about(hx[None, :], sets.centre_ref[s],
                                  sets.boxsize_mpc_h)[0]
            o = (frozen_c - ref_c).cpu().numpy().astype(np.float64)
            radial = (ref_c - host_c).cpu().numpy().astype(np.float64)
            d_host = float(np.linalg.norm(radial))
            rhat = radial / d_host if d_host > 1e-9 else np.zeros(3)
            rows.append({
                "key": task.key,
                "halo_id": int(sets.halo_id[s]),
                "num_p": int(sets.num_p[s]),
                "n_live": int(sets.n_live[s]),
                "scale": float(sets.centre_scale[s]),
                "log_mvir": float(task.sel.log_mvir),
                "d_host": d_host,
                "o": o.tolist(),
                "rhat": rhat.tolist(),
                "o_par": float(o @ rhat),
                "o_perp": float(np.linalg.norm(o - (o @ rhat) * rhat)),
                "o_abs": float(np.linalg.norm(o)),
            })
    return rows


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def _design(rows: Sequence[Dict]) -> np.ndarray:
    """Features a generator could actually condition on. Never the answer."""
    return np.stack([
        np.ones(len(rows)),
        np.array([r["d_host"] for r in rows]),
        np.log10(np.array([max(r["num_p"], 1) for r in rows], dtype=np.float64)),
        np.array([r["log_mvir"] for r in rows]),
    ], axis=1)


def fit_rules(train: Sequence[Dict]) -> Dict[str, object]:
    """Least squares on the training hosts only."""
    if not train:
        return {"radial_a": 0.0, "regressed_beta": [0.0, 0.0, 0.0, 0.0]}
    o = np.array([r["o"] for r in train])
    rhat = np.array([r["rhat"] for r in train])
    # `o ~ a rhat` in 3-D least squares is a = <o . rhat> / <rhat . rhat>, and
    # rhat is a unit vector, so it is simply the mean projection.
    a = float((o * rhat).sum(axis=1).mean())
    x = _design(train)
    y = np.array([r["o_par"] for r in train])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"radial_a": a, "regressed_beta": [float(b) for b in beta]}


def apply_rule(rows: Sequence[Dict], rule: str, fit: Dict) -> np.ndarray:
    """Residual offset per set, in Mpc/h, after the rule's correction."""
    o = np.array([r["o"] for r in rows])
    if rule == "none":
        return np.linalg.norm(o, axis=1)
    rhat = np.array([r["rhat"] for r in rows])
    if rule == "radial":
        pred = float(fit["radial_a"]) * rhat
    elif rule == "regressed":
        pred = (_design(rows) @ np.array(fit["regressed_beta"]))[:, None] * rhat
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return np.linalg.norm(o - pred, axis=1)


def score(rows: Sequence[Dict], fit: Dict) -> Dict[str, Dict[str, float]]:
    """Each rule's residual, in the gate's own units."""
    if not rows:
        return {}
    scale = np.array([r["scale"] for r in rows])
    total = float((np.array([r["o_abs"] for r in rows]) ** 2).sum())
    out: Dict[str, Dict[str, float]] = {}
    for rule in RULES:
        raw = apply_rule(rows, rule, fit)
        res = raw / scale
        out[rule] = {
            # THE statistic. The share of the offset's squared magnitude the
            # rule accounts for -- an R^2 on a vector target, and the direct
            # answer to "how much of this is a rule". `frac_within_1r` is
            # context, not the verdict: a search radius is max(r_vir, 0.15) and
            # is a demanding target, which is why the free field itself -- with
            # every address in hand -- reached only 46.8%.
            "explained_fraction": float(1.0 - (raw ** 2).sum() / max(total, 1e-30)),
            "median_radii": float(np.median(res)),
            "p90_radii": float(np.percentile(res, 90)),
            "frac_within_1r": float(np.mean(res <= 1.0)),
            "frac_within_2r": float(np.mean(res <= 2.0)),
            "n": int(len(rows)),
        }
    return out


#: The three centre terms of ``MemberGatherConfig.centre_mode``, and the offset
#: each one LEAVES BEHIND when it is driven perfectly to zero. This is an
#: arithmetic bound, not a simulation: it needs no optimiser, no generator and
#: no halo finder, because what a term does not charge for is exactly what
#: survives its own optimum.
#:
#:   full    charges the whole vector    -> residual 0
#:   radial  charges |o . rhat| only     -> residual |o_perp|
#:   self    charges the frozen anchor   -> residual |o|, the frozen offset
#:
#: Read it BEFORE spending a gate on an arm. The gate matches inside ONE search
#: radius, so the ``<=1r`` column is a hard ceiling on that arm's per-target
#: score in the free field, where the optimiser sees every address and can
#: therefore reach the arm's optimum exactly.
ARMS = ("full", "radial", "self")


def arm_residuals(rows: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """What each ``centre_mode`` leaves on the table at its own optimum."""
    if not rows:
        return {}
    scale = np.array([r["scale"] for r in rows])
    resid = {
        "full": np.zeros(len(rows)),
        "radial": np.array([r["o_perp"] for r in rows]),
        "self": np.array([r["o_abs"] for r in rows]),
    }
    out: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        u = resid[arm] / scale
        out[arm] = {
            "median_radii": float(np.median(u)),
            "p90_radii": float(np.percentile(u, 90)),
            "frac_within_1r": float(np.mean(u <= 1.0)),
            "frac_within_2r": float(np.mean(u <= 2.0)),
            "frac_within_3r": float(np.mean(u <= 3.0)),
            "n": int(len(rows)),
        }
    return out


def anisotropy(rows: Sequence[Dict]) -> Dict[str, float]:
    """How much of the offset is radial. The isotropic null is exactly 1/3."""
    if not rows:
        return {}
    par = np.array([r["o_par"] for r in rows])
    tot = np.array([r["o_abs"] for r in rows])
    scale = np.array([r["scale"] for r in rows])
    return {
        "radial_variance_fraction": float((par ** 2).sum() / (tot ** 2).sum()),
        "isotropic_null": 1.0 / 3.0,
        "median_o_abs_mpc_h": float(np.median(tot)),
        "median_o_abs_radii": float(np.median(tot / scale)),
        "median_o_par_mpc_h": float(np.median(par)),
        "median_o_perp_mpc_h": float(np.median([r["o_perp"] for r in rows])),
        "frac_o_par_positive": float(np.mean(par > 0.0)),
        "n_sets": int(len(rows)),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(res: Dict) -> None:
    banner("centre offset: rule or address?")
    for label in ("train", "holdout"):
        a = res["anisotropy"].get(label) or {}
        if not a:
            continue
        print(f"\n{label}: {a['n_sets']} sets, median offset "
              f"{a['median_o_abs_mpc_h']:.3f} Mpc/h "
              f"({a['median_o_abs_radii']:.2f} search radii)")
        print(f"  radial variance fraction {a['radial_variance_fraction']:.3f} "
              f"(isotropic null {a['isotropic_null']:.3f})")
        print(f"  median o_par {a['median_o_par_mpc_h']:+.3f} Mpc/h, "
              f"median |o_perp| {a['median_o_perp_mpc_h']:.3f}, "
              f"{a['frac_o_par_positive']:.0%} of sets sit further OUT than "
              f"they should")

    print("\nrules -- fitted on the training hosts, scored on both")
    print(f"  {'rule':<12} {'split':<9} {'explained':>10} {'median':>8} "
          f"{'p90':>8} {'<=1r':>7} {'<=2r':>7}")
    for rule in RULES:
        for label in ("train", "holdout"):
            r = (res["rules"].get(label) or {}).get(rule)
            if not r:
                continue
            print(f"  {rule:<12} {label:<9} {r['explained_fraction']:10.3f} "
                  f"{r['median_radii']:8.2f} "
                  f"{r['p90_radii']:8.2f} {r['frac_within_1r']:7.1%} "
                  f"{r['frac_within_2r']:7.1%}")

    arms = res.get("arms") or {}
    if arms:
        print("\nthe three centre_mode arms -- residual offset AT EACH ARM'S "
              "OWN OPTIMUM")
        print("  (arithmetic, not a run: what a term does not charge for is "
              "what survives it)")
        print(f"  {'centre_mode':<12} {'split':<9} {'median':>8} {'p90':>8} "
              f"{'<=1r':>7} {'<=2r':>7} {'<=3r':>7}")
        for arm in ARMS:
            for label in ("train", "holdout"):
                r = (arms.get(label) or {}).get(arm)
                if not r:
                    continue
                print(f"  {arm:<12} {label:<9} {r['median_radii']:8.2f} "
                      f"{r['p90_radii']:8.2f} {r['frac_within_1r']:7.1%} "
                      f"{r['frac_within_2r']:7.1%} {r['frac_within_3r']:7.1%}")
        ho = arms.get("holdout") or {}
        if ho:
            print("\n  The `<=1r` column is a CEILING on that arm's free-field "
                  "per-target score:")
            print("  the free field sees every address, so it reaches its "
                  "objective's optimum, and")
            print("  what the objective stopped asking for is what the gate "
                  "still measures. Against")
            print("  the measured reference points -- full 72/154 = 46.8%, "
                  "frozen 3/154 = 1.9%,")
            print("  geometric ceiling 151/154 = 98.1%:")
            for arm in ARMS:
                print(f"    {arm:<8} <= {ho[arm]['frac_within_1r']:.1%} of "
                      f"targets inside one search radius")
            print("  So `radial` and `self` CANNOT beat `full` in the free "
                  "field, by construction.")
            print("  Their case is as a GENERATOR objective, where the address "
                  "is not available and")
            print("  `full`'s own learnable content is the "
                  "`explained_fraction` above, not 100%.")

    hold = (res["rules"].get("holdout") or {})
    if hold:
        best_rule = max(RULES, key=lambda r: hold[r]["explained_fraction"])
        best = hold[best_rule]["explained_fraction"]
        print(f"\nHELD OUT: the best rule ({best_rule}) accounts for "
              f"{best:.1%} of the offset, cutting the median from "
              f"{hold['none']['median_radii']:.2f} to "
              f"{hold[best_rule]['median_radii']:.2f} search radii.")
        # The verdict is on the EXPLAINED FRACTION, not on frac_within_1r. A
        # search radius is max(r_vir, 0.15) Mpc/h and a strongly radial offset
        # with modest residual scatter still leaves most sets outside one; the
        # free field, holding every address, reached 46.8% of targets. Reading
        # frac_within_1r as the verdict would call a large real effect a null.
        if best < 0.2:
            print("  VERDICT: the offset is mostly NOT predictable from the "
                  "environment. The centre term is an ADDRESS, a shared "
                  "operator is being charged for information its input does "
                  "not carry, and no rung of the ladder fixes that -- soften "
                  "the term (dead zone + Huber) and put position on a "
                  "self-consistency condition instead.")
        elif best < 0.5:
            print("  VERDICT: PARTLY a rule. Keep the centre term, soften its "
                  "tail so the unpredictable sets stop owning the gradient, "
                  "and expect a generator to land between the frozen field and "
                  "the free field rather than at it.")
        else:
            print("  VERDICT: the offset is largely a LEARNABLE RULE -- a "
                  "systematic infall deficit. The centre term stays as it is "
                  "and the fine-tune's problem is capacity and receptive "
                  "field, not the objective.")
        print("  Two things this does NOT say. It bounds the CENTRE term "
              "alone -- nothing here is about whether the internal moments are "
              "reachable. And the rules fitted here are linear in three "
              "features; a conv sees the whole field, so this is a FLOOR on "
              "what is learnable, not a ceiling. The gate stays real Rockstar.")


# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    if args.from_json:
        res = json.loads(Path(args.from_json).read_text())
        # `arms` was added after the 2026-08-23 pool was written. Backfill it
        # from the rows rather than asking for a 25-minute GPU job again: it is
        # a function of `o_perp`, `o_abs` and `scale`, all of which are stored.
        if "arms" not in res and res.get("rows"):
            rows = res["rows"]
            res["arms"] = {
                lab: arm_residuals([r for r in rows if r.get("split") == lab])
                for lab in ("train", "holdout")}
        report(res)
        return res

    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    pscfg = phase_space_config_of(cfg)
    mcfg = member_config_of(args)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    banner("centre-offset decomposition (no halo finder, no optimisation)")

    train_boxes = [b for b in str(args.train_boxes).replace(",", " ").split() if b]
    hold_boxes = [b for b in str(args.holdout_boxes).replace(",", " ").split() if b]

    frozen = load_controlled_generator(
        model_path_of(cfg), in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    for p in frozen.parameters():
        p.requires_grad_(False)

    tasks = build_pool(args, cfg, geom, scfg, pscfg, mcfg, frozen, device,
                       train_boxes + hold_boxes)
    if not tasks:
        raise SystemExit(
            "no hosts selected -- every box was skipped for want of an owner "
            "array. Build them with scripts/slurm/submit_owner_arrays.sh.")
    split = split_pool([t.sel for t in tasks], train_boxes=train_boxes,
                       holdout_boxes=hold_boxes, holdout_keys=[])
    which = {s.key: "train" for s in split.train}
    which.update({s.key: "holdout" for s in split.holdout})

    kw = dict(ng_hr=NG_HR, tile_hr=TILE, boxsize_mpc_h=BOXSIZE,
              dis_scale_mpc_h=float(scfg.dis_norm_kpc_h) * 1e-3,
              vel_scale_kms=float(pscfg.vel_norm_km_s))
    pos_of: Dict[str, Dict[int, np.ndarray]] = {}
    rows: List[Dict] = []
    for task in tasks:
        box = task.sel.box
        if box not in pos_of:
            cat = _catalog(box, "hr")
            pos_of[box] = {int(h): np.asarray(cat.pos[i], dtype=np.float64)
                           for i, h in enumerate(cat.ids)}
        hp = pos_of[box][int(task.sel.halo_id)]
        r = host_offsets(task, frozen, geom, kw, hp, device)
        for row in r:
            row["split"] = which.get(task.key, "train")
        rows += r
        print(f"  {task.key}: {len(r)} sets, median offset "
              f"{np.median([x['o_abs'] for x in r]):.3f} Mpc/h", flush=True)

    train = [r for r in rows if r["split"] == "train"]
    hold = [r for r in rows if r["split"] == "holdout"]
    fit = fit_rules(train)
    res = {
        "ok": True,
        "config": {k: v for k, v in vars(args).items() if k != "from_json"},
        "fit": fit,
        "anisotropy": {"train": anisotropy(train), "holdout": anisotropy(hold)},
        "rules": {"train": score(train, fit), "holdout": score(hold, fit)},
        "arms": {"train": arm_residuals(train),
                 "holdout": arm_residuals(hold)},
        "per_host": sorted({r["key"] for r in rows}),
        "rows": rows,
    }
    out_dir = paths.subdir("centre_offset", f"pool{args.label}", create=True)
    write_json(out_dir / "offsets.json", res)
    report(res)
    print(f"\n  wrote {out_dir / 'offsets.json'}")
    print(f"  redraw with: python {Path(__file__).name} --from-json "
          f"{out_dir / 'offsets.json'}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    ap.add_argument("--from-json", default="",
                    help="re-print the table from a previous run; no compute")
    ap.add_argument("--train-boxes", default="set3 set4 set5 set6 set7")
    ap.add_argument("--holdout-boxes", default="set9 set10")
    ap.add_argument("--n-tiles", type=int, default=4)
    ap.add_argument("--max-hosts-per-box", type=int, default=8)
    ap.add_argument("--min-log-mvir", type=float, default=13.5)
    ap.add_argument("--min-num-p", type=int, default=200)
    ap.add_argument("--min-purity", type=float, default=0.5)
    ap.add_argument("--min-live-frac", type=float, default=0.5)
    ap.add_argument("--max-sets", type=int, default=256)
    ap.add_argument("--softening", type=float, default=0.01)
    ap.add_argument("--softening-kind", default="plummer")
    ap.add_argument("--pot-chunk", type=int, default=2048)
    ap.add_argument("--bound-tau", type=float, default=0.5)
    ap.add_argument("--bound-temperature", default="adaptive")
    # The background is only used by d6, which this script never evaluates, and
    # building it is the slowest part of the pool. Off by default here.
    ap.add_argument("--bg-k", type=int, default=0)
    ap.add_argument("--bg-radius", type=float, default=4.0)
    ap.add_argument("--w-virial", type=float, default=1.0)
    ap.add_argument("--w-bound", type=float, default=1.0)
    ap.add_argument("--w-d6", type=float, default=1.0)
    ap.add_argument("--w-rrms", type=float, default=0.3)
    ap.add_argument("--w-sigmav", type=float, default=0.3)
    ap.add_argument("--w-centre", type=float, default=1.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="")
    args = ap.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
