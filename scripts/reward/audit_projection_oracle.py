#!/usr/bin/env python
"""Phase 1.2: how much does forbidding the coarse residual actually cost?

Builds, for paired diagnostic boxes and several fixed SR2 seeds,

    X(alpha_dis, alpha_vel) = B + T_{alpha_dis}(r_dis) + T_{alpha_vel}(r_vel),
    B = Psi_SR2,  r = Psi_HR - B,  T_alpha(r) = P_N r + alpha * P_R r,

assembles the complete periodic box, runs the **real** Rockstar pipeline on it,
and measures host-conditioned occupation, subhalo abundance, host recovery
against the HR catalog, density / displacement / velocity power, and the
LR-coarse mismatch.

``alpha = 1`` is ``Psi_HR`` exactly, so the sweep says how much of HR's
advantage survives when the LR-visible part of the residual is removed. The
``hr`` and ``sr2`` reference arms bracket it, and the ``joint_a1`` vs ``hr``
agreement is reported as a self-consistency check on the construction.

This job chooses a constraint. **It writes no training data**: no residual, crop
or replay entry leaves it -- only scored rows. ``scripts/reward/projection_oracle_report.py``
turns those rows into the JSON/CSV/Markdown report and the recommendation.

Resumable (one JSON per (arm, box, seed); existing rows are skipped) and
shardable with ``--shard i --num-shards n``.

    python scripts/reward/audit_projection_oracle.py --run-name proj0 \\
        --boxes set8,set9 --base-seeds 0,1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from _common import (add_common_args, assert_no_leak, banner, bins_of, chunk_grid,
                     constraints_of, hr_path, load_freeze, load_reward_config,
                     lr_path, parse_boxes, write_json, PROJECT_ROOT)

from cosmo_sr.eval.halo_match import match_hosts
from cosmo_sr.eval.rockstar import load_rockstar_ascii
from cosmo_sr.reward import fields as F
from cosmo_sr.reward import paths
from cosmo_sr.reward.base import find_base_field
from cosmo_sr.reward.catalog import pool, write_summaries
from cosmo_sr.reward.constraints import constraint_values
from cosmo_sr.reward.pipeline import existing_catalog, field_to_chunk_summaries
from cosmo_sr.reward.projection import (DEFAULT_ALPHAS, ProjectionArm, arm_plan,
                                        project_residual_field)
from cosmo_sr.reward.reward import RewardModel


# --------------------------------------------------------------------------- #
# Field metrics beyond the standard constraint set
# --------------------------------------------------------------------------- #
def coarse_and_velocity_metrics(
    x: np.ndarray,
    base: np.ndarray,
    hr: np.ndarray,
    y_lr: np.ndarray,
    *,
    scale_factor: int,
    n_bins: int,
    vel_channels: Sequence[int] = (3, 4, 5),
) -> Dict[str, float]:
    """LR/coarse mismatch, low-k change against HR, and velocity power.

    ``constraint_values`` already covers displacement and density against HR and
    the LR consistency of the candidate; what it does not carry is (a) the coarse
    field measured against **HR's** coarse field, which is the quantity ``alpha``
    directly controls, and (b) velocity, which has its own allowance in this
    sweep and would otherwise be unobserved.
    """
    out: Dict[str, float] = {}
    a_x = F.block_average(x, scale_factor)
    a_hr = F.block_average(np.asarray(hr), scale_factor)
    a_b = F.block_average(np.asarray(base), scale_factor)

    out["coarse_mismatch_vs_hr"] = F.rel_rms(a_x, a_hr)
    out["coarse_mismatch_vs_lr"] = F.rel_rms(a_x, np.asarray(y_lr))
    out["coarse_mismatch_base_vs_hr"] = F.rel_rms(a_b, a_hr)
    # Split the same quantity by group, since the two alphas act separately.
    out["coarse_mismatch_vs_hr_disp"] = F.rel_rms(a_x[0:3], a_hr[0:3])
    out["coarse_mismatch_vs_hr_vel"] = F.rel_rms(a_x[3:6], a_hr[3:6])
    del a_x, a_hr, a_b

    k_lr_nyq = float(x.shape[1]) / (2.0 * float(scale_factor))
    t_err, t_low, r_low = [], [], []
    for ch in vel_channels:
        k, p_hat, p_hr, p_x = F.cross_power(np.asarray(x[ch]), np.asarray(hr[ch]), n_bins)
        tk = np.sqrt(np.maximum(p_hat, 0.0) / np.maximum(p_hr, 1e-30))
        rk = p_x / np.maximum(np.sqrt(np.maximum(p_hat * p_hr, 0.0)), 1e-30)
        m = F.band_masks(k, k_lr_nyq)
        t_err.append(float(np.mean(np.abs(tk - 1.0))))
        t_low.append(float(np.mean(np.abs(tk[m["low"]] - 1.0))) if m["low"].any() else np.nan)
        r_low.append(float(np.mean(rk[m["low"]])) if m["low"].any() else np.nan)
    out["velocity_power_error"] = float(np.mean(t_err))
    out["velocity_power_error_low_k"] = float(np.mean(t_low))
    out["velocity_rk_low_k"] = float(np.mean(r_low))
    return out


def matching_metrics(
    cat, hr_cat, *, boxsize_mpc_h: float, mhost_min: float = 0.0
) -> Dict[str, float]:
    """Host recovery against the HR catalog, plus subhalo counts.

    ``host_recovery_fraction`` is the fraction of HR hosts above ``mhost_min``
    that found a counterpart. It is a *primary* metric in the decision rule
    because occupation can be raised by deleting hosts, and a recovery fraction
    that fell while occupation rose is exactly that failure.
    """
    out: Dict[str, float] = {}
    try:
        res = match_hosts(hr_cat, cat, boxsize_mpc_h=boxsize_mpc_h, mhost_min=mhost_min)
    except Exception as exc:                     # pragma: no cover - scipy/edge cases
        out["host_match_error"] = repr(exc)
        out["host_recovery_fraction"] = float("nan")
        return out
    hr_hosts = hr_cat.hosts()
    sel = np.asarray(hr_hosts.mvir, dtype=np.float64) >= float(mhost_min)
    matched = (np.asarray(res.sr_ids) >= 0) & sel
    n_ref = int(np.count_nonzero(sel))
    out["n_hr_hosts_considered"] = n_ref
    out["host_recovery_fraction"] = float(np.count_nonzero(matched) / max(n_ref, 1))
    sc = np.asarray(res.score, dtype=np.float64)[matched]
    out["host_match_score_median"] = float(np.median(sc)) if sc.size else float("nan")
    out["n_hosts"] = int(cat.hosts().n)
    out["n_subs"] = int(cat.subhalos().n)
    out["n_halos"] = int(cat.n)
    out["subhalo_ratio_vs_hr"] = float(
        cat.subhalos().n / max(hr_cat.subhalos().n, 1)
    )
    return out


# --------------------------------------------------------------------------- #
# One arm
# --------------------------------------------------------------------------- #
def build_field(
    arm: ProjectionArm, hr, base, *, scale_factor: int, disp_channels, vel_channels,
    slab: int,
) -> np.ndarray:
    if arm.name == "sr2":
        return np.asarray(base, dtype=np.float32)
    if arm.name == "hr":
        return np.asarray(hr, dtype=np.float32)
    return project_residual_field(
        hr, base, alpha_disp=arm.alpha_disp, alpha_vel=arm.alpha_vel,
        scale_factor=scale_factor, disp_channels=disp_channels,
        vel_channels=vel_channels, slab=slab,
    )


def score_arm(
    arm: ProjectionArm, box: str, seed: Optional[int], *, cfg, freeze, grid, bins,
    model: Optional[RewardModel], reliable, out_dir: Path, args,
    default_seed: int,
) -> Optional[Dict]:
    """One (arm, box, seed) row. ``seed=None`` means the arm is seed-independent
    (the ``hr`` arm), and ``default_seed`` says which cached base field to read
    for the geometry it still needs."""
    tag = f"{arm.name}__{box}" + ("" if seed is None else f"__s{seed}")
    row_path = out_dir / "rows" / f"{tag}.json"
    if row_path.is_file() and not args.overwrite:
        print(f"[{tag}] cached", flush=True)
        return json.loads(row_path.read_text())

    dcfg = cfg["data"]
    geo = cfg.get("geometry", {})
    corr = cfg.get("correction", {})
    sf = int(dcfg["scale_factor"])
    disp_ch = tuple(int(c) for c in corr.get("disp_channels", (0, 1, 2)))
    vel_ch = tuple(int(c) for c in corr.get("vel_channels", (3, 4, 5)))

    hr = np.load(hr_path(cfg, box), mmap_mode="r")
    base_path = find_base_field(box, int(seed if seed is not None else default_seed))
    if base_path is None:
        raise SystemExit(f"no cached SR2 base field for {box} seed {seed}")
    base = np.load(base_path, mmap_mode="r")
    lr = np.load(lr_path(cfg, box))

    t0 = time.time()
    x = build_field(arm, hr, base, scale_factor=sf, disp_channels=disp_ch,
                    vel_channels=vel_ch, slab=int(args.slab))
    t_build = time.time() - t0

    row: Dict = {
        **arm.to_dict(), "box": box, "seed": seed, "tag": tag,
        "base_field": str(base_path), "seconds_build": t_build,
    }
    # alpha=1 must reproduce HR; a drift here would invalidate the whole sweep.
    if arm.alpha_disp == 1.0 and arm.alpha_vel == 1.0:
        row["reconstruction_rel_rms_vs_hr"] = F.rel_rms(x, np.asarray(hr, dtype=np.float32))

    t0 = time.time()
    vals = constraint_values(
        x, np.asarray(base, dtype=np.float32), lr, hr=hr, scale_factor=sf,
        n_bins=int(args.n_bins), boxsize_mpc_h=float(dcfg["boxsize_mpc_h"]),
        dis_norm_kpc_h=float(dcfg["dis_norm_kpc_h"]),
        redshift=float(dcfg.get("redshift", 0.0)),
        compute_density=not args.no_density,
    )
    vals.update(coarse_and_velocity_metrics(
        x, base, hr, lr, scale_factor=sf, n_bins=int(args.n_bins), vel_channels=vel_ch))
    row["field"] = vals
    row.update({k: v for k, v in vals.items()
                if k in ("low_k_change", "lr_consistency_error", "density_power_error",
                         "displacement_power_error", "velocity_power_error",
                         "coarse_mismatch_vs_hr", "coarse_mismatch_vs_lr",
                         "coarse_mismatch_vs_hr_disp", "coarse_mismatch_vs_hr_vel",
                         "density_sigma_ratio", "density_pdf_l1")})
    row["seconds_field"] = time.time() - t0

    if not args.skip_halo:
        res = field_to_chunk_summaries(
            x, box=box, source=f"projection_{arm.name}", tag=tag, grid=grid, bins=bins,
            work_dir=out_dir / "halos" / tag,
            rockstar_binary=PROJECT_ROOT / freeze["rockstar"]["binary"],
            rockstar_cfg=PROJECT_ROOT / freeze["rockstar"]["config"],
            boxsize_kpc_h=float(freeze["cosmology_sim"]["boxsize_kpc_h"]),
            redshift=float(freeze.get("redshift", 0.0)),
            dis_norm_kpc_h=float(dcfg["dis_norm_kpc_h"]),
            purity_grid=int(geo.get("purity_grid", 128)),
            max_half_width=int(geo.get("max_half_width", 4)),
            reuse_catalog=not args.overwrite,
        )
        summary_dir = out_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        write_summaries(summary_dir / f"{tag}.jsonl",
                        [res.summaries[c] for c in sorted(res.summaries)])
        write_summaries(summary_dir / f"{tag}__fullbox.jsonl", [res.full_box])

        ens = pool([res.full_box])
        row["n_hosts_full_box"] = int(res.full_box.n_host_total)
        row["n_subs_full_box"] = int(res.full_box.n_sub_total)
        row["n_sub_per_bin"] = np.asarray(ens.n_sub, dtype=float).tolist()
        row["n_host_per_bin"] = np.asarray(ens.n_host, dtype=float).tolist()
        row["occupation_per_bin"] = np.asarray(ens.occupation(), dtype=float).tolist()
        row["summaries"] = str(summary_dir / f"{tag}.jsonl")
        row["full_box_summary"] = str(summary_dir / f"{tag}__fullbox.jsonl")
        row["seconds_halo"] = res.timings

        if model is not None:
            scores = model.scores(ens, reliable_host_bins=reliable)
            row.update({k: float(v) for k, v in scores.items()})
            row["occupation_gap_per_bin"] = np.asarray(
                model.occupation_gap(ens), dtype=float).tolist()

        hr_cat_path = _hr_catalog_path(out_dir, box)
        if hr_cat_path is not None and hr_cat_path.is_file():
            row.update(matching_metrics(
                res.catalog, load_rockstar_ascii(hr_cat_path),
                boxsize_mpc_h=float(dcfg["boxsize_mpc_h"]),
                mhost_min=float(args.mhost_min),
            ))
        else:
            row["host_recovery_fraction"] = float("nan")
            row["host_match_note"] = (
                "no HR catalog yet; run the `hr` reference arm for this box first "
                "(--references only)"
            )
        print(f"[{tag}] hosts={row['n_hosts_full_box']} subs={row['n_subs_full_box']} "
              f"R_occ_rel={row.get('R_occ_reliable', float('nan')):.4g} "
              f"recov={row.get('host_recovery_fraction', float('nan')):.3f} "
              f"low_k={vals['low_k_change']:.4g}", flush=True)
    else:
        print(f"[{tag}] field only: low_k={vals['low_k_change']:.4g} "
              f"coarse_vs_hr={vals['coarse_mismatch_vs_hr']:.4g}", flush=True)

    del x
    write_json(row_path, row)
    return row


def _hr_catalog_path(out_dir: Path, box: str) -> Optional[Path]:
    tag = f"hr__{box}"
    return existing_catalog(out_dir / "halos" / tag, tag)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-name", default="projection_oracle")
    ap.add_argument("--boxes", default=None, help="comma list; default = the config's boxes")
    ap.add_argument("--split", default="val",
                    choices=["train", "val", "test", "dev", "final"])
    ap.add_argument("--base-seeds", default="0",
                    help="comma list of fixed SR2 seeds (several, so a conclusion "
                         "is not one realisation)")
    ap.add_argument("--alphas", default=None,
                    help=f"comma list; default {list(DEFAULT_ALPHAS)}")
    ap.add_argument("--sweeps", default="joint,disp_only,vel_only")
    ap.add_argument("--references", default="shard0",
                    choices=["shard0", "only", "skip", "all"],
                    help="who runs the `hr`/`sr2` reference arms. Every shard "
                         "needs the HR catalog for matching, but two shards "
                         "running Rockstar in the same work directory corrupt "
                         "each other -- so by default only shard 0 does, and "
                         "`only` exists for a prerequisite array job")
    ap.add_argument("--reward-model", default=None)
    ap.add_argument("--mhost-min", type=float, default=0.0,
                    help="minimum HR host mass counted in host recovery")
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--slab", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--no-density", action="store_true")
    ap.add_argument("--skip-halo", action="store_true",
                    help="field metrics only; a cheap first pass")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_reward_config(args)
    freeze = load_freeze(cfg)
    grid = chunk_grid(cfg)
    bins = bins_of(cfg)
    constraints_of(cfg)   # validated, but not used to filter: this is a measurement

    po = cfg.get("projection_oracle", {})
    boxes = parse_boxes(args.boxes, cfg, args.split) if (args.boxes or not po.get("boxes")) \
        else list(po["boxes"])
    # Paired HR is read here. That is allowed for calibration, and forbidden on
    # the final-eval boxes, which have to stay untouched until the comparison.
    assert_no_leak(cfg, boxes, ["train", "val", "dev"])

    seeds = [int(s) for s in str(args.base_seeds).split(",") if s.strip() != ""]
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else \
        [float(a) for a in po.get("alphas", DEFAULT_ALPHAS)]
    sweeps = [s.strip() for s in args.sweeps.split(",") if s.strip()]
    arms = arm_plan(alphas, sweeps)

    reliable = cfg.get("reward", {}).get("occupation", {}).get(
        "reliable_host_bins", [0, 1, 2, 3])
    rm_path = Path(args.reward_model) if args.reward_model else \
        paths.subdir("reward_model") / "reward_model.json"
    model = None
    if Path(rm_path).is_file():
        model = RewardModel.from_dict(json.loads(Path(rm_path).read_text()))
    else:
        print(f"  ! no reward model at {rm_path}; occupation curves are still "
              f"measured, but R_occ/R_occ_reliable will be absent and the "
              f"decision rule cannot use them. Run "
              f"scripts/reward/fit_reward_model.py.", flush=True)

    out_dir = paths.AUDITS(f"projection_oracle/{args.run_name}", create=True)
    (out_dir / "rows").mkdir(parents=True, exist_ok=True)

    # (arm, box, seed) work items. The reference arms are split out because the
    # `hr` arm is seed-independent and every other row needs its catalog.
    ref_arms = [a for a in arms if a.sweep == "reference"]
    sweep_arms = [a for a in arms if a.sweep != "reference"]
    shard, n_shards = int(args.shard), max(1, int(args.num_shards))

    ref_items: List = []
    for arm in ref_arms:
        for box in boxes:
            # HR does not depend on the SR2 seed; the frozen baseline does.
            for seed in ([None] if arm.name == "hr" else seeds):
                ref_items.append((arm, box, seed))
    if args.references == "shard0":
        ref_items = ref_items if shard == 0 else []
    elif args.references == "only":
        ref_items = ref_items[shard::n_shards]
    elif args.references == "skip":
        ref_items = []

    items: List = []
    if args.references != "only":
        for arm in sweep_arms:
            for box in boxes:
                for seed in seeds:
                    items.append((arm, box, seed))
        items = items[shard::n_shards]

    banner(f"projection oracle '{args.run_name}': {len(ref_items)} reference + "
           f"{len(items)} sweep rows (shard {shard}/{n_shards}); "
           f"boxes={boxes} seeds={seeds} alphas={alphas}")

    rows = []
    for arm, box, seed in ref_items + items:
        r = score_arm(arm, box, seed, cfg=cfg, freeze=freeze, grid=grid, bins=bins,
                      model=model, reliable=reliable, out_dir=out_dir, args=args,
                      default_seed=seeds[0])
        if r is not None:
            rows.append(r)

    write_json(out_dir / f"manifest_shard{shard}.json", {
        "run_name": args.run_name,
        "boxes": boxes, "seeds": seeds, "alphas": alphas, "sweeps": sweeps,
        "arms": [a.to_dict() for a in arms],
        "reward_model": str(rm_path) if model is not None else None,
        "reliable_host_bins": list(reliable),
        "mhost_min": float(args.mhost_min),
        "shard": shard, "num_shards": n_shards,
        "references": args.references,
        "n_rows": len(rows),
        "purpose": (
            "choose alpha_disp and alpha_vel. This job writes NO training data: "
            "no residual, crop or replay entry is produced from these fields."
        ),
    })
    banner(f"{len(rows)} rows -> {out_dir}/rows; "
           f"next: python scripts/reward/projection_oracle_report.py "
           f"--run-name {args.run_name}")


if __name__ == "__main__":
    main()
