#!/usr/bin/env python
"""Stage 1 — localise the SR2 subhalo failure at z=0.

For each test box and each nested seed:
  1. Generate full-box SR (frozen inference).
  2. Run Rockstar on HR and SR (identical config, periodic full box).
  3. Record field controls + halo/subhalo metrics + match classification.

Outputs under ``--out``:
  field_rows.jsonl, halo_rows.jsonl, match_rows.jsonl, summary.json

z=2 is intentionally deferred until matched catnorm pairs exist.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _append_jsonl(path: Path, row: dict):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _field_controls(sr, hr, cell, dis_norm, device):
    from cosmo_sr.eval.density import cic_density
    from cosmo_sr.eval.spectra import power_spectrum, cross_correlation_coefficient
    from cosmo_sr.eval.sr2_stats import equilateral_bispectrum

    out = {}
    for name, ch in (("disp", 0), ("vel", 3)):
        k, pk_sr = power_spectrum(sr[ch])
        _, pk_hr = power_spectrum(hr[ch])
        _, rk = cross_correlation_coefficient(sr[ch], hr[ch])
        denom = np.clip(pk_hr, 1e-30, None)
        hi = slice(int(len(k) * 2 / 3), None)
        out[f"{name}_Tk_high"] = float(np.mean(np.sqrt(pk_sr / denom)[hi]))
        out[f"{name}_rk_high"] = float(np.mean(rk[hi]))
        out[f"{name}_Pk_ratio_high"] = float(np.mean((pk_sr / denom)[hi]))

    d_sr = cic_density(torch.as_tensor(sr[0:3])[None].to(device), cell, dis_norm)[0, 0]
    d_hr = cic_density(torch.as_tensor(hr[0:3])[None].to(device), cell, dis_norm)[0, 0]
    d_sr, d_hr = d_sr.cpu().numpy(), d_hr.cpu().numpy()
    k, pk_sr = power_spectrum(d_sr)
    _, pk_hr = power_spectrum(d_hr)
    _, rk = cross_correlation_coefficient(d_sr, d_hr)
    denom = np.clip(pk_hr, 1e-30, None)
    hi = slice(int(len(k) * 2 / 3), None)
    out["density_Tk_high"] = float(np.mean(np.sqrt(pk_sr / denom)[hi]))
    out["density_rk_high"] = float(np.mean(rk[hi]))
    out["density_Pk_ratio_high"] = float(np.mean((pk_sr / denom)[hi]))
    out["density_sigma_ratio"] = float(d_sr.std() / max(d_hr.std(), 1e-12))
    # PDF L1 on log1p(1+delta) clipped
    bins = np.linspace(-1, 5, 64)
    hs, _ = np.histogram(np.clip(d_sr, -1, 5), bins=bins, density=True)
    hh, _ = np.histogram(np.clip(d_hr, -1, 5), bins=bins, density=True)
    out["density_pdf_l1"] = float(np.mean(np.abs(hs - hh)))
    try:
        _, b_sr = equilateral_bispectrum(d_sr, n_bins=6)
        _, b_hr = equilateral_bispectrum(d_hr, n_bins=6)
        out["bispectrum_eq_rel"] = float(
            np.nanmean(np.abs(b_sr - b_hr) / np.maximum(np.abs(b_hr), 1e-30))
        )
    except Exception as e:
        out["bispectrum_eq_rel"] = float("nan")
        out["bispectrum_error"] = str(e)
    return out, d_sr, d_hr


def _halo_row(tag, box, seed, cat, box_mpc):
    from cosmo_sr.eval import halo_metrics as hm
    hosts, subs = cat.hosts(), cat.subhalos()
    m_c, m_dn = hm.mass_function(hosts.mvir, box_mpc)
    s_c, s_dn = hm.mass_function(subs.mvir, box_mpc)
    v_c, v_dn = hm.vmax_function(subs.vmax, box_mpc)
    n_c, n_mean, n_h = hm.nsub_vs_mhost(cat)
    r_c, r_p = hm.subhalo_radial_profile(cat, boxsize_mpc_h=box_mpc)
    o_c, o_xi = hm.one_halo_correlation(cat, box_mpc)
    vel = hm.host_velocity_dispersion(cat)
    return {
        "tag": tag, "box": box, "seed": seed, "redshift": 0.0,
        "n_hosts": hosts.n, "n_subs": subs.n,
        "hmf_M": m_c.tolist(), "hmf_dn": m_dn.tolist(),
        "shmf_M": s_c.tolist(), "shmf_dn": s_dn.tolist(),
        "vmax_V": v_c.tolist(), "vmax_dn": v_dn.tolist(),
        "nsub_M": n_c.tolist(), "nsub_mean": n_mean.tolist(), "nsub_nhost": n_h.tolist(),
        "rad_r": r_c.tolist(), "rad_pdf": r_p.tolist(),
        "onehalo_r": o_c.tolist(), "onehalo_xi": o_xi.tolist(),
        **vel,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", default=str(ROOT / "configs/sr2_baseline/freeze.yaml"))
    ap.add_argument(
        "--out",
        default="/zfsauton/scratch/yixiz/DMSR/sr2_baseline/stage1",
        help="scratch Stage-1 root (large Rockstar/GADGET artifacts)",
    )
    ap.add_argument("--boxes", default=None, help="comma-separated; default = test_boxes")
    ap.add_argument("--seeds", default="0,1,2,3",
                    help="nested seeds; prefer 2–4 per box for box-to-box variance")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--skip-halo", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite-halo", action="store_true")
    ap.add_argument("--keep-records", action="store_true",
                    help="store per-subhalo match records (large JSONL)")
    ap.add_argument("--density-probe", action="store_true",
                    help="refine missing classes with SR CIC density peaks")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.freeze).read_text())
    from cosmo_sr.data.field_io import load_field
    from cosmo_sr.tts.srs_noise import load_controlled_generator
    from cosmo_sr.tts.sampling import super_resolve_srs_seeded
    from cosmo_sr.eval.rockstar import run_rockstar_on_field
    from cosmo_sr.eval.halo_match import match_hosts, classify_subhalos

    boxes = ([b.strip() for b in args.boxes.split(",")] if args.boxes
             else list(cfg["split"]["test_boxes"]))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    inf = cfg["inference"]
    nsplit, pad = int(inf["nsplit"]), int(inf["pad"])
    scale = int(cfg["model"]["scale_factor"])
    box_kpc = float(cfg["cosmology_sim"]["boxsize_kpc_h"])
    box_mpc = float(cfg["cosmology_sim"]["boxsize_mpc_h"])
    dis_norm = float(cfg["data"]["dis_norm_kpc_h"])
    data_root = Path(cfg["data"]["root"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    field_path, halo_path, match_path = (
        out / "field_rows.jsonl", out / "halo_rows.jsonl", out / "match_rows.jsonl"
    )

    G = load_controlled_generator(
        str(ROOT / cfg["model"]["path"]), scale_factor=scale, device=device,
    )

    for box in boxes:
        lr = load_field(str(data_root / "lr" / f"{box}.npy")).astype(np.float32)
        hr = np.ascontiguousarray(
            load_field(str(data_root / "hr" / f"{box}.npy"), mmap=True), dtype=np.float32
        )
        cell = box_kpc / hr.shape[1]
        hr_halo_dir = out / "halos" / box / "hr"
        if not args.skip_halo:
            t0 = time.time()
            hr_cat = run_rockstar_on_field(
                hr, hr_halo_dir, tag="hr", boxsize_kpc_h=box_kpc, redshift=0.0,
                overwrite=args.overwrite_halo,
            )
            _append_jsonl(halo_path, _halo_row("hr", box, -1, hr_cat, box_mpc))
            print(f"[{box}] HR Rockstar: {hr_cat.n} halos in {time.time()-t0:.1f}s",
                  flush=True)

        for seed in seeds:
            t0 = time.time()
            sr = super_resolve_srs_seeded(
                G, lr, seed, scale_factor=scale, nsplit=nsplit, pad=pad, device=device,
                noise_mode=inf.get("noise_mode", "per_tile"),
            )
            gen_s = time.time() - t0
            dens_sr = dens_hr = None
            if not args.skip_field or args.density_probe:
                controls, dens_sr, dens_hr = _field_controls(
                    sr, hr, cell, dis_norm, device,
                )
                if not args.skip_field:
                    _append_jsonl(field_path, {
                        "box": box, "seed": seed, "redshift": 0.0,
                        "gen_s": gen_s, **controls,
                    })
            if not args.skip_halo:
                sr_dir = out / "halos" / box / f"sr_seed{seed}"
                sr_cat = run_rockstar_on_field(
                    sr, sr_dir, tag=f"sr{seed}", boxsize_kpc_h=box_kpc, redshift=0.0,
                    overwrite=args.overwrite_halo,
                )
                _append_jsonl(halo_path, _halo_row("sr", box, seed, sr_cat, box_mpc))
                hm = match_hosts(hr_cat, sr_cat, boxsize_mpc_h=box_mpc)
                classes = classify_subhalos(hr_cat, sr_cat, hm, boxsize_mpc_h=box_mpc)
                if args.density_probe and dens_sr is not None:
                    from cosmo_sr.eval.halo_density_probe import refine_missing_with_density
                    hr_subs = hr_cat.subhalos()
                    pos_by_id = {int(i): p for i, p in zip(hr_subs.ids, hr_subs.pos)}
                    classes = refine_missing_with_density(
                        classes, dens_sr, dens_hr, box_mpc, pos_by_id,
                    )
                from collections import Counter
                counts = Counter(c["class"] for c in classes)
                n_matched = int((hm.sr_ids >= 0).sum())
                row = {
                    "box": box, "seed": seed, "redshift": 0.0,
                    "n_hr_hosts_matched": n_matched,
                    "n_hr_hosts_total": int(len(hm.hr_ids)),
                    "host_match_rate": float(n_matched / max(len(hm.hr_ids), 1)),
                    "n_hr_subs_classified": len(classes),
                    "class_counts": dict(counts),
                    "matcher": "host_nn_periodic_v2",
                }
                if args.keep_records:
                    row["records"] = classes
                _append_jsonl(match_path, row)
                print(
                    f"[{box} seed={seed}] SR halos={sr_cat.n}  "
                    f"host_match={row['host_match_rate']:.3f}  "
                    f"classes={dict(counts)}  ({time.time()-t0:.1f}s)",
                    flush=True,
                )

    summary = {
        "freeze": str(Path(args.freeze).resolve()),
        "boxes": boxes,
        "seeds": seeds,
        "redshift": 0.0,
        "note": "z=2 deferred; box-level bootstrap to be applied in analyze step",
        "field_rows": str(field_path),
        "halo_rows": str(halo_path),
        "match_rows": str(match_path),
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote Stage-1 outputs under {out}")


if __name__ == "__main__":
    main()
