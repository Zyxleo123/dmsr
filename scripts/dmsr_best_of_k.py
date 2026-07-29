#!/usr/bin/env python
"""Stage 6: cheap test-time-scaling baseline -- is a good candidate even in there?

Draw ``K`` candidates for one region with independent flow noise (optionally at
several noise temperatures and residual scales), then report three numbers as a
function of ``K``:

    random      the expected score of one candidate picked blindly
    oracle      the best candidate, chosen using the test HR box
    selected    the best candidate, chosen by a score computable WITHOUT test HR

``oracle`` measures the ceiling: what selection could buy if the selector were
perfect. If best-of-32 is barely better than a single sample, the generator does
not contain good candidates and no selector can rescue it -- stop. If the oracle
is much better, the gap between ``oracle`` and ``selected`` is what a verifier
would have to learn.

Why a region and not the full box
---------------------------------
The stated design (K up to 32 full 512^3 boxes) is not affordable: one full-box
DMSR draw measured ~5.3 GPU-hours here, so K=32 is ~170 GPU-hours per box. Stage 2
measured that a 64^3 scored region is fully converged once particles within 64 HR
cells of it are deposited, so a 24^3 LR window (192^3 HR = region + 64-cell buffer
on every face) contains everything needed to score that region exactly. That is a
single forward pass per candidate instead of 64 tiles.

The selection score must never see the test HR box; ``--train-hr`` supplies a
*training* box whose density statistics the deployable score matches against.

Usage
-----
    python scripts/dmsr_best_of_k.py \
        --config runs/dmsr/t13_unconstrained_s0/config.yaml \
        --ckpt   runs/dmsr/t13_unconstrained_s0/ckpt_best.pt \
        --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy \
        --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
        --train-hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set0.npy \
        --K 32 --regions 4 --out runs/dmsr/stage6_bok
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dmsr_cic_buffer_audit import cic_block_into_region, cic_into_region, region_metrics  # noqa: E402
from dmsr_context_oracle import deformation_stats, wrapped_block  # noqa: E402


def region_density_from_window(win_disp, buf, R, dscale):
    """CIC the central ``R^3`` of a window, using the window itself as the buffer.

    ``win_disp`` is ``(3, R+2b, R+2b, R+2b)`` normalized displacement covering the
    scored region plus ``buf`` HR cells on every face. At deployment the SR field
    exists everywhere, so scoring a region with the model's own surroundings is
    the honest construction -- and Stage 2 showed ``buf = 64`` makes it exact.
    """
    side = win_disp.shape[-1]
    d = win_disp * dscale
    q = np.arange(side, dtype=np.float64) + 0.5
    pos = np.empty((3, side, side, side), dtype=np.float64)
    pos[0] = d[0] + q[:, None, None]
    pos[1] = d[1] + q[None, :, None]
    pos[2] = d[2] + q[None, None, :]
    # scored cube sits at [buf, buf+R) of a domain we treat as periodic of size `side`
    return cic_into_region(pos.reshape(3, -1), [buf] * 3, R, side)


def pdf_hist(delta, edges):
    h, _ = np.histogram(np.log10(np.clip(delta + 1.0, 1e-3, None)), bins=edges)
    return h / max(h.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--hr", required=True, help="TEST HR: oracle scoring ONLY")
    ap.add_argument("--train-hr", required=True,
                    help="TRAINING HR box: supplies the reference density statistics "
                         "the deployable score matches against")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--regions", type=int, default=4, help="independent regions to average")
    ap.add_argument("--region-hr", type=int, default=64, help="scored cube side, HR cells")
    ap.add_argument("--buffer-hr", type=int, default=64,
                    help="CIC buffer; Stage 2 measured 64 as converged for a 64^3 region")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[1.0],
                    help="noise temperatures (z is scaled by each)")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--boxsize", type=float, default=100000.0)
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--out", default="runs/dmsr/stage6_bok")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    from cosmo_sr.utils.config import load_config
    from dmsr_eval import load_flow

    cfg = load_config(args.config)
    uc = cfg.get("data", {}).get("use_channels") or [0, 1, 2]
    scale = int(cfg.get("factor", 8))
    model = load_flow(cfg, len(uc), args.ckpt, device, use_ema=True)
    model.eval()

    lr = np.load(args.lr).astype(np.float32)[uc]
    Ng = lr.shape[-1]
    hr = np.load(args.hr, mmap_mode="r")
    Nhr = hr.shape[-1]
    cellsize = args.boxsize / Nhr
    dscale = args.dis_norm / cellsize
    R, B = int(args.region_hr), int(args.buffer_hr)
    if (R + 2 * B) % scale:
        raise SystemExit(f"region+2*buffer ({R + 2*B}) must be a multiple of scale {scale}")
    S = (R + 2 * B) // scale                       # LR window side
    print(f"device={device}  scored region {R}^3 HR, buffer {B}, LR window {S}^3 "
          f"-> HR window {S*scale}^3, K={args.K}, regions={args.regions}")

    # reference density statistics from a TRAINING box (never the test box)
    tr = np.load(args.train_hr, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    edges = np.linspace(-3, 3, 41)
    ref_sigmas, ref_pdfs, ref_peaks = [], [], []
    for _ in range(4):
        o = [int(x) for x in rng.integers(0, Nhr, size=3)]
        m, _ = cic_block_into_region(tr, o, R, B, Nhr, args.dis_norm, cellsize)
        rec, dl, _ = region_metrics(m, R ** 3)
        ref_sigmas.append(rec["sigma"]); ref_peaks.append(rec["n_peaks_gt10"])
        ref_pdfs.append(pdf_hist(dl, edges))
    ref_sigma = float(np.mean(ref_sigmas))
    ref_peak = float(np.mean(ref_peaks))
    ref_pdf = np.mean(ref_pdfs, axis=0)
    print(f"training-HR reference: sigma={ref_sigma:.3f}  peaks>10={ref_peak:.0f}")

    all_rows = []
    for ri in range(int(args.regions)):
        o_hr = [int(x) for x in rng.integers(0, Nhr, size=3)]
        o_hr = [(v // scale) * scale for v in o_hr]          # align to the LR lattice
        a_lr = [(v - B) // scale for v in o_hr]
        y = wrapped_block(lr, a_lr, S, channels=len(uc))
        y_t = torch.from_numpy(y)[None].to(device)

        # truth for this region, with the validated buffer
        m_true, _ = cic_block_into_region(hr, o_hr, R, B, Nhr, args.dis_norm, cellsize)
        true_rec, true_delta, _ = region_metrics(m_true, R ** 3)
        true_pdf = pdf_hist(true_delta, edges)
        true_disp = wrapped_block(hr, o_hr, R) * dscale
        true_def, _ = deformation_stats(true_disp)

        for k in range(int(args.K)):
            temp = args.temperatures[k % len(args.temperatures)]
            g = torch.Generator(device="cpu").manual_seed(args.seed + 9973 * ri + k)
            z = torch.randn(1, len(uc), S * scale, S * scale, S * scale, generator=g)
            z = (z * temp).to(device)
            with torch.no_grad():
                x = model.generate(y_t, n_steps=args.n_steps, z=z)[0].cpu().numpy()
            mass = region_density_from_window(x, B, R, dscale)
            rec, delta, _ = region_metrics(mass, R ** 3)
            defr, _ = deformation_stats(
                np.ascontiguousarray(x[:, B:B + R, B:B + R, B:B + R]) * dscale)
            pdf = pdf_hist(delta, edges)

            # ORACLE score (uses test HR): lower is better
            oracle = (abs(np.log(max(rec["sigma"], 1e-6) / max(true_rec["sigma"], 1e-6)))
                      + abs(np.log(max(rec["pk_highk"], 1e-30)
                                   / max(true_rec["pk_highk"], 1e-30)))
                      + np.abs(pdf - true_pdf).sum()
                      + abs(rec["n_peaks_gt10"] - true_rec["n_peaks_gt10"])
                      / max(true_rec["n_peaks_gt10"], 1)
                      + abs(defr["frac_detJ_negative"] - true_def["frac_detJ_negative"])
                      / max(true_def["frac_detJ_negative"], 1e-6))

            # DEPLOYABLE score: training-HR statistics + self-consistency only
            deployable = (abs(np.log(max(rec["sigma"], 1e-6) / ref_sigma))
                          + np.abs(pdf - ref_pdf).sum()
                          + abs(rec["n_peaks_gt10"] - ref_peak) / max(ref_peak, 1))

            all_rows.append({"region": ri, "k": k, "temperature": temp,
                             "oracle_score": float(oracle),
                             "deployable_score": float(deployable),
                             "sigma": rec["sigma"], "sigma_true": true_rec["sigma"],
                             "peaks": rec["n_peaks_gt10"],
                             "peaks_true": true_rec["n_peaks_gt10"],
                             "frac_detJ_negative": defr["frac_detJ_negative"]})
            print(f"  region {ri} k {k:>3} T={temp:.2f}  sigma={rec['sigma']:.3f} "
                  f"(true {true_rec['sigma']:.3f})  oracle={oracle:.4f} "
                  f"deploy={deployable:.4f}", flush=True)

    # ---- best-of-K curves -------------------------------------------------- #
    from cosmo_sr.tts.bootstrap import best_of_k

    rows = np.array([(r["region"], r["oracle_score"], r["deployable_score"])
                     for r in all_rows])
    curves = {"K": [], "random": [], "oracle": [], "selected": []}
    Ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= args.K]
    for K in Ks:
        rand, orac, sel = [], [], []
        for ri in range(int(args.regions)):
            v = rows[rows[:, 0] == ri]
            o, d = v[:, 1], v[:, 2]
            # best_of_k reports `values` and MINIMISES `selector`; both our scores
            # are already "lower is better", so they are passed through unnegated.
            m0, _ = best_of_k(o, np.zeros_like(o), K)   # constant selector = random pick
            rand.append(float(m0))
            m1, _ = best_of_k(o, o, K)                  # oracle picks by the oracle score
            orac.append(float(m1))
            m2, _ = best_of_k(o, d, K)                  # selector picks by deployable score
            sel.append(float(m2))
        curves["K"].append(K)
        curves["random"].append(float(np.mean(rand)))
        curves["oracle"].append(float(np.mean(orac)))
        curves["selected"].append(float(np.mean(sel)))

    print(f"\n{'K':>4} {'random':>10} {'oracle':>10} {'selected':>10}  (lower=better)")
    for i, K in enumerate(curves["K"]):
        print(f"{K:>4} {curves['random'][i]:>10.4f} {curves['oracle'][i]:>10.4f} "
              f"{curves['selected'][i]:>10.4f}")
    gain = curves["random"][0] - curves["oracle"][-1]
    print(f"\noracle gain from K=1 to K={curves['K'][-1]}: {gain:.4f} "
          f"({100 * gain / max(curves['random'][0], 1e-9):.1f}% of the single-sample score)")
    print("GATE: if this gain is small, the generator lacks good candidates -- "
          "stop selector development (Branch C territory).")

    with open(out / "best_of_k.json", "w") as f:
        json.dump({"curves": curves, "rows": all_rows,
                   "reference": {"sigma": ref_sigma, "peaks": ref_peak}}, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for nm in ("random", "oracle", "selected"):
            ax.plot(curves["K"], curves[nm], "o-", label=nm)
        ax.set_xscale("log", base=2); ax.set_xlabel("K candidates")
        ax.set_ylabel("score (lower = better)"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("test-time best-of-K")
        fig.tight_layout(); fig.savefig(out / "best_of_k.png", dpi=120); plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"\nWrote {out}/best_of_k.json, best_of_k.png")


if __name__ == "__main__":
    main()
