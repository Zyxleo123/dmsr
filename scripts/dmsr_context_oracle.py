#!/usr/bin/env python
"""Stage 3: does the central prediction depend on LR context outside the 8^3 crop?

For a FIXED central ``tgt``^3 LR region we run inference with progressively wider
LR context windows centred on the same region, holding the checkpoint and the
stochastic noise fixed, and score the identical central ``tgt*scale``^3 HR block.
If the central output keeps changing as the context grows, the training crop is
truncating information the model would otherwise use.

Why the noise must be *anchored*
--------------------------------
``NullSpaceFlow.sample_residual`` draws ``z ~ N(0, I)`` at HR resolution. Drawing
it per-window would make every context size differ by noise alone. We instead
draw ONE full-box ``z`` and slice each window's HR footprint out of it, so the
central region receives byte-identical noise at every context size and the only
thing that varies is how much surrounding LR the network saw.

Two known reasons the central output can move (both measured here)
------------------------------------------------------------------
1. **Convolutional receptive field** -- the ordinary, intended mechanism.
2. **GroupNorm** -- ``nn.GroupNorm`` reduces over (C/g, D, H, W), so every output
   voxel depends on the mean/variance of the WHOLE window. This makes the
   effective dependency crop-global regardless of kernel reach, and it means a
   window of a different size renormalises the features differently. ``--report-gn``
   dumps the per-layer GroupNorm statistics so the two effects can be told apart.

Modes
-----
``--mode velocity``
    ONE velocity-net evaluation at fixed ``t`` and fixed ``r_t``. 20x cheaper than
    a full ODE and enough to establish whether the network's central output is
    context-dependent at all. Runs on CPU for small windows.
``--mode generate``
    The full ``n_steps`` ODE, i.e. the actual sampler. Needs a GPU beyond ~16^3 LR.

Usage
-----
    python scripts/dmsr_context_oracle.py \
        --config runs/dmsr/t13_unc_fulldisp_pshuffle8_l003_s0/config.yaml \
        --ckpt   runs/dmsr/t13_unc_fulldisp_pshuffle8_l003_s0/ckpt_best.pt \
        --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy \
        --hr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr/set14.npy \
        --contexts 8 12 16 24 32 --mode generate --out runs/dmsr/stage3_context
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dmsr_cic_buffer_audit import cic_into_region, region_metrics  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def wrapped_block(arr, start, size, channels=3):
    """``arr[:, start:start+size, ...]`` with periodic wrap on the 3 spatial axes.

    Indexes the memmap plane-by-plane rather than materialising ``arr`` first: the
    HR field is 1.6 GiB and a whole-array ``np.asarray`` is the single biggest
    avoidable allocation in this script.
    """
    n = arr.shape[-1]
    i1 = np.mod(np.arange(start[1], start[1] + size), n)
    i2 = np.mod(np.arange(start[2], start[2] + size), n)
    outp = np.empty((channels, size, size, size), dtype=np.float32)
    for s in range(size):
        p = int((start[0] + s) % n)
        outp[:, s] = np.asarray(arr[0:channels, p], dtype=np.float32)[:, i1][:, :, i2]
    return outp


# --------------------------------------------------------------------------- #
# position-anchored noise
# --------------------------------------------------------------------------- #
_SM64_A = np.uint64(0x9E3779B97F4A7C15)
_SM64_B = np.uint64(0xBF58476D1CE4E5B9)
_SM64_C = np.uint64(0x94D049BB133111EB)


def _splitmix64(x):
    z = (x + _SM64_A).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(30))) * _SM64_B).astype(np.uint64)
    z = ((z ^ (z >> np.uint64(27))) * _SM64_C).astype(np.uint64)
    return (z ^ (z >> np.uint64(31))).astype(np.uint64)


def anchored_noise(start, size, channels, ngrid_hr, seed):
    """Standard-normal noise indexed by ABSOLUTE periodic HR coordinates.

    ``sample_residual`` would otherwise draw fresh ``z`` per window, so two context
    sizes would differ by noise alone. Drawing one full-box ``z`` and slicing it is
    the obvious fix but costs 1.6 GiB at 512^3; instead each voxel's value is a
    pure function of ``(channel, i, j, k)`` and ``seed`` via a counter-based hash,
    so any window sees identical values on the overlap with O(window) memory.
    Generated plane-by-plane to keep the uint64 temporaries small.
    """
    out = np.empty((channels, size, size, size), dtype=np.float32)
    j = np.mod(np.arange(start[1], start[1] + size, dtype=np.int64), ngrid_hr).astype(np.uint64)
    k = np.mod(np.arange(start[2], start[2] + size, dtype=np.int64), ngrid_hr).astype(np.uint64)
    jj, kk = np.meshgrid(j, k, indexing="ij")
    # uint64 products wrap by design (that IS the mixing step), so silence overflow
    with np.errstate(over="ignore"):
        base_jk = (jj * np.uint64(0x100000001B3)) ^ (kk * np.uint64(0x9E3779B1))
        for c in range(channels):
            cc = np.uint64(seed) + np.uint64(c) * np.uint64(0xA24BAED4963EE407)
            for s in range(size):
                i = np.uint64(int((start[0] + s) % ngrid_hr))
                key = (base_jk ^ (i * np.uint64(0xD6E8FEB86659FD93)) ^ cc).astype(np.uint64)
                h1 = _splitmix64(key)
                h2 = _splitmix64(key ^ np.uint64(0x5DEECE66D))
                u1 = (h1 >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
                u2 = (h2 >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
                u1 = np.clip(u1, 1e-12, 1.0)
                out[c, s] = (np.sqrt(-2.0 * np.log(u1))
                             * np.cos(2.0 * np.pi * u2)).astype(np.float32)
    return out


def deformation_stats(disp_cells):
    """det J and folding statistics for ``(3, N, N, N)`` displacement in HR cells.

    ``J = I + d(Psi)/dq`` is the local deformation matrix on the Lagrangian
    lattice (spacing 1 cell after the conversion to cell units). ``det J`` small
    or negative means strong compression or shell-crossing (folding) -- the
    signature of collapse. Central differences, periodic within the block.
    """
    g = np.empty((3, 3) + disp_cells.shape[1:], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            g[i, j] = 0.5 * (np.roll(disp_cells[i], -1, axis=j) - np.roll(disp_cells[i], 1, axis=j))
    J = g.copy()
    for i in range(3):
        J[i, i] += 1.0
    det = (
        J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
        - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
        + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0])
    )
    div = g[0, 0] + g[1, 1] + g[2, 2]
    return {
        "detJ_mean": float(det.mean()),
        "detJ_median": float(np.median(det)),
        "detJ_p01": float(np.quantile(det, 0.01)),
        "detJ_p99": float(np.quantile(det, 0.99)),
        "frac_detJ_negative": float((det < 0).mean()),
        "frac_detJ_lt_0p1": float((det < 0.1).mean()),
        "div_rms": float(np.sqrt((div ** 2).mean())),
        "div_min": float(div.min()),
    }, det


def band_rms(a, nb=3):
    """RMS of a scalar field in ``nb`` equal radial k-bands (low -> high)."""
    n = a.shape[-1]
    fk = np.fft.rfftn(a)
    kx = np.fft.fftfreq(n) * n
    kz = np.fft.rfftfreq(n) * n
    km = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    kmax = n / 2
    out = []
    for b in range(nb):
        m = (km >= b * kmax / nb) & (km < (b + 1) * kmax / nb)
        out.append(float(np.sqrt((np.abs(fk[m]) ** 2).sum()) / n ** 1.5))
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--hr", required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    ap.add_argument("--tgt", type=int, default=8, help="central LR region scored (LR cells)")
    ap.add_argument("--origin", type=int, nargs=3, default=None,
                    help="central region lower corner in LR cells (default: box centre)")
    ap.add_argument("--mode", choices=["velocity", "generate"], default="velocity")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--t-fixed", type=float, default=0.5, help="velocity mode: ODE time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--boxsize", type=float, default=100000.0)
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--cic-buffer", type=int, default=64,
                    help="true-HR CIC buffer in HR cells; Stage 2 measured 64 as "
                         "converged for a 64^3 scored region on set14")
    ap.add_argument("--report-gn", action="store_true",
                    help="dump per-layer GroupNorm input statistics per context size")
    ap.add_argument("--out", default="runs/dmsr/stage3_context")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  mode={args.mode}")

    from cosmo_sr.utils.config import load_config
    from dmsr_eval import load_flow

    cfg = load_config(args.config)
    uc = cfg.get("data", {}).get("use_channels") or [0, 1, 2]
    scale = int(cfg.get("factor", 8))
    model = load_flow(cfg, len(uc), args.ckpt, device, use_ema=True)
    model.eval()

    lr = np.load(args.lr).astype(np.float32)[uc]        # (3, 64, 64, 64)
    Ng = lr.shape[-1]
    hr = np.load(args.hr, mmap_mode="r")
    Nhr = hr.shape[-1]
    cellsize = args.boxsize / Nhr
    disp_scale = args.dis_norm / cellsize
    tgt = int(args.tgt)
    o = args.origin or [(Ng - tgt) // 2] * 3
    print(f"LR box {Ng}^3, scale {scale}, central region {tgt}^3 LR at {o} "
          f"-> {tgt * scale}^3 HR at {[x * scale for x in o]}")

    contexts = sorted(set(args.contexts))
    for S in contexts:
        if (S - tgt) % 2:
            raise SystemExit(f"context {S} and tgt {tgt} must have even margin")
        if S > Ng:
            raise SystemExit(f"context {S} exceeds LR box {Ng}")

    print(f"anchored noise: position-hashed (seed {args.seed}), materialised per window")

    # --- GroupNorm statistics probe ------------------------------------------ #
    gn_stats = {}
    hooks = []
    if args.report_gn:
        import torch.nn as nn

        def mk(name):
            def hook(mod, inp, _out):
                x = inp[0].detach()
                b, c = x.shape[0], x.shape[1]
                xg = x.reshape(b, mod.num_groups, c // mod.num_groups, -1)
                gn_stats.setdefault(name, []).append(
                    (float(xg.mean()), float(xg.var(dim=(2, 3), unbiased=False).mean())))
            return hook

        for n_, m_ in model.named_modules():
            if isinstance(m_, nn.GroupNorm):
                hooks.append(m_.register_forward_hook(mk(n_)))

    results = {}
    central = {}
    for S in contexts:
        m = (S - tgt) // 2
        a = [o[d] - m for d in range(3)]                       # LR window origin (may be <0)
        y = wrapped_block(lr, a, S, channels=len(uc))          # (3, S, S, S)
        y_t = torch.from_numpy(y)[None].to(device)

        hstart = [x * scale for x in a]
        z = torch.from_numpy(
            anchored_noise(hstart, S * scale, len(uc), Ng * scale, args.seed)
        )[None].to(device)

        if args.report_gn:
            gn_stats.clear()
        t0 = time.time()
        with torch.no_grad():
            if args.mode == "velocity":
                tt = torch.full((1,), float(args.t_fixed), device=device)
                x = model.velocity(z, tt, y_t)
            else:
                x = model.generate(y_t, n_steps=args.n_steps, z=z)
        dt = time.time() - t0
        x = x[0].cpu().numpy().astype(np.float32)              # (3, S*scale, ...)

        h = m * scale
        c = x[:, h:h + tgt * scale, h:h + tgt * scale, h:h + tgt * scale]
        central[S] = np.ascontiguousarray(c)
        rec = {"context_lr": S, "margin_lr": m, "seconds": dt,
               "central_rms": float(np.sqrt((c ** 2).mean()))}
        if args.report_gn:
            rec["groupnorm"] = {k: v[0] for k, v in gn_stats.items()}
        results[f"ctx{S}"] = rec
        print(f"  [ctx {S:>3}^3 -> HR {S*scale}^3] {dt:6.1f}s  "
              f"central RMS {rec['central_rms']:.6f}", flush=True)

    for h_ in hooks:
        h_.remove()

    # --- convergence vs the widest context ----------------------------------- #
    ref_S = contexts[-1]
    ref = central[ref_S]
    ref_rms = float(np.sqrt((ref ** 2).mean()))
    print(f"\nreference context = {ref_S}^3 LR")
    print(f"{'context':>9} {'relRMS_vs_ref':>14} {'r_vs_ref':>9} "
          f"{'dlow':>8} {'dmid':>8} {'dhigh':>8}")
    for S in contexts:
        d = central[S] - ref
        rec = results[f"ctx{S}"]
        rec["rel_rms_vs_ref"] = float(np.sqrt((d ** 2).mean()) / max(ref_rms, 1e-12))
        rec["corr_vs_ref"] = float(np.corrcoef(central[S].ravel(), ref.ravel())[0, 1])
        bands = band_rms(d[0])
        ref_bands = band_rms(ref[0])
        rec["band_rel_change"] = [float(b / max(r, 1e-12)) for b, r in zip(bands, ref_bands)]
        print(f"{S:>9} {rec['rel_rms_vs_ref']:>14.5f} {rec['corr_vs_ref']:>9.5f} "
              + " ".join(f"{v:>8.4f}" for v in rec["band_rel_change"]))

    # --- deformation + density on the identical central block ---------------- #
    # Density uses the *true HR* field outside the central block as the CIC buffer
    # (Stage 2 showed a 64^3 crop needs a ~115 HR-cell buffer). Holding the buffer
    # fixed and true means any density change across context sizes is caused by
    # the central prediction alone, not by the model's own buffer improving.
    R = tgt * scale
    ho = [x * scale for x in o]
    b_ref = int(args.cic_buffer)
    print(f"\ndensity: central {R}^3 block embedded in TRUE HR with buffer {b_ref} HR cells")
    side = R + 2 * b_ref
    lo = [ho[d] - b_ref for d in range(3)]

    def buffered_mass(central_cells, slab=16):
        """CIC into the scored cube from the padded block, slab by slab.

        The central ``R^3`` sub-block is overwritten with ``central_cells`` (the
        model's prediction) while the buffer stays at the TRUE HR field. Holding
        the buffer fixed and true is what makes the density differences across
        context sizes attributable to the central prediction alone, rather than to
        the model's own surroundings getting better.
        """
        mass = np.zeros((R, R, R), dtype=np.float64)
        i1 = np.mod(np.arange(lo[1], lo[1] + side), Nhr)
        i2 = np.mod(np.arange(lo[2], lo[2] + side), Nhr)
        q1 = np.arange(lo[1], lo[1] + side, dtype=np.float64) + 0.5
        q2 = np.arange(lo[2], lo[2] + side, dtype=np.float64) + 0.5
        for s in range(0, side, slab):
            e = min(s + slab, side)
            blk = np.empty((3, e - s, side, side), dtype=np.float32)
            for t_ in range(s, e):
                p = int((lo[0] + t_) % Nhr)
                blk[:, t_ - s] = np.asarray(hr[0:3, p], dtype=np.float32)[:, i1][:, :, i2]
            blk *= disp_scale
            if central_cells is not None:
                for t_ in range(s, e):
                    if b_ref <= t_ < b_ref + R:
                        blk[:, t_ - s, b_ref:b_ref + R, b_ref:b_ref + R] = \
                            central_cells[:, t_ - b_ref]
            q0 = np.arange(lo[0] + s, lo[0] + e, dtype=np.float64) + 0.5
            pos = np.empty((3, e - s, side, side), dtype=np.float64)
            pos[0] = blk[0] + q0[:, None, None]
            pos[1] = blk[1] + q1[None, :, None]
            pos[2] = blk[2] + q2[None, None, :]
            del blk
            mass += cic_into_region(pos.reshape(3, -1), ho, R, Nhr)
            del pos
        return mass

    print(f"{'context':>9} {'sigma':>9} {'pk_highk':>11} {'peaks>10':>9} "
          f"{'detJ<0':>9} {'div_rms':>9}")
    for S in contexts:
        c_cells = central[S] * disp_scale
        mass = buffered_mass(c_cells)
        dm, _, _ = region_metrics(mass, R ** 3)
        ds, _ = deformation_stats(c_cells)
        results[f"ctx{S}"]["density"] = dm
        results[f"ctx{S}"]["deformation"] = ds
        print(f"{S:>9} {dm['sigma']:>9.4f} {dm['pk_highk']:>11.4g} "
              f"{dm['n_peaks_gt10']:>9} {ds['frac_detJ_negative']:>9.5f} "
              f"{ds['div_rms']:>9.4f}", flush=True)

    # --- true-HR reference for the same central block ------------------------ #
    true_c = wrapped_block(hr, ho, R) * disp_scale
    ds_true, _ = deformation_stats(true_c)
    dm_true, _, _ = region_metrics(buffered_mass(None), R ** 3)
    results["truth"] = {"density": dm_true, "deformation": ds_true}
    print(f"{'TRUTH':>9} {dm_true['sigma']:>9.4f} {dm_true['pk_highk']:>11.4g} "
          f"{dm_true['n_peaks_gt10']:>9} {ds_true['frac_detJ_negative']:>9.5f} "
          f"{ds_true['div_rms']:>9.4f}")

    meta = {"config": args.config, "ckpt": args.ckpt, "mode": args.mode,
            "contexts": contexts, "tgt": tgt, "origin": list(map(int, o)),
            "n_steps": args.n_steps, "seed": args.seed, "ref_context": ref_S}
    with open(out / "context_oracle.json", "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    np.savez_compressed(out / "central_fields.npz",
                        **{f"ctx{S}": central[S] for S in contexts})

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(contexts, [results[f"ctx{S}"]["rel_rms_vs_ref"] for S in contexts], "o-")
        ax[0].set_xlabel("LR context [cells]"); ax[0].set_ylabel(f"rel RMS vs {ref_S}^3")
        ax[0].set_title("central displacement change")
        ax[1].plot(contexts, [results[f"ctx{S}"]["density"]["sigma"] for S in contexts], "o-")
        ax[1].axhline(dm_true["sigma"], color="k", ls="--", label="truth")
        ax[1].set_xlabel("LR context [cells]"); ax[1].set_title("central density sigma")
        ax[1].legend(fontsize=8)
        ax[2].plot(contexts,
                   [results[f"ctx{S}"]["deformation"]["frac_detJ_negative"] for S in contexts], "o-")
        ax[2].axhline(ds_true["frac_detJ_negative"], color="k", ls="--", label="truth")
        ax[2].set_xlabel("LR context [cells]"); ax[2].set_title("folding fraction (det J < 0)")
        ax[2].legend(fontsize=8)
        for a_ in ax:
            a_.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "context_curves.png", dpi=120); plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"\nWrote {out}/context_oracle.json, central_fields.npz, context_curves.png")


if __name__ == "__main__":
    main()
