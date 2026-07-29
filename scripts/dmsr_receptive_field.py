#!/usr/bin/env python
"""Stage 4: measure the model's effective receptive field, both directions.

Input-centered
    Perturb ONE LR voxel and map the induced change across the HR output,
    ``E(x) = |Psi_perturbed(x) - Psi_original(x)|^2``. Repeated for LR voxels in
    halo / filament / sheet / void environments (classified by local LR density).

Output-centered
    Take a central HR block and differentiate its summed output with respect to
    every LR input voxel, ``|d sum(Psi_centre) / d y(i,j,k)|``.

Both are reported as radial response curves plus ``r50 / r95 / r99`` -- the radii
containing 50 / 95 / 99 percent of the response energy -- and as the fraction of
response reaching the window boundary (a value that is not small means the window
is clipping real dependence).

Why the normalization matters here
----------------------------------
``UNetResidualFlowModel`` is built with ``norm="group"``. ``nn.GroupNorm`` reduces
over ``(C/g, D, H, W)``, so every output voxel depends on the mean and variance of
the entire window: the effective receptive field is window-global *by
construction*, independent of kernel reach. A radius measured with GroupNorm
active therefore describes the window, not the architecture.

``--ablate-norm channel`` swaps every GroupNorm for a per-voxel channel-group
normalization (same affine parameters, statistics taken over channels only, never
over space). That isolates the convolutional receptive field. NOTE that this
changes the function the trained weights implement, so the ablation answers "how
local could this architecture be?" and NOT "how good is this model?".

Usage
-----
    python scripts/dmsr_receptive_field.py \
        --config runs/dmsr/t13_unconstrained_s0/config.yaml \
        --ckpt   runs/dmsr/t13_unconstrained_s0/ckpt_best.pt \
        --lr /zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr/set14.npy \
        --window 16 --n-steps 4 --out runs/dmsr/stage4_rf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dmsr_context_oracle import anchored_noise, wrapped_block  # noqa: E402


# --------------------------------------------------------------------------- #
class ChannelGroupNorm(nn.Module):
    """GroupNorm statistics taken over channels only -- never over space.

    Drop-in for ``nn.GroupNorm`` carrying the same ``weight``/``bias``. For input
    ``(B, C, D, H, W)`` and ``G`` groups the mean and variance are computed over
    the ``C/G`` channels of each group *at each voxel independently*, so the layer
    becomes strictly pointwise and cannot transport information across space.
    """

    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.num_groups = gn.num_groups
        self.num_channels = gn.num_channels
        self.eps = gn.eps
        self.weight = gn.weight
        self.bias = gn.bias

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        g = self.num_groups
        xs = x.reshape(b, g, c // g, *x.shape[2:])
        mean = xs.mean(dim=2, keepdim=True)
        var = xs.var(dim=2, keepdim=True, unbiased=False)
        xs = (xs - mean) / (var + self.eps).sqrt()
        x = xs.reshape(b, c, *x.shape[2:])
        if self.weight is not None:
            shape = (1, -1) + (1,) * (x.dim() - 2)
            x = x * self.weight.view(shape) + self.bias.view(shape)
        return x


def ablate_norms(module):
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, ChannelGroupNorm(child))
            n += 1
        else:
            n += ablate_norms(child)
    return n


# --------------------------------------------------------------------------- #
def radial_profile(energy, centre):
    """Mean response per integer radius, plus the r50/r95/r99 energy radii."""
    n = energy.shape[-1]
    ax = [np.abs(((np.arange(n) - c + n // 2) % n) - n // 2) for c in centre]
    r = np.sqrt(ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2
                + ax[2][None, None, :] ** 2)
    rr = r.ravel().astype(np.int64)
    e = energy.ravel().astype(np.float64)
    nb = int(rr.max()) + 1
    sums = np.bincount(rr, weights=e, minlength=nb)
    cnts = np.bincount(rr, minlength=nb)
    prof = sums / np.maximum(cnts, 1)
    cum = np.cumsum(sums)
    tot = cum[-1] if cum[-1] > 0 else 1.0
    q = {}
    for frac in (0.5, 0.95, 0.99):
        q[f"r{int(frac*100)}"] = float(np.searchsorted(cum / tot, frac))
    return prof, sums / tot, q


def classify_environment(lr_disp, k=4):
    """Crude environment label per LR voxel from the local Zel'dovich divergence.

    ``div Psi`` is negative where matter converges. Ranking voxels by it and
    cutting at quartiles gives halo (most convergent) through void (most
    divergent) without needing a halo finder -- enough to check whether the
    receptive field depends on environment.
    """
    div = sum(0.5 * (np.roll(lr_disp[i], -1, i) - np.roll(lr_disp[i], 1, i))
              for i in range(3))
    flat = div.ravel()
    order = np.argsort(flat)
    labels = np.empty(flat.shape, dtype=object)
    n = len(flat)
    names = ["halo", "filament", "sheet", "void"]
    for i, nm in enumerate(names):
        labels[order[i * n // 4:(i + 1) * n // 4]] = nm
    return labels.reshape(div.shape), div


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--window", type=int, default=16, help="LR window side")
    ap.add_argument("--n-steps", type=int, nargs="+", default=[4],
                    help="ODE step counts to test (response can grow with steps)")
    ap.add_argument("--eps", type=float, default=0.05,
                    help="LR perturbation amplitude, in units of the LR field std")
    ap.add_argument("--ablate-norm", choices=["none", "channel"], default="none")
    ap.add_argument("--environments", nargs="+",
                    default=["halo", "filament", "sheet", "void"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/dmsr/stage4_rf")
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

    n_ab = ablate_norms(model) if args.ablate_norm == "channel" else 0
    print(f"device={device}  norm ablation={args.ablate_norm} ({n_ab} GroupNorm replaced)")

    lr = np.load(args.lr).astype(np.float32)[uc]
    Ng = lr.shape[-1]
    S = int(args.window)
    a = [(Ng - S) // 2] * 3
    y0 = wrapped_block(lr, a, S, channels=len(uc))
    lr_std = float(y0.std())
    z = torch.from_numpy(
        anchored_noise([x * scale for x in a], S * scale, len(uc), Ng * scale, args.seed)
    )[None].to(device)
    y_t = torch.from_numpy(y0)[None].to(device)
    centre_lr = [S // 2] * 3
    centre_hr = [S * scale // 2] * 3

    labels, div = classify_environment(lr)
    results = {"meta": {"window": S, "scale": scale, "ablate_norm": args.ablate_norm,
                        "eps_rel": args.eps, "lr_std": lr_std,
                        "n_groupnorm_replaced": n_ab}}

    # ---------------- input-centered ---------------------------------------- #
    print(f"\n=== input-centered: perturb 1 LR voxel, eps={args.eps}*std ===")
    print(f"{'env':>9} {'steps':>6} {'r50':>5} {'r95':>5} {'r99':>5} "
          f"{'bnd_frac':>9} {'relRMS':>10}")
    for env in args.environments:
        # pick a voxel of this environment nearest the window centre
        cand = np.argwhere(labels == env)
        gc = np.array([a[d] + centre_lr[d] for d in range(3)])
        d2 = ((cand - gc) ** 2).sum(1)
        gsel = cand[np.argmin(d2)]
        lsel = [int((gsel[d] - a[d]) % Ng) for d in range(3)]
        if any(v >= S for v in lsel):
            print(f"  [{env}] no voxel of this class inside the window; skipped")
            continue
        for ns in args.n_steps:
            with torch.no_grad():
                base = model.generate(y_t, n_steps=ns, z=z)
                yp = y_t.clone()
                yp[0, :, lsel[0], lsel[1], lsel[2]] += args.eps * lr_std
                pert = model.generate(yp, n_steps=ns, z=z)
            e = ((pert - base) ** 2).sum(1)[0].cpu().numpy()
            rel = float(np.sqrt(e.mean()) / (base[0].pow(2).mean().sqrt().item() + 1e-12))
            hc = [v * scale + scale // 2 for v in lsel]
            prof, frac, q = radial_profile(e, hc)
            edge = np.zeros_like(e, dtype=bool)
            edge[0] = edge[-1] = True
            edge[:, 0] = edge[:, -1] = True
            edge[:, :, 0] = edge[:, :, -1] = True
            bnd = float(e[edge].sum() / max(e.sum(), 1e-30))
            key = f"input_{env}_steps{ns}"
            results[key] = {"env": env, "n_steps": ns, "lr_voxel": lsel,
                            "rel_rms_change": rel, "boundary_energy_frac": bnd, **q}
            np.save(out / f"radial_{key}.npy", prof)
            np.save(out / f"slice_{key}.npy", e[hc[0]])
            print(f"{env:>9} {ns:>6} {q['r50']:>5} {q['r95']:>5} {q['r99']:>5} "
                  f"{bnd:>9.4f} {rel:>10.3e}", flush=True)

    # ---------------- output-centered --------------------------------------- #
    print(f"\n=== output-centered: d(sum central HR block)/d(LR voxel) ===")
    blk = scale  # central HR block side = one LR cell's worth
    for ns in args.n_steps:
        yg = y_t.clone().requires_grad_(True)
        x = model.generate(yg, n_steps=ns, z=z)
        h0 = [c - blk // 2 for c in centre_hr]
        sel = x[0, :, h0[0]:h0[0] + blk, h0[1]:h0[1] + blk, h0[2]:h0[2] + blk]
        sel.pow(2).sum().backward()
        g = yg.grad[0].abs().sum(0).cpu().numpy()
        prof, frac, q = radial_profile(g ** 2, centre_lr)
        edge = np.zeros_like(g, dtype=bool)
        edge[0] = edge[-1] = True
        edge[:, 0] = edge[:, -1] = True
        edge[:, :, 0] = edge[:, :, -1] = True
        bnd = float((g[edge] ** 2).sum() / max((g ** 2).sum(), 1e-30))
        key = f"output_steps{ns}"
        results[key] = {"n_steps": ns, "boundary_energy_frac": bnd, **q}
        np.save(out / f"radial_{key}.npy", prof)
        np.save(out / f"slice_{key}.npy", g[centre_lr[0]])
        print(f"  steps={ns}: r50={q['r50']} r95={q['r95']} r99={q['r99']} "
              f"(LR cells)  boundary_frac={bnd:.4f}", flush=True)
        del yg, x, sel

    with open(out / "receptive_field.json", "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = [k for k in results if k.startswith("input_")]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for k in keys:
            p = np.load(out / f"radial_{k}.npy")
            ax[0].semilogy(np.arange(len(p)) / scale, np.maximum(p, 1e-30), label=k[6:])
        ax[0].set_xlabel("radius [LR cells]"); ax[0].set_ylabel("mean |dPsi|^2")
        ax[0].set_title("input-centered response"); ax[0].legend(fontsize=7)
        ax[0].axvline(S / 2, color="r", ls="--", lw=0.8)
        for k in [k for k in results if k.startswith("output_")]:
            p = np.load(out / f"radial_{k}.npy")
            ax[1].semilogy(np.arange(len(p)), np.maximum(p, 1e-30), label=k)
        ax[1].set_xlabel("radius [LR cells]"); ax[1].set_title("output-centered sensitivity")
        ax[1].axvline(S / 2, color="r", ls="--", lw=0.8); ax[1].legend(fontsize=7)
        for a_ in ax:
            a_.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "receptive_field.png", dpi=120); plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")

    print(f"\nWrote {out}/receptive_field.json, radial_*.npy, slice_*.npy, receptive_field.png")


if __name__ == "__main__":
    main()
