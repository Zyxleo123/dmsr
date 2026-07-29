"""map2map (SRS-map2map SR-GAN, Ni et al.) baseline visualisation.

Renders the SAME figures as dmsr_run_viz.py -- density chain / residue / T(k),r(k)
transfer / q+Psi 3-component displacement (SR | GT | SR-GT) -- but for the
pretrained SRS generator ``external/SRS-map2map/SRmodel/G_z0.pt`` instead of one
of our flows. The baseline SR does not depend on any t13 checkpoint, so this is
computed ONCE into a shared directory (default runs/dmsr/figures/baseline_map2map/)
at both 64^3 and 128^3.

  python scripts/dmsr_baseline_viz.py                 # default t13 config + shared out
  python scripts/dmsr_baseline_viz.py --config runs/dmsr/t13_critic_s0/config.yaml

The SRS generator uses valid (unpadded) convs and takes the full 6-channel LR
(displacement+velocity), so we load the 6-ch crop, super-resolve it SRS-style
(periodic pad + narrow, one tile), and compare the displacement channels (0..2)
against the HR crop -- the same channels our flows model. NOTE: a single small
crop is periodic-padded rather than given true neighbour context, matching the
tiled-inference approximation in compare_flow_baseline.py / reproduce_srs.py.
"""
from __future__ import annotations
import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
# Import our (map2map-free) modules FIRST; the SRS fork hijacks `import map2map`
# and must be the first map2map imported in the process (baseline_srs enforces it).
from cosmo_sr.utils.config import load_config                       # noqa: E402
from cosmo_sr.dmsr.data import build_val_dataset, resolve_split     # noqa: E402
from cosmo_sr.dmsr.density import HighPassDensity, cellsizes        # noqa: E402
from cosmo_sr.dmsr.operator import NullSpaceOperator                # noqa: E402

# reuse the shared figure panels
_spec = importlib.util.spec_from_file_location("rv", "scripts/dmsr_run_viz.py")
rv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rv)


def _first_t13_config() -> str:
    for c in sorted(glob.glob("runs/dmsr/t13_*/config.yaml")):
        return c
    raise FileNotFoundError("no runs/dmsr/t13_*/config.yaml found; pass --config")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="a run's config.yaml (for data globs / factor / dis_norm); "
                         "default: first runs/dmsr/t13_*/config.yaml")
    ap.add_argument("--srs-model", default="external/SRS-map2map/SRmodel/G_z0.pt")
    ap.add_argument("--out", default="runs/dmsr/figures/baseline_map2map")
    ap.add_argument("--res", nargs="*", type=int, default=[64, 128])
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--crop", type=int, default=-1, help="crop index; -1 = auto-pick")
    ap.add_argument("--scan", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mmap", action="store_false", dest="mmap",
                    help="(kept for parity) memory-map instead of RAM-load")
    ap.set_defaults(mmap=True)
    args = ap.parse_args()

    cfg_path = args.config or _first_t13_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cfg = load_config(cfg_path)
    dcfg = cfg.get("data", {})
    factor = int(cfg.get("factor", 8))
    n_all = int(dcfg.get("channels", 6))
    label = "map2map (SRS G_z0)"

    op = NullSpaceOperator(factor=factor).to(device)
    hr_cs, lr_cs = cellsizes(dcfg, factor)
    _dn = float(cfg.get("critic", {}).get("dis_norm", dcfg.get("dis_norm", 6000.0)))
    highpass = HighPassDensity(factor=factor, lowpass="blockavg", cellsize=hr_cs, dis_norm=_dn)
    highpass_lr = HighPassDensity(factor=factor, lowpass="blockavg", cellsize=lr_cs, dis_norm=_dn)
    dens = lambda f: highpass.density(f)
    dens_lr = lambda f: highpass_lr.density(f)

    # SRS generator (triggers the SRS map2map fork import; must be after our imports)
    from cosmo_sr.eval.baseline_srs import load_srs_generator  # noqa: E402
    print(f"[baseline] loading SRS generator {args.srs_model} ...")
    srs_G = load_srs_generator(args.srs_model, scale_factor=factor, device=device)

    split = resolve_split(dcfg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    res_to_crop = {64: 8, 128: 16}
    for res in args.res:
        crop_lr = res_to_crop.get(res)
        if crop_lr is None:
            print(f"[baseline] unknown res {res}; skipping"); continue
        suffix = "" if res == 64 else f"_{res}"
        try:
            _render_res(out, label, split, op, dens, dens_lr, factor, _dn,
                        n_all, crop_lr, suffix, args, device, srs_G)
        except Exception as e:  # never let one resolution kill the other
            print(f"[baseline] res {res} failed: {type(e).__name__}: {e}")
    print(f"[baseline] wrote figures to {out}/")


def _center(t, size):
    """Center-crop the last three dims of a (…,N,N,N) tensor/array to ``size``."""
    n = t.shape[-1]
    s = (n - size) // 2
    if s < 0:
        raise ValueError(f"cannot center-crop size {n} to larger {size}")
    return t[..., s:s + size, s:s + size, s:s + size]


def _srs_region(gen, lr_np, tgt_hr, device, seed):
    """Run the SRS generator on one LR region (6,L,L,L) and center-narrow the
    valid-conv output (factor*L - 42) to ``tgt_hr``. Real neighbour context is
    already inside the region, so no periodic self-padding is needed."""
    t = torch.from_numpy(np.ascontiguousarray(lr_np)).float()[None].to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        sr = gen(t).squeeze(0).cpu().numpy()          # (6, factor*L-42, ...)
    return _center(sr, tgt_hr)


def _render_res(out, label, split, op, dens, dens_lr, factor, dis_norm,
                n_all, crop_lr, suffix, args, device, srs_G):
    # SRS uses valid (unpadded) convs: HR = factor*L - 42. To produce a
    # crop_lr*factor HR block we must feed an LR *context region* of side
    # L = crop_lr + 2*ceil(21/factor) (= crop_lr + 6 for factor 8) and then
    # center-narrow. We crop that L-region from the box (real neighbours) and
    # center-align everything (GT, LR panel, A+(y)) to the same target block.
    import math
    pad_cells = math.ceil((42 / factor) / 2)          # 3 for factor 8
    L = crop_lr + 2 * pad_cells                        # 14 (64^3), 22 (128^3)
    tgt_hr = crop_lr * factor
    print(f"[baseline] crop_lr={crop_lr} -> HR {tgt_hr}^3 (SRS LR context region {L}^3)")

    # 6-channel L-regions (SRS needs displacement+velocity); use_channels=None keeps all.
    ds = build_val_dataset(split, crop_lr=L, scale_factor=factor,
                           channels=n_all, use_channels=None, mmap=args.mmap,
                           max_crops=max(args.scan, args.crop + 1), which=args.split)
    if args.crop >= 0:
        idx = min(args.crop, len(ds) - 1)
    else:
        best, idx = -1.0, 0
        for i in range(min(args.scan, len(ds))):     # score the CENTER target block
            hrc = _center(ds[i]["hr"][:3].unsqueeze(0), tgt_hr).to(device)
            s = float(dens(hrc).std())
            if s > best:
                best, idx = s, i
        print(f"[baseline]   auto-picked crop {idx} (density std {best:.3f})")
    sample = ds[idx]
    lr_L = sample["lr"]                                # (6,L,L,L)
    hr_L = sample["hr"]                                # (6,L*factor,...)

    sr6 = _srs_region(srs_G, lr_L.cpu().numpy(), tgt_hr, device, args.seed)  # (6,tgt,tgt,tgt)
    x_sr3 = torch.from_numpy(np.ascontiguousarray(sr6[:3]))[None].float().to(device)
    x_hr3 = _center(hr_L[:3].unsqueeze(0), tgt_hr).to(device)
    y6 = _center(lr_L.unsqueeze(0), crop_lr).to(device)          # center LR crop for panels
    x_base = op.A_plus(y6[:, :3])

    tag = f"{args.split} crop {idx}"
    rv.fig_density_chain(out, label, y6, x_base, x_sr3, x_hr3, dens, dens_lr, tag, suffix)
    rv.fig_density_residue(out, label, x_base, x_sr3, x_hr3, dens, tag, suffix)
    rv.fig_transfer_rk(out, label, x_base, x_sr3, x_hr3, factor, suffix)
    rv.fig_qpsi(out, label, x_sr3, x_hr3, dis_norm, tag, suffix)

    def d_rms(a, b): return float(((dens(a) - dens(b)) ** 2).mean().sqrt())
    print(f"[baseline]   density-residual RMS: A+(y) {d_rms(x_base, x_hr3):.4f}  "
          f"SRS {d_rms(x_sr3, x_hr3):.4f}")


if __name__ == "__main__":
    main()
