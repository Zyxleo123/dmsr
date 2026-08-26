"""Build the projected substructure target x1 = Pi(Psi_HR - Psi_SR2) for a box.

This is the target-precompute of ``docs/sr2_moment_constraint.md`` section 5.1:
the one whole-box point (of two) where the per-host affine-moment projector Pi
enters training. It

1. generates the frozen SR2 displacement field over the whole box, tile by tile
   (the same generator and tiling the pilot's step-4 overfit uses);
2. forms the residual against the paired HR field, ``Psi_HR - Psi_SR2``;
3. builds Pi from ``<box>_lagrangian_host.npz`` and projects the residual, so
   inside every host footprint the affine part -- translation, dilation, rotation
   and shear of the host -- is removed and only substructure remains;
4. writes the target as the training cache, plus the diagnostics that answer *is
   the target correct?* (``moment_target_diag`` module), which the CPU renderer
   turns into a figure without recomputing anything.

GPU, because step 1 runs the generator. Everything after is CPU arithmetic on the
whole-box field; it is kept in this job only so the 3.2 GB field is formed once.
Resumable: an existing target is skipped unless ``MT_FORCE=1``.

    python scripts/features/build_moment_target.py --box set8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _sr2_direct import (  # noqa: E402
    add_direct_args, geometry_of, load_direct_config, load_hr, load_lr,
    model_path_of,
)
from cosmo_sr.data.preprocess_srs import disnorm  # noqa: E402
from cosmo_sr.features.lagrangian_host import LagrangianHostFeatures  # noqa: E402
from cosmo_sr.features.moment_constraint import from_features  # noqa: E402
from cosmo_sr.features.moment_target_diag import (  # noqa: E402
    host_slice_panels, offfootprint_max_abs_diff, per_host_moment_rows,
)
from cosmo_sr.train.sr2_finetune_data import (  # noqa: E402
    tile_lr_crop, tile_noise_stack, trim_to_tile,
)
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402

BOXSIZE = 100.0
NG_HR = 512
TILE = 64
DX = BOXSIZE / NG_HR


def reward_root() -> Path:
    return Path(os.environ.get(
        "DMSR_REWARD_ROOT", "/zfsauton/scratch/yixiz/DMSR/dmsr_reward"))


def features_path(box: str) -> Path:
    return reward_root() / "lagrangian_host" / box / f"{box}_lagrangian_host.npz"


# --------------------------------------------------------------------------- #
# Whole-box SR2 generation
# --------------------------------------------------------------------------- #
def generate_sr2_box(model, lr, geom, device, seed: int, batch: int) -> np.ndarray:
    """``(6, 512, 512, 512)`` frozen SR2 displacement+velocity, in on-disk units.

    Same per-tile forward as ``overfit_host_mse.forward_tiles``, batched over the
    512 tiles and scattered into the box. Kept in on-disk (normalised) units so
    the residual and target are exactly what the model emits and the flow will
    consume; unit conversion happens only for the human-readable slices.
    """
    n = NG_HR // TILE
    lr_np = np.asarray(lr)
    box = np.empty((6, NG_HR, NG_HR, NG_HR), dtype=np.float32)
    tiles = list(range(n ** 3))
    t0 = time.time()
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        lr_b = torch.stack([
            torch.from_numpy(np.ascontiguousarray(tile_lr_crop(lr_np, t, geom)))
            .float() for t in chunk]).to(device)
        noise_list = [tile_noise_stack([seed], t, geom, device=device) for t in chunk]
        keys = list(noise_list[0].keys())
        noise = {k: torch.cat([nl[k] for nl in noise_list], dim=0) for k in keys}
        with torch.no_grad():
            out = trim_to_tile(model(lr_b, noise=noise), geom).float().cpu().numpy()
        for j, t in enumerate(chunk):
            ix, iy, iz = t // (n * n), (t // n) % n, t % n
            box[:, ix * TILE:(ix + 1) * TILE, iy * TILE:(iy + 1) * TILE,
                iz * TILE:(iz + 1) * TILE] = out[j]
        if (i // batch) % 8 == 0:
            done = i + len(chunk)
            print(f"    tiles {done}/{len(tiles)}  ({time.time() - t0:.0f}s)",
                  flush=True)
    return box


def to_mpc_disp(disp_ondisk: np.ndarray) -> np.ndarray:
    """Displacement channels, on-disk units -> Mpc/h. For the slices only."""
    return (disnorm(disp_ondisk.astype(np.float64), z=0.0, undo=True) * 1e-3)


# --------------------------------------------------------------------------- #
# Diagnostic spectra (windowed, on a crop -- respects the pilot's leakage lesson)
# --------------------------------------------------------------------------- #
def crop_around(field: np.ndarray, centre, m: int) -> np.ndarray:
    """``(C, m, m, m)`` crop centred on ``centre`` with periodic wrap."""
    sh = tuple(m // 2 - int(c) for c in centre)
    rolled = np.roll(field, shift=sh, axis=(1, 2, 3))
    return rolled[:, :m, :m, :m]


def windowed_radial_power(disp_crop: np.ndarray):
    """Hann-windowed radial |disp|^2(k) of a ``(3, m, m, m)`` crop.

    A crop of the box is not periodic, so its bare FFT leaks the coherent bulk
    flow into every k (pilot step 3's bug). The Hann window removes it, which is
    exactly what lets this spectrum show the projection stripping the low-k bulk
    inside the host while the high-k substructure survives.
    """
    m = disp_crop.shape[-1]
    w = np.hanning(m)
    win = np.einsum("i,j,k->ijk", w, w, w)
    power = np.zeros((m, m, m // 2 + 1))
    for c in range(3):
        power += np.abs(np.fft.rfftn(disp_crop[c].astype(np.float64) * win)) ** 2
    kx = np.fft.fftfreq(m) * m
    kz = np.fft.rfftfreq(m) * m
    kk = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)
    kb = np.round(kk).astype(int).ravel()
    nb = int(kb.max()) + 1
    prof = (np.bincount(kb, weights=power.ravel(), minlength=nb)
            / np.maximum(np.bincount(kb, minlength=nb), 1))
    k_hmpc = np.arange(nb) * (2.0 * np.pi / (m * DX))
    return k_hmpc, prof


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_direct_args(ap)
    ap.add_argument("--box", default=os.environ.get("MT_BOX", "set8"))
    ap.add_argument("--mode", default=os.environ.get("MT_MODE", "affine"),
                    choices=["affine", "translation"])
    ap.add_argument("--seed", type=int, default=int(os.environ.get("MT_SEED", "0")))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("MT_BATCH", "8")))
    ap.add_argument("--dtype", default=os.environ.get("MT_DTYPE", "float16"),
                    choices=["float16", "float32"])
    ap.add_argument("--top-hosts", type=int,
                    default=int(os.environ.get("MT_TOP_HOSTS", "24")))
    ap.add_argument("--crop", type=int, default=int(os.environ.get("MT_CROP", "128")))
    ap.add_argument("--force", action="store_true",
                    default=os.environ.get("MT_FORCE", "0") == "1")
    ap.add_argument("--device", default=os.environ.get("DEVICE", ""))
    args = ap.parse_args()

    box = args.box
    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = reward_root() / "moment_target" / box
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / f"{box}_moment_target.npy"
    print(f"[moment-target] box={box} mode={args.mode} device={device}", flush=True)

    if target_path.is_file() and not args.force:
        print(f"  target exists ({target_path}); nothing to do. "
              "MT_FORCE=1 to rebuild target + diagnostics.", flush=True)
        return

    # --- fields --------------------------------------------------------------
    feat_path = features_path(box)
    if not feat_path.is_file():
        raise SystemExit(
            f"no host features at {feat_path}; run submit_lagrangian_host.sh first")
    hr = np.asarray(load_hr(cfg, box), dtype=np.float32)
    if hr.shape != (6, NG_HR, NG_HR, NG_HR):
        raise SystemExit(f"HR field is {hr.shape}, expected (6,512,512,512)")

    print("  generating SR2 field ...", flush=True)
    model = load_controlled_generator(
        model_path_of(cfg), in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    for p in model.parameters():
        p.requires_grad_(False)
    lr = load_lr(cfg, box)
    sr2 = generate_sr2_box(model, lr, geom, device, args.seed, args.batch)
    residual = (hr - sr2).astype(np.float32)

    # --- project -------------------------------------------------------------
    print("  building Pi and projecting ...", flush=True)
    feat = LagrangianHostFeatures.from_npz(str(feat_path))
    proj = from_features(feat, mode=args.mode)
    target = proj.apply(residual)

    np.save(target_path, target.astype(np.dtype(args.dtype)))
    print(f"  wrote {target_path} ({target.nbytes / 1e9 * (2 if args.dtype=='float32' else 1):.1f} GB "
          f"as {args.dtype})", flush=True)

    # --- diagnostics ---------------------------------------------------------
    print("  diagnostics ...", flush=True)
    present = np.array([b.row for b in proj.blocks], dtype=int)
    mvir = feat.table.mvir[present]
    order = present[np.argsort(-mvir)]
    top = order[: args.top_hosts]
    rows = per_host_moment_rows(proj, residual, target, rows=top.tolist())
    off = offfootprint_max_abs_diff(proj, residual, target)

    # slice + spectra around the most massive host
    top_row = int(order[0])
    centre_mpc = feat.table.center_lag[top_row]
    centre_site = tuple(int(np.floor(c / DX)) % NG_HR for c in centre_mpc)
    fields = {"hr": hr, "sr2": sr2, "residual": residual, "target": target}
    # convert displacement magnitudes to Mpc/h for legibility
    panels = host_slice_panels(
        {k: np.concatenate([to_mpc_disp(v[0:3]), v[3:6]], axis=0)
         for k, v in fields.items()},
        centre_site, NG_HR, half=args.crop // 2, axis=2)

    res_crop = crop_around(to_mpc_disp(residual[0:3]), centre_site, args.crop)
    tgt_crop = crop_around(to_mpc_disp(target[0:3]), centre_site, args.crop)
    k_hmpc, p_res = windowed_radial_power(res_crop)
    _, p_tgt = windowed_radial_power(tgt_crop)

    np.savez_compressed(
        out_dir / f"{box}_moment_target_diag.npz",
        **{f"panel_{k}": v.astype(np.float32) for k, v in panels.items()},
        k_hmpc=k_hmpc, power_residual=p_res, power_target=p_tgt,
        centre_site=np.array(centre_site))

    summary = {
        "box": box, "mode": args.mode, "seed": args.seed,
        "n_hosts_total": int(present.size),
        "offfootprint_max_abs_diff": off,
        "offfootprint_ok": bool(off == 0.0),
        "global": {
            "rms_residual_disp": float(np.sqrt(np.mean(residual[0:3] ** 2))),
            "rms_target_disp": float(np.sqrt(np.mean(target[0:3] ** 2))),
            "frac_sites_in_footprint": float(proj.footprint_mask().mean()),
        },
        "top_host": {
            "row": top_row,
            "halo_id": int(feat.table.host_id[top_row]),
            "log_mvir": float(np.log10(max(feat.table.mvir[top_row], 1.0))),
            "centre_site": [int(c) for c in centre_site],
        },
        "per_host": [
            {"row": r.row, "n_sites": r.n_sites,
             "log_mvir": float(np.log10(max(feat.table.mvir[r.row], 1.0))),
             "rms_before": r.rms_before, "rms_after": r.rms_after,
             "moment_norm_before": r.moment_norm_before,
             "moment_norm_after": r.moment_norm_after,
             "affine_var_frac": r.affine_var_frac}
            for r in rows],
        "verdict": _verdict(rows, off),
    }
    (out_dir / f"{box}_moment_target_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("  " + summary["verdict"], flush=True)
    print(f"  summary: {out_dir / f'{box}_moment_target_summary.json'}", flush=True)


def _verdict(rows, off: float) -> str:
    if not rows:
        return "NO HOSTS: the projector is empty; check the features npz."
    worst = max(r.moment_norm_after for r in rows)
    ratio = max(r.moment_norm_after / max(r.moment_norm_before, 1e-30) for r in rows)
    med_removed = float(np.median([r.affine_var_frac for r in rows]))
    ok = worst < 1e-4 and off == 0.0
    tag = "PASS" if ok else "CHECK"
    return (f"{tag}: post-projection moment norm <= {worst:.2e} "
            f"(<= {ratio:.1e} of before), off-footprint diff {off:.1e}; "
            f"median {med_removed:.0%} of footprint displacement variance was "
            f"affine and was removed.")


if __name__ == "__main__":
    main()
