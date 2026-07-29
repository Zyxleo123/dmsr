#!/usr/bin/env python
"""Stages 4 and 5: verifier-guided noise refinement, and globally coherent tiling.

Both stages emit rows in the same schema as ``scripts/eval_srs_tts.py``, so
``scripts/tts_final_table.py`` can put every method in one table.

``--mode refine`` (Stage 4)
    Per box: draw ``--k`` independent noise collections, score them with the
    trained verifier, keep the best ``--keep``, then optimise each one's noise
    with SR2 and the verifier frozen. Refinement is coarse -> middle -> fine and
    pays a prior term that keeps ``z`` near N(0, 1); runs whose noise leaves the
    training distribution are recorded as ``rejected`` and excluded from
    selection. A cross-entropy-method arm (``--cem``) provides the gradient-free
    control: if it matches gradient refinement, the gain came from search, not
    from gradients.

    Refinement runs **per tile** -- backpropagating through 512 tiles at once is
    not memory-feasible -- against a linear, differentiable surrogate of the
    verifier (:class:`cosmo_sr.tts.refine.LinearFeatureObjective`). Histogram
    features have zero gradient and are therefore excluded from the surrogate;
    they are still used to *score* the final candidate.

``--mode global`` (Stage 5)
    Per box: build overlapping tiles fed by one coordinate-indexed global noise
    field per candidate, score each tile's candidates with the verifier, choose
    the combination that minimises

        sum_i S_verifier(x_i) + lam_overlap * sum_(i,j) ||x_i - x_j||^2_overlap

    by coordinate descent, and only then blend. Cropping/blending before
    selection would average away the small-scale variance the candidates differ
    in and call the result an improvement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _load_verifier(path: Path, device):
    import torch
    from cosmo_sr.tts.verifier import FeatureRanker, Standardizer

    ckpt = torch.load(path / "verifier.pt", map_location="cpu", weights_only=False)
    std = Standardizer.load(path / "standardizer.json")
    model = FeatureRanker(len(ckpt["keys"]), hidden=ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, std, list(ckpt["keys"])


def _linear_weights(model, keys) -> Optional[Dict[str, float]]:
    """Weights of a purely linear ranker, or ``None`` if it has hidden layers."""
    linear = [m for m in model.net if m.__class__.__name__ == "Linear"]
    if len(linear) != 1:
        return None
    w = linear[0].weight.detach().cpu().numpy().reshape(-1)
    return {k: float(v) for k, v in zip(keys, w)}


def _score_candidate(model, std, keys, feats: Dict[str, float]) -> float:
    import torch
    from cosmo_sr.tts.features import feature_matrix

    x = std(feature_matrix([feats], keys))
    with torch.no_grad():
        return float(model(torch.as_tensor(x, dtype=torch.float32))[0])


def run_refine(args) -> None:
    import torch
    from cosmo_sr.tts.features import (
        DIFFERENTIABLE_FEATURE_KEYS, HRReference, candidate_features,
        differentiable_features,
    )
    from cosmo_sr.tts.metrics import DensityGeometry, candidate_metrics, cic_density_slabs
    from cosmo_sr.tts.refine import (
        LinearFeatureObjective, NoiseRegularizer, cem_refine_tile, default_schedule,
        noise_statistics, refine_tile_noise,
    )
    from cosmo_sr.tts.sampling import iter_srs_candidates, tile_noise, tile_starts
    from cosmo_sr.tts.srs_noise import load_controlled_generator
    from cosmo_sr.data.crops import periodic_crop

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    G = load_controlled_generator(args.model, scale_factor=args.scale, device=device,
                                  chan_base=args.chan_base, chan_min=args.chan_min,
                                  chan_max=args.chan_max)
    for p in G.parameters():
        p.requires_grad_(False)                     # SR2 stays frozen throughout

    model, std, keys = _load_verifier(Path(args.verifier), device)
    weights = _linear_weights(model, keys)
    if weights is None and not args.cem:
        raise SystemExit(
            "gradient refinement needs a linear verifier (train with --hidden), "
            "or pass --cem for the gradient-free arm"
        )
    geometry = DensityGeometry(boxsize=args.boxsize, ng=args.scale * 64, dis_norm=args.dis_norm)
    reference = HRReference.load(Path(args.hr_reference)) if args.hr_reference else None

    rows_path = out / "rows.jsonl"
    for name in args.boxes:
        lr_np = np.load(Path(args.lr) / f"{name}.npy").astype(np.float32)
        hr = torch.from_numpy(np.load(Path(args.hr) / f"{name}.npy").astype(np.float32))
        hr = hr.unsqueeze(0).to(device)
        lr_dev = torch.from_numpy(lr_np).unsqueeze(0).to(device)
        n_hr = hr.shape[-1]
        geo = geometry.for_grid(n_hr)
        with torch.no_grad():
            rho_hr = cic_density_slabs(hr[:, 0:3], geo.cellsize, geo.dis_norm, slab=args.slab)
        tile_hr = (lr_np.shape[-1] // args.nsplit) * args.scale

        # --- 1. sample K, score with the verifier, keep the best few --------- #
        scored: List[tuple] = []
        with torch.no_grad():
            for cand in iter_srs_candidates(
                G, lr_np, list(range(args.k)), nsplit=args.nsplit, pad=args.pad,
                scale_factor=args.scale, device=device, box=name, model_path=args.model,
            ):
                sr = torch.from_numpy(cand.field).unsqueeze(0).to(device)
                feats = candidate_features(sr, lr_dev, factor=args.scale, geometry=geometry,
                                           reference=reference, tile_size=tile_hr,
                                           n_bins=args.n_bins, slab=args.slab)
                scored.append((_score_candidate(model, std, keys, feats), int(cand.seed)))
                del sr
        scored.sort()
        kept = [seed for _s, seed in scored[: args.keep]]
        print(f"[{name}] verifier kept seeds {kept} of {args.k}", flush=True)

        # --- 2. refine each kept realisation, tile by tile ------------------- #
        chunk = lr_np.shape[-1] // args.nsplit
        lr_size = chunk + 2 * args.pad
        starts = tile_starts(lr_np.shape[-1], args.nsplit)
        if args.max_tiles:
            starts = starts[: args.max_tiles]

        for seed in kept:
            t0 = time.time()
            box = np.zeros((6, n_hr, n_hr, n_hr), dtype=np.float32)
            traj_all, stats_all, n_rejected = [], [], 0
            for start in starts:
                crop = periodic_crop(lr_np, start, chunk, pad=args.pad)
                x = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0).to(device)
                z0 = tile_noise(seed, start, lr_size, args.scale, device, pad=args.pad)
                # the LR block this tile is responsible for (unpadded), so the
                # operator-consistency feature compares against the real input
                lr_centre = torch.from_numpy(
                    np.ascontiguousarray(periodic_crop(lr_np, start, chunk))
                ).float().unsqueeze(0).to(device)
                tgt = chunk * args.scale

                def feature_fn(sr_tile, _lr=lr_centre, _tgt=tgt):
                    w = (sr_tile.shape[-1] - _tgt) // 2
                    core = sr_tile[..., w:w + _tgt, w:w + _tgt, w:w + _tgt]
                    return differentiable_features(
                        core, _lr, factor=args.scale, geometry=geometry,
                        reference=reference, slab=args.slab,
                    )

                if args.cem:
                    obj = LinearFeatureObjective(
                        {k: v for k, v in (weights or {}).items()
                         if k in DIFFERENTIABLE_FEATURE_KEYS},
                        dict(zip(keys, std.mean)), dict(zip(keys, std.std)), feature_fn)
                    res = cem_refine_tile(G, x, z0, lambda y: float(obj(y)),
                                          iterations=args.cem_iters,
                                          population=args.cem_pop, seed=seed)
                else:
                    obj = LinearFeatureObjective(
                        {k: v for k, v in weights.items() if k in DIFFERENTIABLE_FEATURE_KEYS},
                        dict(zip(keys, std.mean)), dict(zip(keys, std.std)), feature_fn)
                    res = refine_tile_noise(
                        G, x, z0, obj,
                        schedule=default_schedule(args.steps, args.step_lr),
                        regularizer=NoiseRegularizer(args.lam_mu, args.lam_sigma, args.lam_l2),
                    )
                n_rejected += int(res.rejected)
                traj_all.append(res.trajectory[-1] - res.trajectory[0])
                stats_all.append(res.stats.get("max_dist", 0.0))
                noise = z0 if res.rejected else res.noise
                with torch.no_grad():
                    y = G(x, noise=noise).squeeze(0)
                w = (y.shape[-1] - tgt) // 2
                hs = tuple(s * args.scale for s in start)
                box[:, hs[0]:hs[0] + tgt, hs[1]:hs[1] + tgt, hs[2]:hs[2] + tgt] = (
                    y[..., w:w + tgt, w:w + tgt, w:w + tgt].cpu().numpy()
                )

            sr = torch.from_numpy(box).unsqueeze(0).to(device)
            with torch.no_grad():
                m = candidate_metrics(sr, hr, lr_dev, factor=args.scale, geometry=geometry,
                                      n_bins=args.n_bins, tile_size=tile_hr, rho_hr=rho_hr)
                m.update(candidate_features(sr, lr_dev, factor=args.scale, geometry=geometry,
                                            reference=reference, tile_size=tile_hr,
                                            n_bins=args.n_bins, slab=args.slab))
            row = {
                "box": name, "seed": int(seed),
                "method": "cem_refine" if args.cem else "grad_refine",
                "wall_s": round(time.time() - t0, 2),
                "refine_delta_mean": float(np.mean(traj_all)),
                "refine_noise_dist_mean": float(np.mean(stats_all)),
                "refine_rejected_tiles": int(n_rejected),
                "refine_n_tiles": len(starts),
                **{k: float(v) for k, v in m.items()},
            }
            with rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{name}] seed {seed} refined in {row['wall_s']}s "
                  f"({n_rejected}/{len(starts)} tiles rejected)", flush=True)
            del sr
        del hr, lr_dev, rho_hr
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"wrote {rows_path}")


def run_global(args) -> None:
    import torch
    from cosmo_sr.tts.features import HRReference, candidate_features
    from cosmo_sr.tts.metrics import DensityGeometry, candidate_metrics, cic_density_slabs
    from cosmo_sr.tts.srs_noise import load_controlled_generator
    from cosmo_sr.tts.tiling import (
        TileGrid, generate_tiles, overlap_pairs, select_tiles_coordinate_descent,
        stitch_overlapping,
    )

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    G = load_controlled_generator(args.model, scale_factor=args.scale, device=device,
                                  chan_base=args.chan_base, chan_min=args.chan_min,
                                  chan_max=args.chan_max)
    model, std, keys = _load_verifier(Path(args.verifier), device)
    geometry = DensityGeometry(boxsize=args.boxsize, ng=args.scale * 64, dis_norm=args.dis_norm)
    reference = HRReference.load(Path(args.hr_reference)) if args.hr_reference else None
    rows_path = out / "rows.jsonl"

    for name in args.boxes:
        t0 = time.time()
        lr_np = np.load(Path(args.lr) / f"{name}.npy").astype(np.float32)
        hr = torch.from_numpy(np.load(Path(args.hr) / f"{name}.npy").astype(np.float32))
        hr = hr.unsqueeze(0).to(device)
        lr_dev = torch.from_numpy(lr_np).unsqueeze(0).to(device)
        ng = lr_np.shape[-1]
        grid = TileGrid(ng=ng, chunk=args.chunk, stride=args.stride, pad=args.pad,
                        scale=args.scale)
        geo = geometry.for_grid(hr.shape[-1])
        with torch.no_grad():
            rho_hr = cic_density_slabs(hr[:, 0:3], geo.cellsize, geo.dis_norm, slab=args.slab)

        per_seed = [
            generate_tiles(G, lr_np, grid, seed=s, device=device, noise_mode="global")
            for s in range(args.k)
        ]
        starts = grid.starts()
        # tile-level verifier scores; a tile is scored exactly like a small box
        verifier: Dict[tuple, np.ndarray] = {}
        for s in starts:
            scores = []
            for tiles in per_seed:
                tile = tiles[s].unsqueeze(0)
                lr_tile = torch.nn.functional.avg_pool3d(tile, args.scale)
                feats = candidate_features(tile, lr_tile, factor=args.scale,
                                           geometry=geometry, reference=reference,
                                           n_bins=args.n_bins, slab=args.slab)
                scores.append(_score_candidate(model, std, keys, feats))
            verifier[s] = np.asarray(scores)

        pairs = overlap_pairs(grid)
        disagreement: Dict[tuple, np.ndarray] = {}
        d_hr = grid.stride * grid.scale
        for (a, b) in pairs:
            axis = next(i for i in range(3) if a[i] != b[i])
            mat = np.zeros((args.k, args.k))
            for i in range(args.k):
                for j in range(args.k):
                    ta = per_seed[i][a].movedim(axis + 1, 1)[:, d_hr:]
                    tb = per_seed[j][b].movedim(axis + 1, 1)[:, : grid.tile_hr - d_hr]
                    mat[i, j] = float((ta - tb).pow(2).mean())
            disagreement[(a, b)] = mat

        choice, traj = select_tiles_coordinate_descent(
            verifier, disagreement, lam_overlap=args.lam_overlap, max_sweeps=args.sweeps
        )
        chosen = {s: per_seed[choice[s]][s] for s in starts}
        box = stitch_overlapping(chosen, grid, mode=args.stitch).unsqueeze(0)

        with torch.no_grad():
            m = candidate_metrics(box, hr, lr_dev, factor=args.scale, geometry=geometry,
                                  n_bins=args.n_bins, tile_size=grid.stride * grid.scale,
                                  rho_hr=rho_hr)
            m.update(candidate_features(box, lr_dev, factor=args.scale, geometry=geometry,
                                        reference=reference,
                                        tile_size=grid.stride * grid.scale,
                                        n_bins=args.n_bins, slab=args.slab))
        row = {
            "box": name, "seed": -1, "method": "global_joint",
            "wall_s": round(time.time() - t0, 2),
            "joint_score_start": float(traj[0]), "joint_score_end": float(traj[-1]),
            "n_tiles": len(starts), "k": args.k, "lam_overlap": args.lam_overlap,
            **{k: float(v) for k, v in m.items()},
        }
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{name}] joint selection {traj[0]:.4f} -> {traj[-1]:.4f} "
              f"in {row['wall_s']}s", flush=True)
        del hr, lr_dev, rho_hr, per_seed, chosen, box
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"wrote {rows_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("refine", "global"), required=True)
    ap.add_argument("--lr", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm/lr")
    ap.add_argument("--hr", default="/zfsauton/scratch/yixiz/DMSR/paired_catnorm/hr")
    ap.add_argument("--model", default=str(_ROOT / "external" / "SRS-map2map" / "SRmodel" / "G_z0.pt"))
    ap.add_argument("--verifier", default="runs/tts_verifier")
    ap.add_argument("--hr-reference", default="runs/tts_oracle/hr_reference.npz")
    ap.add_argument("--boxes", nargs="*", default=["set12", "set13", "set14", "set15"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--nsplit", type=int, default=8)
    ap.add_argument("--pad", type=int, default=3)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--slab", type=int, default=32)
    ap.add_argument("--boxsize", type=float, default=100000.0)
    ap.add_argument("--dis-norm", type=float, default=6000.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chan-base", type=int, default=512)
    ap.add_argument("--chan-min", type=int, default=64)
    ap.add_argument("--chan-max", type=int, default=512)
    ap.add_argument("--out", default="runs/tts_stage4")
    # refinement
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--step-lr", type=float, default=0.05)
    ap.add_argument("--lam-mu", type=float, default=1.0)
    ap.add_argument("--lam-sigma", type=float, default=1.0)
    ap.add_argument("--lam-l2", type=float, default=0.1)
    ap.add_argument("--cem", action="store_true", help="gradient-free control arm")
    ap.add_argument("--cem-iters", type=int, default=6)
    ap.add_argument("--cem-pop", type=int, default=12)
    ap.add_argument("--max-tiles", type=int, default=0, help="debug: refine only N tiles")
    # global tiling
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--lam-overlap", type=float, default=1.0)
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--stitch", choices=("crop", "blend"), default="blend")
    args = ap.parse_args()

    if args.mode == "refine":
        run_refine(args)
    else:
        run_global(args)


if __name__ == "__main__":
    main()
