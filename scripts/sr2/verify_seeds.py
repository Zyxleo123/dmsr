#!/usr/bin/env python
"""Verify nested seeds share the LR condition but differ in noise (Stage 0).

Checks on one LR box (default set14):
1. Same seed → bit-identical SR fields.
2. Different seeds → nonzero RMS difference (disp and vel separately).
3. Nested property: seeds 0..N-1 for N=4 are the first four of N=8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", default=str(ROOT / "configs/sr2_baseline/freeze.yaml"))
    ap.add_argument("--box", default="set14")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--out", default=str(ROOT / "runs/sr2_baseline/seed_verify.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke-ng", type=int, default=0,
                    help="If >0, use a random LR cube of this size instead of a real box")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.freeze).read_text())
    from cosmo_sr.tts.srs_noise import load_controlled_generator, ControlledG
    from cosmo_sr.tts.sampling import super_resolve_srs_seeded

    inf = cfg["inference"]
    nsplit, pad = int(inf["nsplit"]), int(inf["pad"])
    scale = int(cfg["model"]["scale_factor"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    if args.smoke_ng:
        # Tiny ControlledG with inflated noise for a CPU unit check.
        # Match tests/tts/test_sampling.py geometry: ng=8, nsplit=4, pad=3
        # (nsplit=8 → chunk=1 is too small for valid-conv SR2).
        torch.manual_seed(0)
        G = ControlledG(6, 6, scale, chan_base=16, chan_min=8, chan_max=16).eval().to(device)
        with torch.no_grad():
            for name, p in G.named_parameters():
                if name.endswith(".std"):
                    p.copy_(torch.ones_like(p) * 0.3)
        lr = np.random.default_rng(0).normal(
            size=(6, args.smoke_ng, args.smoke_ng, args.smoke_ng)
        ).astype(np.float32)
        nsplit = 4 if args.smoke_ng % 4 == 0 else 1
        while nsplit > 1 and args.smoke_ng % nsplit:
            nsplit -= 1
        if args.smoke_ng // nsplit < 2:
            raise SystemExit(
                f"smoke-ng={args.smoke_ng} too small for pad={pad}; use --smoke-ng 8"
            )
    else:
        from cosmo_sr.data.field_io import load_field
        G = load_controlled_generator(
            str(ROOT / cfg["model"]["path"]), scale_factor=scale, device=device,
        )
        lr = load_field(str(Path(cfg["data"]["root"]) / "lr" / f"{args.box}.npy")).astype(
            np.float32
        )

    fields = {}
    for s in seeds:
        fields[s] = super_resolve_srs_seeded(
            G, lr, s, scale_factor=scale, nsplit=nsplit, pad=pad, device=device,
            noise_mode=inf.get("noise_mode", "per_tile"),
        )

    # Re-run seed 0 for bit-identity
    again = super_resolve_srs_seeded(
        G, lr, seeds[0], scale_factor=scale, nsplit=nsplit, pad=pad, device=device,
        noise_mode=inf.get("noise_mode", "per_tile"),
    )
    identical = bool(np.array_equal(fields[seeds[0]], again))

    def rel_rms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a ** 2)), 1e-30))

    pairs = {}
    for i, s0 in enumerate(seeds):
        for s1 in seeds[i + 1:]:
            pairs[f"{s0}_vs_{s1}"] = {
                "disp_rel_rms": rel_rms(fields[s0][0:3], fields[s1][0:3]),
                "vel_rel_rms": rel_rms(fields[s0][3:6], fields[s1][3:6]),
            }

    # Nested: regenerating first k seeds matches the stored ones
    nested_ok = True
    for k in range(1, len(seeds) + 1):
        for s in seeds[:k]:
            regen = super_resolve_srs_seeded(
                G, lr, s, scale_factor=scale, nsplit=nsplit, pad=pad, device=device,
                noise_mode=inf.get("noise_mode", "per_tile"),
            )
            if not np.array_equal(regen, fields[s]):
                nested_ok = False

    report = {
        "box": args.box if not args.smoke_ng else f"smoke_ng={args.smoke_ng}",
        "seeds": seeds,
        "same_seed_bit_identical": identical,
        "nested_prefix_stable": nested_ok,
        "pairwise": pairs,
        "all_pairs_differ": all(
            p["disp_rel_rms"] > 0 or p["vel_rel_rms"] > 0 for p in pairs.values()
        ),
        "nsplit": nsplit,
        "pad": pad,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    ok = identical and nested_ok and report["all_pairs_differ"]
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
