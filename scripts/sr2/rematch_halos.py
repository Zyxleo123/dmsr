#!/usr/bin/env python
"""Rematch existing Rockstar catalogs with the repaired host matcher.

Does not regenerate fields or re-run Rockstar. Writes a new match_rows JSONL
(and optional summary) under ``--out``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _find_ascii(d: Path) -> Path:
    for pat in ("halos*.ascii", "halos*.list"):
        hits = sorted(d.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no ascii catalog in {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--halos-root", required=True,
                    help=".../stage1/halos containing set*/")
    ap.add_argument("--boxes", default="set12")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--boxsize", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-records", action="store_true",
                    help="store per-subhalo records (large)")
    args = ap.parse_args()

    from cosmo_sr.eval.rockstar import load_rockstar_ascii
    from cosmo_sr.eval.halo_match import match_hosts, classify_subhalos

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    match_path = out / "match_rows.jsonl"
    if match_path.exists():
        match_path.unlink()

    boxes = [b.strip() for b in args.boxes.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    root = Path(args.halos_root)
    summary = []

    for box in boxes:
        hr_dir = root / box / "hr" / "hr_rockstar"
        hr_cat = load_rockstar_ascii(_find_ascii(hr_dir))
        for seed in seeds:
            sr_base = root / box / f"sr_seed{seed}"
            sr_dirs = list(sr_base.glob("sr*_rockstar"))
            if not sr_dirs:
                print(f"SKIP {box} seed={seed}: no catalog")
                continue
            sr_cat = load_rockstar_ascii(_find_ascii(sr_dirs[0]))
            hm = match_hosts(hr_cat, sr_cat, boxsize_mpc_h=args.boxsize)
            classes = classify_subhalos(hr_cat, sr_cat, hm, boxsize_mpc_h=args.boxsize)
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
            with open(match_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            summary.append(row)
            print(
                f"[{box} seed={seed}] host_match={row['host_match_rate']:.3f} "
                f"({n_matched}/{len(hm.hr_ids)}) classes={dict(counts)}",
                flush=True,
            )

    with open(out / "rematch_summary.json", "w") as fh:
        json.dump({"rows": summary, "match_rows": str(match_path)}, fh, indent=2)
    print(f"Wrote {match_path}")


if __name__ == "__main__":
    main()
