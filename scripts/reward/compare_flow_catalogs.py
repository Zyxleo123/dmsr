#!/usr/bin/env python
"""Compare flow-cascade Rockstar catalogs against the frozen HR and base ones.

Reads the ``catalog_cache`` metadata JSONs written by
``scripts/reward/catalog_summaries.py`` (one per source), re-parses each ASCII
catalog for its halo-mass function, and emits a compact summary table + a
host/subhalo mass-function overlay. Pure read-of-artifacts: no halo finding, no
GPU, so the figure is cheap to redraw.

    python scripts/reward/compare_flow_catalogs.py \
        --cache /zfsauton/scratch/yixiz/DMSR/dmsr_reward/catalog_cache \
        --box set15 \
        --candidate flow_cascade --candidate flow_unet_cascade \
        --out /zfsauton/scratch/yixiz/DMSR/dmsr_reward/flow_rockstar/set15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _meta_path(cache: Path, box: str, source: str, tag: str) -> Path:
    return cache / f"{box}__{source}__{tag}.json"


def _load_entry(cache: Path, box: str, source: str, tag: str, label: str):
    meta_p = _meta_path(cache, box, source, tag)
    if not meta_p.is_file():
        print(f"  (skip {label}: missing {meta_p.name})", flush=True)
        return None
    meta = json.loads(meta_p.read_text())
    from cosmo_sr.eval.rockstar import load_rockstar_ascii
    cat_path = Path(meta["catalog"])
    if not cat_path.is_file():
        print(f"  (skip {label}: catalog ascii gone: {cat_path})", flush=True)
        return None
    cat = load_rockstar_ascii(str(cat_path))
    hosts = cat.hosts()
    subs = cat.subhalos()
    return {
        "label": label,
        "source": source,
        "tag": tag,
        "n_halos": int(cat.n),
        "n_hosts": int(hosts.n),
        "n_subs": int(subs.n),
        "host_mvir": np.asarray(hosts.mvir, dtype=np.float64),
        "sub_mvir": np.asarray(subs.mvir, dtype=np.float64),
        "field": meta.get("field", ""),
    }


def _mass_function(mvir: np.ndarray, edges: np.ndarray) -> np.ndarray:
    mvir = mvir[np.isfinite(mvir) & (mvir > 0)]
    hist, _ = np.histogram(np.log10(mvir), bins=edges)
    return hist.astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="catalog_cache dir")
    ap.add_argument("--box", default="set15")
    ap.add_argument("--candidate", action="append", default=[],
                    help="candidate tag(s) under source=candidate; repeatable")
    ap.add_argument("--out", required=True, help="output dir for summary + plot")
    args = ap.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plan = [("HR (truth)", "hr", "hr"), ("base (SR2)", "base", "base")]
    for tag in args.candidate:
        plan.append((f"flow:{tag}", "candidate", tag))

    entries = []
    for label, source, tag in plan:
        e = _load_entry(cache, args.box, source, tag, label)
        if e is not None:
            entries.append(e)
    if not entries:
        raise SystemExit("no catalogs found to compare")

    # Common log-mass edges spanning all host masses.
    all_hosts = np.concatenate([e["host_mvir"] for e in entries])
    all_hosts = all_hosts[np.isfinite(all_hosts) & (all_hosts > 0)]
    lo = np.floor(np.log10(all_hosts.min()) * 2) / 2
    hi = np.ceil(np.log10(all_hosts.max()) * 2) / 2
    edges = np.arange(lo, hi + 0.25, 0.25)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ref = next((e for e in entries if e["source"] == "hr"), entries[0])
    table = []
    for e in entries:
        e["host_hist"] = _mass_function(e["host_mvir"], edges).tolist()
        e["sub_hist"] = _mass_function(e["sub_mvir"], edges).tolist()
        row = {
            "label": e["label"], "source": e["source"], "tag": e["tag"],
            "n_halos": e["n_halos"], "n_hosts": e["n_hosts"], "n_subs": e["n_subs"],
            "hosts_vs_hr": e["n_hosts"] / max(ref["n_hosts"], 1),
            "subs_vs_hr": e["n_subs"] / max(ref["n_subs"], 1),
            "host_median_log10mvir": float(np.median(
                np.log10(e["host_mvir"][e["host_mvir"] > 0]))),
        }
        table.append(row)

    summary = {
        "box": args.box,
        "reference": ref["label"],
        "log10_mass_edges": edges.tolist(),
        "table": table,
        "host_mass_function": {e["tag"]: e["host_hist"] for e in entries},
        "sub_mass_function": {e["tag"]: e["sub_hist"] for e in entries},
    }
    (out / "flow_catalog_comparison.json").write_text(json.dumps(summary, indent=2))

    # ---- table to stdout ----------------------------------------------------
    print(f"\n=== {args.box} Rockstar catalog comparison "
          f"(reference: {ref['label']}) ===", flush=True)
    hdr = f"{'catalog':<20}{'halos':>9}{'hosts':>9}{'subs':>9}" \
          f"{'hosts/HR':>10}{'subs/HR':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in table:
        print(f"{r['label']:<20}{r['n_halos']:>9}{r['n_hosts']:>9}{r['n_subs']:>9}"
              f"{r['hosts_vs_hr']:>10.3f}{r['subs_vs_hr']:>10.3f}")

    # ---- mass-function overlay ---------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for kind, ax, key in (("host", axes[0], "host_hist"),
                              ("subhalo", axes[1], "sub_hist")):
            for e in entries:
                h = np.asarray(e[key], dtype=float)
                style = "-" if e["source"] == "hr" else (
                    "--" if e["source"] == "base" else ":")
                lw = 2.5 if e["source"] == "hr" else 1.8
                ax.step(centers, np.where(h > 0, h, np.nan), where="mid",
                        label=e["label"], ls=style, lw=lw)
            ax.set_yscale("log")
            ax.set_xlabel(r"$\log_{10} M_{\rm vir}\ [M_\odot/h]$")
            ax.set_ylabel("count / bin")
            ax.set_title(f"{kind} mass function")
            ax.legend(fontsize=8)
        fig.suptitle(f"{args.box}: flow vs base vs HR (full-box Rockstar)")
        fig.tight_layout()
        fig.savefig(out / "flow_mass_function.png", dpi=120)
        plt.close(fig)
        print(f"\n-> {out / 'flow_mass_function.png'}", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"(plot skipped: {exc})", flush=True)

    print(f"-> {out / 'flow_catalog_comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
