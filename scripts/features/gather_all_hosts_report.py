#!/usr/bin/env python
"""Pool the per-box all-hosts abundance JSONs into ONE cross-box result.

`gather_all_hosts_abundance` answers the question per box. The talk states one
number, so the arms have to be pooled across every gated box -- and pooled by
SUM, not by averaging the per-box percentages: set9 contributes 3775 hosts and
set10 3363, and a mean of ratios would silently weight them equally while
hiding that the denominators differ at all.

Only boxes where an arm actually has a gated catalog are pooled for that arm,
and the boxes are named in the output for every row, so a two-box arm is never
compared against a one-box arm without that being visible.

Emits the table three ways: stdout, a JSON summary, and a LaTeX ``tabular``
ready to paste into docs/subhalo_finetune_slides.tex.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402


def pool(boxes: List[Dict]) -> Dict[str, Dict]:
    """Sum every arm's counts over the boxes that actually gated it."""
    arms: Dict[str, Dict] = {}
    for b in boxes:
        n_bins = len(b["host_mass_edges_log10"]) - 1
        for name, s in b["arms"].items():
            a = arms.setdefault(name, {
                "boxes": [], "n_hosts": 0, "subs_total": 0, "hosts_alive": 0,
                "by_host_mass": [{"n_hosts": 0, "subs_total": 0}
                                 for _ in range(n_bins)],
                "mass_function": {k: 0 for k in b["mass_bin_labels"]},
            })
            a["boxes"].append(b["box"])
            # The host sample is HR's and is shared by every arm of the box, so
            # each arm accumulates that box's host count exactly once.
            a["n_hosts"] += b["n_hosts"]
            a["subs_total"] += s["subs_total"]
            a["hosts_alive"] += s["hosts_alive"]
            for k, v in s["mass_function"].items():
                a["mass_function"][k] = a["mass_function"].get(k, 0) + v
            for j, row in enumerate(s["by_host_mass"]):
                a["by_host_mass"][j]["n_hosts"] += row["n_hosts"]
                a["by_host_mass"][j]["subs_total"] += row["subs_total"]
    for a in arms.values():
        a["subs_per_host"] = a["subs_total"] / max(a["n_hosts"], 1)
        a["hosts_alive_frac"] = a["hosts_alive"] / max(a["n_hosts"], 1)
        for row in a["by_host_mass"]:
            row["subs_per_host"] = row["subs_total"] / max(row["n_hosts"], 1)
    return arms


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="the per-box <out>.json files")
    ap.add_argument("--baseline", default="frozen",
                    help="arm name to quote as the SR2 baseline; falls back to "
                         "'base' when the frozen control was not gated")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    boxes = []
    for p in a.inputs:
        d = json.loads(Path(p).read_text())
        if not d.get("ok"):
            print(f"skipping {p}: {d.get('reason', 'not ok')}")
            continue
        boxes.append(d)
    if not boxes:
        print("GATE: no usable per-box result -- the abundance stage did not run.")
        write_json(Path(a.out).with_suffix(".json"), {"ok": False})
        return 0

    arms = pool(boxes)
    ref = a.baseline if a.baseline in arms else "base"
    hr, bs = arms["hr"], arms[ref]
    order = (["hr", "base"] + ([ref] if ref not in ("hr", "base") else [])
             + sorted(k for k in arms if k not in ("hr", "base", ref)))

    print(f"boxes pooled: {', '.join(b['box'] for b in boxes)}")
    print(f"HR hosts (>= {boxes[0]['host_min_p']}p, in the edited region): "
          f"{hr['n_hosts']}")
    print(f"SR2 baseline for the ratios: {ref}\n")
    hdr = (f"{'arm':<28} {'boxes':>6} {'hosts':>7} {'subs':>8} {'/host':>7} "
           f"{'/HR':>7} {'recov':>7} {'hosts alive':>12}")
    print(hdr)
    print("-" * len(hdr))
    for name in order:
        s = arms[name]
        rec = ((s["subs_total"] - bs["subs_total"])
               / max(hr["subs_total"] - bs["subs_total"], 1))
        print(f"{name:<28} {len(s['boxes']):>6} {s['n_hosts']:>7} "
              f"{s['subs_total']:>8} {s['subs_per_host']:>7.2f} "
              f"{100 * s['subs_total'] / max(hr['subs_total'], 1):>6.1f}% "
              f"{100 * rec:>6.1f}% "
              f"{s['hosts_alive']:>6} ({100 * s['hosts_alive_frac']:>4.1f}%)")

    edges = boxes[0]["host_mass_edges_log10"]
    print(f"\n=== occupation vs host mass (mean subhalos per host, "
          f"{hr['n_hosts']} hosts)")
    print(f"{'log10 M_host':<16} {'hosts':>7} " +
          " ".join(f"{n[:10]:>11}" for n in order))
    for j in range(len(edges) - 1):
        n_h = hr["by_host_mass"][j]["n_hosts"]
        if n_h == 0:
            continue
        cells = " ".join(f"{arms[n]['by_host_mass'][j]['subs_per_host']:>11.1f}"
                         for n in order)
        print(f"{edges[j]:>5.1f}-{edges[j + 1]:<10.1f} {n_h:>7} {cells}")

    latex = [r"\begin{tabular}{l r r r}", r"\hline",
             r"Population & SR2 baseline & Ours & HR \\", r"\hline"]
    write_json(Path(a.out).with_suffix(".json"), {
        "ok": True,
        "boxes": [b["box"] for b in boxes],
        "baseline": ref,
        "host_min_p": boxes[0]["host_min_p"],
        "sub_min_p": boxes[0]["sub_min_p"],
        "n_hosts": hr["n_hosts"],
        "host_mass_edges_log10": edges,
        "arms": arms,
    })
    # The slide table, for whichever arm is the headline: written for every arm
    # so the choice stays the author's.
    tex = Path(a.out).with_suffix(".tex")
    with open(tex, "w") as fh:
        fh.write(f"% pooled over {', '.join(b['box'] for b in boxes)}; "
                 f"{hr['n_hosts']} HR hosts >= {boxes[0]['host_min_p']}p\n")
        for name in order:
            if name in ("hr", ref):
                continue
            s = arms[name]
            fh.write("\n".join(latex) + "\n")
            fh.write(f"Subhalos within $R_{{\\rm vir}}$ & "
                     f"{100 * bs['subs_total'] / max(hr['subs_total'], 1):.0f}\\% & "
                     f"\\textbf{{{100 * s['subs_total'] / max(hr['subs_total'], 1):.0f}\\%}} & "
                     f"100\\% \\\\\n")
            fh.write(f"Hosts resolved & "
                     f"{100 * bs['hosts_alive_frac']:.0f}\\% & "
                     f"\\textbf{{{100 * s['hosts_alive_frac']:.0f}\\%}} & "
                     f"100\\% \\\\\n")
            fh.write("\\hline\n\\end{tabular}\n"
                     f"% arm: {name} ({len(s['boxes'])} boxes, "
                     f"{s['n_hosts']} hosts, {s['subs_total']} subhalos)\n")
    print(f"\nwrote {Path(a.out).with_suffix('.json')}")
    print(f"      {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
