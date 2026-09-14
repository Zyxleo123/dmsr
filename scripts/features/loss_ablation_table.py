"""Tabulate the leave-one-out loss ablation from its Rockstar gate JSONs.

Reads only what the gate already wrote -- ``verdict_vs_self.json`` (or
``region_counts.json`` when no baseline was given) under each arm's
``holdout_<box>`` and ``holdout_<box>_step<N>`` directories, plus the last
held-out row of the arm's ``metrics.jsonl`` for the surrogate terms. No catalog,
no field. Missing rows are reported as missing, not dropped silently.

    python scripts/features/loss_ablation_table.py --reward-root R --box set9 \
        --arms "repro nobound" --gate-at 2000 --out R/member_gather/loss_ablation_set9
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TERMS = ("virial", "bound", "d6", "rrms", "sigmav", "centre")
SURROGATE = ("bound_hard", "highk_ratio_max", "vel_rms_ratio", "low_k")


def bin_order(d) -> list:
    """The bin labels of a region_mass_function dict, LOW TO HIGH.

    ``write_json`` sorts keys, so the stored order is alphabetical --
    "100-200p", "200-500p", "2000-...p", "50-100p", "500-2000p" -- and reading
    the dict in iteration order silently transposes the mass function. Sort by
    the numeric lower edge instead.
    """
    return sorted((k for k in d if k != "total"), key=lambda k: int(k.split("-")[0]))


def read_gate(run_dir: Path):
    for name in ("verdict_vs_self.json", "region_counts.json"):
        p = run_dir / name
        if p.is_file():
            return json.loads(p.read_text())
    return None


def gate_row(d) -> dict:
    new = d["region_counts"]["new"]["in_tiles"]
    row = {"host_subs": d["mass_function"]["new"]["total"],
           "region_subs": new["subs_ge_50p"],
           "region_hosts": new["hosts_ge_200p"],
           "q1_abundance_held": d.get("q1_abundance_held"),
           "q2_hosts_recovered": d.get("q2_hosts_recovered")}
    # Present only on verdicts written from 2026-09-14 (submit_reverdict.sh
    # backfills older ones); absent stays absent rather than faked as 0.
    shape = (d.get("shape_score") or {}).get("new") or {}
    row["shape_sub"] = shape.get("sub")
    row["shape_host"] = shape.get("host")
    row["radial_sub"] = (d.get("radial_score") or {}).get("new")
    rad = (d.get("radial_distribution") or {}).get("new") or {}
    row["median_r_over_rvir"] = rad.get("median_r_over_rvir")
    rmf = d.get("region_mass_function") or {}
    if "new" in rmf and "hr" in rmf:
        bins = bin_order(rmf["hr"]["sub"])
        row["region_bin_ratio"] = [
            (rmf["new"]["sub"][k] + 1.0) / (rmf["hr"]["sub"][k] + 1.0) for k in bins]
    return row


def reference_rows(d) -> dict:
    """base / HR / baseline arm, all carried inside any one verdict file."""
    out = {}
    for who in ("base", "hr", "self"):
        if who in d["region_counts"] and who in d["mass_function"]:
            rc = d["region_counts"][who]["in_tiles"]
            out[who] = {"host_subs": d["mass_function"][who]["total"],
                        "region_subs": rc["subs_ge_50p"],
                        "region_hosts": rc["hosts_ge_200p"]}
    return out


def last_holdout(metrics: Path, max_step=None):
    if not metrics.is_file():
        return None
    row = None
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if max_step is not None and int(r["step"]) > max_step:
            break
        row = r
    if row is None:
        return None
    h = row["holdout"]
    keep = {"step": int(row["step"])}
    keep.update({k: h.get(k) for k in SURROGATE})
    keep.update({f"term_{t}": h.get(f"term_{t}") for t in TERMS})
    return keep


def build(reward_root: Path, box: str, arms, gate_at: int) -> dict:
    mg = reward_root / "member_gather"
    rows, refs = [], {}
    for label in arms:
        arm_dir = mg / f"all_blocks_abl_{label}"
        for stage, sub, cap in (("step%d" % gate_at, f"holdout_{box}_step{gate_at}", gate_at),
                                ("final", f"holdout_{box}", None)):
            d = read_gate(arm_dir / sub)
            row = {"arm": label, "stage": stage, "gated": d is not None}
            if d is not None:
                row.update(gate_row(d))
                refs = refs or reference_rows(d)
            row["holdout_metrics"] = last_holdout(arm_dir / "metrics.jsonl", cap)
            rows.append(row)
    return {"box": box, "gate_at": gate_at, "references": refs, "rows": rows}


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def markdown(table: dict) -> str:
    refs = table["references"]
    ref_subs = refs.get("self", {}).get("region_subs")
    ref_hosts = refs.get("self", {}).get("region_hosts")
    lines = [f"# Loss ablation, held-out {table['box']} (edited region, 32 tiles)", "",
             "Every arm is `repro` with ONE weight zeroed. Read `repro` first.", "",
             "| reference | host subs | region subs>=50p | region hosts>=200p |",
             "|---|--:|--:|--:|"]
    for who, name in (("base", "frozen SR2"), ("self", "all_blocks_self (Aug)"), ("hr", "HR")):
        if who in refs:
            r = refs[who]
            lines.append(f"| {name} | {r['host_subs']} | {r['region_subs']} | {r['region_hosts']} |")
    lines += ["", "| arm | stage | host subs | region subs | d subs vs self | region hosts "
              "| d hosts vs self | shape (subs) | radial | med r/Rvir "
              "| region new/HR per bin | bound_hard | highk max | vel_rms |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|"]
    for r in table["rows"]:
        m = r.get("holdout_metrics") or {}
        if not r["gated"]:
            lines.append(f"| {r['arm']} | {r['stage']} | not gated | | | | | | | | "
                         f"| {fmt(m.get('bound_hard'))} | {fmt(m.get('highk_ratio_max'))} "
                         f"| {fmt(m.get('vel_rms_ratio'))} |")
            continue
        ratio = r.get("region_bin_ratio")
        ratio_s = " ".join(f"{x:.2f}" for x in ratio) if ratio else "-"
        ds = (r["region_subs"] - ref_subs) if ref_subs is not None else None
        dh = (r["region_hosts"] - ref_hosts) if ref_hosts is not None else None
        lines.append(
            f"| {r['arm']} | {r['stage']} | {r['host_subs']} | {r['region_subs']} "
            f"| {fmt(ds)} | {r['region_hosts']} | {fmt(dh)} "
            f"| {fmt(r.get('shape_sub'))} | {fmt(r.get('radial_sub'))} "
            f"| {fmt(r.get('median_r_over_rvir'))} | {ratio_s} "
            f"| {fmt(m.get('bound_hard'))} | {fmt(m.get('highk_ratio_max'))} "
            f"| {fmt(m.get('vel_rms_ratio'))} |")
    lines += ["", "`shape (subs)` is mean |log(new/HR)| over the five region sub bins "
              "(0 = HR's own mass function); the ratio column is that comparison bin by "
              "bin, 50-100 / 100-200 / 200-500 / 500-2000 / 2000+ p. A right TOTAL with "
              "ratios above 1 at the massive end is the open defect.",
              "", "`radial` is mean |log(new/HR)| over the host-centric distance bins "
              "(r / host R_vir), on FRACTIONS -- so it scores WHERE subhalos sit, "
              "independently of how many there are. 0 = HR's own radial distribution. "
              "`med r/Rvir` is the median of that distribution (HR's own value is the "
              "target, printed in each verdict log).",
              "", "Surrogate columns are the last held-out metrics row at or before the "
              "gated step. They are NOT the result; the Rockstar columns are."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reward-root", type=Path, required=True)
    ap.add_argument("--box", default="set9")
    ap.add_argument("--arms", required=True, help="space-separated ablation labels")
    ap.add_argument("--gate-at", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True, help="path stem; writes .json and .md")
    args = ap.parse_args(argv)
    table = build(args.reward_root, args.box, args.arms.split(), args.gate_at)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(table, indent=2))
    md = markdown(table)
    args.out.with_suffix(".md").write_text(md)
    print(md)
    n = sum(r["gated"] for r in table["rows"])
    print(f"{n}/{len(table['rows'])} rows gated -> {args.out}.json / .md")


if __name__ == "__main__":
    main()
