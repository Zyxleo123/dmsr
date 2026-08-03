#!/usr/bin/env python
"""Stage 0: calibrate the local editor's feasibility thresholds by measurement.

``configs/reward/reward.yaml`` carries ``constraints.calibrated: false`` and a
``low_k_change_max`` of 0.02 that nothing derived -- it was chosen for a model
that rewrites 512^3 cells. Reusing it here would be wrong in both directions: a
local edit moving a few hundred of 1.34e8 particles sits orders of magnitude
below 0.02, so the threshold would pass everything the editor can do including
destructive ones; and any threshold set as "a fraction of the difference between
two independent HR boxes" measures cosmic variance, not fidelity.

So the thresholds are set from four measured populations:

``frozen``
    The frozen SR2 box against itself. Every constraint is identically zero by
    construction; it is measured anyway, because a nonzero value here means the
    measurement code is broken and nothing after it can be trusted.
``oracle_success``
    The targeted HR-oracle interventions that **recovered** their subhalo, at
    each alpha. These are known-good localised edits. The threshold must accept
    them: an edit already demonstrated to restore real structure cannot be
    called infeasible.
``oracle_control``
    The oracle's equal-count random-particle controls. Same magnitude of change,
    no structural benefit -- so they bound how much of the accept region is
    explained by "an edit of this size" rather than by "an edit that worked".
``editor``
    Deployment-legal candidates from ``run_editor_candidates.py --measure-only``.
    The population the threshold will actually be applied to.

The proposed threshold for each ``max`` constraint is

    max(accept population) * margin,

with ``accept = frozen + oracle_success + editor``, and the report states the
separation against a *destructive* reference: the whole-field residual scale, and
the oracle at alpha = 1 applied over a random equal-count mask. A threshold that
does not separate them is reported as ``separates: false`` rather than quietly
committed.

    python scripts/reward/audit_local_editor_constraints.py --run-name le_a
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

import numpy as np

from _local_common import (  # noqa: E402
    PIPELINE, add_local_args, banner, load_local_config, read_jsonl, rows_path,
    run_dir, write_json,
)

from cosmo_sr.reward import paths  # noqa: E402

# The constraints a deployment-legal editor can actually measure: both are
# functions of the candidate field, the frozen base and the LR input alone. The
# HR-referenced ones (displacement/density power against HR) are omitted on
# purpose -- measuring them would make the feasibility filter illegal.
LOCAL_CONSTRAINTS = ("low_k_change", "lr_consistency_error")

# Of those, the ones measured as a difference *from the frozen base*, which must
# therefore be exactly zero on the no-op arm.
#
# ``lr_consistency_error`` is not one of them: it is ||A(Psi) - y_lr|| / ||y_lr||,
# measured against the LR *input*, and the frozen baseline's own value is ~0.46
# (constraints.py says so explicitly -- "the frozen baseline's own value is not
# zero, so the threshold is set from it rather than from 0"). Demanding zero
# there would report a correct pipeline as broken.
ZERO_ON_FROZEN = ("low_k_change",)


def oracle_rows(boxes: Sequence[str]) -> List[Dict]:
    """Experiment-1 intervention rows from the reward line, if they exist."""
    out: List[Dict] = []
    for box in boxes:
        p = paths.subdir("oracle_hr", box) / "interventions.jsonl"
        if p.is_file():
            out.extend(read_jsonl(p))
    return out


def _values(rows: Sequence[Dict]) -> Dict[str, np.ndarray]:
    out: Dict[str, List[float]] = {k: [] for k in LOCAL_CONSTRAINTS}
    for r in rows:
        c = r.get("constraints") or {}
        for k in LOCAL_CONSTRAINTS:
            v = c.get(k)
            if v is not None and np.isfinite(v):
                out[k].append(float(v))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _within_box_spread(rows: Sequence[Dict], c: str) -> float:
    """Largest max-min of ``c`` *inside* a single box.

    Pooling boxes would confound the edit's effect with the box-to-box
    difference in the frozen baseline, which is much larger for anything
    measured against the LR input: set8 and set9 sit at lr_consistency_error
    0.4643 and 0.4497 respectively, a 0.015 gap, while a local edit moves the
    quantity in the 6th decimal. A pooled spread would therefore report a
    constraint as discriminating when all it is measuring is which box the
    candidate came from.
    """
    by_box: Dict[str, List[float]] = {}
    for r in rows:
        v = (r.get("constraints") or {}).get(c)
        if v is not None and np.isfinite(v):
            by_box.setdefault(str(r.get("box", "")), []).append(float(v))
    spreads = [max(v) - min(v) for v in by_box.values() if len(v) > 1]
    return float(max(spreads)) if spreads else float("nan")


def _stats(x: np.ndarray) -> Dict:
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "min": float(x.min()), "max": float(x.max()),
            "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "mean": float(x.mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_local_args(ap)
    ap.add_argument("--run-name", default="le_a")
    ap.add_argument("--boxes", default="")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="proposed threshold = margin * max(accept population). "
                         "2 leaves room for a host the search has not met yet "
                         "without opening the filter by an order of magnitude")
    args = ap.parse_args(argv)

    cfg = load_local_config(args)
    boxes = ([b.strip() for b in args.boxes.split(",") if b.strip()]
             or list(cfg.get("search_boxes", [])))

    # --- populations ---------------------------------------------------------
    orows = oracle_rows(boxes)
    pop: Dict[str, List[Dict]] = {
        "frozen": [r for r in orows if float(r.get("alpha", -1)) == 0.0],
        "oracle_success": [r for r in orows
                           if float(r.get("alpha", 0)) > 0
                           and r.get("kind") == "targeted"
                           and int(r.get("n_recovered", 0)) > 0],
        "oracle_control": [r for r in orows if r.get("kind") == "control"],
    }
    erows: List[Dict] = []
    for box in boxes:
        erows.extend([r for r in read_jsonl(rows_path(args.run_name, box))
                      if r.get("constraints")])
    pop["editor"] = erows
    pop["editor_frozen_arm"] = [r for r in erows if r.get("arm") == "frozen"]

    measured = {k: {c: _stats(v) for c, v in _values(rs).items()}
                for k, rs in pop.items()}

    # --- the base-referenced constraints must be exactly zero on the no-op ---
    zero_ok = True
    zero_detail = {}
    for c in ZERO_ON_FROZEN:
        st = measured.get("editor_frozen_arm", {}).get(c, {})
        if st.get("n", 0):
            zero_detail[c] = st["max"]
            if st["max"] > 0.0:
                zero_ok = False

    # --- proposal ------------------------------------------------------------
    accept = ["frozen", "oracle_success", "editor"]
    reject = ["oracle_control"]
    proposal: Dict[str, Optional[float]] = {}
    separation: Dict[str, Dict] = {}
    for c in LOCAL_CONSTRAINTS:
        a = np.concatenate([_values(pop[k])[c] for k in accept
                            if _values(pop[k])[c].size] or [np.zeros(0)])
        r = np.concatenate([_values(pop[k])[c] for k in reject
                            if _values(pop[k])[c].size] or [np.zeros(0)])
        if a.size == 0:
            proposal[f"{c}_max"] = None
            separation[c] = {"status": "no accept population measured"}
            continue
        thr = float(a.max()) * float(args.margin)
        proposal[f"{c}_max"] = thr

        # How much of the accept region the EDITOR actually occupies, and how
        # much the edit moves the quantity at all.
        #
        # Both matter and neither is visible from the threshold alone. The
        # accept population is dominated by the HR oracle, whose interventions
        # replace the field with real HR content over a dilated Lagrangian mask
        # -- a far larger perturbation than a few hundred contracted particles.
        # The plan requires the threshold to accept those, so it does; the
        # consequence is that the bound can sit orders of magnitude above
        # anything the editor produces, and reporting only "feasible: true"
        # would present a filter that never binds as if it were a live gate.
        ev = _values(pop["editor"])[c]
        fv = _values(pop["editor_frozen_arm"])[c]
        base = float(fv.max()) if fv.size else float("nan")
        headroom = (thr / float(ev.max())) if ev.size and ev.max() > 0 else float("inf")
        spread = _within_box_spread(pop["editor"], c)
        separation[c] = {
            "threshold": thr,
            "accept_max": float(a.max()), "accept_n": int(a.size),
            "reject_min": (float(r.min()) if r.size else None),
            "reject_n": int(r.size),
            "editor_max": (float(ev.max()) if ev.size else None),
            "editor_within_box_spread": spread,
            "frozen_arm_value": base,
            "threshold_over_editor_max": headroom,
            # A threshold hundreds of times above the population it filters is
            # not a filter. Say so, so nobody reads "feasible" as evidence.
            "binds_on_editor": bool(np.isfinite(headroom) and headroom < 10.0),
            # If the edit barely moves the quantity relative to its own value,
            # no threshold on it can discriminate between a good edit and a bad
            # one -- e.g. lr_consistency_error is ~0.46 for the FROZEN box and
            # the edit changes it in the 5th decimal.
            # Discriminating power, referenced to the right scale. For a
            # base-referenced constraint the frozen value is zero by
            # construction, so the population's own range is the only sensible
            # yardstick; for the others it is the frozen box's own value.
            "discriminates": bool(
                np.isfinite(spread) and (
                    spread / max(float(ev.max()), 1e-30) > 0.1
                    if c in ZERO_ON_FROZEN else
                    (np.isfinite(base) and base > 0
                     and spread / max(base, 1e-30) > 1e-3))),
            # Separation is a genuine question, not a formality: if the random
            # control produces the same field-level footprint as a successful
            # targeted edit -- which is exactly what an equal-count control is
            # designed to do -- then this constraint cannot distinguish them and
            # should be reported as a size bound, not as a fidelity filter.
            "separates": bool(r.size and float(r.min()) > thr),
            "note": ("an equal-count control has the same magnitude by design, "
                     "so 'separates: false' is the expected outcome and means "
                     "this threshold bounds edit SIZE, not edit quality"),
        }

    report = {
        "pipeline": PIPELINE, "run_name": args.run_name, "boxes": boxes,
        "margin": float(args.margin),
        "populations": {k: len(v) for k, v in pop.items()},
        "measured": measured,
        "frozen_arm_exactly_zero": zero_ok,
        "frozen_arm_values": zero_detail,
        "zero_on_frozen_constraints": list(ZERO_ON_FROZEN),
        "separation": separation,
        "proposed_constraints_block": {
            "calibrated": True,
            **proposal,
            # Left disabled deliberately: measuring them requires the HR field,
            # which a deployment-legal editor may not read.
            "density_power_error_max": None,
            "displacement_power_error_max": None,
            "diversity_min": None,
        },
        "how_to_apply": (
            "paste proposed_constraints_block into the `constraints:` block of "
            "configs/reward/local_editor.yaml, verify frozen_arm_exactly_zero is "
            "true and that the accept population covers the oracle's SUCCESSFUL "
            "interventions, then commit. Do not widen a threshold to make a "
            "candidate pass."),
    }
    out = write_json(paths.LOCAL_EDITOR("audit", create=True) /
                     "constraints_proposal.json", report)
    write_json(run_dir(args.run_name) / "constraints_audit.json", report)

    banner(f"constraint audit -> {out}")
    for k, v in report["populations"].items():
        print(f"    population {k:20s} {v:4d} rows", flush=True)
    for c in LOCAL_CONSTRAINTS:
        s = separation[c]
        print(f"    {c}", flush=True)
        print(f"        threshold {s.get('threshold')}  "
              f"(accept max {s.get('accept_max')}, reject min {s.get('reject_min')}, "
              f"separates={s.get('separates')})", flush=True)
        print(f"        editor max {s.get('editor_max')}  "
              f"frozen arm {s.get('frozen_arm_value')}  "
              f"threshold/editor_max {s.get('threshold_over_editor_max')}", flush=True)
        if not s.get("binds_on_editor", True):
            print("        !! this threshold does NOT bind on the editor: it is "
                  "set by the HR oracle's much larger interventions, which the "
                  "plan requires it to accept. Feasibility here is not evidence "
                  "the edit was gentle; the host-damage term of the object-level "
                  "reward is what actually protects the parent.", flush=True)
        if not s.get("discriminates", True):
            print("        !! and the edit barely moves this quantity relative to "
                  "its frozen-box value, so no threshold on it can separate a "
                  "good edit from a bad one. Report it as measured, not as a "
                  "filter.", flush=True)
    if not zero_ok:
        print(f"    !! the no-op arm did NOT measure exactly zero for "
              f"{ZERO_ON_FROZEN}: {zero_detail}. That is a bug in the "
              "measurement path, not a threshold question; stop here.",
              flush=True)
        return 2
    print(f"    no-op arm exactly zero for {list(ZERO_ON_FROZEN)}: {zero_detail}",
          flush=True)
    if not any(measured.get("oracle_success", {}).get(c, {}).get("n")
               for c in LOCAL_CONSTRAINTS):
        print("    !! no successful oracle interventions were available, so the "
              "accept population rests on the editor's own candidates alone. "
              "Run the Experiment-1 oracle (reward line) first, or state this "
              "limitation wherever the thresholds are used.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
