#!/usr/bin/env python
"""What does a density peak need to be a halo -- and do frozen / fine-tuned peaks have it?

``peak_halo_match.py`` measured that HR turns 65-93% of its host-outskirt
density peaks into Rockstar objects, frozen SR2 22-36%, the fine-tuned arm
32-56%, with peak counts and shapes matching. This asks which per-peak
property carries the difference, using the census's own records:

    log10 contrast (k=32 density / shell mean), log10 T/|U| (64 nearest),
    Lagrangian spread (cells), dvbg (core-vs-surroundings velocity offset in
    units of the surroundings' dispersion), coldbg (core dispersion over the
    surroundings'), c/a.

For each field, restricted to peaks at --x-min..--x-max R_vir:

1. the property's median among all / converted / unconverted peaks;
2. the conversion rate in quintile bins of the property (edges from HR's peaks,
   so every field is binned identically);
3. a logistic model of "is a halo" fitted on HR's peaks (standardised
   properties), then applied unchanged to each field: if the HR-trained model
   predicts a field's low conversion from that field's own property
   distribution, the measured properties explain the deficit; if it predicts
   HR-like conversion and the field still falls short, the cause is something
   these properties do not capture.

Saved records + catalogs only, seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import write_json  # noqa: E402
from peak_halo_match import match  # noqa: E402
from placement_forensics import read_rockstar  # noqa: E402

BASE_PROPS = ("log_ratio", "log_tu", "lag", "dvbg", "coldbg", "c_over_a")
#: added when the census ran with --bind (isolated unbinding + enclosed density)
BIND_PROPS = ("log_nbound", "log_enc128", "log_enc512")
PROPS = BASE_PROPS


def features(pk) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        f = {"log_ratio": np.log10(pk["ratio"]), "log_tu": np.log10(pk["tu"]),
             "lag": pk["lag"], "dvbg": pk["dvbg"], "coldbg": pk["coldbg"],
             "c_over_a": pk["c_over_a"]}
        if "nbound" in pk and np.isfinite(pk["nbound"]).any():
            f["log_nbound"] = np.log10(np.maximum(pk["nbound"], 1.0))
            f["log_enc128"] = np.log10(pk["enc128"])
            f["log_enc512"] = np.log10(pk["enc512"])
        return f


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50):
    """L2-regularised logistic regression by IRLS (intercept in column 0)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        g = X.T @ (p - y) + l2 * w
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", required=True,
                    help="comma list name=peaks.npz:rockstar.ascii; the FIRST is the "
                         "reference the model is fitted on (HR)")
    ap.add_argument("--x-min", type=float, default=0.75)
    ap.add_argument("--x-max", type=float, default=3.0)
    ap.add_argument("--min-match", type=float, default=0.03)
    ap.add_argument("--match-rvir", type=float, default=0.5)
    ap.add_argument("--min-num-p", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    data = {}
    for item in a.fields.split(","):
        name, rest = item.split("=", 1)
        pk_path, cat_path = rest.split(":")[:2]
        pk = dict(np.load(pk_path))
        cat = read_rockstar(cat_path)
        keep = cat["num_p"] >= a.min_num_p
        hit = match(pk["pos"], cat["pos"][keep],
                    cat["rvir"][keep] / 1000.0 * a.match_rvir, a.min_match) >= 0
        sel = (pk["x"] >= a.x_min) & (pk["x"] < a.x_max)
        f = {k: v[sel] for k, v in features(pk).items()}
        data[name] = (f, hit[sel].astype(float))
    names = list(data)
    ref = names[0]
    fr, yr = data[ref]
    global PROPS
    PROPS = BASE_PROPS + (BIND_PROPS if all("log_nbound" in data[n][0] for n in names)
                          else ())
    if "log_nbound" in fr:
        for n in names:
            f, y = data[n]
            nbv = np.power(10.0, f["log_nbound"])
            ok = np.isfinite(nbv)
            print(f"{n:14s} isolated-bound particles per outskirt peak: median "
                  f"{np.median(nbv[ok]):.0f}, frac >=20 {np.mean(nbv[ok] >= 20):.2f}, "
                  f">=50 {np.mean(nbv[ok] >= 50):.2f}; conversion if >=20: "
                  f"{y[ok & (nbv >= 20)].mean():.2f}, if <20: {y[ok & (nbv < 20)].mean():.2f}",
                  flush=True)

    out = {"x_range": [a.x_min, a.x_max], "reference": ref, "fields": {}}
    # 1 + 2: medians and quintile conversion curves on the reference's edges
    edges = {p: np.nanquantile(fr[p], [0.2, 0.4, 0.6, 0.8]) for p in PROPS}
    for n in names:
        f, y = data[n]
        rec = {"n_peaks": int(y.size), "conversion": float(y.mean()) if y.size else np.nan,
               "medians": {}, "quintile_conversion": {}, "quintile_fraction": {}}
        for p in PROPS:
            v = f[p]
            ok = np.isfinite(v)
            rec["medians"][p] = {
                "all": float(np.median(v[ok])) if ok.any() else np.nan,
                "converted": float(np.median(v[ok & (y > 0)])) if (ok & (y > 0)).any() else np.nan,
                "unconverted": float(np.median(v[ok & (y == 0)])) if (ok & (y == 0)).any() else np.nan}
            q = np.digitize(v, edges[p])
            rec["quintile_conversion"][p] = [float(y[ok & (q == i)].mean()) if np.any(ok & (q == i))
                                             else np.nan for i in range(5)]
            rec["quintile_fraction"][p] = [float(np.mean(q[ok] == i)) if ok.any() else np.nan
                                           for i in range(5)]
        out["fields"][n] = rec

    # 2b: explicit contrast bands -- how MANY peaks of each contrast a field
    # has (absolute, per the same 60 hosts) and how many become objects. This is
    # what separates "the low-contrast clumps are missing" from "they are there
    # and do not bind".
    bands = (3.0, 10.0, 30.0, 100.0, np.inf)
    print(f"  contrast bands (k=32 density / shell mean), {a.x_min}-{a.x_max} R_vir:")
    for n in names:
        f, y = data[n]
        ratio = np.power(10.0, f["log_ratio"])
        row = []
        for lo, hi in zip(bands[:-1], bands[1:]):
            m = (ratio > lo) & (ratio <= hi)
            row.append({"band": f"{lo:g}-{hi:g}", "n": int(m.sum()),
                        "conversion": float(y[m].mean()) if m.any() else float("nan"),
                        "n_converted": int(y[m].sum())})
        out["fields"][n]["contrast_bands"] = row
        print(f"    {n:14s} " + "  ".join(f"{r['band']:>7s}: n {r['n']:6d} conv "
                                         f"{r['conversion']:.2f}" for r in row))

    # 3: HR-trained logistic model, applied to every field
    def design(f, mu=None, sd=None):
        M = np.column_stack([f[p] for p in PROPS])
        ok = np.all(np.isfinite(M), axis=1)
        if mu is None:
            mu, sd = M[ok].mean(axis=0), M[ok].std(axis=0) + 1e-12
        Z = (M - mu) / sd
        return np.column_stack([np.ones(len(Z)), Z]), ok, mu, sd
    Xr, okr, mu, sd = design(fr)
    w = fit_logistic(Xr[okr], yr[okr])
    out["logistic"] = {"features": ["intercept", *PROPS], "coef_standardised": w.tolist(),
                       "feature_mean": mu.tolist(), "feature_sd": sd.tolist()}
    print(f"HR-fitted logistic (standardised coefficients): "
          + ", ".join(f"{k} {c:+.2f}" for k, c in zip(["icpt", *PROPS], w)))
    for n in names:
        f, y = data[n]
        X, ok, _, _ = design(f, mu, sd)
        pred = predict(X[ok], w)
        out["fields"][n]["predicted_conversion_hr_model"] = float(pred.mean()) if ok.any() else np.nan
        out["fields"][n]["actual_conversion_rows_used"] = float(y[ok].mean()) if ok.any() else np.nan
        out["fields"][n]["n_rows_used"] = int(ok.sum())
        print(f"{n:14s} peaks {y.size:6d}  actual conv {y[ok].mean():.3f}  HR-model predicts "
              f"{pred.mean():.3f}  | medians: "
              + ", ".join(f"{p} {out['fields'][n]['medians'][p]['all']:.2f}" for p in PROPS),
              flush=True)
    for p in PROPS:
        print(f"  quintile conversion by {p:9s} (HR edges {np.round(edges[p], 2).tolist()}):")
        for n in names:
            qc = out["fields"][n]["quintile_conversion"][p]
            qf = out["fields"][n]["quintile_fraction"][p]
            print(f"    {n:14s} conv {[round(x, 2) for x in qc]}  frac {[round(x, 2) for x in qf]}")
    write_json(Path(a.out), out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
