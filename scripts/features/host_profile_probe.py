#!/usr/bin/env python
"""Does the solve pull candidates inward, and is there anything to pull outward?

Two questions, both owed BEFORE a host-profile arm is trained, because either
answer can kill it:

**1. The supply.** ``--soft-host-profile`` works by refusing a candidate the
core material it currently gathers. If the candidate field has no material at
``0.8-1.5 R_vir`` to gather instead, the refusal does not RELOCATE the object,
it STARVES it: the mass constraint goes unmet, the dual runs away, and the
gather loss is applied to a near-empty pool. That is a worse failure than the
one the term is for. So this measures the radial particle supply around each
gated host, HR against the frozen SR2 field, in shells of ``R_vir``, and writes
``supply_<box>.json``. The number that decides is the frozen/HR ratio in the
``1.0-1.5 R_vir`` shell where HR's subhalos peak: near 1 means the material is
there and only the SELECTION is wrong, well below 1 means the field itself is
missing the outskirts and no membership rule can conjure them.

**2. The drift.** One EM sweep recentres every candidate on the material it
selected (``soft_membership`` M step). If those centres move systematically
INWARD, the solve's fixed point really is the core and the background is the
right thing to change. If they are stable, the misplacement was already in the
field before the loss touched it and this term will not be what moves it. Read
straight off the ``--dump-membership-npz`` snapshots, so no field and no GPU:
the drift is projected on the clustercentric direction and reported per sweep.

Both stages are pure functions of artefacts already on disk, which is why this
runs on the CPU partition and why its figures redraw without recomputation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reward"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from _common import banner, hr_path, load_reward_config, paths, write_json  # noqa: E402

from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.features.host_profile import load_lr_host_table  # noqa: E402
from cosmo_sr.reward.base import find_base_field  # noqa: E402

BOX = 100.0

#: Shell edges in ``R_vir``. The ``1.0-1.5`` shell is the one that decides:
#: ``radial-placement-is-inverted`` puts HR's median subhalo at 1.13 R_vir and
#: every tuned arm at 0.30-0.35, so that is where the material has to exist.
SHELLS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


# --------------------------------------------------------------------------- #
# 1. the supply
# --------------------------------------------------------------------------- #
def radial_counts(pos: np.ndarray, centres: np.ndarray, rvir: np.ndarray, *,
                  boxsize: float, shells=SHELLS, chunk: int = 64) -> np.ndarray:
    """``(H, len(shells) - 1)`` particle counts per host per ``R_vir`` shell.

    Brute force in host chunks, minimum-image periodic. The particle table is
    512^3 and the host list a few hundred, so this is a couple of passes over
    the table rather than anything clever, and clever here would be a second
    definition of "distance" to keep in step with the solve's.
    """
    edges = np.asarray(shells, dtype=np.float64)
    out = np.zeros((centres.shape[0], edges.size - 1), dtype=np.int64)
    rmax = float(edges[-1])
    for s in range(0, centres.shape[0], chunk):
        c = centres[s:s + chunk]
        rv = np.maximum(rvir[s:s + chunk], 1e-12)
        for k in range(c.shape[0]):
            d = pos - c[k][None, :]
            d -= boxsize * np.round(d / boxsize)
            # Cheap box reject before the norm: at 3 R_vir the shell is a tiny
            # fraction of the box and the full norm over 512^3 per host is the
            # only part of this job that would not fit in the time limit.
            lim = rmax * rv[k]
            m = (np.abs(d) <= lim).all(axis=1)
            if not m.any():
                continue
            r = np.sqrt((d[m] ** 2).sum(axis=1)) / rv[k]
            out[s + k] = np.histogram(r, bins=edges)[0]
    return out


def supply_for_box(box: str, cfg, args, out_dir: Path) -> Dict:
    tbl = load_lr_host_table(Path(args.lr_host_table or
                                  paths.subdir("lr_host_table")) / f"{box}.npz")
    keep = np.argsort(-tbl["host_mvir_msun_h"])[:int(args.max_hosts)]
    centres = tbl["host_pos_mpc_h"][keep]
    rvir = tbl["host_rvir_mpc_h"][keep]

    hr_field = np.load(str(hr_path(cfg, box)), mmap_mode="r")
    base_p = find_base_field(box, seed=int(args.seed))
    if base_p is None:
        raise SystemExit(
            f"no cached frozen SR2 box for {box} seed {args.seed}; produce it "
            "with scripts/reward/cache_sr2_base.py")
    sr_field = np.load(str(base_p), mmap_mode="r")

    rows = {}
    for tag, field in (("hr", hr_field), ("frozen", sr_field)):
        p = field_to_particles(np.asarray(field), boxsize_kpc_h=BOX * 1000.0)
        rows[tag] = radial_counts(p.pos_mpc_h, centres, rvir, boxsize=BOX)
        print(f"  {box} {tag}: {p.pos_mpc_h.shape[0]} particles, "
              f"{rows[tag].sum()} inside {SHELLS[-1]} R_vir of "
              f"{centres.shape[0]} hosts", flush=True)

    vol = np.diff(np.asarray(SHELLS, dtype=np.float64) ** 3)
    rep: Dict = {"box": box, "n_hosts": int(centres.shape[0]),
                 "shells_rvir": list(SHELLS), "shells": []}
    for j in range(len(SHELLS) - 1):
        hr_n = rows["hr"][:, j].astype(np.float64)
        fz_n = rows["frozen"][:, j].astype(np.float64)
        # Per host then median, not summed: a summed ratio is one cluster's
        # ratio, and `sr2-deficit-scales-with-host-mass` is the reason that
        # distinction has bitten this line before.
        ratio = fz_n / np.maximum(hr_n, 1.0)
        rep["shells"].append({
            "r_lo": SHELLS[j], "r_hi": SHELLS[j + 1],
            "hr_median": float(np.median(hr_n)),
            "frozen_median": float(np.median(fz_n)),
            "frozen_over_hr_median": float(np.median(ratio)),
            "hr_density_per_rvir3": float(np.median(hr_n) / vol[j]),
        })
    j15 = [i for i, (lo, hi) in enumerate(zip(SHELLS[:-1], SHELLS[1:]))
           if lo == 1.0 and hi == 1.5]
    if j15:
        rep["verdict_shell"] = "1.0-1.5"
        r = rep["shells"][j15[0]]["frozen_over_hr_median"]
        rep["verdict_ratio"] = r
        rep["verdict"] = ("SUPPLY OK -- the material is there and the "
                          "selection is what is wrong" if r >= float(args.supply_floor)
                          else "STARVED -- the frozen field is missing the "
                               "outskirts; a contrast rule cannot relocate "
                               "what is not there")
        print(f"  {box} VERDICT: frozen/HR at 1.0-1.5 R_vir = {r:.3f} "
              f"({rep['verdict']})", flush=True)
    np.savez(out_dir / f"supply_{box}.npz", centres=centres, rvir=rvir,
             hr_counts=rows["hr"], frozen_counts=rows["frozen"],
             shells=np.asarray(SHELLS))
    write_json(out_dir / f"supply_{box}.json", rep)
    return rep


# --------------------------------------------------------------------------- #
# 2. the drift
# --------------------------------------------------------------------------- #
def parent_geometry(halo_id: np.ndarray, cat) -> Dict[str, np.ndarray]:
    """Each set's OWN parent host: position, ``R_vir`` and parent id.

    Why this is not ``centre_host``: a member-gather task is named for one host,
    but its supervised sets are selected by HOME TILE, and a trained tile is
    12.5 Mpc/h a side at the deployed grids. MEASURED on the first real dump --
    159 subhalos of which only **51** belong to the task's host and 108 belong
    to 45 other parents sharing those tiles. Signing a radial drift against
    ``centre_host`` therefore measures the wrong direction for two thirds of
    the sets, and it does not merely add noise: it inverted the verdict, from
    -0.21 R_vir (INWARD, the real answer) to +0.08 R_vir (STABLE). The same
    class of bookkeeping error as `host-row-overlay-collision`.
    """
    ids = np.asarray(cat.ids)
    row = {int(i): k for k, i in enumerate(ids)}
    par = np.asarray(cat.parent_ids)
    pos = np.asarray(cat.pos, dtype=np.float64)
    rv = np.asarray(cat.rvir, dtype=np.float64) / 1000.0
    n = int(np.asarray(halo_id).size)
    ppos = np.full((n, 3), np.nan)
    prvir = np.full(n, np.nan)
    pid = np.full(n, -1, dtype=np.int64)
    for k, h in enumerate(np.asarray(halo_id)):
        r = row.get(int(h))
        if r is None:
            continue
        q = int(par[r])
        pr = row.get(q)
        if q < 0 or pr is None:
            continue
        pid[k], ppos[k], prvir[k] = q, pos[pr], max(rv[pr], 1e-9)
    return {"parent_id": pid, "parent_pos": ppos, "parent_rvir": prvir}


def drift_for_dump(path: Path, lr_table=None, cat=None) -> Dict:
    """Radial component of the M-step recentring, per sweep, from one npz.

    Signed against each set's OWN parent when ``cat`` (the box's HR catalog) is
    supplied -- see :func:`parent_geometry` for why that is not optional in
    practice. Without it, the drift is signed against the task's host for every
    set, which is right only for the subset that actually belongs to it.
    """
    z = np.load(str(path), allow_pickle=True)
    d = {k: z[k] for k in z.files}
    if "centre_host" not in d:
        raise SystemExit(
            f"{path} carries no centre_host, so a radial drift cannot be "
            "signed. Re-dump with a build that writes it (this script's "
            "companion change to finetune_member_gather.py), or the run was "
            "host_pos=None and has no clustercentric direction at all.")
    c = d["sweep_c"].astype(np.float64)          # (n_sweeps, S, 3)
    host = d["centre_host"].astype(np.float64)   # (S, 3)
    rvir = None
    attributed = None
    if cat is not None:
        g = parent_geometry(d["halo_id"], cat)
        ok = np.isfinite(g["parent_rvir"])
        if ok.any():
            # Keep only sets whose parent the catalog resolves; a set with no
            # parent has no clustercentric direction and must be dropped, not
            # silently signed against somebody else's host.
            c, host = c[:, ok, :], g["parent_pos"][ok]
            rvir = g["parent_rvir"][ok]
            d = dict(d, halo_id=np.asarray(d["halo_id"])[ok])
            attributed = {
                "n_sets_total": int(ok.size),
                "n_with_parent": int(ok.sum()),
                "n_distinct_parents": int(np.unique(g["parent_id"][ok]).size),
            }
    if rvir is None:
        hv = d.get("host_rvir")
        if hv is not None and np.any(np.asarray(hv) > 0):
            rvir = np.maximum(np.asarray(hv, dtype=np.float64), 1e-9)
        elif lr_table is not None:
            from cosmo_sr.features.host_profile import lr_host_for
            rvir = np.array([lr_host_for(host[i], lr_table, boxsize_mpc_h=BOX)[0]
                             for i in range(host.shape[0])])
            rvir = np.maximum(rvir, 1e-9) if np.any(rvir > 0) else None

    r0 = c[0] - host
    n0 = np.linalg.norm(r0, axis=1)
    rhat = r0 / np.maximum(n0, 1e-12)[:, None]
    rep = {"path": str(path), "halo_id": np.asarray(d["halo_id"]).tolist(),
           "n_sets": int(c.shape[1]),
           "signed_against": "parent" if attributed else "task_host",
           "sweeps": []}
    if attributed:
        rep.update(attributed)
    for t in range(c.shape[0]):
        r = np.linalg.norm(c[t] - host, axis=1)
        step = (np.sum((c[t] - c[max(t - 1, 0)]) * rhat, axis=1)
                if t else np.zeros(c.shape[1]))
        rep["sweeps"].append({
            "tag": str(d["sweep_tags"][t]),
            "median_r_mpc_h": float(np.median(r)),
            "median_r_over_host_rvir": (None if rvir is None
                                        else float(np.median(r / rvir))),
            "median_radial_step_mpc_h": float(np.median(step)),
            "frac_moving_inward": float(np.mean(step < 0.0)) if t else 0.0,
        })
    net = rep["sweeps"][-1]["median_r_mpc_h"] - rep["sweeps"][0]["median_r_mpc_h"]
    rep["net_radial_drift_mpc_h"] = float(net)
    if rep["sweeps"][-1]["median_r_over_host_rvir"] is not None:
        rep["net_radial_drift_rvir"] = float(
            rep["sweeps"][-1]["median_r_over_host_rvir"]
            - rep["sweeps"][0]["median_r_over_host_rvir"])
    rep["verdict"] = ("INWARD -- the solve's own fixed point is the core, and "
                      "the background is the right thing to change"
                      if net < 0 else
                      "STABLE/OUTWARD -- the misplacement is in the field "
                      "before the loss; a background change is not the lever")
    print(f"  {Path(path).name}: net radial drift {net:+.4f} Mpc/h "
          f"({rep['verdict'].split(' -- ')[0]})", flush=True)
    return rep


def hr_catalog_for(box: str, args):
    """The box's HR Rockstar catalog, for parent attribution. None if absent.

    Advisory rather than required: without it the drift is still computed, and
    says so in `signed_against`, so a dump from a box whose catalog has been
    purged still reports something honest.
    """
    if not box:
        return None
    pat = (Path(args.hr_catalog_root or paths.subdir("halos"))
           / f"{box}__hr__hr")
    hits = sorted(pat.glob("*_rockstar/halos_*.ascii"))
    if not hits:
        print(f"  note: no HR catalog under {pat}; the drift for this dump "
              "will be signed against the TASK host for every set, which is "
              "right only for the subset that belongs to it.", flush=True)
        return None
    from cosmo_sr.eval.rockstar import load_rockstar_ascii
    return load_rockstar_ascii(hits[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/reward/reward.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--boxes", nargs="*", default=[],
                    help="boxes for the SUPPLY stage; empty skips it")
    ap.add_argument("--dumps", nargs="*", default=[],
                    help="--dump-membership-npz files for the DRIFT stage; "
                         "empty skips it")
    ap.add_argument("--reward-root", default="")
    ap.add_argument("--lr-host-table", default="")
    ap.add_argument("--hr-catalog-root", default="",
                    help="where <box>__hr__hr/*_rockstar lives; empty means "
                         "$DMSR_REWARD_ROOT/halos. Used to attribute each set "
                         "to its OWN parent host for the drift stage.")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-hosts", type=int, default=64,
                    help="most massive N gated hosts per box. The supply "
                         "question is about hosts big enough to have "
                         "subhalos, and the pass is O(N) over a 512^3 table.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--supply-floor", type=float, default=0.8,
                    help="frozen/HR median in the 1.0-1.5 R_vir shell below "
                         "which the verdict is STARVED")
    args = ap.parse_args(argv)

    banner("host-profile probe")
    if args.reward_root:
        os.environ["DMSR_REWARD_ROOT"] = str(args.reward_root)
    cfg = load_reward_config(args)
    out_dir = (Path(args.out_dir) if args.out_dir
               else paths.subdir("host_profile_probe", create=True))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict = {"supply": [], "drift": []}
    for b in [x for x in " ".join(args.boxes).replace(",", " ").split() if x]:
        summary["supply"].append(supply_for_box(b, cfg, args, out_dir))
    for p in args.dumps:
        # The host size for a dump is looked up from the LR table of ITS OWN
        # box, so a drift number is in R_vir even when the run that wrote the
        # dump never needed a host profile.
        _t = _c = None
        _b = str(np.load(str(p), allow_pickle=True)["box"]) if Path(p).exists() else ""
        _tp = Path(args.lr_host_table or paths.subdir("lr_host_table")) / f"{_b}.npz"
        if _b and _tp.exists():
            _t = load_lr_host_table(_tp)
        _c = hr_catalog_for(_b, args)
        summary["drift"].append(drift_for_dump(Path(p), lr_table=_t, cat=_c))
    if not summary["supply"] and not summary["drift"]:
        raise SystemExit("nothing to do: pass --boxes, --dumps, or both.")
    s = write_json(out_dir / "summary.json", summary)
    print(f"\nwrote {s}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
