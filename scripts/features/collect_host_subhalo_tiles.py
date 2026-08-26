#!/usr/bin/env python
"""Per-tile subhalo counts of *one selected host*, in HR and in SR2.

The viewer already shows how much of an LR host each tile carries
(``host_fraction_per_tile``). This adds the thing that fraction is supposed to
buy: how many subhalos that tile actually produced **for that host**, on each
side of the comparison. Panel 5 of the viewer answers the same question for a
tile against the whole box; this answers it for a tile against the host.

Why this needs the owner arrays and not just the tile weights
------------------------------------------------------------
``*_tilew.npz`` knows, for every catalog object, how its members split over
tiles -- but it does not know which *host* an LR host corresponds to in the HR
or SR2 catalog, and it cannot separate two structures that share a tile. Both
are settled exactly by ``<box>_<tag>_owner.npy``:
``owner[hr_particle_id]`` is the catalog object the particle is bound to, and
our GADGET2 ids are the flat Lagrangian index, so an LR site's
``upsample**3`` HR children are a lookup
(:meth:`LagrangianGrid.hr_children`). No positional matching is involved.

Two counts per (host, tile), on each side:

``tile_sub`` (the one the page plots)
    The matched host's own subhalos, split over the tiles their Lagrangian
    material came from: subhalo ``s`` contributes its ``weight[s, t]`` from the
    tile-weight file, so a subhalo spanning two tiles is split fractionally and
    the row sums to the host's full subhalo count. Whole-box and unrestricted.
``tile_sub_footprint``
    The same, but counting only particles inside *this LR host's* Lagrangian
    footprint (each such particle contributes ``1 / num_p(subhalo)``). Useful
    when one HR host swallows several LR structures -- and reported second
    rather than first because it is **biased against whichever side builds
    larger subhalos**: more of a big subhalo falls outside the footprint, so
    SR2's inflated survivors lose more of their weight than HR's small ones.

Subhalos of *other* hosts are never counted: the root of every candidate is
compared against the matched host's id first. Where one HR host swallows two LR
structures the page can still overstate a shared tile, which is why
``share_of_match`` (how much of the matched host's material this LR host is) and
``n_lr_hosts_sharing`` are reported per host rather than hidden.

    python scripts/features/collect_host_subhalo_tiles.py --boxes set8
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (  # noqa: E402
    DEFAULT_CONFIG, banner, load_reward_config, paths, write_json,
)

from cosmo_sr.eval.particle_identity import root_lookup  # noqa: E402
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.features import LagrangianHostFeatures  # noqa: E402

SOURCES = ("hr", "base")


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/features/test_host_subhalo_tiles.py)
# --------------------------------------------------------------------------

def hr_children_of_sites(lr_ids, grid) -> np.ndarray:
    """The ``upsample**3`` HR particle ids of every LR site in ``lr_ids``.

    The vectorised form of :meth:`LagrangianGrid.hr_children`, which takes one
    site at a time. Row ``k`` of the (flattened) result is site ``lr_ids[k]``'s
    block, so a per-site quantity can be broadcast with ``repeat``.
    """
    ids = np.asarray(lr_ids, dtype=np.int64).reshape(-1)
    n, f, ngh = grid.ng_lr, grid.upsample, grid.ng_hr
    a, b, c = ids // (n * n), (ids // n) % n, ids % n
    u = np.arange(f, dtype=np.int64)
    ix = (f * a[:, None, None, None] + u[None, :, None, None])
    iy = (f * b[:, None, None, None] + u[None, None, :, None])
    iz = (f * c[:, None, None, None] + u[None, None, None, :])
    return ((ix * ngh + iy) * ngh + iz).reshape(-1)


def catalog_tables(cat, n_ids: int = 0):
    """``(root_of_id, num_p_of_id, mvir_of_id, is_sub_of_id)`` indexed by catalog id.

    Dense id-indexed tables, so the per-particle joins below are array lookups
    rather than dict misses in a 130-million-element loop. ``n_ids`` extends them
    past the catalog's largest id, so indexing them with an owner array cannot
    go out of bounds even if that array names an object the catalog does not
    list (it should not; the tables then report it as unowned rather than
    raising a hundred million rows later).
    """
    n = max(int(cat.ids.max()) + 1 if cat.n else 1, int(n_ids))
    root = root_lookup(cat).astype(np.int64)
    if root.size < n:                                   # pad, never truncate
        root = np.concatenate([root, np.full(n - root.size, -1, np.int64)])
    num_p = np.zeros(n, dtype=np.int64)
    mvir = np.zeros(n, dtype=np.float64)
    is_sub = np.zeros(n, dtype=bool)
    num_p[cat.ids] = cat.num_p
    mvir[cat.ids] = cat.mvir
    is_sub[cat.ids] = cat.parent_ids >= 0
    return root, num_p, mvir, is_sub


class RootTileSubhalos:
    """Per-tile fractional subhalo count of every top-level host, from ``tilew``.

    ``weight[s, t]`` in the tile-weight file is the fraction of subhalo ``s``'s
    members whose Lagrangian site is in tile ``t`` (rows sum to 1), so summing it
    over the substructure of one host gives that host's subhalos split over
    tiles, and summing *that* over tiles returns the host's subhalo count
    exactly. Whole-box and unrestricted, which is the point: a footprint-limited
    count is biased against whichever side builds *larger* subhalos, because more
    of each one then falls outside the footprint.
    """

    def __init__(self, z, root: np.ndarray, is_sub: np.ndarray, n_tiles: int):
        hid = np.asarray(z["halo_id"], dtype=np.int64)
        tid = np.asarray(z["tile_id"], dtype=np.int64)
        w = np.asarray(z["weight"], dtype=np.float64)
        keep = is_sub[hid] & (root[hid] >= 0)
        key = root[hid[keep]] * int(n_tiles) + tid[keep]
        uk, inv = np.unique(key, return_inverse=True)
        self.n_tiles = int(n_tiles)
        self.count = np.bincount(inv, weights=w[keep])
        self.root = uk // int(n_tiles)          # sorted, ascending
        self.tile = uk % int(n_tiles)

    def of(self, host_id: int):
        """``(tile_ids, counts)`` for one top-level host, most populated first."""
        lo = int(np.searchsorted(self.root, int(host_id), side="left"))
        hi = int(np.searchsorted(self.root, int(host_id), side="right"))
        t, c = self.tile[lo:hi], self.count[lo:hi]
        o = np.argsort(-c, kind="stable")
        return t[o], c[o]


def subhalo_lagrangian_centres(owner, grid, is_sub, n_ids: int, *,
                               chunk: int = 4_000_000):
    """Periodic Lagrangian centroid of every subhalo, indexed by catalog id.

    Panel 1 of the viewer is a slice of the **Lagrangian** lattice, so a
    subhalo's Rockstar centre -- an Eulerian position -- cannot be drawn on it.
    What can is the mean lattice site its own member particles came from, which
    is what this returns: "where in the initial conditions did this subhalo's
    material start".

    The mean is circular per axis, so a subhalo whose material straddles the box
    seam lands on the seam instead of in the middle of the box. Returns
    ``(pos_mpc_h[(n_ids, 3)], n_members[n_ids])``; rows of objects that are not
    subhalos, or own no particle, stay at 0 with ``n_members == 0``.
    """
    owner = np.asarray(owner)
    ngh = int(grid.ng_hr)
    sin_ = np.zeros((int(n_ids), 3))
    cos_ = np.zeros((int(n_ids), 3))
    cnt = np.zeros(int(n_ids), dtype=np.int64)
    for s in range(0, owner.size, int(chunk)):
        o = owner[s:s + int(chunk)]
        bound = o >= 0
        if not bound.any():
            continue
        ids = o[bound].astype(np.int64)
        keep = is_sub[ids]
        ids = ids[keep]
        if ids.size == 0:
            continue
        pid = (np.arange(s, s + o.size, dtype=np.int64)[bound])[keep]
        for a, coord in enumerate((pid // (ngh * ngh), (pid // ngh) % ngh,
                                   pid % ngh)):
            th = (2.0 * np.pi / ngh) * (coord + 0.5)
            sin_[:, a] += np.bincount(ids, weights=np.sin(th), minlength=n_ids)
            cos_[:, a] += np.bincount(ids, weights=np.cos(th), minlength=n_ids)
        cnt += np.bincount(ids, minlength=int(n_ids))
    ang = np.arctan2(sin_, cos_)
    pos = np.mod(ang / (2.0 * np.pi), 1.0) * float(grid.boxsize_mpc_h)
    pos[cnt == 0] = 0.0
    return pos, cnt


def host_tile_subhalos(lr_ids, grid, owner, root, num_p, is_sub, *,
                       n_tiles: int, chunk_sites: int = 4096) -> dict:
    """Per-tile subhalo count of the host owning ``lr_ids``, one source.

    ``lr_ids`` is the LR host's Lagrangian footprint. The matched host is the
    top-level object that binds the most of that footprint in this source's
    catalog; only its own substructure is counted. Returns the match plus the
    per-tile fractional counts.
    """
    lr_ids = np.asarray(lr_ids, dtype=np.int64).reshape(-1)
    f3 = grid.upsample ** 3
    site_tile = grid.tile_of_lr_site(lr_ids).astype(np.int64)

    # Pass 1: which top-level object owns the most of this footprint.
    votes: dict[int, int] = {}
    n_bound = 0
    for s in range(0, lr_ids.size, chunk_sites):
        blk = lr_ids[s:s + chunk_sites]
        own = owner[hr_children_of_sites(blk, grid)]
        good = own >= 0
        n_bound += int(good.sum())
        if not good.any():
            continue
        r = root[own[good]]
        u, c = np.unique(r[r >= 0], return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            votes[k] = votes.get(k, 0) + v
    n_part = int(lr_ids.size) * f3
    out = {
        "n_hr_particles": n_part,
        "n_bound": n_bound,
        "match": None,
        "tile_id": [],
        "tile_sub": [],
        "sub_total": 0.0,
        "tile_sub_footprint": [],
        "sub_total_footprint": 0.0,
    }
    if not votes:
        return out
    match = int(max(votes, key=votes.get))
    shared = int(votes[match])

    # Pass 2: fractional subhalo count per tile, this host's material only.
    cnt = np.zeros(n_tiles, dtype=np.float64)
    for s in range(0, lr_ids.size, chunk_sites):
        blk = lr_ids[s:s + chunk_sites]
        own = owner[hr_children_of_sites(blk, grid)]
        tl = np.repeat(site_tile[s:s + chunk_sites], f3)
        good = own >= 0
        leaf = own[good].astype(np.int64)
        # Only substructure, and only substructure of the matched host: a
        # neighbouring host's satellite sharing this tile must not be counted.
        sel = is_sub[leaf] & (root[leaf] == match)
        if not sel.any():
            continue
        cnt += np.bincount(tl[good][sel], weights=1.0 / num_p[leaf[sel]],
                           minlength=n_tiles)
    out.update(
        match={
            "halo_id": match,
            "n_shared_particles": shared,
            "match_frac": float(shared / max(n_part, 1)),
            "match_frac_bound": float(shared / max(n_bound, 1)),
        },
        footprint_count=cnt,
        sub_total_footprint=float(cnt.sum()),
    )
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _load_source(root: Path, box: str, tag: str):
    work = root / f"{box}__{tag}__{tag}"
    own_p = work / f"{box}_{tag}_owner.npy"
    hits = sorted(glob.glob(str(work / f"{tag}_rockstar" / "halos*.ascii"))
                  + glob.glob(str(work / f"{tag}_rockstar" / "halos*.list")))
    if not own_p.is_file() or not hits:
        return None
    cat = load_rockstar_ascii(hits[0])
    z = np.load(work / f"{box}_{tag}_tilew.npz")
    rec = np.zeros(int(max(cat.ids.max(), z["member_halo_id"].max())) + 1,
                   dtype=np.int64)
    rec[z["member_halo_id"]] = z["member_count"]
    return {
        "cat": cat, "owner": np.load(own_p), "recursive_num_p": rec, "tilew": z,
        "owner_path": str(own_p), "catalog_path": hits[0],
    }


def collect(box: str, cfg, args) -> dict:
    banner(f"host subhalo tiles {box}")
    src = paths.subdir("lagrangian_host", box) / f"{box}_lagrangian_host.npz"
    if not src.is_file():
        raise SystemExit(f"no cached features at {src}; run "
                         f"scripts/features/build_lagrangian_host.py first")
    feat = LagrangianHostFeatures.from_npz(src)
    g, t = feat.grid, feat.table
    n_tiles = g.n_tiles
    site_row = feat.host_index.reshape(-1)

    order = np.argsort(-t.mvir)
    if int(args.n_hosts) > 0:
        order = order[:int(args.n_hosts)]
    rows = order.tolist()

    # Group the lattice by host once instead of scanning it per host.
    srt = np.argsort(site_row, kind="stable")
    sorted_rows = site_row[srt]
    lo = np.searchsorted(sorted_rows, np.arange(t.n_hosts), side="left")
    hi = np.searchsorted(sorted_rows, np.arange(t.n_hosts), side="right")

    root_dir = paths.subdir("halos_particles")
    centres: dict[str, dict] = {}
    per_source: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    for tag in SOURCES:
        s = _load_source(root_dir, box, tag)
        if s is None:
            print(f"    GATE: no owner array + catalog for {box}/{tag}; "
                  f"skipping this source")
            continue
        cat = s["cat"]
        root, num_p, mvir, is_sub = catalog_tables(
            cat, n_ids=int(s["owner"].max()) + 1)
        print(f"    {tag}: {cat.n} objects, "
              f"{int((cat.parent_ids < 0).sum())} hosts, "
              f"{int((cat.parent_ids >= 0).sum())} subhalos, "
              f"owner {s['owner'].size} particles")
        rts = RootTileSubhalos(s["tilew"], root, is_sub, n_tiles)
        res = {}
        for k, row in enumerate(rows):
            lr_ids = srt[lo[row]:hi[row]].astype(np.int64)
            r = host_tile_subhalos(lr_ids, g, s["owner"], root, num_p, is_sub,
                                   n_tiles=n_tiles)
            if r["match"] is not None:
                mid0 = int(r["match"]["halo_id"])
                # Whole-host counts (the ones the page plots), plus the
                # footprint-restricted ones on exactly the same tile list so the
                # two are read side by side.
                tiles_w, cnts_w = rts.of(mid0)
                fp = r.pop("footprint_count")
                extra = [t for t in np.flatnonzero(fp > 0) if t not in set(tiles_w)]
                tiles_all = np.concatenate(
                    [tiles_w, np.asarray(extra, dtype=np.int64)])
                cnts_all = np.concatenate(
                    [cnts_w, np.zeros(len(extra), dtype=np.float64)])
                r["tile_id"] = [int(t) for t in tiles_all]
                r["tile_sub"] = [float(c) for c in cnts_all]
                r["sub_total"] = float(cnts_w.sum())
                r["tile_sub_footprint"] = [float(fp[t]) for t in tiles_all]
                mid = int(r["match"]["halo_id"])
                r["match"]["log_mvir"] = float(np.log10(max(mvir[mid], 1.0)))
                r["match"]["num_p"] = int(num_p[mid])
                r["match"]["recursive_num_p"] = int(s["recursive_num_p"][mid])
                r["match"]["is_sub_of_another_host"] = bool(is_sub[mid])
                r["match"]["n_sub_catalog"] = int(
                    ((root[cat.ids] == mid) & (cat.parent_ids >= 0)).sum())
                denom = max(int(s["recursive_num_p"][mid]), 1)
                r["match"]["share_of_match"] = float(
                    r["match"]["n_shared_particles"] / denom)
            else:
                r.pop("footprint_count", None)
            res[row] = r
            if (k + 1) % 100 == 0 or k + 1 == len(rows):
                print(f"      {tag}: {k + 1}/{len(rows)} hosts", flush=True)
        # How many LR hosts landed on the same object: a shared match means the
        # per-tile counts of those hosts overlap where their tiles do.
        seen: dict[int, int] = {}
        for r in res.values():
            if r["match"] is not None:
                mid = int(r["match"]["halo_id"])
                seen[mid] = seen.get(mid, 0) + 1
        for r in res.values():
            if r["match"] is not None:
                r["match"]["n_lr_hosts_sharing"] = int(
                    seen[int(r["match"]["halo_id"])])
        # Lagrangian centres of every subhalo, for panel 1's overlay. Computed
        # here because this is where the owner array is already in memory; one
        # extra streaming pass over it, no second load.
        pos, nmem = subhalo_lagrangian_centres(
            s["owner"], g, is_sub, int(is_sub.size))
        have = nmem > 0
        exact = int((nmem[have] == num_p[have]).sum())
        print(f"      {tag}: centred {int(have.sum())} subhalos, "
              f"{exact} with len(members) == num_p")
        # Which selectable LR host (if any) each subhalo belongs to, so the page
        # can filter the overlay down to the host in the selector.
        row_of_match = {int(r["match"]["halo_id"]): int(row)
                        for row, r in res.items() if r["match"] is not None}
        ids = np.flatnonzero(have)
        rows_u16 = np.full(ids.size, 65535, dtype=np.uint16)
        for k, i in enumerate(ids):
            rows_u16[k] = row_of_match.get(int(root[i]), 65535)
        centres[tag] = {
            "pos": np.clip(np.round(pos[ids] / float(g.boxsize_mpc_h) * 65535.0),
                           0, 65535).astype(np.uint16),
            # Capped at the uint16 limit: the page uses it only to size a dot
            # (log-scaled and clamped at 6.5 px), never as a reported count, and
            # the few subhalos above 65535 particles are already at the cap.
            "num_p": np.clip(num_p[ids], 0, 65535).astype(np.uint16),
            "host_row": rows_u16,
            "halo_id": ids.astype(np.int64),
            "n_members_match_num_p": exact,
        }
        per_source[tag] = res
        provenance[tag] = {"owner": s["owner_path"], "catalog": s["catalog_path"],
                           "n_objects": int(cat.n)}
        del s

    hosts = []
    for row in rows:
        e = {"row": int(row), "lr_host_id": int(t.host_id[row]),
             "lr_log_mvir": float(np.log10(t.mvir[row])),
             "n_lr_sites": int(t.n_particles[row]),
             "lr_n_sub": int(t.n_sub[row])}
        for tag, key in (("hr", "hr"), ("base", "sr2")):
            e[key] = per_source.get(tag, {}).get(row)
        hosts.append(e)

    def total(key, field="sub_total"):
        return float(sum(h[key][field] for h in hosts if h.get(key)))
    summary = {
        "n_hosts": len(hosts),
        "hr_sub_total": total("hr"),
        "sr2_sub_total": total("sr2"),
        "ratio": (total("sr2") / total("hr")) if total("hr") > 0 else None,
        "hr_sub_total_footprint": total("hr", "sub_total_footprint"),
        "sr2_sub_total_footprint": total("sr2", "sub_total_footprint"),
        "sources_present": sorted(per_source),
    }
    out = {"box": box, "n_tiles": n_tiles, "upsample": g.upsample,
           "summary": summary, "provenance": provenance, "hosts": hosts}
    dest = paths.subdir("lagrangian_host", box, create=True) \
        / f"{box}_host_subhalo_tiles.json"
    write_json(dest, out)

    # The centres go to an npz, not the JSON: 147k subhalos x 4 fields is binary
    # data, and JSON would triple it and slow the page's parse for nothing.
    npz = dest.parent / f"{box}_subhalo_centres.npz"
    flat = {"boxsize_mpc_h": np.array(float(g.boxsize_mpc_h)),
            "ng_lr": np.array(int(g.ng_lr))}
    for tag, key in (("hr", "hr"), ("base", "sr2")):
        c = centres.get(tag)
        if c is None:
            continue
        for field in ("pos", "num_p", "host_row", "halo_id"):
            flat[f"{key}_{field}"] = c[field]
    np.savez_compressed(npz, **flat)
    out["subhalo_centres"] = {
        "path": str(npz),
        "n": {("sr2" if k == "base" else k): int(v["halo_id"].size)
              for k, v in centres.items()},
        "n_members_match_num_p": {
            ("sr2" if k == "base" else k): int(v["n_members_match_num_p"])
            for k, v in centres.items()},
    }
    print(f"    wrote {npz} ("
          + ", ".join(f"{k}:{len(v['halo_id'])}" for k, v in centres.items())
          + " centred subhalos)")
    print(f"    over {summary['n_hosts']} LR hosts, matched hosts hold: "
          f"HR {summary['hr_sub_total']:.1f}  SR2 {summary['sr2_sub_total']:.1f}"
          f"  ratio " + (f"{summary['ratio']:.3f}" if summary["ratio"] else "n/a"))
    print(f"    restricted to the LR footprints: "
          f"HR {summary['hr_sub_total_footprint']:.1f}  "
          f"SR2 {summary['sr2_sub_total_footprint']:.1f}  (biased against the "
          f"side with larger subhalos; reported for reference)")
    matched = [h for h in hosts if h.get("hr")]
    if matched:
        mf = np.array([h["hr"]["match"]["match_frac"] for h in matched
                       if h["hr"]["match"]])
        print(f"    HR match fraction of the LR footprint: median "
              f"{np.median(mf):.3f}, p10 {np.quantile(mf, 0.1):.3f}")
    print(f"    wrote {dest}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    ap.add_argument("--n-hosts", type=int, default=0,
                    help="most massive LR hosts to process; 0 (default) = all")
    args = ap.parse_args(argv)
    cfg = load_reward_config(args)
    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        collect(box, cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
