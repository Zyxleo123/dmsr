"""Render the Lagrangian host features as one self-contained interactive page.

Why a static page and not Streamlit
-----------------------------------
This repository already renders its interactive views this way
(``scripts/reward/render_overdensity_html.py``): every array and coordinate is
embedded, so the file opens in any browser after an ``scp`` and needs no server,
no port on the login node, and no extra dependency in the ``pjm`` env. A
Streamlit app would need a long-lived process on a machine that kills inline
compute, which is the wrong shape for this cluster.

It is a pure redraw of the cached ``*_lagrangian_host.npz``: rerun it to change
the host cap, the sampling or the colours without touching Rockstar.

    python scripts/features/render_lagrangian_host_app.py --boxes set8
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (  # noqa: E402
    DEFAULT_CONFIG, banner, load_reward_config, lr_path, paths,
)

from cosmo_sr.eval.particle_identity import periodic_delta  # noqa: E402
from cosmo_sr.eval.particles import field_to_particles  # noqa: E402
from cosmo_sr.features import (  # noqa: E402
    LagrangianHostFeatures, lagrangian_lattice_positions, normalization_report,
)


def _b64_u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def _b64_u16(a: np.ndarray) -> str:
    """Little-endian uint16, to match the JS ``Uint16Array`` view on the bytes."""
    return base64.b64encode(
        np.ascontiguousarray(a, dtype="<u2").tobytes()).decode("ascii")


def _quantize(vol: np.ndarray, mask: np.ndarray) -> tuple[str, float, float]:
    """Volume -> uint8 with 0 reserved for "no host", plus the value range.

    Reserving 0 keeps "this site has no host" distinguishable from "this site
    has the smallest value in range", which matters because 0 is a legitimate
    value of every channel here (a site at its host's centre, a host with no
    budget).
    """
    v = np.asarray(vol, dtype=np.float64)
    if not mask.any():
        return _b64_u8(np.zeros(v.size)), 0.0, 1.0
    lo, hi = float(v[mask].min()), float(v[mask].max())
    if hi <= lo:
        hi = lo + 1.0
    q = np.zeros(v.shape, dtype=np.uint8)
    q[mask] = 1 + np.clip(
        np.round(254.0 * (v[mask] - lo) / (hi - lo)), 0, 254).astype(np.uint8)
    return _b64_u8(q.reshape(-1)), lo, hi


def eulerian_payload(feat: LagrangianHostFeatures, field_path: Path) -> dict:
    """Displaced positions of every LR particle, plus the displacement field.

    The features are Lagrangian scalars; where a particle physically *ends up*
    is not in them, it is in the LR field itself. Both are indexed by the same
    LR particle id, so the join is an array lookup and no matching is involved.

    Positions are quantised to a byte over the box (0.39 Mpc/h, a quarter of an
    LR cell), which is far finer than anything the panel resolves and keeps the
    page around 3 MB.
    """
    g = feat.grid
    box = float(g.boxsize_mpc_h)
    field = np.load(field_path, mmap_mode="r")
    if int(field.shape[1]) != g.ng_lr:
        raise SystemExit(
            f"{field_path} is {field.shape[1]}^3 but the features are "
            f"{g.ng_lr}^3; they are not the same box")
    pb = field_to_particles(np.asarray(field, dtype=np.float32),
                            boxsize_kpc_h=box * 1000.0, redshift=0.0)
    x = pb.pos_mpc_h.astype(np.float64)
    q = lagrangian_lattice_positions(g)
    psi = periodic_delta(x, q, box)
    dmag = np.linalg.norm(psi, axis=1)

    pos_u8 = np.clip(np.round(255.0 * (x % box) / box), 0, 255).astype(np.uint8)
    return {
        "pos": _b64_u8(pos_u8.reshape(-1)),          # (N,3) row-major, /255*box
        "disp": {"data": _b64_u8(np.clip(np.round(
                     255.0 * dmag / max(dmag.max(), 1e-9)), 0, 255).astype(np.uint8)),
                 "lo": 0.0, "hi": float(dmag.max()),
                 "label": "|displacement| [Mpc/h]"},
        "stats": {
            "median": float(np.median(dmag)),
            "p90": float(np.quantile(dmag, 0.9)),
            "max": float(dmag.max()),
            "field": str(field_path),
        },
    }


def subcentres_payload(path: Path) -> dict:
    """The subhalo Lagrangian centres npz, as base64 uint16 for the page.

    Positions are stored as a fraction of the box in uint16 (0.0015 Mpc/h), far
    finer than the LR cell panel 1 draws them on, and ``host_row`` is the index
    into the page's own host list (65535 = this subhalo belongs to no selectable
    LR host), so the overlay can be filtered to the selected host without a
    second lookup table.
    """
    z = np.load(path)
    out = {"boxsize": float(z["boxsize_mpc_h"]), "ng_lr": int(z["ng_lr"])}
    for key in ("hr", "sr2"):
        if f"{key}_pos" not in z:
            out[key] = None
            continue
        out[key] = {
            "n": int(z[f"{key}_num_p"].size),
            "pos": _b64_u16(z[f"{key}_pos"].reshape(-1)),
            "num_p": _b64_u16(z[f"{key}_num_p"]),
            "host_row": _b64_u16(z[f"{key}_host_row"]),
        }
    return out


def build_payload(feat: LagrangianHostFeatures, *, n_hosts: int,
                  n_sample: int, seed: int) -> dict:
    g, t = feat.grid, feat.table
    mask = feat.host_member > 0
    ng = g.ng_lr

    # --- the hosts the page can select -------------------------------------
    # n_hosts <= 0 means every host. Shipping all of them costs ~240 kB at 64^3
    # (the per-host sample cap only ever binds for the few largest), so the
    # default is "all" and the page filters rather than the renderer.
    order = np.argsort(-t.mvir)
    if int(n_hosts) > 0:
        order = order[:int(n_hosts)]
    rng = np.random.default_rng(int(seed))

    site_row = feat.host_index.reshape(-1)
    lam = (feat.subhalo_budget.reshape(-1) if feat.subhalo_budget is not None
           else np.zeros(g.n_lr, dtype=np.float32))
    hosts = []
    palette = np.zeros(g.n_lr, dtype=np.uint8)   # 0 none, 1..K selected, 255 other
    palette[(site_row >= 0)] = 255

    for k, row in enumerate(order):
        ids = np.flatnonzero(site_row == row).astype(np.int64)
        palette[ids] = np.uint8(1 + (k % 254))
        take = ids if ids.size <= n_sample else rng.choice(ids, n_sample, replace=False)
        take = np.sort(take)
        a, b, c = take // (ng * ng), (take // ng) % ng, take % ng
        coords = np.stack([a, b, c], axis=1).astype(np.uint8).reshape(-1)

        tiles, fracs = t.tiles_of(int(row))
        # Budget actually allocated to each tile: sum of lambda over the tile's
        # share of this host. Equals N_h * f[h,t] by construction, but summed
        # from the per-site channel so the page shows the measured number.
        site_tiles = g.tile_of_lr_site(ids)
        counts = np.array([int(np.count_nonzero(site_tiles == tt)) for tt in tiles],
                          dtype=np.int64)
        alloc = np.array(
            [float(lam[ids][site_tiles == tt].sum()) for tt in tiles],
            dtype=np.float64)

        hosts.append({
            "k": k + 1,
            "row": int(row),
            "host_id": int(t.host_id[row]),
            "mvir": float(t.mvir[row]),
            "log_mvir": float(np.log10(t.mvir[row])),
            "rvir_kpc_h": float(t.rvir_kpc_h[row]),
            "num_p_catalog": int(t.num_p_catalog[row]),
            "n_particles": int(t.n_particles[row]),
            "n_sampled": int(take.size),
            "center_lag": [float(x) for x in t.center_lag[row]],
            "center_cell": [float(x) / g.cell_mpc_h for x in t.center_lag[row]],
            "r_lag": float(t.r_lag_mpc_h[row]),
            "rms_lag": float(t.rms_lag_mpc_h[row]),
            "n_sub": int(t.n_sub[row]),
            "lam": float(lam[ids][0]) if ids.size else 0.0,
            "tiles": [int(x) for x in tiles],
            "tile_frac": [float(x) for x in fracs],
            "tile_count": [int(x) for x in counts],
            "tile_alloc": [float(x) for x in alloc],
            "frac_sum": float(fracs.sum()),
            "alloc_sum": float(alloc.sum()),
            "coords": _b64_u8(coords),
        })

    # --- volumes for the slice views ---------------------------------------
    dq_mag = np.linalg.norm(feat.dq_over_rl, axis=0)
    mass_b64, mass_lo, mass_hi = _quantize(feat.log_host_mass, mask)
    lam_b64, lam_lo, lam_hi = _quantize(
        feat.subhalo_budget if feat.subhalo_budget is not None
        else np.zeros_like(feat.log_host_mass), mask)
    dq_b64, dq_lo, dq_hi = _quantize(dq_mag, mask)
    frac_b64, frac_lo, frac_hi = _quantize(feat.host_fraction_per_tile, mask)

    rep = normalization_report(feat)
    return {
        "box": feat.box,
        "grid": {
            "ng_lr": g.ng_lr, "ng_hr": g.ng_hr, "upsample": g.upsample,
            "tile_lr": g.tile_lr, "tile_hr": g.tile_hr,
            "n_per_axis": g.n_per_axis, "n_tiles": g.n_tiles,
            "boxsize": g.boxsize_mpc_h, "cell": g.cell_mpc_h,
        },
        "report": {k: v for k, v in rep.items() if k != "channels"},
        "channels": rep["channels"],
        "n_sub_source": t.n_sub_source,
        "vol": {
            "log_host_mass": {"data": mass_b64, "lo": mass_lo, "hi": mass_hi,
                              "label": "log10 Mvir [Msun/h]"},
            "subhalo_budget": {"data": lam_b64, "lo": lam_lo, "hi": lam_hi,
                               "label": "lambda_i (subhalo budget)"},
            "dq_over_rl": {"data": dq_b64, "lo": dq_lo, "hi": dq_hi,
                           "label": "|dq| / R_L"},
            "host_fraction_per_tile": {"data": frac_b64, "lo": frac_lo,
                                       "hi": frac_hi,
                                       "label": "host fraction in own tile"},
        },
        # The host table row of each site, as uint16 (65535 = no host). This
        # replaces an earlier uint8 palette, which wrapped at 254 hosts and made
        # two distinct hosts share an index -- with every host selectable that
        # would highlight the wrong sites in the slice panel.
        "rowidx": _b64_u16(np.where(site_row >= 0, site_row, 65535)),
        "n_hosts_total": int(t.n_hosts),
        "hosts": hosts,
    }


_TEMPLATE = r"""<title>Lagrangian host features &middot; __BOX__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.55 system-ui, sans-serif;
         background: #0d0d10; color: #e8e8ec; }
  .wrap { max-width: 1280px; margin: 0 auto; padding: 20px 22px 60px; }
  h1 { font-size: 19px; margin: 0 0 2px; }
  h2 { font-size: 14px; margin: 0 0 8px; color: #cfd3da;
       text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
  .sub { color: #9aa0aa; margin: 0 0 16px; }
  .bar { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-end;
         padding: 12px 14px; background: #15151b; border: 1px solid #2a2a30;
         border-radius: 8px; margin-bottom: 18px; position: sticky; top: 0;
         z-index: 5; }
  .fld { display: flex; flex-direction: column; gap: 4px; }
  .fld label { font-size: 11px; color: #9aa0aa; text-transform: uppercase;
               letter-spacing: .05em; }
  select, input[type=range] { background: #22222a; color: #e8e8ec;
      border: 1px solid #3a3a44; border-radius: 5px; padding: 5px 7px;
      font: inherit; min-width: 150px; }
  input[type=range] { padding: 0; }
  .grid { display: grid; gap: 18px; grid-template-columns: repeat(2, 1fr); }
  .card { background: #15151b; border: 1px solid #2a2a30; border-radius: 8px;
          padding: 14px; min-width: 0; }
  .card.full { grid-column: 1 / -1; }
  .note { color: #8b909b; font-size: 12px; margin: 8px 0 0; }
  canvas { display: block; width: 100%; height: auto; background: #08080b;
           border-radius: 5px; border: 1px solid #24242c; touch-action: none; }
  .row { display: flex; gap: 14px; flex-wrap: wrap; }
  .row > div { flex: 1 1 240px; min-width: 0; }
  .cap { font-size: 11px; color: #8b909b; margin: 6px 0 0; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; }
  th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #24242c; }
  th { color: #9aa0aa; font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .ok { color: #6ee7a8; } .bad { color: #ff8b8b; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
            vertical-align: -1px; margin-right: 6px; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11px;
            color: #8b909b; margin-top: 8px; }
</style>

<div class="wrap">
  <h1>LR-Rockstar host features on the Lagrangian lattice &middot; __BOX__</h1>
  <p class="sub" id="subtitle"></p>

  <div class="bar">
    <div class="fld"><label for="massbin">Host mass</label>
      <select id="massbin">
        <option value="all" selected>all masses</option>
        <option value="14">&ge; 1e14</option>
        <option value="13">1e13 &ndash; 1e14</option>
        <option value="12">1e12 &ndash; 1e13</option>
        <option value="0">&lt; 1e12</option>
      </select></div>
    <div class="fld"><label for="host">Host <span id="hcount"></span></label>
      <select id="host"></select></div>
    <div class="fld"><label for="tile">Tile</label>
      <select id="tile"></select></div>
    <div class="fld"><label for="chan">Channel</label>
      <select id="chan"></select></div>
    <div class="fld"><label for="slice">LR slice (x index) <span id="slabel"></span></label>
      <input type="range" id="slice" min="0" max="63" value="32"></div>
    <div class="fld"><label for="subs">Subhalo centres (panel 1)</label>
      <select id="subs">
        <option value="both" selected>HR + SR2</option>
        <option value="hr">HR only</option>
        <option value="sr2">SR2 only</option>
        <option value="off">off</option>
      </select></div>
    <div class="fld"><label for="subscope">Centres from</label>
      <select id="subscope">
        <option value="all" selected>all hosts</option>
        <option value="host">selected host only</option>
      </select></div>
    <div class="fld"><label for="others">Neighbours in 3D</label>
      <select id="others">
        <option value="1" selected>show other hosts</option>
        <option value="0">selected host only</option>
      </select></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>1 &middot; LR lattice slice &rarr; broadcast HR tile</h2>
      <div class="row">
        <div>
          <canvas id="cslice" width="512" height="512"></canvas>
          <p class="cap" id="capslice"></p>
        </div>
        <div>
          <canvas id="chr" width="512" height="512"></canvas>
          <p class="cap" id="caphr"></p>
        </div>
      </div>
      <p class="note">The right panel is the selected tile broadcast to HR: each
        LR site's value is repeated over its
        <span id="upcube"></span> HR children, which is the only broadcast
        consistent with a feature defined per LR site. Faint lines mark the LR
        cell borders inside the tile.</p>
      <p class="note">Circles are <strong>subhalo centres</strong> &mdash; blue
        HR, red SR2 &mdash; sized by member count. Both panels are Lagrangian, so
        a subhalo's Rockstar centre (an Eulerian position) cannot be drawn on
        them; what is drawn is the <em>circular mean of the lattice sites its own
        member particles came from</em>, i.e. where in the initial conditions its
        material started. Only subhalos whose centre falls in the LR cell plane
        on screen are shown, so stepping the slider walks the slab through the
        box. Where blue circles have no red neighbour, SR2 built no subhalo out
        of that material.</p>
    </div>

    <div class="card">
      <h2>2 &middot; Where this tile's material ends up (displaced)</h2>
      <canvas id="ceul" width="1040" height="820"></canvas>
      <p class="cap" id="capeul"></p>
      <div class="fld" style="margin:10px 0">
        <label for="eulcol">Colour by</label>
        <select id="eulcol">
          <option value="log_host_mass" selected>final host mass</option>
          <option value="disp">|displacement|</option>
        </select>
      </div>
      <table id="teul"></table>
      <p class="note">The same 512 LR particles as the selected tile, drawn at
        their <strong>displaced</strong> positions instead of their lattice
        sites. Drag to rotate. The wireframe is where the tile started, so the
        offset between cube and cloud is the tile's bulk translation and the
        cloud's shape is what the displacement field did to it. Colour is the
        mass of the halo each particle finally lands in &mdash; grey means it
        joins no halo. This is the only Eulerian panel; every other one is
        Lagrangian.</p>
    </div>

    <div class="card">
      <h2>3 &middot; Lagrangian 3D view</h2>
      <canvas id="c3d" width="1040" height="1040"></canvas>
      <p class="cap" id="cap3d"></p>
      <p class="note">Drag to rotate. Axes are LR lattice cells, centred on the
        host's periodic Lagrangian centre. Wireframe boxes are the SR2 tiles the
        host intersects. <code>log_host_mass</code> and <code>subhalo_budget</code>
        are constant within one host by construction, so they only vary here when
        neighbours are shown; <code>|dq|/R_L</code> varies inside the host.</p>
    </div>

    <div class="card">
      <h2>4 &middot; One host across tiles</h2>
      <canvas id="ctiles" width="1040" height="620"></canvas>
      <p class="note">Every tile of the box, laid out as
        <span id="tlayout"></span> planes of the tile lattice. Shaded cells hold
        part of the selected host; the number is that tile's share of the host.</p>
    </div>

    <div class="card">
      <h2>5 &middot; SR2 subhalo deficit per tile</h2>
      <canvas id="cdef" width="1040" height="700"></canvas>
      <p class="cap" id="capdef"></p>
      <table id="tdef"></table>
      <p class="note">Every tile, same layout as panel 4, shaded by the
        <strong>relative shortfall</strong> of subhalos,
        (N<sub>SR2</sub> &minus; N<sub>HR</sub>) / N<sub>HR</sub>. Red is short
        of HR, blue is over; white is parity. Counts are fractional &mdash; a
        subhalo is split across the tiles its Lagrangian material came from, so
        each contributes exactly 1 across the box. The selected tile is outlined.</p>
    </div>

    <div class="card">
      <h2>6 &middot; Fraction and budget per tile</h2>
      <canvas id="cbar" width="1040" height="620"></canvas>
      <p class="note">Left bar: fraction of the host's LR particles in the tile
        (sums to 1). Right bar: subhalo budget allocated to the tile, summed from
        the per-particle &lambda; (sums to N<sub>h</sub>).</p>
    </div>

    <div class="card full">
      <h2>7 &middot; Subhalos of <em>this host</em> per tile &mdash; HR vs SR2</h2>
      <canvas id="csub" width="1040" height="560"></canvas>
      <p class="cap" id="capsub"></p>
      <table id="tsub"></table>
      <p class="note">Panel 5 counts every subhalo in a tile; this counts only
        the subhalos of <strong>the selected host</strong> &mdash; a neighbouring
        host's satellite sharing the tile is excluded by construction. The LR
        host is matched to an HR and an SR2 host by Lagrangian material (the
        object that binds most of this host's particles, via
        <code>owner[particle_id]</code>, not by position), and only objects whose
        top-level ancestor <em>is</em> that match are counted. A subhalo is then
        split fractionally over the tiles its Lagrangian material came from, so
        the bars sum to the matched host's full subhalo count. The table also
        gives the count restricted to this LR host's own footprint; read it as a
        cross-check only, since restricting is biased against whichever side
        builds larger subhalos. Watch <em>LR host is this share of the match</em>:
        when it is small, one HR host swallowed several LR structures and its
        subhalos are not all this host's doing.</p>
    </div>

    <div class="card full">
      <h2>8 &middot; Host record and normalisation checks</h2>
      <div class="row">
        <div><table id="thost"></table></div>
        <div><table id="tbox"></table></div>
      </div>
    </div>
  </div>
</div>

<script>
const D = __PAYLOAD__;
const G = D.grid, NG = G.ng_lr, NT = G.n_per_axis, S = G.tile_lr;

function unb64(s) {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function unb64u16(s) {
  const b = unb64(s);
  return new Uint16Array(b.buffer, b.byteOffset, b.length / 2);
}
for (const k in D.vol) D.vol[k].u8 = unb64(D.vol[k].data);
const ROW = unb64u16(D.rowidx);      // host table row per site, 65535 = none
D.hosts.forEach(h => { h.xyz = unb64(h.coords); });

// Perceptually-ordered ramp (dark -> warm), reused by every value panel so the
// same colour means the same normalised value everywhere on the page.
const RAMP = [[13,13,16],[38,26,64],[86,32,96],[143,42,90],[196,64,63],
              [234,110,45],[249,168,58],[252,222,140],[255,255,224]];
function ramp(t) {
  t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
  const a = RAMP[i], b = RAMP[i + 1];
  return `rgb(${a[0]+(b[0]-a[0])*f|0},${a[1]+(b[1]-a[1])*f|0},${a[2]+(b[2]-a[2])*f|0})`;
}

const $ = id => document.getElementById(id);
const state = { host: 0, tile: 0, chan: "log_host_mass", slice: NG >> 1,
                others: 1, yaw: 0.6, pitch: 0.35, subs: "both", subscope: "all" };

// ---------- subhalo Lagrangian centres (panel 1 overlay) ------------------
const SC = D.subcentres;
const SC_COL = { hr: "#7aa2f7", sr2: "#f77a7a" };
if (SC) for (const k of ["hr", "sr2"]) if (SC[k]) {
  SC[k].xyz = unb64u16(SC[k].pos);      // (n,3) fraction of the box
  SC[k].np = unb64u16(SC[k].num_p);
  SC[k].row = unb64u16(SC[k].host_row); // 65535 = no selectable LR host
}
// Which sides the current toggle asks for, and are actually present.
function scSides() {
  if (!SC || state.subs === "off") return [];
  const want = state.subs === "both" ? ["hr", "sr2"] : [state.subs];
  return want.filter(k => SC[k]);
}
// LR-cell coordinate of subhalo i on axis a (0..NG), from the uint16 fraction.
function scAt(s, i, a) {
  return s.xyz[i * 3 + a] / 65535 * SC.boxsize / G.cell;
}
function scRadius(n) {
  return Math.max(1.6, Math.min(6.5, 1.6 + 1.1 * Math.log10(Math.max(n, 1) / 20)));
}
// Every subhalo of `side` whose centre sits in the LR cell plane `plane`,
// as [cy, cz, num_p] in LR-cell units. Filtered to the selected host when the
// scope control asks for it.
function scInPlane(side, plane) {
  const s = SC[side], out = [], row = H().row;
  for (let i = 0; i < s.n; i++) {
    if (state.subscope === "host" && s.row[i] !== row) continue;
    if (Math.floor(scAt(s, i, 0)) !== plane) continue;
    out.push([scAt(s, i, 1), scAt(s, i, 2), s.np[i]]);
  }
  return out;
}
function scDot(g, x, y, r, col) {
  g.beginPath(); g.arc(x, y, r, 0, 6.2832);
  g.fillStyle = col; g.globalAlpha = 0.85; g.fill();
  g.globalAlpha = 1; g.lineWidth = 1;
  g.strokeStyle = "rgba(8,8,11,.85)"; g.stroke();
}

function H() { return D.hosts[state.host]; }
function fmt(x, n) { return Number(x).toFixed(n === undefined ? 3 : n); }
function esc(s) { return String(s).replace(/[&<>]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function wrap(d, n) { return d - n * Math.round(d / n); }   // -> [-n/2, n/2)

// ---------- controls -------------------------------------------------------
function inBin(h, bin) {
  if (bin === "all") return true;
  const L = h.log_mvir;
  if (bin === "14") return L >= 14;
  if (bin === "13") return L >= 13 && L < 14;
  if (bin === "12") return L >= 12 && L < 13;
  return L < 12;
}
function fillHosts() {
  const sel = $("host"), bin = $("massbin").value;
  sel.innerHTML = "";
  let n = 0, firstIdx = null;
  D.hosts.forEach((h, i) => {
    if (!inBin(h, bin)) return;
    n++;
    if (firstIdx === null) firstIdx = i;
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `#${h.host_id}  log M=${fmt(h.log_mvir, 2)}  `
                  + `${h.n_particles} LR part  ${h.tiles.length} tiles`;
    sel.appendChild(o);
  });
  $("hcount").textContent = `(${n} of ${D.hosts.length})`;
  // Keep the current host if the filter still contains it, else take the first.
  if (!D.hosts[state.host] || !inBin(D.hosts[state.host], bin)) {
    state.host = firstIdx === null ? 0 : firstIdx;
  }
  sel.value = state.host;
}
$("massbin").onchange = () => { fillHosts(); fillHosts(); fillTiles(); centreSlice(); draw(); };
for (const k of Object.keys(D.vol)) {
  const o = document.createElement("option");
  o.value = k; o.textContent = D.vol[k].label;
  $("chan").appendChild(o);
}
$("slice").max = NG - 1;
$("upcube").textContent = `${G.upsample}^3 = ${G.upsample ** 3}`;
$("tlayout").textContent = `${NT} x ${NT} x ${NT}`;
$("subtitle").textContent =
  `LR ${NG}^3 -> HR ${G.ng_hr}^3 (x${G.upsample}), ${G.n_tiles} SR2 tiles of `
  + `${G.tile_hr}^3 HR cells (${S}^3 LR sites), box ${G.boxsize} Mpc/h, `
  + `LR cell ${fmt(G.cell, 4)} Mpc/h. ${D.hosts.length} most massive hosts of `
  + `${D.report.n_hosts} shown.`;

// ---------- per-host subhalo counts (HR vs SR2), keyed by host table row ----
const ST = D.subtiles;
const STBY = {};
if (ST) for (const e of ST.hosts) STBY[e.row] = e;
function stOf() { return ST ? (STBY[H().row] || null) : null; }
// null = this side was not collected; 0 = collected and genuinely empty.
function subOfTile(side, tile) {
  const e = stOf();
  if (!e || !e[side]) return null;
  const i = e[side].tile_id.indexOf(tile);
  return i < 0 ? 0 : e[side].tile_sub[i];
}

function fillTiles() {
  const h = H(), sel = $("tile");
  sel.innerHTML = "";
  h.tiles.forEach((t, i) => {
    const o = document.createElement("option");
    o.value = t;
    const nh = subOfTile("hr", t), ns = subOfTile("sr2", t);
    o.textContent = `tile ${t}  (${h.tile_count[i]} of ${h.n_particles} sites`
                  + `, ${(100 * h.tile_frac[i]).toFixed(1)}% of host`
                  + (nh === null && ns === null ? ""
                     : `, subs HR ${nh === null ? "?" : fmt(nh, 1)}`
                       + ` / SR2 ${ns === null ? "?" : fmt(ns, 1)}`)
                  + `)`;
    sel.appendChild(o);
  });
  state.tile = h.tiles[0];
  sel.value = state.tile;
}

$("host").onchange = e => { state.host = +e.target.value; fillHosts(); fillTiles(); centreSlice(); draw(); };
$("tile").onchange = e => { state.tile = +e.target.value; draw(); };
$("chan").onchange = e => { state.chan = e.target.value; draw(); };
$("slice").oninput = e => { state.slice = +e.target.value; draw(); };
$("others").onchange = e => { state.others = +e.target.value; draw(); };
// Panel 1 only, so redraw panel 1 rather than the 3D and Eulerian scatters.
$("subs").onchange = e => { state.subs = e.target.value; drawSlice(); drawHR(); };
$("subscope").onchange = e => {
  state.subscope = e.target.value; drawSlice(); drawHR();
};
if (!SC) {
  for (const id of ["subs", "subscope"]) {
    $(id).disabled = true;
    $(id).title = "no <box>_subhalo_centres.npz; run collect_host_subhalo_tiles.py";
  }
  state.subs = "off";
}

function centreSlice() {
  state.slice = Math.max(0, Math.min(NG - 1, Math.round(H().center_cell[0] - 0.5)));
  $("slice").value = state.slice;
}

// ---------- 1. 3D ----------------------------------------------------------
const c3d = $("c3d");
let drag = null;
c3d.addEventListener("pointerdown", e => {
  drag = { x: e.clientX, y: e.clientY }; c3d.setPointerCapture(e.pointerId);
});
c3d.addEventListener("pointermove", e => {
  if (!drag) return;
  state.yaw += (e.clientX - drag.x) * 0.008;
  state.pitch += (e.clientY - drag.y) * 0.008;
  state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch));
  drag = { x: e.clientX, y: e.clientY };
  draw3d();
});
c3d.addEventListener("pointerup", () => { drag = null; });

function offsets(h) {
  // Periodic offsets of the sampled sites from the host's Lagrangian centre,
  // in LR cells. Wrapping here is what keeps an edge-straddling host compact.
  const n = h.n_sampled, out = new Float32Array(n * 3), c = h.center_cell;
  for (let i = 0; i < n; i++)
    for (let a = 0; a < 3; a++)
      out[i * 3 + a] = wrap(h.xyz[i * 3 + a] + 0.5 - c[a], NG);
  return out;
}

function draw3d() {
  const g = c3d.getContext("2d"), W = c3d.width, Hh = c3d.height;
  g.fillStyle = "#08080b"; g.fillRect(0, 0, W, Hh);
  const h = H(), off = offsets(h);

  let ext = 4;
  for (let i = 0; i < off.length; i++) ext = Math.max(ext, Math.abs(off[i]));
  ext = Math.max(ext, S * 0.9);
  const sc = 0.42 * W / ext;
  const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const proj = (x, y, z) => {
    const X = x * cy - y * sy, Y = x * sy + y * cy;
    return [W / 2 + X * sc, Hh / 2 + (z * cp - Y * sp) * sc, Y * cp + z * sp];
  };

  // Tile wireframes, drawn behind the points.
  g.lineWidth = 2;
  h.tiles.forEach((t, i) => {
    const tx = Math.floor(t / (NT * NT)), ty = Math.floor(t / NT) % NT, tz = t % NT;
    const cc = [tx, ty, tz].map((q, a) =>
      wrap((q + 0.5) * S - h.center_cell[a], NG));
    const lo = cc.map(v => v - S / 2), hi = cc.map(v => v + S / 2);
    const P = [];
    for (const a of [0, 1]) for (const b of [0, 1]) for (const c of [0, 1])
      P.push(proj(a ? hi[0] : lo[0], b ? hi[1] : lo[1], c ? hi[2] : lo[2]));
    const E = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
    g.strokeStyle = t === state.tile ? "rgba(110,231,168,.95)" : "rgba(140,150,170,.30)";
    g.beginPath();
    for (const [a, b] of E) { g.moveTo(P[a][0], P[a][1]); g.lineTo(P[b][0], P[b][1]); }
    g.stroke();
  });

  // Points: the selected host, plus neighbours that fall inside the same view.
  const pts = [];
  const push = (hh, sel) => {
    const o = sel ? off : offsets(hh);
    const dc = [0, 1, 2].map(a => wrap(hh.center_cell[a] - h.center_cell[a], NG));
    for (let i = 0; i < hh.n_sampled; i++) {
      const x = o[i*3] + (sel ? 0 : dc[0]), y = o[i*3+1] + (sel ? 0 : dc[1]),
            z = o[i*3+2] + (sel ? 0 : dc[2]);
      if (!sel && Math.max(Math.abs(x), Math.abs(y), Math.abs(z)) > ext) continue;
      let v;
      if (state.chan === "log_host_mass") v = norm("log_host_mass", hh.log_mvir);
      else if (state.chan === "subhalo_budget") v = norm("subhalo_budget", hh.lam);
      else if (state.chan === "dq_over_rl")
        v = norm("dq_over_rl", Math.hypot(o[i*3], o[i*3+1], o[i*3+2]) * G.cell / hh.r_lag);
      else v = norm("host_fraction_per_tile", hh.tile_frac[0]);
      pts.push([...proj(x, y, z), v, sel]);
    }
  };
  push(h, true);
  if (state.others) for (const hh of D.hosts) if (hh !== h) push(hh, false);
  pts.sort((a, b) => a[2] - b[2]);
  for (const [px, py, , v, sel] of pts) {
    g.fillStyle = ramp(v);
    g.globalAlpha = sel ? 0.95 : 0.35;
    g.beginPath(); g.arc(px, py, sel ? 3.2 : 2.2, 0, 6.2832); g.fill();
  }
  g.globalAlpha = 1;
  legend3d(g, W, Hh);
  $("cap3d").textContent =
    `host #${h.host_id}: ${h.n_sampled} of ${h.n_particles} LR sites drawn, `
    + `R_L = ${fmt(h.r_lag, 3)} Mpc/h (${fmt(h.r_lag / G.cell, 2)} LR cells), `
    + `${h.tiles.length} tile(s) intersected.`;
}

function legend3d(g, W, Hh) {
  const v = D.vol[state.chan];
  g.font = "600 20px system-ui"; g.textBaseline = "middle";
  const x0 = 40, y0 = Hh - 46, w = 260, hgt = 14;
  for (let i = 0; i < w; i++) {
    g.fillStyle = ramp(i / (w - 1));
    g.fillRect(x0 + i, y0, 1.5, hgt);
  }
  g.fillStyle = "#c8ccd4";
  g.fillText(fmt(v.lo, 2), x0, y0 + 34);
  g.fillText(fmt(v.hi, 2), x0 + w - 40, y0 + 34);
  g.fillText(v.label, x0, y0 - 16);
}
function norm(chan, x) {
  const v = D.vol[chan];
  return v.hi > v.lo ? (x - v.lo) / (v.hi - v.lo) : 0.5;
}

// ---------- 2. slices ------------------------------------------------------
function drawSlice() {
  const g = $("cslice").getContext("2d"), W = 512, px = W / NG;
  const u8 = D.vol[state.chan].u8, h = H();
  g.fillStyle = "#08080b"; g.fillRect(0, 0, W, W);
  const x = state.slice;
  for (let b = 0; b < NG; b++) for (let c = 0; c < NG; c++) {
    const q = u8[(x * NG + b) * NG + c];
    if (!q) continue;
    const mine = ROW[(x * NG + b) * NG + c] === h.row;
    g.fillStyle = ramp((q - 1) / 254);
    g.globalAlpha = mine ? 1 : 0.55;
    g.fillRect(c * px, b * px, px, px);
  }
  g.globalAlpha = 1;
  // Tile borders, then the selected tile if this slice cuts it.
  g.strokeStyle = "rgba(150,158,175,.35)"; g.lineWidth = 1;
  for (let i = 0; i <= NT; i++) {
    g.beginPath(); g.moveTo(i * S * px, 0); g.lineTo(i * S * px, W); g.stroke();
    g.beginPath(); g.moveTo(0, i * S * px); g.lineTo(W, i * S * px); g.stroke();
  }
  const t = state.tile;
  const tx = Math.floor(t / (NT * NT)), ty = Math.floor(t / NT) % NT, tz = t % NT;
  if (Math.floor(x / S) === tx) {
    g.strokeStyle = "#6ee7a8"; g.lineWidth = 2.5;
    g.strokeRect(tz * S * px, ty * S * px, S * px, S * px);
  }
  $("slabel").textContent = `= ${state.slice}`
    + (Math.floor(state.slice / S) === tx ? " (cuts the tile)" : " (outside the tile)");

  // Subhalo centres whose Lagrangian mean site lies in this plane.
  const seen = {};
  for (const side of scSides()) {
    const pts = scInPlane(side, x);
    seen[side] = pts.length;
    for (const [cy, cz, np] of pts) {
      scDot(g, cz * px, cy * px, scRadius(np), SC_COL[side]);
    }
  }
  const scope = state.subscope === "host" ? "of this host" : "in the box";
  $("capslice").textContent =
    "LR lattice, one x slice. Grid lines are tile borders; the outlined square "
    + "is the selected tile."
    + (scSides().length
        ? `  Circles: subhalo centres ${scope} whose Lagrangian mean site is in `
          + `this plane — `
          + scSides().map(k => `${k.toUpperCase()} ${seen[k]}`).join(", ") + "."
        : "");
}

function drawHR() {
  const g = $("chr").getContext("2d"), W = 512;
  const u8 = D.vol[state.chan].u8, t = state.tile, f = G.upsample;
  const tx = Math.floor(t / (NT * NT)), ty = Math.floor(t / NT) % NT, tz = t % NT;
  // The HR view of the tile is the LR crop repeated f times per axis; drawing
  // one rectangle per LR site *is* that broadcast, so no HR array is built.
  const x = Math.min(S - 1, Math.max(0, state.slice - tx * S));
  const px = W / (S * f);
  g.fillStyle = "#08080b"; g.fillRect(0, 0, W, W);
  for (let b = 0; b < S; b++) for (let c = 0; c < S; c++) {
    const q = u8[(((tx * S + x) * NG) + ty * S + b) * NG + tz * S + c];
    g.fillStyle = q ? ramp((q - 1) / 254) : "#101015";
    g.fillRect(c * f * px, b * f * px, f * px, f * px);
  }
  g.strokeStyle = "rgba(150,158,175,.22)"; g.lineWidth = 1;
  for (let i = 0; i <= S * f; i++) {
    const p = i * px, big = i % f === 0;
    g.strokeStyle = big ? "rgba(150,158,175,.5)" : "rgba(150,158,175,.13)";
    g.beginPath(); g.moveTo(p, 0); g.lineTo(p, W); g.stroke();
    g.beginPath(); g.moveTo(0, p); g.lineTo(W, p); g.stroke();
  }
  // The same centres, cropped to this tile. The plane is the one actually
  // drawn (clamped into the tile), not the slider's, so the dots and the cells
  // underneath always come from the same slab.
  const inTile = {};
  for (const side of scSides()) {
    let n = 0;
    for (const [cy, cz, np] of scInPlane(side, tx * S + x)) {
      const dy = cy - ty * S, dz = cz - tz * S;
      if (dy < 0 || dy >= S || dz < 0 || dz >= S) continue;
      n++;
      scDot(g, dz * (W / S), dy * (W / S), scRadius(np) * 1.6, SC_COL[side]);
    }
    inTile[side] = n;
  }
  const want = state.slice - tx * S, clamped = want < 0 || want > S - 1;
  $("caphr").textContent =
    `tile ${t} at HR: ${S}^3 LR sites -> ${G.tile_hr}^3 HR cells. `
    + (clamped
        ? `The slider (x=${state.slice}) is outside this tile's x range `
          + `${tx * S}..${tx * S + S - 1}, so the nearest plane in the tile `
          + `(x=${tx * S + x}) is shown.`
        : `LR x index ${tx * S + x}, plane ${x} of ${S} within the tile.`)
    + (scSides().length
        ? "  Subhalo centres in this plane of the tile: "
          + scSides().map(k => `${k.toUpperCase()} ${inTile[k]}`).join(", ") + "."
        : "");
}

// ---------- 3. tile occupancy ---------------------------------------------
function drawTiles() {
  const cv = $("ctiles"), g = cv.getContext("2d");
  const W = cv.width, Hh = cv.height;
  g.fillStyle = "#08080b"; g.fillRect(0, 0, W, Hh);
  const h = H(), fr = {}, al = {};
  h.tiles.forEach((t, i) => { fr[t] = h.tile_frac[i]; al[t] = h.tile_alloc[i]; });
  const maxf = Math.max(...h.tile_frac);
  const cols = Math.ceil(Math.sqrt(NT)), rows = Math.ceil(NT / cols);
  const pw = W / cols, ph = Hh / rows, cell = Math.min(pw, ph) * 0.82 / NT;
  g.font = "600 11px system-ui"; g.textAlign = "center"; g.textBaseline = "middle";
  for (let tx = 0; tx < NT; tx++) {
    const ox = (tx % cols) * pw + (pw - cell * NT) / 2;
    const oy = Math.floor(tx / cols) * ph + (ph - cell * NT) / 2 + 8;
    g.fillStyle = "#8b909b"; g.font = "600 11px system-ui";
    g.fillText(`tile-x ${tx}`, ox + cell * NT / 2, oy - 10);
    for (let ty = 0; ty < NT; ty++) for (let tz = 0; tz < NT; tz++) {
      const t = (tx * NT + ty) * NT + tz;
      const x = ox + tz * cell, y = oy + ty * cell;
      const f = fr[t] || 0;
      g.fillStyle = f > 0 ? ramp(0.15 + 0.85 * f / maxf) : "#131319";
      g.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      if (t === state.tile) {
        g.strokeStyle = "#6ee7a8"; g.lineWidth = 2;
        g.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
      }
      if (f > 0.02) {
        const i = h.tiles.indexOf(t);
        g.fillStyle = f / maxf > 0.6 ? "#1a1206" : "#e8e8ec";
        g.font = "600 10px system-ui";
        g.fillText((100 * f).toFixed(0) + "%", x + cell / 2, y + cell / 2 - 5);
        g.font = "9px system-ui";
        g.fillText(i >= 0 ? String(h.tile_count[i]) : "",
                   x + cell / 2, y + cell / 2 + 6);
      }
    }
  }
  g.textAlign = "left";
  g.fillStyle = "#9aa0aa"; g.font = "12px system-ui";
  g.fillText(`host #${h.host_id} touches ${h.tiles.length} of ${G.n_tiles} tiles`
             + ` (top = % of the host, bottom = LR site count)`, 12, Hh - 10);
}

// ---------- 4. bars --------------------------------------------------------
function drawBars() {
  const cv = $("cbar"), g = cv.getContext("2d");
  const W = cv.width, Hh = cv.height;
  g.fillStyle = "#08080b"; g.fillRect(0, 0, W, Hh);
  const h = H(), n = Math.min(h.tiles.length, 16);
  const L = 70, R = 24, T = 30, B = 56;
  const bw = (W - L - R) / n;
  const maxf = Math.max(...h.tile_frac.slice(0, n));
  const maxa = Math.max(1e-12, ...h.tile_alloc.slice(0, n));
  g.font = "12px system-ui"; g.textBaseline = "middle";
  // axes
  g.strokeStyle = "#2a2a30"; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = T + (Hh - T - B) * i / 4;
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.fillStyle = "#8b909b"; g.textAlign = "right";
    g.fillText((maxf * (1 - i / 4)).toFixed(2), L - 8, y);
    g.textAlign = "left";
    g.fillText((maxa * (1 - i / 4)).toFixed(2), W - R + 6, y);
  }
  g.textAlign = "center";
  for (let i = 0; i < n; i++) {
    const t = h.tiles[i], x = L + i * bw;
    const hf = (Hh - T - B) * h.tile_frac[i] / maxf;
    const ha = (Hh - T - B) * h.tile_alloc[i] / maxa;
    g.fillStyle = t === state.tile ? "#6ee7a8" : "#7aa2f7";
    g.fillRect(x + bw * 0.12, Hh - B - hf, bw * 0.34, hf);
    g.fillStyle = t === state.tile ? "#f0c674" : "#c98cf1";
    g.fillRect(x + bw * 0.52, Hh - B - ha, bw * 0.34, ha);
    g.fillStyle = "#9aa0aa"; g.font = "11px system-ui";
    g.save(); g.translate(x + bw / 2, Hh - B + 16);
    g.fillText(String(t), 0, 0); g.restore();
  }
  g.fillStyle = "#8b909b"; g.font = "12px system-ui"; g.textAlign = "left";
  g.fillText("tile id", L, Hh - 14);
  g.fillStyle = "#7aa2f7"; g.fillRect(W / 2 - 130, Hh - 22, 10, 10);
  g.fillStyle = "#9aa0aa"; g.fillText("fraction of host", W / 2 - 114, Hh - 16);
  g.fillStyle = "#c98cf1"; g.fillRect(W / 2 + 10, Hh - 22, 10, 10);
  g.fillStyle = "#9aa0aa"; g.fillText("subhalo budget", W / 2 + 26, Hh - 16);
  g.textAlign = "center"; g.fillStyle = "#cfd3da"; g.font = "600 13px system-ui";
  g.fillText(`top ${n} of ${h.tiles.length} tiles  |  host has `
             + `${h.n_particles} LR sites, N_h = ${h.n_sub}`, W / 2, 14);
}

// ---------- 7. subhalos of this host per tile, HR vs SR2 -------------------
function drawSubs() {
  const cv = $("csub"), g = cv.getContext("2d");
  const W = cv.width, Hh = cv.height;
  g.fillStyle = "#08080b"; g.fillRect(0, 0, W, Hh);
  const h = H(), e = stOf();
  if (!ST) {
    g.fillStyle = "#8b909b"; g.font = "18px system-ui"; g.textAlign = "center";
    g.fillText("no per-host subhalo file; run collect_host_subhalo_tiles.py",
               W / 2, Hh / 2);
    g.textAlign = "left"; $("capsub").textContent = ""; rows("tsub", []);
    return;
  }
  if (!e || (!e.hr && !e.sr2)) {
    g.fillStyle = "#8b909b"; g.font = "18px system-ui"; g.textAlign = "center";
    g.fillText("this host is not in the collected file "
               + "(it was run with a host cap)", W / 2, Hh / 2);
    g.textAlign = "left"; $("capsub").textContent = ""; rows("tsub", []);
    return;
  }
  // Tile order follows the host's own tile list, so the x axis matches panel 6.
  const n = Math.min(h.tiles.length, 16);
  const hrv = [], srv = [];
  for (let i = 0; i < n; i++) {
    hrv.push(subOfTile("hr", h.tiles[i]) || 0);
    srv.push(subOfTile("sr2", h.tiles[i]) || 0);
  }
  const top = Math.max(1e-9, ...hrv, ...srv);
  const L = 62, R = 20, T = 34, B = 58, bw = (W - L - R) / n;
  g.font = "12px system-ui"; g.textBaseline = "middle";
  g.strokeStyle = "#2a2a30"; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = T + (Hh - T - B) * i / 4;
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke();
    g.fillStyle = "#8b909b"; g.textAlign = "right";
    g.fillText((top * (1 - i / 4)).toFixed(1), L - 8, y);
  }
  g.textAlign = "center";
  for (let i = 0; i < n; i++) {
    const t = h.tiles[i], x = L + i * bw;
    const hh = (Hh - T - B) * hrv[i] / top, hs = (Hh - T - B) * srv[i] / top;
    g.fillStyle = t === state.tile ? "#9ecbff" : "#7aa2f7";
    g.fillRect(x + bw * 0.12, Hh - B - hh, bw * 0.34, hh);
    g.fillStyle = t === state.tile ? "#ffb4b4" : "#f77a7a";
    g.fillRect(x + bw * 0.52, Hh - B - hs, bw * 0.34, hs);
    g.fillStyle = t === state.tile ? "#6ee7a8" : "#9aa0aa";
    g.font = t === state.tile ? "600 11px system-ui" : "11px system-ui";
    g.fillText(String(t), x + bw / 2, Hh - B + 16);
    g.fillStyle = "#8b909b"; g.font = "10px system-ui";
    g.fillText(hrv[i] > 0 ? (srv[i] / hrv[i]).toFixed(2) : "-",
               x + bw / 2, Hh - B + 32);
  }
  g.fillStyle = "#8b909b"; g.font = "12px system-ui"; g.textAlign = "left";
  g.fillText("tile id / SR2:HR", 8, Hh - B + 24);
  g.fillStyle = "#7aa2f7"; g.fillRect(W / 2 - 130, Hh - 22, 10, 10);
  g.fillStyle = "#9aa0aa"; g.fillText("HR subhalos", W / 2 - 114, Hh - 16);
  g.fillStyle = "#f77a7a"; g.fillRect(W / 2 + 20, Hh - 22, 10, 10);
  g.fillStyle = "#9aa0aa"; g.fillText("SR2 subhalos", W / 2 + 36, Hh - 16);
  const tot = s => (e[s] ? e[s].sub_total : null);
  g.textAlign = "center"; g.fillStyle = "#cfd3da"; g.font = "600 13px system-ui";
  g.fillText(`top ${n} of ${h.tiles.length} tiles  |  the matched host has `
             + `${tot("hr") === null ? "?" : fmt(tot("hr"), 1)} subhalos in HR `
             + `vs ${tot("sr2") === null ? "?" : fmt(tot("sr2"), 1)} in SR2`,
             W / 2, 15);

  const nh = subOfTile("hr", state.tile), ns = subOfTile("sr2", state.tile);
  $("capsub").textContent =
    `tile ${state.tile} contributes ${nh === null ? "?" : fmt(nh, 2)} of this `
    + `host's HR subhalos and ${ns === null ? "?" : fmt(ns, 2)} of its SR2 ones`
    + (nh > 0 && ns !== null
        ? ` (${(100 * ns / nh).toFixed(0)}% of HR).` : ".")
    + " Fractional: a subhalo split across tiles counts in each.";

  const side = (s, k, d) => (e[s] && e[s].match ? e[s].match[k] : d);
  const pair = dg => `${e.hr && e.hr.match ? dg(e.hr) : "-"}  /  `
                   + `${e.sr2 && e.sr2.match ? dg(e.sr2) : "-"}`;
  const shareHR = side("hr", "share_of_match", null);
  const shareSR = side("sr2", "share_of_match", null);
  const shaky = (shareHR !== null && shareHR < 0.5)
             || (shareSR !== null && shareSR < 0.5);
  rows("tsub", [
    ["selected tile", state.tile],
    ["subhalos this tile contributes (HR / SR2)",
     `${nh === null ? "?" : fmt(nh, 2)} / ${ns === null ? "?" : fmt(ns, 2)}`,
     nh > 0 && ns !== null && ns < 0.5 * nh ? "bad" : ""],
    ["matched host's total (HR / SR2)",
     `${tot("hr") === null ? "?" : fmt(tot("hr"), 1)} / `
     + `${tot("sr2") === null ? "?" : fmt(tot("sr2"), 1)}`,
     tot("hr") > 0 && tot("sr2") !== null && tot("sr2") < 0.5 * tot("hr")
       ? "bad" : ""],
    ["within this LR footprint only (HR / SR2)",
     pair(s => fmt(s.sub_total_footprint, 1))],
    ["matched halo id (HR / SR2)",
     `${side("hr", "halo_id", "-")} / ${side("sr2", "halo_id", "-")}`],
    ["matched log10 Mvir (HR / SR2)", pair(s => fmt(s.match.log_mvir, 2))],
    ["matched catalog subhalos (HR / SR2)", pair(s => s.match.n_sub_catalog)],
    ["LR footprint matched (HR / SR2)",
     pair(s => `${(100 * s.match.match_frac).toFixed(0)}%`)],
    ["LR host is this share of the match",
     pair(s => `${(100 * s.match.share_of_match).toFixed(0)}%`),
     shaky ? "bad" : "ok"],
    ["LR hosts sharing the match",
     pair(s => s.match.n_lr_hosts_sharing), shaky ? "bad" : "ok"],
    ["matched object is itself a subhalo",
     pair(s => (s.match.is_sub_of_another_host ? "yes &cross;" : "no")),
     side("hr", "is_sub_of_another_host", false)
     || side("sr2", "is_sub_of_another_host", false) ? "bad" : "ok"],
    ["LR N_h (budget, for reference)", h.n_sub],
    ["box totals over all collected hosts (HR / SR2)",
     `${fmt(ST.summary.hr_sub_total, 0)} / ${fmt(ST.summary.sr2_sub_total, 0)}`
     + (ST.summary.ratio === null ? "" : `  (${fmt(ST.summary.ratio, 3)})`),
     ST.summary.ratio !== null && ST.summary.ratio < 0.9 ? "bad" : "ok"],
  ]);
}

// ---------- 5. tables ------------------------------------------------------
function rows(t, pairs) {
  $(t).innerHTML = pairs.map(([k, v, cls]) =>
    `<tr><th>${esc(k)}</th><td class="num ${cls || ""}">${v}</td></tr>`).join("");
}
function drawTables() {
  const h = H(), r = D.report;
  const fracOK = Math.abs(h.frac_sum - 1) < 1e-6;
  const budOK = Math.abs(h.alloc_sum - h.n_sub) < 1e-3;
  rows("thost", [
    ["Rockstar host id (metadata)", h.host_id],
    ["Mvir [Msun/h]", h.mvir.toExponential(3)],
    ["log10 Mvir", fmt(h.log_mvir, 3)],
    ["Rvir [kpc/h] (Eulerian)", fmt(h.rvir_kpc_h, 1)],
    ["LR particles (host + subs)", h.n_particles],
    ["catalog num_p (host row)", h.num_p_catalog],
    ["Lagrangian radius R_L [Mpc/h]", fmt(h.r_lag, 4)],
    ["R_L in LR cells", fmt(h.r_lag / G.cell, 2)],
    ["rms |dq| [Mpc/h]", fmt(h.rms_lag, 4)],
    ["Lagrangian centre [Mpc/h]", h.center_lag.map(v => fmt(v, 2)).join(", ")],
    ["tiles intersected", `${h.tiles.length} / ${G.n_tiles}`],
    ["N_h (subhalo budget)", h.n_sub],
    ["lambda_i = N_h / N_part", fmt(h.lam, 5)],
    ["sum_t f[h,t]", `${fmt(h.frac_sum, 9)} ${fracOK ? "&check;" : "&cross;"}`,
     fracOK ? "ok" : "bad"],
    ["sum_i lambda_i", `${fmt(h.alloc_sum, 6)} vs N_h = ${h.n_sub} `
      + `${budOK ? "&check;" : "&cross;"}`, budOK ? "ok" : "bad"],
  ]);
  rows("tbox", [
    ["box", esc(D.box)],
    ["hosts on the lattice", r.n_hosts],
    ["LR sites", r.n_lr_sites],
    ["sites with a host", `${r.n_sites_with_host} `
      + `(${(100 * r.frac_sites_with_host).toFixed(1)}%)`],
    ["host mass range [log10]", r.mass_range_log10.map(v => fmt(v, 2)).join(" .. ")],
    ["median tiles per host", r.median_tiles_per_host],
    ["max tiles per host", r.max_tiles_per_host],
    ["hosts spanning >1 tile", r.n_hosts_spanning_tiles],
    ["N_h source", esc(D.n_sub_source)],
    ["max |sum_t f - 1| (all hosts)", r.max_abs_tile_frac_error.toExponential(2),
     r.max_abs_tile_frac_error < 1e-6 ? "ok" : "bad"],
    ["max |sum lambda - N_h| (all hosts)", r.max_abs_budget_error.toExponential(2),
     r.max_abs_budget_error < 1e-3 ? "ok" : "bad"],
    ["normalisation ok", r.ok ? "yes &check;" : "no &cross;", r.ok ? "ok" : "bad"],
    ["channels", esc(D.channels.join(", "))],
  ]);
}

// ---------- 6. Eulerian view of the selected tile -------------------------
const EU = D.eulerian;
if (EU) { EU.xyz = unb64(EU.pos); EU.disp.u8 = unb64(EU.disp.data); }
const ceul = $("ceul");
let edrag = null;
state.eyaw = 0.7; state.epitch = 0.3; state.eulcol = "log_host_mass";
$("eulcol").onchange = e => { state.eulcol = e.target.value; drawEul(); };
ceul.addEventListener("pointerdown", e => {
  edrag = { x: e.clientX, y: e.clientY }; ceul.setPointerCapture(e.pointerId);
});
ceul.addEventListener("pointermove", e => {
  if (!edrag) return;
  state.eyaw += (e.clientX - edrag.x) * 0.008;
  state.epitch = Math.max(-1.5, Math.min(1.5,
      state.epitch + (e.clientY - edrag.y) * 0.008));
  edrag = { x: e.clientX, y: e.clientY };
  drawEul();
});
ceul.addEventListener("pointerup", () => { edrag = null; });

function tileSiteIds(t) {
  const tx = Math.floor(t / (NT * NT)), ty = Math.floor(t / NT) % NT, tz = t % NT;
  const out = new Int32Array(S * S * S);
  let n = 0;
  for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) for (let c = 0; c < S; c++)
    out[n++] = ((tx * S + a) * NG + ty * S + b) * NG + tz * S + c;
  return out;
}

function drawEul() {
  const g = ceul.getContext("2d"), W = ceul.width, Hh = ceul.height;
  g.fillStyle = "#08080b"; g.fillRect(0, 0, W, Hh);
  if (!EU) {
    g.fillStyle = "#8b909b"; g.font = "20px system-ui"; g.textAlign = "center";
    g.fillText("no LR field was given to the renderer, so there is no "
               + "displacement to draw", W / 2, Hh / 2);
    g.textAlign = "left";
    $("capeul").textContent = "re-render with --lr-field to enable this panel.";
    return;
  }
  const BOX = G.boxsize, t = state.tile, ids = tileSiteIds(t);
  // Periodic centre of the displaced cloud: the tile can land across the seam.
  let sx = 0, sy = 0, cx = 0, cy = 0, cz = 0, sz = 0;
  for (const i of ids) {
    const th = [0, 1, 2].map(a => 2 * Math.PI * (EU.xyz[i * 3 + a] / 255) );
    sx += Math.sin(th[0]); cx += Math.cos(th[0]);
    sy += Math.sin(th[1]); cy += Math.cos(th[1]);
    sz += Math.sin(th[2]); cz += Math.cos(th[2]);
  }
  const ctr = [Math.atan2(sx, cx), Math.atan2(sy, cy), Math.atan2(sz, cz)]
      .map(a => ((a / (2 * Math.PI)) % 1 + 1) % 1 * BOX);
  const rel = i => [0, 1, 2].map(a =>
      wrap(EU.xyz[i * 3 + a] / 255 * BOX - ctr[a], BOX));

  // Undisplaced tile cube, in the same frame, for the bulk-shift reference.
  const tx = Math.floor(t / (NT * NT)), ty = Math.floor(t / NT) % NT, tz = t % NT;
  const cellMpc = G.cell, sideMpc = S * cellMpc;
  const qc = [tx, ty, tz].map((k, a) =>
      wrap((k + 0.5) * sideMpc - ctr[a], BOX));

  let ext = sideMpc * 0.7;
  const P = [];
  for (const i of ids) { const r = rel(i); P.push(r);
    ext = Math.max(ext, Math.abs(r[0]), Math.abs(r[1]), Math.abs(r[2])); }
  for (const a of [0, 1]) for (const b of [0, 1]) for (const c of [0, 1])
    ext = Math.max(ext,
        Math.abs(qc[0] + (a ? .5 : -.5) * sideMpc),
        Math.abs(qc[1] + (b ? .5 : -.5) * sideMpc),
        Math.abs(qc[2] + (c ? .5 : -.5) * sideMpc));
  const sc = 0.42 * Math.min(W, Hh) / ext;
  const cy2 = Math.cos(state.eyaw), sy2 = Math.sin(state.eyaw);
  const cp = Math.cos(state.epitch), sp = Math.sin(state.epitch);
  const proj = (x, y, z) => {
    const X = x * cy2 - y * sy2, Y = x * sy2 + y * cy2;
    return [W / 2 + X * sc, Hh / 2 + (z * cp - Y * sp) * sc, Y * cp + z * sp];
  };

  const V = [];
  for (const a of [0, 1]) for (const b of [0, 1]) for (const c of [0, 1])
    V.push(proj(qc[0] + (a ? .5 : -.5) * sideMpc,
                qc[1] + (b ? .5 : -.5) * sideMpc,
                qc[2] + (c ? .5 : -.5) * sideMpc));
  const E = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
  g.strokeStyle = "rgba(110,231,168,.55)"; g.lineWidth = 2; g.beginPath();
  for (const [a, b] of E) { g.moveTo(V[a][0], V[a][1]); g.lineTo(V[b][0], V[b][1]); }
  g.stroke();

  const massU8 = D.vol.log_host_mass.u8;
  const pts = [];
  let nBound = 0, sumPsi = 0;
  for (let k = 0; k < ids.length; k++) {
    const i = ids[k], r = P[k];
    const q = massU8[i];
    if (q) nBound++;
    sumPsi += EU.disp.u8[i] / 255 * EU.disp.hi;
    let col;
    if (state.eulcol === "disp") col = ramp(EU.disp.u8[i] / 255);
    else col = q ? ramp((q - 1) / 254) : null;      // null = joins no halo
    pts.push([...proj(r[0], r[1], r[2]), col, q]);
  }
  pts.sort((a, b) => a[2] - b[2]);
  for (const [px, py, , col, q] of pts) {
    g.fillStyle = col || "#4a4a55";
    g.globalAlpha = col ? 0.95 : 0.5;
    g.beginPath(); g.arc(px, py, col ? 3.4 : 2.4, 0, 6.2832); g.fill();
  }
  g.globalAlpha = 1;

  const bulk = Math.hypot(qc[0], qc[1], qc[2]);
  $("capeul").textContent =
    `tile ${t}: ${ids.length} LR particles, ${nBound} of them (`
    + `${(100 * nBound / ids.length).toFixed(0)}%) end up in a halo. `
    + `Cloud centre sits ${bulk.toFixed(2)} Mpc/h from where the tile started.`;

  const v = state.eulcol === "disp" ? EU.disp : D.vol.log_host_mass;
  g.font = "600 20px system-ui"; g.textBaseline = "middle";
  const x0 = 40, y0 = Hh - 46, w = 260;
  for (let i = 0; i < w; i++) { g.fillStyle = ramp(i / (w - 1));
    g.fillRect(x0 + i, y0, 1.5, 14); }
  g.fillStyle = "#c8ccd4";
  g.fillText(fmt(v.lo, 2), x0, y0 + 34);
  g.fillText(fmt(v.hi, 2), x0 + w - 40, y0 + 34);
  g.fillText(v.label, x0, y0 - 16);
  if (state.eulcol !== "disp") {
    g.fillStyle = "#4a4a55"; g.beginPath();
    g.arc(x0 + w + 70, y0 + 7, 6, 0, 6.2832); g.fill();
    g.fillStyle = "#8b909b"; g.fillText("no halo", x0 + w + 86, y0 + 7);
  }

  rows("teul", [
    ["tile", t],
    ["LR particles", ids.length],
    ["end in a halo", `${nBound} (${(100 * nBound / ids.length).toFixed(0)}%)`],
    ["bulk shift of the tile", `${bulk.toFixed(2)} Mpc/h`],
    ["tile side (Lagrangian)", `${sideMpc.toFixed(2)} Mpc/h`],
    ["bulk / tile side", (bulk / sideMpc).toFixed(2)],
    ["mean |displacement|", `${(sumPsi / ids.length).toFixed(2)} Mpc/h`],
    ["box |psi| median", `${EU.stats.median.toFixed(2)} Mpc/h`],
    ["box |psi| p90 / max",
     `${EU.stats.p90.toFixed(2)} / ${EU.stats.max.toFixed(2)} Mpc/h`],
  ]);
}

// ---------- 5. SR2 subhalo deficit ---------------------------------------
const AB = D.abundance;
// Diverging ramp: the deficit has a meaningful zero (parity with HR), which a
// sequential ramp would hide. Red = short of HR, blue = over.
function divRamp(t) {                     // t in [-1, 1]
  const c = Math.max(-1, Math.min(1, t)), a = Math.abs(c);
  if (c < 0) return `rgb(${255 - 40 * (1 - a) | 0},${235 - 200 * a | 0},${235 - 205 * a | 0})`;
  return `rgb(${235 - 195 * a | 0},${240 - 90 * a | 0},${255 - 25 * a | 0})`;
}

function drawDeficit() {
  const cv = $("cdef"), g = cv.getContext("2d");
  const W = cv.width, Hh = cv.height;
  g.fillStyle = "#08080b"; g.fillRect(0, 0, W, Hh);
  if (!AB) {
    g.fillStyle = "#8b909b"; g.font = "20px system-ui"; g.textAlign = "center";
    g.fillText("no tile-abundance file; run collect_tile_abundance.py",
               W / 2, Hh / 2);
    g.textAlign = "left";
    $("capdef").textContent = "";
    rows("tdef", []);
    return;
  }
  const rel = AB.rel_deficit, T = AB.totals;
  const cols = Math.ceil(Math.sqrt(NT)), rowsN = Math.ceil(NT / cols);
  const pw = W / cols, ph = Hh / rowsN, cell = Math.min(pw, ph) * 0.82 / NT;
  g.textAlign = "center"; g.textBaseline = "middle";
  for (let tx = 0; tx < NT; tx++) {
    const ox = (tx % cols) * pw + (pw - cell * NT) / 2;
    const oy = Math.floor(tx / cols) * ph + (ph - cell * NT) / 2 + 8;
    g.fillStyle = "#8b909b"; g.font = "600 11px system-ui";
    g.fillText(`tile-x ${tx}`, ox + cell * NT / 2, oy - 10);
    for (let ty = 0; ty < NT; ty++) for (let tz = 0; tz < NT; tz++) {
      const t = (tx * NT + ty) * NT + tz;
      const x = ox + tz * cell, y = oy + ty * cell;
      const v = rel[t];
      g.fillStyle = v === null ? "#131319" : divRamp(v);
      g.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      if (t === state.tile) {
        g.strokeStyle = "#6ee7a8"; g.lineWidth = 2;
        g.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
      }
    }
  }
  // Legend
  g.textAlign = "left";
  const x0 = 20, y0 = Hh - 26, w = 240;
  for (let i = 0; i < w; i++) {
    g.fillStyle = divRamp(-1 + 2 * i / (w - 1));
    g.fillRect(x0 + i, y0, 1.5, 12);
  }
  g.fillStyle = "#c8ccd4"; g.font = "12px system-ui";
  g.fillText("-100%", x0, y0 + 22);
  g.fillText("parity", x0 + w / 2 - 18, y0 + 22);
  g.fillText("+100%", x0 + w - 34, y0 + 22);
  g.fillText("(N_SR2 - N_HR) / N_HR", x0, y0 - 10);

  const v = rel[state.tile];
  $("capdef").textContent = v === null
    ? `tile ${state.tile}: HR has no subhalo here, so no ratio is defined.`
    : `tile ${state.tile}: SR2 ${AB.n_sub_sr2[state.tile].toFixed(1)} vs HR `
      + `${AB.n_sub_hr[state.tile].toFixed(1)} subhalos `
      + `(${(100 * v).toFixed(1)}% ${v < 0 ? "short" : "over"}).`;

  rows("tdef", [
    ["HR subhalos (box)", T.hr_subhalos.toFixed(0)],
    ["SR2 subhalos (box)", T.sr2_subhalos.toFixed(0)],
    ["SR2 / HR", T.ratio.toFixed(3), T.ratio < 0.9 ? "bad" : "ok"],
    ["tiles short of HR", `${T.tiles_short} / ${T.tiles_scored}`,
     T.tiles_short > T.tiles_scored / 2 ? "bad" : "ok"],
    ["median rel. deficit", (100 * T.median_rel_deficit).toFixed(1) + "%", "bad"],
    ["p10 / p90", `${(100 * T.p10_rel_deficit).toFixed(1)}% / `
      + `${(100 * T.p90_rel_deficit).toFixed(1)}%`],
    ["worst tile", (100 * T.worst_rel_deficit).toFixed(1) + "%", "bad"],
    ["bound-particle occupancy",
     `SR2 ${T.sr2_occupancy.toFixed(4)} vs HR ${T.hr_occupancy.toFixed(4)}`],
    ["min subhalo particles", AB.min_sub_particles],
  ]);
}

function draw() { draw3d(); drawSlice(); drawHR(); drawTiles(); drawBars();
                  drawTables(); drawEul(); drawDeficit(); drawSubs(); }
fillHosts(); fillTiles(); centreSlice(); draw();
</script>
"""


def render(feat: LagrangianHostFeatures, out: Path, *, n_hosts: int,
           n_sample: int, seed: int,
           field_path: Optional[Path] = None,
           abundance_path: Optional[Path] = None,
           subtiles_path: Optional[Path] = None,
           subcentres_path: Optional[Path] = None) -> Path:
    payload = build_payload(feat, n_hosts=n_hosts, n_sample=n_sample, seed=seed)
    # Panel 6 needs the displacement; without the LR field the page still works
    # and simply says so rather than drawing an empty box.
    payload["eulerian"] = (eulerian_payload(feat, field_path)
                           if field_path is not None else None)
    # Per-tile SR2-vs-HR abundance, if it has been collected. Optional so the
    # page still renders on a box whose SR2 catalog does not exist yet.
    payload["abundance"] = (json.loads(Path(abundance_path).read_text())
                            if abundance_path is not None else None)
    # Per-host, per-tile HR/SR2 subhalo counts (panel 7). Also optional: it
    # needs the owner arrays, which a box may not have streamed yet.
    payload["subtiles"] = (json.loads(Path(subtiles_path).read_text())
                           if subtiles_path is not None else None)
    # Subhalo Lagrangian centres for panel 1's overlay (same collector).
    payload["subcentres"] = (subcentres_payload(Path(subcentres_path))
                             if subcentres_path is not None else None)
    html = _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    html = html.replace("__BOX__", feat.box or out.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8")
    ap.add_argument("--n-hosts", type=int, default=0,
                    help="most massive hosts the page can select; 0 (the "
                         "default) means every host, which costs ~240 kB at "
                         "64^3 because the per-host sample cap only binds for "
                         "the largest few")
    ap.add_argument("--n-sample", type=int, default=2500,
                    help="cap on the LR sites drawn per host in 3D "
                         "(default 2500); the full count is still reported")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="",
                    help="where to write the html (default: next to the npz)")
    ap.add_argument("--tile-abundance", default="",
                    help="per-tile SR2-vs-HR abundance JSON (default: the "
                         "<box>_tile_abundance.json next to the features; "
                         "'none' to skip panel 5)")
    ap.add_argument("--host-subhalo-tiles", default="",
                    help="per-host per-tile HR/SR2 subhalo-count JSON (default: "
                         "the <box>_host_subhalo_tiles.json next to the "
                         "features; 'none' to skip panel 7)")
    ap.add_argument("--subhalo-centres", default="",
                    help="subhalo Lagrangian-centre npz for panel 1's overlay "
                         "(default: the <box>_subhalo_centres.npz next to the "
                         "features; 'none' to skip the overlay)")
    ap.add_argument("--lr-field", default="",
                    help="LR field npy for the displacement panel (default: the "
                         "reward config's lr/<box>.npy; pass 'none' to skip)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_reward_config(args)

    for box in [b.strip() for b in args.boxes.split(",") if b.strip()]:
        banner(f"render {box}")
        src = paths.subdir("lagrangian_host", box) / f"{box}_lagrangian_host.npz"
        if not src.is_file():
            raise SystemExit(
                f"no cached features at {src}; run "
                f"scripts/features/build_lagrangian_host.py --boxes {box} first")
        feat = LagrangianHostFeatures.from_npz(src)
        out_dir = Path(args.out_dir) if args.out_dir else src.parent

        if args.lr_field.lower() == "none":
            fp = None
        elif args.lr_field:
            fp = Path(args.lr_field)
        else:
            fp = lr_path(cfg, box)
        if fp is not None and not fp.is_file():
            print(f"    no LR field at {fp}; panel 6 (displacement) is disabled")
            fp = None
        if fp is not None:
            print(f"    displacement from {fp}")

        if args.tile_abundance.lower() == "none":
            ap_path = None
        elif args.tile_abundance:
            ap_path = Path(args.tile_abundance)
        else:
            ap_path = src.parent / f"{box}_tile_abundance.json"
        if ap_path is not None and not ap_path.is_file():
            print(f"    no tile abundance at {ap_path}; panel 5 is disabled")
            ap_path = None
        if ap_path is not None:
            print(f"    abundance from {ap_path}")

        if args.host_subhalo_tiles.lower() == "none":
            st_path = None
        elif args.host_subhalo_tiles:
            st_path = Path(args.host_subhalo_tiles)
        else:
            st_path = src.parent / f"{box}_host_subhalo_tiles.json"
        if st_path is not None and not st_path.is_file():
            print(f"    no per-host subhalo tiles at {st_path}; "
                  f"panel 7 is disabled")
            st_path = None
        if st_path is not None:
            print(f"    per-host subhalo tiles from {st_path}")

        if args.subhalo_centres.lower() == "none":
            sc_path = None
        elif args.subhalo_centres:
            sc_path = Path(args.subhalo_centres)
        else:
            sc_path = src.parent / f"{box}_subhalo_centres.npz"
        if sc_path is not None and not sc_path.is_file():
            print(f"    no subhalo centres at {sc_path}; panel 1's overlay "
                  f"is disabled")
            sc_path = None
        if sc_path is not None:
            print(f"    subhalo centres from {sc_path}")

        out = render(feat, out_dir / f"lagrangian_host_{box}.html",
                     n_hosts=args.n_hosts, n_sample=args.n_sample,
                     seed=args.seed, field_path=fp, abundance_path=ap_path,
                     subtiles_path=st_path, subcentres_path=sc_path)
        n_sel = (feat.table.n_hosts if args.n_hosts <= 0
                 else min(args.n_hosts, feat.table.n_hosts))
        print(f"    {feat.table.n_hosts} hosts, {n_sel} selectable")
        print(f"    wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
