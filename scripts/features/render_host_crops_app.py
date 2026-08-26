#!/usr/bin/env python
"""Render the host-frame crops as one self-contained interactive page.

Same shape as ``render_lagrangian_host_app.py``: every array is embedded, the
file opens from an ``scp`` with no server and no port on the login node, and it
is a pure redraw of ``<box>_host_crops.npz`` -- rerun it to change the colours
or the host cap without touching the collector.

The page answers one question in four panels:

1. **Lagrangian crop.** A slab through the host's crop, showing SR2's local
   density, HR's, or their difference, with **every HR subhalo drawn where its
   Lagrangian material sits**. This is the picture the question "how easy are
   they to learn" asks for: if the circles land on features that are already
   visible in the SR2 panel, a conditional model has something to condition on.
2. **Eulerian cloud.** The same crop as particles, HR beside SR2, which is what
   the crop looks like as a halo rather than as a field.
3. **Learnability.** The ROC of the SR2 scalar against the HR ceiling and the
   geometry-only baseline, plus the per-subhalo view: how much of each HR
   subhalo's material SR2 binds into anything at all.
4. **Across hosts.** Whether any of this depends on host mass, which is the
   axis the deficit lives on.

    python scripts/features/render_host_crops_app.py --boxes set8
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, paths  # noqa: E402

VOL_CHANNELS = ("sr2_rho", "hr_rho", "hr_sub")


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def build_payload(summary: dict, store, *, n_hosts: int) -> dict:
    """Everything the page draws, as plain JSON plus base64 typed arrays.

    Volumes arrive from the collector already quantised with their range in the
    first eight bytes (see ``collect_host_crops._quant``); they are passed
    through untouched so the page inverts exactly the same transform the
    collector applied.
    """
    hosts = list(summary["hosts"])[:n_hosts]
    out_hosts = []
    for h in hosts:
        key = f"h{int(h['halo_id'])}"
        rec = dict(h)
        rec["vol"] = {}
        for ch in VOL_CHANNELS:
            name = f"{key}__{ch}"
            if name in store:
                rec["vol"][ch] = _b64(store[name])
        for side in ("hr", "sr2"):
            name = f"{key}__scatter_{side}"
            if name in store:
                rec[f"scatter_{side}"] = _b64(store[name].astype("<i2"))
        out_hosts.append(rec)
    return {
        "box": summary["box"],
        "meta": {k: summary[k] for k in
                 ("grid", "crop_scale", "k_neighbours", "pad_sites",
                  "boxsize_mpc_h", "ng_hr", "particle_mass_msun_h",
                  "hr_field", "sr2_field", "n_hosts", "seconds")
                 if k in summary},
        "hosts": out_hosts,
    }


CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#26303d;--fg:#d7dee7;--dim:#8b98a8;
      --hr:#f2b544;--sr2:#4aa3ff;--acc:#7ee787;--warn:#ff7b72;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
h1{font-size:17px;margin:0 0 2px}
h2{font-size:13px;margin:0 0 8px;color:var(--acc);text-transform:uppercase;
   letter-spacing:.08em;font-weight:600}
a{color:var(--sr2)}
header{padding:14px 18px;border-bottom:1px solid var(--line);background:var(--panel)}
header p{margin:4px 0 0;color:var(--dim);max-width:98ch}
.wrap{display:grid;grid-template-columns:300px minmax(0,1fr);gap:14px;padding:14px 18px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;
       padding:12px;margin-bottom:14px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:3px 5px;text-align:right;border-bottom:1px solid var(--line);
      white-space:nowrap}
th{color:var(--dim);font-weight:600;cursor:pointer;user-select:none;position:sticky;
   top:0;background:var(--panel)}
th:first-child,td:first-child{text-align:left}
tbody tr{cursor:pointer}
tbody tr:hover{background:#1d242e}
tbody tr.sel{background:#243447;box-shadow:inset 2px 0 0 var(--acc)}
.scroll{max-height:70vh;overflow:auto}
.ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-bottom:10px;
     color:var(--dim)}
.ctl label{display:flex;gap:6px;align-items:center;color:var(--fg)}
input[type=range]{width:150px;accent-color:var(--acc)}
select,button{background:#0d1117;color:var(--fg);border:1px solid var(--line);
              border-radius:4px;padding:3px 7px;font:inherit}
button.on{border-color:var(--acc);color:var(--acc)}
canvas{display:block;background:#05080c;border:1px solid var(--line);border-radius:4px;
       image-rendering:pixelated;max-width:100%}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.cap{color:var(--dim);font-size:12px;margin:6px 0 0;max-width:100ch}
.kv{display:grid;grid-template-columns:auto auto;gap:1px 12px;font-size:12px}
.kv b{font-weight:600;color:var(--dim)}
.num{font-variant-numeric:tabular-nums}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;
        vertical-align:-1px;margin-right:4px}
.note{color:var(--dim);font-size:12px;border-left:2px solid var(--line);
      padding-left:10px;margin:8px 0 0}
.big{font-size:22px;font-variant-numeric:tabular-nums}
"""


APP_JS = r"""
"use strict";
const D = window.__CROPS__;
const H = D.hosts;

/* ---------- decoding ------------------------------------------------- */
function b64bytes(s){
  const bin = atob(s), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}
/* The collector prepends the float32 range to every quantised cube, so the
   page inverts exactly the transform that was applied. */
function decodeVol(s){
  const u = b64bytes(s);
  const dv = new DataView(u.buffer, u.byteOffset, 8);
  return {lo: dv.getFloat32(0, true), hi: dv.getFloat32(4, true), q: u.subarray(8)};
}
function decodeScatter(s){
  const u = b64bytes(s);
  return new Int16Array(u.buffer, u.byteOffset, u.byteLength >> 1);
}
const volCache = new Map(), scatCache = new Map();
function vol(h, ch){
  const k = h.halo_id + ":" + ch;
  if (!volCache.has(k)) volCache.set(k, h.vol[ch] ? decodeVol(h.vol[ch]) : null);
  return volCache.get(k);
}
function scat(h, side){
  const k = h.halo_id + ":" + side, f = "scatter_" + side;
  if (!scatCache.has(k)) scatCache.set(k, h[f] ? decodeScatter(h[f]) : null);
  return scatCache.get(k);
}

/* ---------- colour --------------------------------------------------- */
const VIRIDIS = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
                 [31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]];
const DIVERGE = [[41,98,255],[90,140,220],[150,175,200],[210,210,210],
                 [225,170,140],[230,120,80],[220,50,32]];
function ramp(t, table){
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const x = t * (table.length - 1), i = Math.min(Math.floor(x), table.length - 2);
  const f = x - i, a = table[i], b = table[i + 1];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
}

/* ---------- slab projection ------------------------------------------ */
/* Display axes are the two the view axis leaves over, in increasing order:
   the row axis first, the column axis second. Keeping that rule in one place
   is what makes the overlay and the image agree. */
function planeAxes(axis){ return [[1,2],[0,2],[0,1]][axis]; }

function slabMax(v, g, axis, k0, k1){
  const out = new Float32Array(g * g).fill(-Infinity);
  const [ra, ca] = planeAxes(axis);
  const idx = [0, 0, 0];
  for (let r = 0; r < g; r++){
    for (let c = 0; c < g; c++){
      let m = -Infinity;
      for (let k = k0; k < k1; k++){
        idx[ra] = r; idx[ca] = c; idx[axis] = k;
        const val = v.q[(idx[0] * g + idx[1]) * g + idx[2]];
        if (val > m) m = val;
      }
      out[r * g + c] = m;
    }
  }
  return out;
}

/* ---------- the crop panel ------------------------------------------- */
const S = {
  host: 0, field: "sr2_rho", axis: 2, slab: 0.5, thick: 0.15,
  showSubs: true, minP: 20, inSlab: true, proj: 2, sortKey: "log_mvir", sortDir: -1,
};

const cv = document.getElementById("crop");
const off = document.createElement("canvas");

function curHost(){ return H[S.host]; }

function drawCrop(){
  const h = curHost(), g = h.grid_out, W = cv.width;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, W);
  const half = Math.max(1, Math.round(S.thick * g / 2));
  const kc = Math.round(S.slab * (g - 1));
  const k0 = Math.max(0, kc - half), k1 = Math.min(g, kc + half + 1);

  let img, lo, hi, table = VIRIDIS;
  if (S.field === "diff"){
    const a = vol(h, "sr2_rho"), b = vol(h, "hr_rho");
    if (!a || !b) return;
    const sa = slabMax(a, g, S.axis, k0, k1), sb = slabMax(b, g, S.axis, k0, k1);
    img = new Float32Array(g * g);
    /* Both cubes are quantised on their own range; undo each before
       subtracting or the difference is between two different units. */
    for (let i = 0; i < img.length; i++){
      const va = a.lo + (a.hi - a.lo) * sa[i] / 255;
      const vb = b.lo + (b.hi - b.lo) * sb[i] / 255;
      img[i] = vb - va;
    }
    let m = 0;
    for (let i = 0; i < img.length; i++) m = Math.max(m, Math.abs(img[i]));
    lo = -m; hi = m; table = DIVERGE;
  } else {
    const v = vol(h, S.field);
    if (!v) return;
    const s = slabMax(v, g, S.axis, k0, k1);
    img = new Float32Array(g * g);
    for (let i = 0; i < img.length; i++) img[i] = v.lo + (v.hi - v.lo) * s[i] / 255;
    lo = v.lo; hi = v.hi;
    if (S.field === "hr_sub"){ lo = 0; hi = Math.max(hi, 1e-6); }
  }

  off.width = off.height = g;
  const octx = off.getContext("2d"), id = octx.createImageData(g, g);
  const span = (hi - lo) || 1;
  for (let i = 0; i < g * g; i++){
    const c = ramp((img[i] - lo) / span, table);
    id.data[4 * i] = c[0]; id.data[4 * i + 1] = c[1];
    id.data[4 * i + 2] = c[2]; id.data[4 * i + 3] = 255;
  }
  octx.putImageData(id, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, g, g, 0, 0, W, W);

  if (S.showSubs) overlaySubs(ctx, h, W, k0, k1, g);
  drawLegend(lo, hi, table);
  document.getElementById("slabinfo").textContent =
    `slab ${k0}–${k1 - 1} of ${g} along ${"xyz"[S.axis]} ` +
    `(${(h.crop_side_mpc_h * (k1 - k0) / g).toFixed(2)} Mpc/h thick)`;
}

/* HR subhalo centres, in the crop's own native-site coordinates. The volume
   is a block reduction of that same cube, so the two share an origin and a
   scale and the circles land where the material is. */
function overlaySubs(ctx, h, W, k0, k1, g){
  const [ra, ca] = planeAxes(S.axis);
  /* The reduced cube spans `vol_extent_sites` native sites, which is `side`
     rounded up to a whole number of blocks. Scaling by `side` instead drifts
     every circle outward by extent/side -- the bug that put circles in the
     padded margin. */
  const side = h.crop_side_sites;
  const extent = h.vol_extent_sites || side;
  const px = W / extent;
  const d0 = k0 * extent / g, d1 = k1 * extent / g;
  let shown = 0;
  for (const s of h.hr_subhalos){
    if (s.num_p < S.minP) continue;
    const u = s.u, depth = u[S.axis], r = Math.max(s.rl_sites, 1.0);
    const near = depth + r >= d0 && depth - r <= d1;
    if (S.inSlab && !near) continue;
    shown++;
    const x = u[ca] * px, y = u[ra] * px, rp = Math.max(r * px, 2.5);
    const t = Math.min(1, Math.log10(Math.max(s.num_p, 1)) / 3.5);
    const c = ramp(t, [[255,255,255],[255,220,120],[255,140,60],[255,60,40]]);
    ctx.beginPath(); ctx.arc(x, y, rp, 0, 6.2832);
    ctx.strokeStyle = `rgba(${c[0]|0},${c[1]|0},${c[2]|0},${near ? 0.95 : 0.28})`;
    ctx.lineWidth = near ? 1.6 : 1.0;
    ctx.stroke();
  }
  document.getElementById("nsubshown").textContent = shown;
}

function drawLegend(lo, hi, table){
  const c = document.getElementById("cbar"), ctx = c.getContext("2d");
  const id = ctx.createImageData(c.width, 1);
  for (let i = 0; i < c.width; i++){
    const col = ramp(i / (c.width - 1), table);
    id.data[4*i] = col[0]; id.data[4*i+1] = col[1];
    id.data[4*i+2] = col[2]; id.data[4*i+3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  ctx.drawImage(c, 0, 0, c.width, 1, 0, 0, c.width, c.height);
  document.getElementById("cblo").textContent = lo.toFixed(2);
  document.getElementById("cbhi").textContent = hi.toFixed(2);
}

/* ---------- Eulerian panel ------------------------------------------- */
function drawCloud(){
  const h = curHost();
  const [ra, ca] = planeAxes(S.proj);
  const pts = {hr: scat(h, "hr"), sr2: scat(h, "sr2")};
  let m = 1e-6;
  for (const k of ["hr", "sr2"]){
    const p = pts[k]; if (!p) continue;
    for (let i = 0; i < p.length; i++) m = Math.max(m, Math.abs(p[i] / 64));
  }
  for (const k of ["hr", "sr2"]){
    const c = document.getElementById("cloud_" + k), ctx = c.getContext("2d");
    const W = c.width;
    ctx.fillStyle = "#05080c"; ctx.fillRect(0, 0, W, W);
    const p = pts[k];
    if (p){
      ctx.fillStyle = k === "hr" ? "rgba(242,181,68,0.30)" : "rgba(74,163,255,0.30)";
      for (let i = 0; i + 2 < p.length; i += 3){
        const x = (p[i + ca] / 64 / m * 0.5 + 0.5) * W;
        const y = (p[i + ra] / 64 / m * 0.5 + 0.5) * W;
        ctx.fillRect(x, y, 1.2, 1.2);
      }
    }
    /* HR subhalo centres go on both canvases: on HR they say where the truth
       is, on SR2 they say what is missing at that spot. */
    if (S.showSubs){
      for (const s of h.hr_subhalos){
        if (s.num_p < S.minP || !s.d_mpc_h) continue;
        const x = (s.d_mpc_h[ca] / m * 0.5 + 0.5) * W;
        const y = (s.d_mpc_h[ra] / m * 0.5 + 0.5) * W;
        ctx.beginPath(); ctx.arc(x, y, 3.0, 0, 6.2832);
        ctx.strokeStyle = "rgba(255,110,60,0.85)"; ctx.lineWidth = 1.1; ctx.stroke();
      }
    }
  }
  document.getElementById("cloudscale").textContent =
    `±${m.toFixed(2)} Mpc/h, projected along ${"xyz"[S.proj]}`;
}

/* ---------- learnability panel --------------------------------------- */
function svgLine(pts, colour, width){
  if (!pts.length) return "";
  const d = pts.map(p => `${(p[0] * 200).toFixed(1)},${(200 - p[1] * 200).toFixed(1)}`);
  return `<polyline fill="none" stroke="${colour}" stroke-width="${width}" points="${d.join(" ")}"/>`;
}
function drawLearn(){
  const h = curHost(), L = (h.learnability || {})["footprint"];
  const box = document.getElementById("roc");
  if (!L){ box.innerHTML = '<p class="cap">no subhalos in this host’s footprint.</p>'; 
           document.getElementById("aucbars").innerHTML = ""; return; }
  const z = (r) => (r.fpr || []).map((f, i) => [f, r.tpr[i]]);
  box.innerHTML =
    `<svg viewBox="0 0 200 200" width="230" height="230">
       <rect x="0" y="0" width="200" height="200" fill="#05080c" stroke="#26303d"/>
       <line x1="0" y1="200" x2="200" y2="0" stroke="#3d4a5a" stroke-dasharray="3 3"/>
       ${svgLine(z(L.roc_hr_rho || {}), "#f2b544", 1.6)}
       ${svgLine(z(L.roc_sr2_rho || {}), "#4aa3ff", 1.8)}
     </svg>`;
  const rows = [
    ["SR2 density, σ=0", L.auc_sr2_rho.sigma0, "#4aa3ff"],
    ["SR2 density, σ=1", L.auc_sr2_rho.sigma1, "#4aa3ff"],
    ["SR2 density, σ=2", L.auc_sr2_rho.sigma2, "#4aa3ff"],
    ["SR2 density, σ=4", L.auc_sr2_rho.sigma4, "#4aa3ff"],
    ["SR2 bound-or-not", L.auc_sr2_bound, "#6fb3ff"],
    ["distance from centre", L.auc_radius, "#8b98a8"],
    ["HR density, σ=2 (ceiling)", L.auc_hr_rho.sigma2, "#f2b544"],
  ];
  document.getElementById("aucbars").innerHTML = rows.map(([n, v, col]) => {
    const x = isFinite(v) ? v : 0.5;
    /* Bars start at 0.5, because 0.5 is the score of knowing nothing. */
    const w = Math.abs(x - 0.5) * 2 * 100, left = x >= 0.5 ? 50 : 50 - w;
    return `<div style="display:grid;grid-template-columns:190px 1fr 52px;
              gap:8px;align-items:center;margin:2px 0">
        <span style="color:var(--dim)">${n}</span>
        <span style="position:relative;height:11px;background:#0d1117;
              border:1px solid var(--line);border-radius:2px">
          <span style="position:absolute;left:50%;top:0;bottom:0;width:1px;
                background:#3d4a5a"></span>
          <span style="position:absolute;left:${left}%;width:${w}%;top:0;bottom:0;
                background:${col};opacity:.75"></span></span>
        <span class="num">${isFinite(v) ? x.toFixed(3) : "–"}</span></div>`;
  }).join("");
  document.getElementById("baserate").textContent =
    (100 * L.base_rate).toFixed(1) + "%";
  document.getElementById("nfoot").textContent = L.n_sites.toLocaleString();

  /* per-subhalo: how much of each HR subhalo SR2 binds into anything */
  const c = document.getElementById("persub"), ctx = c.getContext("2d");
  const W = c.width, Hh = c.height;
  ctx.fillStyle = "#05080c"; ctx.fillRect(0, 0, W, Hh);
  ctx.strokeStyle = "#26303d"; ctx.strokeRect(0.5, 0.5, W - 1, Hh - 1);
  for (const s of h.hr_subhalos){
    const x = (Math.log10(Math.max(s.num_p, 1)) / 4.0) * W;
    const y = Hh - s.sr2_bound_frac * Hh;
    ctx.fillStyle = "rgba(126,231,135,0.55)";
    ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 6.2832); ctx.fill();
  }
}

/* ---------- across hosts --------------------------------------------- */
function drawAcross(){
  const c = document.getElementById("across"), ctx = c.getContext("2d");
  const W = c.width, Hh = c.height;
  ctx.fillStyle = "#05080c"; ctx.fillRect(0, 0, W, Hh);
  ctx.strokeStyle = "#26303d"; ctx.strokeRect(0.5, 0.5, W - 1, Hh - 1);
  ctx.strokeStyle = "#3d4a5a"; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(0, Hh * 0.5); ctx.lineTo(W, Hh * 0.5); ctx.stroke();
  ctx.setLineDash([]);
  const xs = H.map(h => h.log_mvir);
  const lo = Math.min(...xs) - 0.15, hi = Math.max(...xs) + 0.15;
  H.forEach((h, i) => {
    const L = (h.learnability || {})["footprint"]; if (!L) return;
    const x = (h.log_mvir - lo) / (hi - lo) * W;
    const y = Hh - L.auc_sr2_rho.sigma2 * Hh;
    ctx.fillStyle = i === S.host ? "#7ee787" : "rgba(74,163,255,0.75)";
    ctx.beginPath(); ctx.arc(x, y, i === S.host ? 5 : 3.2, 0, 6.2832); ctx.fill();
    const y2 = Hh - L.auc_hr_rho.sigma2 * Hh;
    ctx.strokeStyle = "rgba(242,181,68,0.6)";
    ctx.beginPath(); ctx.arc(x, y2, 2.6, 0, 6.2832); ctx.stroke();
  });
  document.getElementById("acrossx").textContent =
    `log10 Mvir  ${lo.toFixed(2)} → ${hi.toFixed(2)}`;
}

/* ---------- host table ------------------------------------------------ */
const COLS = [
  ["halo_id", "id", h => h.halo_id],
  ["log_mvir", "logM", h => h.log_mvir.toFixed(2)],
  ["num_p", "N_p", h => h.num_p.toLocaleString()],
  ["n_subhalos", "n_sub", h => h.n_subhalos],
  ["crop_side_sites", "side", h => h.crop_side_sites],
  ["auc", "AUC", h => {
    const L = (h.learnability || {})["footprint"];
    return L ? L.auc_sr2_rho.sigma2.toFixed(3) : "–";
  }],
];
function keyOf(h, k){
  if (k === "auc"){
    const L = (h.learnability || {})["footprint"];
    return L ? L.auc_sr2_rho.sigma2 : -1;
  }
  return h[k];
}
function buildTable(){
  const t = document.getElementById("hosts");
  t.querySelector("thead").innerHTML =
    "<tr>" + COLS.map(c => `<th data-k="${c[0]}">${c[1]}</th>`).join("") + "</tr>";
  t.querySelectorAll("th").forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    S.sortDir = (S.sortKey === k) ? -S.sortDir : -1;
    S.sortKey = k; renderTable();
  });
  renderTable();
}
function renderTable(){
  const order = H.map((h, i) => i).sort((a, b) =>
    S.sortDir * (keyOf(H[a], S.sortKey) - keyOf(H[b], S.sortKey)));
  const tb = document.querySelector("#hosts tbody");
  tb.innerHTML = order.map(i =>
    `<tr data-i="${i}" class="${i === S.host ? "sel" : ""}">` +
    COLS.map(c => `<td class="num">${c[2](H[i])}</td>`).join("") + "</tr>").join("");
  tb.querySelectorAll("tr").forEach(tr => tr.onclick = () => {
    S.host = +tr.dataset.i; renderTable(); drawAll();
  });
}

/* ---------- host facts ------------------------------------------------ */
function drawFacts(){
  const h = curHost(), r = h.resample;
  const rows = [
    ["log10 Mvir", h.log_mvir.toFixed(2)],
    ["particles", h.num_p.toLocaleString()],
    ["R_vir", h.rvir_mpc_h.toFixed(3) + " Mpc/h"],
    ["R_L (Lagrangian)", h.rl_mpc_h.toFixed(2) + " Mpc/h"],
    ["crop", `${h.crop_side_sites}³ sites = ${h.crop_side_mpc_h.toFixed(2)} Mpc/h`],
    ["drawn at", `${h.grid_out}³ (×${h.block_factor.toFixed(0)} blocks` +
                 `${h.vol_pad_sites ? ", " + h.vol_pad_sites + " pad" : ""})`],
    ["→ 96³ resample", `×${r.ratio.toFixed(2)} per axis`],
    ["HR subhalos", `${h.n_subhalos} (${h.n_subhalos_20p} ≥ 20p)`],
    ["host fills crop", (100 * h.host_fill_frac).toFixed(1) + "%"],
    ["in a subhalo", (100 * h.sub_frac_of_host).toFixed(1) + "% of host sites"],
    ["SR2 binds", (100 * h.sr2_bound_frac_in_host).toFixed(1) + "% of host sites"],
  ];
  document.getElementById("facts").innerHTML =
    rows.map(([k, v]) => `<b>${k}</b><span class="num">${v}</span>`).join("");
}

/* ---------- wiring ---------------------------------------------------- */
function drawAll(){ drawCrop(); drawCloud(); drawLearn(); drawAcross(); drawFacts(); }

function bind(){
  document.querySelectorAll("[data-field]").forEach(b => b.onclick = () => {
    S.field = b.dataset.field;
    document.querySelectorAll("[data-field]").forEach(x =>
      x.classList.toggle("on", x === b));
    drawCrop();
  });
  document.getElementById("axis").onchange = e => {
    S.axis = +e.target.value; drawCrop(); };
  document.getElementById("proj").onchange = e => {
    S.proj = +e.target.value; drawCloud(); };
  document.getElementById("slab").oninput = e => {
    S.slab = +e.target.value / 1000; drawCrop(); };
  document.getElementById("thick").oninput = e => {
    S.thick = +e.target.value / 100;
    document.getElementById("thicklab").textContent = S.thick.toFixed(2);
    drawCrop(); };
  document.getElementById("minp").oninput = e => {
    S.minP = +e.target.value;
    document.getElementById("minplab").textContent = S.minP;
    drawCrop(); drawCloud(); };
  document.getElementById("subs").onclick = e => {
    S.showSubs = !S.showSubs;
    e.target.classList.toggle("on", S.showSubs); drawCrop(); drawCloud(); };
  document.getElementById("inslab").onclick = e => {
    S.inSlab = !S.inSlab;
    e.target.classList.toggle("on", S.inSlab); drawCrop(); };
  document.onkeydown = e => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp"){
      S.host = (S.host + (e.key === "ArrowDown" ? 1 : H.length - 1)) % H.length;
      renderTable(); drawAll(); e.preventDefault();
    }
  };
}

buildTable(); bind(); drawAll();
"""


CONTENT = """<style>{css}</style>
<header>
  <h1>Option A host-frame crops &mdash; {box}</h1>
  <p>One HR host per row. The crop is the cube of
  <b>docs/sr2_substructure_module.md</b> &sect;3: side <b>2&nbsp;R<sub>L</sub></b>
  about the host's <b>Lagrangian</b> centroid, taken on the native 512&sup3;
  lattice and <em>not</em> resampled &mdash; the &times;96&sup3; ratio Option A
  would apply is reported instead. SR2 and HR are indexed by the same Lagrangian
  sites, so the two panels are the same sites twice, and the circles are HR's
  subhalos drawn where their Lagrangian material comes from.
  Colour is a k&#8209;nearest&#8209;neighbour local <b>Eulerian</b> log density
  gathered back onto the Lagrangian site (k={k}), identical estimator on both
  sides. Nothing here is trained or fitted.</p>
</header>
<div class="wrap">
  <div>
    <div class="panel">
      <h2>hosts</h2>
      <div class="scroll"><table id="hosts"><thead></thead><tbody></tbody></table></div>
      <p class="cap">Click a row (or press &uarr;/&darr;). AUC is how well SR2's
      smoothed local density ranks this host's Lagrangian sites by "HR put a
      subhalo here". <b>Distance from 0.5 is the signal, in either direction</b>
      &mdash; 0.5 is no information, and below 0.5 the ranking is simply
      reversed: subhalos sit where the local density is <em>low</em>.</p>
    </div>
    <div class="panel">
      <h2>selected host</h2>
      <div class="kv" id="facts"></div>
    </div>
  </div>

  <div>
    <div class="panel">
      <h2>1 &middot; Lagrangian crop</h2>
      <div class="ctl">
        <span>
          <button data-field="sr2_rho" class="on">SR2 density</button>
          <button data-field="hr_rho">HR density</button>
          <button data-field="diff">HR &minus; SR2</button>
          <button data-field="hr_sub">HR subhalo mask</button>
        </span>
        <label>view along
          <select id="axis"><option value="0">x</option><option value="1">y</option>
          <option value="2" selected>z</option></select></label>
        <label>slab <input id="slab" type="range" min="0" max="1000" value="500"></label>
        <label>thickness <input id="thick" type="range" min="2" max="100" value="15">
          <span class="num" id="thicklab">0.15</span></label>
        <label>min N<sub>p</sub> <input id="minp" type="range" min="0" max="500" value="20">
          <span class="num" id="minplab">20</span></label>
        <button id="subs" class="on">subhalos</button>
        <button id="inslab" class="on">slab only</button>
      </div>
      <div class="row">
        <canvas id="crop" width="560" height="560"></canvas>
        <div>
          <div class="kv">
            <b>slab</b><span id="slabinfo" class="num"></span>
            <b>circles drawn</b><span id="nsubshown" class="num"></span>
          </div>
          <p class="cap" style="margin-top:10px">
            Circle radius is the subhalo's own R<sub>L</sub> &mdash; the
            Lagrangian patch its particles came from &mdash; not its virial
            radius, because that is the region a Lagrangian generator has to
            act on. Faint circles are subhalos whose material lies outside the
            current slab.</p>
          <div style="margin-top:10px">
            <canvas id="cbar" width="200" height="10"></canvas>
            <div style="display:flex;justify-content:space-between;width:200px;
                 color:var(--dim);font-size:11px">
              <span id="cblo" class="num"></span><span id="cbhi" class="num"></span>
            </div>
          </div>
        </div>
      </div>
      <p class="note">The honest read of this panel: does the SR2 image already
      have something at each circle? Where it does, a model only has to sharpen
      what is there. Where the circle sits on smooth SR2 material, the position
      itself has to be invented, and that is the part a deterministic map cannot
      do.</p>
    </div>

    <div class="panel">
      <h2>2 &middot; Eulerian cloud</h2>
      <div class="ctl">
        <label>project along
          <select id="proj"><option value="0">x</option><option value="1">y</option>
          <option value="2" selected>z</option></select></label>
        <span id="cloudscale" class="num"></span>
      </div>
      <div class="row">
        <div><canvas id="cloud_hr" width="330" height="330"></canvas>
          <p class="cap"><span class="swatch" style="background:var(--hr)"></span>HR</p></div>
        <div><canvas id="cloud_sr2" width="330" height="330"></canvas>
          <p class="cap"><span class="swatch" style="background:var(--sr2)"></span>SR2</p></div>
      </div>
      <p class="cap">Both clouds are the <em>same Lagrangian sites</em>, moved by
      each field's own displacement, and share one scale. Orange circles are the
      HR subhalos at their catalog positions &mdash; on the SR2 panel they mark
      what is missing.</p>
    </div>

    <div class="panel">
      <h2>3 &middot; how learnable is a subhalo's position</h2>
      <div class="row">
        <div>
          <div id="roc"></div>
          <p class="cap" style="max-width:230px">ROC over the host's own
            footprint. <span class="swatch" style="background:var(--sr2)"></span>SR2
            density, <span class="swatch" style="background:var(--hr)"></span>HR
            density (the ceiling: the same statistic read off the answer).</p>
        </div>
        <div style="flex:1;min-width:340px">
          <div id="aucbars"></div>
          <div class="kv" style="margin-top:8px">
            <b>sites in footprint</b><span id="nfoot" class="num"></span>
            <b>of them in a subhalo</b><span id="baserate" class="num"></span>
          </div>
        </div>
        <div>
          <canvas id="persub" width="260" height="200"></canvas>
          <p class="cap" style="max-width:260px">Per HR subhalo: x = log&#8321;&#8320;
            N<sub>p</sub> (0&ndash;4), y = fraction of its Lagrangian sites SR2
            binds to <em>any</em> object. High y with a missing subhalo means the
            material is there and unfragmented.</p>
        </div>
      </div>
    </div>

    <div class="panel">
      <h2>4 &middot; across hosts</h2>
      <canvas id="across" width="560" height="200"></canvas>
      <p class="cap">y = AUC 0&ndash;1 (dashed line 0.5).
        <span class="swatch" style="background:var(--sr2)"></span>SR2,
        <span class="swatch" style="background:var(--hr)"></span>HR ceiling.
        x = <span id="acrossx" class="num"></span>. Read the <em>gap between the
        two colours</em>, not the height: HR is the same statistic computed on
        the answer, so where SR2 sits on top of it, SR2's field is as informative
        about subhalo positions as HR's own field is, and the missing
        substructure is not a missing input.</p>
    </div>
  </div>
</div>
<script>window.__CROPS__ = {payload};</script>
<script>{js}</script>
"""

# The standalone file the cluster job writes. ``--body-only`` emits CONTENT
# alone, for a host that wraps the content in its own <head>/<body>.
SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Host-frame crops &mdash; {box}</title>
</head><body>
{content}
</body></html>
"""


def render(summary: dict, store, out: Path, *, n_hosts: int,
           body_only: bool = False) -> Path:
    payload = build_payload(summary, store, n_hosts=n_hosts)
    content = CONTENT.format(
        box=summary["box"], css=CSS, js=APP_JS,
        k=summary.get("k_neighbours", "?"),
        # `</` cannot appear inside a <script> body; base64 and our keys never
        # produce it, but the escape costs nothing and removes the question.
        payload=json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"),
    )
    html = content if body_only else SHELL.format(
        box=summary["box"], content=content)
    out.write_text(html, encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8")
    ap.add_argument("--n-hosts", type=int, default=0, help="0 = every host collected")
    ap.add_argument("--out", default="")
    ap.add_argument("--body-only", action="store_true",
                    help="emit the content without a document shell")
    args = ap.parse_args(argv)

    for box in [b for b in args.boxes.split(",") if b]:
        d = paths.subdir("lagrangian_host", box)
        js_path, npz_path = d / f"{box}_host_crops.json", d / f"{box}_host_crops.npz"
        if not js_path.exists() or not npz_path.exists():
            print(f"GATE: {box} has no crop artifacts; run "
                  f"scripts/features/collect_host_crops.py --boxes {box}")
            continue
        banner(f"render host crops: {box}")
        summary = json.loads(js_path.read_text())
        store = np.load(npz_path)
        n = args.n_hosts or len(summary["hosts"])
        out = Path(args.out) if args.out else d / f"host_crops_{box}.html"
        render(summary, store, out, n_hosts=n, body_only=args.body_only)
        mb = out.stat().st_size / 1e6
        print(f"  wrote {out}  ({mb:.1f} MB, {min(n, len(summary['hosts']))} hosts)")
        print(f"  scp it off and open it: scp rhea:{out} .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
