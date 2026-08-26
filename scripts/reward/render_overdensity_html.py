#!/usr/bin/env python
"""Draw the overdensity slab + centre markers as one self-contained HTML page.

Reads the ``map.npz`` written by :mod:`scripts/reward/collect_overdensity_map.py`
and emits ``overdensity_<box>.html`` -- the projected overdensity as a base64 PNG
background with an SVG overlay of the three marker sets on top. Nothing is fetched
at view time (the image and every coordinate are embedded), so the file opens
anywhere. Because it only reads the npz it is a pure redraw: rerun it to retune
colours, thresholds or the zoom without touching the field.

    python scripts/reward/render_overdensity_html.py --boxes set8,set9
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict

import numpy as np

from _common import banner, paths  # noqa: E402


def _heatmap_png_b64(over: np.ndarray) -> str:
    """Projected overdensity -> base64 PNG (log colour, no axes/margins)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logd = np.log10(np.clip(over, 0.0, None) + 0.1)
    vmin = float(np.quantile(logd, 0.02))
    vmax = float(np.quantile(logd, 0.999))
    buf = io.BytesIO()
    # imsave writes exactly the array as pixels (no axes); origin="lower" puts
    # row 0 at the bottom so data-y increases upward, matching the SVG flip below.
    plt.imsave(buf, logd.T, cmap="magma", vmin=vmin, vmax=vmax,
               origin="lower", format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _payload(z: Dict) -> Dict:
    """Everything the page's JS needs, in plain (JSON-safe) types."""
    return {
        "box_l": float(z["box_l"]),
        "z0": float(z["z0"]), "dz": float(z["dz"]),
        "cell": float(z["cell_mpc"]),
        "base_sub": z["base_sub_xy"].tolist(),
        "dog": z["dog_xy"].tolist(),
        "hr_sub": z["hr_sub_xy"][~z["hr_missing"]].tolist(),
        "missing": z["hr_sub_xy"][z["hr_missing"]].tolist(),
        "target": z["target_xy"].tolist(),
    }


_TEMPLATE = """<title>Overdensity slab &middot; __BOX__</title>
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; font: 14px/1.5 system-ui, sans-serif;
         background: #0d0d10; color: #e8e8ec; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: #9aa0aa; margin: 0 0 14px; }
  .stage { position: relative; width: 100%; aspect-ratio: 1 / 1;
           border: 1px solid #2a2a30; border-radius: 6px; overflow: hidden;
           background: #000; }
  .stage img, .stage svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  .stage img { image-rendering: pixelated; }
  .controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
              margin: 12px 0; }
  .controls label { display: inline-flex; align-items: center; gap: 6px;
                    cursor: pointer; user-select: none; }
  .swatch { width: 14px; height: 14px; border-radius: 50%; display: inline-block; }
  select, button { background: #1b1b20; color: #e8e8ec; border: 1px solid #33333a;
                   border-radius: 5px; padding: 5px 8px; font: inherit; }
  .meta { color: #9aa0aa; font-size: 12.5px; margin-top: 10px; }
  .meta code { color: #cdd2da; }
</style>
<div class="wrap">
  <h1>Frozen-SR2 overdensity slab &mdash; __BOX__</h1>
  <p class="sub">Background: projected <code>1 + &delta;</code> (log colour) through
    a slab <code>|z &minus; __Z0__| &le; __DZ__ Mpc/h</code>. Markers are centres
    that fall inside the same slab.</p>

  <div class="controls">
    <label><input type="checkbox" id="c-base" checked>
      <span class="swatch" style="background:#35d0ff"></span> candidate: SR2 subhalos (<span id="n-base"></span>)</label>
    <label><input type="checkbox" id="c-dog" checked>
      <span class="swatch" style="background:#ffd23f"></span> candidate: DoG peaks (<span id="n-dog"></span>)</label>
    <label><input type="checkbox" id="c-hr" checked>
      <span class="swatch" style="background:#57e389"></span> real HR subhalos (<span id="n-hr"></span>)</label>
    <label><input type="checkbox" id="c-miss" checked>
      <span class="swatch" style="background:#ff4d6d;border-radius:2px"></span> real: MISSING targets (<span id="n-miss"></span>)</label>
  </div>
  <div class="controls">
    <label>zoom to missing target:
      <select id="zoom"></select></label>
    <button id="reset">reset view</button>
    <span class="meta">cell = <code id="cell"></code> Mpc/h &middot; box = <code id="boxl"></code> Mpc/h</span>
  </div>

  <div class="stage">
    <img src="data:image/png;base64,__PNG__" alt="projected overdensity">
    <svg id="ov" preserveAspectRatio="none"></svg>
  </div>
  <p class="meta">Candidates (cyan/yellow) are built from frozen SR2 alone &mdash;
    no ground truth. Green + red are the true HR subhalos; red rings are the ones
    frozen SR2 fails to form. Where a red ring has no cyan/yellow marker on it, the
    reward has no centre to seed that structure.</p>
</div>
<script>
const D = __DATA__;
const L = D.box_l;
const svg = document.getElementById("ov");
svg.setAttribute("viewBox", `0 0 ${L} ${L}`);

// data (x,y up) -> svg (y down): sy = L - y
const layers = {};
function draw(id, pts, kind, color, r) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  for (const [x, y] of pts) {
    let el;
    if (kind === "ring") {
      el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      el.setAttribute("cx", x); el.setAttribute("cy", L - y); el.setAttribute("r", r);
      el.setAttribute("fill", "none"); el.setAttribute("stroke", color);
      el.setAttribute("stroke-width", 1.6);
      el.setAttribute("vector-effect", "non-scaling-stroke");
    } else if (kind === "diamond") {
      el = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      const sy = L - y;
      el.setAttribute("x", x - r); el.setAttribute("y", sy - r);
      el.setAttribute("width", 2 * r); el.setAttribute("height", 2 * r);
      el.setAttribute("transform", `rotate(45 ${x} ${sy})`);
      el.setAttribute("fill", color); el.setAttribute("opacity", 0.85);
    } else {
      el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      el.setAttribute("cx", x); el.setAttribute("cy", L - y); el.setAttribute("r", r);
      el.setAttribute("fill", color); el.setAttribute("opacity", 0.85);
    }
    g.appendChild(el);
  }
  svg.appendChild(g);
  layers[id] = g;
}

const rBase = L * 0.0016, rDog = L * 0.0018, rHr = L * 0.0014, rMiss = L * 0.007;
draw("hr", D.hr_sub, "dot", "#57e389", rHr);
draw("base", D.base_sub, "dot", "#35d0ff", rBase);
draw("dog", D.dog, "diamond", "#ffd23f", rDog);
draw("miss", D.missing.length ? D.missing : D.target, "ring", "#ff4d6d", rMiss);

const bind = (cb, id) => {
  const el = document.getElementById(cb);
  el.addEventListener("change", () =>
    layers[id].style.display = el.checked ? "" : "none");
};
bind("c-base", "base"); bind("c-dog", "dog");
bind("c-hr", "hr"); bind("c-miss", "miss");

document.getElementById("n-base").textContent = D.base_sub.length;
document.getElementById("n-dog").textContent = D.dog.length;
document.getElementById("n-hr").textContent = D.hr_sub.length;
const miss = D.missing.length ? D.missing : D.target;
document.getElementById("n-miss").textContent = miss.length;
document.getElementById("cell").textContent = D.cell.toFixed(3);
document.getElementById("boxl").textContent = L.toFixed(0);

const zoom = document.getElementById("zoom");
zoom.innerHTML = '<option value="-1">(whole box)</option>' +
  miss.map((p, i) => `<option value="${i}">#${i + 1} @ (${p[0].toFixed(1)}, ${p[1].toFixed(1)})</option>`).join("");
const PAD = Math.max(3, L * 0.05);
zoom.addEventListener("change", () => {
  const i = +zoom.value;
  if (i < 0) { svg.setAttribute("viewBox", `0 0 ${L} ${L}`); return; }
  const [x, y] = miss[i], sy = L - y;
  svg.setAttribute("viewBox", `${x - PAD} ${sy - PAD} ${2 * PAD} ${2 * PAD}`);
});
document.getElementById("reset").addEventListener("click", () => {
  zoom.value = "-1"; svg.setAttribute("viewBox", `0 0 ${L} ${L}`);
});
</script>
"""


def render_box(box: str, out_dir: Path) -> Path:
    npz = out_dir / box / "map.npz"
    if not npz.is_file():
        raise SystemExit(f"no map.npz for {box} at {npz}; run collect_overdensity_map first")
    z = dict(np.load(npz))
    png = _heatmap_png_b64(z["overdensity"])
    data = _payload(z)
    html = (_TEMPLATE
            .replace("__BOX__", box)
            .replace("__Z0__", f"{float(z['z0']):.1f}")
            .replace("__DZ__", f"{float(z['dz']):.1f}")
            .replace("__PNG__", png)
            .replace("__DATA__", json.dumps(data)))
    out = out_dir / box / f"overdensity_{box}.html"
    out.write_text(html)
    banner(f"wrote {out}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boxes", default="set8,set9")
    ap.add_argument("--out-name", default="overdensity_map")
    args = ap.parse_args(argv)

    out_dir = paths.subdir("audits", args.out_name, create=True)
    for b in (b.strip() for b in args.boxes.split(",") if b.strip()):
        render_box(b, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
