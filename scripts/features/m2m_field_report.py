#!/usr/bin/env python
"""Score a map2map checkpoint on the yardstick, not on its loss.

``m2m.py train`` logs MSE, and MSE is the wrong number for this line: frozen
SR2's own high-k ``r(k)`` is ~0 on every channel, so it does not reproduce HR's
realisation at small scales by design, and a model that beats SR2 on MSE is most
likely suppressing small-scale power. MEASURED on set14, trilinear interpolation
scores hr_mse 0.241 against SR2's 0.435 and loses on density. The verdict is
density ``P/P_HR`` at high k: frozen SR2 **0.977**, trilinear **0.706**.

This is the map2map twin of ``sr2_scratch_field_report.py`` and imports that
script's ``field_table``/``whole_box_holdout`` rather than restating them, so the
numbers are comparable to every other row this line has produced. It is a
separate script rather than a flag on that one for two reasons, both of which
would otherwise be silently wrong:

* **A different generator.** ``styled_srsgan.G`` is NOT state-dict compatible
  with our ``ControlledG``. MEASURED on ``state_1.pt``: 46 parameters against 26,
  only 8 shared keys, and the shared ones are coincidences of naming. The conv
  stacks are the same architecture under different ``nn.Sequential`` indexing,
  but the styled variant adds ``style_block.*`` modulation (``w = weight * s``,
  then demodulate) that ``ControlledG`` does not have at all. At ``a = 1.0`` that
  ``s`` is a LEARNED linear, not the identity, so remapping the keys and loading
  into ``ControlledG`` would run a different function and report the answer as
  if it were this checkpoint's.

* **A different inference path.** ``super_resolve_srs_seeded`` refuses anything
  but a ``ControlledG``, and says why: the upstream G draws its noise from the
  global RNG and cannot be steered per tile. So this uses
  ``eval.baseline_srs.super_resolve_srs``, which is the mirror of upstream's own
  ``lr2sr.py`` -- one ``torch.manual_seed`` and a fixed tile walk. The
  consequence is stated rather than hidden: the REALISATION depends on tile
  order here, where the seeded path guarantees it does not. That is acceptable
  for ``P/P_HR``, which is a statistic over modes, and it is why ``r(k)`` from
  this script should not be compared across runs at face value.

``sr2_scratch_field_report.py`` refuses checkpoints without a ``scratch`` flag,
on the grounds that filing a fine-tune result under a from-scratch arm is the
failure this line keeps paying for. The same discipline applies here in the
other direction: this script refuses a checkpoint that does NOT look like
map2map's, and stamps the provenance into the output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features", PROJECT_ROOT / "scripts" / "sr2"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, write_json  # noqa: E402
from _sr2_direct import geometry_of, load_direct_config, load_hr, load_lr  # noqa: E402
from overfit_host_mse import BOXSIZE, NG_HR  # noqa: E402
from sr2_scratch_field_report import field_table, whole_box_holdout  # noqa: E402

from cosmo_sr.eval.baseline_srs import super_resolve_srs  # noqa: E402
from cosmo_sr.reward.base import find_base_field  # noqa: E402

#: The bar. See the module docstring.
REFERENCE = {"frozen_sr2_density": 0.977, "trilinear_density": 0.706}


def _as_floats(x):
    """A checkpoint scalar, vector or None as plain JSON-able floats."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x) if x.numel() == 1 else [float(v) for v in x.flatten()]
    return float(x)


class StyleBound(torch.nn.Module):
    """Bind ``style`` so a styled generator matches ``super_resolve_srs``.

    That helper calls ``generator(t)`` with one argument, which is upstream's own
    ``lr2sr.py`` convention; the styled branch's ``train.py`` calls
    ``model(input, style)`` unconditionally. The style is the scale factor ``a``
    and our boxes are z = 0, so it is a constant 1.0 and binding it here is an
    adapter, not a modelling choice.
    """

    def __init__(self, g: torch.nn.Module, a: float, style_size: int):
        super().__init__()
        self.g = g
        self.register_buffer("style",
                             torch.full((1, int(style_size)), float(a)))

    def forward(self, x):
        return self.g(x, self.style.to(x.device, x.dtype).expand(x.shape[0], -1))


def load_m2m_generator(ckpt: Path, repo: Path, *, scale_factor: int,
                       in_chan: int, out_chan: int, style_size: int,
                       style_a: float, model: str, device):
    """``(module, provenance)`` for a map2map ``state_N.pt``.

    Constructs the generator with map2map's own ``import_attr`` and the same
    argument shape ``train.py:139`` uses, so a class that would not have been
    built by a real run cannot be built here either.
    """
    if not (repo / "m2m.py").is_file():
        raise SystemExit(f"no map2map checkout at {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    missing = {"model", "epoch"} - set(blob)
    if missing:
        raise SystemExit(
            f"{ckpt} is missing {sorted(missing)}, so it was not written by "
            "m2m.py train. Scoring it here would file some other run's weights "
            "under a map2map arm's name.")

    from map2map import models as m2m_models
    from map2map.utils import import_attr
    cls = import_attr(model, m2m_models, callback_at=None)
    # The two trees differ in whether the generator is styled, and the
    # constructor signature is the honest way to ask. `styled_srsgan.G` requires
    # `style_size` and its forward takes a style; the colleague's `srsgan.G` has
    # neither -- their train.py:142 is
    #     model(sum(in_chan), sum(out_chan), scale_factor=scale_factor)
    # and their train.py:397 is `output = model(input)`. Passing style_size to
    # the latter is a TypeError, and wrapping it in StyleBound would hand it a
    # second positional argument it does not accept.
    import inspect
    styled = "style_size" in inspect.signature(cls.__init__).parameters
    if styled:
        g = cls(int(in_chan), int(out_chan), style_size=int(style_size),
                scale_factor=int(scale_factor))
    else:
        g = cls(int(in_chan), int(out_chan), scale_factor=int(scale_factor))
    g.load_state_dict(blob["model"])
    # StyleBound exists only so super_resolve_srs can call generator(t) with one
    # argument on a STYLED generator. An unstyled one already has that signature.
    g = (StyleBound(g, style_a, style_size) if styled else g).to(device).eval()
    return g, {
        "checkpoint": str(ckpt),
        "epoch": int(blob.get("epoch", -1)),
        "model": model,
        "style_a": float(style_a),
        # MEASURED: `min_loss` is None before validation has ever run and a
        # (5,) TENSOR afterwards -- map2map tracks the whole epoch_loss vector
        # (loss, loss_adv, adv_loss, adv_loss_fake, adv_loss_real), not a
        # scalar. float() on it raises, which killed job 68925 at load time.
        "min_loss": _as_floats(blob.get("min_loss")),
        "inference": "baseline_srs.super_resolve_srs (global-RNG noise, "
                     "tile-order dependent realisation)",
    }


def markdown(rows: Dict[str, Dict], boxes: List[str], prov: Dict) -> str:
    out = [f"# map2map checkpoint vs frozen SR2 -- {Path(prov['checkpoint']).name} "
           f"(epoch {prov['epoch']})", "",
           f"`{prov['model']}`, style a={prov['style_a']}. "
           f"Inference: {prov['inference']}.", "",
           f"Bar: density P/P_HR -- frozen SR2 "
           f"**{REFERENCE['frozen_sr2_density']}**, trilinear "
           f"**{REFERENCE['trilinear_density']}**. Below trilinear is worse "
           f"than interpolation.", ""]
    for box in boxes:
        r = rows.get(box)
        if not r:
            continue
        out += [f"## {box}", "",
                "| field | P/P_HR (hi-k) | T(k) | r(k) | | frozen P/P_HR | "
                "frozen T(k) | frozen r(k) |",
                "|---|---:|---:|---:|---|---:|---:|---:|"]
        for f in ("displacement", "velocity", "density"):
            c, z = r["candidate"][f], r["frozen"][f]
            out.append(
                f"| {f} | {c['ratio_highk']:.3f} | {c['transfer_highk']:.3f} | "
                f"{c['rk_highk']:.3f} | | {z['ratio_highk']:.3f} | "
                f"{z['transfer_highk']:.3f} | {z['rk_highk']:.3f} |")
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a map2map state_N.pt")
    ap.add_argument("--repo", default="/zfsauton2/home/yixiz/DMSR/map2map_styled")
    ap.add_argument("--model", default="styled_srsgan.G")
    ap.add_argument("--style-a", type=float, default=1.0,
                    help="the scale factor a; our boxes are z=0 so a=1.0")
    ap.add_argument("--style-size", type=int, default=1)
    ap.add_argument("--config", default="configs/reward/sr2_direct_finetune.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--boxes", default="set9 set10",
                    help="held-out boxes; the verdict is a box never trained on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="")
    a = ap.parse_args(argv)

    cfg = load_direct_config(a)
    geom = geometry_of(cfg)
    device = torch.device(a.device
                          or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = Path(a.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint at {ckpt}")

    g, prov = load_m2m_generator(
        ckpt, Path(a.repo), scale_factor=geom.scale_factor, in_chan=6,
        out_chan=6, style_size=a.style_size, style_a=a.style_a,
        model=a.model, device=device)
    banner(f"m2m field report: {ckpt.name} (epoch {prov['epoch']})")
    print(f"  provenance: {json.dumps(prov)}", flush=True)

    boxes = [b for b in str(a.boxes).replace(",", " ").split() if b]
    cell = BOXSIZE * 1e3 / float(NG_HR)
    dis_norm = float(cfg.get("geometry", {}).get("dis_norm_kpc_h", 6000.0))
    rows: Dict[str, Dict] = {}

    for box in boxes:
        t0 = time.time()
        frozen_path = find_base_field(box, seed=int(a.seed))
        if frozen_path is None:
            print(f"  {box}: SKIPPED -- no cached frozen SR2 box for seed "
                  f"{a.seed}. Build it with scripts/reward/cache_sr2_base.py.",
                  flush=True)
            continue
        hr = np.ascontiguousarray(load_hr(cfg, box), dtype=np.float32)
        lr = np.asarray(load_lr(cfg, box), dtype=np.float32)
        sr = super_resolve_srs(g, lr, scale_factor=geom.scale_factor,
                               nsplit=geom.nsplit, pad=geom.pad,
                               seed=int(a.seed), device=device)
        frozen = np.asarray(np.load(str(frozen_path), mmap_mode="r"),
                            dtype=np.float32)
        rows[box] = {
            "candidate": field_table(sr, hr, cell_kpc_h=cell,
                                     dis_norm_kpc_h=dis_norm, device=device),
            "frozen": field_table(frozen, hr, cell_kpc_h=cell,
                                  dis_norm_kpc_h=dis_norm, device=device),
            "candidate_holdout": whole_box_holdout(
                sr, hr, scale_factor=geom.scale_factor, device=device),
            "frozen_holdout": whole_box_holdout(
                frozen, hr, scale_factor=geom.scale_factor, device=device),
            "frozen_field": str(frozen_path),
        }
        c = rows[box]["candidate"]["density"]
        z = rows[box]["frozen"]["density"]
        verdict = ("ABOVE frozen SR2" if c["ratio_highk"] >= z["ratio_highk"]
                   else "above trilinear"
                   if c["ratio_highk"] >= REFERENCE["trilinear_density"]
                   else "BELOW TRILINEAR -- worse than interpolation")
        print(f"  {box} [{time.time() - t0:.0f}s]  density P/P "
              f"{c['ratio_highk']:.3f} (frozen {z['ratio_highk']:.3f})  "
              f"-> {verdict}", flush=True)

    if not rows:
        print("\nno box scored; nothing written.")
        return 0

    out = Path(a.out) if a.out else ckpt.parent / f"field_report_{ckpt.stem}"
    write_json(str(out.with_suffix(".json")),
               {"provenance": prov, "reference": REFERENCE, "boxes": rows})
    out.with_suffix(".md").write_text(markdown(rows, boxes, prov))
    print(f"\nwrote {out.with_suffix('.json')}\n      {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
