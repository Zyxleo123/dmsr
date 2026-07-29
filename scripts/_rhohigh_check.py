"""How much signal did the critic's rho_high channel actually carry?

Compares the density high-pass at the as-run HR cellsize (15625, 80x too large)
against the correct one (100000/512 = 195.3125), on real paired HR crops.
"""
import sys; sys.path.insert(0, "src")
import torch, numpy as np
from cosmo_sr.utils.config import load_config
from cosmo_sr.dmsr.data import resolve_split, build_val_dataset
from cosmo_sr.data.datasets import finite_loader
from cosmo_sr.dmsr.density import HighPassDensity
from cosmo_sr.dmsr.operator import NullSpaceOperator

cfg = load_config("configs/dmsr/stage_c_critic_pairedlr.yaml")
d = cfg["data"]; factor = int(cfg["factor"])
split = resolve_split(d)
ds = build_val_dataset(split, crop_lr=int(d["crop_lr"]), scale_factor=factor,
                       channels=int(d["channels"]), use_channels=d["use_channels"],
                       mmap=True, max_crops=8, which="val")
op = NullSpaceOperator(factor=factor)
batch = next(iter(finite_loader(ds, 4)))
x = batch["hr"]

print(f"HR crop {tuple(x.shape)}")
print(f"{'cellsize':>12}{'std(rho)':>12}{'std(rho_high)':>16}{'rho_high/resid':>17}")
res = op.P_A(x)
res_std = float(res.std())
for label, cs in (("as-run 15625", 15625.0), ("correct 195.3", 100000.0/512.0)):
    hp = HighPassDensity(factor=factor, cellsize=cs, dis_norm=float(d["dis_norm"]))
    rho = hp.density(x); rh = hp(x)
    print(f"{label:>12}{float(rho.std()):>12.5f}{float(rh.std()):>16.5f}"
          f"{float(rh.std())/res_std:>17.5f}")
print(f"\nfield residual P_A(x) std = {res_std:.5f}")
print("The critic sees concat(residual[3ch], rho_high[1ch]). If rho_high's scale is")
print("orders of magnitude below the residual's, that channel contributes ~nothing")
print("and the 'Eulerian critic' is effectively a residual-only critic.")
