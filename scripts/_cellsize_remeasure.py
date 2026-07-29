"""Re-measure the environment descriptors / source-classifier AUC at the CORRECT
LR cellsize, and quantify how much the 10x error suppressed them.

Box is 100 Mpc/h (paramfile.genic BoxSize=100000 kpc/h).
  LR grid 64  -> cellsize = 100000/64  = 1562.5 kpc/h   (environment_descriptors)
  HR grid 512 -> cellsize = 100000/512 =  195.3125 kpc/h (HighPassDensity)
The DMSR configs used 15625.0 for both.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
from cosmo_sr.utils.config import load_config
from cosmo_sr.dmsr.data import LRCropPool, resolve_split, build_balanced_sampler
from cosmo_sr.dmsr.env import DescriptorStandardizer, source_classifier_auc, DESCRIPTOR_NAMES

cfg = load_config("configs/dmsr/stage_d_critic_alllr.yaml")
d, e = cfg["data"], cfg["env"]
split = resolve_split(d)

WRONG = 15625.0
RIGHT = 100000.0 / 64.0          # 1562.5

N = int(e["pool_size"])
print(f"paired boxes={len(split.train_lr)}  unpaired boxes={len(split.lr_only)}  "
      f"pool={N}/{2*N}  use_tidal={e['descriptor']['use_tidal']}")
print()

for label, cs in (("AS RUN  (cellsize 15625, 10x too big)", WRONG),
                  ("CORRECT (cellsize 1562.5)", RIGHT)):
    dk = dict(e["descriptor"]); dk["cellsize"] = cs
    pk = dict(crop_lr=int(d["crop_lr"]), channels=int(d["channels"]),
              use_channels=d.get("use_channels"), mmap=bool(d["mmap"]),
              descriptor_kwargs=dk)
    pp = LRCropPool(split.train_lr, n_crops=N,     seed=0, **pk)
    up = LRCropPool(split.lr_only,  n_crops=2 * N, seed=1, **pk)
    pd_, ud_ = pp.descriptors(), up.descriptors()
    std = DescriptorStandardizer.fit(pd_)
    auc_before = source_classifier_auc(std.transform(pd_), std.transform(ud_), seed=0)
    s, _ = build_balanced_sampler(pp, up, n_dims=int(e["n_dims"]),
                                  n_bins=int(e["n_bins"]), seed=0)
    r = s.report()

    print(f"=== {label} ===")
    print(f"  kept descriptors : {std.kept_names}")
    print(f"  AUC before       : {auc_before:.4f}")
    print(f"  AUC after        : {r.auc_after:.4f}   in_support={r.n_in_support}/{r.n_unpaired}")
    # raw scale of the density-derived descriptors -- this is what the bug suppressed
    for nm in ("var_density", "tidal_I2", "tidal_I3"):
        i = DESCRIPTOR_NAMES.index(nm)
        print(f"    {nm:12s} paired mean={pd_[:, i].mean():+.6g}  sd={pd_[:, i].std():.6g}")
    print()
