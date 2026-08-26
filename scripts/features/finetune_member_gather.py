#!/usr/bin/env python
"""Fine-tune SR2 on the id-gathered member-set objective, over many hosts and boxes.

The successor to ``scripts/features/free_field_gather.py``, and the run that
``docs/sr2_member_gather.md`` section 7 items 3 and 4 were written to ask for.

What the free field settled, and what it did not
------------------------------------------------
The oracle optimised a free ``(4, 6, 64, 64, 64)`` tensor -- 6.3M parameters,
one per field value, no operator -- against the true member sets of one host of
one box, and recovered **72 of 154** supervised subhalos as bound halos that real
Rockstar matches, against a frozen baseline of 3 and a measured ceiling of 151.
That establishes the objective is **satisfiable and externally verifiable**: a
reachable field exists that satisfies the loss and that a 6-D phase-space finder
independently reads as correctly-placed bound substructure.

It establishes nothing about learning. A free field sees the answer, generalises
to nothing, and is not a model. This script replaces it with the generator:

    candidate = model(lr, noise)          instead of   frozen + delta

so one learned operator, applied at every site, has to produce the configuration
at every supervised host -- and is then scored on hosts and boxes it was never
trained on. The supervision (HR member sets and reference statistics) is data in
the loss exactly as HR displacements are data in an MSE; the generator's inputs
are unchanged, so a checkpoint from this run samples from ``(lr, noise)`` alone,
with no catalog and no HR field.

Three things this does that the free field deliberately did not
--------------------------------------------------------------
**The high-k hinge is on.** The free-field runs drove displacement power above
``k_split`` to 5.50x HR and it was still climbing at step 2000 (section 7 item
2). Editing 0.8% of one box, that is a caveat; a *generator* rewrites every site
of every box, so it is corruption. :mod:`cosmo_sr.features.field_guards` supplies
the hinge -- charging only for **exceeding** HR, never for falling short, because
building substructure raises high-k power and an L2 anchor was already measured
moving peak contrast the wrong way (``sr2_gather_finetune.md`` section 3.3).

**No gradient mask.** ``--mask-grad`` existed because the low-k guard's
block-averaged gradient plus Adam rewrote 99.56% of a free tile. A generator has
no such freedom and *should* feel the guards everywhere: masking would hide
exactly the collateral damage this run has to be judged on.

**Gradient accumulation over several hosts per step.** One host per step is a
very noisy estimate of a shared operator's gradient and invites chasing one
region at a time. ``--hosts-per-step`` accumulates before stepping.

What still cannot be claimed from this script alone
---------------------------------------------------
Its own numbers are member-set statistics -- ``bound_frac`` and friends -- and
they are surrogates this module computes about itself. Per
``occupation-ratio-is-gameable`` and ``tile-overfit-proxy-exploitation``, a
differentiable objective that scores itself gets gamed; the tile-overfit run drove
its proxy to +255 while real Rockstar showed no gain at all. **The verdict here is
FEASIBILITY ONLY.** The gate is a whole-box regeneration through Rockstar, scored
on the held-out pool, and it lives in the submitter's later stages.

    python scripts/features/finetune_member_gather.py --pool-only
    python scripts/features/finetune_member_gather.py --steps 8000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, paths, write_json  # noqa: E402
from _sr2_direct import (  # noqa: E402
    add_direct_args, geometry_of, load_direct_config, model_path_of,
    phase_space_config_of, soft_config_of,
)
# The oracle's own selection, tile builder and forward. A second definition of
# "which tiles are this host's" or "how a tile is generated" would mean this run
# is not working the problem docs/sr2_member_gather.md measured.
from overfit_host_mse import (  # noqa: E402
    BOXSIZE, DX, NG_HR, TILE, _catalog, _owner, build_tiles, forward_tiles,
)
# The loss configuration and the LR-scale guard, unchanged from the free-field
# run, for the same reason.
from free_field_gather import guards, member_config_of  # noqa: E402

from cosmo_sr.dmsr.critic import HRCritic, LazyR1, hinge_d_loss, hinge_g_loss  # noqa: E402
from cosmo_sr.eval.particle_identity import build_owner_index, child_map  # noqa: E402
from cosmo_sr.features.field_guards import (  # noqa: E402
    banded_highk_hinge, highk_hinge, highk_power_ratio_torch,
)
from cosmo_sr.features.gather_critic import (  # noqa: E402
    GatherCriticNorm, gather_critic_input, highpass_field,
)
from cosmo_sr.features.member_gather import (  # noqa: E402
    MemberSets, build_member_sets, member_gather_loss, tile_particles,
)
from cosmo_sr.features.member_pool import (  # noqa: E402
    HostSelection, select_hosts, split_pool, summarise_pool,
)
from cosmo_sr.reward.base import find_base_field  # noqa: E402
from cosmo_sr.train.common import finish_wandb, maybe_init_wandb  # noqa: E402
from cosmo_sr.train.sr2_finetune_data import trim_to_tile  # noqa: E402
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402
from cosmo_sr.train.sr2_unfreeze import (  # noqa: E402
    assert_only_trainable_changed, parameter_groups, snapshot_parameters,
    trainable_names,
)


# --------------------------------------------------------------------------- #
# One host's supervision and its tiles
# --------------------------------------------------------------------------- #
@dataclass
class HostTask:
    """Everything one supervised cluster contributes to a step.

    Held on the CPU. One host's tensors are moved to the GPU per step -- ~60 MB,
    a few milliseconds against a step dominated by the ``N^2`` potential -- so
    the pool size is bounded by host RAM rather than by GPU memory.
    """

    sel: HostSelection
    sets: MemberSets
    hr: torch.Tensor                    # (n_tiles, 6, T, T, T)
    tile_data: Dict[int, Dict]          # lr / noise / hr, per tile
    report: Dict
    #: The resolved HOST halos homing in these same tiles, as member sets, or
    #: ``None`` when ``--w-host-sets`` is 0. The same loss on these keeps the
    #: run from fragmenting hosts it is not being asked to build (measured
    #: damage: resolved hosts 3028 base -> 2708 tuned, HR 3775). The reference is
    #: the frozen field's own value, which already resolves these hosts, so the
    #: hinged terms start at ~0 and fire only when a host starts to come apart --
    #: a preservation guard, not a second objective competing for the budget.
    host_sets: Optional[MemberSets] = None

    @property
    def key(self) -> str:
        return self.sel.key

    @property
    def tiles(self) -> List[int]:
        return self.sel.tiles


def host_member_config(mcfg, args):
    """The member-gather config for the HOST preservation sets.

    The estimator (softening, ``bound_tau``, background, per-term weights) is the
    subhalo config's -- the same physics keeps a host bound as keeps a subhalo
    bound, and the overall importance of the host guard is the single
    ``--w-host-sets`` multiplier applied where the term enters the loss, not a
    second copy of six weights. Three fields differ:

    * ``min_num_p`` -> ``--host-min-num-p`` (200): the resolution floor for a
      *host*, independent of the subhalo floor.
    * ``max_sets`` -> ``--host-max-sets``: a cluster's tiles can home many more
      hosts than subhalos.
    * ``centre_mode`` -> ``self``: a host has no clustercentric direction of its
      own (it *is* the centre, so ``radial`` is undefined and would raise), and
      preservation means "stay where the frozen field already put you", which is
      exactly what the self-anchored centre term charges for -- zero at step 0,
      rising only if the host's centroid drifts.
    """
    return dataclasses.replace(
        mcfg,
        min_num_p=int(args.host_min_num_p),
        max_sets=int(args.host_max_sets),
        centre_mode="self",
    )


def build_host_sets(
    sel: HostSelection, cat, oidx, cfg, geom, scfg, pscfg, mcfg, args,
    frozen_model, device,
) -> Optional[HostTask]:
    """Select one host's member sets and measure the reference it can reach.

    ``frozen_field`` is the frozen generator's output on these tiles, not the
    cached full-box field: the two are produced by different generation paths and
    differ slightly, and every reference in the oracle was measured against the
    tile-wise one.
    """
    data = build_tiles(sel.box, sel.tiles, cfg, geom, device, args)
    hr = torch.stack([data[t]["hr"] for t in sel.tiles], dim=0).float()
    with torch.no_grad():
        base = forward_tiles(frozen_model, data, sel.tiles, geom).float()

    box_path = find_base_field(sel.box, seed=int(args.seed))
    if box_path is None:
        raise SystemExit(
            f"no cached frozen SR2 box for {sel.box} seed {args.seed}. Members "
            "outside the trained tiles would be dropped, which computes the "
            "potential of a fragment and OVERSTATES how bound it is. Produce it "
            "with scripts/reward/cache_sr2_base.py.")
    frozen_box = np.load(str(box_path), mmap_mode="r")

    kw = dict(ng_hr=NG_HR, tile_hr=TILE, boxsize_mpc_h=BOXSIZE,
              dis_scale_mpc_h=float(scfg.dis_norm_kpc_h) * 1e-3,
              vel_scale_kms=float(pscfg.vel_norm_km_s))
    report: Dict = {}
    # The host's own position, for `--centre-mode radial`. `cat.ids` is the
    # catalog's own ordering, so the row is looked up rather than assumed.
    _hrow = np.flatnonzero(np.asarray(cat.ids) == int(sel.halo_id))
    host_pos = (np.asarray(cat.pos[int(_hrow[0])], dtype=np.float64)
                if _hrow.size else None)
    sets = build_member_sets(
        cat, oidx, sel.tiles, hr, base, mcfg,
        particle_mass_msun_h=float(pscfg.particle_mass_msun_h),
        frozen_box=frozen_box, host_pos=host_pos, report=report, **kw)
    if sets.n_sets == 0:
        return None

    # --- the host preservation sets (--w-host-sets > 0) --------------------
    # The reference field is the FROZEN generator's own output (`base`), NOT HR,
    # and that is the whole point of the term. Against HR the hinges would start
    # positive and DRIVE hosts toward HR's concentration -- a second objective
    # competing with the subhalos. Against the frozen field the reference is the
    # pre-tune state the run is trying not to wreck: every term is exactly zero at
    # step 0 (cand == frozen in-tile) and rises only as a host falls below where
    # it started -- unbinds, puffs up, drifts. `hr_field=base` makes the reference
    # frozen; the out-of-tile stragglers already come from `frozen_box`, so a host
    # set is measured entirely against frozen. host_pos is None on purpose: a host
    # is its own centre, so there is no radial direction and centre_mode="self"
    # (baked into host_member_config) is the only sound choice. A cluster region
    # with no resolved host homing in its tiles contributes no host term -- not
    # fatal, so the missing-sets ValueError is swallowed.
    host_sets = None
    if float(getattr(args, "w_host_sets", 0.0)) > 0.0:
        hrep: Dict = {}
        try:
            host_sets = build_member_sets(
                cat, oidx, sel.tiles, base, base, host_member_config(mcfg, args),
                particle_mass_msun_h=float(pscfg.particle_mass_msun_h),
                frozen_box=frozen_box, host_pos=None, top_level=True,
                report=hrep, **kw)
        except ValueError:
            host_sets = None
        report["host_sets"] = hrep

    cpu = torch.device("cpu")
    return HostTask(
        sel=sel, sets=sets.to(cpu), hr=hr.to(cpu), report=report,
        tile_data={t: {"lr": data[t]["lr"].to(cpu),
                       "noise": {s: v.to(cpu) for s, v in data[t]["noise"].items()},
                       "hr": data[t]["hr"].to(cpu)}
                   for t in sel.tiles},
        host_sets=None if host_sets is None else host_sets.to(cpu))


def pair_cost(task: "HostTask", cap: int = 0, *,
              which: str = "sets") -> Dict[str, float]:
    """``sum_n_squared`` for one host's ``sets`` (or ``host_sets``): the pair bill.

    It is the only quantity in the pool that is quadratic, so it is the only one
    that can be surprising. A host whose sets are all 200p and a host with one
    118,000p satellite look identical in ``n_sets`` and differ by 4 orders of
    magnitude here -- and the second is what OOMed three GPUs on 2026-08-23.

    ``which="host_sets"`` bills the preservation sets instead. Hosts are the
    LARGER pair-sum risk of the two -- a host is by construction bigger than the
    subhalos inside it -- so ``--w-host-sets`` runs must read this line and set
    ``--max-set-particles`` before spending a GPU.
    """
    ms = task.sets if which == "sets" else task.host_sets
    if ms is None or ms.n_sets == 0:
        return {"sum_n_squared": 0.0, "max_n": 0.0, "n_sets": 0}
    n = np.asarray([int(x) for x in ms.num_p], dtype=np.float64)
    if cap > 0:
        n = np.minimum(n, float(cap))
    return {"sum_n_squared": float((n * n).sum()), "max_n": float(n.max(initial=0)),
            "n_sets": int(ms.n_sets)}


def report_pair_cost(tasks: Sequence["HostTask"], args) -> None:
    """Print the pair-sum bill, capped and uncapped, before spending a GPU on it."""
    cap = int(getattr(args, "max_set_particles", 0))
    raw = [pair_cost(t) for t in tasks]
    tot = sum(r["sum_n_squared"] for r in raw)
    big = max(raw, key=lambda r: r["max_n"]) if raw else {"max_n": 0}
    print(f"  pair sums: sum_n_squared {tot:.3e} over {len(tasks)} hosts, "
          f"largest set {big['max_n']:.0f} particles")
    if cap > 0:
        capped = sum(pair_cost(t, cap)["sum_n_squared"] for t in tasks)
        print(f"    --max-set-particles {cap}: {capped:.3e} "
              f"({tot / max(capped, 1.0):.1f}x cheaper). The estimator becomes "
              f"stochastic; see MemberGatherConfig.max_set_particles.")
    else:
        print("    --max-set-particles 0: the exact estimator, and the full "
              "bill. Every pair block is now recomputed in backward "
              "(member_gather._SpecificPotential), so this is a TIME cost and "
              "no longer a memory one.", flush=True)

    # The host preservation guard is the LARGER pair-sum risk when it is on --
    # a host is bigger than the subhalos inside it -- so its bill is printed
    # separately, and the same cap applies to it.
    if float(getattr(args, "w_host_sets", 0.0)) > 0.0:
        hraw = [pair_cost(t, which="host_sets") for t in tasks]
        htot = sum(r["sum_n_squared"] for r in hraw)
        n_host = sum(r["n_sets"] for r in hraw)
        hbig = max(hraw, key=lambda r: r["max_n"]) if hraw else {"max_n": 0}
        print(f"  host pair sums: sum_n_squared {htot:.3e} over {n_host} host "
              f"sets, largest {hbig['max_n']:.0f} particles")
        if cap > 0:
            hcap = sum(pair_cost(t, cap, which="host_sets")["sum_n_squared"]
                       for t in tasks)
            print(f"    --max-set-particles {cap}: {hcap:.3e} "
                  f"({htot / max(hcap, 1.0):.1f}x cheaper).", flush=True)
        else:
            print("    --max-set-particles 0 with --w-host-sets on: hosts can be "
                  "10^5 particles; set a cap unless the largest above is small.",
                  flush=True)


def _host_row(t: "HostTask") -> Dict:
    """One host's line in pool.json, including the reference it can reach.

    ``reference_median`` is the hybrid HR-in-tile / frozen-outside target, not
    pure HR: a member whose Lagrangian site is outside the trained tiles cannot
    be moved by the run or by the splice, so pure HR is not a reachable target
    and charging for it would charge for material nobody controls.
    """
    return {
        "key": t.key,
        "log_mvir": t.sel.log_mvir,
        "n_sets": int(t.sets.n_sets),
        "tiles": t.sel.tiles,
        "site_coverage": t.sel.site_coverage,
        "median_live_frac": t.report.get("median_live_frac"),
        "reference_median": t.report.get("reference_median"),
        "pure_hr_median": t.report.get("pure_hr_median"),
        # The host preservation guard (present only when --w-host-sets > 0).
        "host_n_sets": int(t.host_sets.n_sets) if t.host_sets is not None else 0,
        "host_reference_median": (t.report.get("host_sets") or {}).get(
            "reference_median"),
        "host_median_live_frac": (t.report.get("host_sets") or {}).get(
            "median_live_frac"),
    }


def build_pool(args, cfg, geom, scfg, pscfg, mcfg, frozen_model, device,
               boxes: Sequence[str]) -> List[HostTask]:
    """Every host of every box, one owner-array load per box.

    ``host_tiles`` reloads a 537 MB owner array and rebuilds its CSR index per
    call. Forty hosts that way is forty loads; here it is one per box, which is
    the whole startup cost -- roughly a minute a box, paid once per job.

    Deliberately NOT cached to disk. A cache keyed on anything less than the full
    member-gather config would silently serve references built under different
    softening, background or purity settings, and a wrong reachable reference is
    invisible: every statistic would look fine and mean something else.
    """
    tasks: List[HostTask] = []
    for box in boxes:
        owner_p = (paths.reward_root() / "halos_particles" / f"{box}__hr__hr"
                   / f"{box}_hr_owner.npy")
        if not owner_p.is_file():
            print(f"  {box}: no owner array -- SKIPPED. Build it with "
                  f"scripts/slurm/submit_owner_arrays.sh", flush=True)
            continue
        t0 = time.time()
        cat = _catalog(box, "hr")
        oidx = build_owner_index(_owner(box, "hr"))
        children = child_map(cat)
        sels = select_hosts(cat, oidx, box, n_tiles=int(args.n_tiles),
                            max_hosts=int(args.max_hosts_per_box),
                            min_log_mvir=float(args.min_log_mvir),
                            ng_hr=NG_HR, tile_hr=TILE, children=children)
        print(f"  {box}: {len(sels)} hosts >= logM {args.min_log_mvir} "
              f"[{time.time() - t0:.0f}s to load owner+catalog]", flush=True)
        for sel in sels:
            task = build_host_sets(sel, cat, oidx, cfg, geom, scfg, pscfg, mcfg,
                                   args, frozen_model, device)
            if task is None:
                print(f"    {sel.key}: 0 supervised sets, skipped")
                continue
            tasks.append(task)
            hn = (f", {task.host_sets.n_sets} host sets"
                  if task.host_sets is not None else
                  (", 0 host sets" if float(getattr(args, "w_host_sets", 0.0)) > 0
                   else ""))
            print(f"    {sel.key}: logM {sel.log_mvir:.2f}, "
                  f"{task.sets.n_sets:4d} sets{hn}, "
                  f"coverage {sel.site_coverage:.3f}, "
                  f"tiles {sel.tiles}", flush=True)
        del cat, oidx, children
    return tasks


# --------------------------------------------------------------------------- #
# The unsupervised-tile high-k support
# --------------------------------------------------------------------------- #
# Why this exists. The high-k hinge in `host_loss` is charged on `cand` vs `hr`
# for the SUPERVISED member tiles only -- the 40 training hosts' 160 tiles, 6.25%
# of the boxes. `all_blocks_selfvel` (2026-08-25) measured the consequence: a
# banded hinge sat at 0.56x HR on those tiles yet 3.9x on held-out tiles, the
# SAME 6x train/holdout gap the scalar had. The defect is not the guard's shape;
# it is that its support is a sliver of the box while a generator rewrites every
# site. This pool charges the (amplitude-only, phase-agnostic) banded hinge on
# RANDOM tiles drawn box-wide, against those tiles' own HR power -- the untried
# fix named in `selfvel-arm-failed-the-gate`. It only constrains the AMOUNT of
# small-scale power, never its placement (r(k)~0 above k_split for every field,
# frozen included): the placement defect is out of reach and out of scope here.
def build_tile_pool(boxes, per_box, cfg, geom, device, args, exclude, rng):
    """Random unsupervised tiles per box, held on the CPU like the host pool.

    ``exclude`` is the set of ``(box, tile)`` already supervised, kept disjoint so
    the pool is genuinely unsupervised support and not a second copy of the host
    tiles. One ``build_tiles`` call per box loads that box's LR/HR field once.
    """
    pool: List[Dict] = []
    if per_box <= 0:
        return pool
    n_tot = (NG_HR // TILE) ** 3
    for box in boxes:
        avail = [t for t in range(n_tot) if (box, t) not in exclude]
        rng.shuffle(avail)
        pick = sorted(avail[:per_box])
        if not pick:
            continue
        data = build_tiles(box, pick, cfg, geom, device, args)
        for t in pick:
            d = data[t]
            pool.append({
                "box": box, "tile": t,
                "lr": d["lr"].detach().cpu(),
                "noise": {s: v.detach().cpu() for s, v in d["noise"].items()},
                "hr": d["hr"].detach().cpu(),
            })
        del data
    print(f"  unsupervised-tile pool: {len(pool)} tiles over {len(boxes)} boxes "
          f"({per_box}/box requested)", flush=True)
    return pool


def _forward_unsup(model, batch: Sequence[Dict], geom, device):
    """``(candidate, hr)`` for a minibatch of unsupervised tiles.

    Mirrors ``overfit_host_mse.forward_tiles`` but takes tile dicts directly, so
    a tile index that repeats across two boxes cannot collide in a keyed dict.
    """
    lr = torch.stack([b["lr"].to(device) for b in batch], dim=0)
    sites = list(batch[0]["noise"].keys())
    noise = {s: torch.cat([b["noise"][s].to(device) for b in batch], dim=0)
             for s in sites}
    cand = trim_to_tile(model(lr, noise=noise), geom).float()
    hr = torch.stack([b["hr"].to(device) for b in batch], dim=0).float()
    return cand, hr


def _unsup_bands(model, batch: Sequence[Dict], geom, device, args):
    """``(penalty, ratio_per_bin, k_centres)`` for a minibatch. One-sided.

    Never two-sided: the k>10 deficit is SR2's own resolution limit, and a lower
    bound there would chase an unreachable target -- the reason `highk-two-sided`
    is documented as displacement-inappropriate. ``--highk-bins`` is forced to at
    least one band because the unsupervised guard is inherently banded.
    """
    cand, hr = _forward_unsup(model, batch, geom, device)
    pen, ratio, kband = banded_highk_hinge(
        cand, hr, dx=DX, k_split=float(args.k_split),
        n_bins=max(int(args.highk_bins), 1),
        k_max=(float(args.highk_k_max) if args.highk_k_max else None),
        tol=float(args.highk_tol), two_sided=False, reduce=str(args.highk_reduce))
    return pen, ratio, kband


@torch.no_grad()
def eval_unsup_highk(model, pool: Sequence[Dict], geom, device, args,
                     batch: int = 8) -> Optional[Dict]:
    """Median/worst per-band high-k ratio over an unsupervised tile pool.

    This is the number the verdict reads to decide whether the box-wide guard
    generalised. It is a diagnostic, not a term: computed under ``no_grad`` in
    chunks so a whole held-out box's tiles never build a tape.
    """
    if not pool:
        return None
    ratios, kband = [], None
    for i in range(0, len(pool), batch):
        _, ratio, kband = _unsup_bands(model, pool[i:i + batch], geom, device, args)
        ratios.append(ratio.detach().cpu().numpy())
    arr = np.stack(ratios, axis=0)          # (n_chunks, n_bins)
    med = np.nanmedian(arr, axis=0)
    return {
        "k": [float(x) for x in kband],
        "ratio": [float(x) for x in med],
        "ratio_max": float(np.nanmax(med)),
    }


# --------------------------------------------------------------------------- #
# One host's contribution to a step
# --------------------------------------------------------------------------- #
def sample_sets(task: HostTask, n: int, rng: random.Random) -> Optional[List[int]]:
    """Which of a host's sets this step supervises. ``None`` means all of them.

    Sets, not hosts, are the natural minibatch unit here: a host contributes ~134
    of them and their gradients are independent given the field, so taking all of
    them every step is a full-batch gradient over the expensive axis. Drawing a
    fixed number also makes a step's cost roughly independent of which host it
    landed on, which the pair sums otherwise make wildly uneven.
    """
    total = int(task.sets.n_sets)
    if n <= 0 or total <= n:
        return None
    return rng.sample(range(total), n)


def sample_host_sets(task: HostTask, n: int,
                     rng: random.Random) -> Optional[List[int]]:
    """Which of a host task's PRESERVATION sets this step charges. ``None`` = all.

    A cluster's tiles can home far more resolved hosts than subhalos, and each is
    an ``N^2`` pair sum, so the host guard is minibatched exactly as the subhalo
    objective is -- the guard's gradient is a mean over whatever it draws, so its
    scale does not depend on the draw size.
    """
    if task.host_sets is None:
        return None
    total = int(task.host_sets.n_sets)
    if n <= 0 or total <= n:
        return None
    return rng.sample(range(total), n)


def host_forward(model, task: HostTask, geom, kw, mcfg, device,
                 frozen_model=None, set_indices=None, term_scale=None,
                 host_set_indices=None):
    """``(candidate, base, hr, gather_loss, diagnostics, host)`` for one host.

    ``base`` is recomputed under ``no_grad`` rather than cached on the GPU: it is
    one frozen forward on four tiles, and caching it for every host of a large
    pool is the difference between fitting in GPU memory and not.

    ``host`` is ``None`` unless the task carries preservation sets, in which case
    it is ``(host_gather, host_diag)`` from the SAME particle table -- the host
    term never triggers a second forward or a second ``tile_particles``.
    """
    data = {t: {"lr": task.tile_data[t]["lr"].to(device),
                "noise": {s: v.to(device)
                          for s, v in task.tile_data[t]["noise"].items()},
                "hr": task.tile_data[t]["hr"].to(device)}
            for t in task.tiles}
    cand = forward_tiles(model, data, task.tiles, geom).float()
    with torch.no_grad():
        base = forward_tiles(frozen_model, data, task.tiles, geom).float()
    hr = task.hr.to(device)
    sets = task.sets.to(device)
    pos, vel = tile_particles(cand, task.tiles, **kw)
    gather, diag = member_gather_loss(pos, vel, sets, mcfg, set_indices,
                                      term_scale=term_scale)
    host = None
    if task.host_sets is not None and task.host_sets.n_sets > 0:
        hsets = task.host_sets.to(device)
        # Only selection and centre_mode distinguish the host config, and both
        # are already baked into `hsets` at build time; the estimator and the
        # per-term weights are mcfg's by design (same physics binds a host as a
        # subhalo, one `--w-host-sets` scales the whole term). centre_mode="self"
        # is required, not cosmetic: host sets carry an all-zero `centre_rhat`
        # (no clustercentric direction of their own), so "radial" would raise and
        # the self-anchor is the mode whose reference these sets were built for.
        # term_scale is dropped: it is measured on the subhalo terms and does not
        # describe the host ones.
        host = member_gather_loss(
            pos, vel, hsets,
            dataclasses.replace(mcfg, centre_mode="self"), host_set_indices)
    return cand, base, hr, gather, diag, host


def host_loss(model, task, geom, kw, mcfg, args, device, frozen_model,
              set_indices=None, term_scale=None, critic=None, critic_norm=None,
              w_adv=0.0, host_set_indices=None):
    """Total loss for one host, plus the row that gets logged.

    ``critic`` is the optional HR critic (:mod:`cosmo_sr.features.gather_critic`).
    When it is supplied the generator's adversarial term ``-D(high-pass(cand))``
    is added at weight ``w_adv`` (0 during warmup, so the critic still trains but
    the generator feels only the gather gradient), and the detached real/fake
    critic inputs are handed back under ``row["_adv"]`` for the critic update --
    the step loop pops that key before the row is logged. With ``critic=None`` the
    call is byte-identical to the four finished arms.
    """
    cand, base, hr, gather, diag, host = host_forward(
        model, task, geom, kw, mcfg, device, frozen_model, set_indices,
        term_scale=term_scale, host_set_indices=host_set_indices)
    g = guards(cand, base, geom)
    # The REPORTED scalar is always the original one, whatever is being
    # penalised: `highk_ratio` is the number every run of this line has quoted
    # and the number `--highk-max` gates on, and silently redefining it would
    # make this run incomparable with the four already on disk.
    hk_pen, hk_ratio = highk_hinge(cand, hr, dx=DX, k_split=float(args.k_split))
    bands = None
    if int(args.highk_bins) > 0:
        # Per-octave instead of one unweighted mean over modes. On the 64^3
        # tile, `k >= 4` admits 99.2% of modes and only 5.7% of them lie at the
        # 4-8 h/Mpc subhalo scale, so the scalar is ~94% an opinion about
        # grid-scale power -- see field_guards' banded section for the counts.
        hk_pen, band_ratio, band_k = banded_highk_hinge(
            cand, hr, dx=DX, k_split=float(args.k_split),
            n_bins=int(args.highk_bins),
            k_max=(float(args.highk_k_max) if args.highk_k_max else None),
            tol=float(args.highk_tol), two_sided=bool(args.highk_two_sided),
            reduce=str(args.highk_reduce))
        bands = {"k": [float(x) for x in band_k],
                 "ratio": [float(x) for x in band_ratio.detach()]}
    # --- velocity small-scale power ----------------------------------------
    # ALWAYS measured, whatever is being penalised. This is the number the four
    # finished arms had no record of, and it is the largest single defect in
    # them: the FROZEN field carries 1.02x HR of velocity power above k_split --
    # i.e. SR2 has this right -- and every arm ended between 0.034x and 0.053x,
    # a 19-30x collapse that the tune caused. `vel_rms_ratio` could not see it:
    # that is one global std over the whole tile, and it read 0.71.
    vk_ratio = highk_power_ratio_torch(cand, hr, dx=DX,
                                       k_split=float(args.k_split),
                                       channels=slice(3, 6))
    vk_pen = torch.zeros((), device=cand.device)
    if float(args.w_vel_highk) > 0:
        # TWO-SIDED, unlike the displacement guard, because the failure here is
        # a collapse and a one-sided-above hinge is exactly zero on all of it.
        vk_pen, _, _ = banded_highk_hinge(
            cand, hr, dx=DX, k_split=float(args.k_split),
            n_bins=max(int(args.highk_bins), 1),
            k_max=(float(args.highk_k_max) if args.highk_k_max else None),
            channels=slice(3, 6), tol=float(args.vel_highk_tol),
            two_sided=True, reduce=str(args.highk_reduce))
    # --- host preservation term (optional; --w-host-sets) ------------------
    # The SAME member-gather loss on the resolved HOST halos homing in these
    # tiles, at its own weight. Hinged against the frozen field's own value --
    # which already resolves these hosts -- so it starts at ~0 and rises only as
    # a host comes apart, charging the run for the collateral damage measured in
    # `gather-holdout-rockstar-gate` (hosts 3028 base -> 2708 tuned) without
    # competing with the subhalo objective while hosts stay intact.
    host_gather = None
    host_diag = None
    if host is not None:
        host_gather, host_diag = host
    loss = (gather
            + float(args.w_low) * g["low"]
            + float(args.w_highk) * hk_pen
            + float(args.w_vel_highk) * vk_pen)
    if host_gather is not None:
        loss = loss + float(args.w_host_sets) * host_gather
    row = {
        "gather": float(gather.detach()),
        "low_k": float(g["low"].detach()),
        "highk_ratio": float(hk_ratio.detach()),
        "highk_pen": float(hk_pen.detach()),
        # The shape of the excess, not just its size. `all_blocks_self` ran to
        # 1.70x held out with no record of WHICH scales carried it, so the run
        # could not say whether it had built substructure or rung the grid.
        "highk_bands": bands,
        "velhighk_ratio": float(vk_ratio.detach()),
        "velhighk_pen": float(vk_pen.detach()),
        "vel_rms_ratio": float(g["vel_rms_ratio"].detach()),
        "bound_hard": diag["median_bound_hard"],
        "virial": diag["median_virial"],
        "centre_offset_radii": diag["median_centre_offset_radii"],
        "r_rms_over_hr": diag["median_r_rms_over_hr"],
        "sigma_v_over_hr": diag["median_sigma_v_over_hr"],
        "n_sets": int(diag["n_sets"]),
        "n_sets_total": int(task.sets.n_sets),
    }
    # The host preservation guard, reported whether or not it is on so the frozen
    # start and every step carry the same keys. `host_gather` is the weighted
    # scalar added to the loss; the medians say whether hosts are staying bound.
    if host_diag is not None:
        row["host_gather"] = float(args.w_host_sets) * float(host_gather.detach())
        row["host_bound_hard"] = host_diag["median_bound_hard"]
        row["host_virial"] = host_diag["median_virial"]
        row["host_r_rms_over_hr"] = host_diag["median_r_rms_over_hr"]
        row["host_centre_offset_radii"] = host_diag["median_centre_offset_radii"]
        row["host_n_sets"] = int(host_diag["n_sets"])
        row["host_n_sets_total"] = int(
            task.host_sets.n_sets if task.host_sets is not None else 0)
    # Where the loss budget actually goes. `member_gather_loss` already computes
    # every term; dropping them here left the run with no way to see that the
    # centre term was ~80% of `gather` and carrying the whole train/holdout gap
    # (2026-08-23). Logged WEIGHTED, so `sum(term_*) == gather` is a standing
    # self-check on the decomposition rather than a number needing the weights.
    for k, w in _TERM_WEIGHTS(mcfg).items():
        row[f"term_{k}"] = float(w) * float(diag[f"term_{k}"])
        # RAW and unweighted: the quantity --term-norm's scales are measured
        # from, and the only one comparable across runs with different weights.
        row[f"term_raw_{k}"] = float(diag[f"term_{k}"])
        # What actually entered the loss. Equal to `term_{k}` when --term-norm
        # is off; when it is on, THIS is the row that sums to `gather`.
        row[f"term_eff_{k}"] = float(diag[f"term_eff_{k}"])

    # --- adversarial term (optional; docs/sr2_gather_critic.md) --------------
    # The critic sees the six-channel HIGH-PASS of cand vs hr, so velocity and
    # small-scale placement -- the two defects the moment loss leaves free (gate
    # section 11.6) -- are both in its view. Built under no_grad while w_adv==0
    # (warmup) so the generator step is pure gather; with grad once w_adv ramps.
    if critic is not None:
        sf = geom.scale_factor
        if w_adv > 0:
            fake_ci = gather_critic_input(cand, sf, normalizer=critic_norm)
            g_adv = hinge_g_loss(critic(fake_ci))
            loss = loss + float(w_adv) * g_adv
            row["loss_G_adv"] = float(g_adv.detach())
            fake_det = fake_ci.detach()
        else:
            with torch.no_grad():
                fake_ci = gather_critic_input(cand, sf, normalizer=critic_norm)
                row["loss_G_adv"] = float(hinge_g_loss(critic(fake_ci)))
                fake_det = fake_ci
        with torch.no_grad():
            real_ci = gather_critic_input(hr, sf, normalizer=critic_norm)
        # Handed to the critic update; popped by the step loop before logging.
        row["_adv"] = (fake_det, real_ci)
    return loss, row


def _TERM_WEIGHTS(mcfg) -> Dict[str, float]:
    """The six ``member_gather_loss`` weights, keyed as its ``term_*`` diag."""
    return {"virial": mcfg.w_virial, "bound": mcfg.w_bound, "d6": mcfg.w_d6,
            "rrms": mcfg.w_rrms, "sigmav": mcfg.w_sigmav,
            "centre": mcfg.w_centre}


_TERM_NAMES = ("virial", "bound", "d6", "rrms", "sigmav", "centre")


def adv_weight_at(step: int, args) -> float:
    """The generator's adversarial weight at ``step``: 0 during warmup, then a
    linear ramp to ``--w-adv`` over ``--adv-ramp-steps``.

    Ramped, never switched on at full strength, for the same reason the DMSR
    trainer ramps ``lambda_adv``: the generator is a fixed operator whose
    adversarial gradient is weak next to the gather loss, and a step change would
    let the critic yank the field before it has calibrated. During warmup the
    critic still trains (on the generator's fakes); only the generator's adv term
    is held at zero, so no gather steps are wasted.
    """
    w = float(args.w_adv)
    if w <= 0:
        return 0.0
    warm = int(args.adv_warmup_steps)
    if step <= warm:
        return 0.0
    ramp = max(1, int(args.adv_ramp_steps))
    return w * min(1.0, (step - warm) / ramp)


_POOL_SCALARS = (("n_hosts", "n_sets_total", "gather", "bound_hard", "virial",
                  "centre_offset_radii", "r_rms_over_hr", "sigma_v_over_hr",
                  "low_k", "highk_ratio", "highk_ratio_max", "vel_rms_ratio")
                 + tuple(f"term_{k}" for k in _TERM_NAMES)
                 + tuple(f"term_raw_{k}" for k in _TERM_NAMES)
                 + tuple(f"term_eff_{k}" for k in _TERM_NAMES))


def wandb_row(row: Dict) -> Dict:
    """Flatten one nested eval row into the scalars wandb can chart.

    ``metrics.jsonl`` keeps the full nested row including every per-host entry
    and stays the source of truth; this is a lossy view for plotting only, so a
    wandb outage can never lose a number.

    Per-host values go in as a **histogram** rather than 56 separate series. The
    question they answer is "is one big cluster carrying the pooled median while
    everything else sits still", and a distribution answers that at a glance
    where 56 lines do not.
    """
    flat: Dict[str, object] = {"step": int(row["step"])}
    for k in ("wall_s", "batch_gather", "batch_unsup_pen"):
        if k in row:
            flat[k] = float(row[k])
    # The box-wide guard's own number, per split -- the worst-band ratio is what
    # the arm exists to move and what the verdict gates on.
    for side, block in (row.get("unsup") or {}).items():
        if block and block.get("ratio_max") is not None:
            flat[f"unsup/{side}/highk_ratio_max"] = float(block["ratio_max"])
    for side in ("train", "holdout"):
        pool = row.get(side) or {}
        for k in _POOL_SCALARS:
            if k in pool and pool[k] is not None:
                flat[f"{side}/{k}"] = pool[k]
        per_host = pool.get("per_host") or []
        if per_host:
            try:
                import wandb
                for k in ("bound_hard", "centre_offset_radii", "highk_ratio"):
                    vals = [h[k] for h in per_host
                            if h.get(k) is not None and np.isfinite(h[k])]
                    if vals:
                        flat[f"{side}/hist_{k}"] = wandb.Histogram(vals)
            except Exception:
                pass
    return flat


def log_wandb(row: Dict, enabled: bool) -> None:
    """Mirror one row. Never raises: the jsonl write has already happened."""
    if not enabled:
        return
    try:
        import wandb
        wandb.log(wandb_row(row), step=int(row["step"]))
    except Exception:
        pass


@torch.no_grad()
def evaluate_pool(model, tasks: Sequence[HostTask], geom, kw, mcfg, args,
                  device, frozen_model, label: str, term_scale=None) -> Dict:
    """Median statistics over a pool. NOT a verdict -- see the module docstring.

    Reported per host as well as pooled: a mean over hosts hides the case this
    line has to watch for, where one big cluster carries the number and every
    other host is untouched.
    """
    rows = []
    # An eval builds a graph it never uses. Before this, one pool eval held a
    # full autograd tape over every host's N^2 pair blocks for nothing -- the
    # single largest avoidable allocation in the job, and pure waste in time too.
    with torch.no_grad():
        for task in tasks:
            _, row = host_loss(model, task, geom, kw, mcfg, args, device,
                               frozen_model, term_scale=term_scale)
            row["key"] = task.key
            row["log_mvir"] = task.sel.log_mvir
            rows.append(row)
    if not rows:
        return {"label": label, "n_hosts": 0, "per_host": []}

    def med(name):
        return float(np.median([r[name] for r in rows]))

    out = {
        "label": label,
        "n_hosts": len(rows),
        "n_sets_total": int(sum(r["n_sets"] for r in rows)),
        # The objective itself, per split. Without it the only loss curve in the
        # run was the train minibatch, so a 3.9x train/holdout divergence on the
        # `gather` median was invisible while the (hinged, log-compressed)
        # diagnostics below still read as converged.
        "gather": med("gather"),
        "bound_hard": med("bound_hard"),
        "virial": med("virial"),
        "centre_offset_radii": med("centre_offset_radii"),
        "r_rms_over_hr": med("r_rms_over_hr"),
        "sigma_v_over_hr": med("sigma_v_over_hr"),
        "low_k": med("low_k"),
        "highk_ratio": med("highk_ratio"),
        "highk_ratio_max": float(np.max([r["highk_ratio"] for r in rows])),
        # Median band ratio across hosts, so the eval row carries the SHAPE of
        # the excess and not only its size. None when --highk-bins is 0.
        "highk_bands": _median_bands(rows),
        "vel_rms_ratio": med("vel_rms_ratio"),
        "velhighk_ratio": med("velhighk_ratio"),
        "velhighk_ratio_min": float(np.min([r["velhighk_ratio"] for r in rows])),
        "per_host": rows,
    }
    for k in _TERM_NAMES:
        out[f"term_{k}"] = med(f"term_{k}")
        out[f"term_raw_{k}"] = med(f"term_raw_{k}")
        out[f"term_eff_{k}"] = med(f"term_eff_{k}")
    return out


def _median_bands(rows: List[Dict]) -> Optional[Dict]:
    """Median per-band high-k ratio over a pool's hosts, or ``None`` if unbanded.

    Kept separate from ``med`` because a band row is a vector and a missing one
    is legitimate: with ``--highk-bins 0`` nothing computes bands and the eval
    row must say so rather than report a length-zero median.
    """
    have = [r["highk_bands"] for r in rows if r.get("highk_bands")]
    if not have:
        return None
    return {"k": have[0]["k"],
            "ratio": np.median([h["ratio"] for h in have], axis=0).tolist(),
            "ratio_max": np.max([h["ratio"] for h in have], axis=0).tolist()}


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    pscfg = phase_space_config_of(cfg)
    mcfg = member_config_of(args)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    banner(f"member-gather fine-tune: rung={args.rung}")

    train_boxes = [b for b in str(args.train_boxes).replace(",", " ").split() if b]
    hold_boxes = [b for b in str(args.holdout_boxes).replace(",", " ").split() if b]
    hold_keys = [k for k in str(args.holdout_hosts).replace(",", " ").split() if k]

    frozen = load_controlled_generator(
        model_path_of(cfg), in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    for p in frozen.parameters():
        p.requires_grad_(False)

    print("building the host pool (one owner-array load per box)", flush=True)
    all_tasks = build_pool(args, cfg, geom, scfg, pscfg, mcfg, frozen, device,
                           train_boxes + hold_boxes)
    if not all_tasks:
        raise SystemExit(
            "no hosts selected. Every box was skipped for want of an owner "
            "array, or --min-log-mvir is above the most massive host. Build the "
            "arrays with scripts/slurm/submit_owner_arrays.sh.")

    split = split_pool([t.sel for t in all_tasks], train_boxes=train_boxes,
                       holdout_boxes=hold_boxes, holdout_keys=hold_keys)
    by_key = {t.key: t for t in all_tasks}
    train_tasks = [by_key[s.key] for s in split.train]
    hold_tasks = [by_key[s.key] for s in split.holdout]

    print(f"\npool: {len(train_tasks)} training hosts, "
          f"{len(hold_tasks)} held out, {len(split.rejected)} rejected")
    for key, why in split.rejected:
        print(f"  REJECTED {key}: {why}")
    tr_sum = summarise_pool([t.sel for t in train_tasks])
    ho_sum = summarise_pool([t.sel for t in hold_tasks])
    print(f"  train:  {tr_sum.get('n_hosts', 0)} hosts over "
          f"{tr_sum.get('n_boxes', 0)} boxes, "
          f"{sum(t.sets.n_sets for t in train_tasks)} supervised sets")
    print(f"  hold:   {ho_sum.get('n_hosts', 0)} hosts over "
          f"{ho_sum.get('n_boxes', 0)} boxes, "
          f"{sum(t.sets.n_sets for t in hold_tasks)} supervised sets", flush=True)
    report_pair_cost(train_tasks + hold_tasks, args)

    out_dir = paths.subdir("member_gather",
                           f"{args.rung}{args.label}", create=True)
    pool_summary = {
        "train": tr_sum, "holdout": ho_sum,
        "rejected": split.rejected,
        "train_hosts": [_host_row(t) for t in train_tasks],
        "holdout_hosts": [_host_row(t) for t in hold_tasks],
    }
    write_json(out_dir / "pool.json", pool_summary)
    print(f"  wrote {out_dir / 'pool.json'}", flush=True)

    if args.pool_only:
        # No wandb run for a shakeout: it trains nothing, and an empty run per
        # shakeout would clutter the group the rung ladder is compared in.
        # The shakeout. Selection and the reachable reference are everything that
        # can be wrong before a step is taken, and a run that supervises three
        # hosts is not worth a GPU allocation.
        return {"ok": True, "pool_only": True, "pool": pool_summary,
                "out_dir": str(out_dir),
                "verdict": {"text": "pool-only shakeout; nothing trained"}}

    if not train_tasks:
        raise SystemExit("no training hosts after the split; nothing to train on.")

    # ---- the unsupervised-tile high-k support -----------------------------
    # Disjoint from every supervised host tile, so the box-wide guard is charged
    # on sites the member-set loss never touches. The training pool DRIVES the
    # loss; the held-out pool only MEASURES, and feeds --unsup-highk-max.
    supervised = {(t.sel.box, tile) for t in all_tasks for tile in t.sel.tiles}
    tile_rng = random.Random(int(args.seed) + 7)
    per_box = int(args.unsup_tiles_per_box)
    unsup_train = build_tile_pool(train_boxes, per_box, cfg, geom, device, args,
                                  supervised, tile_rng)
    unsup_hold = build_tile_pool(hold_boxes, per_box, cfg, geom, device, args,
                                 supervised, tile_rng)

    # ---- wandb ------------------------------------------------------------
    # Grouped by "member_gather" and named by rung, so the rung ladder (fine /
    # middle_fine / all_blocks) lands together and is directly comparable --
    # which is the whole reason the ladder is run as siblings.
    wcfg = dict(cfg.get("wandb", {}) or {})
    if args.no_wandb:
        wcfg["mode"] = "disabled"
    elif args.wandb_mode:
        wcfg["mode"] = args.wandb_mode
    wcfg.setdefault("group", "member_gather")
    wcfg.setdefault("name", f"member_gather-{args.rung}{args.label}")
    cfg["wandb"] = wcfg
    # The pool is part of what a run IS, not a side note: two runs over
    # different hosts are not comparable, so the composition goes into the
    # config where wandb will show it beside the curves.
    cfg["pool"] = {
        "train": pool_summary["train"], "holdout": pool_summary["holdout"],
        "n_train_sets": int(sum(t.sets.n_sets for t in train_tasks)),
        "n_holdout_sets": int(sum(t.sets.n_sets for t in hold_tasks)),
        "train_keys": [t.key for t in train_tasks],
        "holdout_keys": [t.key for t in hold_tasks],
    }
    use_wandb = maybe_init_wandb(cfg, out_dir, job_type="finetune")
    if use_wandb:
        banner(f"wandb: {wcfg.get('project', 'cosmo_sr')} / {wcfg['name']} "
               f"(group {wcfg['group']})")
    else:
        print("wandb: not logging (disabled, unavailable, or no API key); "
              "metrics.jsonl is unaffected and remains the source of truth",
              flush=True)

    model = load_controlled_generator(
        model_path_of(cfg), in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=False)
    lrs = {g: v * args.lr_scale for g, v in
           {"proj_noise": 1e-5, "fine": 3e-6, "middle": 1e-6, "coarse": 3e-7}.items()}
    groups = parameter_groups(model, args.rung, lrs)
    names = trainable_names(model)
    theta0 = snapshot_parameters(model)
    n_train_params = sum(p.numel() for g in groups for p in g["params"])
    n_targets = sum(t.sets.n_sets for t in train_tasks)
    print(f"\nrung {args.rung}: {len(names)} tensors, {n_train_params} of "
          f"{sum(p.numel() for p in model.parameters())} parameters trainable")
    print(f"  {n_train_params / max(n_targets, 1):.0f} parameters per supervised "
          f"set -- pilot_steps_2_4.md section 2 is the reference point: at 4.4 "
          f"parameters per TARGET VALUE the rung could memorise one cluster.",
          flush=True)
    opt = torch.optim.Adam(groups)

    kw = dict(ng_hr=NG_HR, tile_hr=TILE, boxsize_mpc_h=BOXSIZE,
              dis_scale_mpc_h=float(scfg.dis_norm_kpc_h) * 1e-3,
              vel_scale_kms=float(pscfg.vel_norm_km_s))

    # --- the optional HR critic (docs/sr2_gather_critic.md) -----------------
    # Default OFF (--w-adv 0): an unset run is byte-identical to the four arms.
    # The critic sees the six-channel high-pass, and its input normaliser is fit
    # from REAL HR tiles only (the same discipline as the DMSR critic): a handful
    # of the training pool's HR tiles are high-passed and their per-channel RMS
    # measured once, then held fixed and applied identically to real and fake.
    critic = opt_d = lazy_r1 = critic_norm = None
    if float(args.w_adv) > 0:
        n_fit = int(args.critic_norm_fit_tiles)
        fit_tiles = []
        got = 0
        for t in train_tasks:
            fit_tiles.append(highpass_field(t.hr.to(device).float(), geom.scale_factor))
            got += int(t.hr.shape[0])
            if got >= n_fit:
                break
        critic_norm = GatherCriticNorm.fit(fit_tiles).to(device)
        del fit_tiles
        critic = HRCritic(
            in_channels=6, width=int(args.critic_width),
            n_layers=int(args.critic_layers),
            global_pool=bool(args.critic_global_pool)).to(device)
        opt_d = torch.optim.Adam(critic.parameters(), lr=float(args.critic_lr),
                                 betas=(0.0, 0.99))
        lazy_r1 = LazyR1(gamma=float(args.critic_r1_gamma),
                         interval=int(args.critic_r1_interval))
        with open(out_dir / "critic_norm.json", "w") as f:
            json.dump(critic_norm.to_dict(), f, indent=2)
        print(f"\n=== HR critic ON: w_adv {args.w_adv} "
              f"(warmup {args.adv_warmup_steps}, ramp {args.adv_ramp_steps}), "
              f"n_critic {args.n_critic}, R1 gamma {args.critic_r1_gamma}")
        print(f"    sees the 6-channel HIGH-PASS (disp+vel) -- velocity is in the "
              f"input, which is the point; input scales {critic_norm.to_dict()}")
        print("    THIS IS A REGULARISER on the gather loss, not a replacement: "
              "it charges\n    for the field-realism defects the gate found "
              "(velocity collapse, misplaced\n    small-scale power) that no "
              "moment term can see. Feasibility verdict unchanged.", flush=True)

    metrics_p = out_dir / "metrics.jsonl"
    hist: List[Dict] = []

    # --- the budget ---------------------------------------------------------
    # Measured, unweighted, on the FROZEN field over the TRAINING pool, once,
    # and then held fixed: these are constants of the objective, not parameters.
    # Held out is deliberately not used -- the scales would otherwise be a
    # channel from the held-out hosts into the training loss.
    probe = evaluate_pool(model, train_tasks, geom, kw, mcfg, args,
                          device, frozen, "train")
    raw0 = {k: float(probe.get(f"term_raw_{k}", 1.0)) for k in _TERM_NAMES}
    term_scale = None
    if args.term_norm:
        term_scale = {k: v for k, v in raw0.items() if v > 1e-6}
        print("\n=== --term-norm: dividing each term by its frozen-field value")
        print("    so the declared weights ARE the budget. Measured now:")
        w = _TERM_WEIGHTS(mcfg)
        tot = sum(w[k] * raw0[k] for k in _TERM_NAMES) or 1.0
        for k in _TERM_NAMES:
            print(f"      {k:<7} raw {raw0[k]:9.3f}  was "
                  f"{100 * w[k] * raw0[k] / tot:5.1f}% of the budget  "
                  f"-> now {100 * w[k] / (sum(w.values()) or 1):5.1f}%")
        print("    d6's head start is removed by this and not by a separate "
              "knob:\n    it is the same fact -- it starts ~11x above its "
              "reference.", flush=True)
    else:
        w = _TERM_WEIGHTS(mcfg)
        tot = sum(w[k] * raw0[k] for k in _TERM_NAMES) or 1.0
        print("\n=== loss budget at step 0 (weighted share of `gather`)")
        for k in _TERM_NAMES:
            print(f"      {k:<7} {100 * w[k] * raw0[k] / tot:5.1f}%  "
                  f"(raw {raw0[k]:.3f})")
        print("    NOTE: this split is set by the terms' dynamic ranges, not by "
              "the weights.\n    `bound` is the gate's own criterion and "
              "`[1-x/ref]_+^2` caps it at 1.\n    --term-norm and "
              "--bound-penalty log are the two levers.", flush=True)

    def unsup_block() -> Dict:
        """High-k on the unsupervised pools, the box-wide guard's own measure."""
        return {"train": eval_unsup_highk(model, unsup_train, geom, device, args),
                "holdout": eval_unsup_highk(model, unsup_hold, geom, device, args)}

    row0 = {
        "step": 0,
        "term_scale": term_scale or {},
        # With no scales the probe IS the step-0 train row -- same model, same
        # pool, same objective -- so re-running it would be a wasted pool pass.
        "train": (evaluate_pool(model, train_tasks, geom, kw, mcfg, args,
                                device, frozen, "train", term_scale)
                  if term_scale else probe),
        "holdout": evaluate_pool(model, hold_tasks, geom, kw, mcfg, args,
                                 device, frozen, "holdout", term_scale),
        "unsup": unsup_block(),
    }
    hist.append(row0)
    metrics_p.write_text(json.dumps(row0) + "\n")
    log_wandb(row0, use_wandb)
    print(f"\nfrozen start: train bound_frac {row0['train']['bound_hard']:.3f}, "
          f"2T/|W| {row0['train']['virial']:.0f}, "
          f"high-k {row0['train']['highk_ratio']:.2f}x HR", flush=True)

    order = list(range(len(train_tasks)))
    hps = max(1, int(args.hosts_per_step))
    sps = int(args.sets_per_step)
    hsps = int(args.host_sets_per_step)
    set_rng = random.Random(int(args.seed) + 1)
    cursor = len(order)
    t0 = time.time()

    for step in range(1, int(args.steps) + 1):
        w_adv = adv_weight_at(step, args) if critic is not None else 0.0
        opt.zero_grad(set_to_none=True)
        acc: List[Dict] = []
        adv_pairs: List = []
        for _ in range(hps):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            task = train_tasks[order[cursor]]
            cursor += 1
            loss, row = host_loss(model, task, geom, kw, mcfg, args, device,
                                  frozen, sample_sets(task, sps, set_rng),
                                  term_scale=term_scale, critic=critic,
                                  critic_norm=critic_norm, w_adv=w_adv,
                                  host_set_indices=sample_host_sets(
                                      task, hsps, set_rng))
            # Mean over the hosts in the step, so the gradient scale does not
            # depend on --hosts-per-step and the learning rate stays comparable.
            (loss / hps).backward()
            if critic is not None:
                # The detached (fake, real) critic inputs for this host's D
                # update; popped so the row stays JSON-serialisable when logged.
                adv_pairs.append(row.pop("_adv"))
            acc.append(row)
        # The box-wide high-k support. A fresh minibatch of unsupervised tiles
        # each step, backward()'d separately so its (cheap, no N^2) tape is freed
        # before the next step -- Adam accumulates the grads either way, and the
        # clip below sees their sum.
        unsup_pen = 0.0
        if unsup_train and float(args.w_highk_unsup) > 0:
            m = min(int(args.unsup_tiles_per_step), len(unsup_train))
            batch = set_rng.sample(unsup_train, m) if m > 0 else unsup_train
            pen, _, _ = _unsup_bands(model, batch, geom, device, args)
            (float(args.w_highk_unsup) * pen).backward()
            unsup_pen = float(pen.detach())
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], float(args.clip))
        opt.step()

        # ---- critic updates (hinge D loss + lazy R1) --------------------------
        # After the generator step, on the SAME hosts' detached fakes -- one
        # forward reused rather than a fresh generate, which is cheaper and, with
        # a fixed operator, an unbiased sample of what the generator now produces.
        # Any stray critic grads from the generator's adv backward are cleared by
        # opt_d.zero_grad below, so the two optimisers never cross.
        d_stat = None
        if critic is not None and adv_pairs:
            d_losses, r_scores, f_scores, r1_vals = [], [], [], []
            for ci in range(max(1, int(args.n_critic))):
                fake_ci, real_ci = adv_pairs[ci % len(adv_pairs)]
                s_real, s_fake = critic(real_ci), critic(fake_ci)
                loss_d = hinge_d_loss(s_real, s_fake)
                pen, r1m = lazy_r1(critic, real_ci)
                if pen is not None:
                    loss_d = loss_d + pen
                    r1_vals.append(r1m["loss_R1"])
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                d_losses.append(float(loss_d.detach()))
                r_scores.append(float(s_real.detach().mean()))
                f_scores.append(float(s_fake.detach().mean()))
            g_adv = [r["loss_G_adv"] for r in acc if "loss_G_adv" in r]
            d_stat = {
                "w_adv": float(w_adv),
                "loss_D": float(np.mean(d_losses)),
                "loss_G_adv": float(np.mean(g_adv)) if g_adv else 0.0,
                "critic_real_score": float(np.mean(r_scores)),
                "critic_fake_score": float(np.mean(f_scores)),
                "loss_R1": float(np.mean(r1_vals)) if r1_vals else None,
            }

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            row = {
                "step": step,
                "wall_s": time.time() - t0,
                "batch_gather": float(np.mean([r["gather"] for r in acc])),
                "batch_unsup_pen": unsup_pen,
                "train": evaluate_pool(model, train_tasks, geom, kw, mcfg, args,
                                       device, frozen, "train", term_scale),
                "holdout": evaluate_pool(model, hold_tasks, geom, kw, mcfg, args,
                                         device, frozen, "holdout", term_scale),
                "unsup": unsup_block(),
            }
            if d_stat is not None:
                row["adv"] = d_stat
            hist.append(row)
            with metrics_p.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            log_wandb(row, use_wandb)

            # Periodic checkpoint. The authoritative tuned.pt/critic.pt below are
            # written only after the loop, so a walltime or OOM kill near the end
            # left nothing on disk -- the unsup/selfbound/self_critic arms died at
            # steps 5750-6750 of 8000 and had to be re-run from scratch. Mirror
            # the final layout every eval so the gate can always load the latest
            # model; the post-loop write still adds param_drift and the summary.
            if step != int(args.steps):
                torch.save({
                    "model": {n: p.detach().cpu()
                              for n, p in model.state_dict().items()},
                    "rung": args.rung,
                    "trained_names": names,
                    "step": int(step),
                    "partial": True,
                }, out_dir / "tuned.pt")
                if critic is not None:
                    torch.save({"critic": critic.state_dict(),
                                "norm": critic_norm.to_dict(),
                                "step": int(step)}, out_dir / "critic.pt")

            tr, ho = row["train"], row["holdout"]
            un = row.get("unsup") or {}
            ut, uh = un.get("train"), un.get("holdout")
            us = ("" if not (ut or uh) else
                  f" | unsup high-k "
                  f"{(ut or {}).get('ratio_max', float('nan')):.2f}x tr / "
                  f"{(uh or {}).get('ratio_max', float('nan')):.2f}x ho")
            av = ("" if d_stat is None else
                  f" | D {d_stat['loss_D']:.3f} G_adv {d_stat['loss_G_adv']:.3f} "
                  f"(w {d_stat['w_adv']:.3g}, real {d_stat['critic_real_score']:.2f} "
                  f"fake {d_stat['critic_fake_score']:.2f})")
            print(f"  step {step:6d}  loss {row['batch_gather']:.4f}  "
                  f"| train bound {tr['bound_hard']:.3f} "
                  f"2T/|W| {tr['virial']:.1f} dx {tr['centre_offset_radii']:.2f}r "
                  f"| hold bound {ho.get('bound_hard', float('nan')):.3f} "
                  f"dx {ho.get('centre_offset_radii', float('nan')):.2f}r "
                  f"| low_k {tr['low_k']:.4f} high-k {tr['highk_ratio']:.2f}x "
                  f"(max {tr['highk_ratio_max']:.2f}){us}{av}  [{row['wall_s']:.0f}s]",
                  flush=True)

    # Every parameter outside the rung must be bit-identical: a rung that
    # silently moved more than it claims is not the rung that was reported.
    drift = assert_only_trainable_changed(model, theta0, names)

    # Saved as {"model": <FULL state dict>} -- the layout
    # cosmo_sr.tts.srs_noise.load_controlled_generator reads with strict=True.
    # Writing only the rung's tensors would make the checkpoint unloadable by the
    # gate, which regenerates whole boxes through that same loader; the rung
    # metadata rides alongside so what moved is still recoverable from the file.
    ckpt = out_dir / "tuned.pt"
    torch.save({
        "model": {n: p.detach().cpu() for n, p in model.state_dict().items()},
        "rung": args.rung,
        "trained_names": names,
        "step": int(args.steps),
        "param_drift_l2": drift,
    }, ckpt)
    # The critic is not needed by the gate (it regenerates through the generator
    # alone) but is saved for resumability and post-hoc inspection of what it
    # learned to key on. Kept in a separate file so the gate's loader never sees
    # a tensor it does not expect.
    if critic is not None:
        torch.save({"critic": critic.state_dict(),
                    "norm": critic_norm.to_dict(),
                    "step": int(args.steps)}, out_dir / "critic.pt")

    last = hist[-1]
    summary = {
        "ok": True,
        "config": vars(args),
        "pool": pool_summary,
        "rung": args.rung,
        "trainable_parameters": int(n_train_params),
        "param_drift_l2": drift,
        "history": hist,
        "checkpoint": str(ckpt),
        "out_dir": str(out_dir),
        "verdict": verdict(row0, last, args),
    }
    write_json(out_dir / "summary.json", summary)
    log_result_wandb(summary, hist, train_tasks, hold_tasks, use_wandb)
    finish_wandb()
    print(f"\nVERDICT: {summary['verdict']['text']}")
    print(f"  checkpoint -> {ckpt}")
    print(f"  summary    -> {out_dir / 'summary.json'}")
    return summary


def log_result_wandb(summary: Dict, hist: List[Dict], train_tasks, hold_tasks,
                     enabled: bool) -> None:
    """Put the RESULT in wandb's summary, not only the curves.

    A run whose charts must be read to find out what happened is a run whose
    outcome gets misremembered. The verdict string, its three booleans and the
    final pooled numbers go into ``wandb.summary`` so the outcome is visible on
    the runs table itself -- which is where the rung ladder gets compared.

    The per-host final state goes in as a Table for the same reason
    ``evaluate_pool`` reports per host: the pooled median hides the case where
    one big cluster carries the number. Never raises -- summary.json has
    already been written and is the record.
    """
    if not enabled:
        return
    try:
        import wandb
        if wandb.run is None:
            return
        v = summary["verdict"]
        first, last = hist[0], hist[-1]
        wandb.run.summary.update({
            "verdict": v["text"],
            "verdict/guards_held": bool(v["guards_held"]),
            "verdict/moved": bool(v["moved"]),
            "verdict/generalised_by_surrogate": bool(v["generalised_by_surrogate"]),
            "result/steps": int(summary["config"].get("steps", 0)),
            "result/rung": str(summary["rung"]),
            "result/trainable_parameters": int(summary["trainable_parameters"]),
            "result/checkpoint": str(summary["checkpoint"]),
            "result/n_train_hosts": len(train_tasks),
            "result/n_holdout_hosts": len(hold_tasks),
            "result/n_train_sets": int(sum(t.sets.n_sets for t in train_tasks)),
            "result/n_holdout_sets": int(sum(t.sets.n_sets for t in hold_tasks)),
        })
        # Frozen start -> final, for every pooled scalar, both pools. The DELTA
        # is the readable quantity; an absolute bound_frac means nothing without
        # the frozen value it started from.
        for side in ("train", "holdout"):
            a, b = first.get(side) or {}, last.get(side) or {}
            for k in _POOL_SCALARS:
                if k in b and b[k] is not None:
                    wandb.run.summary[f"final/{side}/{k}"] = b[k]
                if k in a and a[k] is not None:
                    wandb.run.summary[f"frozen/{side}/{k}"] = a[k]
                    if k in b and b[k] is not None:
                        try:
                            wandb.run.summary[f"delta/{side}/{k}"] = float(b[k]) - float(a[k])
                        except (TypeError, ValueError):
                            pass

        cols = ["key", "side", "log_mvir", "n_sets", "bound_hard_frozen",
                "bound_hard_final", "bound_hard_delta", "centre_offset_radii",
                "highk_ratio", "low_k"]
        rows = []
        for side in ("train", "holdout"):
            f0 = {h["key"]: h for h in (first.get(side) or {}).get("per_host", [])}
            for h in (last.get(side) or {}).get("per_host", []):
                z = f0.get(h["key"], {})
                b0 = z.get("bound_hard")
                rows.append([
                    h["key"], side, h.get("log_mvir"), h.get("n_sets"),
                    b0, h.get("bound_hard"),
                    (h["bound_hard"] - b0) if (b0 is not None
                                               and h.get("bound_hard") is not None)
                    else None,
                    h.get("centre_offset_radii"), h.get("highk_ratio"),
                    h.get("low_k"),
                ])
        if rows:
            wandb.log({"result/per_host": wandb.Table(columns=cols, data=rows)})
    except Exception as e:  # pragma: no cover - never fail a finished run
        print(f"[wandb] result logging failed ({e}); summary.json is unaffected")


def verdict(first: Dict, last: Dict, args) -> Dict:
    """Feasibility only, and it says so.

    Deliberately NOT a success criterion. Every number it reads is a member-set
    statistic this module computes about itself, and the line's own history is
    that such a statistic can be driven arbitrarily far while a halo finder shows
    nothing (``tile-overfit-proxy-exploitation``: proxy +255, real gain zero).
    The only question answered here is whether the run is worth gating.
    """
    tr0, tr1 = first["train"], last["train"]
    ho1 = last["holdout"]
    # The high-k guard is OPTIMISED on the train pool, so a train-only verdict
    # asks whether the optimiser reached its own constraint -- not whether the
    # field survived. All four finished arms passed on train; `self` was at
    # 3.87x HR held out at the same step and `verdict` never read that row.
    ho_max = float(args.highk_max_holdout)
    hold_highk_ok = (ho_max <= 0 or ho1.get("n_hosts", 0) == 0
                     or ho1.get("highk_ratio_max", 0.0) <= ho_max)
    vk_min = float(args.vel_highk_min)
    vel_highk_ok = (vk_min <= 0
                    or tr1.get("velhighk_ratio_min", 1.0) >= vk_min)
    # The box-wide guard's OWN generalisation test: worst band on the held-out
    # unsupervised tiles. This is the number the whole arm exists to move -- the
    # held-out HOST tiles are still a sliver, so `--highk-max-holdout` above can
    # pass while the box at large is corrupt.
    un_max = float(getattr(args, "unsup_highk_max", 0.0))
    un_ho = (last.get("unsup") or {}).get("holdout") or {}
    unsup_highk_ok = (un_max <= 0 or not un_ho
                      or un_ho.get("ratio_max", 0.0) <= un_max)
    guards_held = bool(
        tr1["low_k"] <= float(args.low_k_max)
        and tr1["highk_ratio_max"] <= float(args.highk_max)
        and hold_highk_ok
        and vel_highk_ok
        and unsup_highk_ok
        and abs(tr1["vel_rms_ratio"] - 1.0) <= float(args.vel_rms_tol))
    moved = bool(tr1["bound_hard"] > tr0["bound_hard"] + 0.05)
    generalised = bool(
        ho1.get("n_hosts", 0) > 0
        and ho1["bound_hard"] > first["holdout"]["bound_hard"] + 0.05)

    if not guards_held:
        text = (f"GUARD FAILED (low_k {tr1['low_k']:.4f}, high-k max "
                f"{tr1['highk_ratio_max']:.2f}x HR train / "
                f"{ho1.get('highk_ratio_max', float('nan')):.2f}x held out, "
                f"unsup high-k max {un_ho.get('ratio_max', float('nan')):.2f}x "
                f"held out, vel rms {tr1['vel_rms_ratio']:.3f}, vel high-k min "
                f"{tr1.get('velhighk_ratio_min', float('nan')):.3f}x HR). The "
                f"field was damaged; nothing about the objective is readable "
                f"from this run.")
    elif not moved:
        text = (f"NOT MOVED: train bound_frac {tr0['bound_hard']:.3f} -> "
                f"{tr1['bound_hard']:.3f}. The operator did not reach the "
                f"objective on the hosts it was supervised on. Gating is "
                f"premature.")
    elif not generalised:
        text = (f"IN-SAMPLE ONLY: train bound_frac {tr0['bound_hard']:.3f} -> "
                f"{tr1['bound_hard']:.3f} but held-out "
                f"{first['holdout'].get('bound_hard', float('nan')):.3f} -> "
                f"{ho1.get('bound_hard', float('nan')):.3f}. Gate the training "
                f"hosts to confirm, but the generalisation question is answered "
                f"NO by this run's own statistics.")
    else:
        text = (f"FEASIBLE ON HELD-OUT HOSTS: bound_frac train "
                f"{tr0['bound_hard']:.3f} -> {tr1['bound_hard']:.3f}, held-out "
                f"{first['holdout']['bound_hard']:.3f} -> "
                f"{ho1['bound_hard']:.3f}, guards held. This is a surrogate and "
                f"is expected to be gameable -- RUN THE WHOLE-BOX ROCKSTAR GATE.")
    return {"text": text, "guards_held": guards_held, "moved": moved,
            "generalised_by_surrogate": generalised}


def build_parser() -> argparse.ArgumentParser:
    """The trainer's parser, separately constructible.

    ``export_gather_tiles.py`` replays a finished run's ``summary.json`` config
    through this parser to rebuild the pool exactly as the run built it. A second
    copy of these defaults there would mean the gate scores a pool assembled
    under different purity, softening or background settings than the one the
    run was evaluated on -- a difference that is invisible in every statistic.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_direct_args(ap)
    # --- the pool ------------------------------------------------------------
    ap.add_argument("--train-boxes", default="set3 set4 set5 set6 set7",
                    help="boxes to fine-tune on. set0-2 are SR2's own paired "
                         "training boxes; set8 is where this line was developed")
    ap.add_argument("--holdout-boxes", default="set9 set10",
                    help="never trained on, never used to develop the line")
    ap.add_argument("--holdout-hosts", default="",
                    help="'box:h<id>' keys carved out of a TRAINING box; the "
                         "weaker same-realisation held-out axis")
    ap.add_argument("--n-tiles", type=int, default=4,
                    help="tiles per host. 4 is the oracle's, whose ceiling is "
                         "151/154; widening buys ~3 targets and halves the pool")
    ap.add_argument("--max-hosts-per-box", type=int, default=8)
    ap.add_argument("--min-log-mvir", type=float, default=13.5)
    ap.add_argument("--pool-only", action="store_true",
                    help="the shakeout: select, report and stop")
    # --- the objective (mirrors free_field_gather.py) ------------------------
    ap.add_argument("--w-virial", type=float, default=1.0)
    ap.add_argument("--w-bound", type=float, default=1.0)
    ap.add_argument("--w-d6", type=float, default=1.0)
    ap.add_argument("--w-rrms", type=float, default=0.3)
    ap.add_argument("--w-sigmav", type=float, default=0.3)
    ap.add_argument("--w-centre", type=float, default=1.0)
    # --- host preservation guard (0 => byte-identical to the finished arms) ---
    ap.add_argument(
        "--w-host-sets", type=float, default=0.0,
        help="weight on the member-gather loss applied to the resolved HOST "
             "halos homing in a cluster's tiles. 0 (default) is OFF and every "
             "run recorded before 2026-08-25 is byte-identical. On, it is a "
             "PRESERVATION guard: the same virial/bound/d6/centre terms charge "
             "the run for fragmenting hosts it is not being asked to build "
             "(measured: resolved hosts 3028 base -> 2708 tuned, HR wants 3775; "
             "docs/sr2_member_gather_training.md). The reference is the frozen "
             "field's own value, which already resolves these hosts, so the "
             "hinged terms start at ~0 and fire only on damage -- they do not "
             "compete with the subhalo objective while hosts stay intact. Uses "
             "the same per-term weights as the subhalo loss, scaled by this.")
    ap.add_argument(
        "--host-min-num-p", type=int, default=200,
        help="resolution floor for a HOST preservation set, independent of "
             "--min-num-p. A host below this is not a reliably resolved object "
             "to preserve.")
    ap.add_argument(
        "--host-max-sets", type=int, default=256,
        help="cap on host preservation sets per cluster, the host analogue of "
             "--max-sets. A cluster's tiles can home many hosts; the cap keeps "
             "the largest, which are the ones a halo finder actually resolves.")
    ap.add_argument(
        "--host-sets-per-step", type=int, default=64,
        help="host preservation sets drawn per step, the analogue of "
             "--sets-per-step. 0 charges all of them (a full-batch host guard).")
    ap.add_argument("--bound-penalty", choices=("hinge", "log"), default="hinge",
                    help="how a boundness DEFICIT is charged. hinge is "
                         "[1-x/ref]_+^2, the form every run so far used, which "
                         "is CAPPED AT 1 because x >= 0 -- measured, it held "
                         "0.24%% of the step-0 budget while d6 held 88%%, and "
                         "it is the term carrying Rockstar's own decision "
                         "rule. log is [log(ref/x)]_+^2: still exactly zero "
                         "for meeting or beating HR, unbounded as x -> 0, and "
                         "the same scale-free form virial/r_rms/sigma_v use.")
    ap.add_argument("--term-norm", action="store_true",
                    help="divide each term by its value on the FROZEN field "
                         "over the training pool, measured once at step 0 and "
                         "held fixed, so the declared weights are the actual "
                         "budget. Also removes d6's head start, which is the "
                         "same fact: it starts ~11x above its reference and "
                         "collapses 32x in the first 250 steps -- the steps "
                         "`bound` most needs.")
    ap.add_argument("--centre-mode", choices=("full", "radial", "self"),
                    default="full",
                    help="what the centre term charges for. full: the whole "
                         "offset to the reachable HR centroid. radial: only its "
                         "clustercentric projection -- the 62.9%% of the offset "
                         "variance that is a signed infall deficit, dropping "
                         "the transverse part LR does not determine. self: "
                         "re-anchor to the set's own frozen centroid.")
    ap.add_argument("--centre-dead-zone", type=float, default=0.0,
                    help="search radii inside which the centre term costs "
                         "nothing. The gate is a threshold at 1 radius, so cost "
                         "paid at 0.1 buys nothing there. 0 = the 72/154 term.")
    ap.add_argument("--centre-huber-radii", type=float, default=0.0,
                    help="search radii beyond which the centre term is linear. "
                         "Frozen sets sit a median 5.6 radii out, where a "
                         "quadratic lets the hopeless sets own the gradient.")
    ap.add_argument("--min-num-p", type=int, default=200)
    ap.add_argument("--min-purity", type=float, default=0.5)
    ap.add_argument("--min-live-frac", type=float, default=0.5)
    ap.add_argument("--max-sets", type=int, default=256)
    ap.add_argument("--softening", type=float, default=0.01)
    ap.add_argument("--softening-kind", default="plummer",
                    choices=("plummer", "spline"))
    ap.add_argument("--pot-chunk", type=int, default=2048)
    ap.add_argument("--pot-max-elems", type=int, default=1 << 24,
                    help="cap on one pair block, in elements. Rows per pass are "
                         "min(--pot-chunk, this // N), which is what keeps a "
                         "118k-particle set from a 2.7 GiB allocation.")
    ap.add_argument("--max-set-particles", type=int, default=0,
                    help="subsample each set to at most this many particles. 0 "
                         "(default) is the exact estimator. On, the pair sum is "
                         "rescaled by (N-1)/(K-1) and 2T/|W| stays unbiased; it "
                         "is a TIME knob, not a memory one.")
    ap.add_argument("--bound-tau", type=float, default=0.5)
    ap.add_argument("--bound-temperature", default="adaptive",
                    choices=("adaptive", "fixed"))
    ap.add_argument("--bg-k", type=int, default=4096)
    ap.add_argument("--bg-radius", type=float, default=4.0)
    # --- the guards ----------------------------------------------------------
    ap.add_argument("--w-low", type=float, default=100.0)
    ap.add_argument("--w-highk", type=float, default=10.0,
                    help="the term docs/sr2_member_gather.md section 7 item 2 "
                         "asks for. A hinge, not an anchor: an L2 anchor was "
                         "measured moving peak contrast the WRONG way")
    ap.add_argument("--k-split", type=float, default=4.0)
    # --- band-resolved high-k. Default OFF: the four arms on disk were trained
    # against the scalar, and changing the objective by default would make this
    # run incomparable with them rather than a controlled follow-up.
    ap.add_argument("--highk-bins", type=int, default=0,
                    help="0 (default) penalises the single mode-count-weighted "
                         "ratio, as the four finished arms did. >0 splits "
                         "k >= --k-split into that many log bins and charges "
                         "each equally, so a 3x excess at the 4-8 h/Mpc "
                         "subhalo scale costs what a 3x excess at Nyquist "
                         "costs. It does not on the scalar: that band is 5.7% "
                         "of the modes and the mean is unweighted")
    ap.add_argument("--highk-k-max", type=float, default=0.0,
                    help="upper k for the bands; 0 (default) runs to the cube "
                         "corner, matching the scalar's mask. pi/dx = 16.08 "
                         "drops the 48% of the guard's modes that sit ABOVE "
                         "Nyquist, where a shell is only the cube's corners")
    ap.add_argument("--highk-tol", type=float, default=0.0,
                    help="dead zone, as a factor either side of HR. The "
                         "one-sided hinge has none, so `all_blocks_self` "
                         "finished with its worst TRAIN host at 0.887 -- the "
                         "term contributing exactly zero gradient -- while its "
                         "worst held-out host stood at 3.87")
    ap.add_argument("--highk-two-sided", action="store_true",
                    help="also charge a band for falling BELOW HR. Not the "
                         "L2-anchor mistake: that anchor was to the FROZEN "
                         "field and charged for changing anything, while a "
                         "lower bound on HR asks for MORE small-scale power. "
                         "It is what would have charged `all_blocks_nocentre` "
                         "for taking high-k to 0.026")
    ap.add_argument("--highk-reduce", default="mean", choices=("mean", "max"),
                    help="how the per-band penalties combine")
    ap.add_argument("--w-vel-highk", type=float, default=0.0,
                    help="weight on a TWO-SIDED band hinge over VELOCITY power "
                         "above --k-split. 0 (default) only measures it. "
                         "Measured on held-out set9: frozen SR2 is 1.02x HR "
                         "here -- it has this right -- and all four finished "
                         "arms landed at 0.034-0.053x, a 19-30x collapse with "
                         "no term in the loss. A halo finder works in 6-D "
                         "phase space; smoothing the small-scale velocity "
                         "field is how bound substructure stops being "
                         "separable from its host")
    ap.add_argument("--vel-highk-tol", type=float, default=0.25,
                    help="dead zone for the velocity band, either side of HR")
    ap.add_argument("--vel-highk-min", type=float, default=0.0,
                    help="verdict gate on the WORST host's velocity high-k "
                         "ratio. 0 (default) keeps the historical verdict, "
                         "which had no velocity-power criterion at all")
    ap.add_argument("--low-k-max", type=float, default=0.02)
    ap.add_argument("--highk-max", type=float, default=1.5,
                    help="verdict gate on the worst host's high-k ratio")
    ap.add_argument("--highk-max-holdout", type=float, default=0.0,
                    help="same gate, on the HELD-OUT pool. 0 (default) keeps "
                         "the historical train-only verdict. The four finished "
                         "arms all passed a train guard while `self` sat at "
                         "3.87x held out, because `verdict` never read that row")
    ap.add_argument("--vel-rms-tol", type=float, default=0.10)
    # --- the box-wide (unsupervised-tile) high-k support ---------------------
    # The fix `selfvel-arm-failed-the-gate` names: the supervised guard held 0.56x
    # on train tiles yet 3.9x held out because its support is 6.25% of the box.
    # These charge the SAME banded hinge (shape from --highk-bins/-k-max/-tol/
    # -reduce, always one-sided) on random tiles drawn box-wide, against those
    # tiles' own HR power. Amplitude only -- it never touches placement.
    ap.add_argument("--unsup-tiles-per-box", type=int, default=0,
                    help="random unsupervised tiles per box, disjoint from every "
                         "supervised host tile. 0 (default) is off -- the "
                         "historical supervised-only guard. Drawn from the train "
                         "boxes (drive the loss) and the holdout boxes (measure "
                         "and gate). Held on the CPU, ~15 MB/tile")
    ap.add_argument("--unsup-tiles-per-step", type=int, default=8,
                    help="how many of the training unsupervised tiles carry the "
                         "hinge each step, drawn fresh. A generator forward with "
                         "no N^2 potential, so this is cheap")
    ap.add_argument("--w-highk-unsup", type=float, default=0.0,
                    help="weight on the box-wide high-k hinge. 0 (default) only "
                         "builds/measures the pool; >0 adds it to the loss")
    ap.add_argument("--unsup-highk-max", type=float, default=0.0,
                    help="verdict gate on the worst band over the HELD-OUT "
                         "unsupervised tiles -- the box-wide guard's own "
                         "generalisation test. 0 (default) leaves it off")
    # --- the HR critic (docs/sr2_gather_critic.md) ---------------------------
    # Default OFF (--w-adv 0): an unset run is byte-identical to the four arms.
    # The hand-crafted high-k/velocity hinges above charge for ONE named statistic
    # each; the critic is their general form -- it sees whole HR tiles vs tuned
    # tiles (the six-channel high-pass, velocity included) and learns to penalise
    # the field-realism defects the gate found that no moment term can see.
    ap.add_argument("--w-adv", type=float, default=0.0,
                    help="weight on the generator's adversarial term "
                         "-D(high-pass(cand)). 0 (default) turns the critic off "
                         "entirely, reproducing the four finished arms. >0 adds a "
                         "trained HR critic over the 6-channel high-pass (disp AND "
                         "vel) as a REGULARISER on the gather loss -- see gate "
                         "section 11.6: velocity power collapsed 19-30x and "
                         "small-scale placement went to the wrong spots, both "
                         "invisible to the moment loss")
    ap.add_argument("--adv-warmup-steps", type=int, default=500,
                    help="steps before the generator feels the adversarial term. "
                         "The critic still TRAINS during warmup (on the "
                         "generator's fakes), so no gather steps are wasted; only "
                         "w_adv is held at 0 so the critic calibrates first")
    ap.add_argument("--adv-ramp-steps", type=int, default=2000,
                    help="linear ramp of w_adv from 0 to --w-adv after warmup. "
                         "Ramped, never a step change, because the operator's "
                         "adversarial gradient is weak next to the gather loss")
    ap.add_argument("--n-critic", type=int, default=1,
                    help="critic updates per generator step")
    ap.add_argument("--critic-lr", type=float, default=2e-4)
    ap.add_argument("--critic-width", type=int, default=64,
                    help="base width of the PatchGAN; doubles per stride-2 stage")
    ap.add_argument("--critic-layers", type=int, default=3,
                    help="number of stride-2 stages in the PatchGAN")
    ap.add_argument("--critic-global-pool", action="store_true",
                    help="collapse the patch grid to ONE score before the final "
                         "1x1, so the critic judges the whole tile jointly and can "
                         "police tile-scale statistics a patch critic is blind to. "
                         "Off (default) is a local PatchGAN")
    ap.add_argument("--critic-r1-gamma", type=float, default=10.0,
                    help="lazy-R1 gradient penalty strength on real examples")
    ap.add_argument("--critic-r1-interval", type=int, default=16,
                    help="apply R1 every N critic steps, scaled by N (the "
                         "effective strength is unchanged; R1 is a double-backward "
                         "and expensive)")
    ap.add_argument("--critic-norm-fit-tiles", type=int, default=32,
                    help="real HR tiles used to fit the critic's per-channel input "
                         "scales, once, held fixed and applied to real and fake "
                         "alike")
    # --- optimisation --------------------------------------------------------
    ap.add_argument("--rung", default="fine")
    ap.add_argument("--lr-scale", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--sets-per-step", type=int, default=0,
                    help="supervise this many of a host's sets per step, drawn "
                         "fresh. 0 (default) takes all of them, which is a "
                         "full-batch gradient over the expensive axis. Evals "
                         "always use every set.")
    ap.add_argument("--hosts-per-step", type=int, default=2,
                    help="gradient accumulation. One host per step is a very "
                         "noisy estimate of a SHARED operator's gradient")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="")
    ap.add_argument("--no-wandb", action="store_true",
                    help="disable Weights & Biases logging for this run "
                         "(overrides the config's wandb.mode)")
    ap.add_argument("--wandb-mode", default="",
                    help="override wandb.mode (online/offline/disabled)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    s = run(args)
    return 0 if s.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
