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
import glob
import os
import random
import sys
import time
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import banner, paths, write_json  # noqa: E402
from _sr2_direct import (  # noqa: E402
    add_direct_args, geometry_of, load_direct_config, load_hr, model_path_of,
    phase_space_config_of, soft_config_of,
)
# The oracle's own selection, tile builder and forward. A second definition of
# "which tiles are this host's" or "how a tile is generated" would mean this run
# is not working the problem docs/sr2_member_gather.md measured.
from overfit_host_mse import (  # noqa: E402
    BOXSIZE, DX, NG_HR, TILE, _catalog, _owner, build_tiles, cond_config_of,
    forward_tiles, load_cond_field,
)
# The loss configuration and the LR-scale guard, unchanged from the free-field
# run, for the same reason.
from free_field_gather import guards, member_config_of  # noqa: E402

from cosmo_sr.dmsr.critic import HRCritic, LazyR1, hinge_d_loss, hinge_g_loss  # noqa: E402
from cosmo_sr.eval.particle_identity import build_owner_index, child_map  # noqa: E402
from cosmo_sr.eval.rockstar import load_rockstar_ascii  # noqa: E402
from cosmo_sr.features.field_guards import (  # noqa: E402
    banded_highk_hinge, highk_hinge, highk_power_ratio_torch,
)
from cosmo_sr.features.host_temperature import (  # noqa: E402
    TempConfig, target_profile, temperature_loss, temperature_profile,
)
from cosmo_sr.features.knn_cdf import (  # noqa: E402
    OWNER_CLASSES, KnnConfig, box_neighbours, box_phase, box_positions,
    classify_owner, knn_tile_loss, tile_rows_flat,
)
from cosmo_sr.features.gather_critic import (  # noqa: E402
    GatherCriticNorm, gather_critic_input, highpass_field,
)
from cosmo_sr.features.lr_host_channel import (  # noqa: E402
    channel_field_path, load_channel_field,
)
from cosmo_sr.features.host_profile import (  # noqa: E402
    load_lr_host_table, lr_host_for,
)
from cosmo_sr.features.member_gather import (  # noqa: E402
    MemberSets, build_member_sets, host_guard_config, member_gather_loss,
    shell_charge,
    shell_radii, shell_reduce, tile_particles,
)
from cosmo_sr.features.member_pool import (  # noqa: E402
    HostSelection, select_hosts, split_pool, summarise_pool,
)
from cosmo_sr.features.subhalo_gather import (  # noqa: E402
    ENV_LABELS, halo_environment,
)
from cosmo_sr.reward.base import find_base_field  # noqa: E402
from cosmo_sr.train.common import finish_wandb, maybe_init_wandb  # noqa: E402
from cosmo_sr.train.sr2_finetune_data import (  # noqa: E402
    tile_noise_stack, trim_to_tile)
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
    #: One :class:`MemberSets` per NOISE DRAW, index-aligned with
    #: ``tile_data[t]["noise_draws"]``. Entry 0 is ``sets`` itself.
    #:
    #: Why per draw and not one shared table: ``centre_self`` -- the anchor
    #: ``--centre-mode self``/``self_tan`` charges against -- is the FROZEN
    #: field's own centroid, and the frozen field is a function of the noise
    #: realisation. Rotating the noise while holding a draw-0 anchor would
    #: charge the run for ordinary realisation scatter and fight the objective.
    #: The reference statistics are measured on the same hybrid field and
    #: inherit the same dependence. So a draw carries its own table or it is not
    #: an honest draw. Length 1 (the default) is the single-seed behaviour every
    #: arm through 2026-09-08 ran.
    sets_by_draw: Optional[List[MemberSets]] = None
    #: The host-preservation tables, one per draw, aligned with
    #: ``sets_by_draw``. These are referenced against the FROZEN field's own
    #: value, so they are draw-dependent in the strongest possible way -- a
    #: draw-0 reference read under draw 3 would charge the guard for ordinary
    #: realisation scatter and report host damage that never happened.
    host_sets_by_draw: Optional[List[Optional[MemberSets]]] = None
    #: The host's HR centre (Mpc/h) and R_vir (Mpc/h), and HR's host-centric
    #: local-temperature profile on these tiles' own particles -- the target of
    #: ``--w-temp`` (cosmo_sr.features.host_temperature). ``temp_target`` is None
    #: unless the term is on or measured, so every arm before it is unchanged.
    host_pos: Optional[np.ndarray] = None
    host_rvir: float = 0.0
    temp_target: Optional[Dict] = None

    def sets_at(self, draw: int) -> MemberSets:
        """The draw's own set table; ``sets`` when the run has only one draw."""
        if not self.sets_by_draw:
            return self.sets
        return self.sets_by_draw[int(draw) % len(self.sets_by_draw)]

    def host_sets_at(self, draw: int) -> Optional[MemberSets]:
        """The draw's own host-preservation table."""
        if not self.host_sets_by_draw:
            return self.host_sets
        return self.host_sets_by_draw[int(draw) % len(self.host_sets_by_draw)]

    @property
    def n_draws(self) -> int:
        return len(self.sets_by_draw) if self.sets_by_draw else 1

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
    second copy of six weights. Four fields differ:

    * ``min_num_p`` -> ``--host-min-num-p`` (200): the resolution floor for a
      *host*, independent of the subhalo floor.
    * ``max_sets`` -> ``--host-max-sets``: a cluster's tiles can home many more
      hosts than subhalos.
    * ``centre_mode`` -> ``self``: a host has no clustercentric direction of its
      own (it *is* the centre, so ``radial`` is undefined and would raise), and
      preservation means "stay where the frozen field already put you", which is
      exactly what the self-anchored centre term charges for -- zero at step 0,
      rising only if the host's centroid drifts.
    * ``w_radius`` and ``w_shell`` -> 0: a host has no host, so "the right
      distance from the centre" and the distribution of those distances are
      both undefined for it. Same mechanism as ``w_radius`` -> 0: a host has
      no host, so "the right distance from the
      centre" is undefined for it. ``build_host_sets`` passes ``host_pos=None``
      for this population and the loss REFUSES a positive ``w_radius`` on an
      all-zero ``centre_host``, so this is what keeps ``--w-radius`` from
      turning the guard into an error rather than a silent zero.
    """
    return dataclasses.replace(
        mcfg,
        min_num_p=int(args.host_min_num_p),
        # The guard's population is `top` unconditionally, whatever the
        # supervised sets are doing. Under `--population all` the subhalo config
        # already carries a `max_num_p` ceiling that exists to keep the cluster
        # OUT of the supervised pool; inheriting it here would exclude the
        # largest hosts from the preservation guard, which is precisely
        # backwards -- the guard is the only term protecting them.
        population="top",
        max_num_p=0,
        max_sets=int(args.host_max_sets),
        centre_mode="self",
        w_radius=0.0,
        # ...and the same for the soft shell, for the same reason: a host has no
        # host, so the DISTRIBUTION of its distance from one is undefined too.
        w_shell=0.0,
        # ...and `env_select` back to `any`, for the reason in the call
        # site's comment: the guard protects every resolved host in the
        # tiles, not only the ones sharing the objective's environment.
        env_select="any",
    )


#: One LR host table per box, cached for the process. The table is a few
#: hundred rows and is read once per host otherwise -- a multi-host run visits
#: the same box dozens of times.
_LR_TABLES: Dict[str, Dict] = {}


def _lr_host_table(box: str, args) -> Dict:
    """The gated LR host table for ``box``, loaded once.

    Refuses rather than degrades when the table is missing: a silent fallback
    to zeros would run the arm with a flat background for EVERY host and report
    itself as the host-profile arm, which is the one failure mode that would
    not show up in any diagnostic the run writes.
    """
    if box in _LR_TABLES:
        return _LR_TABLES[box]
    root = (Path(args.lr_host_table) if getattr(args, "lr_host_table", "")
            else paths.subdir("lr_host_table", create=False))
    p = Path(root) / f"{box}.npz"
    if not p.exists():
        raise SystemExit(
            f"--soft-host-profile lr needs the LR host table for {box} and "
            f"{p} does not exist. Build it with\n"
            f"  scripts/features/build_lr_host_table.py --boxes {box}\n"
            "(seconds: it parses the LR Rockstar catalog that "
            "build_lagrangian_host.py already left on disk), or point "
            "--lr-host-table at the directory holding it.")
    _LR_TABLES[box] = load_lr_host_table(p)
    return _LR_TABLES[box]


def build_host_sets(
    sel: HostSelection, cat, oidx, cfg, geom, scfg, pscfg, mcfg, args,
    frozen_model, device, env=None,
) -> Optional[HostTask]:
    """Select one host's member sets and measure the reference it can reach.

    ``frozen_field`` is the frozen generator's output on these tiles, not the
    cached full-box field: the two are produced by different generation paths and
    differ slightly, and every reference in the oracle was measured against the
    tile-wise one.

    ``env`` is the box's per-catalog-row environment labels, computed once in
    ``build_pool`` because it costs a KD-tree over the whole catalog. It is
    what ``--env-select`` selects on, and it is recorded per set either way so
    an ``any`` run and a targeted run are readable against each other.
    """
    data = build_tiles(sel.box, sel.tiles, cfg, geom, device, args)
    seeds = draw_seeds(args)
    add_noise_draws(data, sel.tiles, geom, device, seeds)
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
    # `--soft-host-profile lr`: the LR catalog's size for this host, looked up
    # by the same r/R_vir rule the conditioning channel assigns sites with. The
    # HR row supplies only the POSITION to look up -- the profile itself is a
    # function of the LR box alone, which is what keeps the arm deployable.
    # Zeros when no gated LR host owns it: `lr-host-channel-fidelity` measured
    # the LR catalog usable only above ~20k HR particles, and the solve falls
    # back to the flat background for the rest rather than borrowing a
    # neighbouring cluster's profile.
    hr_host_rvir = (float(cat.rvir[int(_hrow[0])]) / 1000.0 if _hrow.size else 0.0)
    host_rvir = host_mvir = 0.0
    if str(getattr(args, "soft_host_profile", "off")) == "lr" and host_pos is not None:
        tbl = _lr_host_table(sel.box, args)
        host_rvir, host_mvir, _u = lr_host_for(
            host_pos, tbl, boxsize_mpc_h=BOXSIZE,
            r_max_rvir=float(getattr(args, "lr_host_r_max_rvir", 2.0)))
    sets = build_member_sets(
        cat, oidx, sel.tiles, hr, base, mcfg,
        particle_mass_msun_h=float(pscfg.particle_mass_msun_h),
        frozen_box=frozen_box, host_pos=host_pos, env=env,
        host_rvir_mpc_h=host_rvir, host_mvir_msun_h=host_mvir,
        report=report, **kw)
    if sets.n_sets == 0:
        return None

    # --- relocation (arms B and C): re-choose the material, keep the targets --
    # A separate pass over a finished pool, for the same reason `build_soft_pools`
    # is one: the pool costs a 537 MB owner-array load and a CSR inversion per
    # box, and this needs neither -- only the frozen particle table and the
    # host's position. So arm B and arm C supervise exactly the objects arm A
    # does, in the same order, and the three gates are directly comparable.
    host_rvir = (float(cat.rvir[int(_hrow[0])]) / 1000.0 if _hrow.size else 0.0)

    def _relocate(st, field, into: Dict):
        if str(mcfg.relocate) == "off":
            return st
        from cosmo_sr.features.relocate_sets import relocate_member_sets
        from cosmo_sr.features.member_gather import tile_particles
        with torch.no_grad():
            fp, fv = tile_particles(field, sel.tiles, **kw)
        return relocate_member_sets(st, fp, fv, mcfg,
                                    host_rvir_mpc_h=host_rvir, report=into)

    if str(mcfg.relocate) != "off":
        if not (host_rvir > 0.0):
            raise SystemExit(
                f"{sel.key}: --relocate {mcfg.relocate} needs the host's R_vir "
                "to normalise the radius constraint, and halo "
                f"{int(sel.halo_id)} was not found in the catalog. Without it "
                "--relocate-eps would mean something different per host.")
        rrep: Dict = {}
        sets = _relocate(sets, base, rrep)
        report["relocate"] = rrep

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
    def _build_host_sets(field, into: Dict):
        if float(getattr(args, "w_host_sets", 0.0)) <= 0.0:
            return None
        try:
            return build_member_sets(
                cat, oidx, sel.tiles, field, field,
                host_member_config(mcfg, args),
                particle_mass_msun_h=float(pscfg.particle_mass_msun_h),
                frozen_box=frozen_box, host_pos=None, top_level=True,
                env=env, report=into, **kw)
        except ValueError:
            return None

    host_sets = None
    if float(getattr(args, "w_host_sets", 0.0)) > 0.0:
        hrep: Dict = {}
        try:
            host_sets = build_member_sets(
                cat, oidx, sel.tiles, base, base, host_member_config(mcfg, args),
                particle_mass_msun_h=float(pscfg.particle_mass_msun_h),
                frozen_box=frozen_box, host_pos=None, top_level=True,
                # The guard's population is every resolved host in the
                # tiles, whatever --env-select restricts the SUPERVISED
                # sets to: a run that supervises only the infall region
                # still damages the hosts it is not being asked to build,
                # and narrowing the guard with the objective would remove
                # the protection exactly where the new work is happening.
                env=env, report=hrep, **kw)
        except ValueError:
            host_sets = None
        report["host_sets"] = hrep

    cpu = torch.device("cpu")
    # --- one set table per extra noise draw (--noise-draws / --fresh-noise-eval)
    # The frozen field is regenerated at each draw and the whole reference is
    # remeasured against it, because `centre_self` and the hybrid reference
    # statistics are functions of that field. Selection is NOT redone: which
    # subhalos this host supervises is a property of the catalog and the tiles,
    # so every draw carries the same objects in the same order and the rows of
    # two draws' tables are directly comparable.
    sets_by_draw = [sets.to(cpu)]
    host_sets_by_draw: List[Optional[MemberSets]] = [
        None if host_sets is None else host_sets.to(cpu)]
    for d in range(1, len(seeds)):
        with torch.no_grad():
            base_d = forward_tiles_draw(frozen_model, data, sel.tiles, geom,
                                        d).float()
        rep_d: Dict = {}
        sets_d = build_member_sets(
            cat, oidx, sel.tiles, hr, base_d, mcfg,
            particle_mass_msun_h=float(pscfg.particle_mass_msun_h),
            frozen_box=frozen_box, host_pos=host_pos, env=env,
            report=rep_d, **kw)
        # Relocated against THIS draw's frozen field, not draw 0's rows: which
        # material is compact is a property of the realisation, and pinning
        # draw 0's choice would supervise a group that is only compact in one
        # of the fields the run averages over. The OBJECTS are unchanged -- the
        # assert below still holds, because relocation never drops a set.
        rrep_d: Dict = {}
        sets_d = _relocate(sets_d, base_d, rrep_d)
        rep_d["relocate"] = rrep_d
        if sets_d.n_sets != sets.n_sets:
            raise SystemExit(
                f"{sel.key}: noise draw {d} selected {sets_d.n_sets} sets "
                f"against draw 0's {sets.n_sets}. Selection must not depend on "
                "the draw -- a rotating table with different objects in it "
                "would make the per-draw losses incomparable.")
        sets_by_draw.append(sets_d.to(cpu))
        hrep_d: Dict = {}
        hs_d = _build_host_sets(base_d, hrep_d)
        host_sets_by_draw.append(None if hs_d is None else hs_d.to(cpu))
        rep_d["host_sets"] = hrep_d
        report[f"draw{d}"] = rep_d
    # --- soft membership: attach the candidate pools ------------------------
    # One build per HOST, shared by every draw. The pool is a function of
    # `live_rows` and the tiling only -- both draw-independent, and asserted
    # above ("selection must not depend on the draw") -- so a second build per
    # draw would produce the same object and merely cost the memory.
    if bool(mcfg.soft_membership):
        from cosmo_sr.features.soft_membership import build_soft_pools
        prep: Dict = {}
        pools = build_soft_pools(
            sets_by_draw[0], sel.tiles, mcfg, ng_hr=int(kw["ng_hr"]),
            tile_hr=int(kw["tile_hr"]), factor=int(args.soft_factor),
            seed=int(args.seed), report=prep)
        for st in sets_by_draw:
            st.pools = pools
        sets.pools = pools
        report["soft_pools"] = prep

    temp_target = None
    if temp_enabled(args) and host_pos is not None and hr_host_rvir > 0:
        hpos, hvel = tile_particles(hr, sel.tiles, **kw)
        temp_target = target_profile(
            hpos.detach(), hvel.detach(),
            torch.as_tensor(host_pos, dtype=hpos.dtype, device=hpos.device),
            hr_host_rvir, temp_config_from_args(args),
            queries_per_bin=int(args.temp_target_queries), seed=int(args.seed))
        report["temp_target_bins"] = int((temp_target["n"] > 0).sum())
    return HostTask(
        host_pos=host_pos, host_rvir=hr_host_rvir, temp_target=temp_target,
        sel=sel, sets=sets.to(cpu), hr=hr.to(cpu), report=report,
        tile_data={t: {"lr": data[t]["lr"].to(cpu),
                       "noise": {s: v.to(cpu) for s, v in data[t]["noise"].items()},
                       "noise_draws": [{s: v.to(cpu) for s, v in nd.items()}
                                       for nd in data[t]["noise_draws"]],
                       "hr": data[t]["hr"].to(cpu),
                       # Absent on an unconditioned run, and every consumer
                       # keys on that absence rather than on a flag: there is
                       # no way to be holding a conditioning stack and not use
                       # it, or to be missing one and think it is there.
                       **({"cond": data[t]["cond"].to(cpu)}
                          if "cond" in data[t] else {})}
                   for t in sel.tiles},
        sets_by_draw=sets_by_draw,
        host_sets_by_draw=host_sets_by_draw,
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


def report_relocation(tasks: Sequence["HostTask"], args) -> Dict:
    """The pre-training check: did the requested groups actually get built?

    Seeds at the correct radii are not enough. The nearest-particle group around
    a correct seed can have a displaced centroid, and a radius with no suitable
    material produces a *starved* set that silently keeps its HR membership --
    an arm-A set inside a B/C run. Both are counted here, on the CPU, before a
    GPU-hour is spent, and ``--relocate-check-only`` stops the job right after.

    Three numbers decide whether the arm is worth running:

    * ``within eps`` -- the fraction whose achieved centroid satisfies the
      radius constraint. Far below 1 means the material is not there and the
      answer is a wider ``--relocate-eps`` or a different radius, not training.
    * ``starved`` -- groups that found no unclaimed material in any direction.
      These are arm A rows in a B/C run and every downstream split must exclude
      them; ``MemberSets.relocated`` is the flag that lets an eval do so.
    * ``kept HR frac`` -- how much of the original membership survived. Near 1
      says the selector re-found HR's own particles and the arm is arm A under
      another name; near 0 is the intended regime.
    """
    agg: Dict = {}
    if str(getattr(args, "relocate", "off")) == "off":
        return agg
    reps = [t.report.get("relocate") or {} for t in tasks]
    reps = [r for r in reps if r]
    if not reps:
        print("\n  relocation: ON but no host reported it -- nothing was "
              "relocated. This is a bug, not a result.", flush=True)
        return agg
    ssum = lambda k: sum(int(r.get(k, 0)) for r in reps)      # noqa: E731
    med = lambda k, sub: float(np.median(                     # noqa: E731
        [float((r.get(k) or {}).get(sub, float("nan"))) for r in reps]))
    agg = {
        "mode": str(args.relocate),
        "eps": float(args.relocate_eps),
        "n_sets": ssum("n_sets"),
        "n_relocated": ssum("n_relocated"),
        "n_starved": ssum("n_starved"),
        "n_within_eps": ssum("n_within_eps"),
        "median_radius_err_p50_radii": med("radius_err_radii", "p50"),
        "median_radius_err_p90_radii": med("radius_err_radii", "p90"),
        "median_rrms_over_hr_p50": med("rrms_over_hr", "p50"),
        "median_angular_move_deg_p50": med("angular_move_deg", "p50"),
        "median_kept_hr_member_frac_p50": med("kept_hr_member_frac", "p50"),
        "n_particles_claimed": ssum("n_particles_claimed"),
        "particle_table_size": ssum("particle_table_size"),
    }
    n_rel = max(agg["n_relocated"], 1)
    print(f"\n  relocation: --relocate {args.relocate} "
          f"(arm {'B' if args.relocate == 'frozen' else 'C'}), "
          f"eps {args.relocate_eps} R_vir_host, {args.relocate_dirs} directions")
    print(f"    relocated        {agg['n_relocated']} of {agg['n_sets']} sets; "
          f"{agg['n_starved']} STARVED (kept their HR membership)")
    print(f"    within eps       {agg['n_within_eps']} "
          f"({agg['n_within_eps'] / n_rel * 100.0:.1f}%); radius err median "
          f"{agg['median_radius_err_p50_radii']:.4f}, p90 "
          f"{agg['median_radius_err_p90_radii']:.4f} R_vir_host")
    print(f"    r_rms / HR       median {agg['median_rrms_over_hr_p50']:.2f} "
          f"(1.0 is HR's own size; >>1 is material recruited from far away)")
    print(f"    moved            median {agg['median_angular_move_deg_p50']:.1f} "
          f"deg about the host; kept "
          f"{agg['median_kept_hr_member_frac_p50'] * 100.0:.1f}% of HR's members")
    print(f"    claimed          {agg['n_particles_claimed']} of "
          f"{agg['particle_table_size']} particles in the trained tiles",
          flush=True)
    return agg


def report_population(tasks: Sequence["HostTask"], args) -> None:
    """What the pool actually supervises, by parent split and by mass.

    The one line a `--population all` shakeout is read for. Two ways the run can
    silently not be the run it was configured to be, and both show here:

    * **the cap.** `--max-sets` with the default `largest` policy keeps the top
      of a mass function, so widening the population to 50p halos and leaving
      the cap alone can select the SAME sets as before with more of them thrown
      away. `n_dropped_by_cap` and the `num_p` quantiles say so.
    * **the live-fraction cut.** A field halo whose Lagrangian material is
      mostly outside the cluster's four tiles is dropped at `min_live_frac`.
      That is correct -- these tiles genuinely cannot build it -- but it means
      the reachable field population can be far smaller than the catalog's, and
      that is a fact about the TILING, answered by `--n-tiles`, not by the loss.
    """
    n_top = sum(int(t.report.get("n_top_level") or 0) for t in tasks)
    n_sub = sum(int(t.report.get("n_sub") or 0) for t in tasks)
    n_cap = sum(int(t.report.get("n_dropped_by_cap") or 0) for t in tasks)
    n_live = sum(int(t.report.get("n_dropped_live_frac") or 0) for t in tasks)
    tot = n_top + n_sub
    npall = np.concatenate([t.sets.num_p for t in tasks]) if tasks else np.zeros(0)
    print(f"\n  population: {args.population}, num_p window "
          f"[{args.min_num_p}, {args.max_num_p or 'inf'}), "
          f"cap {args.max_sets} by {args.set_select}")
    print(f"    supervised sets  {tot}: {n_sub} subhalo, {n_top} top-level "
          f"({(n_top / tot * 100.0) if tot else 0.0:.1f}% of the pool)")
    if npall.size:
        q = np.percentile(npall, [10, 50, 90])
        print(f"    num_p            min {int(npall.min())}, "
              f"p10 {q[0]:.0f}, median {q[1]:.0f}, p90 {q[2]:.0f}, "
              f"max {int(npall.max())}")
    print(f"    dropped          {n_cap} by the cap, {n_live} by live fraction")
    # The THIRD axis. Printed unconditionally, including for an
    # `--env-select any` run, because a targeted run's row is only readable
    # against an untargeted one's -- and because the set9 census says this
    # is the axis the deficit lives in: 6,557 of the region's 11,698
    # missing objects are in `near`, and the small-halo arm, which selected
    # on size alone, closed 3.8% of them.
    env_n = {lab: sum(int((t.report.get("n_by_env") or {}).get(lab, 0))
                      for t in tasks) for lab in ENV_LABELS}
    print(f"    env_select       {args.env_select} (host floor "
          f"{args.env_host_floor}p): "
          + ", ".join(f"{lab} {env_n[lab]}" for lab in ENV_LABELS))
    want = [w.strip() for w in str(args.env_select).split(",") if w.strip()]
    if want and want[0] not in ("any", "all"):
        got = sum(env_n[w] for w in want if w in env_n)
        if got == 0:
            print(">>> WARNING: --env-select "
                  f"{args.env_select} selected ZERO sets. The environment "
                  "cut runs BEFORE the cap, so this is the pool, not the "
                  "budget: these tiles genuinely home no halo in that bin. "
                  "Check --env-host-floor and --n-tiles.", flush=True)
        elif got < 0.1 * max(tot, 1):
            print(f">>> WARNING: --env-select {args.env_select} kept {got} "
                  f"of {tot} sets ({got / max(tot, 1) * 100:.1f}%). A "
                  "narrow pool is not the same experiment as a targeted "
                  "one -- raise --n-tiles or widen the selection if this "
                  "is smaller than intended. Continuing.", flush=True)
    if args.population != "sub" and n_top == 0:
        print(">>> WARNING: --population "
              f"{args.population} selected ZERO top-level sets. The run is a "
              "'sub' run with extra arguments. Check --max-num-p against the "
              "num_p range above, and --min-live-frac: a field halo whose "
              "material is mostly outside the cluster's tiles is dropped there.",
              flush=True)
    if args.population != "sub" and args.set_select == "largest" and n_cap:
        print(">>> WARNING: the cap binds and --set-select is 'largest', which "
              "keeps the top of the mass function and drops exactly the small "
              "halos this population was widened to reach. Use "
              "--set-select stratified unless truncation IS the experiment.",
              flush=True)


def _host_row(t: "HostTask") -> Dict:
    """One host's line in pool.json, including the reference it can reach.

    ``reference_median`` is the target the loss actually uses, and which of the
    two readings that is depends on ``--stats-members``. ``pure_hr_median``
    (live members only) and ``hybrid_reference_median`` (live plus the frozen
    out-of-tile stragglers) are BOTH reported under either policy, and the
    ratio between them is the straggler inflation: *measured* 2.18x in
    ``r_rms`` and 1.99x in ``2T/|W|`` at a median live fraction of 0.979
    (`reference-inflated-by-stragglers`).

    The original argument for the hybrid was that a member outside the trained
    tiles cannot be moved by the run or by the splice, so pure HR is not a
    reachable target. What that misses is that ``r_rms`` is an **rms**: the
    unreachable 2% dominate the statistic, so the reachable 98% barely move it,
    and the gradient on the members the run CAN move is diluted by the same
    factor. See ``docs/sr2_outskirt_detection.md`` section 6.
    """
    return {
        "key": t.key,
        "log_mvir": t.sel.log_mvir,
        "n_sets": int(t.sets.n_sets),
        "tiles": t.sel.tiles,
        "site_coverage": t.sel.site_coverage,
        "median_live_frac": t.report.get("median_live_frac"),
        # What was actually selected, after every cut and the cap. Under
        # --population all these are the only fields that say whether the run
        # got the field halos it was reconfigured to get, or whether the cap and
        # the live-fraction cut quietly handed back a subhalo-only pool.
        "population": t.report.get("population"),
        "n_top_level": t.report.get("n_top_level"),
        "n_sub": t.report.get("n_sub"),
        "num_p_min": t.report.get("num_p_min"),
        "num_p_median": t.report.get("num_p_median"),
        "num_p_max": t.report.get("num_p_max"),
        "cap_binds": t.report.get("cap_binds"),
        "n_dropped_by_cap": t.report.get("n_dropped_by_cap"),
        "set_select": t.report.get("set_select"),
        # The environment axis, per host. `n_after_env` against
        # `n_after_purity` is how much the cut took BEFORE the cap, which is
        # the distinction between "these tiles home no infall halos" (a
        # tiling fact) and "the budget went elsewhere" (a cap fact).
        "env_select": t.report.get("env_select"),
        "n_after_env": t.report.get("n_after_env"),
        "n_by_env": t.report.get("n_by_env"),
        "median_iso_radius_mpc_h": t.report.get("median_iso_radius_mpc_h"),
        "stats_members": t.report.get("stats_members"),
        "reference_median": t.report.get("reference_median"),
        "pure_hr_median": t.report.get("pure_hr_median"),
        "hybrid_reference_median": t.report.get("hybrid_reference_median"),
        "outside_excluded_by_policy": t.report.get("outside_excluded_by_policy"),
        # The host preservation guard (present only when --w-host-sets > 0).
        "host_n_sets": int(t.host_sets.n_sets) if t.host_sets is not None else 0,
        "host_reference_median": (t.report.get("host_sets") or {}).get(
            "reference_median"),
        "host_median_live_frac": (t.report.get("host_sets") or {}).get(
            "median_live_frac"),
        # The soft-membership pool (present only under --soft-membership).
        # `pool_over_live` and `sum_pool_squared` are the two numbers that say
        # what the relaxation costs: the first is how much wider the candidate
        # pool is than the HR mask, the second is the O(N^2) pair bill the step
        # now commits to, against the hard arm's `sum_num_p_squared`.
        "soft_pools": t.report.get("soft_pools"),
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
        # One KD-tree per box, not one per host site: the labelling is a
        # property of the catalog and the eight sites share it. Always
        # computed, even at --env-select any, because every run then
        # RECORDS the environment mix it supervised and the arms become
        # comparable on the axis the census says the deficit lives in.
        env = halo_environment(cat, host_floor=int(args.env_host_floor),
                               boxsize_mpc_h=BOXSIZE)
        sels = select_hosts(cat, oidx, box, n_tiles=int(args.n_tiles),
                            max_hosts=int(args.max_hosts_per_box),
                            min_log_mvir=float(args.min_log_mvir),
                            ng_hr=NG_HR, tile_hr=TILE, children=children)
        print(f"  {box}: {len(sels)} hosts >= logM {args.min_log_mvir} "
              f"[{time.time() - t0:.0f}s to load owner+catalog]", flush=True)
        for sel in sels:
            task = build_host_sets(sel, cat, oidx, cfg, geom, scfg, pscfg, mcfg,
                                   args, frozen_model, device, env=env)
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
        del cat, oidx, children, env
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
                **({"cond": d["cond"].detach().cpu()} if "cond" in d else {}),
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
    # Same rule as `stack_cond`: whether the stack is consumed is the MODEL's
    # property. This pool is also forwarded through the frozen reference, which
    # has no conditioning columns.
    cond = (torch.stack([b["cond"].to(device) for b in batch], dim=0)
            if ("cond" in batch[0] and int(getattr(model, "cond_chan", 0)))
            else None)
    cand = trim_to_tile(model(lr, noise=noise, cond=cond), geom).float()
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
# The unsupervised-tile kNN-distance support (src/cosmo_sr/features/knn_cdf.py)
# --------------------------------------------------------------------------- #
# The high-k hinge above "only constrains the AMOUNT of small-scale power, never
# its placement", and `density-power-cannot-see-subhalos` measured that the
# objects this line builds are below the grid any spectrum is taken on. This is
# the grid-free, real-space, catalog-free statistic charged on the SAME pool:
# per tile, the distribution of distances to the k-th nearest particle, against
# that tile's own HR. It is translation-invariant, so it is a support term and
# never a substitute for the supervised centre term.
def temp_enabled(args) -> bool:
    """The temperature term is built when it is charged OR only measured."""
    return float(getattr(args, "w_temp", 0.0)) > 0 or bool(getattr(args, "temp_measure", False))


def temp_config_from_args(args) -> TempConfig:
    return TempConfig(k=int(args.temp_k), queries_per_bin=int(args.temp_queries),
                      max_x=float(args.temp_max_x), min_x=float(args.temp_min_x),
                      min_queries=int(args.temp_min_queries), boxsize_mpc_h=BOXSIZE)


def knn_config_from_args(args) -> KnnConfig:
    return KnnConfig(ks=tuple(int(k) for k in args.knn_ks),
                     n_queries=int(args.knn_queries),
                     n_quantiles=int(args.knn_quantiles),
                     one_sided=bool(args.knn_one_sided),
                     dead_zone=float(args.knn_dead_zone),
                     buffer_cells=float(args.knn_buffer_cells),
                     phase_lambda=float(args.knn_phase_lambda),
                     strata=int(getattr(args, "knn_strata", 0)),
                     strata_rank=int(getattr(args, "knn_strata_rank", 0)),
                     strata_charge=knn_strata_charge(args))


def knn_strata_charge(args) -> Tuple[int, ...]:
    """Parse ``--knn-strata-charge``: a comma or space list of stratum indices.

    Empty charges every stratum, which -- and this is the one thing to know
    before using the flag -- is arithmetically the POOLED loss in finer bins.
    Partitioning is not reweighting; see ``KnnConfig.strata_charge``.
    """
    raw = getattr(args, "knn_strata_charge", None) or []
    out = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            out.append(int(part))
    return tuple(out)


def knn_class_ids(args) -> Tuple[int, ...]:
    """Indices into ``OWNER_CLASSES`` of the classes the term is charged on."""
    names = [c for c in (args.knn_classes or []) if c]
    bad = [c for c in names if c not in OWNER_CLASSES]
    if bad:
        raise SystemExit(f"--knn-classes {bad} not in {list(OWNER_CLASSES)}")
    return tuple(OWNER_CLASSES.index(c) for c in names)


def attach_knn_classes(pool: Sequence[Dict], args) -> None:
    """Give each pool tile its rows' HR owner classes (`classify_owner`).

    MEASURED, round 5b: the pooled statistic ranks the working arms backwards
    because hosts dominate it, so the term is charged per class and only on the
    classes named in ``--knn-classes``. The class vector is HR's, is the same
    for both fields (Lagrangian ids), and is cached per box next to the
    diagnostics.
    """
    # Attached whenever the classes are NAMED, even when they are demoted to a
    # diagnostic: the point of --knn-classes-diagnostic-only is to keep the
    # measurement and drop the supervision, so the labels still have to be here.
    if not knn_class_ids(args) or not pool:
        return
    root = Path(os.environ.get("DMSR_REWARD_ROOT", "")) if os.environ.get(
        "DMSR_REWARD_ROOT") else None
    boxes: Dict[str, List[Dict]] = {}
    for it in pool:
        boxes.setdefault(it["box"], []).append(it)
    for box, items in boxes.items():
        t0 = time.time()
        cache = None if root is None else (
            root / "member_gather" / "knn_cdf" / f"classes_{box}.npy")
        if cache is not None and cache.exists():
            cls = np.load(cache, mmap_mode="r")
        else:
            owner_p = (root / "halos_particles" / f"{box}__hr__hr"
                       / f"{box}_hr_owner.npy") if root else None
            cat_g = sorted(glob.glob(
                f"{root}/halos/{box}__hr__hr/*_rockstar/halos_*.ascii")) if root else []
            if owner_p is None or not owner_p.exists() or not cat_g:
                raise SystemExit(
                    f"--knn-classes needs {box}'s HR owner array and Rockstar "
                    f"catalog under DMSR_REWARD_ROOT (looked for {owner_p})")
            cat = load_rockstar_ascii(cat_g[0])
            cls = classify_owner(np.load(str(owner_p), mmap_mode="r"),
                                 cat.ids, cat.parent_ids, cat.num_p)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, cls)
        for it in items:
            rows = tile_rows_flat(int(it["tile"]), ng=NG_HR, tile=TILE)
            it["knn_classes"] = torch.from_numpy(
                np.ascontiguousarray(np.asarray(cls[rows], dtype=np.int8)))
        counts = {OWNER_CLASSES[i]: int(sum(int((it["knn_classes"] == i).sum())
                                            for it in items))
                  for i in knn_class_ids(args)}
        print(f"  kNN classes for {len(items)} {box} tiles: {counts} "
              f"[{time.time() - t0:.0f}s]", flush=True)


def attach_knn_neighbours(pool: Sequence[Dict], cfg, args, kw,
                          cfg_knn: Optional[KnnConfig] = None) -> None:
    """Give each pool tile the OTHER tiles' particles around it, per field.

    MEASURED, knn_cdf_set9.json (job 49259): a tile's own particles are not a
    usable candidate set at z=0 -- displacements are a whole tile (median |Psi|
    11.2 Mpc/h on the one HR tile checked), and the p95 overestimate of r_128 stayed at +62-80% at every
    buffer. So every pool tile carries, as CPU float32 constants:

    * ``knn_extra_frozen`` -- neighbours from the cached FROZEN SR2 box, for the
      candidate. The generator is not run on neighbouring tiles (27x the forward),
      so the candidate's surroundings are the frozen field's. Whether that
      staleness matters is ``tuned_centroid_frozen`` vs ``tuned_centroid_nbr`` in
      knn_cdf_split_diagnostic.py; read it before trusting this term.
    * ``knn_extra_hr`` -- neighbours from the HR box, for the reference. Exact.

    Selected once, about each field's own circular-mean centroid of the tile,
    ``margin`` cells beyond the tile's half side. Two box loads per box.
    """
    margin = float(args.knn_nbr_margin_cells)
    cfg_knn = cfg_knn if cfg_knn is not None else knn_config_from_args(args)
    if margin <= 0 or not pool:
        return
    cell = BOXSIZE / NG_HR
    half = 0.5 * TILE * cell + margin * cell
    boxes: Dict[str, List[Dict]] = {}
    for it in pool:
        boxes.setdefault(it["box"], []).append(it)
    for box, items in boxes.items():
        t0 = time.time()
        fpath = find_base_field(box, seed=int(args.seed))
        if fpath is None:
            raise SystemExit(
                f"--knn-nbr-margin-cells > 0 needs the cached frozen SR2 box for "
                f"{box} seed {args.seed} (the candidate's neighbours). Produce it "
                "with scripts/reward/cache_sr2_base.py, or set the margin to 0.")
        srcs = (("frozen", np.load(str(fpath), mmap_mode="r")),
                ("hr", load_hr(cfg, box)))
        n_extra = 0
        for tag, field in srcs:
            # 6D: the neighbour material must carry the same velocity columns as
            # the live tile, or a neighbour would sit at v=0 in phase space.
            bp = box_phase(field, ng=NG_HR, boxsize_mpc_h=BOXSIZE,
                           dis_scale_mpc_h=float(kw["dis_scale_mpc_h"]),
                           vel_scale_kms=float(kw["vel_scale_kms"])
                           if cfg_knn.phase_lambda > 0 else 1.0,
                           phase_lambda=float(cfg_knn.phase_lambda))
            for it in items:
                own = bp[tile_rows_flat(int(it["tile"]), ng=NG_HR, tile=TILE), :3]
                ang = own.astype(np.float64) * (2 * np.pi / BOXSIZE)
                c = np.mod(np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
                           * BOXSIZE / (2 * np.pi), BOXSIZE)
                nb, _ = box_neighbours(bp[:, :3], int(it["tile"]), c, half,
                                       ng=NG_HR, tile=TILE, boxsize_mpc_h=BOXSIZE)
                xp = bp[nb]
                it[f"knn_extra_{tag}"] = torch.from_numpy(np.ascontiguousarray(xp))
                n_extra += int(xp.shape[0])
            del bp
        print(f"  kNN neighbours for {len(items)} {box} tiles: mean "
              f"{n_extra / (2 * len(items)) / TILE ** 3:.2f}x a tile per field "
              f"[{time.time() - t0:.0f}s]", flush=True)


def _unsup_knn(model, batch: Sequence[Dict], geom, device, args, kw,
               kcfg: KnnConfig, generator=None, cand=None, hr=None,
               class_ids: Sequence[int] = ()):
    """``(mean penalty, per-tile diagnostics)`` over a minibatch of pool tiles.

    One particle table PER TILE, never one for the batch: pool tiles come from
    different boxes, and a shared table would offer box A's particles as
    neighbours to a query in box B wherever their coordinates overlap. Positions
    go to float64 before the table is built -- a minimum-image difference of two
    ~100 Mpc/h float32 coordinates carries ~1e-5 Mpc/h of rounding, which is not
    small against the r_8 of a subhalo core.
    """
    if cand is None:
        cand, hr = _forward_unsup(model, batch, geom, device)
    pens, diags = [], []
    for b, tile in enumerate(batch):
        cp, cv = tile_particles(cand[b:b + 1].double(), [int(tile["tile"])], **kw)
        with torch.no_grad():
            hp, hv = tile_particles(hr[b:b + 1].double(), [int(tile["tile"])], **kw)
        lam = float(kcfg.phase_lambda)
        if lam > 0:
            # (x, v/lambda): one metric, both halves differentiable, so the term
            # constrains the velocity field as well as the positions.
            cp = torch.cat([cp, cv / lam], dim=1)
            hp = torch.cat([hp, hv / lam], dim=1)
        ce, he = tile.get("knn_extra_frozen"), tile.get("knn_extra_hr")
        kcls = tile.get("knn_classes")
        pen, d = knn_tile_loss(cp, hp, torch.arange(cp.shape[0], device=cp.device),
                               int(tile["tile"]), kcfg, ng_hr=NG_HR, tile_hr=TILE,
                               boxsize_mpc_h=BOXSIZE, generator=generator,
                               cand_extra=None if ce is None else ce.to(cp.device),
                               hr_extra=None if he is None else he.to(cp.device),
                               centre=str(getattr(args, "knn_centre", "lagrangian")),
                               classes=None if kcls is None else kcls.to(cp.device),
                               class_ids=class_ids or ())
        d["box"] = tile["box"]
        diags.append(d)
        if not d.get("skipped"):
            pens.append(pen)
    if not pens:
        return torch.zeros((), device=cand.device, dtype=torch.float64), diags
    return torch.stack(pens).mean(), diags


@torch.no_grad()
def eval_unsup_knn(model, pool: Sequence[Dict], geom, device, args, kw,
                   kcfg: KnnConfig, batch: int = 8,
                   class_ids: Sequence[int] = ()) -> Optional[Dict]:
    """Median over a pool of each tile's bottom-decile ``r_k`` ratio to HR.

    ``ratio_q10`` and not the median ratio: `knn_cdf_loss`'s docstring records
    that a 120-particle clump moves the loss 1500x while moving the MEDIAN ratio
    by 0.2%. A fixed generator seed so an eval is the same number twice.
    """
    if not pool:
        return None
    gen = torch.Generator(device=device).manual_seed(int(kcfg.seed))
    q10, med, loss, skipped = [], [], [], 0
    dense = []
    for i in range(0, len(pool), batch):
        _, diags = _unsup_knn(model, pool[i:i + batch], geom, device, args, kw,
                              kcfg, generator=gen, class_ids=class_ids)
        for d in diags:
            if d.get("skipped"):
                skipped += 1
                continue
            q10.append(d["ratio_q10"])
            med.append(d["ratio_med"])
            loss.append(d["loss"])
            if "ratio_med_dense" in d:
                dense.append(d["ratio_med_dense"])
    if not q10:
        return {"ks": list(kcfg.ks), "n_tiles": 0, "n_skipped": skipped}
    return {
        "ks": list(kcfg.ks),
        "phase_lambda": float(kcfg.phase_lambda),
        "classes": [OWNER_CLASSES[i] for i in class_ids],
        "n_tiles": len(q10), "n_skipped": skipped,
        "ratio_q10": np.median(np.asarray(q10), axis=0).tolist(),
        "ratio_med": np.median(np.asarray(med), axis=0).tolist(),
        # The dense stratum's median ratio: the readout the pooled form cannot
        # give (measured -- pooled reads 1.00 on a field whose dense tail is
        # 1.4-2.0x HR's). Absent unless --knn-strata > 1.
        **({"ratio_med_dense": np.median(np.asarray(dense), axis=0).tolist()}
           if dense else {}),
        "strata": int(kcfg.strata),
        "strata_charge": [int(j) for j in kcfg.strata_charge],
        "loss": float(np.median(loss)),
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


def report_rss(when: str) -> float:
    """Print peak host RSS so far, in GB, and return it.

    Why this exists: the job asks for `--mem=96G` and nothing has ever measured
    what it actually uses, so the request could not be defended or reduced. On
    2026-09-09 that mattered -- 96G did not fit any AVAILABLE node on `general`
    (the only two with a free, unreserved GPU had 20G and 84G), so every arm sat
    PENDING behind a number nobody had checked. `sacct` is not reachable from
    here, so the job has to report it itself.

    `ru_maxrss` is a high-water mark, not current usage: it only rises, and it
    is in KiB on Linux. It does NOT include the mmap'd frozen box's page cache,
    which Slurm may still charge -- so treat this as a floor on what to request.
    """
    import resource
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)
    print(f"  [mem] peak host RSS {when}: {gb:.1f} GB", flush=True)
    # The GPU peak too: the soft arms were pinned to A6000s (--exclude gpu28,
    # the 24 GB A5000s) only because no run had ever recorded this number.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        gpu = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        print(f"  [mem] peak GPU allocated {when}: {gpu:.1f} GB", flush=True)
    return gb


def draw_seeds(args) -> List[int]:
    """Root seeds for the noise realisations this run may use.

    Index 0 is ``--seed`` itself, so a run with ``--noise-draws 1`` and no fresh
    eval is byte-identical to every arm before 2026-09-09. The stride is 1000
    rather than 1 only so a draw index is readable in a log line.

    The list is ``--noise-draws`` training draws followed, when
    ``--fresh-noise-eval`` is set, by ONE reserved draw that no step ever sees.
    That reserved draw is the whole measurement: it turns the train/holdout
    comparison into a 2x2 over (seen box, seen noise), which is what separates
    "the LR field carries this" from "the fixed noise pattern was memorised".
    """
    n = max(1, int(getattr(args, "noise_draws", 1)))
    if bool(getattr(args, "fresh_noise_eval", False)):
        n += 1
    return [int(args.seed) + 1000 * d for d in range(n)]


def fresh_draw_index(args) -> Optional[int]:
    """Index of the reserved never-trained draw, or ``None``."""
    if not bool(getattr(args, "fresh_noise_eval", False)):
        return None
    return max(1, int(getattr(args, "noise_draws", 1)))


def add_noise_draws(data: Dict[int, Dict], tiles: Sequence[int], geom,
                    device, seeds: Sequence[int]) -> None:
    """Attach ``noise_draws`` (one dict per seed) to each tile's data, in place.

    ``build_tiles`` already produced the ``--seed`` draw as ``noise``; that
    tensor is reused as entry 0 rather than regenerated, so entry 0 is the same
    object the single-draw path uses and cannot drift from it.
    """
    for t in tiles:
        draws = [data[t]["noise"]]
        for sd in list(seeds)[1:]:
            draws.append(tile_noise_stack([int(sd)], int(t), geom,
                                          device=device))
        data[t]["noise_draws"] = draws


def forward_tiles_draw(model, data: Dict[int, Dict], tiles: Sequence[int],
                       geom, draw: int = 0):
    """:func:`forward_tiles` on one noise draw.

    Falls straight through to ``forward_tiles`` for draw 0 with no
    ``noise_draws`` present, which keeps the single-draw path untouched.
    """
    if int(draw) == 0 or "noise_draws" not in data[tiles[0]]:
        return forward_tiles(model, data, tiles, geom)
    sub = {t: {"lr": data[t]["lr"],
               "noise": data[t]["noise_draws"][int(draw)],
               "hr": data[t]["hr"],
               **({"cond": data[t]["cond"]} if "cond" in data[t] else {})}
           for t in tiles}
    return forward_tiles(model, sub, tiles, geom)


def host_forward(model, task: HostTask, geom, kw, mcfg, device,
                 frozen_model=None, set_indices=None, term_scale=None,
                 host_set_indices=None, draw: int = 0, shell_draws=(),
                 guard_ref: str = "frozen"):
    """``(candidate, base, hr, gather_loss, diagnostics, host)`` for one host.

    ``base`` is recomputed under ``no_grad`` rather than cached on the GPU: it is
    one frozen forward on four tiles, and caching it for every host of a large
    pool is the difference between fitting in GPU memory and not.

    ``host`` is ``None`` unless the task carries preservation sets, in which case
    it is ``(host_gather, host_diag)`` from the SAME particle table -- the host
    term never triggers a second forward or a second ``tile_particles``.
    """
    nds = task.tile_data[task.tiles[0]].get("noise_draws")
    data = {t: {"lr": task.tile_data[t]["lr"].to(device),
                "noise": {s: v.to(device)
                          for s, v in task.tile_data[t]["noise"].items()},
                "hr": task.tile_data[t]["hr"].to(device),
                **({"cond": task.tile_data[t]["cond"].to(device)}
                   if "cond" in task.tile_data[t] else {})}
            for t in task.tiles}
    if nds is not None:
        for t in task.tiles:
            data[t]["noise_draws"] = [{s: v.to(device) for s, v in nd.items()}
                                      for nd in task.tile_data[t]["noise_draws"]]
    cand = forward_tiles_draw(model, data, task.tiles, geom, draw).float()
    hr = task.hr.to(device)
    # `guard_ref` picks what the LR-scale anchor and `vel_rms_ratio` are measured
    # AGAINST. "frozen" is every arm on disk and stays the default: in a
    # fine-tune the candidate STARTS as the frozen field, so the anchor is a
    # drift detector and the frozen field is the only meaningful zero.
    #
    # "hr" exists for `train_sr2_scratch.py`. A generator trained from a random
    # initialisation has not drifted from the frozen field -- it has never been
    # there -- so anchoring to it would charge the run for failing to reproduce
    # a checkpoint whose central concentration is the very thing under test
    # (`radial-placement-is-inverted`). The operator is unchanged; only the
    # reference moves, and `frozen_model` is then not needed at all.
    if str(guard_ref) == "hr":
        base = hr
    else:
        with torch.no_grad():
            base = forward_tiles_draw(frozen_model, data, task.tiles, geom,
                                      draw).float()
    sets = task.sets_at(draw).to(device)
    pos, vel = tile_particles(cand, task.tiles, **kw)
    gather, diag = member_gather_loss(pos, vel, sets, mcfg, set_indices,
                                      term_scale=term_scale)
    # --- the NOISE-AVERAGED shell (--shell-noise-draws) --------------------
    # The term claims to be about a DISTRIBUTION but is evaluated on one
    # stochastic realisation, which is exactly the freedom the 2026-09-08 sweep
    # exploited: 7.3x on train, 1.4x held out. Averaging the reduced curve over
    # several draws BEFORE charging asks for a distribution no single draw can
    # fake. Note the order: mean-then-charge, not charge-then-mean -- the second
    # is just a lower-variance version of the same per-realisation objective.
    #
    # Only the radii are recomputed on the extra draws (`shell_radii`), never
    # the O(N^2) statistics: the other terms keep their primary-draw values and
    # the extra cost is a forward pass and a gather.
    curves = diag.pop("_shell_curves", None)
    shell_avg = None
    if curves is not None and shell_draws:
        lc0, lt0 = curves
        lcs = [lc0]
        for d2 in shell_draws:
            cand2 = forward_tiles_draw(model, data, task.tiles, geom, d2).float()
            pos2, _v2 = tile_particles(cand2, task.tiles, **kw)
            sets2 = task.sets_at(d2).to(device)
            rc2, rt2 = shell_radii(pos2, sets2, mcfg, set_indices)
            lc2, _lt2 = shell_reduce(rc2, rt2, mcfg)
            # Shapes must agree or the mean would be of two different
            # reductions. They do by construction -- selection is
            # draw-independent, checked when the tables were built -- so a
            # mismatch is a bug, not a case to paper over.
            if lc2.shape != lc0.shape:
                raise RuntimeError(
                    f"{task.key}: shell curve on draw {d2} has shape "
                    f"{tuple(lc2.shape)} against the primary draw's "
                    f"{tuple(lc0.shape)}")
            lcs.append(lc2)
        if len(lcs) > 1:
            lc_bar = torch.stack(lcs, dim=0).mean(dim=0)
            w = float(mcfg.w_shell)
            sc = max(float((term_scale or {}).get("shell", 1.0)), 1e-6)
            # Swap the primary-draw charge out for the averaged one. Exact:
            # `shell_charge(lc0, lt0)` recomputes the very term `gather`
            # already carries, from the same live tensors.
            gather = (gather - w * shell_charge(lc0, lt0, mcfg) / sc
                      + w * shell_charge(lc_bar, lt0, mcfg) / sc)
            shell_avg = float(shell_charge(lc_bar, lt0, mcfg).detach())
    diag["shell_avg_raw"] = (float("nan") if shell_avg is None else shell_avg)
    diag["n_shell_draws"] = 1 + len(shell_draws)
    host = None
    _hs = task.host_sets_at(draw)
    if _hs is not None and _hs.n_sets > 0:
        hsets = _hs.to(device)
        # Only selection and centre_mode distinguish the host config, and both
        # are already baked into `hsets` at build time; the estimator and the
        # per-term weights are mcfg's by design (same physics binds a host as a
        # subhalo, one `--w-host-sets` scales the whole term). centre_mode="self"
        # is required, not cosmetic: host sets carry an all-zero `centre_rhat`
        # (no clustercentric direction of their own), so "radial" would raise and
        # the self-anchor is the mode whose reference these sets were built for.
        # term_scale is dropped: it is measured on the subhalo terms and does not
        # describe the host ones.
        # soft_membership is forced OFF here, and that is not an oversight.
        # The guard says "the frozen field already resolves these hosts, do not
        # break them", referenced to the frozen field's own value on HR's own
        # particles. A relaxed membership could satisfy it with a DIFFERENT
        # collection of particles -- i.e. report a host intact while the host
        # HR named came apart -- which is precisely what a preservation
        # constraint must not permit. The subhalo terms are the ones being
        # asked to build something new; this one is being asked to keep
        # something, and keeping is an identity statement.
        host = member_gather_loss(
            pos, vel, hsets,
            # soft_host_profile goes off with it. The host profile is a
            # property of the RELAXATION's background -- there is no background
            # competing for particles once membership is HR's mask -- and
            # leaving it set here is what killed jobs 56560/56562 five minutes
            # in, on the MemberGatherConfig refusal that exists to catch
            # exactly this combination being set by hand.
            host_guard_config(mcfg),
            host_set_indices)
    return cand, base, hr, gather, diag, host


def host_loss(model, task, geom, kw, mcfg, args, device, frozen_model,
              set_indices=None, term_scale=None, critic=None, critic_norm=None,
              w_adv=0.0, host_set_indices=None, draw: int = 0,
              shell_draws=(), guard_ref: Optional[str] = None):
    """Total loss for one host, plus the row that gets logged.

    ``guard_ref`` defaults to ``args.guard_ref`` ("frozen" for every arm on
    disk), so the eval path and the training path cannot disagree about what the
    LR-scale anchor is measured against. See :func:`host_forward`.

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
        term_scale=term_scale, host_set_indices=host_set_indices,
        draw=draw, shell_draws=shell_draws,
        guard_ref=(str(getattr(args, "guard_ref", "frozen"))
                   if guard_ref is None else str(guard_ref)))
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
    # --- host-centric local temperature (--w-temp; host_temperature.py) -----
    # Velocity-only: neighbours are found without gradient, so this cannot move
    # the density field the velocity oracle showed is already right. Measured
    # under no_grad when only --temp-measure is on.
    temp_pen, temp_info = None, {}
    if task.temp_target is not None:
        pos_c, vel_c = tile_particles(cand, task.tiles, **kw)
        guess = torch.as_tensor(task.host_pos, dtype=pos_c.dtype, device=pos_c.device)
        tcfg = temp_config_from_args(args)
        if float(args.w_temp) > 0:
            prof = temperature_profile(pos_c, vel_c, guess, task.host_rvir, tcfg)
            temp_pen, temp_info = temperature_loss(
                prof, task.temp_target["mean"], task.temp_target["n"], tcfg)
            loss = loss + float(args.w_temp) * temp_pen
        else:
            with torch.no_grad():
                prof = temperature_profile(pos_c, vel_c, guess, task.host_rvir, tcfg)
                temp_pen, temp_info = temperature_loss(
                    prof, task.temp_target["mean"], task.temp_target["n"], tcfg)
    row = {
        # WHICH noise realisation this row was measured under. Without it a
        # rotating run's metrics.jsonl cannot be split by draw, and the whole
        # point of the rotation is to compare seen-noise against fresh-noise.
        "draw": int(draw),
        "shell_avg_raw": float(diag["shell_avg_raw"]),
        "n_shell_draws": int(diag["n_shell_draws"]),
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
        # Logged at --w-radius 0 too: it is the coordinate the `self` arm's
        # radial collapse lives in (set9: median 0.35 host R_vir against HR's
        # 1.13), and an arm that does not log it cannot be read against one
        # that does. Signed, so "too close" stays distinguishable from "too far".
        "radius_err_radii": diag["median_radius_err_radii"],
        "radius_ratio": diag["median_radius_ratio"],
        "radius_too_close": diag["frac_radius_too_close"],
        # The SHELL's own residual, logged at --w-shell 0 too. Read THIS for
        # that term and not `radius_err_radii`: a perfect distribution match
        # still leaves 79%% (train) / 85%% (holdout) of the per-set address
        # error standing, so the address diagnostic barely moves on a total win.
        "shell_resid": diag["median_shell_resid"],
        "shell_resid_signed": diag["median_shell_resid_signed"],
        "r_rms_over_hr": diag["median_r_rms_over_hr"],
        "sigma_v_over_hr": diag["median_sigma_v_over_hr"],
        "n_sets": int(diag["n_sets"]),
        "n_sets_total": int(task.sets.n_sets),
        # The pedestal. > 1 is "more non-member mass sits on this object
        # than HR has", which is the state that makes Rockstar file it as
        # somebody's subhalo -- logged at --w-isolation 0 too, so an arm
        # that does not charge for it can still be read against one that
        # does. `n_iso` is how many sets the term could be taken on.
        "iso_ratio": diag["median_iso_ratio"],
        "iso_above_hr": diag["frac_iso_above_hr"],
        "n_iso": int(diag["n_iso"]),
        # WHICH population and WHICH environment moved, not just the pool.
        "by_population": diag["by_population"],
        "by_env": diag["by_env"],
        # --- the relaxation's own health, NaN on the hard arm ---------------
        # `soft_mass_ratio` must sit at ~1: it is the constraint that stops a
        # candidate keeping a tiny cold core, and a run drifting below it is a
        # run whose lambda made the window unfillable. `soft_hr_fraction` is
        # how much of each solved object is still HR's own material -- 1.0 is
        # the hard arm, and a collapse toward 0 is the relaxation walking off
        # the object, which is the failure mode the Rockstar gate would
        # otherwise be the first to find, ten hours later.
        "soft_mass_ratio": diag.get("soft_mass_ratio", float("nan")),
        "soft_mass_ratio_min": diag.get("soft_mass_ratio_min", float("nan")),
        "soft_hr_fraction": diag.get("soft_hr_fraction", float("nan")),
        "soft_n_eff_over_target": diag.get("soft_n_eff_over_target",
                                           float("nan")),
        # The host-profile background. NaN on every arm that does not use it,
        # so the columns exist from the frozen start and a run can be read
        # against one that predates the term. `soft_host_valid_frac` is the
        # one to watch on the `lr` arm: the LR catalog is incomplete below its
        # gate, and a run near 0 here is the arm doing NOTHING -- which looks
        # exactly like the arm failing if this column is missing. It WAS
        # missing: this row is an explicit whitelist, so a key returned in
        # `diag` but not named here never reaches metrics.jsonl or wandb.
        "soft_host_weight": diag.get("soft_host_weight", float("nan")),
        "soft_host_valid_frac": diag.get("soft_host_valid_frac", float("nan")),
        "soft_host_logrho_p10": diag.get("soft_host_logrho_p10", float("nan")),
        "soft_host_logrho_p90": diag.get("soft_host_logrho_p90", float("nan")),
        # The temperature term, NaN when it is neither charged nor measured.
        # temp_core_ratio < 1 is the cold core Rockstar fragments; > 1 in
        # temp_out_ratio is the hot outskirts (HR = 1.00 on both).
        "temp_raw": (float(temp_pen.detach()) if temp_pen is not None else float("nan")),
        "term_temp": (float(args.w_temp) * float(temp_pen.detach())
                      if temp_pen is not None else float("nan")),
        "temp_core_ratio": temp_info.get("temp_core_ratio", float("nan")),
        "temp_out_ratio": temp_info.get("temp_out_ratio", float("nan")),
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


def save_atomic(obj, path) -> None:
    """``torch.save`` through a temp file and one ``os.replace``.

    A plain ``torch.save`` to a live path is a multi-second, non-atomic write,
    and this run's ``tuned.pt`` is rewritten every eval so a gate can always load
    the latest model. The moment a gate is allowed to fire WHILE training is
    still going -- which is the whole point of the mid-run checkpoints -- those
    two collide: the export spends ~138 s loading a file the trainer is
    overwriting underneath it, and the failure is a torch unpickling error whose
    message says nothing about a race. ``os.replace`` is atomic on POSIX, so a
    reader sees either the old file or the new one, never half of one.

    The temp file sits in the SAME directory on purpose: ``os.replace`` across
    filesystems is not atomic and raises, and ``/tmp`` on a compute node is not
    the shared filesystem the checkpoint lives on.
    """
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def gate_steps(args) -> List[int]:
    """The steps at which to freeze an immutable checkpoint for an early gate.

    Parsed once, up front, so a typo is a startup error rather than a step-2000
    surprise. Out-of-range steps raise for the same reason: a gate chain queued
    against a snapshot that will never be written strands four jobs on
    `Dependency` with nothing to read.
    """
    raw = str(getattr(args, "gate_at_steps", "") or "").strip()
    if not raw:
        return []
    try:
        out = sorted({int(x) for x in raw.replace(",", " ").split()})
    except ValueError:
        raise SystemExit(
            f"--gate-at-steps {raw!r} is not a list of integers "
            "(e.g. '2000' or '1000,2000').")
    bad = [x for x in out if not (0 < x < int(args.steps))]
    if bad:
        raise SystemExit(
            f"--gate-at-steps {bad} outside (0, --steps {args.steps}). The "
            "final checkpoint is already gated by the normal chain; a snapshot "
            "at or past the last step would only duplicate it, and one past it "
            "would never be written -- stranding the gate queued against it.")
    return out


def _TERM_WEIGHTS(mcfg) -> Dict[str, float]:
    """The nine ``member_gather_loss`` weights, keyed as its ``term_*`` diag."""
    return {"virial": mcfg.w_virial, "bound": mcfg.w_bound, "d6": mcfg.w_d6,
            "rrms": mcfg.w_rrms, "sigmav": mcfg.w_sigmav,
            "centre": mcfg.w_centre, "radius": mcfg.w_radius,
            "shell": mcfg.w_shell, "isolation": mcfg.w_isolation}


_TERM_NAMES = ("virial", "bound", "d6", "rrms", "sigmav", "centre", "radius",
               "shell", "isolation")


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
                  "centre_offset_radii", "radius_err_radii", "radius_ratio",
                  "shell_resid", "shell_resid_signed",
                  "r_rms_over_hr", "sigma_v_over_hr",
                  "low_k", "highk_ratio", "highk_ratio_max", "vel_rms_ratio",
                  "iso_ratio", "iso_above_hr",
                  "soft_mass_ratio", "soft_mass_ratio_min",
                  "soft_hr_fraction", "soft_n_eff_over_target")
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
    for k in ("wall_s", "batch_gather", "batch_unsup_pen", "batch_unsup_knn"):
        if k in row:
            flat[k] = float(row[k])
    # The box-wide guard's own number, per split -- the worst-band ratio is what
    # the arm exists to move and what the verdict gates on.
    for side, block in (row.get("unsup") or {}).items():
        if block and block.get("ratio_max") is not None:
            flat[f"unsup/{side}/highk_ratio_max"] = float(block["ratio_max"])
        knn = (block or {}).get("knn") or {}
        for k, r in zip(knn.get("ks", []), knn.get("ratio_q10", [])):
            flat[f"unsup/{side}/knn_q10_k{k}"] = float(r)
        for k, r in zip(knn.get("ks", []), knn.get("ratio_med_dense", [])):
            flat[f"unsup/{side}/knn_dense_k{k}"] = float(r)
    for side in ("train", "holdout"):
        pool = row.get(side) or {}
        for k in _POOL_SCALARS:
            if k in pool and pool[k] is not None:
                flat[f"{side}/{k}"] = pool[k]
        # The splits, one series per (split, bin, field). This is the chart
        # `all_blocks_smallhalo` needed and did not have: a pooled median
        # over both populations could not say whether its top-level sets --
        # its entire hypothesis -- were moving, and the answer only arrived
        # from a Rockstar gate ten hours later.
        for split in ("by_population", "by_env"):
            for name, blk in (pool.get(split) or {}).items():
                for f, v in (blk or {}).items():
                    if v is not None and np.isfinite(v):
                        flat[f"{side}/{name}/{f}"] = float(v)
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
                  device, frozen_model, label: str, term_scale=None,
                  draw: int = 0) -> Dict:
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
                               frozen_model, term_scale=term_scale, draw=draw)
            row["key"] = task.key
            row["log_mvir"] = task.sel.log_mvir
            rows.append(row)
    if not rows:
        return {"label": label, "n_hosts": 0, "per_host": []}

    def med(name):
        return float(np.median([r[name] for r in rows]))

    def nmed(name):
        """Median over the hosts that HAVE a finite value; NaN if none do."""
        v = np.asarray([r.get(name, np.nan) for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else float("nan")

    out = {
        "label": label,
        "draw": int(draw),
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
        # The shell, pooled. `radius_ratio` is the one to read against the set9
        # radial profile: HR's subhalos sit at a median 1.13 host R_vir and the
        # `self` arm's at 0.35, so a ratio drifting below 1 here is that
        # collapse happening on the supervised sets themselves.
        "radius_err_radii": med("radius_err_radii"),
        "radius_ratio": med("radius_ratio"),
        # ...and the SOFT shell's residual, which is a different question from
        # both of the above: how far the set radii are from HR's radius
        # DISTRIBUTION, once the addresses are set aside. `--w-shell` drives
        # this one; `radius_err_radii` stays near where it started even when it
        # wins, because a perfect distribution match still leaves 79%/85% of the
        # per-set address error standing.
        "shell_resid": med("shell_resid"),
        "shell_resid_signed": med("shell_resid_signed"),
        "r_rms_over_hr": med("r_rms_over_hr"),
        "sigma_v_over_hr": med("sigma_v_over_hr"),
        "low_k": med("low_k"),
        "highk_ratio": med("highk_ratio"),
        "highk_ratio_max": float(np.max([r["highk_ratio"] for r in rows])),
        # Median band ratio across hosts, so the eval row carries the SHAPE of
        # the excess and not only its size. None when --highk-bins is 0.
        "highk_bands": _median_bands(rows),
        "vel_rms_ratio": med("vel_rms_ratio"),
        "iso_ratio": med("iso_ratio"),
        "iso_above_hr": med("iso_above_hr"),
        # --- the relaxation's health, pooled. NaN on the hard arm, which is
        # why these use `nmed`: `np.median` of an all-NaN column is NaN with a
        # warning, and one host missing the key would take the whole pool with
        # it. The keys are present in EVERY run so the two arms' rows stay one
        # schema and a soft arm can be charted against a hard one.
        "soft_mass_ratio": nmed("soft_mass_ratio"),
        "soft_mass_ratio_min": nmed("soft_mass_ratio_min"),
        "soft_hr_fraction": nmed("soft_hr_fraction"),
        "soft_n_eff_over_target": nmed("soft_n_eff_over_target"),
        # Pooled over hosts by SUMMING the counts and taking the median of
        # the medians -- a host with four `near` sets must not weigh the
        # same as one with two hundred in the count, and must not be
        # averaged away in the median either.
        "by_population": _pool_splits(rows, "by_population"),
        "by_env": _pool_splits(rows, "by_env"),
        "velhighk_ratio": med("velhighk_ratio"),
        "velhighk_ratio_min": float(np.min([r["velhighk_ratio"] for r in rows])),
        "temp_raw": nmed("temp_raw"),
        "temp_core_ratio": nmed("temp_core_ratio"),
        "temp_out_ratio": nmed("temp_out_ratio"),
        "per_host": rows,
    }
    for k in _TERM_NAMES:
        out[f"term_{k}"] = med(f"term_{k}")
        out[f"term_raw_{k}"] = med(f"term_raw_{k}")
        out[f"term_eff_{k}"] = med(f"term_eff_{k}")
    return out


def _pool_splits(rows: List[Dict], key: str) -> Dict[str, Dict[str, float]]:
    """Pool per-host ``by_population`` / ``by_env`` blocks over the hosts.

    ``n`` sums; every other field is a median over the hosts that HAVE that
    split non-empty. A host contributing zero sets to a split must not drag its
    medians toward NaN, and must not silently vanish from the count either --
    which is the distinction that makes "the near bin got 4 sets" readable as a
    selection failure rather than as a learning one.
    """
    names: List[str] = []
    for r in rows:
        for k in (r.get(key) or {}):
            if k not in names:
                names.append(k)
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        blocks = [(r.get(key) or {}).get(name) for r in rows]
        blocks = [b for b in blocks if b]
        d: Dict[str, float] = {"n": float(sum(b.get("n", 0.0) for b in blocks))}
        for f in ("bound_hard", "virial", "centre_offset_radii", "iso_ratio"):
            v = np.asarray([b.get(f, np.nan) for b in blocks
                            if b.get("n", 0.0) > 0], dtype=np.float64)
            v = v[np.isfinite(v)]
            d[f] = float(np.median(v)) if v.size else float("nan")
        out[name] = d
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
def _box_to_tiles(box: np.ndarray, tiles: Sequence[int], *,
                  ng_hr: int = NG_HR, tile_hr: int = TILE) -> np.ndarray:
    """Slice a full-box field ``(6, ng, ng, ng)`` into ``(B, 6, T, T, T)``.

    The inverse of :func:`tile_particles`'s origin convention -- the same
    ``tid -> (tx, ty, tz)`` decomposition -- so a tile cut here and gathered
    there lands on the rows :func:`build_member_sets` indexed.
    """
    ng, th = int(ng_hr), int(tile_hr)
    n_side = ng // th
    out = []
    for tid in tiles:
        tid = int(tid)
        tx = tid // (n_side * n_side)
        ty = (tid // n_side) % n_side
        tz = tid % n_side
        out.append(box[:, tx * th:(tx + 1) * th,
                       ty * th:(ty + 1) * th, tz * th:(tz + 1) * th])
    return np.stack(out, axis=0)


def _dump_membership_solve(args, mcfg, scfg, pscfg, by_key, all_tasks) -> None:
    """Run one host's membership solve with the per-sweep recorder and save it."""
    from cosmo_sr.features.soft_membership import solve_memberships
    if not bool(mcfg.soft_membership):
        raise SystemExit("--dump-membership-npz needs --soft-membership on: "
                         "there is no P to solve for a hard-mask arm.")
    key = getattr(args, "dump_membership_host", None)
    if key:
        if key not in by_key:
            raise SystemExit(f"--dump-membership-host {key!r} not in the pool; "
                             f"pooled keys: {sorted(by_key)[:8]}...")
        task = by_key[key]
    else:
        # The richest host makes the clearest figure: most candidate subhaloes
        # to show, and the most particle competition in the budget matrix.
        cand = [t for t in all_tasks if getattr(t.sets, "pools", None) is not None]
        if not cand:
            raise SystemExit("no pooled host to dump; pass --dump-membership-host")
        task = max(cand, key=lambda t: int(t.sets.n_sets))
        print(f"  (no --dump-membership-host: picked {task.key}, the richest "
              f"with {int(task.sets.n_sets)} candidate sets)", flush=True)
    if getattr(task.sets, "pools", None) is None:
        raise SystemExit(f"host {task.key} carries no soft pools")

    field_path = args.dump_membership_field
    if not field_path or not os.path.exists(field_path):
        raise SystemExit(f"--dump-membership-field {field_path!r} does not exist")
    box = np.load(field_path, mmap_mode="r")
    if box.ndim != 4 or box.shape[0] < 6:
        raise SystemExit(f"expected a full-box (6, ng, ng, ng) field, got "
                         f"{tuple(box.shape)}")
    tiles = task.tiles
    sub = np.ascontiguousarray(_box_to_tiles(np.asarray(box), tiles))
    field = torch.as_tensor(sub, dtype=torch.float32)
    kw = dict(ng_hr=NG_HR, tile_hr=TILE, boxsize_mpc_h=BOXSIZE,
              dis_scale_mpc_h=float(scfg.dis_norm_kpc_h) * 1e-3,
              vel_scale_kms=float(pscfg.vel_norm_km_s))
    with torch.no_grad():
        pos, vel = tile_particles(field, tiles, **kw)

    sets = task.sets
    idx = list(range(sets.n_sets))
    rec: Dict = {}
    print(f"solving membership for {task.key}: {sets.n_sets} sets, "
          f"prior={mcfg.soft_prior} lambda={mcfg.soft_lambda} on {field_path}",
          flush=True)
    with torch.no_grad():
        solve_memberships(pos.double(), vel.double(), sets, sets.pools, idx, mcfg,
                          record=rec)

    out = args.dump_membership_npz
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sweeps = rec["sweeps"]
    np.savez_compressed(
        out,
        host_key=str(task.key), box=str(getattr(task.sel, "box", "")),
        halo_id=np.asarray(rec["halo_id"], dtype=np.int64),
        n_sets=int(sets.n_sets), lam=float(rec["lam"]),
        x=rec["x"].astype(np.float32), v=rec["v"].astype(np.float32),
        rows=rec["rows"].astype(np.int64),
        setid=rec["setid"].astype(np.int64),
        is_member=rec["is_member"].astype(bool),
        mult=rec["mult"].astype(np.float32),
        logpi=rec["logpi"].astype(np.float32),
        target_mass=rec["target_mass"].astype(np.float32),
        sigma_r=rec["sigma_r"].astype(np.float32),
        sigma_v=rec["sigma_v"].astype(np.float32),
        sweep_tags=np.asarray([s["tag"] for s in sweeps]),
        sweep_w=np.stack([s["w"] for s in sweeps]).astype(np.float32),
        sweep_c=np.stack([s["c"] for s in sweeps]).astype(np.float32),
        sweep_mass=np.stack([s["mass"] for s in sweeps]).astype(np.float32),
        sweep_hr_kept=np.stack([s["hr_kept"] for s in sweeps]).astype(np.float32),
        centre_ref=sets.centre_ref[torch.as_tensor(idx)].cpu().numpy().astype(np.float32),
        # The host point and the R_vir scale, so a RADIAL question can be
        # asked of this dump without reopening the field: `host_profile_probe`
        # signs the M-step drift on the clustercentric direction, and an
        # unsigned drift says nothing about placement.
        centre_host=sets.centre_host[torch.as_tensor(idx)].cpu().numpy().astype(np.float32),
        centre_scale=sets.centre_scale[torch.as_tensor(idx)].cpu().numpy().astype(np.float32),
        host_rvir=np.asarray(getattr(sets, "host_rvir", None)
                             if getattr(sets, "host_rvir", None) is not None
                             else np.zeros(sets.n_sets), dtype=np.float64)[idx],
        host_mvir=np.asarray(getattr(sets, "host_mvir", None)
                             if getattr(sets, "host_mvir", None) is not None
                             else np.zeros(sets.n_sets), dtype=np.float64)[idx],
    )
    print(f"  wrote {out}  ({len(sweeps)} snapshots, "
          f"{rec['x'].shape[0]} pool entries)", flush=True)


def run(args) -> Dict:
    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    pscfg = phase_space_config_of(cfg)
    mcfg = member_config_of(args)
    # --relocate-check-only is --pool-only plus the relocation report, and the
    # report is written by the pool stage either way. Resolved here rather than
    # branched on twice, so there is exactly one place that decides whether this
    # job trains.
    if bool(getattr(args, "relocate_check_only", False)):
        if str(args.relocate) == "off":
            raise SystemExit(
                "--relocate-check-only with --relocate off checks nothing: "
                "arm A's membership is HR's owner array and there is no "
                "selection to inspect. Pass --relocate frozen or hr.")
        args.pool_only = True
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    banner(f"member-gather fine-tune: rung={args.rung}")

    # --- the noise-realisation flags, checked before a pool is built --------
    # Each of these costs a full reference build per host per draw, so a
    # contradiction must fail here and not after twenty minutes of pool work.
    n_draws = max(1, int(args.noise_draws))
    if int(args.noise_draws) < 1:
        raise SystemExit(f"--noise-draws {args.noise_draws} is below 1")
    if str(args.noise_mode) == "rotate" and n_draws < 2:
        raise SystemExit(
            "--noise-mode rotate with --noise-draws 1 rotates over a single "
            "realisation, which is --noise-mode fixed under another name. Pass "
            "--noise-draws 2 or more, or drop the flag.")
    if int(args.shell_noise_draws) > 0:
        if float(args.w_shell) <= 0.0:
            raise SystemExit(
                "--shell-noise-draws averages the SHELL curve over draws, but "
                "--w-shell is 0 so there is no shell term to average.")
        if n_draws < 2:
            raise SystemExit(
                f"--shell-noise-draws {args.shell_noise_draws} needs more than "
                "one realisation to average over; pass --noise-draws 2 or more.")
        if int(args.shell_noise_draws) > n_draws - 1:
            raise SystemExit(
                f"--shell-noise-draws {args.shell_noise_draws} exceeds the "
                f"{n_draws - 1} other training draws available at "
                f"--noise-draws {n_draws}.")
    if bool(args.soft_membership):
        f = int(args.soft_factor)
        if f < 1 or TILE % f:
            raise SystemExit(
                f"--soft-factor {f} does not divide the tile edge {TILE}, so a "
                "coarse cell would straddle two tiles and its pool would "
                "include rows this run cannot move.")
        if f != int(geom.scale_factor):
            print(f"  WARNING: --soft-factor {f} is not the run's "
                  f"super-resolution factor {geom.scale_factor}. The prior's "
                  "cell is then not an LR cell, which is a different "
                  "experiment from the one the flag describes.", flush=True)
        print(f"  soft membership ON: prior={args.soft_prior} "
              f"lambda={args.soft_lambda} factor={f} "
              f"(cell = {f ** 3} HR sites, "
              f"{BOXSIZE / (NG_HR / f):.4f} Mpc/h a side), "
              f"pool ratio={args.soft_pool_ratio}, window={args.soft_sigma}. "
              f"Watch soft_mass_ratio (~1.0) and soft_hr_fraction.", flush=True)

    _seeds = draw_seeds(args)
    _fresh = fresh_draw_index(args)
    print(f"  noise: {n_draws} training draw(s) (mode {args.noise_mode}), "
          f"root seeds {_seeds[:n_draws]}"
          + (f"; draw {_fresh} (seed {_seeds[_fresh]}) RESERVED for the fresh "
             "eval and never trained on" if _fresh is not None else "")
          + (f"; shell averaged over {1 + int(args.shell_noise_draws)} draws"
             if int(args.shell_noise_draws) > 0 else ""), flush=True)
    if int(args.shell_quantiles) > 0:
        print(f"  shell reduced to {args.shell_quantiles} equal-count quantile "
              "bins (full rank pairing is --shell-quantiles 0)", flush=True)

    train_boxes = [b for b in str(args.train_boxes).replace(",", " ").split() if b]
    hold_boxes = [b for b in str(args.holdout_boxes).replace(",", " ").split() if b]
    hold_keys = [k for k in str(args.holdout_hosts).replace(",", " ").split() if k]

    # The frozen reference is built WITHOUT conditioning columns and is fed no
    # channels. That is the definition of the control: it is the deployed SR2,
    # and every reference in this pipeline (`base`, the guards, the gate's
    # `frozen` field) is measured against it. Because `cond0` is zero-initialised
    # the tuned model agrees with it bit-for-bit at step 0 anyway -- which is
    # what `test_cond_zero_init_matches_frozen_bit_for_bit` pins.
    frozen = load_controlled_generator(
        model_path_of(cfg), in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor, device=device, eval_mode=True)
    for p in frozen.parameters():
        p.requires_grad_(False)

    # --- the conditioning channels, resolved and LOADED before the pool -----
    # Loading here (rather than lazily inside the first `build_tiles`) means a
    # missing or stale cached field fails in seconds, not after the twenty-odd
    # minutes a pool build costs.
    ccfg = cond_config_of(args)
    if ccfg is not None:
        print(f"\nconditioning: {ccfg.n_chan} channel(s) {list(ccfg.channels)}, "
              f"mode {ccfg.mode}, gate >={ccfg.min_hr_num_p} HR particles, "
              f"r_max {ccfg.r_max_rvir} R_vir", flush=True)
        for _b in train_boxes + hold_boxes:
            load_cond_field(_b, ccfg, args)
    else:
        print("\nconditioning: none (--cond-channels empty); this is the "
              "unconditioned control", flush=True)

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
    report_population(all_tasks, args)
    reloc_agg = report_relocation(all_tasks, args)
    report_pair_cost(train_tasks + hold_tasks, args)

    # --- membership-solve dump (visualisation only) -------------------------
    # Reuses the exact pool this arm trained on, runs ZERO training steps, solves
    # the membership on a saved field with the per-sweep recorder on, and writes
    # a single npz a CPU viz job renders. Fits the "figures redraw from saved
    # artefacts" rule: the expensive pool build happens here, the plotting reads
    # the npz. It exits before training, so it is safe to point at a real arm.
    if getattr(args, "dump_membership_npz", None):
        _dump_membership_solve(args, mcfg, scfg, pscfg, by_key, all_tasks)
        # "ok" is what main() turns into the process exit code; without it the
        # dump exits 1 despite writing the npz, and the afterok viz job strands
        # on DependencyNeverSatisfied.
        return {"dumped": args.dump_membership_npz, "ok": True}

    out_dir = paths.subdir("member_gather",
                           f"{args.rung}{args.label}", create=True)
    pool_summary = {
        "train": tr_sum, "holdout": ho_sum,
        "rejected": split.rejected,
        "train_hosts": [_host_row(t) for t in train_tasks],
        "holdout_hosts": [_host_row(t) for t in hold_tasks],
        "relocation": reloc_agg,
        "relocation_by_host": {t.sel.key: (t.report.get("relocate") or {})
                               for t in all_tasks} if reloc_agg else {},
    }
    write_json(out_dir / "pool.json", pool_summary)
    print(f"  wrote {out_dir / 'pool.json'}", flush=True)
    # The config, NOW, not only in the end-of-run summary.json. The early
    # MG_GATE_AT_STEPS gate replays the run's config to rebuild its pool, and
    # summary.json does not exist until the last step -- so every early gate
    # died in export on a missing file (job 84389, all_blocks_radparent).
    # `args` is final here: its one mutation (pool_only) is above.
    write_json(out_dir / "run_config.json", {"config": vars(args)})
    # The pool is the memory-heavy stage (every host's tiles and set tables are
    # held on the CPU, one set per NOISE DRAW), so this is the number that sizes
    # --mem. A pool-only shakeout exists to produce exactly this.
    rss_pool = report_rss(f"after the pool build ({len(all_tasks)} hosts, "
                          f"{len(draw_seeds(args))} draw(s))")
    pool_summary["peak_rss_gb_after_pool"] = rss_pool

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
    if float(args.w_knn_unsup) > 0 and not unsup_train:
        raise SystemExit(
            "--w-knn-unsup > 0 charges the kNN term on the unsupervised tile pool, "
            "which is empty -- set --unsup-tiles-per-box. Without it the flag "
            "would silently train nothing.")
    knn_cfg = knn_config_from_args(args)
    knn_cls = knn_class_ids(args)
    if float(args.w_knn_unsup) > 0 and knn_cfg.phase_lambda <= 0:
        # Round 5b: the position-only statistic ranks the working arms at
        # rho -0.23 (p 0.32). Training on it optimises a statistic with no
        # measured ranking power in the regime a fine-tune operates in.
        print("WARNING: --w-knn-unsup > 0 with --knn-phase-lambda 0 trains on "
              "the POSITION-only statistic, which round 5b measured as unable "
              "to rank the working arms. 4000 is the measured pass.", flush=True)
    if bool(getattr(args, "knn_classes_diagnostic_only", False)):
        # The catalog-free arm: the labels are still measured and reported, they
        # just do not reach the gradient. knn_cls is emptied AFTER attach time.
        if not knn_cls:
            raise SystemExit(
                "--knn-classes-diagnostic-only needs --knn-classes to name the "
                "classes to REPORT; with neither, nothing changes and the flag "
                "would be a no-op that reads like an arm.")
        print(f"kNN: classes {[OWNER_CLASSES[i] for i in knn_cls]} are "
              "DIAGNOSTIC ONLY -- no HR label reaches the loss.", flush=True)
        knn_loss_cls: Tuple[int, ...] = ()
    else:
        knn_loss_cls = knn_cls
    if float(args.w_knn_unsup) > 0 and not knn_loss_cls:
        if knn_cfg.strata > 1 and knn_cfg.strata_charge:
            print(f"kNN: catalog-free arm -- {knn_cfg.strata} density strata, "
                  f"charging {list(knn_cfg.strata_charge)} on each field's own "
                  f"r_{knn_cfg.strata_rank or min(knn_cfg.ks)}.", flush=True)
        elif knn_cfg.strata > 1:
            print("WARNING: --knn-strata > 1 with no --knn-strata-charge is the "
                  "POOLED loss in finer bins -- MEASURED, partitioning alone "
                  "does not reweight the dense tail. Set --knn-strata-charge.",
                  flush=True)
        else:
            print("WARNING: --w-knn-unsup > 0 with no --knn-classes charges the "
                  "POOLED form, which round 5b measured as ranking the working "
                  "arms BACKWARDS (hosts dominate it).", flush=True)
    if float(args.w_knn_unsup) > 0 and max(knn_cfg.ks) < 64:
        # Not refused -- the user may be testing exactly this -- but never silent.
        print(f"WARNING: --knn-ks max {max(knn_cfg.ks)} < 64. knn_cdf's measured "
              "rule: ranks below the object's particle count reach zero loss by "
              "FRAGMENTING (largest FOF 13 of 150 at k<=8). The deficit is at "
              "50-200p.", flush=True)

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
        scale_factor=geom.scale_factor, device=device, eval_mode=False,
        cond_chan=(ccfg.n_chan if ccfg is not None else 0))
    lrs = {g: v * args.lr_scale for g, v in
           {"proj_noise": 1e-5, "fine": 3e-6, "middle": 1e-6, "coarse": 3e-7,
            "cond": float(args.cond_lr)}.items()}
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
    if float(args.w_knn_unsup) > 0 or (bool(args.knn_measure) and unsup_train):
        attach_knn_neighbours(list(unsup_train) + list(unsup_hold), cfg, args, kw,
                              knn_cfg)
        attach_knn_classes(list(unsup_train) + list(unsup_hold), args)

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
        """High-k on the unsupervised pools, the box-wide guard's own measure.

        Plus the kNN-distance ratios under ``knn`` whenever the term is on or
        ``--knn-measure`` asks for them, so an arm run WITHOUT the term can still
        be read on the statistic an arm with it is trying to move.
        """
        out = {}
        for side, pool in (("train", unsup_train), ("holdout", unsup_hold)):
            block = eval_unsup_highk(model, pool, geom, device, args)
            if block is not None and (float(args.w_knn_unsup) > 0
                                      or bool(args.knn_measure)):
                block["knn"] = eval_unsup_knn(model, pool, geom, device, args,
                                              kw, knn_cfg,
                                              class_ids=knn_cls)
            out[side] = block
        return out

    # --- the fresh-noise eval (--fresh-noise-eval) --------------------------
    # The 2x2. `train` and `holdout` are measured on the draw the run trains on;
    # these two are the SAME pools under a reserved draw no step has ever seen.
    #
    # Reading it: if `train_fresh` falls back to `holdout`, the train/holdout gap
    # was the fixed noise pattern being memorised, and the objective carries no
    # more transferable signal than the holdout curve already shows. If
    # `train_fresh` tracks `train`, the model learned something from the LR
    # field and the gap is about boxes and hosts instead -- a completely
    # different problem with a completely different fix.
    fresh = fresh_draw_index(args)

    def fresh_block(tag_hist=None):
        if fresh is None:
            return None
        return {
            "train_fresh": evaluate_pool(model, train_tasks, geom, kw, mcfg,
                                         args, device, frozen, "train_fresh",
                                         term_scale, draw=fresh),
            "holdout_fresh": evaluate_pool(model, hold_tasks, geom, kw, mcfg,
                                           args, device, frozen,
                                           "holdout_fresh", term_scale,
                                           draw=fresh),
        }

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
    row0.update(fresh_block() or {})
    hist.append(row0)
    metrics_p.write_text(json.dumps(row0) + "\n")
    log_wandb(row0, use_wandb)
    print(f"\nfrozen start: train bound_frac {row0['train']['bound_hard']:.3f}, "
          f"2T/|W| {row0['train']['virial']:.0f}, "
          f"high-k {row0['train']['highk_ratio']:.2f}x HR", flush=True)

    # The starting pedestal, and the splits, BEFORE a single step. An arm that
    # begins with the supervised sets already sitting in more non-member mass
    # than HR has is starting inside the failure the isolation term is for, and
    # that is worth knowing at step 0 rather than from a gate ten hours later.
    for side in ("train", "holdout"):
        blk = row0[side]
        print(f"  {side:<8} iso {blk.get('iso_ratio', float('nan')):.2f}x HR "
              f"({100 * blk.get('iso_above_hr', float('nan')):.0f}% of sets "
              f"above it)")
        for split in ("by_population", "by_env"):
            parts = [f"{k} n={int(v['n'])} bound {v['bound_hard']:.3f} "
                     f"iso {v['iso_ratio']:.2f}x"
                     for k, v in (blk.get(split) or {}).items()
                     if v.get("n", 0) > 0]
            if parts:
                print(f"    {split:<14} " + " | ".join(parts))
    print(flush=True)

    order = list(range(len(train_tasks)))
    gate_at = set(gate_steps(args))
    hps = max(1, int(args.hosts_per_step))
    sps = int(args.sets_per_step)
    hsps = int(args.host_sets_per_step)
    set_rng = random.Random(int(args.seed) + 1)
    draw_rng = random.Random(int(args.seed) + 3)
    n_train_draws = max(1, int(args.noise_draws))
    n_shell = max(0, int(args.shell_noise_draws))

    def shell_draws_for(primary: int) -> Tuple[int, ...]:
        """The EXTRA draws the noise-averaged shell charges alongside ``primary``.

        Drawn from the training draws only -- averaging over the reserved fresh
        draw would train on the very realisation the fresh eval exists to hold
        out, and the 2x2 would say nothing.
        """
        if n_shell <= 0 or n_train_draws <= 1:
            return ()
        others = [d for d in range(n_train_draws) if d != int(primary)]
        return tuple(others[:n_shell])

    cursor = len(order)
    t0 = time.time()
    n_nonfinite = 0
    n_consec_nonfinite = 0

    for step in range(1, int(args.steps) + 1):
        w_adv = adv_weight_at(step, args) if critic is not None else 0.0
        opt.zero_grad(set_to_none=True)
        acc: List[Dict] = []
        adv_pairs: List = []
        step_hosts: List[str] = []
        step_losses: List[float] = []
        for _ in range(hps):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            task = train_tasks[order[cursor]]
            cursor += 1
            # WHICH noise realisation this host sees this step. Rotating means
            # a pattern that only fits one draw buys nothing on average, so the
            # only way left to lower the loss is through the LR field -- which
            # is the whole point (see `draw_seeds`). `fixed` keeps every arm
            # before 2026-09-09 byte-identical.
            draw = (draw_rng.randrange(n_train_draws)
                    if str(args.noise_mode) == "rotate" else 0)
            loss, row = host_loss(model, task, geom, kw, mcfg, args, device,
                                  frozen, sample_sets(task, sps, set_rng),
                                  term_scale=term_scale, critic=critic,
                                  critic_norm=critic_norm, w_adv=w_adv,
                                  host_set_indices=sample_host_sets(
                                      task, hsps, set_rng),
                                  draw=draw,
                                  shell_draws=shell_draws_for(draw))
            step_hosts.append(str(task.key))
            step_losses.append(float(loss.detach()))
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
        # The box-wide kNN-distance support. Its own minibatch and its own
        # backward, for the same reason as the hinge: the tape is freed before
        # the next piece, and Adam sees the sum.
        knn_pen = 0.0
        if unsup_train and float(args.w_knn_unsup) > 0:
            m = min(int(args.unsup_tiles_per_step), len(unsup_train))
            batch = set_rng.sample(unsup_train, m) if m > 0 else unsup_train
            # knn_loss_cls, not knn_cls: with --knn-classes-diagnostic-only
            # this is empty and the eval above still reports per class, which is
            # the whole shape of the catalog-free arm -- measure the label, do
            # not train on it.
            pen, _ = _unsup_knn(model, batch, geom, device, args, kw, knn_cfg,
                                class_ids=knn_loss_cls)
            (float(args.w_knn_unsup) * pen).backward()
            knn_pen = float(pen.detach())
        # ---- non-finite guard -------------------------------------------------
        # Job 84238 (radparent arm) had every weight NaN by its first eval at
        # step 250, with nothing logged per step: one non-finite gradient fed to
        # Adam poisons every parameter for good. Skip that update instead, say
        # WHICH host and term produced it, and stop if it keeps happening.
        _bad_p = [n for n, p in model.named_parameters()
                  if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        _bad_l = [h for h, l in zip(step_hosts, step_losses)
                  if not np.isfinite(l)]
        if _bad_p or _bad_l:
            n_nonfinite += 1
            n_consec_nonfinite += 1
            if n_nonfinite <= 20:
                print(f"  !! NON-FINITE step {step}: hosts {step_hosts} "
                      f"losses {[f'{l:.4g}' for l in step_losses]} | "
                      f"{len(_bad_p)} params with non-finite grad "
                      f"(first: {_bad_p[:3]}) -- update SKIPPED", flush=True)
                for h, r in zip(step_hosts, acc):
                    odd = {k: v for k, v in r.items()
                           if isinstance(v, float) and not np.isfinite(v)}
                    keep = {k: r[k] for k in sorted(r)
                            if (k.startswith("term_raw_")
                                or k.startswith("soft_"))
                            and isinstance(r[k], (int, float))}
                    print(f"     {h}: non-finite {odd} | {keep}", flush=True)
            opt.zero_grad(set_to_none=True)
            if n_consec_nonfinite >= int(args.max_nonfinite_steps):
                raise SystemExit(
                    f"{n_consec_nonfinite} consecutive non-finite steps (last "
                    f"{step}, {n_nonfinite} total) -- stopping rather than "
                    "training on skipped updates. Read the NON-FINITE lines.")
        else:
            n_consec_nonfinite = 0
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

        # --- the early-gate snapshot -------------------------------------
        # An IMMUTABLE copy, written at the requested step and never touched
        # again, so a Rockstar gate can run CONCURRENTLY with the rest of
        # training instead of waiting for it. `tuned.pt` cannot serve this:
        # it is rewritten every eval, and a gate reading it would be racing
        # the trainer for ~138 s of export.
        #
        # Why this is not a shortcut that costs anything: MEASURED on the
        # three finished arms, the held-out objective is essentially done by
        # step 2000 -- smallhalo bound_frac 0.0874 there against 0.0877 at
        # 8000 (99.7%), selfhostguard 0.0950 against 0.0976 (97%), self
        # 0.1202 against 0.1395 (86%). What keeps moving after 2000 is the
        # high-k EXCESS -- smallhalo 1.52x -> 3.15x, self 1.92 -> 3.87,
        # selfhostguard 2.02 -> 4.79 -- which is damage accumulating, not
        # the objective improving. Training still runs to --steps; this only
        # buys a read at hour ~2.7 instead of hour ~10.3.
        if step in gate_at:
            snap = out_dir / f"tuned_step{step}.pt"
            save_atomic({
                "model": {n: p.detach().cpu()
                          for n, p in model.state_dict().items()},
                "rung": args.rung,
                "trained_names": names,
                "cond": (ccfg.to_dict() if ccfg is not None else None),
                "step": int(step),
                "partial": True,
                "snapshot": True,
            }, snap)
            print(f"  step {step:6d}  early-gate snapshot -> {snap.name} "
                  "(training continues)", flush=True)

        if step % int(args.eval_every) == 0 or step == int(args.steps):
            row = {
                "step": step,
                "n_nonfinite_steps": int(n_nonfinite),
                "wall_s": time.time() - t0,
                "batch_gather": float(np.mean([r["gather"] for r in acc])),
                "batch_unsup_pen": unsup_pen,
                "batch_unsup_knn": knn_pen,
                "train": evaluate_pool(model, train_tasks, geom, kw, mcfg, args,
                                       device, frozen, "train", term_scale),
                "holdout": evaluate_pool(model, hold_tasks, geom, kw, mcfg, args,
                                         device, frozen, "holdout", term_scale),
                "unsup": unsup_block(),
            }
            row.update(fresh_block() or {})
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
                save_atomic({
                    "model": {n: p.detach().cpu()
                              for n, p in model.state_dict().items()},
                    "rung": args.rung,
                    "trained_names": names,
                    "cond": (ccfg.to_dict() if ccfg is not None else None),
                    "step": int(step),
                    "partial": True,
                }, out_dir / "tuned.pt")
                if critic is not None:
                    save_atomic({"critic": critic.state_dict(),
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
            # The pedestal, held out. The one number that says whether the
            # objects being built are being built INTO a pile -- the state
            # that makes Rockstar file them as somebody's subhalos, and the
            # thing no arm before this could see mid-run.
            iso = f" iso {ho.get('iso_ratio', float('nan')):.2f}x"
            print(f"  step {step:6d}  loss {row['batch_gather']:.4f}  "
                  f"| train bound {tr['bound_hard']:.3f} "
                  f"2T/|W| {tr['virial']:.1f} dx {tr['centre_offset_radii']:.2f}r "
                  f"| hold bound {ho.get('bound_hard', float('nan')):.3f} "
                  f"dx {ho.get('centre_offset_radii', float('nan')):.2f}r{iso} "
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
    save_atomic({
        "model": {n: p.detach().cpu() for n, p in model.state_dict().items()},
        "rung": args.rung,
        "trained_names": names,
        # The conditioning config travels IN the checkpoint, not only in
        # summary.json: a loader has to size `cond_chan` before it can load the
        # state dict at all, so a checkpoint that cannot say how wide it is is
        # one that gets loaded at the wrong width -- or, with `cond0` simply
        # absent, silently loaded as the unconditioned control.
        "cond": (ccfg.to_dict() if ccfg is not None else None),
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
                # Per host, because the shell is a per-host question: a cluster
                # whose sets all sit at 0.4 of their true radius is a different
                # failure from one where the median hides two populations.
                "radius_ratio_frozen", "radius_ratio_final",
                "shell_resid_frozen", "shell_resid_final",
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
                    h.get("centre_offset_radii"),
                    z.get("radius_ratio"), h.get("radius_ratio"),
                    z.get("shell_resid"), h.get("shell_resid"),
                    h.get("highk_ratio"), h.get("low_k"),
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
    report_rss("at the end of the run")
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
    # --- the LR-host conditioning channel ------------------------------------
    # Default EMPTY, which is the unconditioned control and is byte-identical to
    # every run recorded before this option existed. See
    # `cosmo_sr.features.lr_host_channel` for what the channels are and
    # `radial-profile-is-flat-in-host-mass` for why a local operator needs them.
    ap.add_argument(
        "--cond-channels", default="",
        help="comma list of LR-host conditioning channels to feed the "
             "generator, e.g. 'r_over_rvir,log_mhost,valid'. EMPTY (default) "
             "is the unconditioned control. The stack is built by "
             "scripts/features/build_lr_host_channel.py and must already exist "
             "for every box in --train-boxes and --holdout-boxes. Order is "
             "significant and round-trips into the Rockstar gate's replay.")
    ap.add_argument("--cond-mode", default="derived",
                    choices=("derived", "membership"),
                    help="how a Lagrangian site is assigned to an LR host. "
                         "'derived' minimises r/R_vir and is defined "
                         "everywhere; 'membership' uses the Rockstar "
                         ".particles table, MEASURED as covered out to 2 R_vir "
                         "(0.79-0.81 correct at 1-1.5 R_vir) and partial "
                         "beyond. The two are close; this is a real A/B.")
    ap.add_argument("--cond-min-hr-num-p", type=int, default=20000,
                    help="LR hosts below this many HR-equivalent particles are "
                         "not conditioned on. 20000 is where "
                         "lr_host_fidelity measured the LR catalog channel "
                         "stops working (57%% of region subhalos survive it).")
    ap.add_argument("--cond-r-max-rvir", type=float, default=3.0,
                    help="sites beyond this many R_vir of every gated host are "
                         "marked invalid (the `valid` channel goes to 0)")
    ap.add_argument("--cond-log-mhost-ref", type=float, default=13.0)
    ap.add_argument("--cond-log-mhost-scale", type=float, default=1.0)
    ap.add_argument("--cond-root", default="",
                    help="directory holding <box>.npz conditioning fields; "
                         "empty means $DMSR_REWARD_ROOT/lr_host_channel")
    ap.add_argument(
        "--cond-lr", type=float, default=1e-5,
        help="learning rate for the conditioning columns, before --lr-scale. "
             "Deliberately the largest group: cond0 starts at exactly ZERO and "
             "has no pretrained function to protect, so a coarse-sized 3e-7 "
             "would leave the channel wired in and inert for the whole run.")
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
    ap.add_argument("--centre-mode",
                    choices=("full", "radial", "self", "self_tan",
                             "self_tan_parent"),
                    default="full",
                    help="what the centre term charges for. full: the whole "
                         "offset to the reachable HR centroid. radial: only its "
                         "clustercentric projection -- the 62.9%% of the offset "
                         "variance that is a signed infall deficit, dropping "
                         "the transverse part LR does not determine. self: "
                         "re-anchor to the set's own frozen centroid. self_tan: "
                         "the same anchor with the radial direction projected "
                         "OUT, which is what --w-radius should be paired with "
                         "-- `self` pins the very radius the shell term is "
                         "trying to move, and at equal weights the two settle "
                         "at exactly half the correction.")
    ap.add_argument("--centre-dead-zone", type=float, default=0.0,
                    help="search radii inside which the centre term costs "
                         "nothing. The gate is a threshold at 1 radius, so cost "
                         "paid at 0.1 buys nothing there. 0 = the 72/154 term.")
    ap.add_argument("--centre-huber-radii", type=float, default=0.0,
                    help="search radii beyond which the centre term is linear. "
                         "Frozen sets sit a median 5.6 radii out, where a "
                         "quadratic lets the hopeless sets own the gradient.")
    # --- the shell: right distance from the host, any direction -------------
    ap.add_argument("--w-radius", type=float, default=0.0,
                    help="weight on the set's host-centric RADIUS matching the "
                         "truth, with the direction left free. 0 = off, which "
                         "is every arm so far. Targets the `self` arm's radial "
                         "collapse: on held-out set9 its subhalos sit at a "
                         "median 0.35 host R_vir against HR's 1.13, 6496 inside "
                         "0.25 R_vir where HR has 593. The radius is 60%% of "
                         "the frozen centre offset by magnitude and the half "
                         "that is a rule (58%% of sets too close); the "
                         "transverse half is noise no input predicts. Pair with "
                         "--centre-mode self_tan.")
    # --- host-centric local temperature (cosmo_sr.features.host_temperature)
    ap.add_argument(
        "--w-temp", type=float, default=0.0,
        help="weight on the host-centric LOCAL TEMPERATURE profile: per r/R_vir "
             "bin, the mean log k-NN velocity dispersion around each training "
             "host, two-sided against HR's value on the same tiles. 0 is off and "
             "every arm before 2026-10-15. Velocity-only (neighbours detached). "
             "Measured: every tuned arm's core is 0.68-0.75x HR (0.34x in massive "
             "hosts) and Rockstar fragments it 12-19x per structure; imposing HR's "
             "profile took SUBFIND core satellites 672 -> 292 (HR 252).")
    ap.add_argument("--temp-max-x", type=float, default=3.0,
                    help="charge bins up to this r/R_vir; 0.5 = the core-only arm "
                         "(the half the velocity oracle validated)")
    ap.add_argument("--temp-min-x", type=float, default=0.0)
    ap.add_argument("--temp-k", type=int, default=32)
    ap.add_argument("--temp-queries", type=int, default=256,
                    help="candidate queries per r/R_vir bin per host per step")
    ap.add_argument("--temp-target-queries", type=int, default=2048,
                    help="HR queries per bin for the (one-off) target profile")
    ap.add_argument("--temp-min-queries", type=int, default=32)
    ap.add_argument("--temp-measure", action="store_true",
                    help="build and LOG the temperature profile at --w-temp 0")
    ap.add_argument(
        "--w-isolation", type=float, default=0.0,
        help="weight on the local background OCCUPANCY, hinged above HR: the "
             "non-member mass sitting ON the object, counted on the SAME "
             "particle ids in the candidate and in HR. 0 is off and is every "
             "arm so far. It is the only INTER-set term in the loss, and the "
             "reason it exists is that 'top-level' is an inter-set property: "
             "Rockstar files a halo as a subhalo when some bigger halo's R_vir "
             "contains its centre, which is not a property of its members at "
             "all. Every other term is a moment about the set's own centroid, "
             "so an intra-set loss pointed at top-level HR halos scores full "
             "marks on an object that a pedestal then swallows -- measured, "
             "set9: --population all built 3,374 net objects and Rockstar "
             "filed every one as a subhalo (subs +4,260, top-level -886). "
             "d6 does not close this: it normalises by the background's "
             "DISPERSION, not its density.")
    ap.add_argument("--iso-radius-frac", type=float, default=0.5,
                    help="counting radius as a fraction of the background "
                         "sphere (--bg-radius * the set's HR rms radius). Keep "
                         "it below 1: the background ids are chosen once on the "
                         "frozen field within --bg-radius, so a radius at the "
                         "sampling edge counts an edge. Widen --bg-radius "
                         "rather than this.")
    ap.add_argument("--iso-softness", type=float, default=0.15,
                    help="sigmoid width of the soft count, as a fraction of the "
                         "counting radius.")
    ap.add_argument("--iso-min-count", type=float, default=8.0,
                    help="minimum HR soft count for a set to be CHARGED. Below "
                         "it the ratio is shot noise and the term would be "
                         "charging the run for the background sampler's seed; "
                         "such sets are skipped and counted in n_iso.")
    # --- soft membership: WHICH particles, as a latent variable -------------
    ap.add_argument(
        "--soft-membership", action="store_true",
        help="relax the HR member mask into a solved weight P_ij. HR keeps "
             "supplying the object's REGION (a coarse Lagrangian prior) and "
             "its PHYSICS (the reference, untouched); the CURRENT SR field "
             "decides which particles participate, under a per-particle "
             "budget of one (shared with a background/host category) and a "
             "per-candidate mass target. Off is every arm on disk, and the "
             "hard path stays byte-identical when it is off. Needs "
             "--soft-factor to match the run's LR/HR ratio.")
    ap.add_argument(
        "--soft-lambda", type=float, default=1.0,
        help="KL weight against the prior, and the ONLY knob that "
             "interpolates: lambda -> inf pins P at the prior, and with "
             "--soft-prior hard that limit IS the arm already on disk. The "
             "cost is a squared distance in units of the window's sigma, so "
             "1.0 means one dispersion of phase-space distance is worth one "
             "nat of prior. Watch soft_mass_ratio: a lambda small enough to "
             "make the window unfillable shows up there as a shortfall.")
    ap.add_argument(
        "--soft-prior", choices=["coarse", "hard"], default="coarse",
        help="coarse: every particle of an LR cell gets that cell's member "
             "FRACTION, the original members included and not privileged. "
             "hard: the HR mask is the prior and only lambda opens it -- more "
             "conservative, and the control whose lambda -> inf limit is the "
             "hard arm.")
    ap.add_argument(
        "--soft-factor", type=int, default=8,
        help="HR sites per LR cell side (the run's super-resolution factor). "
             "The prior's cell is factor^3 sites; at 8 that is 512, and one "
             "cell is 1.5625 Mpc/h a side against a 0.15 Mpc/h search radius "
             "-- which is why the prior is the SUPPORT and never the "
             "membership.")
    ap.add_argument(
        "--soft-pool-ratio", type=float, default=3.0,
        help="non-members sampled per member, per cell. The pool is (1+ratio) "
             "times the member count and the estimator is O(pool^2), so this "
             "multiplies the step cost directly: 3.0 is ~16x the hard arm's "
             "pair bill. Cap it with --max-set-particles.")
    ap.add_argument(
        "--soft-cell-dilate", type=int, default=0,
        help="add the 26 neighbouring LR cells to each set's support, at "
             "--soft-prior-eps. 0 is off: the member cells alone already hold "
             "factor^3 sites each.")
    ap.add_argument("--soft-prior-eps", type=float, default=1e-3,
                    help="floor on a prior membership, and the value a dilated "
                         "cell gets. Never 0: that is an infinite KL and a "
                         "particle the solve can never reach.")
    ap.add_argument("--soft-bg-floor", type=float, default=1e-3,
                    help="floor on the BACKGROUND share of a particle's "
                         "budget -- the category that makes the budget a "
                         "constraint rather than a normalisation.")
    ap.add_argument("--soft-steps", type=int, default=3,
                    help="EM sweeps of the membership solve per generator "
                         "step. The solve is O(pool) against an O(pool^2) "
                         "step, so a handful is free.")
    ap.add_argument("--soft-mass-steps", type=int, default=10,
                    help="Sinkhorn iterations of the mass dual per sweep.")
    ap.add_argument("--soft-mass-eta", type=float, default=1.0)
    ap.add_argument("--soft-mass-step-cap", type=float, default=20.0)
    ap.add_argument(
        "--soft-sigma", choices=["adaptive", "hr", "cand"], default="adaptive",
        help="dispersions of the 6-D window. adaptive is max(HR, candidate) "
             "per set: never narrower than the object actually is, tightening "
             "toward HR as it tightens. 'hr' alone selects NOTHING at the "
             "start (the frozen field is ~6x wider and ~20x hotter per set, "
             "which is 400 sigma^2); 'cand' is free to contract onto whatever "
             "is already cold and call that the object.")
    ap.add_argument("--soft-sigma-scale", type=float, default=1.0)

    # --- the host as a background component (the radial-placement arm) ------
    ap.add_argument(
        "--soft-host-profile", choices=["off", "lr", "self", "parent"],
        default="off",
        help="give the membership solve a HOST density profile instead of a "
             "flat background. The flat background plus the n_c/f^3 prior make "
             "the host core the cheapest place for a candidate to meet its "
             "mass target, which is a selection-side gradient pointing INWARD "
             "and is what radial-placement-is-inverted measures (HR 1.13 R_vir "
             "vs 0.30-0.35 tuned). Subtracting the host's log density turns "
             "membership into a log OVERDENSITY test, so a lump must beat the "
             "material that would be at its radius anyway. 'lr' is an analytic "
             "NFW from the LR catalog (inference-honest, and blank for the 43%% "
             "of hosts below the LR gate); 'self' is binned off the candidate "
             "field itself (complete, partly circular). No per-object radial "
             "target is stated -- that family (one-to-one, rank, quantile) has "
             "already failed on held-out data.")
    ap.add_argument(
        "--soft-host-weight", type=float, default=0.0,
        help="beta: how much of the log host density enters the background. 0 "
             "reproduces the flat background EXACTLY (pinned by a test), 1 is "
             "full overdensity. NOT comparable between --soft-host-profile lr "
             "and self: their log-density ranges differ by ~3x, so read the "
             "ladder off soft_host_logrho_p10/p90 in the diagnostics, in nats.")
    ap.add_argument("--soft-host-conc", type=float, default=5.0,
                    help="NFW concentration for --soft-host-profile lr.")
    ap.add_argument(
        "--lr-host-table", default="",
        help="directory of per-box LR host tables for --soft-host-profile lr; "
             "empty means $DMSR_REWARD_ROOT/lr_host_table. Built by "
             "scripts/features/build_lr_host_table.py.")
    ap.add_argument(
        "--lr-host-r-max-rvir", type=float, default=2.0,
        help="a supervised host further than this many R_vir from every gated "
             "LR host is treated as having no LR owner, and falls back to the "
             "flat background. 2.0 is where lr_membership_coverage measured LR "
             "membership stop being COVERED.")
    ap.add_argument("--soft-host-bins", type=int, default=24)
    ap.add_argument("--soft-host-r-max-rvir", type=float, default=4.0)
    ap.add_argument(
        "--soft-radial-weight", type=float, default=0.0,
        help="gamma of the per-set radial prior in the membership solve: each "
             "pool particle pays 0.5*gamma*((r - r_HR)/(width*R_vir))^2, r from "
             "the set's OWN HR parent. 0 = off (bit-identical).")
    ap.add_argument("--soft-radial-width", type=float, default=0.2,
                    help="the radial prior's width, in the parent's R_vir")

    # --- relocation: the arm A / B / C experiment ---------------------------
    ap.add_argument(
        "--relocate", choices=["off", "frozen", "hr"], default="off",
        help="WHICH PARTICLES each supervised group is made of. 'off' is arm "
             "A: HR's owner array, every result recorded in this line so far. "
             "'frozen' is arm B: the group's particles are re-chosen, once and "
             "before training, as compact material that already exists in the "
             "frozen SR field at the host-centric radius the group's frozen "
             "centroid already sits at -- isolating the effect of replacing "
             "the membership. 'hr' is arm C: the same selection at HR's "
             "radius, which adds the placement change on top of B. The HR "
             "targets are IDENTICAL in all three; only the material changes.")
    ap.add_argument(
        "--relocate-dirs", type=int, default=8,
        help="proposed directions per group on the sphere of the requested "
             "radius. The first is always the group's own frozen direction, so "
             "the minimal answer -- the material already there -- is always in "
             "the running and arm B is a small change when it can be.")
    ap.add_argument(
        "--relocate-eps", type=float, default=0.05,
        help="radius tolerance in units of the HOST's R_vir. A group outside "
             "it is KEPT and flagged, never dropped: dropping would make the "
             "three arms supervise different catalogs and their Rockstar gates "
             "incomparable. The count is in the pre-training check.")
    ap.add_argument("--relocate-oversample", type=float, default=4.0,
                    help="neighbours queried per candidate as a multiple of "
                         "the group size; grown when earlier claims starve it.")
    ap.add_argument("--relocate-refine-iters", type=int, default=3,
                    help="local-swap passes. Seeds at the correct radius are "
                         "not enough -- the nearest-particle group around a "
                         "correct seed can have a displaced centroid.")
    ap.add_argument("--relocate-w-vel", type=float, default=0.5,
                    help="velocity coherence in the candidate score, against "
                         "compactness at 1.0.")
    ap.add_argument("--relocate-w-radius", type=float, default=1.0,
                    help="radius error in the candidate score, in units of "
                         "--relocate-eps.")
    ap.add_argument(
        "--relocate-check-only", action="store_true",
        help="build the pool, write the relocation report, and exit 0 without "
             "training. The proposal's 'check this distribution before "
             "training': it is a CPU job and it answers whether the material "
             "exists at the requested radii before a GPU-hour is spent.")

    ap.add_argument(
        "--w-shell", type=float, default=0.0,
        help="weight on the DISTRIBUTION of the sets' host-centric radii: sort "
             "the candidate and target radii of one host, pair by rank (the "
             "1-D optimal transport map) and charge the log ratio. 0 is off. "
             "It is --w-radius's post-mortem: that term charges each set "
             "against ITS OWN target radius, and that address has an "
             "out-of-sample R^2 of 0.022 on the 7,560-set pool, so the "
             "selfshell run memorised it (train err 2.86 -> 1.32, held out "
             "2.68 -> 2.81) and paid 55%% of the abundance for it (host "
             "subhalos 367 -> 167 against HR's 374). Only the multiplicative "
             "part transferred (held-out ratio 0.974 -> 0.987) and this term "
             "asks for that and nothing else: the sort keeps the signed inward "
             "bias intact while halving the magnitude. Pair with --centre-mode "
             "self_tan, and read `shell_resid`, not `radius_err_radii`.")
    ap.add_argument(
        "--shell-dead-zone", type=float, default=0.02,
        help="FRACTIONAL radius error the shell term treats as free, applied "
             "as log1p(x) in log space. 0.02 charges 81%% / 78%% of sets "
             "(train / holdout) against a sorted median residual of 0.079 / "
             "0.064. Do not carry over --radius-dead-zone's reasoning: 0.05 "
             "charges only 61%% / 55%% and eats most of a signal that is half "
             "the size of the one that number was set against.")
    ap.add_argument(
        "--shell-quantiles", type=int, default=0,
        help="reduce the sorted pairing to this many equal-COUNT quantile bins "
             "and charge the bins' mean log radius. 0 (default) is the full "
             "rank-by-rank pairing. Why the dial exists: the full pairing hands "
             "the loss one degree of freedom per set (~134 for a big host), and "
             "MEASURED 2026-09-08 the w_shell=30 arm drove its train residual "
             "7.3x while holdout moved 1.4x. The low-order shape is what a "
             "shared operator can carry; the rank-by-rank wiggle is what it can "
             "only memorise. Try 3-5.")
    ap.add_argument(
        "--shell-eps-mpc-h", type=float, default=1e-2,
        help="floor on both radii before the log, Mpc/h. On the candidate it "
             "is a SOFTENED norm, sqrt(|d|^2 + eps^2), not a clamp: a clamp is "
             "a gradient plateau below the floor and collapse toward the host "
             "centre is the one place this term must not have one.")
    ap.add_argument("--radius-dead-zone", type=float, default=1.0,
                    help="search radii of slack: the radius term costs nothing "
                         "this close to the truth. The ask is a SHELL, not an "
                         "address, and the gate is a threshold at 1 radius. At "
                         "the default 1.0, 79%% of frozen sets are still "
                         "charged (median error 2.80 radii).")
    ap.add_argument("--radius-huber-radii", type=float, default=0.0,
                    help="search radii beyond which the radius term is linear. "
                         "The p90 set is 8.2 radii out, where a quadratic "
                         "charges 67 and the tail owns the batch gradient.")
    ap.add_argument("--min-num-p", type=int, default=200)
    ap.add_argument(
        "--population", choices=("sub", "top", "all"), default="sub",
        help="WHICH halos the member loss is responsible for. 'sub' "
             "(parent_ids >= 0) is every arm recorded so far and encodes an "
             "assumption the held-out gate contradicts: in the edited region HR "
             "resolves 3,775 top-level halos >=200p against base's 3,028, a "
             "747-halo deficit OUTSIDE hosts that no supervised set has ever "
             "pointed at. 'all' selects on size alone and lets the parent split "
             "fall where it falls. Requires --max-num-p (see there).")
    ap.add_argument(
        "--max-num-p", type=int, default=0,
        help="exclusive upper edge of the supervised num_p window; 0 is no "
             "ceiling. REQUIRED with --population top/all: without it the "
             "selection includes the cluster host itself, whose O(N^2) pair sum "
             "is ~100k particles (1e10 pairs against a 200p set's 2e4) and whose "
             "HR reference asks the generator to concentrate a 1e14 halo -- a "
             "different objective, and one --w-host-sets already handles against "
             "the FROZEN reference.")
    ap.add_argument(
        "--env-select", default="any",
        help="WHICH environment the supervised sets come from: 'any' (every arm "
             "so far), or a comma list of inside/near/field -- distance to the "
             "nearest bigger resolved host, in its R_vir. This is the third "
             "selection axis after population and mass, and the set9 census "
             "says it is the one that was missing: of the region's 11,698 "
             "missing objects 6,557 sit in 'near' (the infall region) and the "
             "small-halo arm closed 249 of them, 3.8%%, while OVERSHOOTING HR "
             "in 'field' (subhalos 3,316 against HR's 2,466). 'near' is inside "
             "a cluster's own Lagrangian tiles, so it is reachable by changing "
             "the selection; 'field' would need a wider --n-tiles.")
    ap.add_argument(
        "--env-host-floor", type=int, default=200,
        help="particle floor for a halo to count as a 'resolved host' when "
             "labelling environments. 200 is the census's and the number every "
             "host count quoted in this line uses.")
    ap.add_argument(
        "--set-select", choices=("largest", "random", "stratified"),
        default="largest",
        help="how --max-sets chooses when it binds. 'largest' is the historical "
             "default and is a trap for a mass-window run: the pool is a mass "
             "function, so it keeps the top of it and drops exactly the small "
             "halos the run exists to build. 'stratified' fills log2(n_sites) "
             "bins round-robin.")
    ap.add_argument("--min-purity", type=float, default=0.5)
    ap.add_argument("--min-live-frac", type=float, default=0.5)
    ap.add_argument("--stats-members", default="all", choices=("all", "live"),
                    help="which members carry every set statistic. 'all' is the "
                         "historical hybrid -- live rows PLUS the out-of-tile "
                         "stragglers at frozen coordinates -- and is what every "
                         "arm on disk trained against. 'live' drops the "
                         "stragglers from the reference, the candidate, the "
                         "centre targets and the background radius at once. "
                         "MEASURED (docs/sr2_outskirt_detection.md section 6): "
                         "at a median live fraction of 0.979 the hybrid "
                         "reference sits at 2.18x HR's own r_rms and 1.99x its "
                         "2T/|W|, because r_rms is an RMS and a tail of two "
                         "parts in a hundred at ~2 Mpc/h carries roughly "
                         "three quarters of r_rms^2. That both loosens the "
                         "target and dilutes the gradient on the members the "
                         "run can actually move. Read pure_hr_median against "
                         "hybrid_reference_median in pool.json.")
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
                         "costs. It does not on the scalar: that band is 5.7%% "
                         "of the modes and the mean is unweighted")
    ap.add_argument("--highk-k-max", type=float, default=0.0,
                    help="upper k for the bands; 0 (default) runs to the cube "
                         "corner, matching the scalar's mask. pi/dx = 16.08 "
                         "drops the 48%% of the guard's modes that sit ABOVE "
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
    # --- the kNN-distance support (src/cosmo_sr/features/knn_cdf.py) --------
    # Default OFF (--w-knn-unsup 0). Grid-free, catalog-free, on the same
    # unsupervised pool as the high-k hinge. Read the module docstring before
    # changing --knn-ks: the largest rank decides halo vs fragments.
    ap.add_argument("--w-knn-unsup", type=float, default=0.0,
                    help="weight on the kNN log-distance distribution match on the "
                         "unsupervised tiles. 0 (default) is off; requires "
                         "--unsup-tiles-per-box > 0")
    ap.add_argument("--knn-ks", type=int, nargs="+", default=[8, 32, 64, 128],
                    help="neighbour ranks. Keep max >= the particle count of the "
                         "smallest object to build (measured: k<=8 fragments)")
    ap.add_argument("--knn-queries", type=int, default=1024,
                    help="query particles per tile per evaluation")
    ap.add_argument("--knn-quantiles", type=int, default=64,
                    help="equal-count bins of the sorted pairing; 0 = full pairing")
    ap.add_argument("--knn-dead-zone", type=float, default=0.02,
                    help="fractional r_k error that costs nothing")
    ap.add_argument("--knn-buffer-cells", type=float, default=0.0,
                    help="Eulerian buffer inside the query cube. 0 by default "
                         "because the neighbour margin below does that job; job "
                         "49259 measured that a buffer ALONE cannot (p95 +62-80%% "
                         "at every buffer to 16 cells)")
    ap.add_argument("--knn-nbr-margin-cells", type=float, default=8.0,
                    help="offer other tiles' particles within this many cells "
                         "beyond the tile as constant candidates (frozen box for "
                         "the candidate, HR box for the reference). 0 = own tile "
                         "only. SET FROM knn_cdf_split_diagnostic.py")
    ap.add_argument("--knn-centre", choices=("centroid", "lagrangian"),
                    default="centroid",
                    help="query cube on the tile particles' circular-mean centroid "
                         "(default) or on the tile's Lagrangian centre (round 1)")
    ap.add_argument("--knn-phase-lambda", type=float, default=0.0,
                    help="km/s per Mpc/h; > 0 measures kNN in 6D on (x, v/lambda). "
                         "MEASURED (round 5b): 4000 ranks the working set9 arms at "
                         "rho -0.65 vs -0.23 for position only. 0 = rounds 1-4")
    ap.add_argument("--knn-classes", nargs="*", default=[],
                    help=f"HR owner classes to charge the term on, from "
                         f"{list(OWNER_CLASSES)}; empty = the pooled form. "
                         "MEASURED (round 5b): hosts belong in the MONITOR, not "
                         "the loss -- the working arms gain subhalos by "
                         "over-concentrating hosts in phase space (rho +0.91)")
    ap.add_argument("--knn-strata", type=int, default=0,
                    help="equal-count density strata, computed by each field "
                         "from its OWN r_k -- the catalog-free stand-in for "
                         "--knn-classes. 0/1 = pooled. Useless on its own: see "
                         "--knn-strata-charge")
    ap.add_argument("--knn-strata-rank", type=int, default=0,
                    help="neighbour rank defining the strata; must be in "
                         "--knn-ks. 0 = min(ks)")
    ap.add_argument("--knn-strata-charge", nargs="*", default=[],
                    help="which strata reach the loss, dense-first (0 is "
                         "densest), e.g. `0` or `0,1`. Empty charges all, which "
                         "is MEASURED to be the pooled loss in finer bins -- "
                         "partitioning is not reweighting")
    ap.add_argument("--knn-classes-diagnostic-only", action="store_true",
                    help="keep --knn-classes measured and REPORTED but out of "
                         "the gradient: the arm in which no Rockstar label "
                         "reaches the loss. Pair with --knn-strata*")
    ap.add_argument("--knn-one-sided", action="store_true",
                    help="charge only under-clumping (default: two-sided)")
    ap.add_argument("--knn-measure", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="report unsup kNN ratios at every eval even at weight 0")
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
    ap.add_argument("--guard-ref", default="frozen", choices=("frozen", "hr"),
                    help="what the LR-scale anchor and vel_rms_ratio are "
                         "measured AGAINST. 'frozen' is every arm on disk: in "
                         "a fine-tune the candidate STARTS as the frozen field, "
                         "so the anchor is a drift detector. 'hr' is for "
                         "train_sr2_scratch.py, where there is no drift to "
                         "detect because the generator was never there.")
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
    ap.add_argument("--detect-anomaly", action="store_true",
                    help="debug: torch autograd anomaly detection")
    ap.add_argument(
        "--max-nonfinite-steps", type=int, default=25,
        help="stop after this many CONSECUTIVE steps whose loss or gradient "
             "was non-finite (each is skipped and logged, not applied)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument(
        "--gate-at-steps", default="",
        help="comma list of steps at which to freeze an IMMUTABLE "
             "tuned_step<N>.pt, so a Rockstar gate can run CONCURRENTLY with "
             "the rest of training. Empty (default) writes none. It costs the "
             "run nothing -- training still goes to --steps -- and it is not a "
             "shortcut that gives up the arm's potential: MEASURED on three "
             "finished arms, the held-out objective is ~97-100%% of its final "
             "value by step 2000 (smallhalo bound_frac 0.0874 there vs 0.0877 "
             "at 8000), and what keeps moving afterwards is the high-k EXCESS "
             "(1.52x -> 3.15x), i.e. damage, not gain. The gate itself is ~19 "
             "min, 3%% of a 10 h train.")
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0)
    # --- the noise realisation (docs/sr2_member_gather_training.md) ----------
    ap.add_argument(
        "--noise-draws", type=int, default=1,
        help="how many NOISE realisations per tile the run may train on. 1 "
             "(default) is the single fixed draw every arm before 2026-09-09 "
             "used. Each extra draw costs one frozen forward and one full "
             "reference build per host at pool time, and holds its own set "
             "table (centre_self and the hybrid reference are functions of the "
             "frozen field, so a draw that borrowed draw 0's anchor would "
             "charge the run for realisation scatter).")
    ap.add_argument(
        "--noise-mode", default="fixed", choices=("fixed", "rotate"),
        help="'fixed' always uses draw 0. 'rotate' picks one of the "
             "--noise-draws per host per step, which removes the fixed noise "
             "pattern as a memorisable input channel: MEASURED 2026-09-08, the "
             "w_shell=30 arm drove its train residual 7.3x while holdout moved "
             "1.4x, and a constant (LR, noise) pair for 8000 steps is the "
             "obvious way that happens.")
    ap.add_argument(
        "--fresh-noise-eval", action="store_true",
        help="reserve ONE extra draw that no step ever trains on, and evaluate "
             "both pools under it as 'train_fresh' / 'holdout_fresh'. This is "
             "the 2x2 over (seen box, seen noise) that says whether the "
             "train/holdout gap is memorised noise or a box/host gap.")
    ap.add_argument(
        "--shell-noise-draws", type=int, default=0,
        help="charge the soft shell on the MEAN reduced radius curve over this "
             "many extra draws instead of on the primary draw alone. "
             "Mean-then-charge, not charge-then-mean: the first asks for a "
             "distribution no single realisation can fake, the second is just a "
             "lower-variance version of the same per-realisation objective. "
             "Only the radii are recomputed on the extra draws, never the N^2 "
             "statistics. Requires --noise-draws > 1.")
    ap.add_argument(
        "--dump-membership-npz", default="",
        help="visualisation only: build the pool, solve one host's membership on "
             "--dump-membership-field with the per-sweep recorder on, write the "
             "snapshots to this npz, and EXIT before training. Needs "
             "--soft-membership on.")
    ap.add_argument(
        "--dump-membership-field", default="",
        help="the full-box (6, ng, ng, ng) SR field to solve the membership on "
             "(e.g. an arm's saved flow_rockstar/fields/...npy). Defaults to the "
             "frozen field is NOT assumed -- pass the arm's own tuned field to "
             "see the converged solve.")
    ap.add_argument(
        "--dump-membership-host", default="",
        help="which pooled host key to dump; default is the first held-out host.")
    ap.add_argument("--device", default="")
    ap.add_argument("--no-wandb", action="store_true",
                    help="disable Weights & Biases logging for this run "
                         "(overrides the config's wandb.mode)")
    ap.add_argument("--wandb-mode", default="",
                    help="override wandb.mode (online/offline/disabled)")
    return ap


def validate(args) -> None:
    """Refusals that are always a configuration error, checked before the pool.

    Building the pool is one 537 MB owner-array load and a CSR inversion per box
    -- minutes of a GPU allocation -- so a config that cannot mean what it says
    is refused here rather than discovered in an OOM or in a metrics file.
    """
    if float(args.w_temp) < 0:
        raise SystemExit("--w-temp must be >= 0")
    if temp_enabled(args):
        temp_config_from_args(args)     # TempConfig's own refusals (bin range, k)
    # --- the conditioning channel ------------------------------------------
    # Parsing catches a typo'd channel name; the existence check catches the
    # far more expensive case of a run that trains for ten hours against a
    # stack built under different settings, or against no stack at all.
    ccfg = cond_config_of(args)
    if ccfg is not None:
        boxes = [b for b in
                 (str(args.train_boxes) + " " + str(args.holdout_boxes))
                 .replace(",", " ").split() if b]
        for b in boxes:
            path = channel_field_path(b, str(args.cond_root) or None)
            if not path.is_file():
                raise SystemExit(
                    f"--cond-channels is set but {path} does not exist. Build "
                    f"the conditioning fields first:\n"
                    f"    CH_BOXES='{' '.join(boxes)}' bash "
                    f"scripts/slurm/submit_lr_host_channel.sh")
            load_channel_field(path, ccfg)      # raises on a settings mismatch
    if float(args.cond_lr) <= 0.0 and ccfg is not None:
        raise SystemExit(
            f"--cond-lr {args.cond_lr} freezes the conditioning columns at "
            "their zero init, which makes this run the unconditioned control "
            "under the conditioned arm's name. Drop --cond-channels instead.")

    if args.population != "sub" and int(args.max_num_p) <= 0:
        raise SystemExit(
            f"--population {args.population} without --max-num-p selects the "
            "cluster host itself as a supervised set. Its pair sum is ~100k "
            "particles (1e10 pairs against a 200p set's 2e4) and its HR "
            "reference asks the generator to CONCENTRATE a 1e14 halo -- a "
            "different objective from this one, and one --w-host-sets already "
            "handles against the frozen reference. Pass an explicit ceiling, "
            "e.g. --max-num-p 2000.")
    if int(args.max_num_p) and int(args.max_num_p) <= int(args.min_num_p):
        raise SystemExit(
            f"--max-num-p {args.max_num_p} is not above --min-num-p "
            f"{args.min_num_p}; the window is empty.")
    if args.population != "sub" and args.centre_mode in ("radial", "self_tan",
                                                         "self_tan_parent"):
        # `radial` projects the centre offset onto the direction from the
        # CLUSTER to the set. For a top-level halo far out in the same tiles
        # that direction is still defined, but the term's whole justification --
        # 62.9% of a SATELLITE's offset is radial about its host, measured over
        # 7,560 sets in centre_offset/pool/offsets.json -- was measured on
        # subhalos only and is not established for the field population.
        print(f">>> WARNING: --centre-mode {args.centre_mode} with "
              f"--population {args.population}. The radial decomposition that "
              "motivates it was measured on SUBHALOS about their own host; for "
              "a field halo the cluster is not its host and the direction "
              "carries no such claim. Continuing.", flush=True)

    # --- the environment axis -----------------------------------------------
    sel = str(args.env_select).strip()
    if sel not in ("", "any", "all"):
        bad = [w.strip() for w in sel.split(",")
               if w.strip() and w.strip() not in ENV_LABELS]
        if bad:
            raise SystemExit(
                f"--env-select {args.env_select!r} names unknown environments "
                f"{bad}; expected 'any' or a comma list of {list(ENV_LABELS)}.")
        if args.population == "sub" and sel == "field":
            # Not fatal, but it is almost certainly not the run that was meant:
            # a `field` SUBHALO is a satellite of a host below --env-host-floor,
            # a small and peculiar population, and the census's field deficit is
            # a TOP-LEVEL one.
            print(">>> WARNING: --env-select field with --population sub selects "
                  "satellites of hosts below --env-host-floor, not the isolated "
                  "field halos the census's `field` deficit is about (that "
                  "deficit is top-level). Add --population all if the field "
                  "halos themselves are the target. Continuing.", flush=True)
    if float(args.w_isolation) > 0.0 and int(args.bg_k) <= 0:
        raise SystemExit(
            "--w-isolation > 0 with --bg-k 0: the isolation term counts LOCAL "
            "NON-MEMBER particles and --bg-k 0 gathers none, so the term would "
            "be identically zero and the run would be a --w-isolation 0 run "
            "under another name. Pass --bg-k (default 4096).")
    if not (0.0 < float(args.iso_radius_frac) < 1.0):
        raise SystemExit(
            f"--iso-radius-frac {args.iso_radius_frac} is not in (0, 1). The "
            "background ids are chosen ONCE, on the frozen field, inside "
            "--bg-radius; a counting radius at or beyond that edge counts the "
            "sampling boundary rather than the neighbourhood. Widen --bg-radius "
            "instead.")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate(args)
    if bool(getattr(args, "detect_anomaly", False)):
        # Debug only (several x slower): a NaN in backward raises with the
        # FORWARD op that produced it, which the non-finite guard cannot say.
        torch.autograd.set_detect_anomaly(True)
        print("=== autograd anomaly detection ON (debug run)", flush=True)

    s = run(args)
    return 0 if s.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
