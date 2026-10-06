#!/usr/bin/env python
"""Train SR2's architecture FROM SCRATCH, with the structure terms in from the start.

Why this job exists
-------------------
Every result on this line so far is a fine-tune of ``SRmodel/G_z0.pt``, and the
one defect none of them has moved is *where the substructure sits*:

* HR puts subhalos at a median **1.13** host ``R_vir``; the frozen checkpoint
  sits at 0.53 and **every tuned arm makes it worse**, 0.30-0.35
  (`radial-placement-is-inverted`, 56 arms).
* The membership solve is not the cause. Its candidate centres sit at median
  **1.42** ``R_vir`` against HR's 1.19 -- it aims correctly and the objects
  still condense at 0.35 (`docs/sr2_soft_membership.md` section 7.9).
* The terms built to move it failed *as terms*: the shell family is dead at
  three doses (`soft-shell-distribution-term`), and relocating the supervised
  material made the inward drift worse, not better
  (`relocate-arms-hypothesis-falsified`).

The hypothesis this job tests is the remaining one: **the mass is already piled
into the host centre before any loss term speaks, and a fine-tune cannot undo
it.** So start from noise, and put the placement-sensitive terms in the
objective from the beginning rather than bolting them onto a converged
checkpoint.

The arms
--------
``base``
    SR2's own recipe from a random init: ``map2map``'s ``MSELoss`` plus a
    WGAN-GP adversary on :class:`cosmo_sr.reward.sr2_adversarial.SR2Critic`,
    whose input contract is SR2's own -- 6 upsampled LR + 6 candidate +
    **8 inverse-pixel-shuffled CIC density channels**. This is the control, and
    it is the arm that answers the question: does a from-scratch SR2 reproduce
    the inverted radial profile at all? If it does, no loss term will fix it and
    the defect is the architecture or the base objective.

``base_noms``
    ``--w-mse 0``. A pointwise loss to one realisation is this repository's
    recorded cause of mean-collapse, and mean-collapse is what a centrally
    concentrated halo looks like. One flag separates "MSE did this" from "the
    architecture did this".

``knn``
    ``base`` + the 6-D class-conditioned kNN-CDF term. Round 5b measured the
    settings and they are not defaults: ``--knn-phase-lambda 4000``,
    ``--knn-ks 8 16 32 64``, ``--knn-classes sub_50_200``. **Hosts must stay out
    of the term** -- the pooled loss ranks the working arms at rho **+0.878**
    (p 4e-07), i.e. the wrong sign, and 6-D makes it worse
    (`knn-phase-space-round5`).

``soft_radial``
    ``base`` + the member-gather objective under soft membership
    (``--soft-prior coarse --soft-lambda 1``, the arm that beat the hard mask on
    the held-out Rockstar gate) **plus the host-centric radius term**
    (``--w-radius``). That term is measured dead as a fine-tune lever; retrying
    it from a random init is the direct test of the hypothesis above.

What is shared with the fine-tune, and what is not
--------------------------------------------------
Shared, deliberately: the pool builder, the member sets and their reference, the
soft-membership solve, the kNN term, the guards, the verdict and the gate
snapshots are all imported from ``finetune_member_gather.py``. A second
definition of any of them would mean this run is not working the same problem.

Not shared, deliberately:

* **the initialisation** -- :func:`cosmo_sr.train.sr2_scratch.build_scratch_generator`
  instead of ``load_controlled_generator``;
* **the optimiser** -- one uniform learning rate over every parameter, not the
  depth-scaled micro-rates of ``sr2_unfreeze``. Those rates exist to protect a
  pretrained function; there is none here, and ``--rung`` is forced to ``full``;
* **the guard reference** -- ``--guard-ref hr``. The LR-scale anchor is a drift
  detector in a fine-tune, where the candidate starts AS the frozen field. From
  a random init there is no drift, and anchoring to the frozen field would
  charge the run for failing to reproduce the very concentration under test;
* **the data** -- a streaming :class:`PairedTileSampler` over every tile of the
  training boxes, not a resident pool of ~56 hosts. A from-scratch generator
  needs the whole box distribution; 8,192 (box, tile) pairs as resident tensors
  would be 48 GB.

One caveat, stated rather than buried: the member-gather **reference** is still
built against the frozen SR2 field (``build_member_sets`` takes it as the
reachable target, and ``frozen_box`` supplies members outside the trained
tiles). That is a constant of the objective, fixed before step 1 and identical
to the one the fine-tune arms used -- it is a supervision target, not an
initialisation -- but it does mean the structure terms are not frozen-SR2-free.

Warm-up is not optional
-----------------------
The soft-membership solve chooses particles from the *current* SR field, and the
kNN term matches the candidate's neighbour distances against HR's. At step 0
that field is noise. ``--term-warmup-steps`` holds both at zero and
``--term-ramp-steps`` ramps them in, so they begin speaking once there is a
field to speak about.

Resumability
------------
``latest.pt`` carries the generator, the critic, both optimiser states, the step
counter and the samplers' RNG. A run that hits its time limit is resubmitted,
not restarted, and ``metrics.jsonl`` is appended to. This is a multi-day job by
construction.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "reward",
           PROJECT_ROOT / "scripts" / "features"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (  # noqa: E402
    append_jsonl, banner, paths, select_device, write_json,
)
from _sr2_direct import (  # noqa: E402
    geometry_of, load_direct_config, load_hr, load_lr, model_path_of,
    phase_space_config_of, soft_config_of,
)
from overfit_host_mse import BOXSIZE, DX, NG_HR, TILE  # noqa: E402
from free_field_gather import member_config_of  # noqa: E402

# The fine-tune harness is imported as a module, not copied: the pool, the
# member sets, the soft solve, the kNN term, the guards and the verdict are its
# definitions and must stay single-sourced.
import finetune_member_gather as MG  # noqa: E402

from cosmo_sr.features.knn_cdf import OWNER_CLASSES  # noqa: E402
from cosmo_sr.features.member_pool import split_pool, summarise_pool  # noqa: E402
from cosmo_sr.train.common import finish_wandb, maybe_init_wandb  # noqa: E402
from cosmo_sr.dmsr.critic import LazyR1  # noqa: E402
from cosmo_sr.train.sr2_scratch import (  # noqa: E402
    ADV_LOSSES, DENSITY_TRANSFORMS, PairedTileSampler, add_instance_noise,
    base_mse, bce_critic_update, bce_generator_loss, build_scratch_critic,
    instance_noise_std, mse_handoff_weight, scale_anchor,
    build_scratch_generator, check_density_channels, critic_update,
    fit_critic_norm, forward_batch, generator_adv_loss, lowk_anchor, lr_core,
    noise_std_report, scratch_critic_input, term_ramp,
)
from cosmo_sr.tts.srs_noise import load_controlled_generator  # noqa: E402


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def build_parser():
    """The fine-tune parser plus a ``scratch`` group.

    Reusing it is what makes every ``MG_*`` knob mean the same thing in both
    jobs. The defaults that do NOT transfer are overridden in
    :func:`apply_scratch_defaults` and printed, rather than silently inherited.
    """
    ap = MG.build_parser()
    ap.description = "Train SR2's architecture from a random initialisation."

    g = ap.add_argument_group("from scratch")
    g.add_argument("--scratch-lr", type=float, default=1e-4,
                   help="one uniform Adam rate for every generator parameter. "
                        "The depth-scaled rates of sr2_unfreeze exist to protect "
                        "a pretrained function; there is none here.")
    g.add_argument("--scratch-betas", type=float, nargs=2, default=[0.9, 0.999],
                   help="Adam betas. map2map passes --optimizer-args '{}', so "
                        "upstream is PyTorch's Adam default (0.9, 0.999). "
                        "(0, 0.99) is the StyleGAN2/WGAN convention and is what "
                        "runs 1-2 used.")
    g.add_argument("--init-weight-std", type=float, default=0.0,
                   help="map2map's --init-weight-std: N(0, std) for every conv "
                        "weight. 0 keeps PyTorch's Kaiming-uniform default, "
                        "which is what upstream uses when the flag is unset.")
    g.add_argument("--init-noise-std", type=float, default=0.0,
                   help="initial AddNoise.std at all six sites. Upstream starts "
                        "at 0 (deterministic until the gradient grows it) and "
                        "the pretrained G_z0 ended at ~1e-3 (z0-z4) / 5e-2 (z5). "
                        "noise_std_* in metrics.jsonl is how a run checks that "
                        "they left zero; a generator whose std stays at 0 has no "
                        "stochasticity to build substructure out of.")
    g.add_argument("--base-batch", type=int, default=4,
                   help="paired tiles per base-objective step.")
    g.add_argument("--w-mse", type=float, default=1.0,
                   help="map2map's --criterion MSELoss on all six channels. "
                        "0 is the `base_noms` arm.")
    g.add_argument("--w-base-low", type=float, default=0.0,
                   help="LR-scale anchor on the base minibatch, referenced to "
                        "HR. **There is no such term in map2map**, so 0 is the "
                        "faithful setting. It defaulted to 100 in runs 1-2, "
                        "copied from the fine-tune's --w-low, where the term "
                        "sits near zero because the candidate STARTS as the "
                        "frozen field. From a random init it does not: MEASURED "
                        "at step 31500 it was 96-100%% of the non-adversarial "
                        "budget, pulling the generator at the LR-smooth "
                        "solution, and both arms ended with ~4x LESS small-scale "
                        "density power than trilinear upsampling.")
    g.add_argument("--w-scale", type=float, default=0.0,
                   help="AMPLITUDE anchor: log(rms_cand/rms_hr)^2 on the "
                        "displacement and velocity channel groups. Two scalars "
                        "per batch, NO spatial information -- it cannot prefer "
                        "one radial profile over another, which is what makes "
                        "it a safe leash to keep when --w-mse goes to 0. It "
                        "exists because the adversary is BLIND to amplitude: "
                        "SR2Critic's GroupNorms make it exactly scale-invariant "
                        "(MEASURED: D constant to 4 digits from 1e3x to 1e6x, "
                        "||grad D|| falling as 1/a), so with MSE off nothing at "
                        "all constrains the output scale and the field "
                        "random-walks to 1e5x HR. See also "
                        "--critic-inner-norm, which fixes the cause.")
    g.add_argument("--mse-handoff-steps", type=int, default=0,
                   help="linearly decay --w-mse to 0 over this many steps after "
                        "the adversary starts, instead of map2map's cliff "
                        "(0, the default). A cliff also strands Adam's "
                        "second-moment estimate on a loss that no longer "
                        "exists. Only meaningful with --mse-mode upstream.")
    g.add_argument("--w-sr2-adv", type=float, default=1.0,
                   help="WGAN-GP adversarial weight on SR2Critic (the DENSITY "
                        "critic). Distinct from --w-adv, which is the "
                        "member-gather high-pass critic and stays off here.")
    g.add_argument("--adv-loss", default="bce", choices=ADV_LOSSES,
                   help="`bce` is what map2map ACTUALLY trains with: "
                        "BCEWithLogitsLoss on labels (its --adv-criterion "
                        "default) plus optional lazy R1 on the REAL batch. "
                        "There is no WGAN and no interpolate penalty anywhere "
                        "in map2map/train.py -- `wgan_gp` is this repo's own "
                        "variant, kept because run 1 used it.")
    g.add_argument("--mse-mode", default="upstream", choices=("upstream", "sum"),
                   help="`upstream` reproduces map2map/train.py exactly: the "
                        "generator steps on MSE ONLY until the adversary "
                        "starts, and on the ADVERSARIAL loss alone after that "
                        "-- the two are never summed, and MSE gets no gradient "
                        "once the critic is live (it is still logged, which is "
                        "all upstream uses it for). `sum` adds them, which is "
                        "what run 1 and 2 did.")
    g.add_argument("--adv-label-smoothing", type=float, default=1.0,
                   help="map2map --adv-label-smoothing; 1.0 is its default "
                        "(off). Below 1 softens D's real target.")
    g.add_argument("--r1-gamma", type=float, default=10.0,
                   help="R1 strength for --adv-loss bce.")
    g.add_argument("--adv-start-epochs", type=float, default=0.0,
                   help="map2map --adv-start, which counts EPOCHS not steps. "
                        ">0 overrides --sr2-adv-warmup-steps with "
                        "round(epochs * len(sampler) / --base-batch). The "
                        "working SR2 recipe uses --adv-start 1, i.e. exactly "
                        "one pass over the data on MSE before the adversary "
                        "joins. Translating that as a flat 1000 steps is what "
                        "run 3 did, and it diverged.")
    g.add_argument("--critic-spectral-norm", action="store_true",
                   help="map2map --adv-model-spectral-norm. Bounds the critic's "
                        "Lipschitz constant STRUCTURALLY, which R1 and WGAN-GP "
                        "cannot: a penalty only charges for a large gradient. "
                        "Run 3 diverged with a critic that saturated at "
                        "d_real +8.5 / d_fake -8.2.")
    g.add_argument("--critic-inner-norm", default="group",
                   choices=("group", "none"),
                   help="normalisation INSIDE SR2Critic's body. `group` is the "
                        "historical default and is the ROOT CAUSE of the "
                        "handoff divergence: GroupNorm divides out the "
                        "per-group std, so GN(a*z) == GN(z) and the critic is "
                        "exactly scale-invariant -- it cannot see that the "
                        "field is 1e5x too large, and its gradient back to the "
                        "generator decays as 1/a, so the runaway is "
                        "self-reinforcing. MEASURED on width=16 depth=4: D "
                        "pinned at 0.009504 for a in [1e3, 1e6] while "
                        "a*||grad D|| stays 0.6846 at every decade, with and "
                        "without spectral norm. `none` makes the critic "
                        "positively homogeneous instead, so amplitude becomes "
                        "something it CAN learn to reject. Use it with "
                        "--critic-spectral-norm (the SNGAN pairing).")
    g.add_argument("--instance-noise", type=float, default=0.0,
                   help="map2map --instance-noise: N(0, sigma) added to the "
                        "critic's inputs, real and fake alike, annealed to 0 "
                        "over --instance-noise-batches. Stops D separating them "
                        "perfectly. Upstream default 0.")
    g.add_argument("--instance-noise-batches", type=float, default=1e4,
                   help="map2map --instance-noise-batches: the anneal length.")
    g.add_argument("--r1-interval", type=int, default=16,
                   help="map2map --adv-r1-reg-interval. ITS default is 0 "
                        "(off); 16 is kept on here because a GAN from a random "
                        "init is unstable without it, and that is a deliberate "
                        "deviation rather than a silent one.")
    g.add_argument("--sr2-critic-lr", type=float, default=1e-4)
    g.add_argument("--sr2-critic-width", type=int, default=64)
    g.add_argument("--sr2-critic-depth", type=int, default=4)
    g.add_argument("--sr2-critic-steps", type=int, default=1,
                   help="critic updates per generator step.")
    g.add_argument("--sr2-gp-lambda", type=float, default=10.0)
    g.add_argument("--sr2-gp-every", type=int, default=16,
                   help="lazy gradient penalty interval (the upstream "
                        "schedule); the coefficient is scaled by this on the "
                        "steps it runs.")
    g.add_argument("--sr2-adv-warmup-steps", type=int, default=1000,
                   help="generator-frozen critic steps before the adversarial "
                        "gradient reaches the generator. The critic is random "
                        "at step 0 and a random critic defines a random "
                        "objective.")
    g.add_argument("--density-grid-mult", type=int, default=2,
                   help="CIC mesh multiple for the critic's density channels; "
                        "2 is the SR2 recipe and gives 8 channels.")
    g.add_argument("--density-valid-center", type=int, default=0,
                   help="valid-centre CIC region in HR cells. 0 means 'tile/2', "
                        "the exact setting (r = 1.0000). It does NOT mean the "
                        "wrapped deposit -- that is refused, because on a tile "
                        "it correlates r = 0.08 with the true density and the "
                        "critic's 8 density channels would be noise.")
    g.add_argument("--allow-wrapped-density", action="store_true",
                   help="run the broken wrapped deposit deliberately.")
    g.add_argument("--density-transform", default="asinh",
                   choices=DENSITY_TRANSFORMS,
                   help="tail transform on the critic's density channels. "
                        "MEASURED raw: max 2786 against RMS 4.9, a 570x spread, "
                        "which makes ||grad D||=1 unenforceable off the density "
                        "directions -- jobs 62067/62068 ran the score gap to "
                        "+2548 with ||grad D||~4. asinh maps 2786 -> 8.6 and is "
                        "finite at delta=-1, the MEDIAN voxel, where log1p is "
                        "-inf. 'none' is the faithful-to-upstream arm and is the "
                        "one measured to diverge.")
    g.add_argument("--sr2-critic-norm-tiles", type=int, default=32,
                   help="real HR tiles used to fit the critic's per-channel "
                        "scales. Fitted ONCE from real data, held fixed, applied "
                        "identically to real and fake -- per-batch statistics "
                        "would normalise the two sides differently and hand the "
                        "critic a shortcut unrelated to sample quality.")
    g.add_argument("--w-gather", type=float, default=1.0,
                   help="weight on the whole member-gather host loss. 0 turns "
                        "the structure supervision off (the `base` arms).")
    g.add_argument("--term-warmup-steps", type=int, default=2000,
                   help="steps with the structure terms (gather, kNN) held at "
                        "zero. They select particles from, and measure, the "
                        "current SR field; at step 0 that field is noise.")
    g.add_argument("--term-ramp-steps", type=int, default=2000,
                   help="linear ramp to full weight after the warm-up.")
    g.add_argument("--base-exclude-supervised", action="store_true",
                   help="keep the supervised host tiles out of the base "
                        "objective's stream. Off by default: the generator has "
                        "to produce a good field on host tiles too, and the "
                        "honest control is the held-out BOX, not a held-out "
                        "tile of a trained box.")
    g.add_argument("--base-noise-seed-stride", type=int, default=1_000_003,
                   help="0 pins every tile to --seed's realisation (replayable "
                        "but memorisable); the default draws a fresh root seed "
                        "per visit.")
    g.add_argument("--resume", action="store_true",
                   help="continue from latest.pt in the output directory. "
                        "This job is time-limited and resubmitted, not "
                        "restarted.")
    g.add_argument("--save-every", type=int, default=500,
                   help="steps between latest.pt writes (the resume point).")
    g.add_argument("--shakeout-steps", type=int, default=0,
                   help=">0 runs that many steps, skips the host pool unless "
                        "the arm needs it, and writes shakeout.json with the "
                        "measured step time and peak memory. This is what "
                        "sizes --time and --mem for the real arms.")
    return ap


def apply_scratch_defaults(args) -> List[str]:
    """Force the settings a from-scratch run cannot meaningfully vary.

    Returns the list of overrides applied, so the job prints what it changed
    instead of inheriting a fine-tune default that reads as deliberate.
    """
    changed: List[str] = []

    def force(name: str, value, why: str):
        cur = getattr(args, name, None)
        if cur != value:
            setattr(args, name, value)
            changed.append(f"{name}: {cur!r} -> {value!r}  ({why})")

    force("rung", "full",
          "no pretrained function to protect; every parameter trains")
    force("guard_ref", "hr",
          "a random init has not DRIFTED from the frozen field")
    force("w_adv", 0.0,
          "--w-sr2-adv is this job's adversary; the gather high-pass critic "
          "is a separate regulariser and stays off")
    return changed


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_state(path: Path, *, model, critic, opt, opt_d, step: int,
               args, rng_state: Dict, critic_norm=None) -> None:
    """The resume point. Atomic, via ``MG.save_atomic``."""
    MG.save_atomic({
        "model": {n: p.detach().cpu() for n, p in model.state_dict().items()},
        "critic": (None if critic is None else
                   {n: p.detach().cpu() for n, p in critic.state_dict().items()}),
        "opt": opt.state_dict(),
        "opt_d": (None if opt_d is None else opt_d.state_dict()),
        "step": int(step),
        "rng": rng_state,
        "scratch": True,
        "init": "random",
        "label": str(args.label),
        # The critic is worthless without the scales its input was normalised
        # by, and a gate that re-derived them from its own tiles would be
        # scoring a different function.
        "critic_norm": (None if critic_norm is None else critic_norm.to_dict()),
        "density_transform": str(getattr(args, "density_transform", "asinh")),
    }, path)


def load_state(path: Path, *, model, critic, opt, opt_d, device,
               expect_transform: Optional[str] = None,
               wants_critic: bool = False) -> Tuple[int, Dict, Optional[Dict]]:
    """Restore a run in place. Returns ``(step, rng_state, saved_critic_norm)``.

    The two refusals here are not defensive clutter; each is a mistake this run
    has already made once.

    ``scratch``
        A fine-tune checkpoint would train PRETRAINED weights under the
        from-scratch arm's name.

    **the critic-input contract**
        A checkpoint's critic was fitted to one input distribution. Resuming it
        against a different one silently continues a *different* experiment:
        run 1's critic saw raw channels at density RMS ~5 and max ~2800, and the
        fixed run feeds it RMS ~1 and max ~10, so its first-layer responses are
        off by orders of magnitude and nothing in the logs says so. Run 1's
        checkpoints carry no ``critic_norm`` at all, which is exactly the case
        this catches.

    The saved scales are RETURNED rather than discarded, and the caller adopts
    them in place of a freshly fitted set. The fit draws random tiles, so a
    12-hour resubmission would otherwise hand the critic slightly different
    scales than it was trained under, every single time it resumes.
    """
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if not blob.get("scratch"):
        raise SystemExit(
            f"{path} was not written by this trainer (no 'scratch' flag). "
            "Resuming a fine-tune checkpoint here would train a pretrained "
            "generator under the from-scratch arm's name.")
    saved_norm = blob.get("critic_norm")
    saved_tf = blob.get("density_transform")
    if wants_critic:
        if saved_norm is None:
            raise SystemExit(
                f"{path} is at step {blob.get('step')} and carries NO "
                "critic_norm, i.e. it predates the input normaliser. Its "
                "critic was fitted to raw channels (density RMS ~5, max ~2800) "
                "and this run feeds normalised ones (RMS ~1, max ~10), so "
                "resuming it continues a different experiment with no log line "
                "saying so.\n"
                "  Start this arm clean: move its output directory aside, or "
                "pass SC_RESUME=0 for the first chunk.")
        if expect_transform is not None and str(saved_tf) != str(expect_transform):
            raise SystemExit(
                f"{path} was trained with --density-transform {saved_tf!r} and "
                f"this run asks for {expect_transform!r}. The critic's input "
                "distribution would change under it mid-run. Start clean or "
                "match the transform.")
    model.load_state_dict(blob["model"])
    model.to(device)
    if critic is not None and blob.get("critic") is not None:
        critic.load_state_dict(blob["critic"])
        critic.to(device)
    opt.load_state_dict(blob["opt"])
    if opt_d is not None and blob.get("opt_d") is not None:
        opt_d.load_state_dict(blob["opt_d"])
    return int(blob.get("step", 0)), dict(blob.get("rng") or {}), saved_norm


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    cfg = load_direct_config(args)
    geom = geometry_of(cfg)
    scfg = soft_config_of(cfg)
    pscfg = phase_space_config_of(cfg)
    mcfg = member_config_of(args)
    device = select_device(args.device or None)

    overrides = apply_scratch_defaults(args)
    banner("SR2 FROM SCRATCH -- random initialisation, no G_z0.pt weights")
    for line in overrides:
        print(f"  forced {line}", flush=True)

    train_boxes = [b for b in str(args.train_boxes).replace(",", " ").split() if b]
    hold_boxes = [b for b in str(args.holdout_boxes).replace(",", " ").split() if b]
    hold_keys = [k for k in str(args.holdout_hosts).replace(",", " ").split() if k]

    # 0 means "the exact tile/2 region", NOT the wrapped deposit: the wrapped
    # deposit is only reachable through --allow-wrapped-density, because a run
    # that used it by accident looks healthy in every line it logs.
    allow_wrapped = bool(args.allow_wrapped_density)
    vc = (0 if allow_wrapped
          else (int(args.density_valid_center)
                if int(args.density_valid_center) > 0 else int(TILE) // 2))
    check_density_channels(vc, int(TILE), allow_wrapped=allow_wrapped)
    if allow_wrapped:
        print(">>> --allow-wrapped-density: the critic's 8 density channels "
              "correlate r = 0.08\n    with the tile's true density. This is "
              "the BROKEN deposit, run deliberately.", flush=True)
    n_density = int(args.density_grid_mult) ** 3
    print(f"\ncritic: SR2Critic, {12 + n_density} channels "
          f"(6 upsampled LR + 6 field + {n_density} DENSITY), "
          f"valid-centre CIC region {vc} HR cells of a {TILE}^3 tile.")
    print("  the density channels are the point: this critic judges the MASS "
          "field, not\n  the displacement that produced it. At valid_center=0 "
          "they would correlate\n  r = 0.08 with the tile's own density "
          "(docs/density_collapse_investigation.md).", flush=True)

    out_dir = paths.subdir("sr2_scratch", f"{args.label or 'base'}", create=True)
    print(f"\noutput: {out_dir}", flush=True)

    # --- the generator: random weights, one uniform rate --------------------
    model = build_scratch_generator(
        in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
        out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
        scale_factor=geom.scale_factor,
        cond_chan=0,
        init_weight_std=float(args.init_weight_std),
        init_noise_std=float(args.init_noise_std),
        seed=int(args.seed), device=device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\ngenerator: ControlledG(scale_factor={geom.scale_factor}), "
          f"{n_par} parameters, ALL trainable, lr {args.scratch_lr} "
          f"betas {tuple(args.scratch_betas)}")
    print(f"  AddNoise.std at init: {noise_std_report(model)}")
    print("  watch noise_std_mean in metrics.jsonl -- SR2's fine-scale "
          "structure is\n  written by these six tensors, and a run that leaves "
          "them at zero is a\n  deterministic upsampler however good its "
          "spectrum looks.", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.scratch_lr),
                           betas=tuple(float(b) for b in args.scratch_betas))

    # --- the paired-tile stream for the base objective ----------------------
    lr_fields = {b: load_lr(cfg, b) for b in train_boxes}
    hr_fields = {b: load_hr(cfg, b) for b in train_boxes}
    hold_lr = {b: load_lr(cfg, b) for b in hold_boxes}
    hold_hr = {b: load_hr(cfg, b) for b in hold_boxes}

    # --- the host pool (only when a structure term needs it) ----------------
    want_gather = float(args.w_gather) > 0
    want_knn = float(args.w_knn_unsup) > 0 or bool(args.knn_measure)
    all_tasks: List = []
    train_tasks: List = []
    hold_tasks: List = []
    frozen = None
    if want_gather:
        # The member-set REFERENCE is built against the frozen field -- it is
        # the reachable target `build_member_sets` was defined with, a constant
        # fixed before step 1 and identical to the fine-tune arms'. The frozen
        # generator is NOT used in the loss: --guard-ref hr removed its only
        # per-step call site.
        frozen = load_controlled_generator(
            model_path_of(cfg),
            in_chan=int(cfg.get("model", {}).get("in_chan", 6)),
            out_chan=int(cfg.get("model", {}).get("out_chan", 6)),
            scale_factor=geom.scale_factor, device=device, eval_mode=True)
        for p in frozen.parameters():
            p.requires_grad_(False)
        print("\nbuilding the host pool (the member-gather supervision)",
              flush=True)
        all_tasks = MG.build_pool(args, cfg, geom, scfg, pscfg, mcfg, frozen,
                                  device, train_boxes + hold_boxes)
        if not all_tasks:
            raise SystemExit(
                "no hosts selected; build the owner arrays with "
                "scripts/slurm/submit_owner_arrays.sh.")
        split = split_pool([t.sel for t in all_tasks], train_boxes=train_boxes,
                           holdout_boxes=hold_boxes, holdout_keys=hold_keys)
        by_key = {t.key: t for t in all_tasks}
        train_tasks = [by_key[s.key] for s in split.train]
        hold_tasks = [by_key[s.key] for s in split.holdout]
        print(f"  {len(train_tasks)} training hosts, {len(hold_tasks)} held out",
              flush=True)
        write_json(out_dir / "pool.json", {
            "train": summarise_pool([t.sel for t in train_tasks]),
            "holdout": summarise_pool([t.sel for t in hold_tasks]),
            "train_keys": [t.key for t in train_tasks],
            "holdout_keys": [t.key for t in hold_tasks],
        })

    supervised = {(t.sel.box, int(tid)) for t in all_tasks for tid in t.tiles}
    sampler = PairedTileSampler(
        train_boxes, lr_fields, hr_fields, geom=geom, ng_hr=NG_HR, tile_hr=TILE,
        exclude=(sorted(supervised) if args.base_exclude_supervised else None),
        base_seed=int(args.seed),
        noise_seed_stride=int(args.base_noise_seed_stride),
        seed=int(args.seed), device=device)
    hold_sampler = (PairedTileSampler(
        hold_boxes, hold_lr, hold_hr, geom=geom, ng_hr=NG_HR, tile_hr=TILE,
        base_seed=int(args.seed), noise_seed_stride=0,
        seed=int(args.seed) + 7, device=device) if hold_boxes else None)
    print(f"\nbase objective: {len(sampler)} (box, tile) pairs over "
          f"{len(train_boxes)} boxes, batch {args.base_batch}"
          + (f", {sampler.n_excluded} supervised tiles excluded"
             if args.base_exclude_supervised else "")
          + (f"; held-out stream {len(hold_sampler)} pairs"
             if hold_sampler is not None else ""), flush=True)

    # --- --adv-start-epochs, resolved against the actual sampler -----------
    steps_per_epoch = max(1, len(sampler) // max(1, int(args.base_batch)))
    if float(args.adv_start_epochs) > 0:
        args.sr2_adv_warmup_steps = int(round(
            float(args.adv_start_epochs) * steps_per_epoch))
        print(f"\n--adv-start-epochs {args.adv_start_epochs} x "
              f"{steps_per_epoch} steps/epoch ({len(sampler)} pairs / batch "
              f"{args.base_batch}) = adversary joins at step "
              f"{args.sr2_adv_warmup_steps}", flush=True)

    # --- the DENSITY critic, and its input scales --------------------------
    # Built HERE, after the sampler, and not earlier: the per-channel scales are
    # fitted from real HR tiles the sampler draws, and a critic without them is
    # the runaway measured in jobs 62067/62068 (score gap +2548 at step 30k,
    # ||grad D|| ~ 4). Both other critics in this repo -- DMSR's
    # CriticInputNormalizer and the gather critic's GatherCriticNorm -- fit the
    # same thing for the same reason; this path did not, which was the bug.
    critic = opt_d = critic_norm = lazy_r1 = None
    if float(args.w_sr2_adv) > 0:
        critic_norm = fit_critic_norm(
            sampler, geom, cellsize_kpc_h=float(scfg.cellsize_kpc_h),
            dis_norm_kpc_h=float(scfg.dis_norm_kpc_h),
            grid_mult=int(args.density_grid_mult), valid_center=vc,
            allow_wrapped=allow_wrapped,
            density_transform=str(args.density_transform),
            n_tiles=int(args.sr2_critic_norm_tiles),
            batch=int(args.base_batch), device=device)
        write_json(out_dir / "critic_norm.json", {
            "density_transform": str(args.density_transform),
            "density_valid_center": int(vc),
            "grid_mult": int(args.density_grid_mult),
            "fit_tiles": int(args.sr2_critic_norm_tiles),
            "fit_source": "real HR tiles only, held fixed",
            **critic_norm.to_dict(),
        })
        _sc = critic_norm.to_dict()["scale"]
        _nd = int(args.density_grid_mult) ** 3
        print(f"\ncritic input: density transform '{args.density_transform}', "
              f"per-channel scales fitted from {args.sr2_critic_norm_tiles} real "
              f"HR tiles")
        print(f"    head (LR+field) scales {min(_sc[:-_nd]):.3f}-"
              f"{max(_sc[:-_nd]):.3f}   density scales "
              f"{min(_sc[-_nd:]):.3f}-{max(_sc[-_nd:]):.3f}")
        print(f"    wrote {out_dir / 'critic_norm.json'} -- the SAME scales are "
              f"applied to real and fake;\n    per-batch statistics would "
              f"normalise the two sides differently.", flush=True)

        critic = build_scratch_critic(width=int(args.sr2_critic_width),
                                      depth=int(args.sr2_critic_depth),
                                      seed=int(args.seed) + 1,
                                      spectral_norm=bool(args.critic_spectral_norm),
                                      inner_norm=str(args.critic_inner_norm),
                                      device=device)
        if bool(args.critic_spectral_norm):
            print("    spectral norm ON (map2map --adv-model-spectral-norm)")
        if str(args.critic_inner_norm) == "none":
            print("    inner norm NONE: the critic is positively homogeneous "
                  "and can SEE\n    amplitude. `group` (the default) is exactly "
                  "scale-invariant.")
        else:
            print("    inner norm GROUP: the critic is exactly scale-invariant "
                  "and CANNOT\n    see the field's amplitude "
                  "(D constant to 4 digits from 1e3x to 1e6x).")
        if float(args.instance_noise) > 0:
            print(f"    instance noise {args.instance_noise} annealed over "
                  f"{args.instance_noise_batches:.0f} steps")
        # map2map passes --adv-optimizer-args '{}' unless told otherwise, so
        # the faithful adversary optimiser is Adam's default betas too.
        _dbetas = ((0.0, 0.9) if str(args.adv_loss) == "wgan_gp"
                   else tuple(float(b) for b in args.scratch_betas))
        opt_d = torch.optim.Adam(critic.parameters(),
                                 lr=float(args.sr2_critic_lr), betas=_dbetas)
        lazy_r1 = (LazyR1(gamma=float(args.r1_gamma),
                          interval=int(args.r1_interval))
                   if str(args.adv_loss) == "bce" and int(args.r1_interval) > 0
                   else None)
        print(f"  adversary: {args.adv_loss}"
              + (f", R1 gamma {args.r1_gamma} every {args.r1_interval}"
                 if lazy_r1 is not None else "")
              + f", label smoothing {args.adv_label_smoothing}"
              + f", mse-mode {args.mse_mode}")
        if str(args.adv_loss) == "bce":
            print("    this is map2map's own objective: BCEWithLogitsLoss on "
                  "labels + lazy R1\n    on the REAL batch. There is no WGAN "
                  "and no interpolate penalty upstream.")
        if str(args.mse_mode) == "upstream":
            print("    mse-mode upstream: the generator steps on MSE until the "
                  "adversary\n    starts and on the ADVERSARIAL loss alone "
                  "after. They are never summed.")
            if int(args.mse_handoff_steps) > 0:
                print(f"    handoff RAMPED over {args.mse_handoff_steps} steps "
                      f"rather than a cliff.")
            # The configuration that MEASURABLY diverges. Loud, but not fatal:
            # reproducing it deliberately is a legitimate thing to want.
            if (float(args.w_scale) <= 0 and float(args.w_base_low) <= 0
                    and str(args.critic_inner_norm) == "group"):
                print("\n    *** WARNING: after the handoff NOTHING constrains "
                      "the output amplitude.\n"
                      "    MSE goes to 0, --w-base-low is 0, --w-scale is 0, and "
                      "the critic is\n"
                      "    scale-invariant (--critic-inner-norm group). This is "
                      "the exact\n"
                      "    configuration of bracket probes C and D, which reached "
                      "vel_rms_ratio\n"
                      "    5.8e5 within one eval interval of the handoff. Set "
                      "--w-scale > 0 or\n"
                      "    --critic-inner-norm none, or expect the same.\n",
                      flush=True)
        _reg = (f"WGAN-GP lambda {args.sr2_gp_lambda} every {args.sr2_gp_every}"
                if str(args.adv_loss) == "wgan_gp" else
                (f"R1 gamma {args.r1_gamma} every {args.r1_interval}"
                 if lazy_r1 is not None else "unregularised"))
        print(f"  critic: {sum(p.numel() for p in critic.parameters())} "
              f"parameters, lr {args.sr2_critic_lr}, betas {_dbetas}, {_reg}, "
              f"{args.sr2_critic_steps} D step(s)/G step, adversarial gradient "
              f"reaches G after {args.sr2_adv_warmup_steps} steps", flush=True)

    # --- the unsupervised tile pool for the kNN term ------------------------
    kw = dict(ng_hr=NG_HR, tile_hr=TILE, boxsize_mpc_h=BOXSIZE,
              dis_scale_mpc_h=float(scfg.dis_norm_kpc_h) * 1e-3,
              vel_scale_kms=float(pscfg.vel_norm_km_s))
    knn_cfg = MG.knn_config_from_args(args)
    knn_cls = MG.knn_class_ids(args)
    knn_loss_cls = () if bool(getattr(args, "knn_classes_diagnostic_only", False)) \
        else knn_cls
    unsup_train: List[Dict] = []
    unsup_hold: List[Dict] = []
    if want_knn:
        if float(args.w_knn_unsup) > 0 and knn_cfg.phase_lambda <= 0:
            print("WARNING: --w-knn-unsup > 0 with --knn-phase-lambda 0 trains "
                  "the POSITION-only statistic, which round 5b measured as "
                  "unable to rank the working arms (rho -0.23, p 0.32). 4000 is "
                  "the measured pass.", flush=True)
        if float(args.w_knn_unsup) > 0 and not knn_loss_cls:
            print("WARNING: --w-knn-unsup > 0 with no --knn-classes is the "
                  "POOLED term. Measured rho vs region subhalos: +0.878 "
                  "(p 4e-07) -- the WRONG SIGN. Hosts must stay out of it "
                  "(--knn-classes sub_50_200).", flush=True)
        rng = random.Random(int(args.seed))
        unsup_train = MG.build_tile_pool(train_boxes,
                                         int(args.unsup_tiles_per_box), cfg,
                                         geom, device, args, supervised, rng)
        unsup_hold = MG.build_tile_pool(hold_boxes,
                                        int(args.unsup_tiles_per_box), cfg,
                                        geom, device, args, set(), rng)
        MG.attach_knn_neighbours(list(unsup_train) + list(unsup_hold), cfg, args,
                                 kw, knn_cfg)
        MG.attach_knn_classes(list(unsup_train) + list(unsup_hold), args)
        if knn_loss_cls:
            print(f"  kNN term: 6-D (phase lambda {knn_cfg.phase_lambda}), "
                  f"ks {list(knn_cfg.ks)}, classes "
                  f"{[OWNER_CLASSES[i] for i in knn_loss_cls]}", flush=True)

    # --- resume -------------------------------------------------------------
    latest = out_dir / "latest.pt"
    step0 = 0
    if bool(args.resume) and latest.is_file():
        step0, rng_state, saved_norm = load_state(
            latest, model=model, critic=critic, opt=opt, opt_d=opt_d,
            device=device, expect_transform=str(args.density_transform),
            wants_critic=(critic is not None))
        if rng_state.get("sampler") is not None:
            sampler.rng.bit_generator.state = rng_state["sampler"]
        # Adopt the SAVED scales over the ones just fitted. The fit draws random
        # tiles, so every 12-hour resubmission would otherwise hand the critic
        # slightly different scales than it was trained under.
        if critic_norm is not None and saved_norm is not None:
            critic_norm.scale.copy_(torch.as_tensor(
                saved_norm["scale"], dtype=critic_norm.scale.dtype,
                device=critic_norm.scale.device))
            print("  adopted the checkpoint's critic scales over the freshly "
                  "fitted ones (the fit is random-tile, the critic is not)",
                  flush=True)
        print(f"\nRESUMED from {latest} at step {step0}", flush=True)
    elif bool(args.resume):
        print(f"\n--resume: no {latest} yet; starting from step 0", flush=True)

    total = int(args.shakeout_steps) if int(args.shakeout_steps) > 0 \
        else int(args.steps)
    if step0 >= total:
        print(f"already at step {step0} of {total}; nothing to do", flush=True)
        return {"ok": True, "step": step0, "done": True}

    # --- wandb --------------------------------------------------------------
    wcfg = dict(cfg.get("wandb", {}) or {})
    if args.no_wandb:
        wcfg["mode"] = "disabled"
    elif args.wandb_mode:
        wcfg["mode"] = args.wandb_mode
    wcfg.setdefault("group", "sr2_scratch")
    wcfg.setdefault("name", f"sr2_scratch-{args.label or 'base'}")
    cfg["wandb"] = wcfg
    cfg["scratch"] = {
        "init": "random", "scratch_lr": float(args.scratch_lr),
        "w_mse": float(args.w_mse), "w_sr2_adv": float(args.w_sr2_adv),
        "w_scale": float(args.w_scale),
        "mse_handoff_steps": int(args.mse_handoff_steps),
        "critic_inner_norm": str(args.critic_inner_norm),
        "w_gather": float(args.w_gather), "w_knn_unsup": float(args.w_knn_unsup),
        "w_radius": float(args.w_radius), "soft_prior": str(args.soft_prior),
        "density_valid_center": vc, "density_grid_mult": int(args.density_grid_mult),
        "density_transform": str(args.density_transform),
        "critic_norm_tiles": int(args.sr2_critic_norm_tiles),
        "adv_loss": str(args.adv_loss), "mse_mode": str(args.mse_mode),
        "w_base_low": float(args.w_base_low),
        "r1_gamma": float(args.r1_gamma), "r1_interval": int(args.r1_interval),
        "critic_spectral_norm": bool(args.critic_spectral_norm),
        "instance_noise": float(args.instance_noise),
        "adv_warmup_steps": int(args.sr2_adv_warmup_steps),
        "adv_start_epochs": float(args.adv_start_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "scratch_betas": [float(b) for b in args.scratch_betas],
        "term_warmup_steps": int(args.term_warmup_steps),
        "term_ramp_steps": int(args.term_ramp_steps),
    }
    use_wandb = maybe_init_wandb(cfg, out_dir, job_type="sr2_scratch")

    metrics_p = out_dir / "metrics.jsonl"
    gate_at = set(MG.gate_steps(args))
    set_rng = random.Random(int(args.seed) + 3)
    order = list(range(len(train_tasks)))
    random.Random(int(args.seed) + 5).shuffle(order)
    cursor = len(order)
    hps = max(1, int(args.hosts_per_step))
    sps = int(args.sets_per_step)
    hsps = int(args.host_sets_per_step)

    t0 = time.time()
    step_times: List[float] = []
    # Steps where the kNN minibatch produced no differentiable term at all. A
    # silent zero here would look like "the term is satisfied" in the logs.
    n_knn_skipped = 0
    print(f"\n=== training: steps {step0 + 1}..{total} "
          f"(warm-up {args.term_warmup_steps}, ramp {args.term_ramp_steps})",
          flush=True)

    for step in range(step0 + 1, total + 1):
        t_step = time.time()
        opt.zero_grad(set_to_none=True)
        ramp = term_ramp(step, warmup=int(args.term_warmup_steps),
                         ramp=int(args.term_ramp_steps))
        w_adv_now = (float(args.w_sr2_adv)
                     if (critic is not None
                         and step > int(args.sr2_adv_warmup_steps)) else 0.0)

        # ---- 1. the base objective ----------------------------------------
        batch = sampler.batch(int(args.base_batch))
        cand = forward_batch(model, batch, geom)
        mse = base_mse(cand, batch.hr)
        g_low = lowk_anchor(cand, batch.hr, geom.scale_factor)
        # `upstream` = map2map/train.py: MSE carries the generator until the
        # adversary starts, then the adversarial loss carries it ALONE. The two
        # are never summed there, and MSE keeps being logged because that is all
        # upstream uses it for (logging and ReduceLROnPlateau).
        if str(args.mse_mode) == "upstream" and critic is not None:
            w_mse_now = mse_handoff_weight(
                step, float(args.w_mse),
                adv_start=int(args.sr2_adv_warmup_steps),
                handoff=int(args.mse_handoff_steps))
        else:
            w_mse_now = float(args.w_mse)
        loss_base = w_mse_now * mse
        if float(args.w_base_low) > 0:
            loss_base = loss_base + float(args.w_base_low) * g_low["low"]
        # The amplitude leash. Independent of w_mse on purpose: it is what is
        # meant to still be holding the field's scale once MSE is gone, and the
        # adversary cannot do it (GroupNorm makes the critic scale-blind).
        g_scale = scale_anchor(cand, batch.hr)
        if float(args.w_scale) > 0:
            loss_base = loss_base + float(args.w_scale) * g_scale["scale"]
        fake_in = None
        if critic is not None:
            fake_in = scratch_critic_input(
                lr_core(batch.lr, geom), cand,
                cellsize_kpc_h=float(scfg.cellsize_kpc_h),
                dis_norm_kpc_h=float(scfg.dis_norm_kpc_h),
                grid_mult=int(args.density_grid_mult), valid_center=vc,
                allow_wrapped=allow_wrapped,
                density_transform=str(args.density_transform), norm=critic_norm)
            _in_std = instance_noise_std(
                step, sigma=float(args.instance_noise),
                batches=float(args.instance_noise_batches))
            if _in_std > 0:
                fake_in = add_instance_noise(fake_in, _in_std)
            if w_adv_now > 0:
                g_adv = (bce_generator_loss(critic, fake_in)
                         if str(args.adv_loss) == "bce"
                         else generator_adv_loss(critic, fake_in))
                loss_base = loss_base + w_adv_now * g_adv
        loss_base.backward()
        fake_det = None if fake_in is None else fake_in.detach()

        # ---- 2. the structure terms, ramped -------------------------------
        gather_rows: List[Dict] = []
        if want_gather and ramp > 0 and train_tasks:
            for _ in range(hps):
                if cursor >= len(order):
                    random.shuffle(order)
                    cursor = 0
                task = train_tasks[order[cursor]]
                cursor += 1
                loss_h, row = MG.host_loss(
                    model, task, geom, kw, mcfg, args, device, frozen,
                    MG.sample_sets(task, sps, set_rng),
                    host_set_indices=MG.sample_host_sets(task, hsps, set_rng),
                    draw=0, guard_ref="hr")
                (ramp * float(args.w_gather) * loss_h / hps).backward()
                row.pop("_adv", None)
                gather_rows.append(row)
        knn_pen = 0.0
        if float(args.w_knn_unsup) > 0 and ramp > 0 and unsup_train:
            m = min(int(args.unsup_tiles_per_step), len(unsup_train))
            sub = set_rng.sample(unsup_train, m) if m > 0 else unsup_train
            pen, _ = MG._unsup_knn(model, sub, geom, device, args, kw, knn_cfg,
                                   class_ids=knn_loss_cls)
            # `_unsup_knn` returns a CONSTANT zero when every tile in the
            # minibatch was skipped (no query survived the boundary buffer), and
            # backward() on a tensor with no grad_fn raises
            # "element 0 of tensors does not require grad". That killed knn_r5 at
            # step 11500. The odds of an all-skipped minibatch are what changed:
            # SC_UNSUP_TILES_PER_STEP went 8 -> 1 to make the term affordable, so
            # one unlucky tile is now the whole batch.
            if pen.requires_grad:
                (ramp * float(args.w_knn_unsup) * pen).backward()
            else:
                n_knn_skipped += 1
            knn_pen = float(pen.detach())

        if float(args.clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip))
        opt.step()

        # ---- 3. the critic -------------------------------------------------
        d_stat = None
        if critic is not None and fake_det is not None:
            with torch.no_grad():
                real_in = scratch_critic_input(
                    lr_core(batch.lr, geom), batch.hr,
                    cellsize_kpc_h=float(scfg.cellsize_kpc_h),
                    dis_norm_kpc_h=float(scfg.dis_norm_kpc_h),
                    grid_mult=int(args.density_grid_mult), valid_center=vc,
                    allow_wrapped=allow_wrapped,
                    density_transform=str(args.density_transform),
                    norm=critic_norm)
                if _in_std > 0:
                    real_in = add_instance_noise(real_in, _in_std)
            stats = []
            for ci in range(max(1, int(args.sr2_critic_steps))):
                if str(args.adv_loss) == "bce":
                    stats.append(bce_critic_update(
                        critic, opt_d, real_in, fake_det, lazy_r1=lazy_r1,
                        label_smoothing=float(args.adv_label_smoothing)))
                else:
                    stats.append(critic_update(
                        critic, opt_d, real_in, fake_det,
                        gp_lambda=float(args.sr2_gp_lambda),
                        do_gp=(step % max(1, int(args.sr2_gp_every)) == 0),
                        gp_every=int(args.sr2_gp_every)))
            d_stat = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
            d_stat["w_adv"] = float(w_adv_now)

        step_times.append(time.time() - t_step)

        # ---- snapshots, checkpoints, metrics -------------------------------
        if step in gate_at:
            snap = out_dir / f"tuned_step{step}.pt"
            MG.save_atomic({
                "model": {n: p.detach().cpu()
                          for n, p in model.state_dict().items()},
                "rung": "full", "cond": None, "step": int(step),
                "partial": True, "snapshot": True, "scratch": True,
                "init": "random",
            }, snap)
            print(f"  step {step:6d}  gate snapshot -> {snap.name}", flush=True)

        if step % max(1, int(args.save_every)) == 0 or step == total:
            save_state(latest, model=model, critic=critic, opt=opt, opt_d=opt_d,
                       step=step, args=args, critic_norm=critic_norm,
                       rng_state={"sampler": sampler.rng.bit_generator.state})

        if step % int(args.eval_every) == 0 or step == total or step == 1:
            nstd = noise_std_report(model)
            row = {
                "step": step,
                "wall_s": time.time() - t0,
                "s_per_step": float(np.mean(step_times[-int(args.eval_every):])),
                "ramp": float(ramp),
                "base_mse": float(mse.detach()),
                "w_mse_now": float(w_mse_now),
                "base_scale": float(g_scale["scale"].detach()),
                "rms_ratio_dis": float(g_scale["rms_ratio_dis"]),
                "rms_ratio_vel": float(g_scale["rms_ratio_vel"]),
                "base_low_k": float(g_low["low"].detach()),
                "base_low_dis": float(g_low["low_dis"].detach()),
                "base_low_vel": float(g_low["low_vel"].detach()),
                "base_vel_rms_ratio": float(g_low["vel_rms_ratio"].detach()),
                "noise_std_mean": float(np.mean(list(nstd.values()))),
                "noise_std": nstd,
                "knn_pen": knn_pen,
                "knn_skipped_steps": int(n_knn_skipped),
            }
            if d_stat is not None:
                row["adv"] = d_stat
            if gather_rows:
                row["batch_gather"] = float(np.mean(
                    [r["gather"] for r in gather_rows]))
            if want_gather and hold_tasks and (step % (int(args.eval_every) * 4) == 0
                                               or step == total):
                row["holdout"] = MG.evaluate_pool(
                    model, hold_tasks, geom, kw, mcfg, args, device, frozen,
                    "holdout")
            if hold_sampler is not None:
                row["holdout_base"] = holdout_base_block(
                    model, hold_sampler, geom, args)
            append_jsonl(metrics_p, row)
            print(f"  step {step:6d}  mse {row['base_mse']:.4f}  "
                  f"low_k {row['base_low_k']:.4f}  ramp {ramp:.2f}  "
                  f"noise_std {row['noise_std_mean']:.2e}  "
                  f"{row['s_per_step']:.2f} s/step", flush=True)
            MG.log_wandb(row, use_wandb)

    summary = {
        "ok": True,
        "step": total,
        "label": str(args.label),
        "out_dir": str(out_dir),
        "s_per_step_median": float(np.median(step_times)) if step_times else None,
        "s_per_step_p90": (float(np.percentile(step_times, 90))
                           if step_times else None),
        "peak_gpu_gb": (float(torch.cuda.max_memory_allocated() / 2**30)
                        if torch.cuda.is_available() else None),
        "peak_host_gb": MG.report_rss("at the end"),
        "noise_std": noise_std_report(model),
        "density_valid_center": vc,
        "density_transform": str(args.density_transform),
        "critic_norm": (None if critic_norm is None else critic_norm.to_dict()),
        "scratch": True,
    }
    name = "shakeout.json" if int(args.shakeout_steps) > 0 else "summary.json"
    write_json(out_dir / name, summary)
    print(f"\nwrote {out_dir / name}", flush=True)
    if use_wandb:
        finish_wandb()
    return summary


@torch.no_grad()
def holdout_base_block(model, sampler: PairedTileSampler, geom,
                       args, n_batches: int = 4) -> Dict:
    """The base objective on a box never trained on.

    The one number that says whether the from-scratch generator learned the
    super-resolution map or memorised the training boxes. Fixed noise seeds
    (``noise_seed_stride=0``), so successive evals are comparable.
    """
    mses, lows, vels = [], [], []
    for _ in range(int(n_batches)):
        b = sampler.batch(int(args.base_batch))
        c = forward_batch(model, b, geom)
        mses.append(float(base_mse(c, b.hr)))
        g = lowk_anchor(c, b.hr, geom.scale_factor)
        lows.append(float(g["low"]))
        vels.append(float(g["vel_rms_ratio"]))
    return {"mse": float(np.mean(mses)), "low_k": float(np.mean(lows)),
            "vel_rms_ratio": float(np.mean(vels))}


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    MG.validate(args)
    out = run(args)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
