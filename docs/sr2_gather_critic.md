# An HR critic on the member-gather fine-tune: the general form of the velocity term

**Scope.** A build note, not a result: this records adding a trained HR critic
(a 3D PatchGAN) as an optional **regulariser** on the member-gather objective,
default OFF, and the reasoning that says it is the right response to the held-out
gate's two unexplained failures. Nothing here has been trained or gated. The four
finished arms of `docs/sr2_member_gather_training.md` section 11 remain the
baseline, and an unset run (`--w-adv 0`) is byte-identical to them.

Numbers are tagged *measured* (read from an artifact on disk, all inherited from
`sr2_member_gather_training.md`), *derived* (arithmetic on them) or *design* (a
proposed rule, not yet run).

Depends on `docs/sr2_member_gather_training.md` section 11.6 for the defect this
targets, and on `src/cosmo_sr/dmsr/critic.py` for the critic, the hinge losses
and the lazy-R1 stabiliser, which are reused unchanged.

## 1. Why a critic, and why now

The member-gather loss supervises the **bound moments** of member sets -- virial
ratio, boundness, size, dispersion, centre -- and *nothing else*. The held-out
Rockstar gate (section 11) showed exactly the gap that leaves. The `self` arm
recovered HR's subhalo mass function out of sample (*measured*: base 20 -> tuned
366 vs HR 369), the first real win of this line, and in the same run:

- **collapsed velocity small-scale power 19-30x** (*measured*, section 11.6-3):
  the frozen field carries 1.02x HR of velocity power above `k_split` -- SR2 has
  this right -- and every arm ended between 0.034x and 0.053x. The tune *caused*
  this, and `vel_rms_ratio` (one global std) could not see it, reading 0.71
  throughout;
- **put its high-k displacement power in the wrong places** (*measured*, section
  11.6-2): the real-space correlation of `|Psi_{k>4}|^2` with HR fell from frozen
  SR2's 0.915 to 0.318 -- it built new small-scale concentrations at neither
  field's locations;
- **lost resolved hosts locally** (*measured*, section 11.4: -343 hosts in the
  edited region where HR wants *more*), the likely phase-space consequence of the
  velocity collapse.

Section 11.5's prescription was to **hand-craft a hinge per artifact**: a
two-sided velocity-power term (`--w-vel-highk`), a lower arm on the high-k hinge.
Those are built and correct, and each charges for **one named statistic** of one
band. The critic is the **general form of that idea**: instead of enumerating the
statistics a real field satisfies and a tuned one violates, it *learns* the
discriminating direction from whole HR tiles against tuned tiles, and penalises
whatever separates them -- including artifacts no one has thought to name.

This is not a new bet on adversarial training in the abstract. The
prior in this project is unfavourable (`sr2_member_gather_training.md` section 12
item 2: the window-gather adversarial objectives scored 0/43). The claim is
narrower and rests on **what the critic sees** (section 2): a field-realism
regulariser fitted to the exact band where the measured defects live.

## 2. What it sees, and why that is the whole design

The critic input is the **null-space (high-pass) of all six channels**:

    x_hp = x - U(A(x))          A = block-average by the degrade factor s
                                U = block (nearest) upsample by s

i.e. `null_projection(x, s)` applied to `(B, 6, N, N, N)`. Three properties, each
load-bearing:

1. **Velocity is in the input by construction.** Channels `[3:6]` are the
   velocity field, high-passed by the same operator as displacement. This is the
   one property that makes the critic worth adding over the hand-crafted velocity
   hinge -- it does not need to be told *which* velocity statistic to preserve,
   only that real high-pass velocity fields look a certain way and the collapsed
   ones do not. *Derived*: the 19-30x collapse is a gross distributional
   difference at exactly these frequencies, the easiest possible thing for a
   critic to key on.

2. **Displacement placement is in the input too.** The high-pass displacement
   field is where the 0.915 -> 0.318 placement defect lives. A patch critic scores
   *locally*, so it can charge "small-scale power, but not arranged like HR's"
   in a way a global power ratio (which is placement-blind, section 11.6-1)
   structurally cannot.

3. **The LR-resolvable coarse field is withheld.** `cand` and `hr` share the same
   LR tile, so their block-averaged (resolved) component is *identical* --
   `A(x_hp) == 0` exactly (pinned in the tests). Feeding it to the critic would
   spend its capacity discriminating structure that is byte-for-byte equal in real
   and fake, learning nothing about the unresolved detail. This is the same
   argument the DMSR critic makes for stripping `A_plus(y)` -- here it falls out
   of the high-pass for free, and needs no null-space operator.

**Normalisation.** The high-pass mixes displacement (Mpc/h) and velocity (km/s);
their raw RMS differ by orders of magnitude, and the first spectral-normed conv
cannot rescale them, so one group would starve the other of gradient (the failure
`CriticInputNormalizer` was written for). `GatherCriticNorm` fixes one scale per
channel, **measured from real HR high-pass tiles only** and applied identically to
real and fake -- per-batch statistics are forbidden, because normalising real and
fake differently is itself a discriminative shortcut unrelated to sample quality.

*Design, deliberately deferred.* A CIC **density** high-pass channel (the DMSR
critic's `rho_high`) is **not** added in this first draft: the displacement
high-pass already encodes placement, and a differentiable-CIC channel on a crop
brings the `valid_center` geometry with it. It is the obvious first extension if
the six-channel critic underperforms.

## 3. The training scheme

Standard simultaneous GAN, structured like the DMSR trainer's stages c/d/e but
wrapped around the gather loss instead of a flow-regression loss:

- **Generator step** (every step): `loss = gather + guards + w_adv * hinge_g(D(x_hp^fake))`,
  where `w_adv` follows a warmup+ramp (`adv_weight_at`): **0 for the first
  `--adv-warmup-steps`**, then a **linear ramp to `--w-adv`** over
  `--adv-ramp-steps`. Ramped, never a step change, because the fixed operator's
  adversarial gradient is weak next to the gather loss and a step change lets the
  critic yank the field before it has calibrated.
- **Critic step** (`--n-critic` per generator step): hinge D loss on the same
  hosts' **detached** fakes, plus **lazy R1** on the real inputs
  (`--critic-r1-gamma`, `--critic-r1-interval`). The fake is *reused* from the
  generator's forward rather than regenerated -- cheaper, and with a fixed
  operator an unbiased sample of what the generator now produces.

**Warmup trains the critic, not nothing.** Unlike the DMSR warmup (which freezes
the generator entirely), here `w_adv=0` during warmup only zeroes the
*generator's* adversarial term -- the generator still takes full gather steps and
the critic still trains on the resulting fakes. No gather steps are wasted; the
warmup just lets the critic calibrate against a moving generator before its
gradient is allowed to reach the weights.

*Derived*, the gradient bookkeeping that keeps the two optimisers from crossing:
the generator's `loss.backward()` accumulates stray grads on the critic's
parameters (the critic is in the adv term's graph), but the critic update calls
`opt_d.zero_grad()` before its own backward, so those are discarded and never
applied. The generator optimiser never sees the critic's parameters and vice
versa.

## 4. What it does NOT change

- **The feasibility verdict.** Every number this trainer prints is still a
  member-set surrogate it computes about itself; adding a critic does not make it
  a halo-finder result. The gate is unchanged: whole-box (or tile-splice)
  Rockstar on the held-out pool, `export_gather_tiles.py` ->
  `submit_gather_holdout_rockstar.sh`. The critic is a *regulariser on the way
  there*, not a new score.
- **The four arms on disk.** `--w-adv 0` is the default and is byte-identical to
  the finished runs, so the critic is a controlled addition, not a redefinition.
- **The checkpoint the gate loads.** `tuned.pt` is still `{"model": full state
  dict}` in the loader's format; the critic is saved separately as `critic.pt`
  and the gate never sees it (it regenerates through the generator alone).

## 5. Cautions, stated up front

1. **The prior is unfavourable and the critic could cheat differently.** A critic
   that is too strong, or fed the wrong view, degrades the field -- this is the
   0/43 history. The mitigations are structural (high-pass-only input, real-only
   normalisation, ramped weak weight, R1) but the outcome is unmeasured. The first
   run should be scored on the section 11.6 diagnostics (`velhighk_ratio`, the
   placement correlation from `highk_spectrum_diagnostic.py`) *before* a gate.
2. **It regularises, it does not supervise placement.** The critic can only charge
   "these small-scale arrangements are unlike HR's"; it cannot say "this subhalo
   belongs at that address", which is the address problem section 9.4 proved
   unlearnable from LR. Expect it to help the *distributional* defects (velocity
   power, local realism) and not the strict per-target recovery (7.4%).
3. **`self` + critic is the run to make, not `full` + critic.** The address arms
   already failed the gate destructively (section 11.2). The critic should be
   added to the arm that won -- `self` (or `nocentre`) -- not resurrect a losing
   one.

## 6. Module map

| file | role |
| --- | --- |
| `src/cosmo_sr/features/gather_critic.py` | the six-channel high-pass view (`highpass_field`, `gather_critic_input`) and its real-only normaliser (`GatherCriticNorm`) |
| `src/cosmo_sr/dmsr/critic.py` | reused unchanged: `HRCritic`, `hinge_g_loss`, `hinge_d_loss`, `LazyR1` |
| `scripts/features/finetune_member_gather.py` | `adv_weight_at` (the ramp); critic setup, the adversarial term in `host_loss`, and the critic update in the step loop, all behind `--w-adv > 0` |
| `scripts/slurm/member_gather_train_gpu.sbatch` | `MG_W_ADV` and the `MG_CRITIC_*` knobs, all default OFF |

Tests: `tests/features/test_gather_critic.py` (15) -- the high-pass strips the LR
band and keeps velocity, the normaliser is real-only/per-channel/fixed, the
adversarial gradient reaches the generator's output, and the warmup+ramp
schedule.

## 7. Reproduce

```bash
# Default OFF is the four-arm baseline. The critic is added to the arm that WON
# the gate (self), never to a losing address arm. Start small: a short run to
# read the section-11.6 diagnostics before committing to a full 8000 steps.
MG_CENTRE_MODE=self MG_RUNG=all_blocks \
  MG_W_ADV=0.1 MG_ADV_WARMUP=500 MG_ADV_RAMP=2000 \
  MG_LABEL=_self_critic SKIP_SHAKEOUT=1 \
  bash scripts/slurm/submit_member_gather_train.sh

# A single-score critic instead of the local PatchGAN, to police tile-scale
# statistics (e.g. the global velocity-power level) a patch critic averages over.
MG_CENTRE_MODE=self MG_RUNG=all_blocks \
  MG_W_ADV=0.1 MG_CRITIC_GLOBAL_POOL=1 \
  MG_LABEL=_self_critic_global SKIP_SHAKEOUT=1 \
  bash scripts/slurm/submit_member_gather_train.sh

# Then the held-out gate, unchanged, exactly as for the four arms:
EXPORT_ONLY=1 MG_ARMS=all_blocks_self_critic HG_MAX_HOSTS=1 \
  bash scripts/slurm/submit_gather_holdout_rockstar.sh
MG_ARMS=all_blocks_self_critic \
  bash scripts/slurm/submit_gather_holdout_rockstar.sh
```
