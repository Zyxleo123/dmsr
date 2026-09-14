# From one host to fifty-six: the member-gather objective gets a training run

**Scope.** Began as a build-and-setup note (sections 1-10); **section 11 is now
the result.** It records turning `docs/sr2_member_gather.md`'s free-field oracle
into a real generator fine-tune with a held-out pool, and then gating the
finished checkpoints on a box never trained on. **Headline (section 11): the
tuned generator reproduces HR's subhalo mass function on held-out set9** (base 20
-> tuned 366 vs HR 369, within R_vir of the example host), the first time this
line has moved a halo finder out of sample. The cost is a velocity guard failure
and a local host loss (section 11.4), and the fix is a loss term, not a new
experiment (section 11.5). Sections 9-10 record the surrogate-only phase and the
step-250 rankings they report were inverted by the finished runs.

Numbers are tagged *measured* (read from an artifact on disk), *derived*
(arithmetic on measured constants) or *design* (a proposed rule, not yet run).

Depends on `docs/sr2_member_gather.md` for the objective and its calibration
(ceiling 151/154, noise ±9, frozen 3/154, free field 72/154), and on
`docs/sr2_gather_finetune.md` section 8.1 for the gate.

## 1. The supervision was never box-limited

The line ran on set8 for months because set8 was the only box with an HR `owner`
array -- the per-particle "which halo binds you" map that member sets are slices
of. That read as a data constraint. It was a *missing job*.

*Measured.* Inventory across all 16 paired boxes:

| artifact | needed for | had it |
| --- | --- | --- |
| HR field `(6, 512^3)` | supervision | all 16 |
| LR field `(6, 64^3)` | generator input | all 16 |
| cached frozen SR2 box | baseline, and the reachable reference | all 16, seeds 0/1/2 |
| HR Rockstar catalog | subhalo positions and masses | all 16 |
| HR **`owner` array** | **the member sets** | **set8 only** |

*Measured*, from the two pre-existing `particles_report.json` files: an owner
array costs **Rockstar 15.4-16.8 min + owner stream ~1.8 min**, holds ~9.6 GB of
`.particles` ASCII transiently, and leaves 537 MB. `rockstar_particles.py
--write-assignment` builds it and correctly defeats its own reuse
short-circuit, so a box with tile weights but no owner array is rebuilt rather
than skipped.

*Derived.* Twelve boxes is ~3.6 CPU-hours. **The one-box limitation was ~18
minutes of unspent CPU per box.**

set9 is the instructive case: it had `members.npz` and `"particles_deleted":
true` but no owner array, because the run that produced it predated
`--write-assignment`. Nothing about that is visible from the file listing.

## 2. The build, and its consistency

*Measured*, 13 boxes (set0-set12; set13-15 stay sealed), throttled 2 at a time.

| check | result |
| --- | --- |
| `len(members) == num_p` | **exact for every object in every box** (e.g. 315409/315409) |
| `verify_frozen` | `True` for all 13 |
| unowned particle fraction | 0.4582-0.4607 |
| halo-finder time | 15.3-16.8 min |
| `.particles` cleanup | complete; no leftovers |

Two of these are load-bearing rather than routine:

**`verify_frozen: True` everywhere** means the member-id output flag
(`FULL_PARTICLE_CHUNKS = 1`) did not change the halo finding. Had it disagreed,
every catalog number in this project would have been in question. It did not.

**set8's rebuild reproduces its original numbers exactly** (315998 objects,
unowned 0.4603). That is the cheapest available check that the twelve new arrays
are the same kind of object as the one the oracle was measured against.

## 3. The pool

*Measured*, job 35689, one A5000, 19 min wall (~10 s per box to load owner +
catalog, the rest in per-host selection and one frozen forward each).

| | hosts | boxes | supervised sets | sets/host min/med/max |
| --- | ---: | --- | ---: | --- |
| train | 40 | set3-set7 | **5,396** | 91 / 134 / 194 |
| held out | 16 | set9, set10 | **2,164** | 105 / 137 / 159 |
| rejected | 0 | | | |

*Derived.* **35x the oracle's 154 sets**; the held-out pool alone is 14x.

The split follows `configs/reward/sr2_direct_finetune.yaml`'s own discipline.
set0-2 are excluded because they are SR2's paired training boxes; **set8 is
excluded from both sides** because it is where this entire line was developed --
it is retained as a reported anchor, since it is the only box with a calibrated
ceiling and therefore the only place to ask "how far short of the free field did
the generator fall on the identical problem".

## 4. The finding: live fraction is what binds, and host coverage is not it

This corrects an expectation the oracle's numbers had set.

*Measured.* Host-level **site coverage** -- the fraction of a host's Lagrangian
sites inside its 4 trained tiles -- is broad and sometimes poor:

| | min | median | max |
| --- | ---: | ---: | ---: |
| site coverage, train | **0.285** | 0.694 | 0.906 |
| site coverage, held out | 0.488 | 0.703 | 0.820 |

Per-**set** live fraction -- the fraction of a supervised subhalo's own members
inside those tiles, which is what the loss and the splice actually see -- is
not broad at all:

| | min | median | max |
| --- | ---: | ---: | ---: |
| live fraction, train | **0.939** | **0.981** | 1.000 |
| live fraction, held out | 0.962 | 0.979 | 0.992 |
| *oracle, for reference* | -- | 0.946 | -- |

**The worst-covered host in the pool is not a compromised target.**
`set6:h15933` (`logM` 14.96, the most massive host selected) holds only 28.5% of
its Lagrangian sites in its 4 tiles -- and its median live fraction is still
0.939, with 102 supervised sets.

*Derived.* This is the Lagrangian-purity result doing work
(`subhalos-are-lagrangian-pure`: a subhalo's members originate from a median of
**one** tile). A massive host's overall footprint sprawls across many tiles, but
each individual subhalo is compact in Lagrangian space, so the 4 tiles that hold
most of the host's *subhalos* need not hold most of the host's *mass*.

**Consequence for selection:** ranking tiles by host member-site count -- the
oracle's rule, kept unchanged -- selects well for subhalo coverage even when it
scores badly on host coverage. Site coverage is the wrong statistic to filter or
worry about; live fraction is the right one, and it is reported per host in
`pool.json`.

## 5. The pool is better conditioned than the oracle's single host

*Measured*, medians over hosts. "Reachable" is the hybrid HR-in-tile /
frozen-outside reference (`sr2_member_gather.md` section 2.2).

| median | oracle (set8 h271800) | train pool | held-out pool |
| --- | ---: | ---: | ---: |
| live fraction | 0.946 | **0.981** | 0.979 |
| reachable `r_rms` (Mpc/h) | 0.388 | **0.258** | 0.277 |
| reachable `sigma_v` (km/s) | 248 | **194** | 198 |
| reachable `2T/\|W\|` | 10.0 | **4.54** | 4.26 |
| reachable `bound_frac` | 0.534 | **0.650** | 0.673 |
| pure-HR `r_rms`, for contrast | 0.149 | 0.132 | 0.127 |

*Derived.* Higher live fractions leave fewer members stranded at frozen
coordinates, and stranded members dominate `r_rms` quadratically -- so the
reachable reference sits **closer to pure HR** than the oracle's did. The targets
are both more achievable and, if achieved, closer to true HR.

**Train and held-out are well matched** (`r_rms` 0.258 vs 0.277, `2T/|W|` 4.54
vs 4.26, `bound_frac` 0.650 vs 0.673). A held-out pool that were systematically
easier or harder would confound the only claim the run is being built to make.

## 6. Two scope limits of the pool as selected

1. **Cluster-scale only.** *Measured*: `--min-log-mvir 13.5` but
   `--max-hosts-per-box 8`, and the cap binds first -- every selected host landed
   at `logM >= 14.16` (train median 14.38, held-out median 14.41). set8 alone has
   101 hosts above 13.5 and 333 above 13.0, so the mass range is available; this
   run simply does not use it. Defensible as a first run, since the deficit
   scales with host mass (`sr2-deficit-scales-with-host-mass`), but the
   generalisation tested is across cluster environments, not across mass.
2. **Capacity is severe, and that is the experiment.** *Derived*: the oracle had
   6.3M **free** parameters for 154 sets, ~41,000 per set, optimised against the
   answer. Rung `fine` has 335,954 **shared** parameters for 5,396 sets, ~62 per
   set, applied at every site of every box -- a ~660x reduction in per-set
   capacity, and shared rather than free. *Design*: run the rung ladder
   (`fine` / `middle_fine` / `all_blocks`) as siblings, because a single null at
   `fine` cannot distinguish "the operator cannot express this" from "this rung
   is too small".

## 7. Two infrastructure defects, both found the expensive way

**Sourced env files must quote list-valued variables.** *Measured*, jobs
35593-35605: all 13 owner-array tasks died in seconds with
`<envfile>: line 6: set1: command not found`. The submitter wrote
`BOXES=set0 set1 set2 ...`, and to a **sourced** file that is not an assignment
-- it is "run the command `set1` with `BOXES=set0` in its environment". The
standing rule to pass configuration through one sourced env file rests on "a file
handles spaces, so no comma encoding is needed", which is true only if the value
is quoted *in the file*. Every earlier submitter escaped this because every one
of their values was a single word. Both gather submitters now quote every value
and verify the file round-trips through `env -i` **before** calling `sbatch`;
that check reproduces the exact failure on the old form. Blast radius was nil --
the failure precedes the halo finder, so nothing was written.

**The sbatch preamble does not carry `WANDB_API_KEY`.** *Measured*: it lives in
`~/.bashrc`, and the preamble deliberately builds a minimal environment and never
sources it, so every run would have silently downgraded to offline logging. The
job now extracts that one variable from `$HOME` at start. It is **not** written
into the env file: those live on world-readable shared scratch.

## 8. The first launch: three rungs, three OOMs, and a memory bound that was not one

*Measured*, 2026-08-23, jobs 35745-35748. The ladder of section 11 item 2 was
submitted and **every rung died inside the loss**, before the first optimiser
step, on three different cards:

| job | rung | GPU | allocated at failure | asked for |
| ---: | --- | --- | ---: | ---: |
| 35746 | `fine` | a5000, 23.55 GiB | 21.05 GiB | 682 MiB |
| 35747 | `middle_fine` | a5000, 23.55 GiB | 21.06 GiB | 682 MiB |
| 35748 | `all_blocks` | a6000, 47.40 GiB | 41.20 GiB | **2.70 GiB** |

All three at the same line, `member_gather.py`'s `r2 = (d * d).sum(dim=-1)`
inside `specific_potential_torch`. The pool build and the step-0 eval had already
completed, so nothing about the selection or the reference was at fault.

### 8.1 Why chunking was not a memory bound

*Derived, and the arithmetic closes on the observed numbers.* The potential was
chunked over rows with the comment "chunked over rows so the `N x N` block never
materialises". That is true of the **forward pass and false of the tape**.
Autograd saves `d` (`c x N x 3`) and `inv` (`c x N`), plus the `.clone()` the
diagonal zeroing needed, for **every** chunk until backward. So the live memory
was

    5 floats a pair x 4 bytes  =  20 bytes x sum_s N_s^2

independent of `pot_chunk` entirely. At the measured `sum_n_squared = 8.8e8`
(`sr2_member_gather.md` section 6.2, four tiles) that is **17.6 GB**, against the
21 GB the tracebacks report -- the rest being the generator's own activations.

The second number is the sharper one. A 2.70 GiB single-chunk allocation at
`chunk = 2048` decodes to **N = 118,000 particles in one set**: 2048 x 118000 x 3
x 4 bytes. The oracle never met one -- at four tiles on set8 its largest set was
~2e4 -- and the pool, which is 35x larger and spans five boxes, contains massive
satellites. That one set is `1.4e10` pairs, i.e. **~280 GB of tape on its own**.
No GPU in the cluster runs it, and no `pot_chunk` setting changes that.

**So the ladder was never a test of the rungs.** It is not evidence about
capacity, receptive field, or the objective.

### 8.2 The fix, and what it costs

`member_gather._SpecificPotential` is a custom `autograd.Function` that
recomputes each pair block in backward instead of saving it, with the gradient
written out rather than differentiated -- the expression collapses to a single
symmetric pass, because a position enters both as the field point of its own
`phi` and as a source in every other:

    phi_i    = -Gm sum_{j != i} u_ij,          u_ij = (r_ij^2 + eps^2)^{-1/2}
    dL/dx_a  =  Gm sum_j (g_a + g_j) u_aj^3 d_aj

*Measured* (`tests/features/test_member_gather.py`): the forward is unchanged to
1e-12 relative against the previous implementation written out, for both
softening kinds and at every chunk size; the analytic gradient matches autograd
through that reference to **3e-16 relative**; and `torch.autograd.gradcheck`
passes. **This is a memory fix, not an estimator change.** Three smaller wins
rode along -- the self term is subtracted as the constant `1/eps` it is under
both kinds rather than zeroing a diagonal (which cost a whole extra `N^2`
tensor), the gradient contracts with a `bmm` instead of a broadcast multiply
(another `c x N x 3` temporary), and `pot_max_elems` bounds a block in
**elements** rather than rows, so `min(2048, 2^24 / N)` gives the 118,000-particle
set 142 rows and a ~400 MB block instead of 2.70 GiB.

*Derived*: live memory goes from `O(sum_s N_s^2)` to one block. The cost is one
extra pass over the pairs, so the pair sums become a **time** budget rather than
a memory wall -- which is the regime the two knobs below were written for.

### 8.3 Three knobs, all off by default, and why they are off

The control has to stay the objective that scored 72/154, so nothing here is
enabled unless asked for. Each is reported in the job banner.

| knob | what it does | what it costs |
| --- | --- | --- |
| `MG_SETS_PER_STEP` | supervise `n` of a host's ~134 sets per step, drawn fresh | nothing -- the loss is a mean, so the gradient scale is unchanged. Taking all of them is a full-batch gradient over the expensive axis |
| `MG_MAX_SET_PARTICLES` | subsample a set to `K` particles | the estimator becomes **stochastic**. The pair sum is rescaled by `(N-1)/(K-1)`, which keeps `phi` and hence `2T/|W|` unbiased (pinned in the tests, including a test that the *missing* rescaling would move the virial by `N/K`); `bound_soft` is mildly biased and is a surrogate already |
| `MG_CENTRE_DEAD_ZONE`, `MG_CENTRE_HUBER` | dead zone and linear tail on the centre term | section 9 |

*Derived*, and it is the reason `max_set_particles` exists at all: at the pool's
set-size distribution a handful of massive satellites carry most of
`sum_n_squared`, and those are precisely the objects SR2 **already builds
correctly** -- `sr2_substructure_module.md` section 2.1 measured hosts above
~200 particles matching HR at mass ratio 1.03. Spending most of the compute
there is backwards. The shakeout now prints the bill (`report_pair_cost`), capped
and uncapped, before a GPU is spent on it.

*Also fixed, and pure waste*: `evaluate_pool` ran without `torch.no_grad`, so
every eval built a full autograd tape over 56 hosts' pair blocks and threw it
away.

## 9. Is the centre term learnable at all? The measurement that decides the design

**This is the open question the OOM interrupted, and it is prior to the rung
ladder.** Five of the six terms are moments about the set's *own* centroid --
`virial`, `bound`, `d6`, `r_rms`, `sigma_v`. They are invariant under
translation and permutation, which is to say they are **rules**, and a shared
convolutional operator can learn a rule. `centre` is a per-object **address**,
and section 3's pool asks for 5,396 of them from 335,954 shared parameters.

It is also load bearing: `sr2_member_gather.md` section 6 measured adding it as
8/154 -> 72/154.

*Measured*, and this is the number that motivates the whole question -- the
step-0 rows the three dead jobs did write before failing:

| median over sets, frozen SR2 | train (40 hosts, 5,396 sets) | held out (16, 2,164) |
| --- | ---: | ---: |
| `centre_offset_radii` | **5.59** | **5.90** |
| `2T/\|W\|` | 204 | 208 |
| `bound_hard` | 0.007 | 0.007 |
| `r_rms` / reachable | 2.99 | 2.97 |
| `sigma_v` / reachable | 2.49 | 2.40 |
| high-k / HR | 0.49 (max 1.14) | 0.51 (max 0.82) |

The pool is well matched across the split on every row, which is what section 5
predicted and is the precondition for reading any held-out number later.

**Note the frozen centre offset is 5.59 search radii, not 1.11.** The 1.11 in
`sr2_member_gather.md` section 6 is the *step-100* value of the free-field run,
after 100 optimisation steps; that run's own step-0 row reads **7.06** radii
(*measured*, its `metrics.jsonl`). So the term is asking each object to move of
order 1 Mpc/h, and the free field closed it with ~41,000 free parameters per set
while holding the answer.

### 9.1 The decomposition

`scripts/features/centre_offset_decompose.py`, one frozen forward per host and a
least-squares fit -- no halo finder, no optimisation. For each supervised set,
with `xbar_ref` the reachable centroid the loss targets and `x_host` the
cluster's centre:

    o      = xbar_frozen - xbar_ref            the offset the term must close
    rhat   = unit(xbar_ref - x_host)
    o_par  = o . rhat                          infall deficit, signed
    o_perp = |o - o_par rhat|                  the part with no direction

An **infall deficit** -- material that never fell far enough into the potential
-- is radial, systematic and a function of the environment, so it is learnable
and a convolution can express it. Isotropic scatter is realisation noise that LR
does not contain, and no architecture fits it. The isotropic null for
`sum o_par^2 / sum |o|^2` is exactly **1/3**.

Three rules are then fitted **on the training hosts and scored on the held-out
ones**, using only quantities available at inference -- clustercentric distance,
host mass, set size, never the answer: `none` (the frozen field), `radial` (one
scalar over the pool), and `regressed` (`o_par` linear in those three features).

### 9.2 What to read, and one trap

The verdict statistic is the **explained fraction** -- the share of the offset's
squared magnitude the rule accounts for, an `R^2` on a vector target. It is
deliberately *not* "fraction of sets inside one search radius", and the reason is
a calibration trap that a synthetic check surfaced before the job was written:
a search radius is `max(r_vir, 0.15)` Mpc/h, so an offset that is **80%**
systematic can still leave ~90% of sets outside one radius. Reading that column
as the verdict would report a large real effect as a null. It is kept in the
table as context, against the reference points that make it readable: the free
field 72/154 = 46.8%, frozen 3/154 = 1.9%, ceiling 151/154 = 98.1%.

| explained fraction, held out | reading |
| --- | --- |
| `< 0.2` | the offset is an **address**. A shared operator is charged for information its input does not carry, and no rung fixes that. Soften the term and put position on a self-consistency condition -- the clump sits at a local minimum of the candidate field's own potential -- instead of an address |
| `0.2 - 0.5` | **partly a rule.** Keep the term, soften its tail so the hopeless sets stop owning the gradient, and expect a generator between the frozen field and the free field |
| `> 0.5` | a **learnable rule**, a systematic infall deficit. The objective stands and the fine-tune's problem is capacity and receptive field |

*Design.* Two limits, stated because they cut opposite ways. The fitted rules are
linear in three scalars while a convolution sees the whole field, so this is a
**floor** on what is learnable rather than a ceiling. And it bounds the centre
term **alone**: nothing in it says whether the internal moments are reachable by
a shared operator, which is what the rung ladder is for. The gate stays real
Rockstar either way.

### 9.3 The result: the direction is a rule, the distance is an address

*Measured*, 2026-08-23, `centre_offset/pool/offsets.json`: 7,560 supervised sets
over seven boxes, 5,396 train and 2,164 held out. Reprint any time with
`python scripts/features/centre_offset_decompose.py --from-json <path>`.

| | train | held out |
| --- | ---: | ---: |
| sets | 5,396 | 2,164 |
| median \|o\| | 0.891 Mpc/h (5.68 radii) | 0.878 Mpc/h (5.65 radii) |
| **radial variance fraction** | **0.630** | **0.628** |
| isotropic null | 0.333 | 0.333 |
| median `o_par` | **-0.371** Mpc/h | **-0.379** Mpc/h |
| median \|`o_perp`\| | 0.475 | 0.456 |
| sets sitting further OUT than they should | 30% | 30% |

**Both halves of this table matter and they say opposite things.**

The offset is **twice as radial as chance** (0.628 against 1/3), `o_par` is
negative for 70% of sets, and the median is a consistent -0.38 Mpc/h. That is
not scatter. It is a signed, systematic **infall deficit** shared across
objects and across boxes: frozen SR2's material has not fallen far enough into
its host's potential. A deficit like that is a function of the environment, and
a shared convolutional operator can express it.

And then the rules table, which is the same rows fitted and scored out of sample:

| rule | explained fraction, held out | median residual |
| --- | ---: | ---: |
| `none` | 0.000 | 5.65 r |
| `radial` (one scalar, `a = -0.429`) | 0.115 | 5.12 r |
| `regressed` (`o_par ~ d_host, num_p, M_host`) | **0.134** | 5.03 r |

**13.4%.** By section 9.2's own decision rule, written before the number
existed, that is the `< 0.2` row: *the offset is an address*. The strongly
radial direction and the barely-predictable magnitude are consistent, and
together they are the finding: **the generator can be told which way to push,
but not how far.**

### 9.4 Three arms, and what the arithmetic already settles about two of them

Three ways to act on 9.3, all built and unit-pinned as of 2026-08-24. They are
`MemberGatherConfig.centre_mode` plus the two shaping knobs of section 8.3, and
**every one defaults to the term that scored 72/154**, so an unset run is
bit-for-bit the run section 6 of `sr2_member_gather.md` records.

| arm | what the term charges for | flag |
| --- | --- | --- |
| **full** | the whole offset to the reachable HR centroid | *(default)* |
| **radial** | only \|`o . rhat`\| -- the 63% that is a rule, dropping the transverse part no input determines | `--centre-mode radial` |
| **self** | the offset from the set's **own frozen centroid** -- "concentrate where SR2 already put you", zero at step 0, no address at all | `--centre-mode self` |
| **soft** | still `full`, but linear beyond `h` radii and free inside `d` | `--centre-huber-radii 2 --centre-dead-zone 0.3` |

**Two of the four are already decided, and no GPU was needed to decide them.**
What a term does not charge for is exactly what survives its own optimum, so the
residual offset at each arm's optimum is arithmetic on the rows `offsets.json`
already holds. `arm_residuals` computes it and the report prints it:

| `centre_mode` | median residual, held out | **inside 1 search radius** |
| --- | ---: | ---: |
| `full` | 0.00 r | **100%** |
| `radial` | 2.92 r | **12.3%** |
| `self` | 5.65 r | **3.6%** |

*Derived.* The free field **sees every address**, so it reaches its objective's
optimum; the `<= 1r` column is therefore a hard **ceiling** on that arm's
per-target free-field score. Against the measured reference points -- `full`
72/154 = 46.8%, frozen 3/154 = 1.9%, geometric ceiling 151/154 = 98.1%:

- **`radial` cannot exceed ~19/154 in the free field**, against `full`'s 72.
- **`self` cannot exceed ~6/154**, which is the frozen field.

So the answer to "do their oracles give better Rockstar numbers" is **no, and
the arithmetic says so without spending the gate**. Not because the ideas are
wrong -- because the *oracle is the wrong instrument for them*. Both arms
deliberately discard information, and the free field is defined as the run that
has all of it.

**The reason those arms exist is unaffected by that, and it is section 9.3.**
The comparison that matters is between GENERATORS, where the address is not
available:

| objective | free-field ceiling | what a generator can actually reach |
| --- | ---: | --- |
| `full` | 151/154 | its learnable content is the 13.4% of 9.3 -- the fitted rule lands **3.0%** inside one radius, *below the frozen field's 3.6%* |
| `radial` | ~19/154 | up to **12.3%**, if the network predicts `o_par` from the field better than a 3-feature linear fit does |
| `self` | ~6/154 | 3.6% by construction, and it asks for nothing it cannot have |

That is the case for `radial`: it is a **lower ceiling on a target that is
actually reachable**, against a higher ceiling on one that is not. It is a case
about generalisation, and only a generator run can test it.

**`soft` is the one arm with genuine upside at the oracle**, and it is the only
one worth a free-field gate. It does not change the optimum at all -- `full`'s
optimum is still offset zero -- it changes the gradient budget. Run 2 drove `dx`
to **0.00 radii by step 1300** and still scored 72/154 against a ceiling of 151,
so the residual 79 targets are **not** a position failure: they are the
concentration failure section 6.1 diagnosed (threshold-insensitive at the
ceiling, and worth 65 extra targets to run 2). At the measured median 5.6 radii
a quadratic charges ~31 with gradient ~11; a Huber knee at 2 radii cuts that
far-field gradient by ~10x and hands the budget to `virial`, `bound`, `d6`,
`r_rms` and `sigma_v` -- the five terms that own exactly that residual. Nothing
in the arithmetic predicts what that does, which is why it is the one to run.

## 10. The loss budget: what the other five terms actually get

*Measured*, 2026-08-24, the generator pool (40 train hosts / 5,396 sets, held-out
set9+set10). Section 9 asked whether the CENTRE term is learnable. This section
is the other half, and it was opened by adding per-term logging: the terms were
already all computed, and dropping them from the row had left every run with no
way to see where its gradient went.

### 10.1 The budget is set by the terms' dynamic ranges, not by the weights

Weighted term values at step 0, held out, at `w = 1, 1, 1, 0.3, 0.3` on
virial / bound / d6 / r_rms / sigma_v:

| step | virial | **bound** | d6 | rrms | sigmav |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 14.05 | **0.31** | 111.78 | 0.61 | 0.36 |
| 250 | 3.95 | **0.22** | 3.95 | 0.44 | 0.06 |
| 500 | 3.62 | **0.21** | 3.52 | 0.43 | 0.05 |

`d6` holds **88%** of the gradient at step 0 and `bound` holds **0.24%** --
and `bound` is the term that encodes **Rockstar's own decision rule**. Over 500
steps that collapsed `d6` 32x and `virial` 4x, `bound` moved 0.31 -> 0.21.

*Derived*, and it is not a judgement about importance -- it is hinge geometry:

```
bound:  [1 - x/ref]_+^2    x < ref   ->  CAPPED AT 1 for any x >= 0
d6:     [x/ref - 1]_+^2    x > ref   ->  unbounded
```

Which side of its reference the frozen field starts on decides the term's whole
dynamic range. `d6` starts ~11x above and charges 112; `bound` starts at ~1.3%
of reference and can never charge more than 1. The declared weights say the
terms matter about equally. The delivered budget is 11% / 0.24% / 88% /
0.5% / 0.3%.

### 10.2 The centre term is what was blocking the other five

*Measured.* Two runs from the identical step-0 row (`bound` 0.007, `2T/|W|`
208.0, `r_rms` 2.97, `dx` 5.90 r), so this comparison is clean. Held out:

| | `full` centre, 2500 steps | centre off, 750 steps | HR | oracle |
| --- | ---: | ---: | ---: | ---: |
| `bound_frac` | 0.061 @250 -> **0.052** | **0.137** and rising | 0.534 | 0.623 |
| `2T/\|W\|` | 51 @250 -> **71** | **32.7** | 10.0 | 10.0 |
| `r_rms` / HR | 2.81 @250 -> **4.72** | **2.21** | 1.00 | 1.00 |
| `sigma_v` / HR | **1.00** | 1.05 | 1.00 | 1.10 |
| `dx` (search radii) | 5.90 -> 5.32 | 5.90 -> **7.37** | -- | 0.00 |
| high-k ratio | 0.51 -> **0.87** | **0.030** | -- | -- |

**With the centre term in, everything except `sigma_v` peaks at step 250 and
then degrades.** `r_rms` goes 2.97 -> 4.72: the generator is *spreading* the
sets, worse than the frozen field it started from. High-k power climbs toward
the 1.5 guard. With the term removed every moment improves monotonically, and
does so at a fifth of the steps.

*Derived.* This is section 9.3's finding as a training dynamic rather than a
regression statistic. A shared operator handed a per-object target its input
does not determine cannot fit it and cannot ignore it either; the best it can do
is smear, and a smear is a real field change, not a no-op. It costs `r_rms`,
`bound`, `virial` and the high-k budget at once. The price of dropping the term
is the `dx` column: position drifts 5.90 -> 7.37 radii, which is exactly the
cost `centre_mode=radial` and `self` exist to recover.

### 10.3 What is solved, and what the residual is

Held out, centre off, step 750, against HR and against the free-field oracle:

| | frozen | generator | oracle | HR | reading |
| --- | ---: | ---: | ---: | ---: | --- |
| `sigma_v` / HR | 2.40 | **1.05** | 1.10 | 1.00 | **done**, and it beats the oracle |
| `2T/\|W\|` | 208 | 32.7 | 10.0 | 10.0 | two thirds of the way in log |
| `r_rms` / HR | 2.97 | 2.21 | 1.00 | 1.00 | about a third |
| `bound_frac` | 0.007 | **0.137** | 0.623 | 0.534 | **a quarter -- and it is the gate** |

Train/holdout on `gather` is 1.26x (6.2 vs 7.8 at step 500), against the 3.9x
divergence recorded with the centre term in. **A shared operator does generalise
on the moments.** That is the claim section 9 could not make and the rung ladder
existed to test, and it is the first evidence in this line for it.

The residual is concentrated in `bound`, which is both the least-satisfied term
and the one holding 0.24% of the gradient. Those two facts are the same fact.

### 10.4 Two levers, both off by default

Built and unit-pinned 2026-08-24. Every number above was measured with both
**off**, so an unset run stays comparable to everything already recorded.

**`--bound-penalty log`** replaces `[1 - x/ref]_+^2` with `[log(ref/x)]_+^2`.
Identical where it matters -- exactly zero at and above the reference, so the
anti-over-sharpening property the hinge exists for (`pilot_steps_2_4.md` step 4)
is untouched -- and unbounded as `x -> 0`. At the measured starting ratio of
1.3% of reference it charges **18.8 instead of 0.97**, with >10x the gradient,
and it is the same scale-free form `virial`, `r_rms` and `sigma_v` already use.

**`--term-norm`** divides each term by its value on the frozen field over the
TRAINING pool, measured once and held fixed, so the declared weights become the
actual budget. *Verified on the toy*: shares land on 21.7 / 21.7 / 21.7 / 6.5 /
6.5 / 21.7, exactly `w / sum(w)`. Held out of the held-out pool deliberately --
otherwise the scales are a channel from the held-out hosts into the training
loss.

Note these fix different things and compose: `log` fixes the term's **shape**
(gradient where the hinge is nearly flat), `--term-norm` fixes its **share**.
And `--term-norm` is also the fix for `d6`'s head start rather than a separate
knob -- dividing `d6` by its step-0 111.8 *is* removing the head start, which is
the same fact stated twice.

### 10.5 The four arms, in flight

*Measured*, step 250, held out. **Early and not a result** -- the `full` run of
10.2 also peaked at 250 and then degraded, so nothing here separates a trend
from a transient.

| arm | `bound` | `2T/\|W\|` | `r_rms` | `dx` | high-k |
| --- | ---: | ---: | ---: | ---: | ---: |
| centre off | **0.121** | **35.1** | **2.37** | 7.35 | **0.026** |
| `self` | 0.094 | 38.5 | 2.70 | **5.83** | 0.144 |
| `full` | 0.062 | 51.4 | 2.82 | 4.71 | 0.198 |
| `radial` | 0.052 | 62.4 | 3.27 | 5.26 | 0.286 |

Two things to watch, stated now so they are not read backwards later:

- **`self` is doing what it was designed to do.** It holds position at the
  frozen field's own level (5.83 against frozen 5.90) instead of drifting to
  7.35, while reaching better concentration than `full` on every term. An
  anchor, not an absence -- section 9.4's prediction.
- **`radial` is currently the worst arm, worse than `full`.** If that survives,
  the likely mechanism is that a projection permits unlimited TRANSVERSE motion
  at zero cost, so the operator can reduce `o_par` by moving material in a way
  `full` would have charged for -- more smearing, not less. That would be a
  real finding against 9.4's argument and it is not what the arithmetic
  predicted. It is 250 of 8000 steps; do not bank it either way.

## 11. The held-out Rockstar gate: the first real result of this line

All four arms **finished** on 2026-08-24 (8,000 steps; `centre off` 4,000), and
were gated on **set9, a box never trained on and never used to develop the
line** -- the first time anything in this line reached a halo finder out of
sample. Section 10.5's step-250 ranking inverted by the end, exactly as its own
caveat warned; the final surrogate rows are in
[[four-arms-failed-the-velocity-guard]].

**The gate exists now.** `scripts/features/export_gather_tiles.py` writes a
finished run's held-out tiles in the splice layout by replaying its
`summary.json` config through the trainer's own parser (verified bit-identical to
the trainer's held-out row: `gather` 30.11248, `bound_hard` 0.04483250,
`dx` 7.18459). `scripts/slurm/submit_gather_holdout_rockstar.sh` then runs the
existing splice -> Rockstar -> compare chain, one Rockstar box per arm plus one
shared frozen control on the same splice edges. The 8 held-out hosts of set9 own
32 tiles with no overlap (6.25% of the box), 1,127 supervised targets.

### 11.1 The win: the subhalo mass function is recovered

Within R_vir of the example host (168880, log Mvir 14.63), *measured* by real
Rockstar, binned by particle count:

| bin | HR | base SR2 | frozen ctrl | **`self`** | nocentre | radial | full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50-100p | 169 | 3 | 0 | **168** | 134 | 15 | 1 |
| 100-200p | 97 | 0 | 1 | **95** | 76 | 7 | 0 |
| 200-500p | 56 | 6 | 7 | **56** | 48 | 6 | 3 |
| 500-2000p | 34 | 8 | 5 | **33** | 34 | 1 | 0 |
| 2000+p | 13 | 3 | 5 | **14** | 9 | 1 | 1 |
| **total** | **369** | **20** | 18 | **366** | 301 | 30 | 5 |

`self` reproduces HR's subhalo mass function bin by bin, from a base of 20, out
of sample, from a shared convolutional operator. The frozen control (+1 target,
-2 subhalos against base) confirms the chain and the splice are clean, so the
gain is attributable.

### 11.2 The address arms failed at the halo finder, as 9.4 predicted

`full` produced **5** subhalos against a base of 20 -- it *destroyed* the ones
SR2 already had -- and recovered 26 supervised targets against the frozen
control's 30, i.e. **below doing nothing**. `radial` reached 30. The centre term
is not merely unhelpful at the gate; it is destructive, harder than the surrogate
said. This is section 9.4's arithmetic confirmed by Rockstar: the address is not
learnable, and charging for it corrupts the field. Only the position-abandoning
arms (`self`, `nocentre`) moved the halo finder toward truth.

### 11.3 Right objects, wrong addresses -- confirmed, and visible

Strict per-target recovery (subhalo *i* at GT position *i*) is **83/1127 = 7.4%**
for `self` against the control's 30/1127 -- a real gain but low, exactly the
`self` oracle ceiling (~frozen per-target, section 9.4). So the two readings
coexist and are the same fact: **`self` builds the right population of subhalos
in the right abundance and mass distribution, but not at the right individual
positions**, because position is not recoverable from the LR input. For a
statistical gate (abundance, SHMF, clustering) this passes; for a per-object gate
it does not, and cannot.

The figure job (`scripts/features/gather_holdout_figures_{data,plot}.py`,
`scripts/slurm/submit_gather_figures.sh`) makes this a picture. Zoomed to R_vir,
`self`'s 366 subhalos cluster in the central ~1 Mpc/h while HR's 369 fill the
whole R_vir disk -- the anchor pulls gathered material inward instead of
dispersing it to true subhalo sites. The frozen density is a smooth blob; the
tuned density has grown filaments and clumps.

The HR panel of `fig1_host_density` came out **blank**: the stage-1 data job was
run without `--hr-field`, so HR had subhalo markers on a neutral ground and no
density behind them, and the tuned field it would need to rebuild the whole npz
has since been purged. `gather_holdout_add_density.py`
(`scripts/slurm/submit_gather_add_density.sh`) patches the saved npz instead:

```bash
bash scripts/slurm/submit_gather_add_density.sh      # add HR + re-render
```

The panel has to be built the way its neighbours were, and the way is not
obvious: **these panels are not a spatial window**. Stage 1 selects the particles
of the host's four LAGRANGIAN tiles -- the exact material the operator edited --
and bins where they landed, over the tile union padded 4 Mpc/h
(`x[33.5, 66.5] x z[46, 79]`, 320 bins). `frozen_density` sums to exactly
`4 x 64^3 = 1048576` particles, `self_density` to 1048415: the 161 missing ones
drifted past the pad. A panel built by binning everything inside a box would
render a perfectly convincing picture of something else, so the job first
**re-derives the frozen panel from the frozen field and requires an exact match**
before it writes anything.

`fig1` now also puts **one colour scale across the three panels**, taken from HR.
Per-panel normalisation was hiding the thing the figure exists to show: an arm
renormalised to its own peak looks the same whether or not it has HR's mass.

**The R_vir zoom.** Those panels are 0.10 Mpc/h a pixel, so the host's R_vir
spans ~30 of them and cropping to the circle only enlarges a thumbnail. The
material survives at full resolution somewhere else, though: the export's
`tiles.npz` keeps every spliced tile for all three arms at 64^3 (`out` = tuned,
`frozen` = control, `hr` = truth, 545 MB), so the window can be **rebinned**
rather than cropped -- and those tiles are the only surviving source for the
tuned panel at all, its spliced field having been purged.
`gather_holdout_zoom_data.py` (`scripts/slurm/submit_gather_zoom.sh`) writes a
second data stem in the same key scheme, so the existing plot job renders it
with no new plotting code:

```bash
bash scripts/slurm/submit_gather_zoom.sh                          # 1.15 R_vir
GF_ZOOM_RVIR=1.5 GF_ZOOM_BINS=320 bash scripts/slurm/submit_gather_zoom.sh
```

It differs from stage 1 in one way that is a judgement, not a detail: it applies
a **depth cut**, by default the window's own half-width. Stage 1 projects the
full 25 Mpc/h Lagrangian column of the four tiles, which at 5 R_vir the host
dominates anyway; zoomed into the circle that column is mostly foreground and
background and washes out the substructure the figure exists to show.
`GF_ZOOM_DEPTH_RVIR=-1` restores the full column.

The same verification discipline applies, and here it carries more weight: before
writing, the job re-derives stage 1's own frozen **and** tuned panels from these
tiles, in stage 1's window with no depth cut, and requires an exact match. That
is what establishes both that the tiles reconstruct to the same particles the
3.2 GB fields gave, and that `out` is the arm stage 1 called `self`.

### 11.4 The damage: local, real, and pointing the wrong way

Every arm wrote **GUARD FAILED on `vel_rms`** (0.66-0.71 against a 0.10
tolerance) -- the one guard with no term in the loss (`host_loss` charges only
`w_low*low + w_highk*hk_pen`). So every number here is from a field that failed
its own velocity guard, and is a lower bound on what a clean run could do.

Box-wide, `self` vs base: **hosts>=200p -343**, subs>=50p +5035, n_halos +7343.
But the splice touches 6.25% of the box, so the box-wide-vs-HR comparison is
93.75% untouched frozen SR2 and says nothing. Restricting to the 32 spliced tiles
(dilated 2 Mpc/h) gives the like-for-like reference:

| in the edited region | base SR2 | **tuned `self`** | HR truth |
| --- | ---: | ---: | ---: |
| subhalos >=50p | 4,541 | **9,187** | 12,367 |
| hosts >=200p | 3,028 | **2,708** | 3,775 |

Two clean findings:

- **No excess.** Even inside the edited region `self` (9,187) stays *below* HR
  (12,367); it recovers ~59% of the local subhalo deficit, a move toward truth,
  not past it. There is no spurious over-production at any scale -- an earlier
  "7,000 spurious" framing was a box-wide scope error.
- **The host loss is real and local.** HR wants *more* resolved hosts than base
  (3,775 vs 3,028); tuning delivered *fewer* (2,708). In the region where truth
  has a host surplus the operator is destroying/fragmenting resolved hosts. This
  is the genuine cost, and the velocity term is the prime suspect: `bound` = 2T/|W|
  is cheapest to satisfy by cooling velocities globally, and only ~5% of the
  edited particles are in supervised sets, so the operator cools the field it
  cannot see. Whether the lost hosts sit inside the spliced tiles is not yet
  separately measured.

### 11.5 The next step is a loss term, not a new experiment

The critical path is no longer "which arm" -- it is `self` (or `nocentre`) plus
**a velocity term in the loss**: a two-sided hinge on tile velocity power against
the frozen field, mirroring the displacement guard, so the global-cooling exploit
that produced both the vel_rms failure and (likely) the host loss is charged for.
A second, cheaper lever is a LOWER arm on the high-k hinge referenced to *frozen*
rather than HR -- the current hinge is one-sided above HR, so `centre off`'s
0.507 -> 0.026 loss of high-k power was never charged. Neither is built. Rerun
`self` alone with the velocity term; that is the experiment that would turn "the
objective can build the right substructure" into "this checkpoint is usable".

## 11.6 What `self`'s high-k excess actually is

`all_blocks_self` won the gate and ended at **1.70x HR** displacement power held
out (worst host 3.87x, `--highk-max` 1.5). Measured on the tiles the gate already
exported (`highk_spectrum_diagnostic.py`, job 36394, 8 held-out set9 hosts, no
model and no GPU). Three results, and the third is the important one.

### 1. The scalar averages an overshoot and an undershoot into one number

The first guess -- that `sel.mean()` is dominated by the outermost shells, since
`k >= 4` admits 99.2% of a 64^3 tile's modes and 48% of them lie above Nyquist --
**is wrong.** HR's share of that mean per log bin:

| k h/Mpc | 4.4 | 5.2 | 6.2 | 7.4 | 8.8 | 10.4 | 12.4 | 14.8 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| share of the guard | 21.3% | 15.0% | 12.9% | 10.4% | 9.7% | 9.1% | 9.7% | 11.9% |
| share of the modes | 1.2% | 1.7% | 3.1% | 5.0% | 8.7% | 14.4% | 24.5% | 41.3% |

A 2.3x spread in power share against a 35x spread in mode count: `P(k) ~ k^-2`
very nearly cancels `N(k) ~ k^2`. **The scalar is a fair power-weighted
average.** That is exactly why it hides the defect:

| `P_cand/P_HR`, median | 4.4 | 5.2 | 6.2 | 7.4 | 8.8 | 10.4 | 12.4 | 14.8 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| frozen SR2 | 1.16 | 0.95 | 0.78 | 0.67 | 0.52 | 0.31 | 0.18 | 0.10 |
| **`self`** | **3.34** | **3.39** | **2.94** | **2.09** | 1.39 | 0.64 | 0.27 | **0.11** |
| `nocentre` | 0.35 | 0.25 | 0.19 | 0.14 | 0.09 | 0.05 | 0.02 | 0.01 |
| `full` | 4.05 | 2.92 | 1.84 | 1.02 | 0.54 | 0.26 | 0.12 | 0.06 |
| `radial` | 3.22 | 2.39 | 1.55 | 0.90 | 0.49 | 0.24 | 0.11 | 0.05 |

`self` is **+3.4x at the subhalo scale and 9x short at the grid scale**, and a
fair average calls that 1.59. The excess is at `k = 4-8`, not at Nyquist. One
number cannot say "far too much here, far too little there," and the one-sided
hinge cannot charge the second half at all.

### 2. The excess went to the WRONG PLACES -- this is the real defect

`r(k)` against HR is ~0.00 above `k_split` for **every** arm *and for frozen SR2*
(0.040 at k=4.4, falling to 0.004): mode phases at these scales are simply not
recoverable and never were the goal. The question is whether the small-scale
power is in the same *places*, which is a real-space question. Correlating the
high-pass map `|Psi_{k>4}(x)|^2` against HR's:

| | half-power volume | corr with HR | corr with frozen |
| --- | --: | --: | --: |
| HR | 0.058 | 1.000 | -- |
| frozen SR2 | 0.048 | **0.915** | -- |
| `nocentre` | 0.043 | 0.916 | 0.945 |
| `full` | 0.053 | 0.767 | 0.791 |
| `radial` | 0.053 | 0.776 | 0.795 |
| **`self`** | **0.033** | **0.318** | **0.336** |

**Frozen SR2 already puts its small-scale power in the right places** (0.915) at
0.46x the amplitude. `self` roughly tripled the amplitude and collapsed the
placement to 0.318 -- and to 0.336 against *frozen*, so it did not merely keep
SR2's addresses, it built new concentrations at neither field's. Half-power
volume 0.033 against HR's 0.058: fewer, sharper spikes. This is the same defect
section 11 read at Rockstar -- right mass function, 366 subhalos "clustered in
the central ~1 Mpc" where HR's fill `R_vir`, strict per-target 7.4%. **The high-k
excess and the address failure are one defect, not two.**

### 3. Every arm destroyed 95% of the velocity field's small-scale power

Velocity channels, power above `k_split`, against HR:

| frozen SR2 | `self` | `nocentre` | `full` | `radial` |
| --: | --: | --: | --: | --: |
| **1.023x** | 0.053x | 0.052x | 0.034x | 0.036x |

**The frozen field has this right** -- SR2's velocity small-scale power is HR's
to 2% -- and the tune took it to a **19-30x collapse**, on every arm. This is not
the inherited defect of `sr2-halos-are-sub-virial`; it is damage the fine-tune
did. `vel_rms_ratio` could not see it: that is one global std over the whole
tile, and it read 0.71 throughout. A halo finder works in 6-D phase space, so
smoothing the small-scale velocity field is precisely how bound substructure
stops being separable from its host -- which is the mechanism behind section 11's
unexplained host loss (HR wants 3775 hosts, base has 3028, tuning made 2708).

### What was built

All default OFF, so the four arms on disk stay the comparison baseline:

- `banded_highk_hinge` / `banded_power_ratio_torch` (`field_guards.py`) -- log
  bins over `k >= k_split`, each charged equally, with an optional dead zone and
  an optional lower arm. A one-bin call reproduces the scalar exactly.
- `--highk-bins N`, `--highk-k-max`, `--highk-tol`, `--highk-two-sided`,
  `--highk-reduce`; `highk_bands` logged per host and per eval row.
- `--w-vel-highk` / `--vel-highk-tol` -- a **two-sided** band hinge on velocity
  power, two-sided because the failure is a collapse and a one-sided-above hinge
  is exactly zero on all of it (measured: 0.0000 vs 6.96 at the observed 0.053x).
  `velhighk_ratio` is now measured and logged **unconditionally**, term or no
  term, because having no record of it is what let this run for four arms.
- `--highk-max-holdout` and `--vel-highk-min` extend `verdict()`, which read only
  the train row and had no velocity-power criterion at all.

### The arm this points to

`--w-vel-highk` is the highest-value change and is what section 11.5 asked for,
now with a measured target rather than a guess. For displacement, the excess is
at `k = 4-8`, so `--highk-bins 4 --highk-k-max 10` aims the hinge at it; do
**not** turn on `--highk-two-sided` for displacement, because the k>10 deficit it
would then charge is frozen SR2's own resolution limit (0.10x HR at k=14.8) and
would dominate the loss with a target the operator cannot reach.

What no high-k term fixes is result 2. Placement is the address problem of
section 9.4, and the 0.915 -> 0.318 collapse says `self` makes it *worse* than
doing nothing -- so a run with the velocity term should be scored on that
correlation, which is now cheap to compute from a tile export.

## 11.7 The box-wide high-k support (the guard's SUPPORT, not its shape)

`all_blocks_selfvel` (2026-08-25, `selfvel-arm-failed-the-gate`) settled two
things. The velocity term is a dead end: it lifted velocity high-k 4x (0.053 ->
0.205 held out) and moved the resolved-host count by **6 halos** while costing
32% of the subhalo population (367 -> 251, mostly the 50-100p bin). Drop
`--w-vel-highk`. And the displacement banded hinge showed the **same ~6x
train/holdout gap** the scalar had -- train max 0.56x, held-out bands 3.9 / 3.0 /
1.9 / 1.1 at k = 4.5 / 5.6 / 7.1 / 8.9. The defect was never the guard's SHAPE;
it is that its SUPPORT is the 40 training hosts' 160 tiles -- **6.25% of the
box** -- while a generator rewrites every site. The hinge overshoots exactly
where it cannot see.

### What was built (2026-08-25, all default OFF)

A pool of random **unsupervised** tiles, disjoint from every supervised host
tile, charging the SAME banded hinge box-wide against those tiles' own HR power.
Amplitude only -- it constrains the AMOUNT of small-scale power, never its
placement (r(k)~0 above k_split for every field, frozen included), so it does
nothing for result 2 by construction.

- `build_tile_pool` / `_forward_unsup` / `_unsup_bands` / `eval_unsup_highk`
  (`finetune_member_gather.py`) -- one `build_tiles` load per box, tiles held on
  the CPU (~15 MB each), a fresh minibatch each step `backward()`'d separately
  (cheap: a generator forward, no `N^2` potential).
- `--unsup-tiles-per-box` (0 = off), `--unsup-tiles-per-step`, `--w-highk-unsup`,
  and the verdict gate `--unsup-highk-max` on the worst band over the **held-out**
  unsupervised tiles -- the box-wide guard's own generalisation test, which the
  held-out HOST gate (`--highk-max-holdout`, still 6.25% of the box) cannot be.
- `unsup/train` and `unsup/holdout` high-k logged every eval; `verdict` names the
  held-out box-wide number when it fails. Env knobs `MG_UNSUP_TILES_PER_BOX`,
  `MG_UNSUP_TILES_PER_STEP`, `MG_W_HIGHK_UNSUP`, `MG_UNSUP_HIGHK_MAX`.

### The arm this points to

```
MG_CENTRE_MODE=self MG_W_VEL_HIGHK=0 \
  MG_HIGHK_BINS=4 MG_HIGHK_K_MAX=10 \
  MG_UNSUP_TILES_PER_BOX=16 MG_UNSUP_TILES_PER_STEP=8 \
  MG_W_HIGHK_UNSUP=10 MG_UNSUP_HIGHK_MAX=1.5 \
  MG_HIGHK_MAX_HOLDOUT=1.5 MG_RUNG=all_blocks MG_LABEL=_unsup \
  SKIP_SHAKEOUT=1 bash scripts/slurm/submit_member_gather_train.sh
```

`self` is the only centre mode that reached HR's mass function out of sample
(section 11); the velocity term is off (refuted); the banded hinge stays (cheap,
helped 1.70 -> 1.46); one-sided only, because the k>10 deficit is SR2's own
resolution limit. Score it on the held-out unsupervised high-k (did the box-wide
guard generalise, unlike the supervised one?), then the whole-box Rockstar gate.
Placement (result 2) is still out of reach for any high-k term.

## 12. What is not established

1. ~~**NOTHING HAS BEEN GATED.**~~ **Gated on held-out set9, section 11.** What
   remains unestablished is narrower and listed below. The original text stands
   as the reason the gate was built: every number
   in sections 9 and 10 is a surrogate scoring itself, and
   `tile-overfit-proxy-exploitation` measured such a surrogate reaching +255
   while the halo finder showed no gain at all. `bound_frac` 0.007 -> 0.137 is
   the loss's opinion of its own progress. Until a checkpoint goes through
   `evaluate_sr2_direct.py --checkpoint` -> Rockstar -> `compare_gather_catalog`,
   this line has moved a differentiable statistic and nothing else.
0. ~~**No run has finished.**~~ **All four finished 2026-08-24** (section 11).
   Section 10.5's step-250 caveat was right: the ranking inverted by the end.
0. ~~**`radial`'s case is now in doubt.**~~ **Settled against it.** `radial`
   produced 30 subhalos and `full` 5 at the held-out gate (section 11.2); both
   position-chasing arms lost to the position-abandoning ones. Section 9.4's
   arithmetic is confirmed by Rockstar.
2. **The prior is unfavourable and should be stated.** *Measured*, across this
   project: plain MSE plateaued at 0.39x frozen (`pilot_steps_2_4.md` section 2);
   the window gather scored **0/43** across runs A, B and C; its preserve variant
   **0/43**; the substructure module destroyed the field (904 halos against a base
   of 213,080). **The free field is the only thing in this line that has ever
   moved a halo finder**, and it saw the answer. The oracle was built precisely to
   separate "is the loss right" from "can the generator reach it" and answered
   only the first.
3. **The high-k hinge is untested at scale.** Built and unit-pinned against the
   numpy spectra (`features/field_guards.py`, 15 tests), but no training run has
   exercised it. It exists because the free field ran to 5.50x HR unguarded.
4. **The three-arm reading of section 9.4 is arithmetic, and the generator
   claim inside it is not tested.** The decomposition itself HAS run (9.3), and
   the free-field ceilings of 9.4 are exact. What is not established is the only
   thing that would justify `radial`: that a convolution predicts `o_par` from
   the whole field better than the 13.4% a three-feature linear fit manages. The
   fit is a floor, not a ceiling, and the distance between them is unmeasured.
   No arm has been trained and no arm has been gated.
5. **The knobs are unexercised at scale.** `--sets-per-step`,
   `--max-set-particles`, the centre shaping, `--bound-penalty log` and
   `--term-norm` are unit-pinned and all default to off, so every run recorded
   here is the objective the free field ran. `--term-norm`'s effect on the
   budget is verified arithmetically and on the toy; its effect on a real run is
   not measured, and rebalancing toward `bound` could as easily surface a new
   cheat as fix the deficit -- `bound_soft` is a surrogate for boundness, and
   giving a surrogate 20x more gradient is exactly how the window objective was
   satisfied by a pedestal (`sr2_gather_finetune.md` section 6).
6. **The held-out gate is a TILE SPLICE, not a whole-box regeneration.** Section
   11 splices the tuned held-out tiles into an otherwise-frozen box, so it
   measures "did the operator build the right substructure where supervised" and
   holds the other 93.75% fixed by construction. Collateral damage across a
   *fully regenerated* box -- the field the operator produces everywhere, not
   just in supervised tiles -- is still untested, and matters because every arm
   failed the velocity guard. Needs `evaluate_sr2_direct.py --checkpoint` for
   full-box generation through the tuned weights, then the same catalog+compare.
7. **The velocity guard is unbuilt, and it is the critical path.** Section 11.4:
   every gated arm failed `vel_rms` (0.66-0.71), the one guard with no loss term.
   The host loss was *hypothesised* to be its consequence -- but
   `selfvel-arm-failed-the-gate` refuted that (a velocity term moved the host
   count by 6 halos while costing 32% of subhalos). The host fix is now the
   supervised preservation guard of **section 13**, not a velocity term.
8. **The host-loss mechanism is not fully localised** -- and section 13 argues it
   no longer needs to be. Section 11.4 shows the -343 hosts are real and in the
   edited region; whether they fragment inside the spliced tiles or are cooling's
   reach beyond them is unmeasured, but a preservation constraint referenced on
   the frozen field charges for the damage either way. What is unrun is whether it
   recovers the count without costing subhalos (section 13's owed A/B).

## 13. The host preservation guard (`--w-host-sets`)

The collateral host loss of section 11.4 -- resolved hosts 3028 base -> 2708
tuned where HR wants 3775 (`gather-holdout-rockstar-gate`) -- has an obvious
supervised fix that the objective simply never expressed: **the loss is applied
only to subhalos.** `subhalo_home_tiles` selects `parent_ids >= 0`, so every set
in `member_gather_loss` has a parent, and nothing ever charges the run for
tearing a host apart. Item 7 named a velocity term as the likely cure;
`selfvel-arm-failed-the-gate` then **refuted** that -- the velocity-guard arm
improved velocity high-k 4x and moved the host count by 6 halos. Velocity cooling
is not the mechanism, so the fix cannot be a mechanism fix. A *supervised
preservation constraint* is the right kind of answer regardless of mechanism, and
it is the advisor's suggestion: apply our loss to the hosts too.

**What it is.** A second `MemberSets`, built from the resolved HOST halos
(`parent_ids < 0`, `num_p >= --host-min-num-p`) homing in a cluster's tiles, fed
through the identical `member_gather_loss`. Two design choices make it a *guard*
and not a second objective:

- **The reference is the FROZEN field, not HR.** `build_host_sets` calls
  `build_member_sets(..., base, base, ..., top_level=True)` -- both the reference
  and the straggler field are the frozen generator's own output. Against HR the
  hinges would start positive and *drive* hosts toward HR's concentration,
  competing with the subhalo objective for the same gradient. Against the frozen
  field, every term is **exactly zero at step 0** (the candidate IS the frozen
  field in-tile, and every out-of-tile member is a frozen straggler either way)
  and rises only as a host falls below where it started: unbinds, puffs up,
  drifts. *Verified* on the toy (`test_the_host_guard_is_silent_at_the_frozen_reference`,
  `test_the_host_guard_fires_when_a_host_is_destroyed`).
- **`centre_mode="self"`.** A host has no clustercentric direction of its own, so
  `radial` is undefined; the self-anchor charges "stay where the frozen field put
  you", which is what preservation means and is zero at step 0.

**Why it does not compete with the subhalo win.** While hosts stay intact the
term is ~0 with ~0 gradient (hinged), so the run that scored 366/369 keeps its
gradient budget until it starts damaging a host -- at which point the guard, and
only then, pushes back. The subhalo sets still *drive* (reference HR, 64x away);
the host sets *guard* (reference frozen, 0 away). One `--w-host-sets` scales the
whole host term; it reuses the subhalo per-term weights rather than a second copy
of six.

**Cost is the real risk.** A host is by construction bigger than the subhalos
inside it, and the potential is an `O(N^2)` pair sum, so `--w-host-sets` runs are
the pair-sum bomb of `member-gather-real-training-run`. `report_pair_cost` now
prints a **separate host bill** (capped and uncapped); `--max-set-particles`
applies to the host sets too and should be set before a GPU is spent.
`--host-sets-per-step` (64) minibatches them exactly as `--sets-per-step` does the
subhalos. Hosts are also less Lagrangian-pure than subhalos, so `min_purity` and
the live-fraction cut drop the spread-out ones -- correct, since this tiling's
particles genuinely cannot rebuild a host whose mass is mostly elsewhere. This
targets the ~3000 resolved hosts the A/B measures as destroyed, **not** the giant
central cluster host (mostly out-of-tile, and the OOM risk).

**Self-check when it runs.** `host_gather` and `host_bound_hard` are logged every
step; at the frozen start `host_gather` must read ~0. If it does not, the
reference is not the frozen field and something in the wiring is wrong.

**Status:** *design + built*, all off by default (`--w-host-sets 0` is
byte-identical to the four finished arms). Not yet run. The owed measurement is a
Rockstar A/B against the `self` arm: does the host count recover toward HR's 3775
**without** costing subhalos, the way the velocity term cost 32% of them?

```bash
# On the self arm, host guard on, with a particle cap for the pair sums:
MG_RUNG=all_blocks MG_W_HOST_SETS=1.0 MG_MAX_SET_PARTICLES=2048 \
  MG_LABEL=_hostguard SKIP_SHAKEOUT=0 \
  bash scripts/slurm/submit_member_gather_train.sh
```

## 14. The shell arm: the right distance from the host (`--w-radius`)

Section 11.4 counted the collateral host damage; `host-damage-is-mostly-reparenting`
found ~88% of it was bookkeeping, with 41 massive, intact hosts re-parented by
**central over-concentration**. The subhalo side of that same defect is now
measured directly. `scripts/features/subhalo_radial.py` pairs every Rockstar
subhalo with its host's centre on held-out **set9**, restricted to the dilated
tile region:

| r / host R_vir | HR | `self` |
| --- | ---: | ---: |
| median r / R_vir | **1.13** | **0.35** |
| < 0.25 R_vir | 593 | **6,496** |
| 0.25 - 0.5 | 2,187 | 2,673 |
| 0.5 - 0.75 | 3,487 | 1,600 |
| 0.75 - 1.0 | 3,846 | **932** |
| 1.0 - 1.5 | 6,545 | 1,065 |

An 11x excess in the innermost bin against a 4x deficit at the virial radius.
The tuned field puts subhalos in the middle and nowhere else.

### 14.1 What the supervised sets contribute to that, measured

Two numbers bound what a loss term can do about it, and they point in different
directions.

**The sets do not migrate.** Over the whole `all_blocks_self` run the supervised
sets' centroid offset never moves: 5.59 -> 5.53 search radii on train, 5.90 ->
5.93 held out, step 0 to step 8000. `centre_mode="self"` is doing exactly what
it was built to do -- anchoring each set where the frozen field put it -- so the
inner excess above is **not** the supervised sets being dragged inward. It is
mostly a different population: re-parented hosts and unsupervised material piling
up at the centre, which is `host-damage-is-mostly-reparenting`'s finding and is
what `--w-host-sets` targets. **This arm will not remove that pile.**

**But the anchor is anchoring the wrong radius.** Recomputing from
`centre_offset/pool/offsets.json` (7,560 sets; `r_frozen = |d_host rhat + o|`):

| | train | holdout |
| --- | ---: | ---: |
| median `r_frozen / r_true` | 0.974 | 0.978 |
| sets sitting too CLOSE | **58.3%** | 57.6% |
| median \|dr\| in search radii | **2.80** | 2.77 |
| p90 \|dr\| | 8.18 | 8.51 |
| \|dr\| as a fraction of \|o\| | **0.60** | 0.61 |

So the frozen field's own supervised sets carry a **signed, systematic inward
bias** -- small fractionally (2.6%), large against the gate's search radius (2.80
radii) -- and `self` pins them there for 8,000 steps. The radius is **60% of the
whole centre offset by magnitude**, and it is the half that is a rule: the
transverse remainder has no direction to predict from and a linear fit on
(`d_host`, `num_p`, `M_host`) explains 11.5% of it out of sample (section 9).

### 14.2 The term

`w_radius` charges `| |x_bar - x_host| - |x_target - x_host| |` in gate search
radii, through the same softened quadratic the centre term uses
(`_soft_quadratic`: dead zone, then `u^2`, then optionally linear). The set's
**distance** from its host is specified and its **direction is not** -- it may
sit anywhere on the sphere its truth defines. That is deliberate on both counts:
the shell is the learnable part, and a term that also demanded the direction
would be asking the generator for realisation noise.

`--radius-dead-zone` defaults to **1.0** search radius, which is the "not too
strict" of the ask: the gate is a threshold at one radius, so cost paid inside it
buys nothing there while competing with `virial`, `bound` and `d6` for the same
gradient. At 1.0 the term still charges 79% of frozen sets.

### 14.3 It conflicts with `centre_mode="self"`, and `self_tan` is the fix

`self` anchors the whole position, radius included, so the radial step this term
asks for is a step `self` charges for. Both are quadratic, so a radial move
`delta` from the frozen centroid costs

```
w_centre (delta/s)^2  +  w_radius ((|dr| - delta)/s)^2
```

which is minimised at `delta* = |dr| w_radius / (w_centre + w_radius)`. **At equal
weights the run closes exactly half the radius error it was added to close**, and
pays the centre term for the half it does close. Outbidding the anchor
(`w_radius >> w_centre`) buys the correction by dissolving the anchor.

Not merely derived: minimising the assembled loss numerically over a radial
sweep of the toy returns **0.500** of the correction under `self` and **1.000**
under `self_tan`, to three decimals, for both sets
(`test_the_two_terms_settle_at_half_the_correction_under_self`).

`centre_mode="self_tan"` removes the competition instead of outbidding it: the
self offset with its radial component projected out. The anchor keeps the two
transverse coordinates -- the ones this term says nothing about -- and the shell
term owns the radial one. Orthogonal by construction, `delta* = |dr|`, and the
step-0 value is still exactly zero. `centre_rhat` is taken at the *target*, so
the projection plane is a constant and not a function of the optimisation
variable.

The submitter warns before the queue if `self` and `w_radius` are combined; the
job echoes it again at startup.

### 14.4 What is refused, and where the term is off

- `w_radius > 0` with an all-zero `centre_host` **raises**. Without a host
  position the term would measure distance from the origin of the box; the
  radius *diagnostics* report `nan` in that case rather than 0, because a
  plotted zero reads as "the shell is already right".
- `centre_mode="self_tan"` with an all-zero `centre_rhat` **raises** -- it would
  silently be a plain `self` run, the one mode it is not.
- The **host preservation sets** (`--w-host-sets`) are forced to `w_radius=0` in
  both `host_member_config` and at the call site. A host has no host, and
  `build_host_sets` passes `host_pos=None` by design.

### 14.5 How to read it

`radius_ratio` (median `r_now / r_true`, pooled and per host) and
`radius_err_radii` are logged **at `--w-radius 0` too**, so the four finished
arms can be re-read against this one. A ratio drifting below 1 is the radial
collapse happening on the supervised sets themselves; a ratio held at ~0.974
while the catalog collapses is the pile coming from somewhere else. The verdict
is unchanged and is still the held-out Rockstar A/B (section 11): the number that
matters is whether the set9 profile above moves toward HR's, not whether this
term went down.

**Status:** *run, 2026-08-28, as `all_blocks_selfshell`* -- `self_tan`,
`w_radius=1.0`, `radius_dead_zone=1.0`, 8000 steps. **It failed the gate.**

```bash
MG_RUNG=all_blocks MG_CENTRE_MODE=self_tan MG_W_RADIUS=1.0 \
  MG_LABEL=_selfshell bash scripts/slurm/submit_member_gather_train.sh
```

`holdout_set9/verdict_vs_self.json`: **`PARTIAL: abundance LOST, hosts NOT
recovered`**.

| set9, host 168880 | base | `self` | `selfshell` | HR |
| --- | ---: | ---: | ---: | ---: |
| subhalos | 20 | 367 | **167** | 374 |
| in-tile subs >= 50p | 4541 | 9187 | **6019** | 12367 |
| in-tile hosts >= 200p | 3028 | 2708 | **2658** | 3775 |

55% of the abundance win, spent, with no host recovered. Section 14.6 is the
post-mortem and the replacement.

### 14.6 Why it failed: the address is not there (`--w-shell`)

The verdict says the arm was expensive. The **trajectory** says what it bought,
and it is the whole reason there is a second shell term:

| | step 0 | step 8000 |
| --- | ---: | ---: |
| train `radius_err_radii` | 2.860 | **1.318** |
| holdout `radius_err_radii` | 2.678 | **2.808** |
| holdout `radius_ratio` | 0.9744 | **0.9872** |
| holdout `centre_offset_radii` | 5.899 | 5.859 |

Train error halved. Held-out error went nowhere. Held-out *ratio* closed half
the systematic bias. So the per-set correction was **memorised**, and the only
component that transferred was the multiplicative one.

That is not a tuning failure, it is the target. A linear probe on
(`log d_host`, `log num_p`, `log Mvir`, `log n_live`), fit on the pool's train
hosts and scored on its holdout hosts:

| target | in-sample R^2 | **out-of-sample R^2** |
| --- | ---: | ---: |
| signed `dr` in search radii -- the units `w_radius` charges in | 0.025 | **0.022** |
| `log(r_frozen / r_true)` | 0.022 | **0.015** |

Section 14.1's "the radius is 60% of the offset and it is the half that is a
**rule**" is true of the **aggregate** -- 58.3% of sets sit too close, median
ratio 0.974 -- and **false per set**. Those are different claims and `w_radius`
was built on the wrong one: at ~2% explained it is *less* predictable than the
transverse direction section 9 rejected for being only 11.5% explained. The
network has far more capacity than a linear probe and sees the field, so the
probe is a weak bound -- but the held-out trace is the independent confirmation
that the network did not find the address either.

#### The soft shell

`--w-shell` asks for the component that transferred, and for nothing else. One
sort per call over a host's sets: sort the candidate radii, sort the target
radii, **pair by rank** -- the 1-D optimal transport map, exact and
`O(N log N)` -- and charge `log(r_cand_sorted[j] / r_tgt_sorted[j])^2`. No set
is ever named against its own target.

Measured on the same 7,560-set pool, sorted pairing against identity pairing
(train / holdout):

| | identity | sorted |
| --- | ---: | ---: |
| median `\|log ratio\|` | 0.164 / 0.143 | **0.079 / 0.064** |
| **signed** median -- the inward bias | -0.026 / -0.022 | **-0.041 / -0.026** |
| mean square | 0.345 / 0.320 | 0.071 / 0.061 |
| charged at `shell_dead_zone` 0.02 | -- | 81% / 78% |
| charged at 0.05 | -- | 61% / 55% |

The sort halves the magnitude while the **signed** bias, which is the entire
signal, survives intact. The monotone map removes **79% of the mean square**
where a per-host constant rescale would remove **1.9%** (the between-host share
of the variance; 98.1% is within-host), so what this term fixes is the **shape**
of the radial distribution, not its mean. The 5x drop in magnitude needs no new
weight: `--term-norm` divides each term by its own step-0 value, so `w_shell=1`
means what it says.

#### Three things that follow, and one that does not

* **Read `shell_resid`, not `radius_err_radii`.** A *perfect* distribution match
  still leaves **79% (train) / 85% (holdout)** of the per-set address error
  standing -- by construction, since the term never asks for the address. Gate
  on the address metric and a total win reads as a failure. `shell_resid` (and
  `shell_resid_signed`) are logged per step, per split and per host, at
  `--w-shell 0` too, so the finished arms can be re-read against this one.
* **`shell_dead_zone` is 0.02, not `radius_dead_zone`'s reasoning rescaled.**
  It is a *fractional* radius error, applied as `log1p(x)`. 0.05 would charge
  only 61%/55% of sets and eat most of a signal that is half the size of the one
  the radius term's dead zone was set against.
* **The candidate radius is a softened norm**, `sqrt(|d|^2 + eps^2)`, not
  `clamp_min(eps)`. A clamp is a gradient *plateau* below the floor and collapse
  toward the host centre is the one place this term must not have one. It does
  not bind on the frozen field (the smallest frozen radius in a host is a median
  0.142, min 0.041 Mpc/h); it is there for where the run can go.

What it does **not** fix: section 14.1's bound is untouched. The supervised
sets' own radius ratio is 0.974 -- 2.6% off -- while the catalog defect is a
median 0.35 host `R_vir` against HR's 1.13, a factor of 3. The pile is
re-parented hosts and unsupervised material, and no loss over the supervised
sets' radii can reach a population that is not in the sum. This term is an
**abundance-recovery** fix for what `selfshell` spent, not a fix for the radial
profile; section 16's isolation arm is what attacks the profile where it lives.

Its minimum is a *distribution*, so it is reachable by sets exchanging radii
along their own radial lines. Per-host `spearman(r_frozen, r_true)` is 0.92
median (0.62 worst) and 31% of sets sit more than `0.1N` from their own target
rank, so that freedom is real -- which is why both diagnostics are logged.
Distribution matched with the addresses rearranged is a different outcome from
both being right, and only the pair can tell them apart.

**Status:** *built, tested, not run.* `MG_W_SHELL=0` leaves the loss and its
gradient unchanged. The controls are `all_blocks_self` (w_shell 0) and
`all_blocks_selfshell` (the address version that failed).

```bash
MG_RUNG=all_blocks MG_CENTRE_MODE=self_tan MG_W_SHELL=1.0 \
  MG_LABEL=_softshell bash scripts/slurm/submit_member_gather_train.sh

# then the cheap held-out A/B (section 11.5) against the self arm -- 19 min,
# 3% of the train -- before reading anything else.
```

The submitter refuses nothing here but warns on both traps before the queue:
`--centre-mode self` pins the radii this term redistributes (use `self_tan`),
and `MG_W_RADIUS>0` alongside it puts the memorisation straight back in.

## 15. The small-halo arm: supervise on SIZE, not on the parent split

Every arm above supervised `parent_ids >= 0` -- HR's subhalos. Section 11.4
measured that this was aimed at half the problem. In the edited region:

| in the edited region | base SR2 | tuned `self` | HR truth |
| --- | ---: | ---: | ---: |
| subhalos >=50p | 4,541 | 9,187 | 12,367 |
| hosts >=200p | 3,028 | 2,708 | **3,775** |

HR wants **747 more top-level halos than base has**, and no supervised set in
any run has ever pointed at one of them. The subhalo cut encodes an assumption
that was never tested: *that the halos SR2 is missing are the ones Rockstar
files inside a bigger halo's radius.*

### 15.1 The question the arm cannot answer for itself

"Top-level halo" spans four decades of mass, and the 747 has two readings that
lead to different runs:

| reading | signature | what it implies |
| --- | --- | --- |
| **small-halo deficit** | `base/HR` climbs with mass | a `num_p` WINDOW is the fix; the objective is right and is pointed too narrowly |
| **flat deficit** | `base/HR` roughly constant | SR2 loses halos at every scale; a window buys nothing over the subhalo cut and the missing thing is not object size |

Neither is established. **Run the census before choosing the ceiling** -- the
ceiling *is* the hypothesis:

```bash
bash scripts/slurm/submit_halo_gap_census.sh          # ~minutes, CPU
```

`scripts/features/halo_gap_census.py` reads the saved ASCII catalogs (no field,
no halo finder), counts every halo by population x `num_p` bin x environment,
and prints a verdict naming which of the two readings the data supports.

### 15.2 The third axis, which decides the TILING rather than the population

A count split by mass alone cannot say whether the missing top-level halos are
*reachable*. The fine-tune's supervised tiles are the top four Lagrangian tiles
of massive clusters -- proto-cluster patches -- so a deficit sitting in
genuinely isolated field halos is out of reach of a wider **population** and
needs a wider **tiling**, which is a different change with a different cost.

So the census also labels every top-level halo by its distance to the nearest
more massive resolved host, in units of that host's `R_vir`:

| label | range | why the bin exists |
| --- | --- | --- |
| `inside` | < 1 R_vir | top-level in the catalog but inside a bigger halo's virial radius. `host-damage-is-mostly-reparenting` measured ~88% of `self`'s host loss to be objects crossing exactly this line, so a difference here is most likely bookkeeping |
| `near` | 1-3 R_vir | the infall region: a cluster's own Lagrangian tiles plausibly hold this material |
| `field` | > 3 R_vir | isolated. Reachable only by changing which tiles are trained |

These are **counts in bins, never matched objects**: two Rockstar catalogs of
two different fields share no halo ids. `scripts/features/host_forensics.py`
remains the matched-descendant tool and answers a different question (what
happened to a *given* base host). A bin count is the right estimator for a mass
function and the wrong one for a fate; the two are not mixed.

### 15.3 The term

Nothing in the loss changes. What changes is which halos it is pointed at --
three arguments, all defaulting to the finished arms' behaviour byte for byte:

| flag | env | meaning |
| --- | --- | --- |
| `--population {sub,top,all}` | `MG_POPULATION` | which half of the catalog, or both. `sub` is every recorded arm |
| `--max-num-p` | `MG_MAX_NUM_P` | exclusive upper edge of the supervised `num_p` window; 0 is no ceiling |
| `--set-select {largest,random,stratified}` | `MG_SET_SELECT` | how `--max-sets` chooses when it binds |

**`--max-num-p` is required with `top`/`all`, and both the job and the submitter
refuse without it.** Without a ceiling the selection includes the cluster host
itself: ~100k particles, so `O(N^2)` is 1e10 pairs against a 200p set's 2e4, and
its HR reference asks the generator to *concentrate a 1e14 halo* -- a different
objective from this one, and one `--w-host-sets` already handles against the
**frozen** reference (section 13). The two references are the whole distinction:
the supervised sets are charged against HR because the run is meant to *build*
them; the host guard is charged against frozen because it is meant to *preserve*.

**`--set-select` is the trap.** The cap has always kept the largest sets, which
is harmless for a subhalo run and exactly wrong for a mass-window one: the pool
of a cluster region *is* a mass function, so `largest` keeps the top of it and
drops precisely the small halos the arm exists to build. A `--population all`
run left on the default cap can select almost the same sets as before and look
like it did something. `stratified` fills `log2(n_sites)` bins round-robin;
the submitter warns when the cap binds under `largest`, and the pool banner
prints the population mix, the `num_p` quantiles and the drop counts so a
shakeout shows which of the two happened.

Two ways the run can silently not be the run it was configured to be, both
visible in `pool.json` and in the shakeout log:

* **the cap**, above -- `n_dropped_by_cap`, `set_select`, and the `num_p`
  quantiles.
* **the live-fraction cut.** A field halo whose Lagrangian material is mostly
  outside the cluster's four tiles is dropped at `min_live_frac`. That is
  correct -- these tiles genuinely cannot build it -- but it means the reachable
  field population can be far smaller than the catalog's, and that is a fact
  about the **tiling**, answered by `--n-tiles`, not by the loss. A shakeout
  whose `n_top_level` is 0 under `--population all` is a `sub` run with extra
  arguments, and says so.

### 15.4 How to run it

```bash
# 1. the census FIRST. Its verdict sets MG_MAX_NUM_P.
bash scripts/slurm/submit_halo_gap_census.sh

# 2. the shakeout: does the widened population actually select field halos in
#    a cluster's tiles, or does the live-fraction cut take them all?
MG_POPULATION=all MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified MG_MIN_NUM_P=50 \
  SHAKEOUT_ONLY=1 bash scripts/slurm/submit_member_gather_train.sh
#    read: n_top_level, num_p quantiles, n_dropped_by_cap, n_dropped_live_frac

# 3. the arm. `self` + the host guard is the control's objective, unchanged;
#    the population is the only difference, which is what makes the A/B read.
MG_POPULATION=all MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified MG_MIN_NUM_P=50 \
  MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0 MG_LABEL=_smallhalo \
  SKIP_SHAKEOUT=1 bash scripts/slurm/submit_member_gather_train.sh
```

The control it must beat is **`all_blocks_selfhostguard`** -- the same objective
with `population=sub` -- on the same held-out gate. Two questions, and the
second is the one that can kill it:

* **Q1** did the top-level count in the edited region move toward HR's 3,775?
* **Q2** did the subhalo mass function `self` already recovered (9,187 against
  HR's 12,367) *deteriorate*? The supervised budget is now split across a wider
  population at the same `--max-sets`, so paying for field halos with subhalos
  is the failure mode to look for, and `gate_verdict_vs_baseline.py` is already
  the tool that asks both.

### 15.5 What this does not do

`--population all` changes **which objects are supervised**, not **where**. If
the census returns its `field` bin as the bulk of the deficit, this arm cannot
reach it and the honest next step is a tiling change, not a weight. Nothing here
touches the velocity defect either: `highk-guard-is-blind` measured every arm
destroying 95% of the velocity field's small-scale power, and a wider population
neither helps nor is charged for that.

## 16. What the small-halo arm measured, and the two things it says to fix

The arm of section 15 ran (job 39083, 8,000 steps, ~6.9 h on an a6000) and was
gated on both held-out boxes with real Rockstar. Section 15's own reading was
"Q1 failed". The census, re-run with the arm's catalog in it
(`halo_gap_census.py --kinds base,hr,smallhalo --run-dir <holdout export>`),
says something sharper. Set9, **region scope** (32 tiles, dilate 2), summed
over every mass bin:

| population | env | base | `smallhalo` | HR | deficit before | after |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| top-level | `inside` | 7 | 21 | 21 | 14 | 0 |
| top-level | `near` | 670 | 575 | 2,819 | 2,149 | **2,244** |
| top-level | `field` | 9,871 | 9,066 | 11,580 | 1,709 | **2,514** |
| sub | `inside` | 2,064 | 3,912 | 4,264 | 2,200 | 352 |
| sub | `near` | 1,229 | 1,573 | 5,637 | 4,408 | **4,064** |
| sub | `field` | 1,248 | 3,316 | 2,466 | 1,218 | **-850** |
| **all** | | 15,089 | 18,463 | 26,787 | 11,698 | 8,324 |

Three facts, none of which is "the arm did nothing".

1. **It built objects, and Rockstar filed every one of them as a subhalo.**
   Net +3,374 in the region: subhalos +4,260, top-level **-886**. The top-level
   deficit the arm exists to close went *backwards*, 3,872 -> 4,758.
2. **It overshot in the field and stopped there.** Field subhalos 3,316 against
   HR's 2,466, and above 200p 625/269/155 against HR's 328/108/72 -- 2-3x too
   much substructure inside sub-200p hosts. Meanwhile field *top-level* fell by
   805.
3. **The deficit is in the infall region, and the arm never reached it.**
   `near` holds 6,557 of the region's 11,698 missing objects (56%); the arm
   closed 249 of them, 3.8%. `near` top-level at 50-100p is base 252 /
   arm 229 / HR 1,547.

### 16.1 Why: the loss is intra-set and "top-level" is inter-set

Every term in `member_gather.py` -- virial, bound, d6, r_rms, sigma_v, centre,
radius -- is a moment of the members about their own centroid, or that
centroid's address. Rockstar's top-level/subhalo decision is **not a property of
the members at all**: it is whether some other, more massive halo's `R_vir`
contains this centre. So an intra-set loss pointed at top-level HR halos scores
full marks on an object that a pedestal then swallows, which is fact 1 stated as
a mechanism. Pointing the *same* loss at a *different* population could not have
produced top-level halos, and the run is the measurement of that.

`d6` is the only existing term that looks outside the set, and it does not close
this. It normalises by the local background's **dispersion** (`sx`, `sv` in
`set_statistics_torch`), not its **density**, so a set can be maximally compact
against its neighbourhood's position and velocity spread while sitting in a
growing mass pile. It is also one-sided (`_hinge_above`), as is `bound`
(`_hinge_below`), so over-concentration is free in both.

This is the same defect as `host-damage-is-mostly-reparenting` (~88% of `self`'s
host loss was objects crossing the parent split with no particle lost) and as
the radial collapse the shell arm was built for and failed to move (section 14).
Those are one emergent pedestal seen from three sides. Section 14's post-mortem
concluded that the collapse "lives in the emergent density field ... NOT in the
supervised member sets, so a placement loss on those sets cannot touch it". That
is right about *placement*. A **density** loss on the material around those sets
can, and that is the term below.

### 16.2 The isolation term (`--w-isolation`)

The soft count of local **non-member** particles inside a fixed radius of the
set's centroid, hinged above HR's own count **on the same particle ids**.

| flag | env | default | meaning |
| --- | --- | ---: | --- |
| `--w-isolation` | `MG_W_ISOLATION` | 0 | weight; 0 is every arm so far |
| `--iso-radius-frac` | `MG_ISO_RADIUS_FRAC` | 0.5 | counting radius, as a fraction of the background sphere |
| `--iso-softness` | `MG_ISO_SOFTNESS` | 0.15 | sigmoid width, as a fraction of that radius |
| `--iso-min-count` | `MG_ISO_MIN_COUNT` | 8 | minimum HR count for a set to be *charged* |

Five choices, each of which is the answer to a way this could have been gamed
or made meaningless:

* **id-matched, not field-matched.** The background rows are chosen once, on the
  frozen field, inside `bg_radius_factor * r_ref`; the candidate and the HR
  reference count *those same rows*. The `bg_k` subsample fraction therefore
  cancels exactly, and the ratio is a statement about the same material rather
  than about two different samples.
* **the centroid is detached.** The gradient reaches the background particles
  only. The term must push the surrounding mass out to where HR has it, not let
  the set run away from the pile -- a live centroid would permit the second, and
  would put this term in exactly the tug-of-war with `centre_mode="self"` that
  `w_radius` lost in section 14.
* **the radius is a constant of the data.** Derived from `r_ref`, HR's own rms
  radius for the set, and stored per set at build time. Deriving it from the
  *candidate's* radius would let a run satisfy the term by shrinking the sphere
  instead of clearing it, which is the section-6 window-loss gaming failure of
  `docs/sr2_gather_finetune.md` in a new costume.
* **hinged above.** Matching or beating HR costs exactly zero with zero
  gradient. The module's standing property -- the loss can never ask for more
  than HR has -- is kept.
* **sparse sets are skipped, not charged.** A ratio taken on three background
  particles is the sampler's seed, not a measurement. `n_iso` reports how many
  sets the term could be taken on; an arm whose `n_iso` is a fraction of
  `n_sets` is not running this term on the pool it thinks it is.

**Known limit, and it is the one to watch.** The background ids are fixed on the
frozen field inside `bg_radius_factor` (4) set radii, so mass arriving from
*further out* is invisible to the term. If the pedestal the gate reports is
wider than that sphere, the fix is `MG_BG_RADIUS`, not `MG_ISO_RADIUS_FRAC` --
the counting radius must stay well inside the sampled one or it is counting a
sampling boundary. The submitter prints this note whenever the term is on.

The `iso_ratio` diagnostic is measured and logged **at `--w-isolation 0` too**,
per host, per population and per environment. The count is a sum over at most
`bg_k` particles against a step dominated by `O(N^2)` pair blocks, so being
blind here would cost nothing and buy nothing -- and being blind to exactly one
un-penalised quantity is how the four arms of section 11.5 failed.

### 16.3 The environment axis (`--env-select`)

`--population` chose *which half of the catalog*, `--max-num-p` chose *which
sizes*. Neither can express *which surroundings*, and the census says that is
where the deficit is.

| flag | env | default | meaning |
| --- | --- | ---: | --- |
| `--env-select` | `MG_ENV_SELECT` | `any` | `any`, or a comma list of `inside` / `near` / `field` |
| `--env-host-floor` | `MG_ENV_HOST_FLOOR` | 200 | particle floor for a "resolved host" when labelling |

The labels are `subhalo_gather.halo_environment`, which is now the **single**
definition: `halo_gap_census.py` imports it rather than carrying its own copy,
so the environment the census measures a deficit in and the environment an arm
is pointed at are the same quantity by construction. `member_pool`-style
per-host recomputation is avoided by computing it once per box in `build_pool`
(one KD-tree over ~100k halos, seconds).

Three details that decide whether the run is the run it was configured to be:

* **the cut runs BEFORE the cap.** Capping first would spend the whole
  `--max-sets` budget on the pool as it stands -- which a cluster region's field
  halos dominate -- and hand back a handful of `near` sets. `n_after_env`
  against `n_after_purity` in `pool.json` is how much the cut took.
* **`stratified` now bins on (mass x environment).** Mass alone is not enough
  once the pool spans environments, for the same reason `largest` was not enough
  once it spanned masses.
* **the host preservation guard is NOT narrowed.** `host_member_config` forces
  `env_select="any"` back on: a run supervising only the infall region still
  damages hosts it is not being asked to build, and narrowing the guard with the
  objective would remove the protection exactly where the new work is happening.

`field` is the bin this cannot help with. A genuinely isolated halo whose
Lagrangian material is mostly outside a cluster's four tiles is dropped at
`min_live_frac`, and that is a fact about the **tiling** (`--n-tiles`), as
section 15.5 said. `near` is different: it is the infall region of the very
cluster whose tiles are being trained, so it is reachable by selection alone.

### 16.4 The eval can now say WHICH population moved

`MemberSets.parent_id` was added in section 15 so an eval could split by
population, and nothing ever read it. That is why `all_blocks_smallhalo` spent
6.9 GPU-hours on a hypothesis entirely about top-level halos and reported one
median pooled over both -- and why "the objective is wrong" and "the objective
never got any budget" were indistinguishable until a Rockstar gate answered ten
hours later. Those need different next runs.

Every eval row now carries `by_population` (`sub` / `top`) and `by_env`
(`inside` / `near` / `field`), each with `n` and the medians of `bound_hard`,
`virial`, `centre_offset_radii` and `iso_ratio`; they are printed at step 0,
written to `metrics.jsonl`, and mirrored to wandb as `{split}/{bin}/{field}`.
Counts sum; medians are taken over the hosts whose split is non-empty, so a host
contributing zero `near` sets is counted as zero rather than dragging the median
to NaN. An empty split reports `n = 0` with NaN medians rather than vanishing --
a missing key reads as a bug, a zero reads as a fact.

### 16.5 How to run it

```bash
# The arm. `near` + `inside` is the 56% of the deficit that is reachable by
# selection; the isolation term is what makes a top-level object stay one.
# MG_GATE_AT_STEPS buys a real Rockstar read at hour ~2.7 for nothing --
# see 16.7 for the measurement that says step 2000 is enough.
MG_POPULATION=all MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified MG_MIN_NUM_P=50 \
  MG_ENV_SELECT=inside,near MG_W_ISOLATION=1.0 \
  MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0 MG_MAX_SET_PARTICLES=2048 \
  MG_GATE_AT_STEPS=2000 MG_LABEL=_nearclean \
  bash scripts/slurm/submit_member_gather_train.sh
```

The control it must beat is **`all_blocks_smallhalo`** -- same objective,
`env_select=any`, no isolation term -- on the same held-out gate. Read, in this
order:

* the **shakeout**, before believing anything: `env_select` in the pool banner,
  `n_by_env`, and `n_iso` against `n_sets`. A run whose `near` count is four
  sets is a selection failure, not a learning one, and says so at minute two.
* **Q1**, `by_population.top`: did the top-level sets themselves improve? This
  is the question section 15's arm could not answer about itself.
* **Q2**, the held-out census by environment: did `near` move, and did `field`
  stop overshooting? The net top-level count is *not* the number to read --
  `host-damage-is-mostly-reparenting` measured it to be dominated by VANISHED
  small hosts HR does not want.
* **the guards**, unchanged and still the ones with no loss term behind them:
  `vel_rms_ratio` and the banded high-k. Nothing here touches the velocity
  collapse of `highk-guard-is-blind`.

### 16.6 What this does not do

It does not address the velocity defect, and it is not a second attempt at the
shell arm: the isolation term says nothing about *where* a set is, only about
what is piled on top of it. If the held-out `iso_ratio` comes back at ~1 while
the top-level count still falls, the pedestal is outside the background sphere
and `MG_BG_RADIUS` is the next knob -- not a bigger `--w-isolation`.

### 16.7 Getting the Rockstar answer at hour 2.7 instead of hour 10.3

The gate has never been the expensive part -- `cheap-rockstar-ab-gate` measured
it at ~19 min, 3% of a 10 h train. The wait is. So the question is whether a
*mid-run* checkpoint gives a real read, and the three finished arms answer it
from their own `metrics.jsonl`, held out:

| arm | `bound_frac` @2000 | @8000 | share | high-k max @2000 -> @8000 |
| --- | ---: | ---: | ---: | --- |
| `all_blocks_smallhalo` | 0.0874 | 0.0877 | **99.7%** | 1.52x -> 3.15x |
| `all_blocks_selfhostguard` | 0.0950 | 0.0976 | **97.3%** | 2.02x -> 4.79x |
| `all_blocks_self` | 0.1202 | 0.1395 | 86.2% | 1.92x -> 3.87x |

The objective is essentially finished by step 2000. What the remaining 6,000
steps (75% of the run, ~7.5 h) reliably produce is **more high-k excess** --
which is the damage of section 11.6, not the objective improving. So a
step-2000 gate is a read, not a peek.

And it takes nothing from the arm, because the two do not compete for anything:

* **training runs to `--steps` regardless.** The snapshot is a file write; the
  loop does not pause, branch, or stop.
* **the gate's 19 min are on other nodes** -- one GPU export, then CPU.
* **artifacts cannot collide.** An early gate's run dir gets `_step<N>` and its
  Rockstar tag `_step<N>_<box>`, and `export.json` records
  `checkpoint.{file,step,partial,snapshot}` so a mid-run catalog can never be
  read later as a finished arm's.

**Read it one-sided.** A clear negative at step 2000 -- top-level still falling,
subhalos still exploding -- says the mechanism did not change, and that is worth
acting on with 7 h of the allocation still ahead. A null is **not** evidence of
failure: the last 6,000 steps are still 3-14% of the objective.

#### The pieces

| flag / env | where | what |
| --- | --- | --- |
| `--gate-at-steps` / `MG_GATE_AT_STEPS` | trainer | freeze an immutable `tuned_step<N>.pt` at those steps |
| `--checkpoint` / `HG_CHECKPOINT` | export, gate submitter | which checkpoint to gate; default `tuned.pt` |
| `HG_WAIT_MINUTES` | gate submitter | how long the wait stage polls for the snapshot |

Three details that are the difference between this working and this corrupting a
run:

* **`save_atomic`.** Every checkpoint write now goes through a temp file in the
  same directory and one `os.replace`. `tuned.pt` is rewritten every eval, and
  the export spends ~138 s reading a checkpoint; the moment a gate is allowed to
  fire during training those overlap, and the failure is a torch unpickling
  error whose message says nothing about a race. This was a latent bug in the
  existing `HG_AFTER_JOB` chain too, since a walltime-killed trainer could be
  mid-write when the dependent export started.
* **the snapshot is immutable.** `tuned.pt` cannot serve as the early-gate
  target however atomic its writes are, because the export would gate whichever
  step happened to be current -- a number nothing else in the run records.
* **`gather_wait_checkpoint_cpu.sbatch`.** Slurm has no "depend on a file", so a
  1-core CPU job polls for the snapshot and the export depends on *it*. It
  deliberately does **not** depend on the training job: the snapshot appears
  while that job is still `RUNNING`, so `afterok` would defeat the purpose and
  `after` only says "has started". On timeout it exits non-zero -- the dependents
  then show `DependencyNeverSatisfied`, which is correct (there is nothing to
  gate) and is not unexplained: the reason is in its log, and the submitter
  prints that log's path before queueing anything.

#### How to run it

```bash
# 1. training, with a snapshot at step 2000. Note the job id it prints.
MG_POPULATION=all MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified MG_MIN_NUM_P=50 \
  MG_ENV_SELECT=inside,near MG_W_ISOLATION=1.0 \
  MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0 MG_MAX_SET_PARTICLES=2048 \
  MG_GATE_AT_STEPS=2000 MG_LABEL=_nearclean \
  bash scripts/slurm/submit_member_gather_train.sh

# 2. the EARLY gate -- submit immediately, alongside training. The wait job
#    holds the chain until the snapshot lands (~2.7 h in), then it runs while
#    training continues.
MG_ARMS=all_blocks_nearclean HG_CHECKPOINT=tuned_step2000.pt NO_CONTROL=1 \
  HG_BASELINE_ARM=all_blocks_smallhalo HG_BASELINE_NAME=smallhalo \
  bash scripts/slurm/submit_gather_holdout_rockstar.sh

# 3. the end-of-run gate, chained behind the training job as usual.
MG_ARMS=all_blocks_nearclean HG_AFTER_JOB=<training job id> \
  HG_BASELINE_ARM=all_blocks_smallhalo HG_BASELINE_NAME=smallhalo \
  bash scripts/slurm/submit_gather_holdout_rockstar.sh
```

`NO_CONTROL=1` on the early gate because the frozen control does not depend on
the checkpoint: if set9 already has one, re-gating it is ~13 min of Rockstar for
a number that cannot have changed. The submitter says so when it would.

**Cheaper still, and free:** `iso_ratio` is now on every eval row from step 250
(~19 min), held out, split by population and environment. It measures the exact
mechanism the step-2000 gate is being asked about -- whether the objects being
built are being built into a pile. If it is above 1 and rising on the top-level
split, the Rockstar gate is going to say what section 16 says, and it will say
so two and a half hours earlier for no compute at all.

## 16b. Why the soft shell fits on train and does not transfer (2026-09-09)

### 16b.1 What the sweep measured

The `w_shell` reweight sweep ran three arms to 8000 steps. Raw, unweighted
`term_raw_shell` (the quantity `--term-norm`'s scales are measured from, and the
only one comparable across runs with different weights):

| arm | train | holdout |
|---|---|---|
| `all_blocks_softshell`   (w=1)  | 0.0507 -> 0.0589 (**worse**) | 0.0539 -> 0.0844 (worse) |
| `all_blocks_softshell10` (w=10) | 0.0507 -> 0.0205 (2.5x better) | 0.0539 -> 0.0599 (worse) |
| `all_blocks_softshell30` (w=30) | 0.0507 -> 0.0069 (7.3x better) | 0.0539 -> 0.0390 (1.4x better) |

Only w=30 improves out of sample at all, at a 5.6x train/holdout gap.

**Two explanations that are ruled out.** It is not a pooling artifact:
`--sets-per-step` defaults to 0, so train and holdout both charge *every* set of
a host, and holdout is different BOXES (set9/set10), not held-out hosts inside a
trained field. And it is not that the target is realisation noise. Splitting each
HR host's subhalos in half at random and comparing the two halves' log-radius
quantile curves (`subhalo_radial/set9/hr.npz`, 101,467 subhalos, in `r/R_vir`):

| hosts with >= 100 subhalos | sd across hosts | R^2 from host mass | split-half reliability |
|---|---|---|---|
| u = 0.25 | 0.146 | 0.011 | **0.84** |
| u = 0.50 | 0.145 | 0.014 | **0.91** |
| u = 0.75 | 0.157 | 0.021 | **0.94** |

A second independent draw of the same host reproduces its deviation at rho ~ 0.9,
so each host genuinely has its own radial distribution, with a real signal of
sd ~ 0.14 in log `r/R_vir` to learn. Host **mass**, though, explains 1-2% of it.
The quantity is real and is simply not carried by the obvious summary feature.

### 16b.2 The prime suspect: the fixed noise draw

`build_tiles` (`scripts/features/overfit_host_mse.py`) takes
`tile_noise_stack([args.seed], t, geom)` **once**, and the fine-tune has a single
`--seed`. So the generator's input is `(LR tile, fixed noise tile)`, constant for
all 8000 steps.

That is a second input channel carrying real per-tile detail and **zero**
generalisable information. Any host-specific radius structure the LR field does
not carry can still be fitted by keying on that tile's particular noise pattern,
and on a held-out box the noise is a fresh draw, so all of it evaporates. The
dose-dependence fits: a larger `w_shell` is more pressure to exploit the
memorisable channel, which is exactly what the train curve shows and the holdout
curve does not.

Receptive field is **not** the explanation. `HBlockFlow` runs `num_levels=3`
(8x downsample) over a 64-cell tile at `DX = 100/512`, i.e. 12.5 Mpc/h, which
spans the host for any host in the pool. A set at 1-3 R_vir can see its own
centre. (Estimated from the layer stack, not from a measured gradient trace.)

### 16b.3 The three levers, and the measurement that arbitrates

All four flags default OFF, so an unset run reproduces every arm on disk.

**The measurement.** `--fresh-noise-eval` reserves ONE extra draw that no step
ever trains on and evaluates both pools under it, adding `train_fresh` and
`holdout_fresh` to every `metrics.jsonl` row. That is the 2x2 over (seen box,
seen noise), and it is what decides which lever matters:

* `train_fresh` falls back to `holdout` -> the train/holdout gap **was** the
  fixed noise being memorised. The objective carries no more transferable signal
  than the holdout curve already shows, and (a)/(b) are the fixes.
* `train_fresh` tracks `train` -> the model learned something from the LR field
  and the gap is about **boxes and hosts** instead: a different problem with a
  different fix, and (c) is the lever to read.

**(a) `--noise-draws N --noise-mode rotate`.** N realisations per tile, one
picked per host per step. A pattern that only fits one draw then buys nothing on
average, so the only way left to lower the loss is through the LR field.

Each draw carries **its own set table**. This is not optional: `centre_self` --
the anchor `--centre-mode self`/`self_tan` charges against -- is the frozen
field's own centroid, and the frozen field is a function of the noise. A draw
that borrowed draw 0's anchor would charge the run for ordinary realisation
scatter and fight the objective. The host-preservation tables (`--w-host-sets`)
are referenced against the frozen field's own value and are draw-dependent in the
strongest possible way, so they are rebuilt per draw too. The bill lands on the
**pool build**, roughly N times the single-draw run; the step loop is unchanged.

**(b) `--shell-noise-draws M`.** Charge the shell on the MEAN reduced radius
curve over M+1 draws instead of on the primary draw alone. Note the order:
**mean-then-charge, not charge-then-mean**. The second is only a lower-variance
version of the same per-realisation objective; the first asks for a distribution
that no single realisation can fake, which is the freedom the sweep exploited.
Only the radii are recomputed on the extra draws (`shell_radii`), never the
O(N^2) statistics, so an extra draw costs a forward pass and a gather.

**(c) `--shell-quantiles Q`.** Reduce both sorted arrays to Q equal-COUNT bins
and charge the bins' mean log radius. The full pairing hands the loss one degree
of freedom per set (~134 for a big host); the low-order shape is what a shared
operator can plausibly carry, and the rank-by-rank wiggle is what it can only
memorise. Q >= N is the full pairing again, so a small host is never charged on
emptier bins than it has sets. Try 3-5.

**`term_raw_shell` is not comparable across Q** -- a different Q is a different
number of terms in the mean. Compare `shell_resid`, which is measured the same
way at every Q.

### 16b.4 Running it

```
bash scripts/slurm/submit_shell_levers.sh          # all five arms
ARMS="ss30_fresh" bash scripts/slurm/submit_shell_levers.sh   # just the 2x2
```

Five siblings on `all_blocks` + `w_shell=30` + `centre_mode=self_tan`, i.e.
`all_blocks_softshell30` with one variable changed each: `ss30_fresh` (control,
measurement only), `ss30_rot` (a), `ss30_navg` (b), `ss30_q4` (c), and
`ss30_rot_q4`. Read `ss30_fresh` first and write the interpretation down before
the others land. None of these is a result until it passes the Rockstar A/B gate
(`scripts/slurm/submit_gate_ab.sh`); `metrics.jsonl` is a surrogate the objective
computes about itself.

### 16b.5 Result (2026-09-10): none of the levers generalise

All five arms ran to step 8000 (`navg` read at 7500). Read on the honest,
Q-comparable `shell_resid` (the median log-radius residual, measured on the full
rank pairing at every Q; `term_raw_shell` is NOT comparable across Q because the
mean is over a different number of terms). Frozen holdout baseline is 0.085.

| arm | holdout `shell_resid` | vs frozen |
|---|---|---|
| `ss30_fresh` (control) | 0.082 | -3.2% |
| `ss30_rot` (a) | 0.083 | -2.6% |
| `ss30_q4` (c) | 0.085 | +0.4% |
| `ss30_rot_q4` (a+c) | 0.088 | +3.7% (worse) |
| `ss30_navg` (b) | 0.094 | +10.4% (worse) |

**1. The noise draw is not the cause.** The 2x2's noise contrast
(`train_fresh - train`) is ~0 in every arm: swapping in a never-trained draw
costs nothing, so the training fit was not keyed to the fixed noise pattern. The
2026-09-08 hypothesis that the gap was a memorised noise channel is falsified.
Rotating (a) and noise-averaging (b) confirm it independently -- both leave the
holdout gap where the control had it.

**2. None of the three levers makes the shell generalise.** The best holdout is
~3% below frozen and it is not durable: `shell_resid` on the held-out pool bottoms
near step 1000 (~0.077) and drifts back up while train descends monotonically --
textbook overfitting. Quantile binning (c) leaves holdout unchanged; noise
averaging (b) makes it worse.

**3. What the gap actually is.** `term_raw_shell` (squared, dead-zoned -- the loss)
had a large train/holdout gap; `shell_resid` (a median) barely did. So the loss's
apparent "win" is concentrated in the worst-offending sets on the *training*
boxes -- per-box/host tail structure that does not transfer. The per-host radial
distribution is real (split-half rho ~0.9; section 16b.1) but a shared operator
cannot predict WHICH host gets WHICH distribution from the LR field.

**Verdict.** This confirms rather than overturns the `w_shell` reweight sweep: the
shell-distribution family is closed. The Rockstar A/B gate was not run -- with
zero holdout generalisation on the surrogate a pass is not plausible, and the
gate time is better spent elsewhere. The four flags stay in the code as
default-OFF levers for reuse on other terms; the fresh-noise 2x2 in particular is
a general overfit diagnostic (is a train/holdout gap noise, or box/host?).

## 16c. The LR-host conditioning channel (`--cond-channels`), 2026-09-10

**Built, tested, NOT run.** This is the first change in this line that alters what
the generator can *see* rather than what it is charged for.

### Why no loss term can fix the radial profile

Section 16's diagnostic, and `radial-profile-is-flat-in-host-mass`, measured it on
held-out set9: HR's median subhalo host-centric radius scales with host mass over a
factor of 6.5 (0.181 Mpc/h at 200-1k particles to 1.186 at >=100k, i.e. `r ~ R_vir ~
M^(1/3)`), and every candidate catalog is **flat** at ~0.1 Mpc/h.

That is not a weighting failure, which is why three separate shell arms (`w_shell`,
`w_radius`, and the 2026-09-09 reweight sweep) all failed and why section 16b closed
the family. `G_theta(Y, z)` is local and translation-equivariant. Fed a 14^3 LR crop
it cannot distinguish "1.2 Mpc from a cluster" from "0.3 Mpc from a group" -- the two
look alike locally -- so it emits the conditional mean over host masses, and the
conditional mean IS a flat profile. The information is not in the input.

### What the channel is

`src/cosmo_sr/features/lr_host_channel.py`, on the LR Lagrangian lattice, from an
**LR-only** Rockstar catalog (no HR is read; the catalog exists at inference for any
box the generator can run on):

| channel | what it carries |
| --- | --- |
| `r_over_rvir` | the site's distance to its owning host in that host's R_vir -- the coordinate the profile is a function of |
| `log_mhost` | `(log10 Mvir - 13) / 1` -- the scale the operator cannot infer locally |
| `valid` | 1 where a gated host owns the site. Without it "3 R_vir from a cluster" and "no host here" carry identical fill values |
| `rhat_x/y/z` | optional unit vector toward the host centre. A radius says how far to move material, never which way (`centre-offset-is-an-address`). OFF by default |

Gated at **>=20k HR-equivalent particles**, which is where `lr-host-channel-fidelity`
measured the LR catalog channel stops working. LR is 64^3, so that is a ~39-particle
LR floor; the catalog falls off a cliff below it rather than degrading.

### The membership question, measured

`lr_membership_coverage.py` had already run (`lr_host_fidelity/membership_summary.json`,
set8 and set9), and the fear it was written to test **did not materialise**. Strict
Rockstar membership, resolved to the top-level ancestor:

| r / R_vir | n_sub (set9) | median correct-assignment frac | verdict |
| --- | --- | --- | --- |
| 0-0.25 | 822 | 0.942 | COVERED |
| 0.75-1 | 6521 | 0.867 | COVERED |
| **1-1.5** | **9973** | **0.793** | **COVERED** |
| 1.5-2 | 3499 | 0.673 | COVERED |
| 2-3 | 1012 | 0.493 | PARTIAL |
| >3 | 104 | 0.147 | PARTIAL |

Membership is covered through the 1.0-1.5 R_vir shell HR's subhalos peak in, and
assigns 31.6-31.9% of LR sites against `derived`'s 29% valid fraction. It only fails
past 2 R_vir. So the two modes are CLOSE and the choice is a judgement call, not a
settled one.

`derived` is the default for the reasons that survive that measurement, none of them
decisive: it is defined everywhere with no PARTIAL tail past 2 R_vir (which is the
range `r_max_rvir = 3` is trying to cover), it degrades continuously in `r/R_vir`
rather than categorically, and it needs no `halos_*.particles` table -- the file that
filled the per-user quota once already. `--cond-mode membership` is a genuine
alternative arm.

### How it is wired, and the three things that had to be true

`ControlledG` gains `cond_chan`, implemented as a **separate** zero-initialised 1x1
conv `cond0` added into `block0`'s pre-activation. Algebraically identical to widening
`block0.0` (a 1x1 conv on `cat([x, cond])` is `conv(x) + conv(cond)`), and the
separation buys two things widening would have cost:

* the pretrained state dict still loads with every original key matching exactly, so
  a mis-sized checkpoint still raises;
* `cond0` is its own tensor, so it is unfrozen at **every** rung. Widening
  `block0.0.weight` would have filed it under `block0.`, which only rung `full`
  trains -- the channel would have been wired in and then held at zero for ten hours,
  producing a null result that reads as "conditioning does not help".

It is trained at the largest learning rate of any group (`--cond-lr`, default 1e-5),
for the same reason: it starts at exactly zero and has no pretrained function to
protect.

Three invariants, pinned in `tests/features/test_lr_host_channel.py`:

1. **Zero-init equivalence.** At step 0 the conditioned generator is bit-identical to
   frozen for *any* conditioning input. Every baseline in this pipeline -- `base` in
   `host_forward`, the `frozen` array in the gate's `tiles.npz`, the low-k guards --
   assumes it.
2. **Channel/tile alignment.** `tile_cond_crop` is the same `periodic_crop` call as
   `tile_lr_crop`, same start, same pad; and `super_resolve_srs_seeded` crops the
   whole-box stack identically. A one-cell slip would hand the generator a coherent,
   plausible, wrong host coordinate and nothing downstream would raise.
3. **Replay parity.** `export_gather_tiles.py` rebuilds the channels from the run's
   stored config and now *compares that against the checkpoint's own `cond` key*,
   refusing to export on a mismatch. A gate whose replay builds different channels
   scores a different model under this arm's name.

### Running it

The fields are built once per box by a CPU job -- it runs a halo finder, which has no
business inside a 10-hour GPU allocation -- and the trainer refuses to start if a box
in `--train-boxes` or `--holdout-boxes` has no field, or if the cached field was built
under different settings:

```bash
bash scripts/slurm/submit_lr_host_channel.sh          # CPU, ~9 s/box, resumable
MG_COND_CHANNELS=r_over_rvir,log_mhost,valid \
    MG_LABEL=_cond bash scripts/slurm/submit_member_gather_train.sh
```

`MG_COND_CHANNELS` empty is the unconditioned control and is byte-identical to every
arm recorded before this existed, which is what makes the A/B clean.

**Read the radial diagnostic, not the loss.** A falling loss with a still-flat profile
means the channel is being ignored, and that is the one outcome this build cannot
rule out in advance.

**Built 2026-09-10** (job 47459, 7 boxes in ~65 s -- a 262k-particle halo find is
seconds, not minutes). Uniform across boxes: 451-497 gated top-level LR hosts,
**valid fraction 0.288-0.300**, median `r/R_vir` 0.83-0.90 among valid sites, median
gated host `log10 Mvir` 13.61-13.70. So the channel speaks about ~29% of the box, and
the supervised pool's floor (`--min-log-mvir 13.5`, i.e. ~54k HR particles) sits
comfortably above the 20k gate -- the hosts the loss is about are hosts the channel
knows.

## 16d. kNN-distance statistics: a gate-side damage monitor, not an objective (2026-09-13)

Full write-up: **`docs/sr2_knn_distance.md`**. A differentiable k-nearest-neighbour
distance loss (grid-free, catalog-free; the detached-search / live-distance gradient is exact)
was taken through four CPU measurement rounds before any GPU was spent. In short:

- **It sees the deficit** that P(k) cannot. On HR's 50-200p subhalo particles, frozen SR2's
  neighbour distances are 1.35-1.88x HR's.
- **It cannot rank working arms.** Across 20 set9 arms with 4,768-7,936 region subhalos,
  |rho| <= 0.4 against every gate metric.
- **It is a strong damage detector.** Across 12 arms that lost 513-742 of frozen's >=200p
  halos, |rho| ~ 0.8-0.9.
- **The max rank must reach the object size** or the loss's minimum is fragments (largest
  FOF group 13 vs 107 of 150).

It therefore runs as an advisory stage of `submit_gather_holdout_rockstar.sh`, writing
`knn_report.json` per arm (set9-calibrated `DAMAGE` flag, never fails the chain).
`--w-knn-unsup` exists and stays 0. Section 7 of that doc also corrects the `condsmall`
reading of section 16c's gate: in the region it does not beat frozen at 50-200p
(2,587 vs 2,762); the +43% was against its `smallctl` control.

## 17. Module map

| file | role |
| --- | --- |
| `features/member_pool.py` | multi-host selection; one owner load per box; train/held-out tile disjointness |
| `features/field_guards.py` | the high-k hinge, section 7 item 2 of `sr2_member_gather.md` |
| `features/lr_host_channel.py` | the LR-host conditioning channels, section 16c: build, cache, tile crop |
| `scripts/features/build_lr_host_channel.py` | the CPU job that builds one box's channel stack |
| `slurm/submit_lr_host_channel.sh`, `slurm/lr_host_channel_cpu.sbatch` | that job, submitted |
| `scripts/features/finetune_member_gather.py` | the trainer: generator in place of the free field |
| `slurm/submit_owner_arrays.sh`, `slurm/owner_arrays_cpu.sbatch` | the supervision build, throttled |
| `slurm/submit_member_gather_train.sh`, `slurm/member_gather_train_gpu.sbatch` | shakeout -> fine-tune |
| `features/member_gather.py` `_SpecificPotential` | section 8.2: the pair sum with a recomputed, analytic backward |
| `scripts/features/centre_offset_decompose.py` | section 9: rule or address, without a halo finder; `arm_residuals` prices the three arms with no GPU |
| `features/member_gather.py` `centre_mode` | section 9.4: `full` / `radial` / `self` / `self_tan`, plus the dead zone and Huber knee |
| `features/member_gather.py` `w_radius` | section 14: the shell -- host-centric radius, direction free. RAN, failed |
| `features/member_gather.py` `w_shell` | section 14.6: the SOFT shell -- the radius distribution, no addresses |
| `features/member_gather.py` `w_isolation` | section 16.2: the only INTER-set term -- non-member mass on the object, id-matched to HR |
| `features/subhalo_gather.py` `halo_environment`, `env_mask` | section 16.3: the one definition of `inside`/`near`/`field`, shared with the census |
| `features/member_gather.py` `env_select`, `_apply_cap(env=)` | section 16.3: the third selection axis, and the (mass x env) cap |
| `features/member_gather.py` `_split_rows` | section 16.4: `by_population` / `by_env`, so an eval can say WHICH population moved |
| `finetune_member_gather.py` `save_atomic`, `gate_at_steps` | section 16.7: immutable mid-run snapshots, safe to read while training |
| `slurm/gather_wait_checkpoint_cpu.sbatch` | section 16.7: the file-dependency Slurm does not have |
| `scripts/features/subhalo_radial.py`, `subhalo_radial_plot.py` | section 14: subhalo mass vs distance-to-host per catalog, the measurement the shell arm answers |
| `features/member_gather.py` `bound_penalty`, `term_scale` | section 10.4: the two budget levers |
| `slurm/submit_centre_offset.sh`, `slurm/centre_offset_gpu.sbatch` | its job |
| `scripts/features/export_gather_tiles.py` | section 11: a finished run's held-out tiles in the splice layout, config replayed from `summary.json` |
| `slurm/gather_export_gpu.sbatch`, `slurm/submit_gather_holdout_rockstar.sh` | export -> splice -> Rockstar -> compare, one box per arm + a shared frozen control |
| `scripts/features/gather_holdout_figures_{data,plot}.py`, `slurm/submit_gather_figures.sh` | section 11.3-11.4: the density/mass-function/local-excess/cosmic-web figures, redrawable from a saved npz |
| `scripts/features/gather_holdout_add_density.py`, `slurm/submit_gather_add_density.sh` | section 11.3: repair a blank density panel (HR's) by selecting the host's Lagrangian tiles out of ONE field, verified against a panel already in the npz |
| `scripts/features/gather_holdout_zoom_data.py`, `slurm/submit_gather_zoom.sh` | section 11.3: the R_vir zoom -- panels REBINNED from the export's tiles.npz at 64^3, not cropped; the only route to a tuned panel now its field is purged |
| `features/subhalo_gather.py` `population_rows`, `subhalo_home_tiles(population=, max_num_p=)` | section 15: one definition of "which halos is this run responsible for", shared by the census and the trainer |
| `features/member_gather.py` `population`, `max_num_p`, `set_select` | section 15.3: the num_p window and the cap policy that keeps it from being truncated away |
| `scripts/features/halo_gap_census{,_plot}.py`, `slurm/submit_halo_gap_census.sh` | section 15.1-15.2: WHERE the count deficit is -- population x mass x environment, from catalogs alone, with a verdict |
| `features/knn_cdf.py` | section 16d: the differentiable kNN log-distance estimator, neighbour tiles, centroid cube |
| `scripts/features/knn_cdf_diagnostic.py`, `knn_cdf_split_diagnostic.py`, `knn_gate_crosscheck.py` (+ plots) | section 16d: the four go/no-go rounds of `sr2_knn_distance.md` |
| `scripts/features/knn_gate_report.py`, `slurm/gate_knn_cpu.sbatch` | section 16d: the advisory kNN damage report in every holdout gate chain (`KNN_REPORT=0` skips) |
| `slurm/submit_knn_gate_backfill.sh` | section 16d: that report for already-gated arms / other boxes, from fields on disk |

Tests: `tests/features/test_field_guards.py` (15),
`tests/features/test_member_pool.py` (16),
`tests/features/test_finetune_member_gather.py` (19),
`tests/features/test_halo_gap_census.py` (7 -- the environment label across a
periodic box face and against a smaller neighbour, and the three verdicts on
catalogs whose mass function is known by construction),
`tests/features/test_member_gather.py` (90 -- the potential rewrite, the
subsample rescaling, the centre shaping and the set minibatching, the four
`centre_mode` arms, the two budget levers of section 10.4, and section 14's
shell term (direction-blindness, sign symmetry, the dead zone, and the
`self` / `self_tan` conflict priced by numerical minimisation, and section
15.3's three cap policies) are pinned there),
`tests/features/test_subhalo_gather.py` (32, including section 15's three
populations, the `max_num_p` window and the two refusals),
`tests/features/test_centre_offset_decompose.py` (13),
`tests/features/test_export_gather_tiles.py` (7 -- the config replay of section
11, including the list-valued-option bug that killed the first launch),
the kNN line of section 16d: `test_knn_cdf.py` (22), `test_knn_cdf_diagnostic.py` (3),
`test_knn_cdf_split_diagnostic.py` (5), `test_finetune_knn.py` (7),
`test_knn_gate_crosscheck.py` (7), `test_knn_gate_report.py` (4 -- including a replay of all 32
set9 arms through the damage thresholds).

## 18. Reproduce

```bash
# the supervision, once: ~18 min CPU per box, throttled 2 at a time
bash scripts/slurm/submit_owner_arrays.sh

# the pool, minutes on a GPU -- read pool.json before spending anything
SHAKEOUT_ONLY=1 bash scripts/slurm/submit_member_gather_train.sh

# the rung ladder, as siblings (section 6 item 2). Unchanged objective: every
# knob of section 8.3 defaults to off, so this is the run that scored 72/154 in
# the free field, now with a generator and a pair sum that fits in a GPU.
for r in fine middle_fine all_blocks; do
  MG_RUNG=$r MG_LABEL=_$r SKIP_SHAKEOUT=1 \
    bash scripts/slurm/submit_member_gather_train.sh
done

# section 9: rule or address. Run it ALONGSIDE the ladder, not after -- it is
# ~25 min on one GPU and it decides how to read a null at every rung.
bash scripts/slurm/submit_centre_offset.sh

# --- section 15: the small-halo arm -----------------------------------------
# The census FIRST: its verdict sets MG_MAX_NUM_P, and the ceiling IS the
# hypothesis. Minutes on a CPU, no field read, no halo finder.
bash scripts/slurm/submit_halo_gap_census.sh

# then the shakeout (does the widened population survive the live-fraction cut
# in a cluster's tiles?), then the arm.
MG_POPULATION=all MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified MG_MIN_NUM_P=50 \
  SHAKEOUT_ONLY=1 bash scripts/slurm/submit_member_gather_train.sh
MG_POPULATION=all MG_MAX_NUM_P=2000 MG_SET_SELECT=stratified MG_MIN_NUM_P=50 \
  MG_CENTRE_MODE=self MG_W_HOST_SETS=1.0 MG_LABEL=_smallhalo \
  SKIP_SHAKEOUT=1 bash scripts/slurm/submit_member_gather_train.sh

# --- the three arms of section 9.4 ------------------------------------------
# Section 9 returned 13.4% -- "the offset is an ADDRESS" -- so these are the
# response, not an optional extra. Every knob defaults to the 72/154 objective,
# so an unset run is unchanged.

# (a) THE ONE TO RUN AT THE ORACLE. Same optimum, different gradient budget:
# run 2 already reached dx = 0.00 radii and still scored 72/154, so the residual
# is concentration, and this is what hands those five terms the budget. The
# arithmetic of 9.4 does NOT predict its outcome, which is the point.
FF_CENTRE_HUBER=2.0 FF_CENTRE_DEAD_ZONE=0.3 FF_LABEL=_soft \
  bash scripts/slurm/submit_free_field_gather.sh

# (b) and (c) at the oracle: CONFIRMATION ONLY, and the numbers are known in
# advance -- <= 12.3% and <= 3.6% of targets inside one search radius, against
# full's 46.8%. Run them to check the implementation against the arithmetic,
# never to look for a win.
FF_CENTRE_MODE=radial FF_LABEL=_radial \
  bash scripts/slurm/submit_free_field_gather.sh
FF_CENTRE_MODE=self   FF_LABEL=_self \
  bash scripts/slurm/submit_free_field_gather.sh

# (b) and (c) where they actually make their case -- a GENERATOR, which does
# not have the address the free field has. This is the comparison of 9.4's
# second table and the only one that can support `radial`.
for m in full radial self; do
  MG_CENTRE_MODE=$m MG_RUNG=all_blocks MG_LABEL=_$m SKIP_SHAKEOUT=1 \
    bash scripts/slurm/submit_member_gather_train.sh
done

# the softened centre as a generator sibling, for the same reason as (a)
MG_CENTRE_DEAD_ZONE=0.3 MG_CENTRE_HUBER=2.0 MG_RUNG=all_blocks \
  MG_LABEL=_soft SKIP_SHAKEOUT=1 bash scripts/slurm/submit_member_gather_train.sh

# --- the loss budget, section 10 --------------------------------------------
# Section 10.3: the residual is `bound`, which holds 0.24% of the gradient. Run
# these on the centre-OFF arm, because section 10.2 measured that the centre
# term degrades the very terms these are trying to feed -- rebalancing while it
# is still smearing the field would confound the two.
MG_W_CENTRE=0 MG_BOUND_PENALTY=log MG_RUNG=all_blocks MG_LABEL=_boundlog \
  SKIP_SHAKEOUT=1 bash scripts/slurm/submit_member_gather_train.sh
MG_W_CENTRE=0 MG_TERM_NORM=1 MG_RUNG=all_blocks MG_LABEL=_termnorm \
  SKIP_SHAKEOUT=1 bash scripts/slurm/submit_member_gather_train.sh
# both levers: `log` fixes the term's shape, `--term-norm` fixes its share
MG_W_CENTRE=0 MG_BOUND_PENALTY=log MG_TERM_NORM=1 MG_RUNG=all_blocks \
  MG_LABEL=_budget SKIP_SHAKEOUT=1 bash scripts/slurm/submit_member_gather_train.sh

# --- THE HELD-OUT GATE, section 11: the tile-splice gate, now built ---------
# Export a finished arm's held-out tiles, splice into a frozen box, one Rockstar
# run per arm + a shared frozen control. `tuned.pt` is in the loader's own
# format so the export needs no conversion, and the config is replayed from
# summary.json so the pool matches the run bit-identically.
#   shakeout: one arm, one host, no Rockstar (minutes)
EXPORT_ONLY=1 MG_ARMS=all_blocks_self HG_MAX_HOSTS=1 \
  bash scripts/slurm/submit_gather_holdout_rockstar.sh
#   the real thing: 4 arms + control on set9's 8 held-out hosts, 1,127 targets
bash scripts/slurm/submit_gather_holdout_rockstar.sh
HG_BOX=set10 bash scripts/slurm/submit_gather_holdout_rockstar.sh   # 2nd box

# the figures (section 11.3-11.4): density zoom, mass function, local excess,
# cosmic web. Two CPU jobs, redrawable from the saved npz.
bash scripts/slurm/submit_gather_figures.sh
PLOT_ONLY=1 bash scripts/slurm/submit_gather_figures.sh             # re-render only
bash scripts/slurm/submit_gather_add_density.sh                     # fill a blank panel
bash scripts/slurm/submit_gather_zoom.sh                            # zoom to R_vir

# STILL OWED (section 12 item 6): a WHOLE-BOX regeneration through tuned.pt --
# evaluate_sr2_direct.py --checkpoint -> flow_rockstar_catalog_cpu.sbatch ->
# compare_gather_catalog.py -- to measure the collateral damage the tile splice
# holds frozen by construction.

# if the pair sums make the wall clock bind (section 8.3), in this order:
# minibatch the sets first -- it is free -- and only then subsample particles,
# which changes the estimator.
MG_SETS_PER_STEP=32 MG_RUNG=all_blocks MG_LABEL=_sps32 SKIP_SHAKEOUT=1 \
  bash scripts/slurm/submit_member_gather_train.sh
```

Artifacts under `$DMSR_REWARD_ROOT/member_gather/<rung><label>/`: `pool.json`
(composition and every host's reachable reference), `metrics.jsonl` (one row per
eval, train and held-out, per host), `summary.json`, `tuned.pt`. wandb group
`member_gather`; the verdict and the final per-host table land in the run
summary.
