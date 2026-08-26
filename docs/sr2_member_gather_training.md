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

## 14. Module map

| file | role |
| --- | --- |
| `features/member_pool.py` | multi-host selection; one owner load per box; train/held-out tile disjointness |
| `features/field_guards.py` | the high-k hinge, section 7 item 2 of `sr2_member_gather.md` |
| `scripts/features/finetune_member_gather.py` | the trainer: generator in place of the free field |
| `slurm/submit_owner_arrays.sh`, `slurm/owner_arrays_cpu.sbatch` | the supervision build, throttled |
| `slurm/submit_member_gather_train.sh`, `slurm/member_gather_train_gpu.sbatch` | shakeout -> fine-tune |
| `features/member_gather.py` `_SpecificPotential` | section 8.2: the pair sum with a recomputed, analytic backward |
| `scripts/features/centre_offset_decompose.py` | section 9: rule or address, without a halo finder; `arm_residuals` prices the three arms with no GPU |
| `features/member_gather.py` `centre_mode` | section 9.4: `full` / `radial` / `self`, plus the dead zone and Huber knee |
| `features/member_gather.py` `bound_penalty`, `term_scale` | section 10.4: the two budget levers |
| `slurm/submit_centre_offset.sh`, `slurm/centre_offset_gpu.sbatch` | its job |
| `scripts/features/export_gather_tiles.py` | section 11: a finished run's held-out tiles in the splice layout, config replayed from `summary.json` |
| `slurm/gather_export_gpu.sbatch`, `slurm/submit_gather_holdout_rockstar.sh` | export -> splice -> Rockstar -> compare, one box per arm + a shared frozen control |
| `scripts/features/gather_holdout_figures_{data,plot}.py`, `slurm/submit_gather_figures.sh` | section 11.3-11.4: the density/mass-function/local-excess/cosmic-web figures, redrawable from a saved npz |

Tests: `tests/features/test_field_guards.py` (15),
`tests/features/test_member_pool.py` (16),
`tests/features/test_finetune_member_gather.py` (19),
`tests/features/test_member_gather.py` (69 -- the potential rewrite, the
subsample rescaling, the centre shaping and the set minibatching, the three
`centre_mode` arms, and the two budget levers of section 10.4 are pinned
there), `tests/features/test_centre_offset_decompose.py` (13),
`tests/features/test_export_gather_tiles.py` (7 -- the config replay of section
11, including the list-valued-option bug that killed the first launch).

## 14. Reproduce

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
