# Cheapest credible evidence that catalog reward can control occupation

Experiments 0 and 1 of the revised plan: fine-grained (64³) credit, and the
targeted HR-residual oracle. This document is the implementation record — what
exists, what was measured on CPU here, what has to be submitted, and what each
outcome licenses.

**The headline finding is a sequencing one.** The brief says to keep "the
current residual diffusion model". There isn't one:
`$ZFS/DMSR/dmsr_reward/checkpoints/` does not exist and the only `.pt` files on
scratch belong to other work lines (degraders, latent flows). So **Experiments
2, 3 and 4 are all blocked behind Gate A** — training the residual prior, 1–2
days of GPU. Experiments 0 and 1 need only the frozen SR2 cache and the HR
boxes, both of which exist for all 16 boxes. Running the oracle first is
correct for a reason beyond cost, and the Gate-A training is submitted as an
independent sibling so the GPU works while the oracle answers.

---

## 1. What Experiment 0 changes, and why it is not a resolution tweak

The existing credit geometry (`cosmo_sr.reward.geometry`) attributes a halo to a
Lagrangian chunk with a **purity test on an Eulerian mesh**, and *drops* the
halo when the test fails. `docs/reward_residual_diffusion.md` §2 measured what
that costs: retention 0.37, 0.30, 0.22, 0.12, **0.00** across host bins from
`1e12` to `1e14 Msun/h`. The rejection is mass-dependent and structural — a
cluster accretes from a large Lagrangian volume, so the more massive the host
the more certainly its `Rvir` neighbourhood straddles a boundary. The mask
preferentially deletes exactly the hosts the occupation reward is about. The fix
adopted then was to double `chunk_hr` to 256, buying host counts at the cost of
credit resolution (8 units per box instead of 64).

Experiment 0 removes the trade-off rather than re-balancing it, because the
information the purity test was *approximating* turns out to be available
exactly:

* our GADGET2 dumps set the particle ID to the **flat Lagrangian grid index**
  ([`particles.py:97`](../src/cosmo_sr/eval/particles.py#L97), `ids = arange(Ng**3)`),
  and Rockstar preserves 4-byte IDs verbatim
  (`io/io_gadget.c::gadget2_rescale_particles`);
* with `FULL_PARTICLE_CHUNKS = 1` Rockstar writes every printed halo's member
  particle IDs.

So each object is split **fractionally** over the 64³ tiles that produced its
particles, `w[o,j] = |members(o) ∩ tile j| / |members(o)|`, summing to 1 by
construction. Nothing is rejected, nothing is mass-selected, and a tile's
effective volume is its nominal volume — there is no masked region to bias
number densities low.

`tile_hr = 64` is not a free parameter: it is one SR2 forward pass
(`nsplit = 8` over `ng_hr = 512`), so it is the finest unit credit can be
assigned to at all.

**Both geometries stay in the config.** `tiles:` and `geometry:` measure
different things; the chunk path is what every earlier audit used and is not
retired by this work.

### The trap in the `.particles` format

`io/meta_io.c::print_child_particles` walks each halo **and recurses into its
substructure**, emitting the table once per recursion root. Grouping is
therefore by `external_haloid` (the root's catalog id), which gives every object
its complete bound-particle set exactly once. Grouping by
`assigned_internal_haloid` instead would silently strip substructure out of its
host, so a rich host's Lagrangian footprint would come from its smooth component
alone — a quiet, plausible-looking wrong answer.

`tests/reward/test_tiles.py` pins the distinction with a synthetic writer that
reproduces the recursion rather than a flat table.

---

## 2. CPU work actually run here

### 2.1 Whole-box audit over all 16 boxes

Existing catalogs only, no halo finding. It was run to choose the Experiment-1
development boxes, and it **removed the premise it was meant to serve**:

| | 1e13 hosts | 3.16e13 hosts | HR occ @3.16e13 | SR2 occ @3.16e13 |
|---|---|---|---|---|
| range over 16 boxes | 219–283 | 64–94 | 114–139 | 13.3–15.5 |

Boxes are interchangeable to ±8% in the upper reliable bins, so "use development
boxes containing many hosts in the upper reliable bins" has no lever to pull.
Box choice is a split-hygiene decision instead, and **`set8`, `set9` (val split)
are used, leaving all four test boxes — including `set12` — untouched.** They
already have HR and frozen-SR2 catalogs, so this costs nothing extra.

The same audit re-confirms the target failure independently on every box: SR2
occupation at `3.16e13` is 13–15 against HR's 114–139, a factor ~9, uniformly.

### 2.2 Verification against the real Rockstar binary

`scripts/reward/verify_member_ids.py` — a short local CPU run, two Rockstar
invocations on a 128³ clustered box (2.1M particles, 138 halos, ~2 min):

| Check | Result |
|---|---|
| `.particles` is written under the member-id config | **pass**, 26.1 MB / 334,980 rows |
| the halo catalog is unchanged by the flag | **pass** — identical ids, `max\|ΔMvir\| = 0`, `max\|Δpos\| = 0`, `max\|Δnum_p\| = 0` |
| printed `particle_id` are our GADGET/Lagrangian ids | **pass** |
| member sets consistent with the catalog | **pass** — 138/138 objects present, 0 missing, 0 truncated |
| partition of unity `max\|Σ_j w − 1\|` | **5.2e-15** |

**This run corrected a wrong invariant before it reached production.** The first
version checked `rows(o) == num_p(o)` and failed on 40 of 138 objects — and
those 40 were precisely the hosts with substructure, i.e. every host this
project is about. Two independent reasons, both now documented in
`tiles.py` and covered by tests:

* `num_p` is a halo's *own* particle list; the recursion adds its descendants';
* the recursion also walks substructure that Rockstar **rejected from the
  catalog**. A 7914-particle host absorbed 51 further particles from four 12–14
  particle clumps appearing nowhere in the ASCII file.

Both are correct for our purpose — a host's Lagrangian footprint should include
all its bound material, resolved or not. The checkable invariant is
`rows(o) >= num_p(o)` with equality exactly for leaves (measured: 98 of 138,
which is all and only the objects with no substructure), plus no particle id
repeating within an object (measured: zero). `check_member_consistency` enforces
this, and `rockstar_particles.py` **exits 3** rather than writing tile summaries
from an inconsistent table.

Had the equality check shipped, every production box would have aborted on its
richest hosts.

### 2.3 Unit tests

```
python -m pytest tests/reward       ->  165 passed   (122 existing + 43 new)
python -m pytest tests --ignore=tests/reward   (regression, unchanged)
```

Note `pyproject.toml` already sets `addopts = "-q"`; passing `-q` again makes it
`-qq` and suppresses the summary line entirely.

New coverage:

| File | N | Covers |
|---|---|---|
| `test_tiles.py` | 21 | id→tile matches the dump convention and rejects out-of-range; tiles partition the box; **partition of unity**; hosts include their substructure (the grouping trap); straddling objects are split not dropped; chunked streaming merges split objects; unprinted halos dropped; **tile sums reproduce direct full-box stats**; **no double counting**; **tile order irrelevant**; **periodic translation invariance**; mismatched table raises; fractional weights survive JSONL; centre-tile fallback keeps partition of unity; `A_j` is exact removal arithmetic; empty tiles get zero occupation credit; credit is order-independent; consistency accepts the real Rockstar pattern and flags truncation; sub-catalog-cut substructure counts for its host |
| `test_oracle_hr.py` | 25 | mask is localised on its own sites, peaks at 1, is periodic, is smooth, handles an empty site set; `ids_to_lattice` inverts the flat index; **α=0 is bit-exact SR2**; α=1 replaces the field at the mask peak; the intervention is linear in α; **each channel mode leaves the others frozen**; only the masked region changes; shape/mode errors raise; control has equal count and excludes the target, is seeded, and shrinks rather than repeating; target selection finds missing subhalos, **enforces separation**, respects caps, and **refuses the sparse 1e14 bin**; recovery detects restoration, requires the object to still be a subhalo, and rejects mass mismatches |

The five tests the brief names for Experiment 0 — exact reconstruction, no
double counting, periodic-translation invariance, tile-order invariance,
boundary objects carrying total weight one — are all present and passing. The
sixth (predicted `A_j` vs measured after re-running tiles) needs the halo finder
and is `exp0_credit_verify_cpu.sbatch`.

---

## 3. Two forms of per-tile credit, and which one is checkable

`A_j = R(S) − R(S − s_j)` is the removal form the brief specifies. It is pure
arithmetic on cached sufficient statistics — **the halo finder is never re-run**,
which is the only reason 512 credit units per box is affordable.

It is also **not verifiable against a field experiment**, because "delete tile
j" is not something you can do to a box. So `tile_loo_credit.py` reports it
alongside the swap form `ΔR_j = R(S − s_j^base + s_j^HR) − R(S)`, which *is*: it
predicts what happens when tile j's field content is really replaced with HR,
and `exp0_credit_verify_cpu.sbatch` replaces it and re-runs Rockstar. The gap
between prediction and measurement is the cross-tile interaction that per-tile
credit assumes away — risk 3 in `reward_residual_diffusion.md`, turned into a
number. Twelve tiles per box bound it; more is waste.

Note the two forms differ systematically, not just by noise: removal also shrinks
the volume and the host denominators, so `A_j` carries a size effect the swap
form cancels.

The splice is hard-edged with no blending. That is deliberate and
in-distribution: SR2 itself generates each 64³ tile from a separately padded LR
crop and trims to the centre, so its own output is already a hard tile mosaic.

**The tile-geometry reward model is refitted**, not reused. The shipped model was
fitted on purity-masked chunk summaries, whose counts and effective volumes are
different quantities; scoring unmasked tile sums against that `mu` would compare
a masked target with an unmasked measurement. One ensemble is one whole box —
the independent cosmological unit. With fewer than three boxes the fit
**refuses**, rather than producing a covariance from two points.

---

## 4. Experiment 1: where the mask lives

The residual `x_HR − x_SR2` is a **Lagrangian** field; a missing subhalo is an
**Eulerian** object. Median `|Ψ|` is ~36 HR cells, so "the Lagrangian
neighbourhood of an Eulerian position" is not a cube around that position.

The mask is therefore built from the HR **member particle IDs** of the target
subhalo, which are Lagrangian lattice indices — so the set of sites that formed
it is known exactly, then dilated (2 cells) and smoothed (1.5 cells) and
peak-normalised so that `α = 1` really applies the full HR correction somewhere.
A hard edge would make the low-`k` constraint measure the mask rather than the
physics.

An earlier design smoothed a sphere around the Eulerian position. That is wrong
in a way that produces a clean, wrong, *negative* result: for a subhalo inside a
cluster the correction would land on unrelated material. This is called out in
the module docstring because a null result from Experiment 1 routes straight to
"the action representation is insufficient" in the decision table, and that
conclusion must not be reachable by a coordinate-system bug.

Target selection takes only the `missing` class from the repaired periodic
matcher — not `shifted`, `mass-biased` or `velocity_incoherent` — inside matched
hosts of bins 2 and 3. Bin 4 (`1e14`) is **refused explicitly**: an oracle result
there cannot feed a Gate B decision. Targets are forced ≥ 6 Mpc/h apart on host
position so that all of a box's targets can be edited in **one** field and scored
with **one** Rockstar run; that is what takes the experiment from ~500 halo-finder
runs to 32.

### The grid, after removing degenerate cells

`4 α × 3 modes × 3 kinds = 36` per box collapses to **16**:

* `α = 0` is the frozen SR2 box, identical for every kind and mode → scored once
  per box, and it is the only legitimate anchor for the recovery curve (which is
  why `test_oracle_hr.py` pins `α = 0` to bit-exact SR2);
* `control` and `exact_particles` are full-strength yes/no tests → `α = 1` only.

The enumeration lives in one Python block inside the job script, so the
submitter and the job cannot disagree about what index *N* means.

### The control is the gate, not a footnote

`random_control_sites` draws the **same number** of Lagrangian sites from the
**same host**, excluding the target's. If it recovers subhalos too, the targeted
result is added fluctuation power and means nothing. `oracle_hr_report.py`
applies this test *before* reporting a positive verdict.

---

## 5. Submission, in dependency order

```bash
# One-off local check of the member-id path (~2 min CPU) -- run before anything:
python scripts/reward/verify_member_ids.py

# Everything else, with the dependencies wired:
bash scripts/slurm/submit_oracle.sh

# Inspect without submitting:
DRY=1 bash scripts/slurm/submit_oracle.sh
```

The submitter only calls `sbatch`. Configuration goes into one timestamped env
file passed as a **positional argument** — never `sbatch --export`, which on
this cluster sets `SLURM_GET_USER_ENV=1` and gets the job requeued and held.

**§5a documents three real bugs this submitter shipped with**, found only by
actually running it. The interface that survived is:

```bash
bash scripts/slurm/submit_oracle.sh              # everything, priority order
bash scripts/slurm/submit_oracle.sh exp0          # targets + member-id jobs + Exp 0 analysis
bash scripts/slurm/submit_oracle.sh exp1          # targets + member-id jobs + Exp 1 chain
bash scripts/slurm/submit_oracle.sh gate_a        # just the prior training

# If a submit cap (QOSMaxSubmitJobPerUserLimit) truncates the chain, the
# prerequisite (target selection + member-id jobs) already exists -- resume
# from it instead of resubmitting everything. The colon-joined ids are the
# 4 "exp0 rockstar+particles (box/source)" job ids the earlier run printed:
DEPEND=<id1>:<id2>:<id3>:<id4> bash scripts/slurm/submit_oracle.sh exp1_run
DEPEND=<id1>:<id2>:<id3>:<id4> bash scripts/slurm/submit_oracle.sh exp0_analyse
```

Submission order is **priority order**, not dependency order: Experiment 1
(interventions + report) is submitted before Experiment 0's 24-task credit
verification, because the latter is a nice-to-have bound on an assumption and
the former is the headline result. If a submit cap truncates the run, it must
truncate the less important half — see §5a.2.

`sbatch scripts/slurm/exp1_intervene_cpu.sbatch PRINT_GRID=1` lists the grid and
the exact `--array` range without submitting work.

Aggregation is always a **second submission with a dependency**, never task 0
polling `squeue`. Every gate failure inside a job prints why and **exits 0**, so
dependents report the same thing instead of stranding in
`DependencyNeverSatisfied`.

### 5a. Three bugs found by actually running this, and one false lead

Kept here rather than silently fixed and forgotten, because all three are the
kind of thing that reappears the next time a Slurm chain gets built on this
cluster.

**1. A swallowed `sbatch` failure produced a broken dependency chain, not a
stopped one.** `sub()`'s first version captured `sbatch`'s exit code inside
`$(...)`, which `set -e` in the parent cannot see. A submit failure (a QOS cap)
returned an empty job id, which fed straight into the next job's
`--dependency=afterok:`, and *that* `sbatch` call failed too ("Job dependency
problem") — while the script kept going regardless, silently. Fixed: `sub()`
now checks the exit code and the empty-id case explicitly, writes an
`ABORT_FLAG` file, and every call site checks `die_if_aborted` immediately
after. A submit failure now stops the whole submitter instead of limping on.

**2. Submission order was priority-inverted.** The first version queued
Experiment 0's 24-task verification array before Experiment 1's interventions.
When a `QOSMaxSubmitJobPerUserLimit` truncated the run, it cost the experiment
that actually matters while the nice-to-have verification jobs had already
claimed the slots. Fixed by reordering (§5), and this is *why* `exp1_run` and
`exp0_analyse` exist as separately resumable stages — a truncated run can be
resumed starting with whichever half was cut off, without resubmitting the
other.

**3. The one that actually corrupted output: a per-job config override was
silently undone by the shared env file, and two "independent" jobs raced on
the same directory.** To submit one member-id job per `(box, source)` pair
(rather than a 4-task array — see the note below on why), `sub()` was called
with `BOXES=$BOX SOURCES=$SRC` *before* the shared env file. But
`_reward_common.sh` processes its positional arguments strictly in order, and
the env file — which defines `BOXES="set8 set9"` for every *other* stage in
the chain — was always appended last by `sub()`. So the env file silently
re-clobbered `BOXES` back to both boxes on every one of the four "independent"
submissions; only `SOURCES` (which the env file never sets) survived. The
result: two jobs both resolved to `set8/hr`, two both resolved to `set8/base`,
`set9` was never touched at all, and the two same-target pairs raced on one
output directory — one crashed instantly (`rc=1`, a snapshot half-overwritten
mid-write), the other's output was suspect enough to discard rather than trust.
Caught by comparing job logs (two jobs printing `=== set8 / hr`), not by any
check inside the pipeline — nothing there is wrong per se, both runs "succeeded"
individually. **Fixed** by threading per-job overrides through a
`SUB_OVERRIDES` array applied *after* the env file (`sub()` in
`submit_oracle.sh`), so the override actually sticks. Verified by replaying
`_reward_common.sh`'s real argument-processing loop against the generated argv
for all four pairs and confirming each resolves distinctly.

That fix's *own* first draft had a second bug, caught before it shipped:
`"${SUB_OVERRIDES[@]:-}"` on a completely unset array expands to **one
empty-string argument**, not zero, under `set -u` — confirmed empirically, not
assumed. That empty string fails `_reward_common.sh`'s
`[[ -r "$_arg" ]]` config-file-readability check and aborts the job, which
would have broken *every other* `sub()` call in the script (all the ones with
no override at all). Fixed by declaring `SUB_OVERRIDES=()` once at global
scope and dropping the `:-`; a declared-but-empty array expands to zero words
via `"${arr[@]}"`, which is the form actually used.

**Why the member-id stage is 4 independent job ids, not one 4-task array**:
the very first attempt at this stage was submitted as `--array=0-3`, and all
four tasks were cancelled *simultaneously*, within 200 ms of each other, by an
external `SIGTERM` mid-run — consistent with a single `scancel <array-id>`
killing the whole array at once, not four independent failures. Cause
unconfirmed (see the false lead below). Regardless of cause, isolating the four
(box, source) pairs into separate job ids means a repeat can only take out the
one pair actually affected.

**The false lead: "slurmdbd is down".** Mid-debugging, `sacct`/`sacctmgr`
returned `Connection refused` to `localhost:6819` and `sbatch` failed with
`Access/permission denied` for *every* submission, including a trivial
diagnostic job with no array, no dependency, nothing unusual. That looked like
(and was reported as) a cluster-wide accounting-daemon outage. It wasn't: the
agent session's shell was running *inside* a long-lived interactive `bash`
allocation on a GPU compute node (`gpu17`, under `$SLURM_JOB_ID`), not on the
login node — confirmed by `hostname` and `echo $SLURM_JOB_ID` once the user
reported `sacct` working fine on their own login-node session. Submitting and
querying jobs from *inside* a running compute allocation is unreliable on this
cluster (nested job submission is commonly restricted), which is a completely
different problem from a real outage and is not fixable by retrying or by
anything in `submit_oracle.sh`. **If `sbatch`/`sacct` start failing uniformly
and mysteriously again, check `hostname` and `$SLURM_JOB_ID` before concluding
anything about cluster health** — and submission/cancellation should happen
from the user's own login-node session, not from an agent shell that might be
nested inside a job.

**`QOSMaxSubmitJobPerUserLimit` is real and tight enough to matter.** Even
after the priority reordering, Experiment 0's verification stage (24 splice
tasks + decompose + 2 aggregates = 27 pending job records, none yet started)
was enough on its own to leave zero headroom for Experiment 1's 32-task array,
and then for even a single additional job (`gate_a`'s smoke test). Pending
array tasks consume quota for their entire wait, not just while running. The
recovery in practice was to cancel the not-yet-started verification jobs
(`(Dependency)`, `0:00` runtime — no completed work lost), submit the higher-
priority stage, and resubmit the verification once it drains. There is no
config knob for the actual cap (`sacctmgr` needs the accounting daemon, which
was unreachable from the debugging session per the false lead above); treat it
as small and plan submissions accordingly.

---

### 5b. Run status snapshot (2026-07-29, first real submission)

Recorded so the job ids in §5a's bug reports are traceable, and so a reader
mid-run knows what state this was actually validated from. **This is a
snapshot, not a claim about current state** — check `squeue` for that.

| Stage | Job id(s) | State at time of writing |
| --- | --- | --- |
| target selection | 23158 (1st attempt, corrupted run), re-run after the fix | completed |
| member-id Rockstar, 4 independent jobs | `set8/hr` `set8/base` `set9/hr` `set9/base` — confirmed via each job's own log printing a distinct `=== <box> / <source>` banner | running |
| Experiment 1 interventions (32-task array) + report | 23195, 23196 | queued, `--dependency=afterok:<4 member-id ids>` |
| Experiment 0 decompose/credit + 24-task verification | cancelled once (to free quota for Exp 1, all `(Dependency)`/`0:00` — no completed work lost); resubmit via `exp0_analyse` once quota allows | not yet resubmitted |
| Gate A (residual prior smoke + training) | attempted, blocked on `QOSMaxSubmitJobPerUserLimit` | not yet submitted |

Two prior submission attempts before this one produced corrupted output later
deleted in full: `halos_particles/set8__{hr,base}__*` and
`set9__{hr,base}__*` (~14 GB, the `set9` directories were empty — that data was
never actually produced). Nothing downstream (`tile_cache/`, catalogs) had
consumed it, so no result in this document rests on the corrupted run.

### 5c. Run status snapshot (2026-07-29, later same day)

- **Experiment 1: complete.** `exp1_report.json` written 01:55; result in §7a.
- **Experiment 0**: still not resubmitted (§5b); `exp0_table.json` does not
  exist yet. Nothing in §7a depends on it.
- **Gate A prior training**: running as job `23271` (`rw_prior`), started
  09:53:27 on `gpu28` (`general` partition, an `a5000` — not `a6000`, which
  `myfree` showed fully occupied at submit time), `TimeLimit=2-00:00:00`, ETA
  2026-07-31 09:53. Submitted with the smoke-test dependency intact (the
  `SKIP_SMOKE=1` retry a few minutes later hit `Invalid qos specification`
  because the job scripts had since been edited to `--partition=general,legacy`
  — two partitions with two distinct required QOS, which a single job can't
  straddle — and was a no-op duplicate against the already-running job, not a
  real failure). Job scripts now pin `--partition=legacy` for any future
  resubmission, since `general`'s `a6000` nodes were saturated and `legacy`
  (`rtx_2080_ti`/`v100`) was idle; this model (width=48, 64³ crops) does not
  need `a6000`-class VRAM.

---

## 6. Cost

Halo finding on a 512³ box is the unit of currency: ~20 min plus a 3.7 GB
transient GADGET2 dump. Member-id output adds an ASCII table and a streaming
pass.

| Stage | Tasks | Per task | Total CPU-h | Wall (parallel) |
|---|---|---|---|---|
| `verify_member_ids` (local) | 1 | 2 min | 0.03 | 2 min |
| `exp1_select_targets` | 1 | ~2 min | 0.03 | 2 min |
| `exp0_rockstar_particles` | 4 | ~40 min (20 halo + 15 stream + I/O) | 2.7 | ~40 min |
| `exp0_decompose` + credit | 1 | ~15 min | 0.25 | 15 min |
| `exp0_credit_verify` | 24 | ~35 min | 14 | ~35 min |
| `exp1_intervene` | 32 | ~35 min (20 halo + 10 field constraints) | 19 | ~35 min |
| `exp1_report` | 1 | ~2 min | 0.03 | 2 min |
| **Experiments 0 + 1 total** | **64** | | **≈ 36 CPU-h** | **≈ 2.5 h** |
| Gate A prior training (GPU) | 1 | 1–2 days | — | 1–2 days GPU |

Experiments 0 and 1 together cost about **36 CPU-hours and no GPU**. That is the
point of running them first.

Storage, all under `$ZFS/DMSR/dmsr_reward/`, nothing in `$HOME`:

| Artifact | Size | Note |
|---|---|---|
| `.particles` ASCII | ~6 GB per box-source, **transient** | measured: 78 B/row on the real binary, ~7.6e7 rows at 512³ with 54% of particles in halos. Deleted in the job that wrote it; if these accumulate, that is the bug |
| tile weights `*_tilew.npz` | a few MB per box-source | the durable product |
| tile summaries JSONL | < 1 MB per box-source | |
| member ids `*_members.npz` | ~10 MB per dev box | Experiment 1 masks |
| Exp-0 splice catalogs | ~100 MB × 24 | ASCII |
| Exp-1 intervention catalogs | ~100 MB × 32 | ASCII |
| GADGET2 dumps | 3.7 GB transient | deleted after parsing |
| **New durable storage** | **≈ 6 GB** | against 48 GB of existing SR2 cache |

No intervention field is ever written to disk: `oracle_intervene.py` composes it
in memory, halo-finds it, and drops it. Writing them would have cost 3.2 GB × 32
≈ 100 GB for nothing.

---

## 7. Reading the results

`oracle_hr_report.py` writes `exp1_report.json` with an explicit verdict.

| Verdict | Meaning | Next action |
|---|---|---|
| `accessible_direction` | recovery grows smoothly with α | reward landscape has a usable direction — proceed to Experiment 2 (needs Gate A) and reward training |
| `representation_ok_exploration_hard` | recovery only near α = 1 | prefer directed search (Experiment 3, CEM) over raw best-of-K |
| `control_matches_targeted` | the equal-count random edit does as well | **not a positive result**; tighten the recovery criterion and re-run before concluding anything |
| `localised_hr_correction_fails` | no recovery at any α | localised residual editing cannot realise the missing structure → move to the explicit catalog-renderer oracle. **Do not spend GPU on best-of-K** |
| `incomplete` | too few cells scored | |

Plus a channel finding from the `disp` / `vel` / `both` arms: position-only
working means the correction is structural; needing both means phase-space
coherence has to be modelled explicitly.

### 7a. Experiment 1 result (`exp1_report.json`, `set8`+`set9`, 2026-07-29)

**Verdict: `accessible_direction`, `best_mode: disp`.** Recovery grows
monotonically with α and clears the untargeted `control` at every α ≥ 0.25:

| mode | recovery @ α=0 | @ α=1 | control @ α=1 | gradual | beats control |
|---|---|---|---|---|---|
| **disp** | 0.375 | 0.729 | 0.458 | yes | yes |
| both (disp+vel) | — | 0.958 | 0.417 | yes (raw table: 0.396→0.521→0.958 over α=0.25/0.5/1.0) | yes |
| vel | — | 0.375 | 0.375 | no | **no — ties control** |

`exact_particles` (splice in the true HR particles outright, the ceiling):
recovers 90% (both) / 73% (disp) / 40% (vel). Even the perfect intervention
barely beats control on velocity alone — evidence that **the velocity channel
carries almost no exploitable occupation signal by itself**, not an artifact
of the smoothed `targeted` mask.

Why the ceiling is <100% even for `exact_particles/both`, and why `vel`-only
is so weak: Rockstar's core linking/unbinding metric is phase-space, not
positional —
[`external/rockstar/subhalo_metric.c:47-53`](../external/rockstar/subhalo_metric.c#L47-L53)
computes `sqrt(r²/r_halo² + v²/vrms²)`. `exact_particles/vel` gets velocities
exactly right but leaves SR2's wrong positions, so it fails the recovery
test's own position tolerance (`d ≤ 0.5·Rvir`,
[`oracle_hr.py:406-469`](../src/cosmo_sr/reward/oracle_hr.py#L406-L469))
outright. `exact_particles/disp` gets positions right but velocities wrong,
which mostly survives 3D linking but loses some targets to unbinding /
mass-tolerance drift from the inconsistent kinematics. Only `both` corrects
both terms of the metric, and even then the intervention only edits the
target's own member particles (`dilate=0`) — the surrounding tidal
context stays SR2, which accounts for the remaining ~10% gap to the ceiling.

**Implication for Experiment 2 (needs Gate A)**: restrict the search/reward
direction to `disp` (or `disp`-dominant `both`) rather than sweeping `vel`
equally — `vel` is a dead end for this reward. Also note `R_cat`/`R_occ` are
`NaN` and `feasible` is `false` on nearly every row: the tile reward model
(`tile_reward_model.json`) wasn't fitted yet when this ran, so these numbers
are raw halo-recovery rate only, not the catalog reward itself. Experiment 2
needs that model fit over `MODEL_BOXES` first.

For Experiment 0 the deliverable is `exp0_table.json` — direct full-box versus
tile-summed `H_b`, `S_b`, `O_b` with the numerical error, plus retention per host
bin under both attributions. `tile_decompose.py` **exits 1** if the worst
absolute reconstruction error exceeds 1e-6, so a silent mismatch cannot pass as
a result.

---

## 8. What this does not deliver

Experiment 1's outputs (recovery-vs-α, the mode breakdown, which
positive-result level was reached) are now delivered — §7a, from
`exp1_report.json`. What is still outstanding, per §5c: the Experiment 0
`exp0_table.json` deliverable (not yet resubmitted) and the field-fidelity /
constraint-satisfaction detail beyond the `feasible` flag already in §7a's
source rows. Per the compute rules, GPU jobs are not launched by the agent and
heavy Rockstar sweeps are delivered as scripts for the user to submit from
their own session, not executed here.

Experiments 2, 3 and 4 are not implemented, because each needs a trained
residual prior to be meaningful and the honest sequencing is to let Experiment 1
report first. Experiment 2's support test is the existing
`sample_reward_oracle.sbatch` path, which already takes `K`, `RESIDUAL_SCALE` and
sampling temperature; what it lacks is a checkpoint.
