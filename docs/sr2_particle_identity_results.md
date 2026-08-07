# SR2 vs HR particle identity: measured results (set8, z=0)

Results from [`scripts/sr2/particle_identity.py`](../scripts/sr2/particle_identity.py)
(SLURM jobs `pid_prep` 26260/26280/26281, `pid_analyse` 26289/26290). Companion
to [`sr2_particle_identity.md`](sr2_particle_identity.md), which defines the
method; this document records what the catalogs actually say.

**Question.** When SR2 produces a halo that looks valid, is it built from the
same *mass elements* as its HR counterpart? If not, a residual trained pointwise
on SR2 → HR must take a self-consistent SR2 object apart and rebuild an HR one
from different particles.

## Scope and provenance

| | |
|---|---|
| Box | `set8` only, \(z=0\) (a **val** box, not dev/final-eval) |
| Sides | HR, SR2 seed 0, SR2 seed 1 |
| Catalogs | Rockstar with `FULL_PARTICLE_CHUNKS=1`, frozen `rockstar_particles.cfg` |
| Model | `G_z0.pt`, `nsplit=8`, `pad=3`, `scale_factor=8`, \(64^3 \to 512^3\) |
| Selection | `num_p >= 50`; `--max-pairs 20000` per class |
| Artifacts | `dmsr_reward/particle_identity/set8__hr__base0/`, `set8__base0__base1/` |

**Why this is exact.** HR and SR2 are displacement fields on the same Lagrangian
lattice with `id = arange(Ng**3)`, so particle *i* is the same mass element in
both. Membership compares as sets of integers — no matching, no tolerance.

**Leaf attribution verified on real data.** `len(members(o)) == num_p(o)` held
for **every object in all three boxes**, `max |diff| = 0`:

| side | objects | exact | unowned particles |
|---|---|---|---|
| HR | 315,998 | 315998/315998 | 0.460 |
| SR2 seed 0 | 213,080 | 213080/213080 | 0.485 |
| SR2 seed 1 | 213,256 | 213256/213256 | 0.484 |

The unowned fraction is diffuse material bound to no halo, as expected at
\(z=0\). Both frozen-configuration re-runs reproduced the frozen catalog
object-for-object (`same_ids: True`, `max_dmvir 0.0`, `max_dpos 0.0`,
`max_dnum_p 0`), so re-running the halo finder cost nothing.

## Headline

**SR2 halos are not made of the same particles as their HR counterparts.** The
failure is severe at \(R_{\rm vir}\), absent at chunk scale, and strongly
mass-dependent.

| matched pairs, SR2 vs HR | hosts | subhalos |
|---|---|---|
| matched / analysed | 17,062 / 20,000 | 12,581 / 20,000 |
| **median Jaccard** | **0.042** | **0.000** |
| median purity / completeness | 0.159 / 0.060 | 0.000 / 0.000 |
| share with Jaccard \(>0.5\) | 6.3% | 0.06% |
| share sharing **zero** particles | 28.0% | 59.3% |
| random-id null | 0.000 | 0.000 |

## 1. Identity — fails at \(R_{\rm vir}\), and the scatter is incoherent

The correction a pointwise residual must apply, split into a coherent bulk
translation and the scatter about it:

| | hosts | subhalos |
|---|---|---|
| median \(R_{\rm vir}\) | 0.141 Mpc/h | 0.110 Mpc/h |
| bulk shift | 0.483 Mpc/h (**3.7 \(R_{\rm vir}\)**) | 0.663 (**6.1 \(R_{\rm vir}\)**) |
| residual scatter | 0.949 Mpc/h (**7.2 \(R_{\rm vir}\)**) | 0.816 (**6.9 \(R_{\rm vir}\)**) |
| coherent fraction | 0.213 | 0.426 |

Most of the required motion is **scatter, not translation**. The residual is not
being asked to move a valid object; it is being asked to disperse one and
reassemble it.

### Matching-free corroboration

Matching is the weakest link in every SR2-vs-HR comparison, so these two need no
matcher — the ids are known, and the other box's ownership of them is a lookup:

| for an HR object's particles, in SR2 | hosts | subhalos |
|---|---|---|
| largest single SR2 destination holds | 36.1% | 44.1% |
| unbound in SR2 entirely | 41.5% | 31.7% |

## 2. Mass dependence — identity survives at the top, collapses at the bottom

| selection | hosts Jaccard | scatter/\(R_{\rm vir}\) | subhalos Jaccard |
|---|---|---|---|
| all matched | 0.042 | 7.16 | 0.000 |
| `num_p >= 1,000` | 0.257 | 4.23 | 0.004 |
| `num_p >= 10,000` | **0.576** | 2.00 | 0.234 |

\(n = 5{,}710\) and \(603\) hosts; \(2{,}358\) and \(187\) subhalos. This is the
particle-level counterpart of the abundance/occupancy result in
[`sr2_subhalo_results.md`](sr2_subhalo_results.md) §1–§2: SR2 reproduces the
massive end and loses the small end, and the same boundary shows up in *which
particles* it uses.

## 3. Radius — spatial slack does not rescue it

Fraction of one object's members lying within \(1\,R_{\rm vir}\) of the other
object's centre, measured in the other box:

| | hosts | subhalos |
|---|---|---|
| SR2 members near the HR centre | 0.092 | 0.000 |
| HR members near the SR2 centre | 0.044 | 0.004 |

Relaxing membership to proximity does not recover the correspondence, because
the displacements (\(\sim 7\,R_{\rm vir}\)) are large compared with the objects.

## 4. Chunk — the disagreement is local, in the Eulerian sense only

The generator's geometry, from [`tile_sr.py`](../src/cosmo_sr/inference/tile_sr.py):

| | LR cells | Mpc/h |
|---|---|---|
| output tile (`nsplit=8` on `ng_lr=64`) | 8 → 64 HR | **12.5** |
| input crop (`pad_lr=3` per side) | 8 + 2·3 = 14 | **21.9** |

Comparing each particle's **HR cell** against its **SR2 cell** on a 12.5 Mpc/h
Eulerian grid:

| | hosts | subhalos | all particles |
|---|---|---|---|
| same cell, **particle-weighted** | 0.884 | 0.882 | 0.874 |
| same cell, median over objects | 0.997 | 1.000 | — |
| objects losing \(>10\%\) from their cell | 28.6% | 28.0% | — |
| **same or adjacent cell** | **1.0** | **1.0** | **1.0** |

`same_or_adjacent` is exactly 1.0 — not rounded — over 1,985,201 sampled
particles and all 40,000 objects, in both the main run and the control. **No
particle's HR and SR2 cells ever differ by more than one.**

> **Report the particle-weighted number, not the median over objects.** A
> compact halo lying inside one cell scores 1.0 whatever its members do, and
> those halos are the majority, so the median saturates at ~1 and hides the
> straddling tail (0.997 vs 0.884). Chunk crossing is driven by proximity to a
> face, not by how far particles move. `summary.json` now carries
> `weighted_frac_same_chunk` alongside the median for this reason.

### What this does *not* say

This measures the **disagreement between the two boxes**, both at their final
\(z=0\) positions. It says nothing about how far a particle travels from its
Lagrangian lattice site — which is a much larger quantity:

| \(\lvert x_{\rm final} - q \rvert\) | median | rms | p90 | max |
|---|---|---|---|---|
| HR | 7.31 | 7.50 | 9.47 | 13.19 Mpc/h |
| SR2 seed 0 | 7.22 | 7.40 | 9.40 | 12.07 Mpc/h |

Particles travel \(\sim\)**7.3 Mpc/h**, comparable to the whole 12.5 Mpc/h
output tile and, at the tail, exceeding it. So it is **false** that a particle
stays near its Lagrangian chunk. What is true is that HR and SR2 agree with each
other to \(\sim\)0.9 Mpc/h *on that 7.3 Mpc/h journey* — a ~12% relative
agreement on the displacement, which is why the two boxes' large-scale structure
matches while the small-scale membership does not.

Decomposed within one \(64^3\) tile (HR):

| | Mpc/h |
|---|---|
| tile bulk translation \(\lvert\langle d\rangle\rvert\) (moves the tile) | 5.48 |
| within-tile dispersion rms (deforms/smears it) | **3.60** |

### How many Lagrangian tiles does one halo draw from?

| | median | mean | p90 | max | from a single tile |
|---|---|---|---|---|---|
| hosts | 2 | 2.24 | 4 | 36 | **36.8%** |
| subhalos | 2 | 2.51 | 4 | 23 | 30.3% |

Only about a third of halos come from one generator tile. Matched pairs
nevertheless draw from nearly the *same* tiles — profile overlap **0.941**
(hosts) / 0.725 (subhalos) — so tile attribution is stable even though it is
rarely single-valued.

## 5. The control — this is systematic, not sampling noise

Two SR2 noise seeds, same LR, same frozen model:

| | SR2 vs HR | SR2 seed0 vs seed1 | ratio |
|---|---|---|---|
| typical particle displacement (median) | 0.918 Mpc/h | **0.028** Mpc/h | 33× |
| hosts median Jaccard | 0.042 | **0.906** | — |
| hosts, Jaccard \(>0.5\) | 6.3% | 92.7% | — |
| hosts residual scatter / \(R_{\rm vir}\) | 7.16 | 0.23 | 31× |
| median match distance | 0.720 Mpc/h | 0.015 Mpc/h | 47× |
| same Eulerian cell (particle-weighted) | 0.884 | 0.995 | — |

**Two consequences.**

1. **The residual has a well-defined target.** SR2's own sampling noise does not
   reshuffle particles, so the SR2↔HR mismatch is systematic and reproducible —
   a residual would not be chasing noise. What it must apply is ~1 Mpc/h of
   incoherent per-particle displacement (5–7 \(R_{\rm vir}\)).
2. **Best-of-N over noise seeds has almost nothing to exploit.** At the particle
   level the generator is nearly deterministic given LR. This corroborates the
   "seed-to-seed scatter ~0.3%" of [`sr2_subhalo_results.md`](sr2_subhalo_results.md)
   and points at Gate A's `fail_noise_ignore` branch rather than `pass_tts`.
   *Caveat: two seeds, one box, measured on positions rather than on catalog
   statistics.*

### Subhalo membership has its own instability floor

Between two nearly identical SR2 fields (particles moved 0.028 Mpc/h), subhalo
Jaccard is only **0.113** — 25.3% share zero particles, versus 0.906 and 6.1%
for hosts. Rockstar's host/sub boundary is itself unstable at low mass.

**Read the subhalo numbers against that floor, not against 1.0.** At
`num_p >= 10,000` the floor is 0.695 and SR2-vs-HR gives 0.234, so the
mismatch is real and far worse than finder noise — but the raw 0.000 overstates
it. Hosts have no such excuse: floor 0.906, measured 0.042.

## What these numbers do *not* support

**(a) The matching is loose, and some zero-overlap is "not the same object".**
Median match distance is 0.720 Mpc/h \(= 5.1\,R_{\rm vir}\), against 0.015
Mpc/h in the control. `match_hosts` searches \(\max(1\,{\rm Mpc}/h, 3R_{\rm vir})\),
so near the floor almost anything counts as matched — the same trap as
[`sr2_subhalo_results.md`](sr2_subhalo_results.md) (a). **Use the matching-free
numbers (§1) for low-mass claims.** They agree: 41.5% of an HR host's particles
are unbound in SR2 regardless of any matcher.

**(b) The quoted medians are optimistic.** `--max-pairs 20000` takes the top
10,000 by mass plus 10,000 at random, and identity improves steeply with mass
(§2). The true all-object medians are **worse** than quoted.

**(c) One box, one seed pair.** No box-to-box error bar. set9 has
`halos_particles` and is the cheap next box; bootstrap over **boxes**, never
over objects — the 20,000 objects here are not independent.

**(d) Sign convention.** The code computes `d = pos_to − pos_from`, i.e.
SR2 − HR in the main run, which is the *negative* of the residual correction as
described in the module docstring. Every quantity reported here is a norm or a
ratio, so nothing changes numerically; only the unreported `bulk_vec` component
flips sign.

## Implications for a chunk-local reward

**Supported.** Scoring a reward on chunk-SR2 against chunk-HR is well-posed. The
SR2↔HR disagreement never exceeds one output tile (§4, exactly 1.0), matched
pairs draw from nearly the same Lagrangian tiles (overlap 0.941), and the credit
machinery in [`tiles.py`](../src/cosmo_sr/reward/tiles.py) is exact on real data
(partition of unity \(2.2\times10^{-16}\), tile-sum vs direct full box
\(\sim10^{-12}\)).

**Forbidden.** The reward must be a **permutation-invariant statistic of the
particle set within the chunk** — never a per-particle or per-membership match.
Matched pairs share Jaccard 0.042/0.000, so a reward for "the same particles in
the same halo" targets something SR2 essentially never achieves. This is an
argument *for* the reward route over an \(L_2\) residual: a set statistic does
not care which particle went where, and the identity scrambling measured here is
exactly what breaks pointwise targets.

**Three constraints from the numbers.**

1. **Fractional credit is mandatory, not optional.** Only 36.8% of hosts come
   from a single generator tile (§4). Hard tile assignment would misattribute
   the majority; the existing fractional member weights already handle it
   exactly.
2. **Do not compute a per-chunk reward independently.** A chunk is 1/512 of the
   box, and the occupancy failure lives in massive hosts — 14 above
   \(1.3\times10^{14}\) in the whole box, i.e. ~0.03 per chunk. Bin by host mass
   with the chunk as the *credit* unit, which is what \(H[j,b]\), \(S[j,b]\) do.
3. **A Eulerian region cannot be attributed to one tile.** Particles travel
   ~7.3 Mpc/h from their Lagrangian sites with 3.6 Mpc/h of within-tile
   dispersion, so the material in a 12.5 Mpc/h Eulerian region originates across
   a \(\pm 1\)-tile neighbourhood. Lagrangian (id-based) attribution is exact;
   Eulerian-crop attribution is not.

**Convergent with the mass budget.** §5 of
[`sr2_subhalo_results.md`](sr2_subhalo_results.md) finds the missing mass never
leaves the halo — it moves from body to envelope at \(\delta \approx 10\)–100,
0.00% reaching a void. The particle view says the same thing: ~0.9 Mpc/h of
incoherent displacement, \(\approx 6\,R_{\rm vir}\), smearing material that HR
binds into subhalos through the host envelope. That suggests a reward term on
the **body-vs-envelope radial mass split within hosts** — chunk-computable, and
aimed at the measured failure rather than at halo counts. It also closes the
open gap §5 flagged ("no intra-halo destination for the mass"), for set8/set9.

## Open gaps

- **One box.** set9 next (`halos_particles` already exists, needs the owner
  arrays); set12 has none and is the box the rest of the study uses.
- **`--chunks 4`** (25 Mpc/h) would test against the 21.9 Mpc/h *input* receptive
  field rather than the output tile. Expected: zero crossings.
- **No random-membership null for the radius test** — whether 9.2%-within-
  \(R_{\rm vir}\) beats scattering the same number of particles at random in the
  host is unmeasured.
- **Displacement-field numbers are an 8-block sample** (262,144 of \(512^3\)
  particles), adequate for a magnitude but not for a tail.
- **`weighted_frac_same_chunk` post-dates the runs.** The summaries on disk lack
  it; the values in §4 were computed from `metrics.npz`. Re-running the analyse
  stage (~5 min, no new halo finding) regenerates them.
- \(z=2\) untouched.

## Reproduce

```bash
bash scripts/slurm/submit_particle_identity.sh          # prep + main + control
DRY=1 bash scripts/slurm/submit_particle_identity.sh    # print, submit nothing

# by hand
sbatch scripts/slurm/particle_identity_prep_cpu.sbatch BOXES=set8 SOURCES=hr
sbatch scripts/slurm/particle_identity_cpu.sbatch PID_BOX=set8 PID_A=hr PID_B=base:0
sbatch scripts/slurm/particle_identity_cpu.sbatch PID_BOX=set8 PID_STAGE=plot
```

Prep is ~20 min/side (Rockstar) and deletes the ~9.6 GB member table in-job,
keeping a 537 MB owner array. Analysis is CPU-only, ~5 min. Figures redraw from
`metrics.npz` alone: `fig1` identity vs null · `fig2` translation-vs-reshuffle
in \(R_{\rm vir}\) · `fig3` radius · `fig4` chunk · `fig5` matching-free fate vs
mass.
