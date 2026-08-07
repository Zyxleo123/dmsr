# SR2 halo / subhalo failure: measured results (set12, z=0)

Results from the Stage-1b report ([`scripts/sr2/subhalo_report.py`](../scripts/sr2/subhalo_report.py),
SLURM job `sr2_subrep`). Companion to [`sr2_subhalo_study.md`](sr2_subhalo_study.md),
which defines the study design; this document records what the frozen catalogs
actually say.

## Scope and provenance

| | |
|---|---|
| Box | `set12` only, \(z=0\) |
| Catalogs | Rockstar, HR + 8 SR2 noise seeds (`sr_seed0..7`) |
| Model | `external/SRS-map2map/SRmodel/G_z0.pt`, `nsplit=8`, `pad=3`, `noise_mode=per_tile` |
| Box size | 100 Mpc/h, particle mass \(5.819\times10^8\,M_\odot/h\) |
| Resolution floor | `MIN_HALO_OUTPUT_SIZE=20` particles \(= 1.16\times10^{10}\,M_\odot/h\) |
| Artifacts | `/zfsauton/scratch/yixiz/DMSR/sr2_baseline/stage1/subhalo_report/` |
| Positional analysis | seed 0 only (abundance/occupancy use all 8 seeds) |

**Caveats that apply to everything below.** One box, so there is no box-to-box
error bar — set13–15 have never been run (`DIAGNOSIS_STATUS.json`). These
catalogs were found with `PERIODIC=0` (upstream Rockstar forced it in serial
mode; since patched), which is identical for HR and SR but biases absolute
counts near box faces. Seed-to-seed scatter is ~0.3%, so the deficits below are
systematic, not sampling noise.

## Headline

| | HR | SR2 (8-seed mean) | ratio |
|---|---|---|---|
| all halos | 314,649 | 210,026 | 0.667 |
| hosts | 214,389 | 165,405 | 0.772 |
| subhalos | 100,260 | 44,621 | **0.445** |

Three distinct failures, in increasing order of severity:

1. **Abundance** — too few halos, confined to the resolution scale.
2. **Occupancy** — massive hosts are nearly empty. The dominant failure.
3. **Position** — massive halos are placed correctly; low-mass ones are not.

§5 then follows the mass those failures displace. It does not leave the halo.

## 1. Abundance deficit is confined to the resolution scale

*(fig1, fig2)*

The mass-function ratio is **0.9–1.0 above \(10^{11}\,M_\odot/h\)** and collapses
only near the 20-particle floor:

| | min ratio | at |
|---|---|---|
| host HMF | 0.41 | \(1.2\times10^{10}\) |
| subhalo SHMF | 0.21 | \(8.3\times10^{9}\) |
| \(V_{\max}\) function (hosts and subs) | 0.36 | 44 km/s |

The \(V_{\max}\) function localizes it most cleanly, since \(V_{\max}\) is less
sensitive to the finder's mass assignment: the deficit bottoms at 44 km/s and
recovers to ~1 by 200 km/s (hosts) / 500 km/s (subs).

**SR2 reproduces the massive end and loses the small end.** Any statistic
computed with a \(>10^{11}\) cut will look healthy — see the methodological
note below.

## 2. Occupancy collapse — the dominant failure

*(fig3)*

\(\langle N_{\rm sub}\,|\,M_{\rm host}\rangle\) tracks HR up to ~\(10^{12}\) and
then **saturates at 15–18 subhalos regardless of host mass**:

| \(M_{\rm host}\) | \(N_{\rm host}\) (HR) | HR | SR2 | ratio |
|---|---|---|---|---|
| \(4.2\times10^{11}\) | 3,514 | 1.36 | 1.36 | 1.00 |
| \(1.3\times10^{12}\) | 1,254 | 3.98 | 3.51 | 0.88 |
| \(4.2\times10^{12}\) | 412 | 11.05 | 6.07 | 0.55 |
| \(1.3\times10^{13}\) | 131 | 33.9 | 10.7 | 0.31 |
| \(4.2\times10^{13}\) | 49 | 94.6 | 14.1 | 0.15 |
| \(1.3\times10^{14}\) | 14 | 281 | 16.4 | 0.06 |
| \(7.5\times10^{14}\) | 1 | 1406 | 44 | **0.03** |

Host counts per bin match HR to within a few percent throughout. **SR2 builds
the right hosts and then leaves them nearly empty.** Meanwhile
\(P(N_{\rm sub}\ge1\,|\,M_{\rm host})\) is essentially identical to HR: SR2 puts
*some* substructure in nearly every host, just an order of magnitude too little
in the big ones.

The clearest single image is the fig6 zoom: within \(3R_{\rm vir}\) of the most
massive HR host (\(7.4\times10^{14}\), \(R_{\rm vir}=1.86\) Mpc/h), **HR has
2409 halos and SR2 has 133** — an 18× local deficit against a 1.5× global one.
The deficit is concentrated in the densest regions. (One host, \(n=1\); the
mass-stratified version of that statement is the table above.)

## 3. Position: right at the top, wrong at the bottom

*(fig4, fig5, fig6)*

**Large scales and massive halos are correct.** The slab maps are visually
indistinguishable in filament and void structure. Matched-pair mass and
\(V_{\max}\) ratios reach 1.0 within a few percent above \(5\times10^{12}\), with
a tight 16–84% band. Spurious SR2 hosts fall to zero above \(10^{13}\). The most
massive host sits 0.22 Mpc/h from its HR counterpart \(= 0.12\,R_{\rm vir}\).

**Low-mass halos are not where the HR ones are.** Matcher-free nearest-neighbour
test, HR halo → nearest SR2 halo of any mass, against a uniform-random null at
the same number density:

| | observed | null | ratio |
|---|---|---|---|
| median NN distance, HR hosts | 0.643 Mpc/h | 0.924 Mpc/h | — |
| within 0.2 Mpc/h, HR hosts | 3.65% | 0.71% | 5.1× |
| within 0.2 Mpc/h, HR subhalos | 8.54% | 0.71% | 12× |

So there is real positional information — SR2 is ~5× better than random — but in
absolute terms only a few percent of HR hosts have an SR2 halo within 0.2 Mpc/h.
The signal strengthens steeply with mass (3.8× over null at \(10^{10}\), 54× at
\(1.3\times10^{13}\)). Median NN distance falls from 0.70 Mpc/h at the floor to
0.18 Mpc/h at \(10^{14}\) — roughly a third of an LR cell (the LR grid is
\(64^3\) over 100 Mpc/h = 1.56 Mpc/h cells) down to ~0.1 \(R_{\rm vir}\).

Reverse direction (false positives): **70% of SR2 hosts near the floor have no
HR halo of any kind within 0.5 Mpc/h**, falling to 51% at \(10^{11}\), 20% at
\(5\times10^{11}\), and ~0 above \(10^{13}\).

## 4. Subhalo classification

From `classify_subhalos` ([`halo_match.py`](../src/cosmo_sr/eval/halo_match.py)),
seed 0, rematched with the v2 matcher. Denominator is HR subhalos inside
successfully matched hosts (79,526 of 100,260 = 79.3%).

| class | n | % | definition |
|---|---|---|---|
| `missing` | 70,731 | 88.9% | no SR partner; HR sub outside \(0.25R_{\rm vir}\) |
| `spatially_shifted` | 6,614 | 8.3% | partner found, but \(\Delta x > 0.3R_{\rm vir,host}\) |
| `merged_into_host` | 1,842 | 2.3% | no partner, HR sub inside \(0.25R_{\rm vir}\) — plausibly absorbed into the central by the finder |
| `recovered_biased` | 286 | 0.36% | position OK, mass or \(V_{\max}\) outside a factor 1.65 |
| `velocity_incoherent` | 44 | 0.06% | position and mass OK, \(|\Delta v| > 0.75\max(V_{\max},50)\) |
| `recovered` | 9 | 0.01% | all four tolerances |

**The 89% `missing` is mostly forced by the occupancy deficit, not by
misplacement.** SR2 has 44,621 subhalos against HR's 100,260, so over half must
be unpaired by counting alone; in a \(10^{14}\) host holding 281 HR subs against
~16 SR subs, ≥94% *must* land in `missing`. This table largely restates §2.

The independent placement signal is in the conditional: of the 79,526 classified
subhalos only **6,953 (8.7%) found a partner at all**, and among those only
**339 (4.9%) landed within \(0.3R_{\rm vir,host}\)**. When SR2 does put a
subhalo in the right host, it is usually in the wrong place inside it.

## 5. The missing mass is not in the void — it never leaves the halo

*(`mass_budget` fig1–fig4)*

From [`scripts/sr2/mass_budget.py`](../scripts/sr2/mass_budget.py), SLURM job
`sr2_massbud`, artifacts in
`/zfsauton/scratch/yixiz/DMSR/sr2_baseline/stage1/mass_budget/`. Fields come
from the cached SR2 base displacements (`dmsr_reward/cache/sr2_base`, seeds
0–2) because the set12 SR `.gadget2` snapshots were deleted after Rockstar ran;
same frozen model sha, `nsplit=8`, `pad=3`, `per_tile`. Catalog numbers use all
8 seeds.

**SR2 cannot lose mass.** HR and SR2 are displacement fields on the same
Lagrangian lattice with `id = arange(Ng**3)`
([`particles.py`](../src/cosmo_sr/eval/particles.py)), so both boxes hold
exactly \(512^3\) equal-mass particles. Whatever §1–§2 fail to bind is
relocated, not destroyed, and particle identity makes the relocation directly
measurable.

### The collapsed-mass budget

Fraction of box mass bound in **host** halos (subhalos excluded — Rockstar's
host `mvir` already contains them):

| halo mass cut | HR | SR2 (8-seed) | ratio | \(\Delta M/M_{\rm box}\) |
|---|---|---|---|---|
| all | 0.3859 | 0.3522 | 0.913 | **−0.0337** |
| \(>10^{10}\) | 0.3814 | 0.3477 | 0.912 | −0.0337 |
| \(>10^{11}\) | 0.3314 | 0.3144 | 0.949 | −0.0169 |
| \(>10^{12}\) | 0.2582 | 0.2506 | 0.971 | −0.0076 |
| \(>10^{13}\) | 0.1657 | 0.1578 | 0.952 | −0.0079 |

Seed scatter 0.0007. **SR2 loses 33% of the halos but only 8.7% of the
collapsed mass** — the count deficit sits at the 20-particle floor where halos
weigh nothing, which is §1 restated in mass. 3.4% of total box mass is at
stake.

### Voids are *emptier* in SR2, not fuller

Mass-weighted fraction below a density threshold, CIC on a common mesh:

| | HR | SR2 | ratio |
|---|---|---|---|
| **128³ mesh** (0.78 Mpc/h) | | | |
| \(f_M(\delta<-0.9)\) | 0.01529 | 0.01690 | 1.105 |
| \(f_M(\delta<-0.8)\) | 0.05118 | 0.04990 | **0.975** |
| \(f_M(\delta<-0.5)\) | 0.13437 | 0.12628 | 0.940 |
| \(f_M(\delta<0)\) | 0.21984 | 0.20776 | 0.945 |
| **256³ mesh** (0.39 Mpc/h) | | | |
| \(f_M(\delta<-0.9)\) | 0.01964 | 0.01846 | 0.940 |
| \(f_M(\delta<-0.8)\) | 0.05238 | 0.04766 | **0.910** |
| \(f_M(\delta<-0.5)\) | 0.11752 | 0.11097 | 0.944 |

Seed scatter \(\le2\times10^{-5}\). The single bin that goes the other way
(\(\delta<-0.9\) at 128³) does not survive the change of mesh, so it carries no
weight. \(\sigma_\delta\) is also marginally *higher* in SR2 (8.588 vs 8.529 at
128³): SR2 is slightly more clumpy, not less.

### Where the 3.4% goes: into the halo envelope, from both sides

Net mass flux SR2 − HR as a fraction of total box mass, per-particle sampled on
the 128³ mesh, seed 0:

| \(\delta\) band | HR | SR2 | net |
|---|---|---|---|
| deep void, \(<-0.8\) | 0.0366 | 0.0361 | −0.0005 |
| void, \(-0.8\ldots-0.5\) | 0.0734 | 0.0658 | **−0.0076** |
| underdense, \(-0.5\ldots0\) | 0.0791 | 0.0745 | −0.0046 |
| mild, \(0\ldots10\) | 0.3449 | 0.3505 | +0.0056 |
| halo envelope, \(10\ldots100\) | 0.2792 | 0.2917 | **+0.0124** |
| halo body, \(100\ldots10^3\) | 0.1809 | 0.1763 | −0.0047 |
| core, \(10^3\ldots10^4\) | 0.0059 | 0.0052 | −0.0006 |

Mass converges on \(\delta\approx10\)–100 from **both directions** — out of the
voids *and* out of the halo bodies. That is a halo that is too puffy, not mass
dumped in a void.

### The migration matrix settles it

Same particle, both boxes: \(\delta_{\rm HR}\) at its HR position against
\(\delta_{\rm SR}\) at its SR2 position (fig3). Of the mass HR binds at
\(\delta>100\) (18.7% of the box), the SR2 destination is:

| destination | share |
|---|---|
| \(\delta<-0.8\) (void) | **0.00%** |
| \(\delta<0\) | 0.05% |
| \(0\ldots10\) | 3.3% |
| \(10\ldots100\) | 21.3% |
| \(>100\) | **75.3%** (median \(\delta=187\)) |

For HR's densest mass (\(\delta>10^3\)), \(P(\text{lands in an SR2 void})\) is
exactly **0.0** and 95.5% stays above \(\delta=100\). Reverse direction: of
SR2's void mass, **99.98% came from HR regions that were already underdense**
(53% HR voids, 36% \(-0.8\ldots-0.5\), 8% \(-0.5\ldots0\)). Essentially none of
it came out of a halo.

**Reading.** The 3.4% that SR2 fails to bind never leaves the halo — it moves
from the body out into the envelope at \(\delta\approx10\)–100, still dense,
just too diffuse and phase-space-smooth for Rockstar to bind. This is §2 seen
from the field side: SR2 builds the right hosts and fills them with a smooth
component instead of subhalos. With the 70% spurious near-floor hosts of §3, SR2
is if anything moving mass *up* the density ladder.

**Two limits on how far this goes.**

- The grid-based void fractions and the per-particle ones differ in absolute
  level (0.0512 vs 0.0366 at \(\delta<-0.8\)) because trilinear interpolation at
  a particle's own position smooths across cell boundaries. Each estimator is
  applied identically to HR and SR2, so the comparisons are fair, but quote the
  **grid** numbers for absolute void mass fractions.
- **A 0.78 (or 0.39) Mpc/h CIC cell cannot resolve a subhalo.** This rules out
  the void hypothesis and locates the mass at halo scale; it says nothing about
  how mass is redistributed *inside* a host. That needs the particle-membership
  route ([`particle_identity.py`](../scripts/sr2/particle_identity.py)), which
  has `halos_particles` for set8/set9 but not set12.
  **Now run for set8** — see
  [`sr2_particle_identity_results.md`](sr2_particle_identity_results.md). It
  agrees from the particle side: ~0.9 Mpc/h of *incoherent* per-particle
  displacement, \(\approx 6\,R_{\rm vir}\), smearing through the envelope the
  material HR binds into subhalos. Matched hosts share a median Jaccard of
  0.042; matched subhalos 0.000.

## What these numbers do *not* support

Three traps found while building this report; all are live risks for anyone
reading the figures.

**(a) The host-match rate of 0.683 is soft.** `match_hosts` searches
\(\max(1\ {\rm Mpc}/h,\ 3R_{\rm vir})\). For a near-floor halo
(\(R_{\rm vir}\approx50\) kpc/h) that 1 Mpc/h floor is ~20 \(R_{\rm vir}\), so
almost anything counts as matched. Consequences:

- The median matched-pair displacement of **14.7 \(R_{\rm vir}\)** is not a
  physical measurement — it is the linking floor divided by a steeply
  mass-dependent yardstick.
- The low-mass end of the fig5 mass-bias curve (median \(M_{\rm SR}/M_{\rm HR}
  = 0.25\) at \(4\times10^{11}\)) is **not a mass bias**. The 16–84% band there
  spans 0.04–1.20, a factor of 30, and the curve is non-monotonic — it recovers
  at *both* ends, which no genuine bias would do. It reflects the greedy matcher
  pairing HR halos with whatever near-floor SR halo was nearby and unclaimed.

Use the matcher-free tests (§3) for low-mass positional claims.

**(b) A \(10^{11}\) mass cut hides the entire abundance deficit.** Same slab,
varying the cut:

| mass cut | HR | SR2 | ratio | HR subs | SR2 subs | sub ratio |
|---|---|---|---|---|---|---|
| none | 26,637 | 17,142 | **0.64** | 8,864 | 3,806 | **0.43** |
| \(>3\times10^{10}\) | 8,235 | 6,588 | 0.80 | 2,703 | 2,105 | 0.78 |
| \(>10^{11}\) | 2,953 | 2,712 | **0.92** | 901 | 895 | **0.99** |
| \(>10^{12}\) | 394 | 344 | 0.87 | 113 | 87 | 0.77 |

fig6 now renders both cuts side by side plus a ratio-vs-cut curve, precisely so
this cannot be read selectively.

**(c) The fixed 0.2 Mpc/h placement tolerance is mass-inappropriate at the top
end.** At \(10^{14}\) the median NN distance is 0.18 Mpc/h — right at the
threshold — so `place_host ≈ 0.57` there measures the threshold, not a failure.
For massive hosts the mass-relative statement (0.18 Mpc/h against
\(R_{\rm vir}=1.2\)–1.9 Mpc/h, i.e. ~0.1 \(R_{\rm vir}\)) is the meaningful one.

## Open gaps

- **No box-to-box error bar.** set13–15 Stage-1 has never run. Every number here
  is one box.
- **No SR→HR null.** The uniform-random null covers HR→SR only. The spurious-host
  fraction (§3, 70% at the floor) therefore has an uncalibrated 0.5 Mpc/h
  threshold.
- **No random-placement null for subhalos inside hosts.** Whether 4.9%-within-
  \(0.3R_{\rm vir}\) beats scattering subhalos at random inside the host is
  unmeasured, so §4's conditional cannot yet be called worse than chance.
- **Positional analysis is seed 0 only.** Cheap to extend via `SR2_MATCH_SEEDS`.
- **No intra-halo destination for the §5 mass.** The mesh is too coarse to say
  where inside a host the unbound 3.4% sits. Needs Rockstar particle membership
  for set12 (`halos_particles` currently covers set8/set9 only).
- **§5 fields and §5 catalogs are not the same noise draw.** The cached SR2 base
  fields match the frozen config but the realisation for a given seed index is
  not guaranteed to be the `sr_seed*` draw of the same index. Seed scatter is
  \(\le2\times10^{-5}\) on the field statistics, so this cannot change the sign
  of anything above.
- \(z=2\) untouched; `G_z2.pt` is frozen but unused pending matched catnorm pairs.

## Reproduce

```bash
sbatch scripts/slurm/sr2_subhalo_report_cpu.sbatch
sbatch scripts/slurm/sr2_subhalo_report_cpu.sbatch SR2_STAGE=plot   # redraw only
```

Analysis writes `metrics.npz` + `summary.json`; plotting reads only those, so
figures are redrawable without re-parsing catalogs. Parsed catalogs are cached
per catalog under `cache/`. Overrides are `SR2_`-prefixed because the cluster
already exports generic names such as `SCRATCH=/home/scratch/$USER`.

Figures: `fig1` HMF/SHMF + ratio · `fig2` \(V_{\max}\) function · `fig3`
occupancy · `fig4` placement tests vs null · `fig5` matched-pair bias and
spurious fraction · `fig6` slab maps at two mass cuts, cluster zoom, and
ratio-vs-cut.

§5 is a separate job with the same two-stage split:

```bash
sbatch scripts/slurm/sr2_mass_budget_cpu.sbatch SR2_SEEDS=0,1,2
sbatch scripts/slurm/sr2_mass_budget_cpu.sbatch SR2_STAGE=plot   # after analyze
```

Sequential, not concurrent — every invocation writes `metrics.npz` into the
same `SR2_OUT`, so two analyze jobs submitted together race; pass a distinct
`SR2_OUT` for variants. Gates on the HR field, the cached SR2 base field and
the HR catalog, exiting 0 with a message when one is missing. Figures:
`fig1` \(\delta\) PDFs volume- and mass-weighted · `fig2` cumulative mass
fraction and the SR2−HR difference · `fig3` the migration matrix, raw and
row-normalised — **the one that carries the argument** · `fig4` mesh dependence
of the void fractions and the collapsed budget.
