# SR2 subhalo deficit: what is lost, and what kind of failure it is

**Scope.** A diagnosis of *why* the SR2 super-resolution catalog holds far fewer
subhalos than HR, and a design note for enforcing a subhalo budget. No model is
trained here. Every number in sections 1-5 is reproduced by
`scripts/features/analyze_sr2_subhalo_deficit.py --boxes set8`
(`sbatch scripts/slurm/subhalo_deficit_cpu.sbatch <envfile>`), which reads only
the committed `*_tilew.npz` weights and the existing Rockstar catalogs (no halo
finding, no owner arrays, no GPU) and writes `set8_subhalo_deficit.json`.
Section 6 has a different source -- the two `particle_identity/` runs -- and
**no committed script yet**; see Reproduce.

Context: `docs/lagrangian_host_features.md` builds LR host features on the 64^3
Lagrangian lattice; this document uses them, plus the SR2/HR tile weights, to
localise where SR2 fails.

> **Revision, 2026-08-18.** An earlier version of this document led with a
> per-tile correlation ("the more host material SR2 sees, the fewer subhalos it
> makes", corr = -0.50) and read an inflated mean subhalo size as evidence of
> *merging*. Both were re-derived. The correlation is real but is a shadow of
> host **mass**, not of local host fraction; the size inflation is survivorship.
> The section "What the earlier version got wrong" at the end lists the four
> corrections and the numbers that settle each.

> **Revision, 2026-08-18 (later).** Section 6 added: SR2 seed 0 vs seed 1 shows
> the model is a deterministic map in practice, so the deficit is systematic and
> not a bad draw. The same comparison retires the "go/no-go check" this document
> previously proposed -- it was run and cannot separate reproduced from lost
> objects, because subhalo-scale particle-identity metrics are saturated. Both
> are recorded under "Enforcing a subhalo budget in SR".

## The headline

On set8, over 512 Lagrangian tiles:

| | HR | SR2 | SR2/HR |
| --- | ---: | ---: | ---: |
| top-level hosts | 215399 | 167013 | 0.78 |
| subhalos | 100599 | 46067 | **0.46** |
| subhalos per host | 0.47 | 0.28 | 0.60 |
| bound-particle occupancy | 0.5683 | 0.5537 | 0.97 |

SR2 keeps **97%** of the bound particles but only **46%** of the subhalos. The
mass is there; the substructure is not. "Bound particle" means a particle
Rockstar judged gravitationally trapped by a halo (velocity test, not a radius
cut) -- the halo's genuine members, as opposed to the ~43% of particles in the
smooth field between halos.

And the deficit is not spread evenly over hosts. Matching every HR host above
1e14 to its nearest SR2 host (periodic, object by object -- 22 of 24 land within
0.5 Mpc/h, median separation 0.20 Mpc/h, median mass ratio **1.03**, median Rvir
ratio 1.00):

> those 24 clusters hold **12582** subhalos in HR and **924** in SR2 -- a factor
> **13.6**, against 2.2 for the box as a whole.

The clusters themselves are reproduced almost exactly. Their interiors are not.

## 1. The clean statement: subhalos per unit host mass

Subhalo abundance in HR is *linear* in host mass -- a fixed number of subhalos
per unit of host, at every mass. That is the premise everything else here rests
on, and it holds:

| host log M | HR mean N_sub | HR subs per 1e12 Msun | SR2 mean N_sub | SR2 subs per 1e12 Msun |
| --- | ---: | ---: | ---: | ---: |
| 10.0-10.5 | 0.05 | 2.97 | 0.04 | 1.99 |
| 10.5-11.0 | 0.23 | 4.20 | 0.11 | 1.90 |
| 11.0-11.5 | 0.69 | 4.02 | 0.49 | 2.93 |
| 11.5-12.0 | 1.99 | 3.71 | 2.03 | **3.71** |
| 12.0-12.5 | 5.79 | 3.41 | 5.30 | 3.08 |
| 12.5-13.0 | 17.98 | 3.30 | 9.16 | 1.69 |
| 13.0-13.5 | 53.26 | 3.13 | 17.11 | 1.02 |
| 13.5-14.0 | 173.04 | 3.39 | 26.52 | 0.49 |
| 14.0+ | 524.25 | 2.90 | 38.74 | **0.20** |

log-log slope of N_sub vs Mvir above 1e13: HR **1.01** (linear), SR2 **0.38**.

Read the two right-hand columns together. HR is flat at ~3-4 subhalos per 1e12
Msun/h across five decades of host mass: substructure *is* spread evenly over
host mass. SR2 matches HR exactly at 1e11.5-12 (3.71 vs 3.71) and then falls off
a cliff, to 0.20 at cluster mass -- a **19x** shortfall per unit mass at the top
end and none at all at the bottom.

**This is the failure, stated without tiles or correlations:** SR2's substructure
does not scale with host mass. The bigger the host, the emptier it is.

## 2. Which objects are lost, at what size

The earlier version tested only hosts above 1e11.5 and concluded "the hosts are
correct". That range covers **4.0%** of hosts. Over the full range:

| host log M | HR | SR2 | SR2/HR |
| --- | ---: | ---: | ---: |
| 9.5-10.0 | 44048 | 48158 | 1.09 |
| 10.0-10.5 | 103676 | 46001 | **0.44** |
| 10.5-11.0 | 39821 | 32004 | 0.80 |
| 11.0-11.5 | 15497 | 13254 | 0.86 |
| 11.5-12.0 | 5540 | 4989 | 0.90 |
| 12.0-12.5 | 2079 | 1996 | 0.96 |
| 12.5-13.0 | 764 | 734 | 0.96 |
| 13.0-13.5 | 232 | 224 | 0.97 |
| 13.5-14.0 | 77 | 69 | 0.90 |
| 14.0+ | 24 | 23 | 0.96 |

Below 1e11.5 (96% of all hosts) SR2/HR is **0.77**; above it, 0.92. So SR2 loses
small *hosts* too, ~23% of them. Comparing hosts and subhalos at equal particle
count says how much worse substructure has it:

| num_p | HR hosts | SR2 hosts | ratio | HR subs | SR2 subs | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20-50 | 118351 | 81573 | 0.69 | 52077 | 16040 | **0.31** |
| 50-100 | 46899 | 38353 | 0.82 | 23441 | 11115 | 0.47 |
| 100-200 | 24030 | 24073 | 1.00 | 12415 | 8599 | 0.69 |
| 200-500 | 15188 | 13127 | 0.86 | 7581 | 5933 | 0.78 |
| 500+ | 10931 | 9887 | 0.90 | 5085 | 4380 | 0.86 |

Two things at once: **small objects are lost** (both kinds, worst at the
20-50-particle floor), and at *equal size* a subhalo is roughly **twice** as
likely to be missing as a free host (0.31 vs 0.69). Being inside a bigger halo is
an extra penalty on top of being small. The corrected version of "the hosts are
correct" is: *hosts above ~200 particles are correct; everything small is thinned,
and small objects inside big hosts most of all.*

Raising a particle floor on the subhalo count confirms the size dependence:

| min particles | SR2/HR ratio | rank corr(LR occupancy, SR2 count) |
| ---: | ---: | ---: |
| 0 | 0.46 | -0.55 |
| 50 | 0.62 | -0.35 |
| 100 | 0.75 | -0.12 |
| 200 | 0.81 | +0.16 |
| 500 | 0.86 | +0.36 |

(The old table's third column, rank corr of the *relative deficit* with density,
is dropped from the argument: `rel = N_SR2/N_HR - 1`, so with the SR2 count flat
and the HR count rising in occupancy, its negative sign follows by arithmetic and
is not independent evidence. It is still in the JSON as
`spearman_lrocc_deficit`.)

## 3. Host fraction or host mass? Host mass.

Per tile define `L` = local host fraction (bound particles / 262144) and `logM` =
mass-weighted mean *total* mass of the hosts the tile's material belongs to, both
from HR truth. The two are **corr = +0.75**, so a marginal correlation with
either one cannot name the driver. Both partials, both directions:

| per-tile count | corr with L | corr with logM | partial(logM \| L) | partial(L \| logM) |
| --- | ---: | ---: | ---: | ---: |
| HR subhalos | +0.73 | +0.58 | **+0.07** | **+0.55** |
| SR2 subhalos | -0.50 | -0.62 | **-0.43** | **-0.07** |

* **HR is the local model.** At fixed total host mass the count tracks local
  material (+0.55); at fixed local material it does not care about total mass
  (+0.07). That is exactly section 1's linearity seen through tiles.
* **SR2 is the opposite.** Its dependence on local host fraction vanishes once
  mass is held fixed (**-0.07**); what survives is mass (-0.43).

So the statement "the more host material SR2 sees, the fewer subhalos it makes"
is **false as stated**. SR2's per-tile count is not driven by host fraction at
all. Tiles rich in host material are tiles whose material belongs to *massive*
hosts, and it is the mass that SR2 fails on. The marginal -0.50 is that
correlation's shadow.

Mean subhalo count per tile, binned by local fraction (rows) x total host mass
(columns), with the number of tiles in each cell -- the tertiles of two variables
correlated at 0.75 leave the off-diagonal corners nearly empty, so those cells
carry far less weight than the diagonal:

```
             low mass    mid    high mass      cell counts
HR   L low  :   145      168      181          113   43   15
     L mid  :   200      195      215           51   79   40
     L high :   245      231      235            7   48  116

SR2  L low  :   100       97       89
     L mid  :   118      102       84
     L high :   112       92       56
```

Along a *row* (mass rising at fixed L) SR2 falls everywhere; along a *column*
(L rising at fixed mass) it is flat or rising except in the top mass column. Same
conclusion as the partials.

## 4. SR2 loses objects in dense regions; it does not reclassify them

An alternative to "SR2 fails to build substructure" is bookkeeping: if SR2's big
hosts were slightly smaller, clumps that HR calls subhalos would fall outside
R_vir and be counted as separate hosts. Then the *total* object count would hold
up. It does not -- per-tile means by quintile of `L`:

| L | tiles | HR hosts | HR subs | SR2 hosts | SR2 subs | hosts | subs | all objects |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.39 | 103 | 567 | 139 | 536 | 96 | 0.95 | 0.69 | 0.89 |
| 0.49 | 102 | 515 | 178 | 438 | 106 | 0.85 | 0.59 | 0.78 |
| 0.56 | 102 | 441 | 202 | 337 | 103 | 0.76 | 0.51 | 0.68 |
| 0.64 | 102 | 364 | 226 | 226 | 89 | 0.62 | 0.39 | 0.53 |
| 0.76 | 103 | 217 | 237 | 95 | 57 | **0.44** | **0.24** | **0.33** |

In the densest quintile SR2 holds a third of HR's objects: 44% of the hosts and
24% of the subhalos. Objects are genuinely absent, and hosts go missing there
almost as badly as subhalos. (Note also that HR's own *host* count falls with `L`
while its subhalo count rises -- in dense tiles most small objects are somebody's
satellite, which is why the two curves cross.)

## 5. Attrition, not merging

Mean subhalo size (member particles) by LR host-occupancy quartile, with a
survivorship control -- the mean of HR's own **N largest** subhalos in the same
tiles, where N is however many SR2 produced there:

| LR occupancy | HR subs | SR2 subs | HR mean | SR2 mean | SR2/HR | HR mean of top-N | SR2 / top-N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00-0.13 | 19089 | 13091 | 119 | 180 | 1.51 | 163 | 1.11 |
| 0.13-0.27 | 23939 | 13134 | 157 | 255 | 1.63 | 262 | 0.97 |
| 0.27-0.45 | 27416 | 12182 | 197 | 369 | 1.87 | 402 | 0.92 |
| 0.45-0.95 | 30155 | 7660 | 281 | 790 | **2.81** | 972 | **0.81** |

The raw ratio rises to 2.8x, which reads as "SR2's subhalos are far too massive"
-- but keeping only HR's largest N reproduces it entirely. Against that control
SR2 sits at 1.11, 0.97, 0.92, **0.81**: its survivors are the same size as HR's
largest, and in the densest quartile 19% *lighter*. Nothing is oversized.

The subhalo size function inside massive hosts says the same thing. Within hosts
above 1e14, by num_p:

```
            20-50  50-100  100-200  200-500  500-2k   >2k     total sub particles
HR           5516    3133     1838     1206     641    248     4.60e6
SR2           136     125      134      164     188    144     2.86e6
ratio        0.02    0.04     0.07     0.14    0.29   0.58
```

SR2 is **below** HR at every size -- there is no excess anywhere for merged mass
to have gone into. Its distribution is flat where HR's is steep. And subhalos hold
62% as many particles in SR2 as in HR inside those hosts, so the missing
substructure's mass did not move into surviving subhalos: it is in the smooth host
body.

So the mechanism is **attrition**: the deterministic MSE-trained generator cannot
resolve small bound clumps in a deep potential and leaves that material smooth,
worst where the host is most massive. Not "many small subhalos merged into fewer
big ones" -- the big ones were already there and are, if anything, slightly
under-massive.

**Consequence for a fix.** Because the hosts (above ~200 particles) are already
right and the mass is already in place, the cure is not to help SR2 find more
halos -- it is to make it *fragment* mass it currently prefers to keep smooth,
and to do so in proportion to host mass, which is exactly where it currently
does the least.

## 6. The deficit is deterministic, not sampling variance

Everything above compares one SR2 field to HR. That leaves open whether the
missing subhalos are simply a draw -- SR2 rolling badly on a stochastic map. They
are not. Comparing SR2 **seed 0 to seed 1** (same checkpoint, same LR input,
different noise seed; `particle_identity/set8__base0__base1`, 50-particle floor)
against the HR-vs-SR2 pair (`set8__hr__base0`) on the same metrics:

| | seed0 vs seed1 | HR vs SR2 |
| --- | ---: | ---: |
| median particle displacement | **0.0282** Mpc/h | 0.918 Mpc/h |
| field residual rms | 0.0507 | 1.257 |
| hosts: n_a / n_b | 85440 / 85666 | 97048 / 85440 |
| hosts: median completeness | **0.954** | 0.060 |
| hosts: median Jaccard | 0.906 | 0.042 |
| subhalos: n_a / n_b | **30027 / 30199** | 48522 / 30027 |
| subhalos: fraction matched | 0.876 | 0.629 |

Changing the seed moves a particle **33x less** than SR2 sits from HR, leaves
host membership 95% identical, and produces the *same number of subhalos in the
same places* (30027 vs 30199, 88% matched). The stochastic channel SR2 nominally
has is doing essentially nothing: in practice the model is a deterministic map.

**Consequence.** The deficit is a systematic property of the learned map, not
variance -- no amount of resampling SR2 recovers it, and any fix that only
re-rolls the existing noise input is dead on arrival. This is the measured
version of the "architectural prerequisite" asserted below on theoretical
grounds: the generator has to actually become stochastic, not merely have a noise
input.

> **Caveat, unconfirmed.** This rests on `base:0` and `base:1` being genuine noise
> seeds of the same SR2 checkpoint (the field paths differ only in `seed0`/`seed1`).
> If they are instead the same deterministic model run twice, the table measures
> nothing and this section must be withdrawn. Confirm before building on it.

### The membership metrics are unreliable at subhalo scale

The same pair gives a calibration that invalidates a diagnostic route, so it is
recorded here rather than left to be rediscovered. Two SR2 fields **0.028 Mpc/h
apart** -- for practical purposes the same field -- agree on host membership
(completeness 0.954) but agree on *subhalo* membership only at **0.189**
(purity 0.251, Jaccard 0.113). Rockstar's boundedness cut is sharp enough that
subhalo member sets are unstable to perturbations far smaller than the object.

So a low HR-vs-SR2 subhalo completeness is not by itself evidence that SR2
failed to build the object. Host-level membership metrics are trustworthy
(0.95 when the answer is "same object"); subhalo-level ones are not.

## Enforcing a subhalo budget in SR

The budget field `subhalo_budget` (lambda_i = N_h / N_particles,h, so
sum_{i in h} lambda_i = N_h) is the natural lever, and section 1 says why it is
the right shape: N_h is linear in host mass in HR, so a per-particle constant is
the correct carrier. Enforcing it is harder than conditioning on it, because the
target is discrete (a count), global (per host, spanning tiles), and
non-differentiable (measured by Rockstar), while the generator is smooth,
tile-local, and MSE-trained. Three levels, weakest first.

**Level 1 -- Condition on it.** Feed lambda (with `log_host_mass`,
`host_fraction_per_tile`) as extra generator input channels, broadcast to HR
within the tile. *Necessary* -- a tile-local generator cannot otherwise see the
host size, and section 3 shows host size is precisely what it gets wrong -- but
*not sufficient*: MSE still rewards staying smooth.

**Level 2 -- Soft budget loss (reward fine-tune).** Add a term pushing the
tile's *predicted* substructure toward sum_tile lambda, using a differentiable
proxy for subhalo count (excess high-k power, a smooth peak count on the CIC
density) -- the role of the existing `reward/` proxies (`soft_rockstar`,
`soft_structure`, `phase_space`) and `train_sr2_direct`. **Caveat, from this
project's own history:** soft proxies are gameable (see the "occupation ratio is
gameable" and "tile-overfit proxy exploitation" findings). A network told to
raise high-k power will inject high-k *noise*, not bound subhalos, unless the
proxy is tightly bounded and verified against real Rockstar.

**Level 3 -- lambda as a point-process intensity (hard, by construction).**
lambda is already a Poisson intensity: it sums to N_h over the host. So sample
N_h subhalo *seeds* with location probability proportional to lambda, and have a
generative module realise a bound clump at each seed. The count is then correct
by construction, with no soft penalty to game. This also dissolves the
tile-boundary non-locality: because sum over a tile of lambda equals
`host_fraction_per_tile * N_h`, sampling per tile at rate sum_tile lambda makes
the per-tile seed counts sum to the global N_h automatically -- no tile needs to
know the whole host, only its local lambda sum, which is already in the feature
set.

**Architectural prerequisite.** Levels 2 and 3 need the generator to place
*discrete* structure, which a deterministic MSE model cannot -- averaging is its
optimum, and section 5 is what that optimum looks like in a cluster. That means
the stochastic generator (the diffusion / flow-matching branches), with lambda as
a conditioning signal. **Section 6 measures this rather than assuming it:** SR2's
existing noise input changes the field by 0.028 Mpc/h and reproduces the same
subhalos, so "add stochasticity" means a genuinely different generator, not a
larger noise weight on this one.

**Recommended shape.** Decouple rather than retrain SR2's regression against its
own MSE. SR2's host mass function is already correct above ~200 particles, so
freeze it and add a lambda-conditioned **substructure module**: it samples N_h
seeds per host from the point process and injects the perturbations that collapse
them into subhalos. Verify against real Rockstar on the assembled box -- never
trust the soft count alone.

**Go/no-go check -- run, and it does not work.** The proposed gate was: take HR
subhalos SR2 smoothed away, split the SR2->HR displacement on those particles
into bulk translation vs internal residual (`cosmo_sr.eval.particle_identity.
displacement_stats`), and call a coherent local collapse *addable* and a
large-scale rearrangement *not*. Run over the 20000 subhalo rows of
`particle_identity/set8__hr__base0/pairs.jsonl`, splitting on whether SR2
reproduced the object (`matched` and `set.completeness >= 0.5`) or lost it
(unmatched or `completeness < 0.1`) -- median `residual_rms / a_rvir_mpc_h`:

| a_num_p | reproduced | n | lost | n |
| --- | ---: | ---: | ---: | ---: |
| 50-100 | 5.83 | 18 | 11.85 | 5951 |
| 100-200 | 5.37 | 17 | 9.99 | 3143 |
| 200-500 | 4.30 | 32 | 7.90 | 5340 |
| 500-2000 | 3.87 | 33 | 6.24 | 3594 |
| 2000+ | 2.24 | 134 | 4.04 | 849 |

Objects SR2 **did** reproduce carry residuals 2-6x their own virial radius, so
the criterion ("residual comparable to the object's size -> structure must be
destroyed and reassembled") fires for both arms and separates nothing. The
reference scale from section 6 makes the saturation explicit: two SR2 seeds
0.028 Mpc/h apart give `residual_rms / rvir` = **0.30** for the same subhalo,
against 2.2-11.9 here. Both arms sit far above the "same object" scale, and the
lost/reproduced gap is only ~2x. Compounding it, section 6 shows subhalo
`completeness` is unreliable, so the split itself is noisy -- only 234 of 20000
HR subhalos clear `completeness >= 0.5` at all.

**What answers the question instead.** Section 4, which needs no particle
identity: in the densest tile quintile SR2 holds **0.33** of HR's total objects
(hosts *and* subhalos). The clumps are not misplaced and not relabelled -- they
are absent from the catalog. Treat "SR2 never builds them" as settled and do not
re-derive it from particle identity, which cannot resolve a 0.1 Mpc/h object
across fields that differ by ~0.92 Mpc/h per particle.

## What the earlier version got wrong

1. **"The larger the host fraction SR2 sees, the fewer subhalos it makes."**
   The measurement (corr(count, L) = -0.50) is reproducible, but it is
   collinearity: corr(L, logM) = +0.75, and `partial(L | logM) = -0.07`. Host
   fraction does nothing once mass is held fixed. Only `partial(logM | L)` was
   reported, which cannot separate the two. **Fixed:** the script now reports
   corr(L, logM) and both partials, and the case is made per host (section 1)
   before any tile correlation.
2. **"The hosts are correct; only their interiors are missing."** The supporting
   table started at 1e11.5 and so covered 4% of hosts. Below it SR2/HR is 0.77,
   and per tile in the densest quintile SR2 has 44% of HR's hosts. **Fixed:** the
   mass function runs from 1e9.5, a `num_p` function compares hosts and subhalos
   at equal size, and per-tile host counts are reported alongside subhalo counts.
3. **"SR2 merges many small subhalos into fewer oversized ones (3.7x too
   massive)."** Not supported. Survivorship reproduces the whole effect (ratio
   0.81-1.11 against HR's own top-N), and SR2 is below HR at *every* subhalo size
   inside massive hosts, with 62% of the subhalo mass. **Fixed:** the size table
   carries the top-N control, and the mechanism is restated as attrition.
   (The mean sizes themselves also changed slightly -- 3.74 -> 2.81 in the top
   quartile -- because the mean is now particle-weighted over the quartile's
   subhalos instead of an unweighted average of per-tile means, which
   over-weighted sparse tiles.)
4. **"23 of HR's 24 hosts above 1e14 are present, at matching masses."** True, but
   it was asserted from binned counts; no object was ever matched. **Fixed:** the
   script matches them periodically and reports separations, masses, radii and
   per-object subhalo counts -- which is where the 12582-vs-924 headline comes
   from. (The 24th is present too, at the right mass: SR2 id 69101, log M 14.17
   vs HR 14.20, 0.27 Mpc/h away -- but Rockstar files it as a *subhalo* of a
   neighbour, so it is not in the host list.)

Not changed: the headline totals, the occupancy, the floor sweep, and the overall
direction of the conclusion -- SR2 loses small substructure, worst in dense
cluster regions, and the remedy is a discrete substructure module rather than
more MSE.

## Reproduce

```bash
bash scripts/slurm/submit_lagrangian_host.sh          # build -> join + deficit -> viewer
# or, individually:
python scripts/features/collect_tile_abundance.py --boxes set8       # per-tile deficit (panel 5)
python scripts/features/collect_host_subhalo_tiles.py --boxes set8   # per-host per-tile counts (panel 7)
python scripts/features/analyze_sr2_subhalo_deficit.py --boxes set8  # the tables above
```

All three read committed artifacts only and write JSON next to the features. The
viewer (`docs/lagrangian_host_features.md`) shows the per-tile deficit as panel 5
and, per selected host, its own HR-vs-SR2 subhalo count per tile as panel 7.

**Section 6 is not covered by those.** Its comparison table reads the two
existing summaries directly:

```
$DMSR_REWARD_ROOT/particle_identity/set8__base0__base1/summary.json   # seed0 vs seed1
$DMSR_REWARD_ROOT/particle_identity/set8__hr__base0/summary.json      # HR vs SR2
```

and the retired go/no-go table is a streaming pass over
`set8__hr__base0/pairs.jsonl` (20000 subhalo rows), grouping on `matched` /
`set.completeness` and taking the median of
`disp_all.residual_rms_mpc_h / a_rvir_mpc_h` per `a_num_p` bin. **TODO:** fold
both into `analyze_sr2_subhalo_deficit.py` with a CPU sbatch, so section 6 is
reproducible on the same footing as sections 1-5. Until then the numbers above
are a one-off and should be re-derived before being relied on.
