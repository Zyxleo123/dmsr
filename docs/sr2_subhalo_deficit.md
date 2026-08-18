# SR2 subhalo deficit: what is lost, and what kind of failure it is

**Scope.** A diagnosis of *why* the SR2 super-resolution catalog holds far fewer
subhalos than HR, and a design note for enforcing a subhalo budget. No model is
trained here. Every number is reproduced by
`scripts/features/analyze_sr2_subhalo_deficit.py --boxes set8`, which reads only
the committed `*_tilew.npz` weights and the existing Rockstar catalogs (no halo
finding, no owner arrays, no GPU) and writes `set8_subhalo_deficit.json`.

Context: `docs/lagrangian_host_features.md` builds LR host features on the 64^3
Lagrangian lattice; this document uses them, plus the SR2/HR tile weights, to
localise where SR2 fails.

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

## Three questions, each ruling out a mechanism

### 1. Does SR2 turn big hosts into small ones? No.

If SR2 were fragmenting massive hosts or failing to recognise them, the host
mass function would shift toward low mass. It does not -- SR2 recovers hosts at
90-97% in **every** mass bin:

| host log M | HR | SR2 | SR2/HR |
| --- | ---: | ---: | ---: |
| 11.5-12.0 | 5540 | 4989 | 0.90 |
| 12.0-12.5 | 2079 | 1996 | 0.96 |
| 12.5-13.0 | 764 | 734 | 0.96 |
| 13.0-13.5 | 232 | 224 | 0.97 |
| 13.5-14.0 | 77 | 69 | 0.90 |
| 14.0-15.0 | 24 | 23 | 0.96 |

23 of HR's 24 hosts above 1e14 are present, at matching masses. The hosts are
correct. Only their interiors are missing.

### 2. Which subhalos are lost? The small ones.

Raising a particle floor on the subhalo count lifts the SR2/HR ratio steadily,
and flips the sign of its correlation with local density:

| min particles | SR2/HR ratio | rank corr(LR occupancy, SR2-HR deficit) |
| ---: | ---: | ---: |
| 0 | 0.46 | -0.85 |
| 50 | 0.62 | -0.80 |
| 100 | 0.75 | -0.73 |
| 200 | 0.81 | -0.59 |
| 500 | 0.86 | -0.48 |

The deficit is dominated by small subhalos, and it concentrates where host
material is densest.

### 3. Is SR2 a context-blind local model? No -- it fails *worse* on big hosts.

The natural hypothesis: SR2 sees only a tile-sized fragment of a host, cannot
tell a slice of a giant from a whole dwarf, and so responds to the **local**
host fraction while being blind to the host's **total** mass. That predicts the
per-tile subhalo count should track local fraction and be flat in total mass.

Test: for each tile define `L` = local host fraction (bound particles / 262144)
and `logM` = mass-weighted mean *total* mass of the hosts the tile's material
belongs to, both from HR truth. Then the **partial** correlation of the count
with `logM` *after holding L fixed* asks whether total mass still matters:

| | corr with L | corr with logM | **partial(logM \| L)** |
| --- | ---: | ---: | ---: |
| HR subhalo count | +0.73 | +0.58 | **+0.07** |
| SR2 subhalo count | -0.50 | -0.62 | **-0.43** |

Mean subhalo count per tile, binned by local fraction (rows) x total host mass
(columns):

```
                low mass   mid    high mass
HR   L low  :    145       167     181     rows rise, columns ~flat
     L mid  :    200       195     215
     L high :    240       232     234

SR2  L low  :    100        97      89     columns FALL, worst at L high
     L mid  :    118       102      87
     L high :    115        94      56
```

Two findings:

* **HR is the local model.** At fixed local host material, HR's subhalo count is
  essentially flat in total host mass (partial +0.07). Physically: subhalo count
  scales ~linearly with host material, so a fixed amount of it carries about the
  same number of subhalos regardless of the parent's total mass. *The intuition
  that the honest signal is local host fraction is correct -- it describes HR.*
* **SR2 is not merely local; it is actively suppressed on big-host fragments.**
  A context-blind model would be flat in mass (partial ~0) like HR. SR2's
  partial is **-0.43**: at the same local material, it makes fewer subhalos when
  that material belongs to a bigger host, worst in the densest tiles
  (115 -> 56).

### The mechanism: merging, not misjudgement

Mean subhalo size (member particles) by LR host-occupancy quartile:

| LR occupancy | HR | SR2 | SR2/HR |
| --- | ---: | ---: | ---: |
| 0.00-0.13 | 117 | 179 | 1.53 |
| 0.13-0.27 | 157 | 258 | 1.64 |
| 0.27-0.45 | 196 | 389 | 1.98 |
| 0.45-0.95 | 280 | 1047 | **3.74** |

In the densest regions SR2's subhalos are **3.7x too massive**. SR2 merges what
should be many small subhalos into fewer oversized ones, and does so worst
exactly where the true host is big and the tile is crowded. That is a
**resolution / merging failure** -- the deterministic MSE-trained generator
cannot resolve tightly packed small clumps and smears them into the smooth host
body -- not a **context** failure. It explains every row above: hosts intact
(mass preserved), small subhalos lost (merged), deficit concentrated in dense
big-host fragments (where merging is worst), size inflated there (the lost
subhalos' mass absorbed into the host).

**Consequence for a fix.** Because the hosts are already right and only their
interiors are missing, the cure is not to help SR2 find more halos -- it is to
make it *fragment* mass it currently prefers to keep smooth.

## Enforcing a subhalo budget in SR

The budget field `subhalo_budget` (lambda_i = N_h / N_particles,h, so
sum_{i in h} lambda_i = N_h) is the natural lever. Enforcing it is harder than
conditioning on it, because the target is discrete (a count), global (per host,
spanning tiles), and non-differentiable (measured by Rockstar), while the
generator is smooth, tile-local, and MSE-trained. Three levels, weakest first.

**Level 1 -- Condition on it.** Feed lambda (with `log_host_mass`,
`host_fraction_per_tile`) as extra generator input channels, broadcast to HR
within the tile. *Necessary* -- a tile-local generator cannot otherwise see the
host size -- but *not sufficient*: MSE still rewards staying smooth, and the
diagnosis above shows context is not the missing ingredient.

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
optimum. That means the stochastic generator (the diffusion / flow-matching
branches), with lambda as a conditioning signal.

**Recommended shape.** Decouple rather than retrain SR2's regression against its
own MSE. SR2's host mass function is already correct, so freeze it and add a
lambda-conditioned **substructure module**: it samples N_h seeds per host from
the point process and injects the perturbations that collapse them into
subhalos. Verify against real Rockstar on the assembled box -- never trust the
soft count alone.

**Go/no-go check before building any of this.** Whether a subhalo is even
*addable* by a local displacement perturbation, or whether it needs
globally-consistent tidal context the tile cannot supply. Take HR subhalos SR2
merged, and split the SR2->HR displacement on exactly those particles into bulk
translation vs internal residual (`cosmo_sr.eval.particle_identity.
displacement_stats`). A coherent local collapse is addable; a large-scale
rearrangement is not. Cheap, on the set8 owner arrays already on disk, and it
decides whether Level 3 is physically possible.

## Reproduce

```bash
python scripts/features/collect_tile_abundance.py --boxes set8      # per-tile deficit (panel 5)
python scripts/features/analyze_sr2_subhalo_deficit.py --boxes set8 # the tables above
```

Both read committed artifacts only and write JSON next to the features. The
viewer (`docs/lagrangian_host_features.md`) shows the per-tile deficit as
panel 5.
