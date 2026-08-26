# A substructure module for SR2: two parameterizations

**Scope.** A design note. No model is trained here and nothing below is a
measured result of a training run. It records (a) the mechanism this project now
believes drives the subhalo deficit, (b) one new piece of evidence read directly
out of the SRS checkpoint, and (c) two concrete parameterizations for a fix --
**Option A, host-frame crops** and **Option B, native Lagrangian tiles** -- with
the reasons to prefer B.

Numbers are tagged by provenance: *measured* (read from an artifact on disk),
*derived* (arithmetic from measured constants), or *estimated* (an inference that
still needs a script). Anything untagged in a table caption is derived.

Depends on `docs/sr2_subhalo_deficit.md` for the deficit itself and
`docs/lagrangian_host_features.md` for the conditioning channels.
`docs/host_crop_learnability.md` measures this design against real crops and
**contradicts item 4 of section 2**; read its section 4 before acting on the
high-pass, and its section 6 for two corrections to the text below.

## Constants

*Measured*, from `dmsr_reward/halos/set8__hr__hr/hr_rockstar/rockstar.cfg`:

| | |
| --- | --- |
| `PARTICLE_MASS` | 5.81881e8 Msun/h |
| `BOX_SIZE` | 100 Mpc/h |
| Om / Ol / h | 0.2814 / 0.7186 / 0.697 |
| HR / LR grid | 512^3 / 64^3, upsample 8 |

*Derived* from those: mean density `rho_bar` = 7.809e10 Msun/h per (Mpc/h)^3;
HR lattice spacing 0.1953 Mpc/h; LR cell 1.5625 Mpc/h; **LR particle mass
2.98e11 Msun/h**. The Lagrangian radius of a mass M is
`R_L = (3M / 4 pi rho_bar)^(1/3)`, and `R_vir ~= R_L / 200^(1/3)`.

## 1. What the fix has to attack

### 1.1 The deficit is a contrast failure with no intrinsic mass scale

`sr2_subhalo_deficit.md` establishes that HR holds a flat ~3-4 subhalos per 1e12
Msun/h across five decades of host mass, while SR2 falls from 3.71 at 1e11.75 to
0.20 above 1e14. Subhalo *density* per unit host mass is uniform in truth, so a
mode-counting argument cannot explain the mass dependence: a 1e12 host and a
1e14 cluster both hold roughly one subhalo per LR cell, since abundance and
Lagrangian volume both scale as M_host.

What is *not* scale-free is amplitude. The perturbation needed to make a subhalo
is set by the subhalo: a 50-particle object has `R_L = 0.446` Mpc/h and
`r_vir = 0.076` Mpc/h, independent of its host. The bulk displacement it must
ride on is set by the host: `R_L = 1.45` Mpc/h at 1e12, **6.74** Mpc/h at 1e14.
A network whose error is proportional to local field amplitude therefore has

    SNR ~ (M_sub / M_host)^(1/3)

which is monotonic in host mass and crosses unity near `M_host / M_sub ~ 30`,
i.e. `M_host ~ 1e12` for a 3e10 subhalo. That is the observed threshold. The same
statement in velocity: the host's dispersion scales as `M_host^(1/3)`
(611 km/s at 1e14, 132 km/s at 1e12) while the subhalo's own coherence does not.

**Consequence.** The controlling variable is the mass *ratio*
`mu = M_sub / M_host`, not `M_host`. Any fix must equalize contrast across mu;
nothing else about it matters as much.

### 1.2 The mu collapse -- *estimated*, and the first thing to check

Reading the two `num_p` tables of `sr2_subhalo_deficit.md` together and
estimating a typical host mass per bin (HR subhalos are spread near-uniformly
over host-mass decades, median host ~1e12.5-13):

| source | M_sub | typical M_host | mu | SR2/HR |
| --- | ---: | ---: | ---: | ---: |
| all hosts, 20-50p | 2.2e10 | ~3e12 | 7e-3 | 0.31 |
| hosts >1e14, 500-2k p | 6.2e11 | 1.5e14 | 4e-3 | 0.29 |
| hosts >1e14, >2k p | ~2.5e12 | 1.5e14 | 1.7e-2 | 0.58 |
| all hosts, 200-500p | 2.2e11 | ~3e12 | 7.3e-2 | 0.78 |

At fixed `M_sub`, clusters are 10-15x worse than the box average. At fixed `mu`,
two samples 1.7 dex apart in host mass agree to within 0.02. **Mass ratio
collapses the deficit; host mass does not.**

This is estimated, not measured -- the host masses in rows 1 and 4 are inferred
from the abundance table rather than joined per object. **Confirming it is the
cheapest useful next step:** bin the SR2/HR subhalo ratio on `mu` directly in
`scripts/features/analyze_sr2_subhalo_deficit.py`, one curve per host-mass
decade. If the curves overlap, section 1.1 is settled and the design below is
justified. If they separate, the mechanism is something else and this note
should be revisited before any training.

### 1.3 New evidence: the generator is deterministic by training

*Measured*, read from `external/SRS-map2map/SRmodel/G_z0.pt` (epoch 635).
`AddNoise.std` at `map2map/models/srsgan.py:107` is an `nn.Parameter` initialized
to **exactly zero**; these are its trained values:

| layer | scale | n | mean abs std | max |
| --- | --- | ---: | ---: | ---: |
| `blocks.0.conv.0` | 1x | 512 | 7.7e-4 | 3.5e-3 |
| `blocks.0.conv.4` | 2x | 256 | 1.9e-3 | 6.4e-3 |
| `blocks.1.conv.0` | 2x | 256 | 2.1e-3 | 7.6e-3 |
| `blocks.1.conv.4` | 4x | 128 | 1.8e-3 | 5.8e-3 |
| `blocks.2.conv.0` | 4x | 128 | 1.4e-3 | 6.5e-3 |
| `blocks.2.conv.4` | 8x | 64 | **5.1e-2** | 1.98e-1 |

Adversarial training had 635 epochs to raise these and left five of six near
1e-3. The one live channel is a **single-channel, spatially white** map at the
finest scale -- dither at output resolution, not a mechanism for coherent clumps.

**This retires the caveat on section 6 of `sr2_subhalo_deficit.md`.** That
section's conclusion no longer depends on `base:0` / `base:1` being genuine
seeds: the stochastic pathway is measurably off at the parameter level. Any fix
that only re-rolls this noise is dead.

(Unrelated upstream bug, noted so it is not rediscovered: `AddNoise.forward`
adds the noise twice -- `x = x + noise` then `return x + noise`. A factor of 2,
harmless unless those `std` values are ever fine-tuned.)

## 2. The skeleton both options share

Five commitments that do not depend on the parameterization.

1. **Freeze SR2.** Hosts above ~200 particles are already correct, the box holds
   97% of HR's bound particles, and the 24 clusters above 1e14 match at mass
   ratio 1.03 and 0.20 Mpc/h separation. The module is additive.
2. **Sample, do not regress.** The target is a *conditional distribution*.
   Conditional flow matching on `p(HR | SR2, conditioning)`; no adversary, no
   soft reward. See section 6 for why the alternatives are excluded.
3. **Emit 6 channels** (`d_disp`, `d_vel`) per particle. Velocity is not
   optional: the channel swap (`sr2-halos-are-sub-virial`) showed HR loses 65% of
   its subhalos when given SR2's velocities.
4. **High-pass the output in Lagrangian k.** *(Contradicted at the stated cut --
   `docs/host_crop_learnability.md` §4: 8 h/Mpc passes only wavelengths under
   0.785 Mpc/h, the Lagrangian diameter of a **34-particle** object, so every
   resolvable subhalo -- 7.0 h/Mpc at 50p, 3.6 at the 366p cluster median -- is
   below the cut and the filter forbids assembling substructure rather than
   restricting the module to it. A `1e12` host at 2.16 h/Mpc and a 2000p subhalo
   at 2.06 also show no single k separates the two. The guarantee is worth
   keeping; the instrument has to change, probably to something host-relative.)*
   Low-k Lagrangian modes carry where a
   host goes and how massive it is; high-k modes carry substructure. Filtering
   the output above a cut makes "cannot move or resize a host" a property of the
   parameterization rather than something a loss has to enforce. 8 h/Mpc is a
   natural first choice -- it is where SR2's own displacement `r(k)` falls off.
   *(Now measured and found wrong at this cut: `docs/pilot_steps_2_4.md` §3.5.
   The exact whole-box spectrum puts SR2's `r(k)` through 0.5 at k=2.3 and 0.1 at
   3.7 h/Mpc -- its error is at 2-4, not 8 -- and only ~10% of the residual sits
   above 8. A fixed Lagrangian-k cut also cannot separate a 2000p subhalo (2.06)
   from a 1e12 host (2.16): the distinction is a mass ratio, not a wavenumber.
   The guarantee stands; the instrument must become host-relative. Deciding it is
   now the critical path.)* **Decided: `docs/sr2_moment_constraint.md`** replaces
   this fixed spectral cut with a per-host affine-moment projection -- relational
   (by host footprint at its own `R_L`), no overlap rule, and it leaves the
   compaction mode of open risk 5 available where an 8 h/Mpc cut would not.
5. **Instrument conditional spread from the first checkpoint.** Sample twice on
   identical conditioning; log seed-to-seed subhalo count and positions every
   eval. This is the diagnostic that would have caught section 1.3 in hours.

Evaluation is realization-independent throughout: subhalo mass function per host,
radial distribution in `R_vir` units, and **real Rockstar on the reassembled
box**. Never a paired per-particle or subhalo-membership metric (section 6.2).

## 3. Option A -- host-frame crops

**The unit is one host, in its own frame.**

* **Crop.** A cube around the host's Lagrangian centroid (already available as
  `dq_over_rl` in `lagrangian_host`), side ~2 `R_L(host)`, resampled to a fixed
  96^3 Lagrangian grid.
* **Normalization.** Displacements divided by `R_L(host)`, velocities by
  `v_vir = sqrt(G M_vir / R_vir)`, bulk position and velocity removed.
* **Condition.** SR2's displacement and velocity on that grid (6 ch), plus
  scalars `log M_vir`, `log N_p`, `lambda`.
* **Target.** HR's displacement and velocity on the same grid, same
  normalization.
* **Inference.** Un-normalize by `R_L` and `v_vir`, resample back to native
  sites, add to SR2's arrays.

**What it buys.** Explicit self-similarity in `mu`. On a fixed 96^3 crop a
subhalo of ratio `mu` is always `~96 * mu^(1/3)` sites across -- 21 sites at
mu=1e-2, 9.6 at 1e-3, 4.5 at 1e-4 -- *independent of host mass*. Because a
cluster's resolution floor is `mu_min = 50 m_p / M_vir = 2.9e-4`, its smallest
resolvable subhalo is ~6 sites across, and that number does not degrade for any
host. A cluster and a dwarf become literally the same training example, so 384
clusters are not drowned by 3.4M small hosts.

**What it costs.**

* **Requires SR2 to HR host matching** to define the two frames. Matching is
  validated only for the >1e14 objects; below that it is an unquantified error
  source injected into every training pair.
* **Overlapping crops.** A cluster's Lagrangian patch contains many smaller
  hosts' patches, so crops are not a partition and need an ownership or
  partition-of-unity rule.
* **Resampling waste at the bottom.** A 1e12 host is 15 sites across natively;
  upsampling to 96^3 interpolates a field that holds only 1,719 particles of real
  content.
* Extra plumbing: centroid extraction, resample in and out, scatter back.

Option A is not wrong, and it is the natural fallback if Option B underperforms
specifically on clusters. It is simply more machinery than the problem needs.

## 4. Option B -- native Lagrangian tiles (recommended)

**The unit is SR2's own output tile: 64^3 Lagrangian sites, no crop, no
resample.**

*Derived* tile scales: 262,144 particles, 12.5 Mpc/h on a side, **1.53e14
Msun/h** of mass -- about one cluster's worth. A 1e14 host's Lagrangian patch
(`R_L` = 6.74 Mpc/h) fills 66% of a tile's volume.

* **Condition.** SR2's `disp` and `vel` on the tile (6 ch), amplitude-normalized
  (below), plus the `lagrangian_host` channels where they exist --
  `log_host_mass`, `dq_over_rl`, `host_fraction_per_tile`, `lambda`.
* **Target.** HR's `disp` and `vel` on the **same sites**, same normalization.
* **Inference.** `disp_final = disp_SR2 + d_disp`, element-wise, in place. Then
  `field_to_particles` and Rockstar, exactly as today.

### 4.1 Why native tiles are better

1. **The pairing is exact and free.** SR2 and HR are indexed by the same
   Lagrangian sites. No halo matching anywhere in the training path -- Option A's
   worst dependency simply does not exist.
2. **No resampling.** One site, one particle, at every density, at every host
   mass.
3. **Objects already have host-independent pixel size.** Subhalos are
   Lagrangian-pure (median 1 tile of origin, `subhalos-are-lagrangian-pure`), so
   an N-particle subhalo occupies N sites: 3.7 across at 50 particles, 7.2 at
   366 (the HR median inside clusters), 12.6 at 2000 -- regardless of host. The
   `mu`-normalization Option A works for turns out to be unnecessary; absolute
   mass is the more natural target for a fixed-resolution generator, and
   `lambda` supplies the abundance.
4. **Tiles partition the lattice**, so there is no overlap rule.
5. **Context is sufficient.** Subhalo material is Lagrangian-local, and the
   SR2/HR ratio is flat against distance to the tile face -- there is no boundary
   suppression to design around.
6. **Reuses the existing pipeline**: `tile_cache`, `*_tilew.npz`, the
   conditioning channels, the assembly and Rockstar path.

### 4.2 The one thing Option A bought that must be replaced

Amplitude normalization. In a native tile the displacement values still span the
full dynamic range -- ~6.7 Mpc/h across a cluster patch against ~1 in a void --
so without intervention the cluster dominates the loss and section 1.1's disease
returns unchanged. Geometry was only ever the vehicle; normalize the **values**
pointwise instead:

    s(q)   = smoothed rms |Psi_SR2| over ~3-4 Lagrangian sites
    x_in   = Psi_SR2 / s
    v_in   = V_SR2  / s_v

and weight the flow-matching loss per site by `1 / s^2`. This equalizes the
*per-subhalo* gradient while correctly leaving a cluster's total weight ~100x a
1e12 host's -- it has ~100x more subhalos to build.

**Derive `s` from SR2's own field, not from a host catalog.** The obvious choice
is `R_L(host)` from `lagrangian_host`, and it fails: LR particle mass is 2.98e11,
so LR-Rockstar finds nothing below ~1.5e13 (50 LR particles), while the deficit
is already at 0.51 in the 1e12.5-1e13 bin -- 764 hosts per box holding ~13,700
subhalos, entirely under that floor. A locally-estimated scale has none of that
problem: defined everywhere, continuous across host boundaries, no catalog
dependence, and available at inference by construction. Keep the catalog channels
as *conditioning* (they cover the >1e13 regime where the damage is worst); just
do not let the normalization depend on them.

The transform must be exactly invertible, so `s` is saved with the tile.

### 4.3 Costs to handle

* **Tile imbalance.** ~24 clusters per box means at most ~5% of the 512 tiles
  contain one; most tiles are field and void. Importance-sample tiles by host
  mass content or by per-tile `sum lambda` (already computed) rather than
  uniformly. Skipping this spends most of the compute on tiles SR2 already gets
  right.
* **Memory.** 64^3 x 6 x fp32 = 6.3 MB per tile of data; a 64^3 U-Net's
  activations dominate and need sizing before committing. The
  `gaussian-training-step-cost` note records a 240^3 batch-2 shape OOMing a 48 GB
  A6000, so this is not automatic.

## 5. Side by side

| | A: host-frame crops | B: native tiles |
| --- | --- | --- |
| unit | one host, ~2 R_L, resampled to 96^3 | SR2's own 64^3 tile |
| pairing | needs SR2<->HR host matching | exact, by Lagrangian site |
| resampling | in and out | none |
| overlap rule | required | none (partition) |
| contrast fix | geometric (R_L, v_vir) | pointwise (local scale `s`) |
| object pixel size | fixed in `mu` | fixed in particle count |
| conditioning floor | needs a host catalog | none required |
| plumbing | centroids, resample, scatter | element-wise add |
| clusters vs dwarfs | same example by construction | equalized by loss weight |

Both use the same objective, the same 6-channel output, the same high-pass
guarantee and the same Rockstar gate. **B is recommended**: same guarantees,
roughly half the machinery, and better-conditioned training data. A is the
fallback if B turns out to underperform on clusters specifically.

## 6. Ruled out, and why

### 6.1 Paired per-particle residual regression

Training `Delta = x_HR - x_SR2` under L2 fails twice. First, SR2 and HR are
different realizations at small scale, so the target is dominated by
delete-and-rebuild: the retired go/no-go table in `sr2_subhalo_deficit.md` gives
`residual_rms / r_vir` of 2.2-11.9 on subhalo particles, against **0.30** for two
SR2 fields that are effectively the same field. Second, that target's conditional
mean is empty in the fine modes, so an L2 regressor converges back to SR2. This
is the same trap as the original generator, one level up.

The concern does *not* apply to flow matching, which never computes a difference:
it learns a conditional distribution, and is free to produce substructure
unrelated to SR2's realization. A broad conditional in the fine modes is the
correct answer there, not an error.

(`docs/host_crop_learnability.md` §6.1: the conclusion holds, the stated reason
does not. If the generated variable is `d_disp` then the flow's data *is* the
difference field; what saves it is that L2 converges to `E[Delta | SR2]`, empty
in the fine modes, while flow matching samples the conditional. That section also
notes this design has never said in writing that the flow's `x_1` is the delta
rather than the full field -- only that reading is consistent with section 4's
inference rule and skeleton item 4.)

**The deletion burden is small anyway** -- *derived* from the `num_p` table
inside hosts above 1e14:

| | clumps | particles | mean size |
| --- | ---: | ---: | ---: |
| HR | 12,582 | 4.60e6 | 366 |
| SR2 | ~890 | 2.86e6 | ~3,210 |

SR2's cluster interiors hold ~7% of HR's clump count, and its survivors average
9x larger -- they are the >2k-particle objects it already gets right (ratio
0.58), which the high-pass constraint protects. The work is *fragmenting smooth
material*, not deleting and rebuilding.

### 6.2 Paired subhalo-membership metrics, as loss or as gate

Section 6 of `sr2_subhalo_deficit.md`: two SR2 fields 0.028 Mpc/h apart agree on
host membership at 0.954 but on **subhalo** membership at 0.189. Rockstar's
boundedness cut is unstable to perturbations far smaller than the object. Neither
completeness nor Jaccard is usable at subhalo scale in either role.

### 6.3 Eulerian density grids

An earlier version of this design cropped in Eulerian space and rasterized to a
density + momentum grid. It reintroduces the dynamic-range failure as a
resolution failure. *Derived*, for a 1e14 host with `R_vir` = 1.15 Mpc/h, a
+/-2 `R_vir` crop on a 128^3 grid gives cells of 0.036 Mpc/h. A 50-particle
subhalo is ~4 cells across -- but in the inner cluster (r ~ 0.1 `R_vir`,
rho ~ 3000 rho_bar) the interparticle spacing is 0.0135 Mpc/h, so **~20 particles
land in every cell**. The subhalo is smeared across 2-3 cells that each already
hold 20 particles of smooth host material, and the effect worsens with host mass
because core density does. Lagrangian sampling is exactly uniform by
construction and has none of this.

### 6.4 Density as an output

A density field cannot be corrected back into particles: infinitely many
arrangements produce a given density, and choosing one is the problem itself.
Density, momentum and `sigma_v` are acceptable as *inputs* only. Likewise
`M_vir`, `N_p` and `lambda` are conditioning scalars read from a catalog -- the
module never predicts them.

A related point that also kills grid-sampled displacement: a smooth field
evaluated at particle positions moves every particle in a cell *together*. It can
translate a blob but never fragment one. Two particles sharing an Eulerian cell
have distinct Lagrangian sites, so differential motion is free in the Lagrangian
parameterization. Fragmenting smooth material is the whole task.

### 6.5 Soft substructure rewards

Excluded by this project's own history -- `occupation-ratio-is-gameable`,
`tile-overfit-proxy-exploitation`, and the arm A-D rank-gate failures. A network
told to raise high-k power injects high-k noise. A likelihood objective has
nothing to game; keep any boundedness diagnostic as a **monitor**, never a loss.

## 7. Open risks

1. **Boundedness is a nonlinear functional of the output.** The module can match
   the HR field distribution and still emit clumps Rockstar rejects. Flow
   matching does not optimize for it and (6.5) it must not be added as a soft
   term. This is the largest unknown; gate on real Rockstar.
2. **Conditional collapse** of the flow itself -- the failure that produced this
   whole document. Mitigated only by instrumenting it from the first checkpoint
   (skeleton item 5).
3. **Is `p(HR | SR2)` actually broad at subhalo scale?** The entire "must sample"
   premise assumes it is. If paired HR tiles turn out nearly determined by their
   SR2 tiles, a regressor would do and this design is overbuilt. **Measured on
   set8 (`docs/pilot_steps_2_4.md` §1): BROAD.** No local linear or
   random-feature map from SR2's 11^3 neighbourhood recovers HR's high-pass
   displacement (held-out `R^2` +0.006) while the same pipeline recovers the
   smoothed field (+0.976); copying SR2's own fine structure scores *negative*.
   The run's own consistency gate now passes (§4). Evidence, not proof --
   residual variance bounds `Var(HR|SR2)` from above only -- and the held-out
   split is spatial, not yet cross-realisation (set9 owner array pending). But
   the premise is now measured rather than assumed. Earlier,
   `docs/host_crop_learnability.md` §1.2 ruled out one half of it (SR2 has not
   thrown the information away); §1 now closes the other.
4. **`mu` collapse unconfirmed** (1.2). If it fails, revisit before building.
5. **The residual has a large low-k component the design may forbid.**
   `docs/host_crop_learnability.md` §2: inside a host's footprint HR exceeds SR2
   in local density by +0.28 to +0.74 dex *everywhere*, not only at subhalos --
   SR2's clusters carry the right total mass at too low a concentration. That
   compaction is a ~0.5 h/Mpc mode and skeleton item 4 would filter it out
   entirely. Decide whether it is this module's job or another's.

## 8. Data inventory

*Measured*, by listing `/zfsauton/scratch/yixiz/DMSR`:

| artifact | coverage |
| --- | --- |
| paired HR fields `(6,512,512,512)` | **16 boxes**, `paired_catnorm/hr/set0..15` |
| HR Rockstar catalogs | **16 boxes**, `dmsr_reward/halos/setN__hr__hr` |
| SR2 catalogs | **15 boxes** (`setN__base__base`; set12 absent) |
| particle -> halo owner arrays | **set8, set9 only**, `dmsr_reward/halos_particles` |
| LR fields | 353 boxes, but only 16 are paired with HR |

`FULL_PARTICLE_CHUNKS = 0` in the reward-tree catalogs, so they carry no member
lists; the owner arrays are the membership source and exist for two boxes. **This
does not gate training** -- tile pairs need only the fields, which exist for all
16 boxes. Membership is needed for evaluation and analysis.

*Derived* training-set size, counting objects rather than halos:

| | per box | x16 |
| --- | ---: | ---: |
| tiles | 512 | 8,192 |
| HR subhalos | 100,599 | ~1.61M |
| hosts > 1e14 | 24 | 384 |
| subhalos inside those | 12,582 | ~201k |
| hosts 1e13-1e14 | 309 | 4,944 |

The tempting count is "384 clusters, hopeless". The other one is ~201k cluster
subhalos, because a convolutional module on a tile sees each subhalo-scale patch
as an independent example. **Both are right, for different quantities**
(`docs/host_crop_learnability.md` §6.2): a subhalo is never a training example --
the unit is a host under A and a tile under B, and section 6.2 keeps subhalo
membership out of the loss entirely. The subhalo count bounds the local gradient
a fully-convolutional module receives *if* the mechanism is local and
translation-invariant; it bounds nothing that varies at host scale, for which the
number is 384. They are not iid -- subhalos in one cluster share a host, an
environment and a realization -- so the effective size sits between the two,
nearer 384 for anything host-scale. Section 4.3's importance sampling is a bet on
the local reading.

## 9. Pilot

Ordered so that each step can kill the next.

1. **Confirm the `mu` collapse** (1.2). CPU, existing artifacts, minutes.
2. **Measure the conditional spread** of `p(HR tile | SR2 tile)` at subhalo scale
   on set8 + set9 (7.3). CPU. *Not done and now the critical path* -- step 1's
   sibling work in `docs/host_crop_learnability.md` narrowed 7.3 but did not
   close it, and steps 3-5 all assume this answer.
3. **Re-choose the high-pass** (7.5 and skeleton item 4) on a measured spectrum
   of `Psi_HR - Psi_SR2`, one `rfftn` per crop. Blocks step 5, not step 4. CPU.
4. **Capacity vs incentive.** Fine-tune SR2's last block to reproduce a *single*
   1e14 host's Lagrangian region under plain MSE. If it can, capacity is not the
   limit and the loss is -- which is the expectation. One short GPU job.
5. **Train Option B** on importance-sampled tiles, 6-channel high-pass output,
   local-scale normalization. One GPU job.
6. **Gate on real Rockstar** on the reassembled box. Success criterion: the
   subhalo ratio inside hosts above 1e14 moves from 0.07 toward 0.4+, with the
   host mass function above 200 particles unchanged.

Per this project's conventions every step is an `sbatch` job with a submitter
that only calls `sbatch`; CPU-only steps (1, 2, 3, 6's aggregation) go to the CPU
partition, and figures are redrawn from written JSON rather than recomputed.

## 10. Reproduce

Section 1.3 is a direct read of the checkpoint and takes seconds on a login node:

```python
import torch
sd = torch.load('external/SRS-map2map/SRmodel/G_z0.pt', map_location='cpu')
for k, v in sd['model'].items():
    if k.endswith('.std'):
        print(k, v.numel(), v.abs().mean().item(), v.abs().max().item())
```

Sections 1.1, 6.1 and 6.3 are arithmetic on the constants in the header plus the
tables in `sr2_subhalo_deficit.md`. Section 1.2 is *estimated* and has no script
yet -- see the note there. Section 8 is a directory listing.

Steps 2, 3 and 4 of the pilot are built and submit together (steps 2 and 3 are
one CPU job, because they are the same FFT; step 4 is a separate GPU job and the
two are siblings, not a chain):

```bash
bash scripts/slurm/submit_pilot_steps.sh                    # both
ONLY=spread  bash scripts/slurm/submit_pilot_steps.sh       # steps 2+3 only
ONLY=overfit OH_RUNG=middle_fine bash scripts/slurm/submit_pilot_steps.sh
DRY=1 bash scripts/slurm/submit_pilot_steps.sh
```

| | code | job | result |
| --- | --- | --- | --- |
| 2 + 3 | `src/cosmo_sr/features/cond_spread.py`, `scripts/features/measure_conditional_spread.py` | `cond_spread_cpu.sbatch` | `dmsr_reward/cond_spread/cond_spread_set8_set9.json` |
| 4 | `scripts/features/overfit_host_mse.py` | `overfit_host_mse_gpu.sbatch` | `dmsr_reward/host_overfit/<box>_h<id>_<rung>/summary.json` |

Both write a `verdict` key stating what the numbers do and do not license. Step
2's is deliberately one-sided: residual variance bounds `Var(HR | SR2)` from
**above**, so a high `R^2` would kill "must sample" outright while a low one is
evidence for it and not a proof.

**Step 4 is a capacity probe and only a capacity probe**, and reading it
requires one ratio the step's first draft omitted -- trainable parameters over
target values:

| rung | trainable params | over 1 tile | over 4 tiles |
| --- | ---: | ---: | ---: |
| `proj_noise` | 4,050 | 0.003 | 0.001 |
| `fine` | 335,954 | 0.214 | 0.053 |
| `middle_fine` | 1,663,314 | 1.058 | 0.264 |
| `all_blocks` | 6,972,242 | **4.433** | 1.108 |

Below ~1 the rung cannot memorise the region: it has to fit a shared local
function across every subhalo-scale patch inside it, and a squared loss over
patches whose fine realisation it cannot predict is minimised by *averaging*.
So a blurred result at `fine`/4 tiles is equally consistent with "the rung
cannot express substructure" and with "L2 declined to commit to a realisation",
which are the two things step 4 exists to separate. Only the over-parameterised
row separates them, so the submitter runs the three as a ladder and the verdict
gates its wording on the ratio.

Neither the ladder nor any single MSE run says whether a **regressor** would
work across examples -- whether `E[HR | SR2]` is full or empty in the fine
modes. That is a property of the conditional distribution rather than of one
optimisation, and it is step 2's question.

**TODO.** Fold 1.2 and the section 8 inventory into
`scripts/features/analyze_sr2_subhalo_deficit.py` with a CPU sbatch, so the
`mu`-binned ratio is reproducible on the same footing as that document's
sections 1-5. Step 1 (the `mu` collapse) is still unrun and still gates the
design; steps 2-4 do not depend on it.
