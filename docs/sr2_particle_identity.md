# Are SR2 and HR halos made of the same particles?

**Research question.** When SR2 produces a subhalo that looks perfectly valid,
is it built from the *same mass elements* as the HR subhalo it is supposed to
correspond to? If not, a residual model trained pointwise on \(\mathrm{SR2} \to
\mathrm{HR}\) is being asked to take a self-consistent SR2 object apart and
rebuild an HR one out of different particles — which is not a correction, it is
a demolition, and no pointwise map can do it without destroying the object it
starts from.

## Why this is exactly answerable, not a matching heuristic

SR2 and HR are two displacement fields on the **same Lagrangian lattice**, and
[`field_to_particles`](../src/cosmo_sr/eval/particles.py) labels particle
\((i_x,i_y,i_z)\) with the flat index \((i_x N_g + i_y) N_g + i_z\) in both. So
"particle 917 in SR2" and "particle 917 in HR" are the *same mass element by
construction*. Rockstar preserves 4-byte IDs verbatim, and with
`FULL_PARTICLE_CHUNKS = 1` it prints the member IDs of every halo. Membership
is therefore comparable as **sets of integers** — no nearest-neighbour
matching, no tolerance, nothing to tune.

This is the same fact Experiment 0's tile decomposition rests on; see
[`src/cosmo_sr/reward/tiles.py`](../src/cosmo_sr/reward/tiles.py) for the
column-level description of Rockstar's recursive `.particles` emission.

### Leaf attribution

Rockstar prints each particle once per *ancestor*, so the table is not one row
per particle. The row where the recursion root **is** the binding object
(`assigned_internal_haloid == internal_haloid`) is unique per particle and names
the deepest object that owns it. That gives a per-particle array
`owner[particle_id]`, and its checkable invariant is sharp:

    len(members(o)) == num_p(o)   for every catalog object o

because `num_p` is a halo's *own* particle list. `check_owner_consistency`
enforces it and the job prints the result; if it does not hold, the id sets are
not the objects' particle lists and every number below means something else.

Comparisons at **host** granularity remap ownership to the top-level ancestor
first (`remap_to_roots`). Without that, a host whose material the other box
binds perfectly — into that host's own subhalos — would score as fragmented.

## The three granularities

| Granularity | What it asks | Functions |
|---|---|---|
| **identity** | Do the matched pair hold the same id set? | `set_metrics` → Jaccard, purity, completeness |
| **radius** | For ids they don't share, *how far* are they? Fraction of one object's members within \(r\) of the other's centre, measured in the other box, at \(r = 0.5, 1, 2\,R_{\rm vir}\) and at fixed 0.1–2 Mpc/h | `radius_fractions`, `displacement_stats` |
| **chunk** | Does the disagreement stay inside one generator tile? | `tile_profile` (Lagrangian), `eulerian_chunk_shift` (Eulerian) |

**"Same chunk" has two meanings and only one of them can fail.** A particle's
*Lagrangian* tile is a function of its id alone, so it is identical in SR2 and
HR — that comparison is only meaningful *between objects* (did the two boxes
build this halo out of the same patch of the ICs?). The *Eulerian* chunk a
particle occupies is a function of position and really does change; a residual
that must push a particle out of the spatial region its generator tile covers is
asking for information that tile never saw.

## The measurement the residual question turns on

For an object's members, \(d_i = x_{\rm HR}(i) - x_{\rm SR2}(i)\) is exactly the
correction a pointwise residual must apply. Split it:

* **bulk** \(= \lVert \langle d \rangle \rVert\) — a coherent translation. SR2
  built a valid object in the wrong place; the residual only has to move it.
* **residual scatter** \(= \mathrm{rms}(d_i - \langle d \rangle)\) — a
  reshuffle. If this is comparable to \(R_{\rm vir}\), the residual has to
  disperse the SR2 object and reassemble it.

`coherent_fraction` is the share of the mean-square displacement carried by the
bulk term. Figure 2 plots the two against each other in units of \(R_{\rm vir}\).

## Controls (all on by default — read them before the headline numbers)

1. **Whole-field baseline.** Displacement statistics over a random sample of all
   particles. Halo members must be judged against what a *typical* particle
   does, not against zero.
2. **Random-id null.** A size-matched random id set fixes the null level of
   every overlap metric.
3. **Two matching-free measurements.** For each object,
   `owner_other[members]` says what the other box did with exactly those
   particles: bound them to one object, split them across several, or left them
   unbound. No halo matching is involved, which matters because matching is the
   weakest link in every SR2-vs-HR comparison. The matched-pair numbers are
   reported alongside so the two views can be checked against each other.
4. **Seed-vs-seed control.** Run the same analysis with A and B both SR2 at
   different noise seeds. **If two SR2 seeds disagree about a subhalo's
   particles as much as SR2 and HR do, no residual trained pointwise on
   (SR2 → HR) can fix it**, because the input does not determine the answer.
   This control is the denominator for the headline number: what it does *not*
   explain is what a residual could in principle repair.

## Running it

Stage A (Rockstar with member IDs → per-particle owner array, ~20 min/box/source,
the ~10 GB ASCII table is deleted inside the same job) and Stage B (CPU-only
analysis, no model, no GPU) are separate jobs. Figures redraw from
`metrics.npz` without re-reading the catalogs.

```bash
# everything: 3 sibling Rockstar passes per box, then the main analysis and the
# seed control, both with --dependency=afterok
bash scripts/slurm/submit_particle_identity.sh
PID_BOXES="set8 set9" bash scripts/slurm/submit_particle_identity.sh
DRY=1 bash scripts/slurm/submit_particle_identity.sh     # print, submit nothing

# or by hand
sbatch scripts/slurm/particle_identity_prep_cpu.sbatch BOXES=set8 SOURCES=hr
sbatch scripts/slurm/particle_identity_cpu.sbatch PID_BOX=set8 PID_A=hr PID_B=base:0
sbatch scripts/slurm/particle_identity_cpu.sbatch PID_BOX=set8 PID_STAGE=plot
```

Knobs are `PID_`-prefixed because the cluster login environment already exports
generic names (`SCRATCH`, `OUT`, `STAGE`).

Seed 0 keeps the plain `base` tag so the existing Experiment-0 catalogs under
`halos_particles/set8__base__base/` are reused; further seeds get their own
directory. Runs that predate `--write-assignment` have the tile weights but no
owner array, and only a new halo-finder pass can produce one — the reuse
short-circuit is deliberately defeated when the array is requested and missing.

## Outputs

`$DMSR_REWARD_ROOT/particle_identity/<box>__<A>__<B>/`

| File | Contents |
|---|---|
| `summary.json` | headline medians, both matched-pair and matching-free, plus the owner-consistency reports and the field baseline |
| `pairs.jsonl` | one row per analysed object |
| `metrics.npz` | flat arrays; the plot stage reads only this |
| `figures/fig1_identity.png` | overlap distributions against the random-id null |
| `figures/fig2_translation_vs_reshuffle.png` | bulk shift vs residual scatter, both in \(R_{\rm vir}\) |
| `figures/fig3_radius.png` | recovery as the radius is relaxed, both directions |
| `figures/fig4_chunk.png` | Eulerian chunk crossings; Lagrangian tile-profile overlap |
| `figures/fig5_fate_vs_mass.png` | matching-free fate of A's particles vs mass |

## How to read the result

* **High Jaccard, high `coherent_fraction`** — SR2 assembles the right mass
  elements and misplaces them as a group. A residual has a well-posed job.
* **High Jaccard, low `coherent_fraction`** — right particles, internally
  scrambled. Learnable in principle, but the correction is not a translation.
* **Low Jaccard, but `radius_a_in_b` high** — the HR particles *are* sitting
  there in SR2; SR2 simply did not bind them. This is a halo-finder/boundedness
  story as much as a field story, and it is the case where the residual is being
  asked to move particles that are already in the right place.
* **Low Jaccard and low `radius_a_in_b`** — the two boxes built this object out
  of different material. Check the seed control before concluding anything: if
  seed-to-seed looks the same, the pointwise residual target is not identifiable
  and the fix belongs in the loss (or in what is conditioned on), not in more
  training.

Bootstrap over **boxes**, never over crops or seeds — the same rule as the rest
of the SR2 study.
