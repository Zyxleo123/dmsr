# LR-Rockstar host features on the Lagrangian lattice

**Scope.** Feature construction, validation, and visualization only. Nothing
here is connected to SR2, diffusion, or flow matching, and no model is trained.
The deliverable is a cached `(64^3)` feature set per box, a test suite that
pins the mapping, and a self-contained page for looking at it.

## Why this is exact

A halo found in the **LR** box is available before any super-resolution runs, so
it can condition the generator. The only question is how to move it from a
catalog row onto the lattice the generator indexes, and here that is a lookup
rather than a match:

* `cosmo_sr.eval.particles.field_to_particles` writes GADGET2 ids as
  `arange(Ng**3)`, the flat C-order index of the Lagrangian lattice site;
* Rockstar preserves 4-byte ids verbatim, and with `FULL_PARTICLE_CHUNKS = 1`
  prints the member ids of every halo it outputs;
* `cosmo_sr.eval.particle_identity.stream_owner_assignment` already inverts that
  member table into `owner[particle_id]`.

So "which LR site does this bound particle come from" has an exact answer. Every
identity below is checked numerically in `tests/features/`, not asserted.

## Coordinate convention

Site `(a, b, c)` of the `ng_lr**3` lattice has

    particle id  = (a * ng_lr + b) * ng_lr + c            # C-order flat index
    position     = ((a, b, c) + 0.5) * boxsize / ng_lr    # Mpc/h, cell centres

which is the same `q` that `field_to_particles` adds the displacement to. Every
volume is stored so that `vol.reshape(-1)[pid]` is the value at particle `pid`.

Defaults: `ng_lr = 64`, `ng_hr = 512` (upsample **8**), `tile_hr = 64`, i.e. one
SR2 output tile (`nsplit = 8`), `boxsize = 100 Mpc/h`, LR cell `1.5625 Mpc/h`.
Because the upsample factor divides the tile size, a tile is exactly `8^3` LR
sites and every HR child of an LR site lies in its parent's tile —
`test_lr_site_and_its_hr_children_share_a_tile` checks the two `TileGrid`s agree.

## Channels

All at `(64, 64, 64)`; `dq_over_rl` is `(3, 64, 64, 64)`. `H` is the number of
hosts that own at least one LR site.

| channel | shape | definition |
| --- | --- | --- |
| `host_member` | `(64,)*3` | 1.0 where the site's particle is bound to a host, else 0.0 |
| `log_host_mass` | `(64,)*3` | `log10(Mvir / (Msun/h))` of the owning **host** |
| `dq_over_rl` | `(3, 64,)*3` | periodic Lagrangian offset from the host's Lagrangian centre, divided by `R_L` |
| `host_fraction_per_tile` | `(64,)*3` and `(H, 512)` | fraction of the host's members in a tile; per-site value is `f[host(i), tile(i)]` |
| `subhalo_budget` (optional) | `(64,)*3` | `lambda_i = N_h / N_particles,h` for members of host `h` |

Two definitions that are choices, and why:

* **Host, not leaf.** Rockstar attributes a satellite's particles to the
  satellite. `remap_to_roots` lifts that to the top-level object before anything
  else, because the conditioning question is about the region the host governs;
  leaving a satellite's mass under the satellite would make a rich host look
  like it lost the substructure it actually holds.
* **Lagrangian radius `R_L`.** The volume-equivalent sphere for the host's
  member count at mean lattice density,
  `R_L = (3 N V_cell / 4 pi)^(1/3)`. It is defined for every host including a
  one-site one, monotone in particle count, and independent of the host's shape.
  The shape information is not thrown away: `rms |dq|` is reported alongside it
  in the host table.
* **Periodic centre.** The circular mean of the phase angles per axis, not the
  arithmetic mean — a host straddling the box edge would otherwise be centred in
  the middle of the box and its offsets would span half the volume.

`N_h` is an **input**. The default counts the host's catalog descendants, which
makes the budget self-consistent with the catalog it came from; that is a
placeholder, not a target. Pass `n_sub_per_host={host_id: N}` for an HR or
predicted count.

Rockstar host ids live in `HostTable.host_id` as **metadata only**. An id is a
nominal label with no physics in its numeric value, so it is never a channel.

## Storage and the HR broadcast

Features are stored **only at `64^3`** (~5 MB for the whole box). A dense
multi-channel `512^3` array is never materialised. The HR view of a tile is
produced on demand:

```python
feat = LagrangianHostFeatures.from_npz(path)
feat.stack_lr()      # (C, 64, 64, 64)   whole box, LR
feat.tile_lr(37)     # (C,  8,  8,  8)   one tile's LR crop
feat.tile_hr(37)     # (C, 64, 64, 64)   the same tile broadcast to HR
```

`tile_hr` is a nearest-neighbour repeat by `upsample` on each axis: every HR
child takes its parent LR site's value, the only broadcast consistent with a
feature defined per LR site.

## Missing data

Most of the volume is not in a halo. A site is unowned when the particle is
genuinely unbound *or* bound only to a clump below `MIN_HALO_OUTPUT_SIZE`; both
are represented identically — `host_index = -1`, `host_member = 0`, and exact
zeros in every other channel.

**Zero is a legitimate value** of `dq_over_rl` (the host centre), of
`subhalo_budget` (a host with no budget), and of `log_host_mass` only in the
absent case. A consumer must therefore read `host_member` alongside the other
channels rather than treating 0 as "absent".

At the LR particle mass (`2.98e11 Msun/h`) the 20-particle output floor is a
floor on `num_p`, **not** on `Mvir`: Rockstar's spherical-overdensity `Mvir` runs
below `num_p * m_p` (measured on set8: median ratio 0.65, and 0.73 in the frozen
HR catalog, so this is not an LR artifact). The measured set8 host range is
`10^11.78 .. 10^14.84 Msun/h`. Massive hosts only, but the low end reaches
~`6e11`, not the ~`6e12` a naive `20 * m_p` would suggest.

## Halo-finder config

`configs/sr2_baseline/rockstar_lr_particles.cfg` — **not** the frozen HR config,
and it must never produce an HR catalog. It differs from
`configs/sr2_baseline/rockstar.cfg` in two lines: `FULL_PARTICLE_CHUNKS = 1`
(output only, needed for member ids) and `FORCE_RES = 0.04` (the LR lattice is
8x coarser, so the frozen `0.005` would claim a resolution the LR simulation
does not have). The `.particles` table is ~20 MB at `64^3` — three orders of
magnitude smaller than the HR one — so it is kept as the rebuild cache instead
of being streamed and deleted.

## Running it

Everything is submitted; nothing runs on the login node.

```bash
# build (LR Rockstar + features + tests) then render the page, chained afterok
bash scripts/slurm/submit_lagrangian_host.sh                  # BOXES=set8
BOXES=set8,set9 bash scripts/slurm/submit_lagrangian_host.sh

# re-render only (pure redraw from the cached npz, no Rockstar)
RENDER_ONLY=1 N_HOSTS=40 bash scripts/slurm/submit_lagrangian_host.sh

# skip a stage: SUBTILES=0 (panel 7's join), DEFICIT=0 (the deficit tables)
SUBTILES=0 DEFICIT=0 bash scripts/slurm/submit_lagrangian_host.sh
```

The submitter chains build → (per-host subhalo join ‖ deficit tables) → render.
The join reads the two 512 MB `<box>_<tag>_owner.npy` arrays, so it is its own
CPU job rather than work inside the render job; both extra stages gate themselves
and exit 0 when their inputs are absent, so a missing owner array costs a panel,
not the page.

Outputs land in `$DMSR_REWARD_ROOT/lagrangian_host/<box>/`:

| file | what |
| --- | --- |
| `<box>_lagrangian_host.npz` | the `64^3` channels + host table (the redraw source) |
| `<box>_lagrangian_host.json` | normalisation report and provenance |
| `<box>_lr_rockstar/` | LR catalog + member-particle table |
| `<box>_host_subhalo_tiles.json` | per-host, per-tile HR/SR2 subhalo counts (panel 7) |
| `<box>_subhalo_centres.npz` | Lagrangian centre, size and host of every subhalo (panel 1's overlay) |
| `<box>_subhalo_deficit.json` | the deficit tables of `docs/sr2_subhalo_deficit.md` |
| `lagrangian_host_<box>.html` | the viewer — `scp` it off and open it |

The underlying commands, if you want them directly:

```bash
python scripts/features/build_lagrangian_host.py --boxes set8
python scripts/features/collect_tile_abundance.py --boxes set8        # panel 5
python scripts/features/collect_host_subhalo_tiles.py --boxes set8    # panel 7
python scripts/features/render_lagrangian_host_app.py --boxes set8 --n-hosts 24
```

## Validation

`pytest tests/features -q` (27 tests, also run inside the build job so a broken
build is reported there rather than in the viewer).

`tests/features/test_lagrangian_host.py` pins the mapping:

* every LR site has exactly `upsample^3` distinct HR children, and over a whole
  tile those children partition the HR block with no gaps or overlaps;
* an LR site and all of its HR children carry the same tile id under the two
  `TileGrid`s — the crop/broadcast agrees with the existing SR2 tile coordinates;
* `log_host_mass` is constant across a host's particles, and a satellite's
  particles carry the **host's** mass;
* periodic offsets match an independent recomputation from the lattice, and an
  edge-wrapping host stays compact instead of spanning half the box;
* `sum_t f[h, t] == 1` for every host, and the per-site channel is the site's
  own tile's entry;
* `sum_{i in h} lambda_i == N_h`, uniformly over the host's particles;
* unowned sites are exactly zero in every channel with `host_index == -1`;
* npz round-trip preserves every array and the grid.

`tests/features/test_host_app_payload.py` pins what the page draws — sampled
coordinates really belong to their host, the palette marks exactly the selected
host's sites, quantised volumes reserve 0 for "no host" and round-trip within
one step, and the bar chart's per-tile allocation equals `N_h * f[h,t]`. These
are the errors a screenshot cannot catch: the picture still looks like a halo.

## The viewer

`lagrangian_host_<box>.html` is one self-contained page — every array and
coordinate is embedded, so it needs no server, no port on the login node, and no
new dependency in the `pjm` env. This follows the pattern already used by
`scripts/reward/render_overdensity_html.py`; Streamlit was not used because it
would need a long-lived process on a machine that kills inline compute.

Controls: host, tile, channel, LR slice index, and whether neighbouring hosts
appear in 3D. Eight panels:

1. **LR slice → broadcast HR tile** — one x slice of the LR lattice with tile
   borders and the selected tile outlined, next to that tile broadcast to HR
   with the LR cell borders visible inside it. Both carry an optional overlay of
   **subhalo centres**, blue for HR and red for SR2, sized by member count and
   toggled by the *Subhalo centres* control (HR + SR2 / HR / SR2 / off) with a
   scope of all hosts or the selected host only. Since the panel is Lagrangian, a
   subhalo is placed at the circular mean of the lattice sites its own member
   particles came from — where its material started — not at its Eulerian
   Rockstar centre, and only subhalos whose centre falls in the plane on screen
   are drawn, so the slider walks the slab through the box. Blue circles with no
   red neighbour are substructure SR2 did not build; on set8 a typical plane
   holds ~1570 HR centres against ~730 SR2 ones.
2. **Where this tile's material ends up** — the same 512 LR particles at their
   *displaced* positions, drag to rotate, with the undisplaced tile as a
   wireframe so the bulk shift is visible. Coloured by the mass of the halo each
   particle lands in, or by `|displacement|`. The only Eulerian panel.
3. **Lagrangian 3D view** — the selected host's sites in LR cell coordinates,
   centred on its periodic Lagrangian centre, drag to rotate, with wireframe
   boxes for the SR2 tiles it intersects (the selected one highlighted).
   Sampling is capped at `--n-sample` (default 2500) per host so the scatter
   stays responsive; the full count is still reported in the caption and table.
   `log_host_mass` and `subhalo_budget` are constant within one host by
   construction, so they only vary here with neighbours shown — `|dq|/R_L` is
   the channel that varies inside a host.
4. **One host across tiles** — the whole `8x8x8` tile lattice as 8 planes, with
   each intersected tile shaded and labelled by its share of the host.
5. **SR2 subhalo deficit per tile** — the same lattice shaded by
   `(N_SR2 - N_HR) / N_HR` over *all* subhalos in the tile, from
   `<box>_tile_abundance.json`.
6. **Fraction and budget per tile** — paired bars for `f[h,t]` (sums to 1) and
   the budget allocated to the tile, summed from the per-particle `lambda`
   (sums to `N_h`).
7. **Subhalos of *this host* per tile, HR vs SR2** — paired bars of how many of
   the selected host's own subhalos each tile produced, on each side, from
   `<box>_host_subhalo_tiles.json`. Where panel 5 counts everything in a tile,
   this counts only the selected host's substructure: the LR host is matched to
   an HR and an SR2 host by Lagrangian material (the object binding most of its
   particles, via `owner[particle_id]` — not by position), and only objects whose
   top-level ancestor *is* that match are counted, so a neighbour's satellite
   sharing the tile is excluded. A subhalo is split fractionally over the tiles
   its material came from, so the bars sum to the matched host's full subhalo
   count. The tile dropdown carries the same two numbers, and the table adds the
   match provenance — matched ids, masses, how much of the LR footprint the match
   binds, and `share_of_match`, which flags the case where one HR host swallowed
   several LR structures. See `docs/sr2_subhalo_deficit.md` for what the numbers
   mean.
8. **Host record and normalisation checks** — host mass, LR particle count,
   `R_L` in Mpc/h and LR cells, rms `|dq|`, Lagrangian centre, number of
   intersected tiles, `N_h`, `lambda_i`, and both normalisation sums with a
   pass/fail mark, plus the box-level report.

Panels 2, 5 and 7 each need an optional input (the LR field, the tile-abundance
JSON, the per-host subhalo JSON), as does panel 1's overlay (the subhalo-centres
npz, written by the same job as panel 7's JSON). Any of them missing disables
just that feature, with the reason printed on it — the page still renders.

## Measured on set8 (jobs 32641 / 32642)

LR Rockstar took **0.9 s** on the 262144-particle box.

| quantity | value |
| --- | --- |
| catalog | 1028 objects — 832 hosts, 196 subhalos |
| host mass range | `10^11.78 .. 10^14.84 Msun/h` |
| LR sites bound to a host | 81480 / 262144 (**31.1%**) |
| tiles per host | median 3, mean 3.66, max 23 |
| hosts spanning >1 tile | **738 / 832 (88.7%)** |
| `max abs(sum_t f[h,t] - 1)` | `4.5e-08` |
| `max abs(sum_i lambda_i - N_h)` | `2.5e-07` |
| total budget | 196.0000 = the catalog's 196 subhalos |
| `\|dq\|/R_L` inside hosts | median 0.84, p90 1.24, max 5.52 |
| feature stack | 7.3 MB at `64^3`; a dense HR stack would be **3758 MB** |

Independent cross-checks against the catalog, after the build:

* `n_particles` equals `num_p(host) + sum num_p(transitive descendants)` for
  **100.0%** of the 832 hosts, max deviation 0. (Counting only *direct*
  children instead misses sub-subhalos and disagrees for 20 hosts — the naive
  check is wrong, not the build.)
* Every channel is exactly 0 off the `host_member` mask: 0 nonzero entries in
  `log_host_mass`, `dq_over_rl` and `subhalo_budget` on unowned sites.
* The most massive LR host is `10^14.84` against the frozen HR catalog's
  `10^14.81` — 0.03 dex apart, so the LR box is finding the same big objects.

**The headline number for the SR2 question is 88.7%.** Hosts routinely straddle
tile boundaries — the median host touches 3 of the 512 tiles and the largest
touches 23 — so a per-tile conditioning signal that assumed one host per tile
would be wrong for the large majority of hosts. That is what
`host_fraction_per_tile` and the budget split exist to represent.

## Not done here, on purpose

No SR2 wiring, no diffusion or flow-matching conditioning, no training. `N_h`
stays a caller input rather than being predicted, and no claim is made that
these channels help — that is the next experiment, not this one.
