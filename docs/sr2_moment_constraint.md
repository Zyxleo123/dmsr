# The moment constraint: the guarantee without a spectral cut

**Scope.** A design note. No model is trained here and nothing below is a measured
result of a training run. It specifies the operator that replaces skeleton item 4
of `docs/sr2_substructure_module.md` -- the high-pass filter -- with a
**per-host affine-moment projection** of the module's added field. It closes the
"decision owed" on the critical path of `docs/pilot_steps_2_4.md` sections 3.5 and
6.2, and it is the last item feeding a step-5 training run.

Numbers are tagged *measured* (read from an artifact on disk), *derived*
(arithmetic on measured constants), or *design* (a proposed rule, not yet run).

Depends on `docs/sr2_substructure_module.md` for the module and its skeleton,
`docs/pilot_steps_2_4.md` §3.5 for why a fixed spectral cut fails, and
`docs/lagrangian_host_features.md` for the conditioning channels this operator
reads (`host_index`, `dq_over_rl`, the host table's `R_L`).

## 1. The guarantee, restated relationally

Skeleton item 4 wants one thing: **the module may add substructure but may not
move or resize a host SR2 already got right.** Hosts above ~200 particles match
HR at mass ratio 1.03 and 0.20 Mpc/h separation (`sr2_substructure_module.md`
§2.1); the module must not disturb them.

The high-pass tried to make this a property of the parameterization by filtering
the output above a fixed Lagrangian `k`. `docs/pilot_steps_2_4.md` §3.5 measured
that it cannot: a 2000-particle subhalo (2.06 h/Mpc) and a 1e12 host (2.16) sit
at the same wavenumber, so no `k` separates "host bulk" from "substructure." The
distinction is **relational -- a mass ratio -- not spectral.** Any field-local
operator (a single-`k` filter, a threshold, a mask) fails the same way.

The fix is to stop asking a wavenumber to encode a relation, and constrain the
relation directly. "Move or resize a host" is not a band of the field; it is the
**coherent affine motion of one host's footprint**. That is a handful of scalars
per host, and they can be removed exactly.

## 2. The affine decomposition

Let the module emit a residual displacement field `d(q)` on the HR Lagrangian
lattice, so `Psi_final(q) = Psi_SR2(q) + d(q)` (and identically for the three
velocity channels). Fix one top-level host `h` with footprint `Omega_h` = the set
of HR sites whose `host_index` maps to `h`. Write the centred, scaled coordinate
of site `i`

    xi_i = (q_i - q_c(h)) / R_L(h)              # dimensionless, 3-vector

where `q_c(h)` is the host's periodic Lagrangian centre and `R_L(h)` its
Lagrangian radius -- both already in the host table, and `xi` is exactly the HR
evaluation of the `dq_over_rl` channel (see §6 for why it is recomputed at HR
rather than broadcast).

Over `Omega_h`, split `d` into its best-fit affine part plus a remainder:

    d(q_i) = T_h + M_h xi_i + d_perp(q_i),       T_h in R^3,  M_h in R^{3x3}

`(T_h, M_h)` are 12 numbers per host. They *are* the six rigid modes plus dilation
and shear:

| moments | dof | what they move |
| --- | ---: | --- |
| `T_h` | 3 | **translation** -- the host's centre of mass |
| `tr(M_h)` | 1 | **dilation** -- uniform resize (breathing) |
| antisym `M_h` | 3 | **rotation** -- net spin |
| sym-traceless `M_h` | 5 | **shear** -- coherent tidal reshaping |

`d_perp` is everything the affine model cannot express: the anharmonic,
site-to-site variation that fragments smooth material into subhalos. **That is the
part the module exists to add** (`sr2_substructure_module.md` §6.1: "the work is
fragmenting smooth material"). Removing `(T_h, M_h)` and keeping `d_perp` is the
guarantee, stated exactly.

## 3. Which moments to constrain

Constrain the **full affine part (all 12)** per host, on both the displacement and
the velocity residual (24 dof/host total). Reasons, in order:

1. It is the complete "leave the host's bulk as SR2 had it" statement. SR2 already
   has the host's translation, size, spin and tidal deformation right; forbidding
   `d` from touching any of them asserts exactly that and nothing more.
2. It is the **simplest** projector. Because each of the three output components
   gets its own row of `(T_h, M_h)`, the full-affine projection is independent per
   component against a shared 4-column basis `[1, xi^1, xi^2, xi^3]` (§6). Any
   proper subset (translation-only; translation+dilation) *couples* components --
   dilation is the single scalar `d = lambda xi` shared across all three -- and is
   more code for a weaker guarantee.
3. It leaves the one low-k mode the module may legitimately need. `d = lambda xi`
   with constant `lambda` is uniform dilation and is removed; but the measured
   compaction of SR2's clusters -- right total mass, too low a concentration,
   +0.28 to +0.74 dex in local density (`docs/host_crop_learnability.md` §2, open
   risk 5) -- is a *radial-profile* contraction, `d ~ g(|xi|) xi` with the inner
   region moving more than the outer. That is **nonlinear in `xi`, so it survives
   the affine projection.** The 8 h/Mpc high-pass would have filtered it out
   entirely; the moment constraint keeps it available to this module. Whether it
   is this module's job stays a separate decision (open risk 5), but the
   instrument no longer forecloses it.

Velocity uses the identical basis on the velocity residual: forbidding `T` there
is "no net momentum kick," forbidding `M` is "no coherent bulk velocity gradient."

## 4. Why this is the right instrument

Point by point against the high-pass it replaces:

* **Relational, not spectral.** It removes host motion by *host footprint*, so a
  2000p subhalo and a 1e12 host are handled by their own footprints at their own
  `R_L` -- the mass ratio is built into the coordinate `xi`, not asked of a `k`.
  §3.5's overlapping-band failure cannot occur.
* **No overlap rule.** `host_index` is the **top-level** host of each site
  (`remap_to_roots`, `lagrangian_host_features.md`), so bound sites *partition* by
  host. Each projection is independent; a cluster's footprint and a satellite
  inside it are one footprint, not two overlapping ones. The unbound 68.9% of
  sites (`lagrangian_host_features.md`, set8) are in no footprint and are left
  **entirely free** -- exactly the field and void where the module should build
  without constraint.
* **Catalog-legitimate.** The footprints come from the **LR** Rockstar catalog,
  which is available before super-resolution runs and is a valid input
  (`lagrangian_host_features.md`, "Why this is exact"). This is the one place the
  catalog *should* drive the operator: the constraint protects catalog objects, so
  being defined only where a catalog object exists is correct, not a limitation.
  This is deliberately opposite to the local scale `s` of `sr2_substructure_module`
  §4.2, which must be catalog-free because it acts everywhere; the two divide the
  labour -- `s` equalizes contrast on every site, the moment constraint guards the
  resolved hosts.
* **Not a soft reward.** It is a hard linear projection, not a term added to the
  loss, so §6.5's objection (a network told to raise high-k power injects high-k
  noise; anything gameable will be gamed) does not apply -- there is nothing to
  optimize against, the forbidden modes are simply absent from the output space.

## 5. How it enters training and inference

The constraint is a single linear, idempotent projector on the whole field,

    Pi = block-diag over hosts of (I - P_h) on Omega_h,  identity off the union of Omega_h

where `P_h = Phi_h (Phi_h^T Phi_h)^{-1} Phi_h^T` projects onto the affine
subspace `span[1, xi^1, xi^2, xi^3]` of host `h`. `Pi^2 = Pi` (each block is a
projector and the blocks are disjoint), so range(`Pi`) is the linear subspace of
fields that move no host affinely.

Flow matching is then run **entirely inside range(`Pi`)** -- the clean way to make
a linear constraint a property of the parameterization:

1. **Target.** Project the residual data: `x_1 <- Pi (Psi_HR - Psi_SR2)`. This
   asserts SR2 keeps each host's affine motion; only the anharmonic remainder is
   learned. (Consistent with `sr2_substructure_module.md` §6.1's note that the
   flow's `x_1` is the delta field.)
2. **Base + field.** Draw the base sample from `Pi`-projected noise and apply `Pi`
   to the learned velocity field `v_theta` at each ODE step. Since `Pi` is linear
   and both endpoints and the field lie in range(`Pi`), every point of every
   trajectory does -- no host is moved at any integration time, not merely at the
   endpoint.
3. **Inference.** `Psi_final = Psi_SR2 + Pi(sample)`, element-wise, then
   `field_to_particles` and Rockstar, exactly as `sr2_substructure_module.md` §4.

Because `Pi` is applied identically to the target and to the output, its
non-invertibility is not a problem: the affine part of the residual is never asked
of the module and never scored against it.

### 5.1 Whole-box or per-tile? Resolved: whole-box, at two points only

`Pi` is defined per host over the host's *full* footprint, and 88.7% of hosts
straddle more than one SR2 tile (`docs/lagrangian_host_features.md`). The module
trains per tile, so the naive reading -- "apply `Pi` inside the training step" --
would fit each host's affine part on a *partial* footprint, a different and
ill-conditioned operator (a thin cluster slice in a boundary tile can have fewer
than four sites). That is the wrong instrument for the same reason the fixed
spectral cut was.

It is also unnecessary. **The guarantee constrains only the emitted `d`**, because
`d` is the *only* thing added to `Psi_SR2`; the interior of the flow's trajectory
is never used. So `Pi` need not touch the per-tile training step or the ODE
integration at all. It enters at exactly two whole-box points, both of which have
the whole box in hand:

1. **Target precompute (train), once per box.** Generate `Psi_SR2` from the LR
   field (`load_controlled_generator`, GPU), load `Psi_HR` (`load_hr`), build
   `Pi` from `<box>_lagrangian_host.npz` (`moment_constraint.from_features`), and
   cache `x_1 = Pi (Psi_HR - Psi_SR2)`. Training then samples tiles from the
   cached, already-projected target -- the per-tile loss sees a target that is
   globally in range(`Pi`), with no per-tile projection anywhere.

2. **Final projection (inference), once per sample.** After the flow emits its
   sample and the tiles are assembled into a whole-box `d`, apply `Pi` once:
   `Psi_final = Psi_SR2 + Pi(d)`. This makes `d in range(Pi)` *exactly*,
   regardless of how well the network learned to stay there, so the guarantee is
   airtight at the endpoint -- which is the only place it must hold.

Training on projected targets teaches the network to produce range(`Pi`) fields;
the single final projection makes it exact. Projecting the base noise or the ODE
trajectory (spec's earlier "every point of every trajectory") is optional
elegance, not required for the guarantee, and is dropped -- it would reintroduce
the per-tile-footprint problem for no benefit. The per-tile training loop is left
untouched, and no small-slice guard is needed.

This makes the target precompute a GPU job (it generates `Psi_SR2`) followed by a
CPU projection, and the inference hook a single `projector.apply` on the assembled
`d`. Both use the exact whole-box operator; neither is on the login node.

## 6. Implementation

The projector is never materialized densely -- a cluster footprint is ~10^5 HR
sites and `P_h` would be `|Omega_h|^2`. Only the 4x4 normal equations are formed:

    for host h:
        Phi_h  = [1, xi^1, xi^2, xi^3]  over Omega_h        # |Omega_h| x 4
        G_h    = (Phi_h^T Phi_h)^{-1}                        # 4 x 4, precomputed
        # applying (I - P_h) to a component d (a length-|Omega_h| vector):
        beta   = G_h @ (Phi_h^T d)                           # 4-vector
        d_perp = d - Phi_h @ beta

This is `O(|Omega_h|)` work and `O(1)` storage per host: a gather over the
footprint, a 4x4 solve, a scatter. Precompute `G_h` and the site index list per
host on CPU from `<box>_lagrangian_host.npz`; apply the gather-solve-scatter each
training step on GPU.

Two resolution points, because the features are stored at LR and the module acts
at HR:

* **Membership** `Omega_h^HR` is the `tile_hr` nearest-neighbour broadcast of
  `host_index == h` -- every HR child of a bound LR site inherits the host, which
  is right because the LR site's Lagrangian patch *is* the host's material at LR
  resolution.
* **Coordinate** `xi_i` is recomputed at HR from the host's `q_c` and `R_L`, **not**
  broadcast from `dq_over_rl`. The broadcast would give all 8^3 HR children of an
  LR site one identical `xi`, collapsing the gradient basis; the affine part needs
  the true HR-site offset. `q_c` and `R_L` are per-host scalars in the table, so
  the exact HR `xi` costs one subtraction per site.

Suggested code seam: `src/cosmo_sr/features/moment_constraint.py` building `Pi`
from a `LagrangianHostFeatures`, with the operator applied in the step-5 training
loop and at inference. Precompute per box alongside the existing
`<box>_lagrangian_host.npz`.

## 7. Verification

Field-only, no halo finder, all pinnable in `tests/features/`:

1. **Idempotence.** `Pi(Pi x) == Pi x` to machine precision on a random field.
2. **Kills exactly the forbidden modes.** For each host, inject a pure translation
   `d = T`, a pure dilation `d = lambda xi`, a pure rotation and a pure shear over
   `Omega_h`; each maps to ~0 there. A field constant off all footprints is
   unchanged.
3. **Preserves substructure.** A synthetic anharmonic bump inside a footprint
   (a few-site clump) survives `Pi` with its integral over the footprint and its
   affine moments removed but its shape intact.
4. **Post-projection moments vanish.** After `Pi`, `Phi_h^T d` is ~0 for every
   host -- the defining property, checked on set8's real footprints.
5. **Partition.** The footprints from `host_index` are disjoint and their union is
   the `host_member` mask -- so the block-diagonal form is valid (guards against a
   future non-top-level `host_index`).

## 8. Open risks and decisions still owed

1. **Shear may be too strong to forbid.** Removing the full `M_h` forbids the
   module from applying *any* coherent tidal reshaping to a host. The claim that
   this is safe rests on SR2 already having host-scale tides right (§2.1, bulk
   match); if a step-5 Rockstar gate shows suppressed triaxiality or tidal
   features, drop shear from the basis (project against `[1, xi]` isotropic-only,
   the 3+1 dof translation+dilation variant of §3). This is a one-line change to
   `Phi_h` and the cheapest thing to sweep if the full projection underperforms.
2. **Footprint edge.** `Omega_h` from LR broadcast has an 8-HR-site-quantized
   boundary; a subhalo straddling a host's LR edge is partly constrained, partly
   free. Likely negligible (subhalos are Lagrangian-pure, median 1 tile of origin,
   `subhalos-are-lagrangian-pure`), but unmeasured.
3. **Interaction with the local scale `s`.** `Pi` acts on the un-normalized
   residual; the `1/s^2` loss weighting of §4.2 acts on the normalized one. They
   commute (both linear, `s` is diagonal) but the order of operations must be
   fixed once and tested -- project, then normalize for the loss.
4. **It does not by itself guarantee boundedness.** `Pi` guarantees the host is not
   moved; it says nothing about whether the added `d_perp` makes *bound* clumps.
   That remains `sr2_substructure_module.md` open risk 1 and is settled only by
   real Rockstar (skeleton item 6), never by a soft term.

## 9. Where it sits in the pilot

This is the deliverable `docs/pilot_steps_2_4.md` §6 item 2 calls "the last item
feeding the module." With it specified:

| pilot step | status after this note |
| --- | --- |
| 3 re-choose the high-pass | **decided** -- per-host affine-moment projection, this doc |
| 5 train Option B | **unblocked**; the operator is §6, the objective §5 |
| 1 `mu` collapse | still unrun, still gates the design (independent, cheap, CPU) |
| 6 Rockstar gate | unchanged -- the only test that closes open risks 1, 8.1, 8.4 |

Per project convention the precompute is a CPU sbatch reading the existing
`<box>_lagrangian_host.npz`, and step 5 is a GPU sbatch that applies the
precomputed operator in-loop. Nothing here runs on a login node.
