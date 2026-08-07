# Chunk reward: host-attributed vs naive crop Rockstar

Status as of 2026-08-04. Box: **set8**, HR vs frozen SR2 base (seed 0).

## Verdict

**Find with full-box Rockstar; assign credit locally.** Do not train on bare
periodic Lagrangian crops. A crop-Rockstar audit still ranks HR above SR2, but
it is a fake small universe (`PERIODIC=1` on the crop, face wrapping) and the
absolute `R_occ` scale is meaningless (reward moments are whole-box).

Host-level \(E^{\rm cat}\) on full-box catalogs agrees with the naive ranking
(\(\mathrm{corr}(E_{\rm cat},\Delta n_{\rm sub})\approx 0.86\),
\(\mathrm{corr}(E_{\rm cat},\Delta R_{\rm occ})\approx 0.79\)) while exposing
*why* (mass + sub count), not a single Mahalanobis scalar.

## Two pipelines (do not mix)

| | Naive crop | Host-attributed \(E^{\rm cat}\) |
|---|---|---|
| Halo finding | Rockstar **per** Lagrangian crop | **One** full-box Rockstar (HR + base already on disk) |
| Geometry | crop as its own `BOX_SIZE`, `PERIODIC=1` | full periodic box; host → 64³ by Eulerian centre |
| Score unit | binned occupation / abundance of the crop | **per host**, averaged over hosts in the chunk |
| Scripts | `scripts/reward/audit_chunk_rockstar.py` | `scripts/reward/audit_host_chunk_reward.py` |
| Reports | `$DMSR_REWARD_ROOT/audits/chunk_rockstar/` | `$DMSR_REWARD_ROOT/audits/host_chunk_reward/` |

SLURM for the crop audit: `scripts/slurm/audit_chunk_rockstar_cpu.sbatch`
(pass overrides as positional `BOX=… CHUNKS=…`, not env-before-sbatch).

## Equations

### Naive \(N_{\rm sub}\) and \(R_{\rm occ}\)

Qualifying subhalo = parent ≥ 0, \(N_p \ge N_p^{\min}\), parent is a qualifying
host.

\[
N_{\rm sub}=\#\{\text{qualifying subs in the catalog}\},\qquad
\Delta n_{\rm sub}=N_{\rm sub}^{\rm HR}-N_{\rm sub}^{\rm SR2}.
\]

Occupation in host-mass bin \(b\):

\[
O_b=\frac{S_b}{H_b},\qquad
s^{\rm occ}_b=\log_{10}(O_b+\varepsilon),
\]

\[
R_{\rm occ}=-D^2_{\rm occ}
=-\bigl(s^{\rm occ}-\mu^{\rm occ}\bigr)^{\mathsf T}
\bigl(C^{\rm occ}_{\rm reg}\bigr)^{-1}
\bigl(s^{\rm occ}-\mu^{\rm occ}\bigr),
\]

\[
\Delta R_{\rm occ}=R_{\rm occ}({\rm HR\,crop})-R_{\rm occ}({\rm SR2\,crop}).
\]

\(\mu,C\) are fit on **whole-box** HR vectors — absolute \(R_{\rm occ}\) on a
25 Mpc crop is misspecified; use sign / ranking only.

### Host-level \(E_j^{\rm cat}\)

Full-box catalogs; match SR2→HR hosts; for tile/chunk \(j\) with HR hosts
\(H_j^{\rm HR}\):

\[
E_j^{\rm cat}
=
\frac{1}{\max\bigl(1,|H_j^{\rm HR}|\bigr)}
\Bigg[
\sum_{h\in H_j^{\rm HR}}
\Bigl(
\lambda_{\rm miss}\,\mathbf{1}_{h\text{ unmatched}}
+
\mathbf{1}_{h\text{ matched}}\,e_h
\Bigr)
+
\lambda_{\rm fp}\,N_j^{\rm FP}
\Bigg].
\]

Default matched error (radial/vel W1 off):

\[
\begin{aligned}
e_h
&=
w_M\,\rho\!\left(\frac{\Delta\log M_{\rm host}}{s_M}\right)
+
w_N\,\rho\!\left(\frac{\log(1+N_{\rm sub}^{\rm gen})-\log(1+N_{\rm sub}^{\rm HR})}{s_{\log}}\right)
\\
&\quad+
w_m\,W_1(F_m^{\rm gen},F_m^{\rm HR})
\quad\text{(omit if either side has 0 subs)}.
\end{aligned}
\]

\(\rho\) = Huber. \(N_{\rm sub}\) here is **per-host** (full-box), not the crop
total. Lower \(E^{\rm cat}\) = SR2 closer to HR.

## set8 results (2026-08-04)

Crop audit used `chunk_hr=128` (25 Mpc). Host audit used `tile_hr=64`, pooled
to the same 128³ chunks 0–3.

### Separated terms (global, host-weighted, base vs HR)

| Term | Mean | Share of \(E_{\rm cat}\) |
|---|---:|---:|
| `dlogM` (Huber) | 3.01 | 64% |
| `dN_sub` (Huber) | 1.42 | 30% |
| false positives | 0.17 | 4% |
| W1 mass-ratio | 0.08 | 2% |
| miss | ~0 | ~0% |

Global \(E_{\rm cat}({\rm base}|{\rm HR})=4.68\) over 3176 HR hosts
(\(M\ge 10^{12}\,M_\odot/h\)). Matched pairs: median \(\Delta\log M\approx -0.11\)
dex (SR2 light), 71% have fewer subs than HR. Misses ≈ 0 — failure is wrong
mass/occupation, not missing hosts.

### Chunk table (128³)

| chunk | host \(E_{\rm cat}\) | naive \(\Delta R_{\rm occ}\) | naive \(\Delta n_{\rm sub}\) |
|------:|---------------------:|----------------------------:|-----------------------------:|
| 0 | 5.12 | +5783 | +654 |
| 1 | 2.47 | +3087 | +436 |
| 2 | 5.17 | +6241 | +775 |
| 3 | 4.55 | +7722 | +537 |

Chunk 1 is least bad under both schemes.

## Design implications

1. **Production credit:** full-box Rockstar + centre→tile (or margin-context
   Rockstar when only a sparse crop exists). Never bare periodic 64³/128³ crops
   for training rewards.
2. **If every chunk is scored:** one full-box find is cheaper than overlapping
   context finds; attribute afterward.
3. **Before training on \(E^{\rm cat}\):** recalibrate / downweight `dlogM`
   (`s_M=0.15` is tight vs ~0.4 dex mean SR2 bias) so `dN_sub` and mass-ratio
   W1 can drive learning. Add radial/vel W1 only after those move
   (`--full-phase`).
4. **Empty tiles:** only FP + field terms; no large “correctly empty” bonus.
   Batch by host count / mass, not by chunk count.

## Artifacts

```
$DMSR_REWARD_ROOT/audits/chunk_rockstar/set8_chunks0-1-2-3.json
$DMSR_REWARD_ROOT/audits/host_chunk_reward/set8_host_chunk_reward.json
$DMSR_REWARD_ROOT/halos/set8__hr__hr/hr_rockstar/halos_0.0.ascii
$DMSR_REWARD_ROOT/halos/set8__base__base/base_rockstar/halos_0.0.ascii
```

Re-run:

```bash
python scripts/reward/audit_chunk_rockstar.py --box set8 --chunks 0,1,2,3 \
  --sources hr,base --set geometry.chunk_hr=128
python scripts/reward/audit_host_chunk_reward.py --box set8
```
