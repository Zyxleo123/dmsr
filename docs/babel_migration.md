# Moving the project from rhea (autonlab) to babel

The autonlab cluster (rhea) is scheduled to shut down around 2026-10-20. Before then, everything the IC
experiments need was uploaded to Backblaze B2 and the code was pushed to GitHub. This
document explains how to rebuild the working state on babel.

- **Code:** GitHub `Zyxleo123/dmsr`, branch `feat/direct-sr2-reward-ft` (the active
  branch; `master` and `fix/reward-pipeline-review` are older and also pushed).
- **Data:** `b2:proteina/dmsr/`. `MANIFEST.txt` at that root lists every prefix with
  its size and an OK/FAIL from a one-way check against rhea. If the file is missing,
  the upload did not reach its last step, so check each prefix before you trust it.
- **Scripts:** `scripts/transfer/b2_upload.sh` (what was uploaded, and from where),
  `b2_download.sh` (the reverse), and `rewrite_paths.sh` (rhea paths to babel paths).

## 1. What is in B2

The layout mirrors rhea's two roots, so every path can be restored by swapping the
root:

| B2 prefix | rhea path | contents |
|---|---|---|
| `scratch/DMSR/...` | `/zfsauton/scratch/yixiz/DMSR/...` (`$ZFS/DMSR`) | bulk data |
| `home/DMSR/cosmo_sr_project/{runs,docs/figures}` | `/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/...` | git-ignored run outputs and figures |
| `bundle/` | (none) | tarballs and exports, listed below |

Upload tiers, from `b2_upload.sh`:

| tier | what | size | needed for |
|---|---|---|---|
| 0 | `bundle/`: a `git bundle --all` of this repo; `repos/<name>.{txt,patch,untracked.tar.gz}` for SRS-map2map, map2map, map2map_styled and MP-Gadget (remote, branch, commit, uncommitted diff, untracked files); `MP-Gadget.Options.mk`; `map2map_colleague.tar.gz`; `external_rockstar_gadget4.tar.gz` (our patched Rockstar and the GADGET-4 SUBFIND build config); `conda_{pjm,g4}.yml` + `pip_{pjm,g4}.txt`; `claude_memory.tar.gz` + `claude_global_CLAUDE.md`. Also the repo's `runs/` and `docs/figures/`. | ~1 GB | everything |
| 1 | `dmsr_reward/ic_determinism` (all the IC results: set9 and set10 at 256³, set9 at 128³, the scaling run, and the dmo-256/128 Rockstar catalogs), `dmsr_reward/halos` (HR and frozen-SR2 Rockstar catalogs, set0-15), `dmsr_reward/{logs,env}`, and `lr_sims{,_256}` paramfiles, seed manifests and logs | ~4 GB | reading the results; the HR/SR2 catalogs the match stage needs |
| 2 | `int_redshift_same_cosmology/dmo-{64,128,256,512}/set*/PART_009` (`disp`, `vel`, `cat`, `style` .npy, z=0) | ~110 GB | the source of everything: HR, LR, and the 128³/256³ runs used as IC proxies |
| 3 | `paired_catnorm/{lr,hr}` (49 GB), `dmsr_reward/cache/sr2_base` (frozen SR2 fields, 48 GB), `lr_sims*/set*/catnorm*.npy` | ~100 GB | **the IC stages read `paired_catnorm/hr/<box>.npy` and `cache/sr2_base/<box>_seed0_*.npy` directly**, so pull these for at least set9 and set10 |

**Not uploaded** (not part of the IC line, and all regenerable or dead): trained arm
checkpoints (`dmsr_reward/checkpoints`, `sr2_direct`, `sr2_scratch`, `m2m`,
`member_gather`, ...), `flow_rockstar` and SUBFIND snapshot dirs, the other
`dmsr_reward/*` diagnostics, `lr_sims*/set*/output/` raw snapshots (regenerate them from
the uploaded paramfiles and seeds), `cosmo_sr_runs`, `dmsr_local_editor`, `sr2_baseline`,
and `m2m_layout`. The frozen SR2 weights `G_z0.pt` are tracked in the public
SRS-map2map repo, so they were not uploaded either.

## 2. Credentials

The B2 key is **not** in B2 or GitHub. Copy rhea's `~/.config/rclone/rclone.conf` to
babel by hand (from your laptop, or from a password manager), or run `rclone config`
and create a `b2` remote with the same account ID and application key. Test it with
`rclone lsd b2:proteina`, which should list `dmsr`.

## 3. Code

```bash
mkdir -p $HOME/DMSR && cd $HOME/DMSR
git clone git@github.com:Zyxleo123/dmsr.git cosmo_sr_project   # ssh key auth; https had no credentials on rhea
git -C cosmo_sr_project switch feat/direct-sr2-reward-ft

# siblings (external/map2map and external/SRS-map2map are symlinks to ../../<name>)
git clone https://github.com/yueyingn/SRS-map2map
git clone https://github.com/eelregit/map2map
git -C map2map remote add xw https://github.com/sagasv5-xw/map2map.git && git -C map2map fetch xw
git clone https://github.com/MP-Gadget/MP-Gadget
```

Then, with the bundle downloaded (step 4, tier 0) to `$B=$NEW_SCRATCH/DMSR/_b2_bundle`:

```bash
for r in SRS-map2map map2map MP-Gadget; do
    git -C $r checkout $(awk '/^commit:/{print $2}' $B/repos/$r.txt)
    [ -s $B/repos/$r.patch ] && git -C $r apply $B/repos/$r.patch
    [ -f $B/repos/$r.untracked.tar.gz ] && tar -C $r -xzf $B/repos/$r.untracked.tar.gz
done
cp $B/repos/MP-Gadget.Options.mk MP-Gadget/Options.mk
tar -xzf $B/repos/map2map_colleague.tar.gz             # -> ./map2map_colleague
# map2map_styled was a clone of eelregit/map2map with local edits:
git clone https://github.com/eelregit/map2map map2map_styled
git -C map2map_styled checkout $(awk '/^commit:/{print $2}' $B/repos/map2map_styled.txt)
git -C map2map_styled apply $B/repos/map2map_styled.patch
# patched Rockstar + GADGET-4 SUBFIND config, under the repo's external/
tar -C cosmo_sr_project/external -xzf $B/repos/external_rockstar_gadget4.tar.gz
```

`external/COMMITS.txt`, `external/rockstar/BUILD_NOTE.txt` and
`docs/sr2_subfind_crosscheck.md` describe how to rebuild Rockstar, GADGET-4 and MP-Gadget.
Rebuild them all on babel, because rhea's binaries link rhea's libraries. Rhea had no
system gcc or MPI, so the `g4` conda env was the GADGET-4 toolchain. If babel has
modules, the systype in `Makefile.systype` may not be needed.

If the GitHub branch is ever in doubt, `git clone $B/cosmo_sr_project.bundle` restores
every branch as it stood on rhea.

### Python envs

```bash
conda env create -n pjm -f $B/conda_pjm.yml     # main env (torch); base had none on rhea
conda env create -n g4  -f $B/conda_g4.yml      # GADGET-4 build toolchain
```

If the solve fails, create a bare env with the same Python and install from
`pip_pjm.txt`. torch/CUDA wheels must match babel's driver, so reinstall torch from
the right index rather than pinning rhea's build.

## 4. Data

```bash
cd $HOME/DMSR/cosmo_sr_project
NEW_SCRATCH=<bulk root> NEW_HOME=$HOME bash scripts/transfer/b2_download.sh 0 1   # ~5 GB
NEW_SCRATCH=<bulk root> NEW_HOME=$HOME bash scripts/transfer/b2_download.sh 2 3   # ~210 GB
```

`<bulk root>` takes over the role of `/zfsauton/scratch/yixiz`, so the data lands in
`<bulk root>/DMSR/...`. Pick a filesystem with at least ~400 GB of headroom, because
Rockstar dumps and arm fields grow fast. On rhea, invalid labels (empty `rockstar.log`,
truncated `.particles`) turned out to be a full per-user quota, not a RAM problem.
The script resumes when re-run. For a long pull, check whether babel's login node
kills long processes (rhea's did) and whether its compute nodes can reach B2. If they
can, run the pull as a CPU batch job.

## 5. Paths and cluster policy

There are about 300 sbatch scripts. They hard-code `/zfsauton2/home/yixiz` and
`/zfsauton/scratch/yixiz`, both in `#SBATCH --output=` lines (which no variable can
override) and in preamble fallbacks. Rewrite them on a branch:

```bash
git switch -c babel
NEW_HOME=$HOME NEW_SCRATCH=<bulk root> bash scripts/transfer/rewrite_paths.sh
git diff --stat; git commit -am "Point paths at babel"
```

The preamble (`scripts/slurm/_reward_common.sh`) also honours `PROJECT`, `ZFS`,
`DMSR_REWARD_ROOT`, `DATA` and `CONDA_ENV`. It sources conda from
`$ZFS/miniconda3/etc/profile.d/conda.sh` after the rewrite, so either install conda
there or edit that one line.

**Re-measure these before trusting anything on babel.** They are rhea facts, recorded in
`~/.claude/CLAUDE.md` and the memory files:

- Partitions, QOS and GPU classes. rhea used `general`, `cpu` and `debug`, with one QOS
  per partition, a ~50 queued-job cap on `cpu`, and ~8 concurrent GPUs per user. Every
  `#SBATCH --partition/--qos/--exclude/--nodelist` line is rhea-specific. `myfree` was
  a rhea shell function.
- `sbatch --export=<anything>` held jobs with `(user env retrieval failed requeued
  held)` on rhea. That is why configuration travels as positional env-file arguments.
  The convention costs nothing, so keep it, even if babel does not have the bug.
- On rhea, `sacct` was dead and exited 0. On babel it may work. Test it once.
- On rhea, `$SCRATCH` was exported by the cluster, which is why the job variables are
  prefixed (`SR2_*`, `ICD_*`). Check babel's environment for name clashes.

## 6. Smoke test: re-run the IC test on set9

```bash
DRY=1 bash scripts/slurm/submit_ic_determinism.sh      # check the env file + sbatch lines
bash scripts/slurm/submit_ic_determinism.sh            # set9 at 256^3, CPU only
```

Pass condition: the new `set9_ripples.json` reproduces the uploaded one: r ≈ 0.90
(dmo-256) vs 0.06 (frozen SR2) in the 2.5-6 h/Mpc band. The verdict should read
`IC CARRIES PLACEMENT`, with a 1-3 R_vir match of ~0.56 against a ~0.01 null. Write the
new outputs to a separate `ICD_ROOT` so they do not overwrite the uploaded reference.

## 7. Claude Code memory

The project memory is in `bundle/claude_memory.tar.gz`. Claude Code keys it by the
project's absolute path, with `/` replaced by `-`:

```bash
P=$HOME/DMSR/cosmo_sr_project
D=$HOME/.claude/projects/$(echo "$P" | sed 's|/|-|g')
mkdir -p "$D" && tar -C "$D" -xzf $B/claude_memory.tar.gz   # -> $D/memory/
cp $B/claude_global_CLAUDE.md $HOME/.claude/CLAUDE.md        # then fix the rhea-only rules
```

The global `CLAUDE.md` describes rhea's slurm rules (no ssh to rhea, `lov4`,
`myfree`, the `--export` bug, the dead `sacct`). Edit it for babel before letting an
agent submit jobs. Many memory files cite rhea job IDs and paths. They are history and
do not need rewriting.

## 8. Where the IC line stands

The docs and memories hold the details (`ic-determinism-test`, `ic-conditioned-sr-branch`).

- **Result.** The IC carries outskirt placement. On set9 and set10, a 256³ run of the
  same seed matches HR's 1-3 R_vir objects at ~0.56, against a null of ~0.01, and almost
  all matches come from the same IC patch. Frozen SR2 manages 0.02-0.06. Recovery scales
  with how many particles the input gives the forming ripple: half-recovery comes at
  ~35-100 input particles. Direction from the host survives, but radius inside R_vir
  does not.
- **Blocker.** set0-15 have no ICs or seeds on disk. The data request to Xiaowen Zhang
  (seeds or ICs, the noise-pattern script, and the SR-IV checkpoint from
  `sagasv5-xw/map2map` branch `map2map_sr`) was drafted for the user to send.
- **Fallback.** If the request yields nothing, run our own paired LR/HR boxes with known
  seeds. The MP-GenIC paramfile templates and seed manifests are in
  `lr_sims*/set*/paramfile.genic` and `lr_sims*/manifest.tsv` (tier 1). MP-Gadget
  builds from the `Options.mk` in the bundle.
