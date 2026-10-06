#!/bin/bash
# Render the soft-membership solve from dumped npz snapshots. Pure plotting:
# reads the npz the dump job wrote, produces the evolution and budget-matrix
# PNGs. No model, no field, no GPU -- exactly the "figures redraw from saved
# artefacts" split, so it runs on the CPU partition behind the dump.
#
#   sbatch scripts/slurm/membership_viz_cpu.sbatch <envfile>
#SBATCH --job-name=mgmemviz
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/DMSR/dmsr_reward/logs/slurm-mgmemviz-%j.out

set -euo pipefail
_D="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/DMSR/cosmo_sr_project}/scripts/slurm"
[[ -r "$_D/_sr2_direct_common.sh" ]] \
    || _D=/zfsauton2/home/yixiz/DMSR/cosmo_sr_project/scripts/slurm
source "$_D/_sr2_direct_common.sh" "$@"
set -euo pipefail

: "${MV_NPZ:?MV_NPZ must be the dumped membership npz path(s) (space-separated)}"
: "${MV_OUT_DIR:=}"
: "${MV_N_EVOLUTION:=4}"
: "${MV_N_MATRIX:=6}"
: "${MV_MAX_MATRIX_ROWS:=400}"
: "${MV_SELECT:=spread}"    # spread | mass | substitution
: "${MV_HEATMAP:=0}"        # 1 = also render the region heatmap + centres
: "${MV_HEATMAP_ONLY:=0}"   # 1 = render ONLY the heatmap (no GPU re-dump needed)
: "${MV_HEATMAP_HALFWIDTH:=10.0}"
: "${MV_RADIAL:=0}"      # 1 = also render membership vs host radius
: "${MV_RADIAL_ONLY:=0}" # 1 = render ONLY that figure
: "${MV_PANEL:=0}"       # 1 = also render the mean-<P> x-z panel alone
: "${MV_PANEL_ONLY:=0}"  # 1 = render ONLY that panel
: "${MV_HOST_RVIR:=0}"   # parent host R_vir [Mpc/h]; 0 = npz, then ascii, then Mpc/h
: "${MV_ROCKSTAR_ASCII:=}"
: "${MV_R_MAX_RVIR:=4.0}"
: "${MV_CENTRE_RADIAL:=0}"       # 1 = radial dist of optimized c_j vs HR
: "${MV_CENTRE_RADIAL_ONLY:=0}"
: "${MV_CENTRE_RADIAL_BARS:=0}"   # grouped bar chart of c_j vs HR per range
: "${MV_CENTRE_RADIAL_BARS_ONLY:=0}"
: "${MV_SR_SUBHALO_MEDIAN:=nan}" # gate's SR subhalo median r/Rvir, as a ref line

# A dump job can fail its gate and leave no npz; say so and EXIT 0 rather than
# strand nothing (nothing depends on this) but make the reason legible.
_have=0
for f in $MV_NPZ; do [[ -r "$f" ]] && _have=$((_have + 1)); done
if [[ "$_have" == 0 ]]; then
    echo ">>> no readable npz among: $MV_NPZ"
    echo ">>> the dump job likely did not produce one; nothing to plot."
    exit 0
fi

echo "=== membership viz: $_have npz file(s)"
python -u scripts/features/viz_membership_solve.py $MV_NPZ \
    ${MV_OUT_DIR:+--out-dir "$MV_OUT_DIR"} \
    --n-evolution "$MV_N_EVOLUTION" \
    --n-matrix "$MV_N_MATRIX" \
    --max-matrix-rows "$MV_MAX_MATRIX_ROWS" \
    --select "$MV_SELECT" \
    --heatmap-halfwidth "$MV_HEATMAP_HALFWIDTH" \
    --host-rvir "$MV_HOST_RVIR" \
    --r-max-rvir "$MV_R_MAX_RVIR" \
    ${MV_ROCKSTAR_ASCII:+--rockstar-ascii "$MV_ROCKSTAR_ASCII"} \
    $([[ "$MV_RADIAL_ONLY" == 1 ]] && echo --radial-only \
       || { [[ "$MV_RADIAL" == 1 ]] && echo --radial; }) \
    $([[ "$MV_PANEL_ONLY" == 1 ]] && echo --panel-only \
       || { [[ "$MV_PANEL" == 1 ]] && echo --panel; }) \
    --sr-subhalo-median "$MV_SR_SUBHALO_MEDIAN" \
    $([[ "$MV_CENTRE_RADIAL_ONLY" == 1 ]] && echo --centre-radial-only \
       || { [[ "$MV_CENTRE_RADIAL" == 1 ]] && echo --centre-radial; }) \
    $([[ "$MV_CENTRE_RADIAL_BARS_ONLY" == 1 ]] && echo --centre-radial-bars-only \
       || { [[ "$MV_CENTRE_RADIAL_BARS" == 1 ]] && echo --centre-radial-bars; }) \
    $([[ "$MV_HEATMAP_ONLY" == 1 ]] && echo --heatmap-only \
       || { [[ "$MV_HEATMAP" == 1 ]] && echo --heatmap; })

echo "=== done. PNGs (per npz stem):"
echo "    <stem>_evolution.png   membership P per pool particle, one col per EM sweep"
echo "    <stem>_matrix.png      particle x candidate budget heatmap (final sweep)"
echo "    <stem>_radial.png      max_j P per particle vs r/Rvir + where the mass comes from"
echo "    <stem>_meanP_xz.png    mean <P> per bin, x-z panel alone (slide size)"
echo "    <stem>_centre_radial.png  radial dist of optimized c_j vs HR (own parent)"
echo "    <stem>_centre_radial_bars.png  same, as a grouped bar chart per range"
