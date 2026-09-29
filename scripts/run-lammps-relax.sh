#!/usr/bin/env bash
# Relax one of the same fixed-cell structures with LAMMPS on one GPU.
set -euo pipefail
lesson_root=$1
lmp=$2
model=$3
start=$4

exec "$lmp" -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -log none -in "$lesson_root/examples/lammps_relax.in" \
    -var model "$model" -var start "$start"
