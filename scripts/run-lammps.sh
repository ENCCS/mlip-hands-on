#!/usr/bin/env bash
# One LAMMPS rank on one already-allocated GPU.
set -euo pipefail
lesson_root=$1
lmp=$2
model=$3
cells=${4:-2}
steps=${5:-200}

launch=()
if [[ "${MLIP_SITE:-}" == arrhenius ]]; then
    launch=(srun --mpi=pmi2 --nodes=1 --ntasks=1 --gpus=1)
fi
exec "${launch[@]}" "$lmp" -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -log none -in "$lesson_root/examples/lammps_mace.in" \
    -var model "$model" -var cells "$cells" -var warmup 10 \
    -var steps "$steps" -var seed 20260924 -var bath_seed 20260925
