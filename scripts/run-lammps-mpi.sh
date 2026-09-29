#!/usr/bin/env bash
# One coupled trajectory, one GPU per rank, within an existing allocation.
set -euo pipefail
lesson_root=$1
ranks=$2
lmp=$3
model=$4
cells=${5:-4}
steps=${6:-100}
case "$ranks" in 1|2|4) ;; *) echo 'use one, two, or four ranks' >&2; exit 2;; esac
# Kokkos -k g is GPUs per node, not GPUs per MPI rank. Each local rank
# selects its device from this common visible set.
: "${CUDA_VISIBLE_DEVICES:?allocation must expose GPUs}"
export MLIP_ALLOCATED_CUDA_DEVICES="$CUDA_VISIBLE_DEVICES"

mpi_option=()
if [[ -n "${MLIP_SRUN_MPI:-}" ]]; then
    mpi_option+=("--mpi=$MLIP_SRUN_MPI")
fi
exec srun "${mpi_option[@]}" --ntasks="$ranks" --gpu-bind=none \
    /bin/bash -c 'export CUDA_VISIBLE_DEVICES="$MLIP_ALLOCATED_CUDA_DEVICES"; exec "$@"' _ \
    "$lmp" -k on g "$ranks" -sf kk \
    -pk kokkos newton on neigh half gpu/aware on \
    -log none -in "$lesson_root/examples/lammps_mace.in" \
    -var model "$model" -var cells "$cells" -var warmup 0 \
    -var steps "$steps" -var seed 20260924 -var bath_seed 20260925
