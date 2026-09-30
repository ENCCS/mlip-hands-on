#!/usr/bin/env bash
# One coupled trajectory: two or four MPI ranks share one allocated GPU.
# Run only inside a one-node, one-GPU allocation with CUDA MPS enabled.
set -euo pipefail

lesson_root=$1
ranks=$2
lmp=$3
model=$4
cells=$5
steps=${6:-100}

case "$ranks" in 2|4) ;; *) echo 'use two or four ranks' >&2; exit 2;; esac
: "${CUDA_VISIBLE_DEVICES:?allocate one visible GPU first}"
export MLIP_ALLOCATED_CUDA_DEVICES="$CUDA_VISIBLE_DEVICES"

mpi_option=()
if [[ -n "${MLIP_SRUN_MPI:-}" ]]; then
    mpi_option+=("--mpi=$MLIP_SRUN_MPI")
fi
exec srun "${mpi_option[@]}" --nodes=1 --ntasks="$ranks" --gpus=1 \
    --gpu-bind=none --wait=30 \
    /bin/bash -c 'export CUDA_VISIBLE_DEVICES="$MLIP_ALLOCATED_CUDA_DEVICES"; exec "$@"' _ \
    "$lmp" -k on g 1 -sf kk \
    -pk kokkos newton on neigh half gpu/aware on \
    -log none -in "$lesson_root/examples/lammps_mace.in" \
    -var model "$model" -var cells "$cells" -var warmup 10 \
    -var steps "$steps" -var seed 20260924 -var bath_seed 20260925
