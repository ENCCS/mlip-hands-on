#!/usr/bin/env bash
# One NVE trajectory across one, two, or four GPUs on one allocated node.
set -euo pipefail
lesson_root=$1
ranks=$2
lmp=$3
model=$4
start=$5
warmup=$6
steps=$7
case "$ranks" in 1|2|4) ;; *) echo 'use one, two, or four ranks' >&2; exit 2;; esac
test "${SLURM_JOB_NUM_NODES:-}" = 1
: "${CUDA_VISIBLE_DEVICES:?allocated GPUs must be visible}"
export MLIP_ALLOCATED_CUDA_DEVICES="$CUDA_VISIBLE_DEVICES"
mpi_option=()
if [[ -n "${MLIP_SRUN_MPI:-}" ]]; then
    mpi_option+=("--mpi=$MLIP_SRUN_MPI")
fi
exec srun "${mpi_option[@]}" --nodes=1 --ntasks="$ranks" --gpu-bind=none \
    /bin/bash -c 'export CUDA_VISIBLE_DEVICES="$MLIP_ALLOCATED_CUDA_DEVICES"; exec "$@"' _ \
    "$lmp" -k on g "$ranks" -sf kk \
    -pk kokkos newton on neigh half gpu/aware on \
    -log none -in "$lesson_root/maintainer/benchmarks/lammps_mace_nve.in" \
    -var start "$start" -var model "$model" \
    -var warmup "$warmup" -var steps "$steps"
