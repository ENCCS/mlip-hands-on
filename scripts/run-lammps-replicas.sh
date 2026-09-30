#!/usr/bin/env bash
# Eight independent LAMMPS processes sharing one already-allocated GPU.
set -euo pipefail
lesson_root=$1
lmp=$2
model=$3
result=$4
steps=${5:-200}
: "${SLURM_JOB_ID:?run inside one Slurm GPU allocation}"
: "${CUDA_VISIBLE_DEVICES:?one allocated GPU must be visible}"
mkdir "$result"                    # Refuse an existing output directory.
pids=()
for ((i = 0; i < 8; i++)); do
    "$lmp" -k on g 1 -sf kk -pk kokkos newton on neigh half \
        -log none -in "$lesson_root/examples/lammps_mace.in" \
        -var model "$model" -var cells 2 -var warmup 10 \
        -var steps "$steps" -var seed "$((20260924 + 2*i))" \
        -var bath_seed "$((20260925 + 2*i))" \
        >"$result/replica-$i.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
for ((i = 0; i < 8; i++)); do
    grep -q 'Total wall time:' "$result/replica-$i.log" || status=1
done
printf 'Eight LAMMPS process logs: %s\n' "$result"
exit "$status"
