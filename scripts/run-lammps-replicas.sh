#!/usr/bin/env bash
# Independent LAMMPS processes sharing one already-allocated GPU.
set -euo pipefail
lesson_root=$1
lmp=$2
model=$3
replicas=$4
result=$5
steps=${6:-2000}
case "$replicas" in 1|2|4|8) ;; *) echo 'replicas must be 1, 2, 4, or 8' >&2; exit 2 ;; esac
mkdir "$result"                    # Refuse an existing output directory.
pids=()
for ((i = 0; i < replicas; i++)); do
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
printf 'LAMMPS replica logs: %s\n' "$result"
exit "$status"
