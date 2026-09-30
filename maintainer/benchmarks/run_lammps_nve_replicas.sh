#!/usr/bin/env bash
# Run matched independent NVE trajectories on one already-allocated GPU.
set -euo pipefail
umask 077
lesson_root=$1
lmp=$2
model=$3
starts=$4
result=$5
replicas=$6
warmup=$7
steps=$8
: "${SLURM_JOB_ID:?run inside an allocated GPU job}"
: "${CUDA_VISIBLE_DEVICES:?one allocated GPU must be visible}"
case "$replicas" in 1|2|4|8|16|32|64) ;; *) echo 'replicas must be a power of two up to 64' >&2; exit 2;; esac
mkdir "$result"  # Never reuse a result directory after an uncertain outcome.
pids=()
for ((i = 0; i < replicas; i++)); do
    test -f "$starts/replica-$i.data"
    "$lmp" -k on g 1 -sf kk -pk kokkos newton on neigh half \
        -log none -in "$lesson_root/maintainer/benchmarks/lammps_mace_nve.in" \
        -var start "$starts/replica-$i.data" -var model "$model" \
        -var warmup "$warmup" -var steps "$steps" \
        >"$result/replica-$i.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
for ((i = 0; i < replicas; i++)); do
    grep -q 'Total wall time:' "$result/replica-$i.log" || status=1
done
if ((status != 0)); then
    echo 'At least one LAMMPS replica did not complete; do not summarize' >&2
    exit 1
fi
printf 'Completed %s matched LAMMPS NVE replicas; logs: %s\n' "$replicas" "$result"
