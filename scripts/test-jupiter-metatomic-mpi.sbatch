#!/bin/bash
#SBATCH --partition=booster
#SBATCH --time=00:15:00
# Short Metatomic/Kokkos silicon check. Supply account, ranks, GPUs, and memory.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${SLURM_NTASKS:?request ranks explicitly}"
: "${SLURM_JOB_NUM_NODES:?request nodes explicitly}"
: "${MLIP_JUPITER_ROOT:?set the private artifact root}"
: "${MLIP_LESSON_ROOT:?set the staged lesson source root}"
: "${MLIP_JUPITER_METATOMIC_BUILD_ID:?set a completed build job ID}"
: "${MLIP_METATOMIC_MODEL:?set the reviewed exported Metatomic model}"
: "${MLIP_METATOMIC_MODEL_SHA256:?set its reviewed SHA-256}"
: "${MLIP_METATOMIC_OVERLAY:?set the reviewed Metatomic package overlay}"
: "${MLIP_JUPITER_ALCHEMI_ENV_ID:?set the qualified native environment job ID}"
case "$MLIP_JUPITER_METATOMIC_BUILD_ID" in ''|*[!0-9]*) exit 2 ;; esac
case "$MLIP_JUPITER_ALCHEMI_ENV_ID" in ''|*[!0-9]*) exit 2 ;; esac
case "${SLURM_JOB_NUM_NODES}:${SLURM_NTASKS}" in
  1:1|1:2|1:4|2:8) ;;
  *) echo 'unreviewed JUPITER Metatomic geometry' >&2; exit 2 ;;
esac
if test -n "${SLURM_GPUS_PER_TASK:-}" && test "$SLURM_GPUS_PER_TASK" != 1; then
  echo 'request one GPU per rank' >&2
  exit 2
fi
build="$MLIP_JUPITER_ROOT/builds/lammps-metatomic-$MLIP_JUPITER_METATOMIC_BUILD_ID"
model="$MLIP_METATOMIC_MODEL"
overlay="$MLIP_METATOMIC_OVERLAY"
venv="$MLIP_JUPITER_ROOT/env-alchemi-$MLIP_JUPITER_ALCHEMI_ENV_ID"
input="${MLIP_METATOMIC_INPUT:-$MLIP_LESSON_ROOT/examples/lammps_metatomic_si.in}"
case "$input" in
  "$MLIP_LESSON_ROOT/examples/lammps_metatomic_si.in"|"$MLIP_LESSON_ROOT/examples/lammps_metatomic_si_512.in") ;;
  *) echo 'unreviewed Metatomic input' >&2; exit 2 ;;
esac
test -f "$input" && test -f "$model" && test -d "$overlay/metatomic"
test -x "$venv/bin/python" && test -x "$build/prefix/bin/lmp"
printf '%s  %s\n' "$MLIP_METATOMIC_MODEL_SHA256" "$model" | sha256sum -c - >/dev/null
(cd "$build" && sha256sum -c lmp.sha256 >/dev/null)
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 CUDA/12 OpenMPI/5.0.5 \
  Python/3.12.3 >/dev/null
export LD_LIBRARY_PATH="$venv/lib/python3.12/site-packages/nvidia/nccl/lib:$build/prefix/lib64:$build/prefix/lib:$venv/lib/python3.12/site-packages/torch/lib:$overlay/metatensor/lib:$overlay/metatensor_torch/torch-2.13/lib:$overlay/metatomic/torch/torch-2.13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
result="$MLIP_JUPITER_ROOT/results/metatomic-$SLURM_JOB_ID"
mkdir -p -m 700 "$MLIP_JUPITER_ROOT/results"
mkdir -m 700 "$result"
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_NTASKS" \
  --gpus-per-task=1 --cpu-bind=cores \
  "$build/prefix/bin/lmp" -k on g 1 -sf kk \
  -var model "$model" -in "$input" -log "$result/lammps.log" \
  >"$result/run.out" 2>"$result/run.err"
grep -q 'Loop time' "$result/lammps.log"
printf 'JUPITER Metatomic/Kokkos functional check passed: %s rank(s)\n' "$SLURM_NTASKS"
