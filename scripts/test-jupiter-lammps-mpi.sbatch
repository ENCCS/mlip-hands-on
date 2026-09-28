#!/bin/bash
#SBATCH --partition=booster
#SBATCH --time=00:15:00
# One private MPI/Kokkos functional check, not a benchmark.
# Supply --account, --nodes, --ntasks, --gpus-per-task=1 and memory to sbatch.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${SLURM_NTASKS:?request MPI ranks explicitly}"
: "${SLURM_JOB_NUM_NODES:?request nodes explicitly}"
: "${MLIP_JUPITER_ROOT:?set a private artifact root}"
: "${MLIP_LESSON_ROOT:?set the staged lesson source root}"
: "${MLIP_JUPITER_BUILD_ID:?set a completed build job ID}"
case "$MLIP_JUPITER_BUILD_ID" in ''|*[!0-9]*) exit 2 ;; esac
case "${SLURM_JOB_NUM_NODES}:${SLURM_NTASKS}" in
  1:1|1:2|1:4|2:8) ;;
  *) echo 'unreviewed JUPITER MPI geometry' >&2; exit 2 ;;
esac
if test -n "${SLURM_GPUS_PER_TASK:-}" && test "$SLURM_GPUS_PER_TASK" != 1; then
  echo 'the job did not request one GPU per rank' >&2
  exit 2
fi
model="$MLIP_JUPITER_ROOT/inputs/mace-mliap-db578c55.pt"
venv="$MLIP_JUPITER_ROOT/env-mace-0315"
build="$MLIP_JUPITER_ROOT/builds/lammps-mliap-$MLIP_JUPITER_BUILD_ID"
test -f "$model" && test -x "$venv/bin/python"
test -f "$MLIP_LESSON_ROOT/examples/lammps_mpi.py"
printf '%s  %s\n' db578c556298ad3bb1f4a93faa50d540eb2b9792215e81ef7548dd7e20a746e8 "$model" | sha256sum -c - >/dev/null
(cd "$build" && sha256sum -c lmp.sha256 >/dev/null)
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 CUDA/12 OpenMPI/5.0.5 \
  Python/3.12.3 PyTorch/2.5.1 >/dev/null
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export LD_LIBRARY_PATH="$build/prefix/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
result="$MLIP_JUPITER_ROOT/results/lammps-mpi-$SLURM_JOB_ID"
mkdir -p -m 700 "$MLIP_JUPITER_ROOT/results"
mkdir -m 700 "$result"
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_NTASKS" \
  --gpus-per-task=1 --cpu-bind=cores \
  "$venv/bin/python" "$MLIP_LESSON_ROOT/examples/lammps_mpi.py" \
  --model "$model" --output-dir "$result" --cells 4 --steps 10 \
  --gpu-binding per-rank >"$result/mpi.out" 2>"$result/mpi.err"
"$venv/bin/python" - "$result" "$SLURM_NTASKS" <<'PY'
import json, math, pathlib, sys
root, count = pathlib.Path(sys.argv[1]), int(sys.argv[2])
for rank in range(count):
    data = json.loads((root / f"rank-{rank}.json").read_text())
    assert data["state"] == "completed" and data["rank"] == rank
    assert data["gpu_binding"] == "per-rank" and data["visible_gpus"] == 1
    assert data["atoms"] == 512 and data["measured_steps"] == 10
    assert math.isfinite(data["final_energy_ev"])
print(f"JUPITER ML-IAP/Kokkos MPI functional check passed: {count} ranks")
PY
