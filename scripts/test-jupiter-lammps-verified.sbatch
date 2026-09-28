#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
# Test a pinned, previously validated ML-IAP export on JUPITER.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_JUPITER_ROOT:?set the private artifact root}"
: "${MLIP_LESSON_ROOT:?set the staged lesson source root}"
: "${MLIP_JUPITER_BUILD_ID:?set a completed build job ID}"
case "$MLIP_JUPITER_BUILD_ID" in ''|*[!0-9]*) exit 2 ;; esac
model="$MLIP_JUPITER_ROOT/inputs/mace-mliap-db578c55.pt"
venv="$MLIP_JUPITER_ROOT/env-mace-0315"
build="$MLIP_JUPITER_ROOT/builds/lammps-mliap-$MLIP_JUPITER_BUILD_ID"
test -f "$model" && test -x "$venv/bin/python"
test -f "$MLIP_LESSON_ROOT/examples/lammps_si.py"
printf '%s  %s\n' db578c556298ad3bb1f4a93faa50d540eb2b9792215e81ef7548dd7e20a746e8 "$model" | sha256sum -c - >/dev/null
(cd "$build" && sha256sum -c lmp.sha256 >/dev/null)
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 CUDA/12 OpenMPI/5.0.5 \
  Python/3.12.3 PyTorch/2.5.1 >/dev/null
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export LD_LIBRARY_PATH="$build/prefix/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
result="$MLIP_JUPITER_ROOT/results/lammps-verified-$SLURM_JOB_ID"
mkdir -p -m 700 "$MLIP_JUPITER_ROOT/results"
mkdir -m 700 "$result"
srun --nodes=1 --ntasks=1 --gpus=1 --cpu-bind=cores \
  "$venv/bin/python" "$MLIP_LESSON_ROOT/examples/lammps_si.py" \
  --mliap-model "$model" --integrator nve --cells 2 --warmup 1 --steps 5 \
  >"$result/md.json" 2>"$result/md.err"
"$venv/bin/python" - "$result/md.json" <<'PY'
import json, math, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data['state'] == 'completed' and data['atoms_per_replica'] == 64
assert data['measured_steps'] == 5 and math.isfinite(data['measured_seconds'])
print('JUPITER ML-IAP/Kokkos one-GPU silicon smoke passed')
PY
