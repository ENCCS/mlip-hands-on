#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
# Short native ALCHEMI functional check in a separate reviewed venv.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_JUPITER_ROOT:?set a private artifact root}"
: "${MLIP_LESSON_ROOT:?set the staged lesson source root}"
: "${MLIP_JUPITER_ALCHEMI_ENV_ID:?set the completed environment job ID}"
: "${MLIP_JUPITER_REPLICAS:=1}"
case "$MLIP_JUPITER_ALCHEMI_ENV_ID" in ''|*[!0-9]*) exit 2 ;; esac
case "$MLIP_JUPITER_REPLICAS" in 1|8) ;; *) exit 2 ;; esac
model="$MLIP_JUPITER_ROOT/inputs/mace-mp-0a-small-2ddb079cee0e131e.model"
venv="$MLIP_JUPITER_ROOT/env-alchemi-$MLIP_JUPITER_ALCHEMI_ENV_ID"
test -f "$model" && test -x "$venv/bin/python"
test -f "$MLIP_LESSON_ROOT/examples/alchemi_si.py"
printf '%s  %s\n' 2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736 "$model" | sha256sum -c - >/dev/null
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 Python/3.12.3 >/dev/null
export PYTHONNOUSERSITE=1
export TORCH_COMPILE_DISABLE=1
export TORCH_DISABLE_NATIVE_JIT=1
export MLIP_ALCHEMI_RUNNER=native
export MLIP_MODEL="$model"
export MLIP_NATIVE_ALCHEMI_PYTHON="$venv/bin/python"
result="$MLIP_JUPITER_ROOT/results/alchemi-${MLIP_JUPITER_REPLICAS}-$SLURM_JOB_ID"
mkdir -p -m 700 "$MLIP_JUPITER_ROOT/results"
mkdir -m 700 "$result"
srun --nodes=1 --ntasks=1 --gpus=1 --cpu-bind=cores \
  bash "$MLIP_LESSON_ROOT/scripts/run-alchemi.sh" \
  --integrator nve --replicas "$MLIP_JUPITER_REPLICAS" --cells 2 \
  --warmup 1 --steps 5 >"$result/md.json" 2>"$result/md.err"
"$venv/bin/python" - "$result/md.json" "$MLIP_JUPITER_REPLICAS" <<'PY'
import json, math, pathlib, sys
data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data['state'] == 'completed' and data['engine'] == 'alchemi'
assert data['replicas'] == int(sys.argv[2]) and data['atoms_per_replica'] == 64
assert data['measured_steps'] == 5 and math.isfinite(data['measured_seconds'])
print(f"JUPITER native ALCHEMI one-GPU silicon smoke passed: {data['replicas']} replicas")
PY
