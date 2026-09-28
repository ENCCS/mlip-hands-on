#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_LESSON_ROOT:?stage the lesson outside the allocation}"
: "${MLIP_LEONARDO_PYTHON:?set the private native Python 3.12 executable}"
: "${MLIP_MODEL:?set the original MACE checkpoint outside Git}"
: "${MLIP_RESULTS_DIR:?set a private, existing results directory}"
case "${MLIP_REPLICAS:-}" in 1|8) ;; *) echo 'MLIP_REPLICAS must be 1 or 8' >&2; exit 2 ;; esac
test -x "$MLIP_LEONARDO_PYTHON"
test -f "$MLIP_LESSON_ROOT/examples/alchemi_si.py"
test -d "$MLIP_RESULTS_DIR"
printf '%s  %s\n' 2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736 "$MLIP_MODEL" | sha256sum -c - >/dev/null
module load profile/base >/dev/null 2>&1
module load gcc/12.2.0 cuda/12.2 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 >/dev/null 2>&1
python_root=$(dirname -- "$(dirname -- "$MLIP_LEONARDO_PYTHON")")
export LD_LIBRARY_PATH="$python_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1 TORCH_COMPILE_DISABLE=1 TORCH_DISABLE_NATIVE_JIT=1
result="$MLIP_RESULTS_DIR/leonardo-alchemi-${MLIP_REPLICAS}-${SLURM_JOB_ID}.json"
test ! -e "$result"
srun --nodes=1 --ntasks=1 --gpus=1 --cpu-bind=cores \
  "$MLIP_LEONARDO_PYTHON" "$MLIP_LESSON_ROOT/examples/alchemi_si.py" \
  --model "$MLIP_MODEL" --integrator nve --replicas "$MLIP_REPLICAS" \
  --cells 2 --warmup 1 --steps 5 >"$result"
jq -e --argjson n "$MLIP_REPLICAS" \
  '.state == "completed" and .replicas == $n and .measured_steps == 5' "$result" >/dev/null
printf 'Leonardo ALCHEMI smoke passed: %s trajectories\n' "$MLIP_REPLICAS"
