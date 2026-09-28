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
: "${MLIP_LEONARDO_LAMMPS_PREFIX:?set the verified CUDA-Kokkos build}"
: "${MLIP_LEONARDO_WRAPPER_ROOT:?set the matching source Python wrapper root}"
: "${MLIP_LEONARDO_OVERLAY_ROOT:?set the matching CuPy overlay root}"
: "${MLIP_MLIAP_MODEL:?set the verified ML-IAP export outside Git}"
: "${MLIP_RESULTS_DIR:?set a private, existing results directory}"
test -x "$MLIP_LEONARDO_PYTHON"
test -x "$MLIP_LEONARDO_LAMMPS_PREFIX/bin/lmp"
test -f "$MLIP_LEONARDO_WRAPPER_ROOT/lammps/mliap/__init__.py"
test -f "$MLIP_LEONARDO_OVERLAY_ROOT/cupy/__init__.py"
test -d "$MLIP_RESULTS_DIR"
printf '%s  %s\n' db578c556298ad3bb1f4a93faa50d540eb2b9792215e81ef7548dd7e20a746e8 "$MLIP_MLIAP_MODEL" | sha256sum -c - >/dev/null
module load profile/base >/dev/null 2>&1
module load gcc/12.2.0 cuda/12.2 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 >/dev/null 2>&1
python_root=$(dirname -- "$(dirname -- "$MLIP_LEONARDO_PYTHON")")
export LD_LIBRARY_PATH="$python_root/lib:$MLIP_LEONARDO_LAMMPS_PREFIX/lib:$MLIP_LEONARDO_LAMMPS_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$MLIP_LEONARDO_WRAPPER_ROOT:$MLIP_LEONARDO_OVERLAY_ROOT"
export PYTHONNOUSERSITE=1 TORCH_COMPILE_DISABLE=1 TORCH_DISABLE_NATIVE_JIT=1
result="$MLIP_RESULTS_DIR/leonardo-lammps-${SLURM_JOB_ID}.json"
test ! -e "$result"
srun --nodes=1 --ntasks=1 --gpus=1 --cpu-bind=cores \
  "$MLIP_LEONARDO_PYTHON" "$MLIP_LESSON_ROOT/examples/lammps_si.py" \
  --mliap-model "$MLIP_MLIAP_MODEL" --integrator nve --replicas 1 \
  --cells 2 --warmup 1 --steps 5 >"$result"
jq -e '.state == "completed" and .replicas == 1 and .measured_steps == 5' "$result" >/dev/null
printf 'Leonardo LAMMPS ML-IAP/Kokkos smoke passed\n'
