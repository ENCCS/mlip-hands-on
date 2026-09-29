#!/usr/bin/env bash
# Source this in a JUPITER Booster GPU shell before the native examples.
: "${MLIP_NATIVE_PREFIX:?set native LAMMPS build root}"
: "${MLIP_NATIVE_PYTHON:?set the matching MACE Python environment}"
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 CUDA/12 OpenMPI/5.0.5 \
    Python/3.12.3 PyTorch/2.5.1 >/dev/null 2>&1
export MLIP_LMP="$MLIP_NATIVE_PREFIX/bin/lmp"
export PATH="$MLIP_NATIVE_PYTHON/bin:$PATH"
export LD_LIBRARY_PATH="$MLIP_NATIVE_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
