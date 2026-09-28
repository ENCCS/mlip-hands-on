#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:45:00
# Build a private JUPITER/Grace-Hopper LAMMPS candidate; do not select it yet.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_JUPITER_ROOT:?set a private JUPITER artifact root}"
test "$(uname -m)" = aarch64
test "${SLURM_JOB_NUM_NODES:-}" = 1

archive="$MLIP_JUPITER_ROOT/inputs/lammps-stable_22Jul2025_update6-c52ecdb29dcf464a58e9cbf03fddab0c9e5389df31923347f55a4d8ce4dd7dd0.tar.gz"
venv="$MLIP_JUPITER_ROOT/env-mace-0315"
test -f "$archive" && test -x "$venv/bin/python"
printf '%s  %s\n' c52ecdb29dcf464a58e9cbf03fddab0c9e5389df31923347f55a4d8ce4dd7dd0 "$archive" | sha256sum -c - >/dev/null

module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 CMake/3.29.3 CUDA/12 \
  OpenMPI/5.0.5 Python/3.12.3 PyTorch/2.5.1 >/dev/null
"$venv/bin/python" -c 'import numpy,torch,cupy,mace; assert numpy.__version__ == "1.26.4"'
test -f "$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')/Python.h"

scratch=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mlip-jupiter-lammps-${SLURM_JOB_ID}.XXXXXX")
tar -xzf "$archive" -C "$scratch"
source_dir="$scratch/lammps-lammps-751b42d"
test -f "$source_dir/cmake/CMakeLists.txt"
result="$MLIP_JUPITER_ROOT/builds/lammps-mliap-${SLURM_JOB_ID}"
mkdir -p -m 700 "$MLIP_JUPITER_ROOT/builds"
mkdir -m 700 "$result"
prefix="$result/prefix"
python_include=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')
python_library=$("$venv/bin/python" -c 'import sysconfig,os; print(os.path.join(sysconfig.get_config_var("LIBDIR"),sysconfig.get_config_var("LDLIBRARY")))')
test -f "$python_library"

cmake -S "$source_dir/cmake" -B "$scratch/build" -G 'Unix Makefiles' \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX="$prefix" \
  -D CMAKE_C_COMPILER="$(command -v mpicc)" \
  -D CMAKE_CXX_COMPILER="$source_dir/lib/kokkos/bin/nvcc_wrapper" \
  -D MPI_C_COMPILER="$(command -v mpicc)" \
  -D MPI_CXX_COMPILER="$(command -v mpicxx)" \
  -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON -D LAMMPS_EXCEPTIONS=ON \
  -D PKG_KOKKOS=ON -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON \
  -D PKG_PYTHON=ON -D MLIAP_ENABLE_PYTHON=ON \
  -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_OPENMP=ON \
  -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
  -D Kokkos_ARCH_ARMV9_GRACE=ON -D Kokkos_ARCH_HOPPER90=ON \
  -D FFT_KOKKOS=KISS \
  -D Python_EXECUTABLE="$venv/bin/python" \
  -D Python_INCLUDE_DIR="$python_include" \
  -D Python_LIBRARY="$python_library" \
  >"$result/configure.log" 2>&1
grep -q '^BUILD_MPI:BOOL=ON$' "$scratch/build/CMakeCache.txt"
cmake --build "$scratch/build" --parallel 24 >"$result/build.log" 2>&1
cmake --install "$scratch/build" >"$result/install.log" 2>&1
test -x "$prefix/bin/lmp"
export LD_LIBRARY_PATH="$prefix/lib64:$prefix/lib:$(dirname "$python_library")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$prefix/bin/lmp" -h >"$result/lmp-help.txt" 2>"$result/lmp-help.err"
grep -q 'ML-IAP' "$result/lmp-help.txt"
grep -q 'KOKKOS' "$result/lmp-help.txt"
sha256sum "$prefix/bin/lmp" >"$result/lmp.sha256"
echo "JUPITER native LAMMPS candidate built: $result"
