#!/usr/bin/env bash
# Core CMake build from an extracted LAMMPS source tree and prepared Python.
# Run inside a GPU build allocation after loading the site's build modules.
set -euo pipefail
source_dir=$1
build_dir=$2
prefix=$3
python=$4
wrapper=$source_dir/lib/kokkos/bin/nvcc_wrapper
python_include=$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')
python_library=$("$python" -c 'import os,sysconfig; print(os.path.join(sysconfig.get_config_var("LIBDIR"),sysconfig.get_config_var("LDLIBRARY")))')
python_root=$(cd "$(dirname "$python")/.." && pwd -P)
python_version=$("$python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
# A relocatable Python archive can retain sysconfig paths from its original
# build location; prefer headers and libpython beside that extracted Python.
if [[ -f "$python_root/include/python$python_version/Python.h" ]]; then
    python_include="$python_root/include/python$python_version"
fi
if [[ -f "$python_root/lib/libpython$python_version.so" ]]; then
    python_library="$python_root/lib/libpython$python_version.so"
fi

cmake -S "$source_dir/cmake" -B "$build_dir" -G 'Unix Makefiles' \
    -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX="$prefix" \
    -D CMAKE_C_COMPILER="$(command -v mpicc)" \
    -D CMAKE_CXX_COMPILER="$wrapper" \
    -D MPI_C_COMPILER="$(command -v mpicc)" \
    -D MPI_CXX_COMPILER="$(command -v mpicxx)" \
    -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON -D LAMMPS_EXCEPTIONS=ON \
    -D PKG_KOKKOS=ON -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON \
    -D PKG_PYTHON=ON -D MLIAP_ENABLE_PYTHON=ON \
    -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
    -D Kokkos_ARCH_ARMV9_GRACE=ON -D Kokkos_ARCH_HOPPER90=ON \
    -D FFT_KOKKOS=KISS -D Python_EXECUTABLE="$python" \
    -D Python_INCLUDE_DIR="$python_include" \
    -D Python_LIBRARY="$python_library"
cmake --build "$build_dir" --parallel 24
cmake --install "$build_dir"
