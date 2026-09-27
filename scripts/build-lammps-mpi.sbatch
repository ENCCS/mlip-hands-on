#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=02:00:00
# Build a DISTINCT MPI-enabled ML-IAP/Kokkos candidate on one GH200.
# No accepted serial artifact is modified. The output remains unqualified.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_NATIVE_ARCHIVE:?pinned serial Python dependency archive}"
: "${MLIP_LAMMPS_SOURCE_ARCHIVE:?pinned LAMMPS source archive}"
: "${MLIP_MPI_BUILD_ROOT:?existing private artifact root}"
test "$(uname -m)" = aarch64
test -d "$MLIP_MPI_BUILD_ROOT"
test -f "$MLIP_NATIVE_ARCHIVE" && test -f "$MLIP_LAMMPS_SOURCE_ARCHIVE"
printf '%s  %s\n' \
  837bc9690f8ee8c43727a47c7f9007c8fa964d8bded4bc0cc373c875d56aecc8 \
  "$MLIP_NATIVE_ARCHIVE" | sha256sum -c - >/dev/null
printf '%s  %s\n' \
  c52ecdb29dcf464a58e9cbf03fddab0c9e5389df31923347f55a4d8ce4dd7dd0 \
  "$MLIP_LAMMPS_SOURCE_ARCHIVE" | sha256sum -c - >/dev/null

source /software/sse2/init/hpc_init_sse.sh >/dev/null 2>&1
module load GPU/buildenv-gcccuda/2026.03-cu13.0 >/dev/null 2>&1
test -x "$(command -v make)"
scratch=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mlip-mpi-build-${SLURM_JOB_ID}.XXXXXX")
tar -xzf "$MLIP_NATIVE_ARCHIVE" -C "$scratch"
tar -xzf "$MLIP_LAMMPS_SOURCE_ARCHIVE" -C "$scratch"
python="$scratch/python/bin/python3"
source_dir="$scratch/lammps-lammps-751b42d"
prefix="$scratch/mpi-prefix"
test -x "$python" && test -f "$source_dir/cmake/CMakeLists.txt"
"$python" -c 'import sys,torch,mace,cupy; assert sys.version_info[:2] == (3,12)'
test -f "$scratch/python/include/python3.12/Python.h"
test -f "$scratch/python/lib/libpython3.12.so"
# The verified dependency archive contains a build-time cythonize entry point
# with its original temporary shebang. Repair only this extracted job-local
# copy, and only when the exact known old line is present.
cythonize="$scratch/python/bin/cythonize"
test -f "$cythonize"
test "$(head -n 1 "$cythonize")" = \
  '#!/tmp/mlip-lammps-native-build-2000103/prefix/python/bin/python3'
sed -i "1c#!$python" "$cythonize"
test "$(head -n 1 "$cythonize")" = "#!$python"

result="$MLIP_MPI_BUILD_ROOT/build-${SLURM_JOB_ID}"
mkdir -m 700 "$result"
cmake -S "$source_dir/cmake" -B "$scratch/mpi-build" -G 'Unix Makefiles' \
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
  -D Python_EXECUTABLE="$python" \
  -D Python_INCLUDE_DIR="$scratch/python/include/python3.12" \
  -D Python_LIBRARY="$scratch/python/lib/libpython3.12.so" \
  >"$result/configure.log" 2>&1
grep -q '^BUILD_MPI:BOOL=ON$' "$scratch/mpi-build/CMakeCache.txt"
grep -Eq '^MPI_CXX_COMPILER:[^=]+=.+/MPICH/4\.3\.2/' "$scratch/mpi-build/CMakeCache.txt"
cmake --build "$scratch/mpi-build" --parallel 36 >"$result/build.log" 2>&1
cmake --install "$scratch/mpi-build" >"$result/install.log" 2>&1
cmake --build "$scratch/mpi-build" --target install-python \
  >"$result/install-python.log" 2>&1
test -x "$prefix/bin/lmp"
# The fresh prefix uses lib64 on this site; keep it ahead of inherited
# toolchain paths for the candidate's own linkage and help checks. The
# copied Python environment also carries libpython3.12.so for ML-IAP.
test -f "$prefix/lib64/liblammps.so.0"
test -f "$scratch/python/lib/libpython3.12.so"
export LD_LIBRARY_PATH="$prefix/lib64:$scratch/python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$prefix/bin/lmp" >"$result/lmp-ldd.txt"
if grep -q 'not found' "$result/lmp-ldd.txt"; then
  echo 'built lmp has unresolved shared libraries' >&2
  exit 1
fi
grep -q 'libmpi' "$result/lmp-ldd.txt"
"$prefix/bin/lmp" -h >"$result/lmp-help.txt"
grep -q 'ML-IAP' "$result/lmp-help.txt"
grep -q 'KOKKOS' "$result/lmp-help.txt"
sha256sum "$prefix/bin/lmp" >"$result/lmp.sha256"

# The candidate contains the copied Python environment plus only the new
# MPI prefix; the accepted serial LAMMPS prefix is deliberately omitted.
tar -czf "$result/mpi-runtime-candidate.tar.gz" -C "$scratch" python mpi-prefix
sha256sum "$result/mpi-runtime-candidate.tar.gz" >"$result/archive.sha256"
printf 'MPI LAMMPS build candidate created; private evidence: %s\n' "$result"
