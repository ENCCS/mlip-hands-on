#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:45:00
# Build one private MPI/CUDA/Hopper Metatomic/Kokkos LAMMPS candidate.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${MLIP_JUPITER_ROOT:?set the private artifact root}"
: "${MLIP_METATOMIC_SOURCE:?set the reviewed patched LAMMPS source}"
: "${MLIP_METATOMIC_OVERLAY:?set the reviewed Metatomic overlay}"
: "${MLIP_JUPITER_ALCHEMI_ENV_ID:?set the qualified native environment job ID}"
case "$MLIP_JUPITER_ALCHEMI_ENV_ID" in ''|*[!0-9]*) exit 2 ;; esac
test "$(uname -m)" = aarch64
test "${SLURM_JOB_NUM_NODES:-}" = 1
venv="$MLIP_JUPITER_ROOT/env-alchemi-$MLIP_JUPITER_ALCHEMI_ENV_ID"
nccl_dir="$venv/lib/python3.12/site-packages/nvidia/nccl/lib"
test -x "$venv/bin/python" && test -f "$nccl_dir/libnccl.so.2"
test -d "$MLIP_METATOMIC_OVERLAY/metatomic"
printf '%s  %s\n' 0bae52bfd833f9dd6294382b67bd17dd314c29dd7f8e3a39e8a42774ebfa2220 "$MLIP_METATOMIC_SOURCE/src/KOKKOS/pair_metatomic_kokkos.h" | sha256sum -c - >/dev/null
printf '%s  %s\n' a134f3d98e949a99233203016c545c37840e7a7f13b5d5706614a883e5a989eb "$MLIP_METATOMIC_SOURCE/src/KOKKOS/pair_metatomic_kokkos.cpp" | sha256sum -c - >/dev/null
printf '%s  %s\n' d21ea1aeab8794531cf17c4d5c3d3ca78945b78b3e611c0088bd06238c9c7f1f "$MLIP_METATOMIC_SOURCE/src/KOKKOS/fix_metatomic_kokkos.h" | sha256sum -c - >/dev/null
printf '%s  %s\n' 31ca266720d169b4dd693158b0d95e6c7fdf9836a28b6a2a9ad3d1cf16f0a3a3 "$MLIP_METATOMIC_SOURCE/src/KOKKOS/fix_metatomic_kokkos.cpp" | sha256sum -c - >/dev/null
module purge >/dev/null 2>&1
module load Stages/2025 GCC/13.3.0 CMake/3.29.3 CUDA/12 \
  OpenMPI/5.0.5 Python/3.12.3 >/dev/null
export PYTHONNOUSERSITE=1
export PYTHONPATH="$MLIP_METATOMIC_OVERLAY"
export LD_LIBRARY_PATH="$nccl_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
prefixes=$("$venv/bin/python" - <<'PY'
import metatensor, metatensor.torch, metatomic.torch, torch
print(';'.join((
    torch.utils.cmake_prefix_path,
    metatensor.utils.cmake_prefix_path,
    metatensor.torch.utils.cmake_prefix_path,
    metatomic.torch.utils.cmake_prefix_path,
)))
PY
)
scratch=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/mlip-metatomic-${SLURM_JOB_ID}.XXXXXXXX")
result="$MLIP_JUPITER_ROOT/builds/lammps-metatomic-$SLURM_JOB_ID"
mkdir -p -m 700 "$MLIP_JUPITER_ROOT/builds"
mkdir -m 700 "$result"
prefix="$result/prefix"
cmake -S "$MLIP_METATOMIC_SOURCE/cmake" -B "$scratch/build" -G 'Unix Makefiles' \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX="$prefix" \
  -D CMAKE_C_COMPILER="$(command -v mpicc)" \
  -D CMAKE_CXX_COMPILER="$MLIP_METATOMIC_SOURCE/lib/kokkos/bin/nvcc_wrapper" \
  -D MPI_C_COMPILER="$(command -v mpicc)" \
  -D MPI_CXX_COMPILER="$(command -v mpicxx)" \
  -D CMAKE_PREFIX_PATH="$prefixes" \
  -D CMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -Wl,-rpath-link,$nccl_dir -Wl,-rpath,$nccl_dir $nccl_dir/libnccl.so.2" \
  -D BUILD_MPI=ON -D BUILD_SHARED_LIBS=ON -D LAMMPS_INSTALL_RPATH=ON \
  -D PKG_KOKKOS=ON -D PKG_ML-METATOMIC=ON \
  -D DOWNLOAD_METATENSOR=OFF -D DOWNLOAD_METATOMIC=OFF \
  -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_OPENMP=ON \
  -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
  -D Kokkos_ARCH_ARMV9_GRACE=ON -D Kokkos_ARCH_HOPPER90=ON \
  -D FFT_KOKKOS=KISS >"$result/configure.log" 2>&1
grep -q '^BUILD_MPI:BOOL=ON$' "$scratch/build/CMakeCache.txt"
cmake --build "$scratch/build" --parallel 24 >"$result/build.log" 2>&1
cmake --install "$scratch/build" >"$result/install.log" 2>&1
test -x "$prefix/bin/lmp"
export LD_LIBRARY_PATH="$prefix/lib64:$prefix/lib:$LD_LIBRARY_PATH"
"$prefix/bin/lmp" -h >"$result/lmp-help.txt" 2>"$result/lmp-help.err"
grep -q 'ML-METATOMIC' "$result/lmp-help.txt"
grep -q 'KOKKOS' "$result/lmp-help.txt"
sha256sum "$prefix/bin/lmp" >"$result/lmp.sha256"
printf 'JUPITER Metatomic/Kokkos build completed: %s\n' "$SLURM_JOB_ID"
