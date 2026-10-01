# Prepare native LAMMPS ML-IAP/Kokkos

LAMMPS needs an MPI-enabled build with ML-IAP and Kokkos CUDA support, plus
a Python/PyTorch environment compatible with the exported MACE model.
Build it separately on each site: their MPI and CUDA libraries differ.

On Arrhenius, load **`GPU/buildenv-gcccuda/2026.03-cu13.0`** after the SSE
initialization; its compiler wrappers provide MPICH. On JUPITER, load
`Stages/2025 GCC/13.3.0 CMake/3.29.3 CUDA/12 OpenMPI/5.0.5 Python/3.12.3
PyTorch/2.5.1`. These are different native MPI stacks. Use a matching
Python environment with MACE, Torch, NumPy, Cython and headers; a general
login-shell Python may lack the packages or matching libraries.

The [build script](../../scripts/build-lammps-mpi.sh) enables MPI, ML-IAP,
and Kokkos CUDA. It selects the GH200 CPU/GPU architecture flags
`Kokkos_ARCH_ARMV9_GRACE` and `Kokkos_ARCH_HOPPER90`, plus the matching
Python executable, headers and library. Starting from the pinned LAMMPS source
archive, unpack it in scratch, then run the build inside a GPU allocation:

```bash
scratch=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}
tar -xzf "$MLIP_LAMMPS_SOURCE_ARCHIVE" -C "$scratch"
bash scripts/build-lammps-mpi.sh \
  "$scratch/lammps-lammps-751b42d" \
  "$scratch/lammps-build" \
  "$MLIP_NATIVE_PREFIX" \
  "$MLIP_NATIVE_PYTHON/bin/python3"
```

`MLIP_LAMMPS_SOURCE_ARCHIVE` and `MLIP_NATIVE_PYTHON` are build inputs,
not needed to run ALCHEMI. The site setup pages explain how the native
runtime is selected after installation. Arrhenius's previously qualified
archive contains `python/` and `mpi-prefix/` under one root. For that archive,
set `MLIP_NATIVE_PYTHON` and `MLIP_NATIVE_PREFIX` to those subdirectories.
The short build script writes an ordinary install prefix: use that prefix
as `MLIP_NATIVE_PREFIX` and keep the matching Python environment separate.

After installation, source the site's environment script to set `MLIP_LMP`.
Its `-h` output should list ML-IAP and KOKKOS. Then run the
[single-trajectory example](04-silicon-md.md) on a GPU to check that the
exported model loads and MD completes.

The simulation inputs are plain text. The single-trajectory input is
[`lammps_mace.in`](../../examples/lammps_mace.in); the relaxation input is
[`lammps_relax.in`](../../examples/lammps_relax.in). The launcher supplies
the model and run settings with `-var`; no Python LAMMPS driver is used.
