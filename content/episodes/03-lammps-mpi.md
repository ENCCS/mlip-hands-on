# Prepare native LAMMPS ML-IAP/Kokkos

LAMMPS needs an MPI-enabled build with ML-IAP and Kokkos CUDA support, plus
a Python/PyTorch environment compatible with the exported MACE model.
The two GH200 sites use different MPI and CUDA module stacks; do not copy
one site's native executable to the other.

On Arrhenius, load `GPU/buildenv-gcccuda/2026.03-cu13.0` after the SSE
initialization; its wrappers provide MPICH. On JUPITER, load
`Stages/2025 GCC/13.3.0 CMake/3.29.3 CUDA/12 OpenMPI/5.0.5 Python/3.12.3
PyTorch/2.5.1`. These are different native MPI stacks. Use a matching
Python environment with MACE, Torch, NumPy, Cython and headers; a general
login-shell Python is not enough.

The [short build script](../../scripts/build-lammps-mpi.sh) shows the
essential CMake configuration. Starting from the pinned LAMMPS source
archive, unpack it in scratch, then run the build inside a GPU allocation:

```bash
tar -xzf "$MLIP_LAMMPS_SOURCE_ARCHIVE" -C "$SLURM_TMPDIR"
bash scripts/build-lammps-mpi.sh \
  "$SLURM_TMPDIR/lammps-lammps-751b42d" \
  "$SLURM_TMPDIR/lammps-build" \
  "$MLIP_NATIVE_PREFIX" \
  "$MLIP_NATIVE_PYTHON/bin/python3"
```

`MLIP_LAMMPS_SOURCE_ARCHIVE` and `MLIP_NATIVE_PYTHON` are build inputs,
not needed to run ALCHEMI. The site setup pages explain how the native
runtime is selected after installation. Arrhenius's previously qualified
runtime is packaged with `python/` and `mpi-prefix/` under one root; the
script above writes an ordinary installed prefix, so set `MLIP_LMP` to its
`bin/lmp` directly unless you package the paired runtime in that layout.

After building, `MLIP_LMP` names the `lmp` executable. Check that its
help output lists ML-IAP and KOKKOS, and that the pinned export loads in
an allocated GPU run. A successful `lmp -h` alone is not a GPU test.

The simulation inputs are plain text. The single-trajectory input is
[`lammps_mace.in`](../../examples/lammps_mace.in); the relaxation input is
[`lammps_relax.in`](../../examples/lammps_relax.in). The launcher supplies
the model and run settings with `-var`; no Python LAMMPS driver is used.
