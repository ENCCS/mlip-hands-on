# JUPITER

JUPITER Booster also has GH200 GPUs, but its Slurm and MPI configuration
differs from Arrhenius.

The same hash-verified ARM/GH200 ALCHEMI SIF has passed bounded one-GPU
single-trajectory, eight-trajectory, and relaxation checks on both sites.
Stage the image and original checkpoint in private project storage and run:

```bash
bash scripts/run-alchemi.sh "$PWD" "$MLIP_ALCHEMI_SIF" "$MLIP_MODEL" 1 200
```

A pinned native ALCHEMI environment is also tested, but it is a separate
runtime route. The short SIF checks do not establish long-run performance or
scientific equivalence.

Build native MPI ML-IAP/Kokkos LAMMPS with JUPITER's matching compiler,
OpenMPI, CUDA and Python libraries. The [build episode](../episodes/03-lammps-mpi.md)
shows the participant-facing command. Earlier guarded jobs remain in Git
history only; they target the old layout.
The participant MD command uses the `lmp` executable and an input file:

Set `MLIP_NATIVE_PREFIX` to the native LAMMPS installed prefix and
`MLIP_NATIVE_PYTHON` to its matching MACE environment, then run
`source scripts/jupiter-lammps-env.sh` in the allocated shell.

```bash
bash scripts/run-lammps.sh "$PWD" "$MLIP_LMP" "$MLIP_MLIAP_MODEL" 2 200
```

Run a one-GPU command inside a GPU-bound Slurm step. A batch shell can see
more GPUs than requested. The scaling episode gives the MPI pattern.
