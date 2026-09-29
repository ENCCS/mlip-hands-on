# Arrhenius

Arrhenius GH200 nodes provide the ARM/NVIDIA GPU route used in the examples.
Request a GPU allocation with your own project account and current site
policy. A reservation is optional and must not be assumed.

The original MACE checkpoint stays outside the repository. Build the
ALCHEMI SIF from [the definition file](../../alchemi-aarch64.def) into a
new path:

```bash
bash scripts/build-alchemi.sh "$PWD" "$MLIP_ALCHEMI_SIF"
```

The model is not baked into the SIF. The run script binds it read-only at
`/models/mace.model`.

For native LAMMPS, use the Arrhenius GCC/CUDA environment and site MPICH
wrappers. The [build episode](../episodes/03-lammps-mpi.md) shows the
participant-facing CMake command. Older guarded build jobs remain in the
`main` branch's Git history and target its old layout. Select a verified MPI-enabled
ML-IAP/Kokkos `lmp` as `MLIP_LMP`.
After extracting the native runtime, set `MLIP_NATIVE_PREFIX` and run
`source scripts/arrhenius-lammps-env.sh` in the allocated shell. This loads
the site compiler/CUDA module and the matching Python and library paths.

Inside a one-GPU allocation, run:

```bash
bash scripts/run-alchemi.sh "$PWD" "$MLIP_ALCHEMI_SIF" "$MLIP_MODEL" 1 200
bash scripts/run-lammps.sh "$PWD" "$MLIP_LMP" "$MLIP_MLIAP_MODEL" 2 200
```

The MPI episode gives the separate launch pattern for one coupled trajectory
on multiple GPUs. The previous qualified native MPI build used the site's
MPICH and Slingshot provider (`FI_PROVIDER=cxi`, GPU-memory support enabled),
with `srun --mpi=pmi2`; `scripts/arrhenius-lammps-env.sh` sets that runtime
environment. For a multi-rank allocation, request the site's single-node
VNI setting if required. This lesson does not use `mpprun`.
