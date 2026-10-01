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
CMake configuration for an MPI-enabled ML-IAP/Kokkos executable.
Set `MLIP_NATIVE_PREFIX` to the installed LAMMPS prefix and
`MLIP_NATIVE_PYTHON` to its matching MACE Python environment, then run
`source scripts/arrhenius-lammps-env.sh` in the allocated shell. For an older
archive containing `python/` and `mpi-prefix/`, use those two subdirectories
as the respective values. The helper loads the site compiler/CUDA module
and the matching Python and library paths.

Inside a one-GPU allocation, run:

```bash
bash scripts/run-alchemi.sh "$PWD" "$MLIP_ALCHEMI_SIF" "$MLIP_MODEL" 1 200
bash scripts/run-lammps.sh "$PWD" "$MLIP_LMP" "$MLIP_MLIAP_MODEL" 2 200
```

The MPI episode gives the separate launch pattern for one coupled trajectory
on multiple GPUs. The previous qualified native MPI build used the site's
MPICH and Slingshot provider (`FI_PROVIDER=cxi`, GPU-memory support enabled),
with `srun --mpi=pmi2`; `scripts/arrhenius-lammps-env.sh` sets that runtime
environment. The tested route requests `#SBATCH --network=single_node_vni`
even for a one-rank GPU check; omitting it caused an OFI domain failure in
a fresh build check. This lesson does not use `mpprun`.
