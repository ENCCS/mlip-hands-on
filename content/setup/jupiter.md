# JUPITER

JUPITER Booster also has GH200 GPUs, but its Slurm and MPI configuration
differs from Arrhenius.

Use the same ARM/GH200 ALCHEMI SIF as on Arrhenius.
JUPITER production compute nodes have no external internet access, according
to the [site environment guide](https://apps.fz-juelich.de/jsc/hps/jupiter/environment.html).
Do not run the Docker-backed image build there unless its OCI and Python
inputs have been staged for offline use. Build on a permitted networked
ARM64 builder, then stage the image and original checkpoint in private
project storage and run:

```bash
bash scripts/run-alchemi.sh "$PWD" "$MLIP_ALCHEMI_SIF" "$MLIP_MODEL" 1 200
```

A native ALCHEMI environment is an alternative; the examples here use the
SIF so both sites run the same container environment.

Build native MPI ML-IAP/Kokkos LAMMPS with JUPITER's matching compiler,
OpenMPI, CUDA and Python libraries. The [build episode](../episodes/03-lammps-mpi.md)
shows the CMake configuration. Run MD with the resulting `lmp` executable
and the supplied input file.

Set `MLIP_NATIVE_PREFIX` to the native LAMMPS installed prefix and
`MLIP_NATIVE_PYTHON` to its matching MACE environment, then run
`source scripts/jupiter-lammps-env.sh` in the allocated shell.

```bash
bash scripts/run-lammps.sh "$PWD" "$MLIP_LMP" "$MLIP_MLIAP_MODEL" 2 200
```

Run this command inside a GPU allocation and check `CUDA_VISIBLE_DEVICES`
before running; using one GPU need not mean the scheduler allocates only
one GPU. The notebook starts in the batch
shell so its native LAMMPS child does not inherit a separate step's MPI
descriptors. The scaling episode gives the distinct multi-rank MPI pattern.
