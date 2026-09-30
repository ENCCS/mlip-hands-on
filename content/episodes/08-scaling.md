# One trajectory across GPUs

The earlier examples use one GPU. For a larger *single* LAMMPS trajectory,
MPI ranks can divide the simulation domain, with one GPU per rank. The
[input file](../../examples/lammps_mace.in) accepts `cells`: a diamond
supercell has `8 × cells³` atoms. The
[MPI command](../../scripts/run-lammps-mpi.sh) launches the same input with
`srun` from an existing allocation.

On one node with four allocated GPUs, the launcher amounts to this pattern.
`-k on g 4` is the number of GPUs per node, not per rank; keeping a common
visible GPU set for all ranks avoids per-rank GPU-mask mismatches:

```bash
export MLIP_ALLOCATED_CUDA_DEVICES="$CUDA_VISIBLE_DEVICES"
srun --ntasks=4 --gpu-bind=none /bin/bash -c \
  'export CUDA_VISIBLE_DEVICES="$MLIP_ALLOCATED_CUDA_DEVICES"; exec "$@"' _ \
  "$MLIP_LMP" -k on g 4 -sf kk \
  -pk kokkos newton on neigh half gpu/aware on \
  -log none -in examples/lammps_mace.in \
  -var model "$MLIP_MLIAP_MODEL" -var cells 8 -var warmup 0 \
  -var steps 100 -var seed 20260924 -var bath_seed 20260925
```

Add your site's required `--mpi` setting to `srun`; Arrhenius's tested
MPICH route uses `--mpi=pmi2`. The supplied launcher reads `MLIP_SRUN_MPI`
for this and checks the allocation boundary before calling `srun`.

For strong scaling, fix `cells` and vary GPU count. For weak scaling,
increase `cells` with GPU count so atoms per GPU stay approximately fixed.
Record the actual atoms, ranks, nodes, warm-up, measured steps, elapsed
seconds, and GPU type alongside every result. A calculation that merely
finishes is not a scaling measurement.

```{warning}
Do not launch this from a login node. Request an allocation for the chosen
site first. Multi-node runs may require a different partition or MPI/GPU
transport configuration than a one-node example. The Arrhenius short-test
reservation must be checked for the requested node count; `--test-only`
does not establish that the job will run promptly.
```

This episode focuses on the LAMMPS MPI route. The ALCHEMI one-GPU batch
demonstration is not a claim of multi-GPU domain decomposition.
