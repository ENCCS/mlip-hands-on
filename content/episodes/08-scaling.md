# One trajectory across GPUs

The earlier examples use one GPU. For a larger *single* LAMMPS trajectory,
MPI ranks can divide the simulation domain, with one GPU per rank. The
[input file](../../examples/lammps_mace.in) accepts `cells`: a diamond
supercell has `8 × cells³` atoms. The
[MPI command](../../scripts/run-lammps-mpi.sh) launches the same input with
`srun` from an existing allocation.

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
