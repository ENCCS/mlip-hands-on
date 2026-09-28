---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.13'
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Capacity and more than one GPU

Batching independent systems and splitting **one coupled system** across
GPUs solve different problems. For one large system, strong scaling holds
the atom count fixed while increasing GPUs; weak scaling increases atoms
with GPUs so the work per GPU stays roughly fixed.

The MPI-enabled LAMMPS example includes a one-node 1/2/4-GPU size job at
32,768 atoms. Review its account, partition, time, memory, and GPU request
for the current site before submitting it. The script does not submit
itself, and a `sbatch --test-only` check is not a real submission.

```{literalinclude} ../scripts/lammps-mpi-size-benchmark.sbatch
:language: bash
:lines: 1-16
```

ALCHEMI Toolkit also has a separate `DomainParallel` route for spatially
partitioning one system. It uses a distributed communication setup and must
not be confused with the independent-replica batch above. Our current
multi-GPU records include an unresolved rank-count energy difference, so
this lesson does **not** present cross-engine multi-GPU scientific agreement
or a two-node speed ranking as established results.

JUPITER has a separate [per-rank GPU binding and native MPI setup](../setup/jupiter.md).
One-, two-, and four-GPU one-node runs and an eight-GPU two-node run have
passed short functional checks there. Ten measured steps on 512 atoms are
not a scaling study; do not plot those elapsed times as speedups.

For a capacity experiment, increase atoms or replicas by predeclared steps,
record completed runs and failures, and stop at the first resource limit.
The largest completed case is a **tested workload**, not an intrinsic GPU
atom limit: memory depends on the checkpoint, neighbor list, batch size,
precision, and other GPU allocations. See [Limits and provenance](../reference/limits.md).
