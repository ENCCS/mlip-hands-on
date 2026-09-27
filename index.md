# Molecular dynamics with MACE on GPUs

Follow one silicon system from a single trajectory to independent replicas on
a GPU. The steps use the same pinned MACE potential with ALCHEMI Toolkit and
LAMMPS ML-IAP/Kokkos, then show how to read their timings without conflating
single-trajectory speed and aggregate throughput.

Run the examples in order, or use the setup pages to prepare artifacts in
advance. Each runnable section shows its inputs and a small result table;
the reviewed-results page keeps the repeated measurements separate from a
single notebook run.

## Setup

```{toctree}
:maxdepth: 1

setup/index
setup/arrhenius
setup/notebook
```

## Episodes

```{toctree}
:maxdepth: 1

episodes/01-model
episodes/02-alchemi-image
episodes/03-lammps-mpi
episodes/04-silicon-md
episodes/05-batched-md
episodes/06-lammps-replicas
episodes/07-reviewed-results
episodes/08-scaling
```

## Reference

```{toctree}
:maxdepth: 1

reference/environment
reference/instructor
reference/limits
```
