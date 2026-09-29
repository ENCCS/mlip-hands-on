# Universal MLIPs on HPC: hands-on

This lesson runs universal machine-learned interatomic potentials (MLIPs)
from the MACE-MP family on GPU nodes of HPC systems. It has two parts that
share one idea: keep the GPU busy with many independent systems, and measure
what that buys you honestly.

**Part A** screens many small crystals at once: it relaxes a set of
structures one at a time with ASE, then all together in one batched TorchSim
call. **Part B** follows one silicon system from a single molecular-dynamics
trajectory to many independent replicas with ALCHEMI Toolkit and LAMMPS
ML-IAP/Kokkos, and shows how to read single-trajectory speed and aggregate
throughput separately.

Contributors: Wei Li (Part B) and Karim Elgammal (Part A).

## Setup

```{toctree}
:maxdepth: 1

setup/index
setup/arrhenius
setup/jupiter
setup/leonardo
setup/notebook
```

## Part A: batched screening

```{toctree}
:maxdepth: 1

episodes/a1-batched-relaxation
episodes/a2-orb-models
```

## Part B: molecular dynamics

Run the examples in order, or use the setup pages to prepare artifacts in
advance. Each runnable section shows its inputs and a small result table;
the reviewed-results page keeps the repeated measurements separate from a
single notebook run.

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
