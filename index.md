# Universal MLIPs on HPC: hands-on

Machine-learned interatomic potentials (MLIPs) learn energies and forces
from quantum-mechanical data and then run at a small fraction of the cost of
density functional theory. Universal, or foundation, MLIPs are trained once on
large datasets that span the periodic table, and are used directly or
fine-tuned for one system. On HPC systems most of their cost is GPU inference,
so the way work is placed on the GPU sets the throughput.

This lesson shows two cases. **Part A** relaxes many small crystals in one
batched TorchSim call and compares it with one-at-a-time relaxation in ASE,
using MACE-MP and Orb-v3. **Part B** runs molecular dynamics of silicon with
MACE-MP in ALCHEMI Toolkit and LAMMPS, from one trajectory to many replicas and
many GPUs.

New to MLIPs? Start with {doc}`episodes/00-background`.

![Lesson map: Part A screening and Part B molecular dynamics.](_static/lesson-map.drawio.png)

## Background

```{toctree}
:maxdepth: 1

episodes/00-background
```

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
reference/reading
```
