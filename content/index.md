# Molecular dynamics with MACE on GPUs

Run silicon molecular dynamics and geometry relaxation with NVIDIA ALCHEMI
Toolkit and LAMMPS ML-IAP/Kokkos. Start with one trajectory, run independent
trajectories together, then explore how system size and GPU count affect
performance. Follow Setup first, then the Episodes in order.

The examples run on allocated GH200 GPUs. Their MyST Markdown pages also
open as notebooks, so you can read and run the same code without a separate
`.ipynb` copy. Reference explains inputs, measurements and known limits.

```{toctree}
:caption: Setup
:maxdepth: 1

setup/index
setup/hardware
setup/arrhenius
setup/jupiter
setup/notebook
```

```{toctree}
:caption: Episodes
:maxdepth: 1

episodes/01-model
episodes/02-alchemi-image
episodes/03-lammps-mpi
episodes/04-silicon-md
episodes/05-batched-md
episodes/06-lammps-replicas
episodes/07-relaxation
episodes/08-scaling
episodes/09-results
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/inputs
reference/benchmarks
reference/limits
reference/instructor
```
