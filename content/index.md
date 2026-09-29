# Molecular dynamics with MACE on GPUs

Use one silicon system to explore molecular dynamics and geometry relaxation
with NVIDIA ALCHEMI Toolkit and LAMMPS ML-IAP/Kokkos. The pages show commands
you can run on an allocated GH200 GPU. The same MyST Markdown opens as a
notebook; no separate `.ipynb` files are maintained.

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
reference/limits
reference/instructor
```
